#include "../include/nr_int8_handoff.h"
#include "../include/nr_split_resident.h"
#include <unordered_map>
#include <memory>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <map>
#include "../include/nr_weight_package.h"
#include "../include/nr_gpu_model.h"
#include "../include/nr_spatial_graph.h"
#include "../include/nr_extent.h"
#include "../include/nr_execution_profile.h"

static thread_local const NrV2::SpatialGraphProfile* active_graph_profile=nullptr;
static void stage_event(const char* name,bool begin){
    if(active_graph_profile&&active_graph_profile->stageEvent)
        active_graph_profile->stageEvent(active_graph_profile->stageEventContext,name,begin,current_stream());
}

using HalfBuffer=Buffer<__half>;
using HalfPtr=std::unique_ptr<HalfBuffer>;

static bool trace_requested(int block){
    if(block!=2&&block!=70)return false;
    char folder[4096];size_t size=0;
    if(NrExecution::ReadSetting(&size,folder,sizeof(folder),"RDNA2_SPATIAL_TRACE"))
        throw std::runtime_error("Invalid spatial trace path");
    return size&&folder[0];
}

// Optional local diagnostic copies; never fed back into the connected graph.
template<class T> static void trace_stage(int block,const char*name,Buffer<T>&buffer){
    if(!trace_requested(block))return;
    char folder[4096];size_t size=0;
    if(NrExecution::ReadSetting(&size,folder,sizeof(folder),"RDNA2_SPATIAL_TRACE"))
        throw std::runtime_error("Invalid spatial trace path");
    if(!size||!folder[0])return;
    auto values=buffer.read();
    std::string path=std::string(folder)+"/block"+std::to_string(block)+"_"+name+".bin";
    std::ofstream output(path,std::ios::binary);
    if(!output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T)))
        throw std::runtime_error("Unable to write spatial stage trace");
}

__global__ void round_float_kernel(const float*in,__half*out,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2half(in[i]);}
__global__ void publish_half_kernel(const __half*in,__half*out,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=pubh(in[i]);}
__global__ void pool2_kernel(const __half*in,__half*out,int h,int w,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(h/2)*(w/2)*c)return;int ch=i%c,p=i/c,x=p%(w/2),y=p/(w/2);size_t a=(size_t(y*2)*w+x*2)*c+ch,b=a+c,d=a+w*c,e=d+c;__half top=__hadd(in[a],in[b]),bottom=__hadd(in[d],in[e]);out[i]=__hmul(__hadd(top,bottom),__float2half(.25f));}
__global__ void pad_end_kernel(const __half*in,__half*out,int h,int w,int ph,int pw,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ph*pw*c)return;int ch=i%c,p=i/c,x=p%pw,y=p/pw;out[i]=(y<h&&x<w)?in[(y*w+x)*c+ch]:__float2half(0);}
__global__ void partition_features(const __half*in,__half*out,int h,int w,int c,int top,int left,int ph,int pw){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ph*pw*c)return;int ch=i%c,e=i/c,ix=e%8;e/=8;int iy=e%8;e/=8;int wx=e%(pw/8),wy=e/(pw/8),y=wy*8+iy-top,x=wx*8+ix-left;out[i]=(y>=0&&y<h&&x>=0&&x<w)?in[(y*w+x)*c+ch]:__float2half(0);}
__global__ void reverse_features(const __half*windows,__half*out,int h,int w,int c,int top,int left,int pw){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h*w*c)return;int ch=i%c,p=i/c,y=p/w+top,x=p%w+left;size_t wi=(((size_t(y/8)*(pw/8)+x/8)*64+(y%8)*8+x%8)*c)+ch;out[i]=windows[wi];}
__global__ void upsample_merge_kernel(const __half*low,const __half*skip,const __half*sine,__half*out,int lw,int th,int tw,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=th*tw*c)return;int ch=i%c,p=i/c,x=p%tw,y=p/tw;__half up=low[((y/2)*lw+x/2)*c+ch];out[i]=pubh(__hadd(up,__hmul(skip[i],sine[ch])));}
__global__ void final_merge_kernel(const __half*low,const __half*skip,const __half*sine,const __half*cosine,__half*out,int lw,int th,int tw,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=th*tw*c)return;int ch=i%c,p=i/c,x=p%tw,y=p/tw;__half a=__hmul(low[((y/2)*lw+x/2)*c+ch],sine[ch]),b=__hmul(skip[i],cosine[ch]);out[i]=__hadd(a,b);}
__global__ void head_kernel(const __half*in,const __half*gain,const __half*conv,__half*out,int tokens){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*4)return;int row=i/4,col=i%4;float a=0,b=0;for(int k=0;k<16;k+=2){a=amd_mixed_dot(__halves2half2(in[row*32+k],in[row*32+k+1]),__halves2half2(gain[k*4+col],gain[(k+1)*4+col]),a,false);b=amd_mixed_dot(__halves2half2(in[row*32+16+k],in[row*32+17+k]),__halves2half2(conv[k*4+col],conv[(k+1)*4+col]),b,false);}out[i]=__hadd(__float2half(a),__float2half(b));}

__global__ void normalize_global(const float*projected,const __half*scale,__half*published,int m){int v=blockIdx.x*blockDim.x+threadIdx.x;if(v>=3*32*m)return;int kind=v/(32*m),head=(v/m)%32,token=v%m;__half x[32];for(int i=0;i<32;++i)x[i]=__float2half(projected[token*3072+kind*1024+head*32+i]);__half inv=__float2half(1);if(kind<2){__half part[4][2],pair[4][2];for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){int q=lane*2+p;part[lane][p]=__hadd(__hfma(x[q+8],x[q+8],__hmul(x[q],x[q])),__hfma(x[q+24],x[q+24],__hmul(x[q+16],x[q+16])));}for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);__half n=__hadd(__hadd(pair[0][0],pair[1][0]),__hadd(pair[0][1],pair[1][1]));n=__float2half(fmaxf(__half2float(n),0.00006198883056640625f));inv=__float2half(rsqrtf(__half2float(n)));}for(int i=0;i<32;++i){__half z=kind<2?__hmul(x[i],inv):x[i];if(kind==0)z=__hmul(z,scale[head]);published[v*32+i]=pubh(z);}}
__global__ void global_scores(const __half*qkv,__half*scores,int m){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=32*m*m)return;int head=i/(m*m),row=(i/m)%m,col=i%m;const __half*q=qkv+(head*m+row)*32,*k=qkv+32*m*32+(head*m+col)*32;float sum=0;for(int j=0;j<32;j+=2)sum=amd_mixed_dot(__halves2half2(q[j],q[j+1]),__halves2half2(k[j],k[j+1]),sum,false);scores[i]=__float2half(fminf(3.f,fmaxf(-3.f,__half2float(__float2half(sum)))));}
__global__ void global_probs(const __half*scores,__half*probs,int m){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=32*m)return;__half total=__float2half(0);for(int j=0;j<m;j+=2){unsigned bits[2];for(int p=0;p<2;++p){__half a=__hfma(scores[row*m+j+p],__float2half(.044921875f),__float2half(1.30078125f));a=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(a))));bits[p]=__half_as_ushort(a);}unsigned t=((bits[0]|bits[1]<<16)<<5)+0x7ff88000u;probs[row*m+j]=__ushort_as_half(t&65535);probs[row*m+j+1]=__ushort_as_half(t>>16);total=__hadd(total,probs[row*m+j]);total=__hadd(total,probs[row*m+j+1]);}__half inv=__float2half(1.f/__half2float(total));for(int j=0;j<m;++j)probs[row*m+j]=pubh(__hmul(probs[row*m+j],inv));}
__global__ void global_attend(const __half*probs,const __half*qkv,__half*out,int m){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m*1024)return;int row=i/1024,head=(i%1024)/32,ch=i%32;float sum=0;for(int j=0;j<m;++j)sum=fmaf(__half2float(probs[(head*m+row)*m+j]),__half2float(qkv[(2*32*m+head*m+j)*32+ch]),sum);out[i]=pubh(__float2half(sum));}
// One workgroup owns a global head/row. Scores and published probabilities stay
// in LDS. Thread zero preserves the original sequential half reduction; the
// first 32 lanes preserve the ascending FMA order of the value reduction.
__global__ void global_attention_fused(const __half*qkv,__half*out,int m){
    __shared__ __half q[32];
    extern __shared__ __half probabilities[];
    const int group=blockIdx.x,head=group/m,row=group%m,tid=threadIdx.x;
    if(group>=32*m)return;
    if(tid<32)q[tid]=qkv[(head*m+row)*32+tid];
    __syncthreads();
    for(int col=tid;col<m;col+=blockDim.x){
        const __half*k=qkv+(32*m+head*m+col)*32;float sum=0;
        for(int channel=0;channel<32;channel+=2)
            sum=amd_mixed_dot(__halves2half2(q[channel],q[channel+1]),
                              __halves2half2(k[channel],k[channel+1]),sum,false);
        probabilities[col]=__float2half(fminf(3.f,fmaxf(-3.f,__half2float(__float2half(sum)))));
    }
    __syncthreads();
    if(tid==0){
        __half total=__float2half(0);
        for(int col=0;col<m;col+=2){
            unsigned bits[2];
            for(int pair=0;pair<2;++pair){
                __half value=__hfma(probabilities[col+pair],__float2half(.044921875f),__float2half(1.30078125f));
                value=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(value))));
                bits[pair]=__half_as_ushort(value);
            }
            const unsigned transformed=((bits[0]|bits[1]<<16)<<5)+0x7ff88000u;
            probabilities[col]=__ushort_as_half(transformed&65535);
            probabilities[col+1]=__ushort_as_half(transformed>>16);
            total=__hadd(total,probabilities[col]);
            total=__hadd(total,probabilities[col+1]);
        }
        const __half inverse=__float2half(1.f/__half2float(total));
        for(int col=0;col<m;++col)probabilities[col]=pubh(__hmul(probabilities[col],inverse));
    }
    __syncthreads();
    if(tid<32){
        float sum=0;
#if defined(NR_ATTENTION_DOT2) && NR_ATTENTION_DOT2
        for(int col=0;col<m;col+=2)
            sum=amd_mixed_dot(__halves2half2(probabilities[col],probabilities[col+1]),
                __halves2half2(qkv[(2*32*m+head*m+col)*32+tid],
                              qkv[(2*32*m+head*m+col+1)*32+tid]),sum,false);
#else
        for(int col=0;col<m;++col)
            sum=fmaf(__half2float(probabilities[col]),
                     __half2float(qkv[(2*32*m+head*m+col)*32+tid]),sum);
#endif
        out[(row*32+head)*32+tid]=pubh(__float2half(sum));
    }
}

static const std::string& int8_capture_folder(){
    static const std::string folder=[](){char value[4096];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_INT8_CAPTURE"))throw std::runtime_error("Invalid INT8 capture path");return size&&value[0]?std::string(value):std::string();}();
    return folder;
}
static bool int8_capture_requested(int block){
    return (block==2||block==31)&&!int8_capture_folder().empty();
}

// Diagnostic-only real graph activations. NRW8ACT1 stores little-endian
// version/block/rows/columns followed by row-major FP16 values.
static void capture_int8_activation(int block,const char*name,HalfBuffer&buffer,
                                    unsigned rows,unsigned columns){
    if(!int8_capture_requested(block))return;
    if(buffer.count!=size_t(rows)*columns)throw std::runtime_error("INT8 capture shape mismatch");
    const auto values=buffer.read();
    const unsigned header[4]={1,unsigned(block),rows,columns};
    const std::string path=int8_capture_folder()+"/block"+std::to_string(block)+"_"+name+".nract";
    static_cast<void>(std::remove(path.c_str()));
    std::ofstream output(path,std::ios::binary);
    if(!output.write("NRW8ACT1",8)||
       !output.write(reinterpret_cast<const char*>(header),sizeof(header))||
       !output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(__half)))
        throw std::runtime_error("Unable to write INT8 activation capture");
}

constexpr size_t Int8GuardBytes=256;
template<class T>struct AlignedInt8Buffer{
    unsigned char*allocation=nullptr;T*data=nullptr;size_t count=0;
    explicit AlignedInt8Buffer(size_t elements,hipStream_t stream):count(elements){
        const size_t payload=count*sizeof(T);HIP_CHECK(hipMalloc(&allocation,payload+2*Int8GuardBytes));
        data=reinterpret_cast<T*>(allocation+Int8GuardBytes);
        if(reinterpret_cast<std::uintptr_t>(data)%Int8GuardBytes)throw std::runtime_error("unaligned INT8 buffer");
        HIP_CHECK(hipMemsetAsync(allocation,0xa5,Int8GuardBytes,stream));
        HIP_CHECK(hipMemsetAsync(allocation+Int8GuardBytes+payload,0xa5,Int8GuardBytes,stream));
    }
    AlignedInt8Buffer(const AlignedInt8Buffer&)=delete;
    ~AlignedInt8Buffer(){if(allocation&&hipFree(allocation)!=hipSuccess)cleanup_failed=true;}
    bool release()noexcept{if(!allocation)return true;if(hipFree(allocation)!=hipSuccess)return false;allocation=nullptr;data=nullptr;count=0;return true;}
    size_t payload_bytes()const{return count*sizeof(T);}
};

__global__ void validate_int8_guards(const unsigned char*a,size_t an,const unsigned char*b,size_t bn,
                                     const unsigned char*c,size_t cn,const unsigned char*d,size_t dn,
                                     const unsigned char*e,size_t en,const unsigned char*f,size_t fn,
                                     const unsigned char*g,size_t gn,unsigned*status){
    for(unsigned i=threadIdx.x;i<Int8GuardBytes;i+=blockDim.x){
    const bool bad=a[i]!=0xa5||a[Int8GuardBytes+an+i]!=0xa5||
        b[i]!=0xa5||b[Int8GuardBytes+bn+i]!=0xa5||c[i]!=0xa5||c[Int8GuardBytes+cn+i]!=0xa5||
        (d&&(d[i]!=0xa5||d[Int8GuardBytes+dn+i]!=0xa5))||e[i]!=0xa5||e[Int8GuardBytes+en+i]!=0xa5||
        (f&&(f[i]!=0xa5||f[Int8GuardBytes+fn+i]!=0xa5))||g[i]!=0xa5||g[Int8GuardBytes+gn+i]!=0xa5;
    if(bad)atomicOr(status,2u);
    }
}
__global__ void validate_one_int8_guard(const unsigned char*a,size_t bytes,unsigned*status){
    for(unsigned i=threadIdx.x;i<Int8GuardBytes;i+=blockDim.x){
    if(a[i]!=0xa5||a[Int8GuardBytes+bytes+i]!=0xa5)atomicOr(status,2u);
    }
}

enum class Int8Role:unsigned{FfnExpand,FfnContract,Qkv,AttentionProjection,Downsample,GlobalProjection,BridgeProjection,UpsampleProjection,SplitFirst,SplitOutput,SplitExpand,SplitProject,BranchedExpand,BranchedOutput};
enum class Int8Scope:unsigned{Block31,GlobalExpand,GlobalDense,MixedV1,Screened,Window512Attention,Window512Dense,MixedV2,Window512Split,MixedV3,BranchedExpand,MixedV4,JuniorOutput,JuniorAttention,JuniorDense,MixedV5};

struct GroupedInt8Matrix{
    unsigned k=0,n=0,groupSize=0,groups=0,batches=1,epilogue=0;bool publish=false,packed=false,hardwarePacked=false;
    const __half*source=nullptr;
    std::unique_ptr<AlignedInt8Buffer<std::int8_t>>weight;
    std::unique_ptr<AlignedInt8Buffer<float>>scale;
    GroupedInt8Matrix(unsigned valueK,unsigned valueN,unsigned groupSize,unsigned valueEpilogue,bool valuePublish,hipStream_t stream,unsigned valueBatches=1):
        k(valueK),n(valueN),groupSize(groupSize),groups(groupSize==1?1:valueK/groupSize),batches(valueBatches),epilogue(valueEpilogue),publish(valuePublish),
        weight(std::make_unique<AlignedInt8Buffer<std::int8_t>>(size_t(batches)*k*n,stream)),
        scale(std::make_unique<AlignedInt8Buffer<float>>(size_t(batches)*groups*n,stream)){
        if(!k||!n||(n&31)||(groupSize!=1&&(k%groupSize)))throw std::runtime_error("grouped INT8 matrix shape");
        if(!batches||epilogue>4)throw std::runtime_error("grouped INT8 epilogue");
    }
    size_t reserved_bytes()const{return weight->payload_bytes()+scale->payload_bytes()+2*2*Int8GuardBytes;}
    bool release()noexcept{bool ok=true;ok=scale->release()&&ok;ok=weight->release()&&ok;return ok;}
};

struct PackedBranchProjection{
    unsigned channels=0,batches=0;const __half*source=nullptr;
    std::unique_ptr<AlignedInt8Buffer<__half2>>weight;
    PackedBranchProjection(unsigned valueChannels,hipStream_t stream):
        channels(valueChannels),batches(4*valueChannels/32),
        weight(std::make_unique<AlignedInt8Buffer<__half2>>(size_t(batches)*16*32,stream)){}
    size_t reserved_bytes()const{return weight->payload_bytes()+2*Int8GuardBytes;}
    bool release()noexcept{return weight->release();}
};

struct PackedC32Attention{
    const __half*qkvSource=nullptr,*projectionSource=nullptr;
    std::unique_ptr<AlignedInt8Buffer<__half2>>qkv,projection;
    PackedC32Attention(hipStream_t stream):
        qkv(std::make_unique<AlignedInt8Buffer<__half2>>(16*96,stream)),
        projection(std::make_unique<AlignedInt8Buffer<__half2>>(16*32,stream)){}
    size_t reserved_bytes()const{return qkv->payload_bytes()+projection->payload_bytes()+4*Int8GuardBytes;}
    bool release()noexcept{bool ok=true;ok=projection->release()&&ok;ok=qkv->release()&&ok;return ok;}
};

struct GroupedInt8Plan{
    hipStream_t stream=nullptr;unsigned groupSize=0;Int8Scope scope=Int8Scope::Block31;
    bool validateGuards=false,wavePack=false,tile64=true,row32=true,group128Global=false,hardwareLayout=false,qkvHardwareLayout=false,branchFp16Packed=false,c32AttentionWave=false,prefetch64=false;
    unsigned splitResident=4;
    std::unordered_map<unsigned,std::unique_ptr<GroupedInt8Matrix>>matrices;
    std::unordered_map<unsigned,std::unique_ptr<PackedBranchProjection>>branchProjects;
    std::unordered_map<unsigned,std::unique_ptr<PackedC32Attention>>c32Attention;
    std::unique_ptr<AlignedInt8Buffer<std::int8_t>>activation;
    std::unique_ptr<AlignedInt8Buffer<float>>activationScale;
    std::unique_ptr<AlignedInt8Buffer<unsigned>>status;
    std::unique_ptr<AlignedInt8Buffer<std::int8_t>>handoffCodes;
    std::unique_ptr<AlignedInt8Buffer<float>>handoffScales;
    size_t activationElements=0,activationScaleElements=0,launches=0;
    GroupedInt8Plan(hipStream_t value,unsigned valueGroupSize,Int8Scope valueScope):stream(value),groupSize(valueGroupSize),scope(valueScope),
        status(std::make_unique<AlignedInt8Buffer<unsigned>>(1,value)){
        if(!stream||(groupSize!=1&&groupSize!=32&&groupSize!=64))throw std::runtime_error("grouped INT8 plan configuration");
        if(groupSize==1&&scope!=Int8Scope::Block31)throw std::runtime_error("row/column W8A8 supports block31 only");
        wavePack=groupSize==64;
        char valueText[16];size_t valueSize=0;if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_INT8_VALIDATE_GUARDS"))throw std::runtime_error("Invalid INT8 guard switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))validateGuards=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_INT8_VALIDATE_GUARDS must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_INT8_WAVE_PACK"))throw std::runtime_error("Invalid INT8 pack switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))wavePack=true;else if(!std::strcmp(valueText,"0"))wavePack=false;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_INT8_WAVE_PACK must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_INT8_TILE64"))throw std::runtime_error("Invalid INT8 tile switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))tile64=true;else if(!std::strcmp(valueText,"0"))tile64=false;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_INT8_TILE64 must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_INT8_ROW32"))throw std::runtime_error("Invalid INT8 row tile switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))row32=true;else if(!std::strcmp(valueText,"0"))row32=false;else throw std::runtime_error("RDNA2_INT8_ROW32 must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_GLOBAL_GROUP128"))throw std::runtime_error("Invalid global group128 switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))group128Global=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_GLOBAL_GROUP128 must be 0 or 1");}
        if(group128Global&&groupSize!=64)throw std::runtime_error("global group128 requires the group64 base plan");
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_INT8_HW_LAYOUT"))throw std::runtime_error("Invalid INT8 hardware layout switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))hardwareLayout=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_INT8_HW_LAYOUT must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_QKV_HW_LAYOUT"))throw std::runtime_error("Invalid QKV hardware layout switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))qkvHardwareLayout=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_QKV_HW_LAYOUT must be 0 or 1");}
        if(qkvHardwareLayout&&!hardwareLayout)throw std::runtime_error("QKV hardware layout requires INT8 hardware layout");
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_BRANCH_FP16_PACKED"))throw std::runtime_error("Invalid branch FP16 layout switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))branchFp16Packed=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_BRANCH_FP16_PACKED must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_C32_ATTENTION_WAVE"))throw std::runtime_error("Invalid C32 attention wave switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))c32AttentionWave=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_C32_ATTENTION_WAVE must be 0 or 1");}
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_INT8_PREFETCH"))throw std::runtime_error("Invalid INT8 prefetch switch");
        if(valueSize&&valueText[0]){if(!std::strcmp(valueText,"1"))prefetch64=true;else if(std::strcmp(valueText,"0"))throw std::runtime_error("RDNA2_INT8_PREFETCH must be 0 or 1");}
        if(prefetch64&&!hardwareLayout)throw std::runtime_error("INT8 prefetch requires hardware layout");
        if(NrExecution::ReadSetting(&valueSize,valueText,sizeof(valueText),"RDNA2_SPLIT_RESIDENT"))throw std::runtime_error("Invalid split resident switch");
        if(valueSize&&valueText[0]){
            if(std::strlen(valueText)!=1||valueText[0]<'0'||valueText[0]>'4')throw std::runtime_error("RDNA2_SPLIT_RESIDENT must be 0 through 4");
            splitResident=unsigned(valueText[0]-'0');
        }
        HIP_CHECK(hipMemsetAsync(status->data,0,sizeof(unsigned),stream));
    }
    static unsigned key(unsigned block,Int8Role role){return block*16+unsigned(role);}
    bool enabled(unsigned block,Int8Role role)const{
        if(scope==Int8Scope::Block31)return block==31&&role==Int8Role::FfnExpand;
        if(block>=31&&block<=38){
            if(scope==Int8Scope::GlobalExpand)return role==Int8Role::FfnExpand;
            if(scope==Int8Scope::GlobalDense||scope==Int8Scope::MixedV1||scope==Int8Scope::Screened||scope==Int8Scope::MixedV2||scope==Int8Scope::MixedV3||scope==Int8Scope::MixedV4||scope==Int8Scope::MixedV5)
                return role==Int8Role::FfnExpand||role==Int8Role::FfnContract||role==Int8Role::Qkv||role==Int8Role::AttentionProjection;
        }
        if((scope==Int8Scope::MixedV1||scope==Int8Scope::MixedV2||scope==Int8Scope::MixedV3||scope==Int8Scope::MixedV4||scope==Int8Scope::MixedV5)&&
           ((block==39&&role==Int8Role::BridgeProjection)||(block==48&&role==Int8Role::UpsampleProjection)))return true;
        if(scope==Int8Scope::Screened)return(block==22&&role==Int8Role::Downsample)||(block==30&&role==Int8Role::GlobalProjection)||
            (block==39&&role==Int8Role::BridgeProjection)||(block==48&&role==Int8Role::UpsampleProjection);
        const bool branched=(block>=5&&block<=22)||(block>=48&&block<=65);
        if(branched){
            if(scope==Int8Scope::BranchedExpand||scope==Int8Scope::MixedV4)return role==Int8Role::BranchedExpand;
            if(scope==Int8Scope::JuniorOutput)return role==Int8Role::BranchedOutput;
            if(scope==Int8Scope::JuniorAttention)return role==Int8Role::Qkv||role==Int8Role::AttentionProjection;
            if(scope==Int8Scope::JuniorDense)return role==Int8Role::BranchedOutput||role==Int8Role::Qkv||role==Int8Role::AttentionProjection;
            if(scope==Int8Scope::MixedV5)return role==Int8Role::BranchedExpand||role==Int8Role::BranchedOutput||role==Int8Role::Qkv||role==Int8Role::AttentionProjection;
        }
        const bool window512=(block>=23&&block<=30)||(block>=40&&block<=47);
        if(!window512)return false;
        if(scope==Int8Scope::Window512Attention)return role==Int8Role::Qkv||role==Int8Role::AttentionProjection;
        if(scope==Int8Scope::Window512Dense||scope==Int8Scope::MixedV2)
            return role==Int8Role::Qkv||role==Int8Role::AttentionProjection||role==Int8Role::SplitFirst||role==Int8Role::SplitOutput;
        if(scope==Int8Scope::Window512Split)return role==Int8Role::SplitExpand||role==Int8Role::SplitProject;
        if(scope==Int8Scope::MixedV3||scope==Int8Scope::MixedV4||scope==Int8Scope::MixedV5)
            return role==Int8Role::Qkv||role==Int8Role::AttentionProjection||role==Int8Role::SplitFirst||role==Int8Role::SplitOutput||
                role==Int8Role::SplitExpand||role==Int8Role::SplitProject;
        return false;
    }
    void ensure_workspace(unsigned m,unsigned k){
        const size_t values=size_t(m)*k,scales=size_t(m)*(groupSize==1?1:k/groupSize);
        if(values>activationElements){activation=std::make_unique<AlignedInt8Buffer<std::int8_t>>(values,stream);activationElements=values;}
        if(scales>activationScaleElements){activationScale=std::make_unique<AlignedInt8Buffer<float>>(scales,stream);activationScaleElements=scales;}
    }
    template<unsigned GroupSize,unsigned TileRows=16,unsigned TileK=32,int Layout=0,int Unroll=TileK/4>void launch_dot_tiled(GroupedInt8Matrix&entry,const __half*skip,const __half*cosine,__half*output,unsigned m){
        const dim3 grid(entry.n/32,(m+TileRows-1)/TileRows),threads(32,TileRows/4);
        if(entry.epilogue==0)rdna2_nr::dot4_grouped_w8a8<GroupSize,4,0,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else if(entry.epilogue==1)rdna2_nr::dot4_grouped_w8a8<GroupSize,4,1,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else if(entry.epilogue==2)rdna2_nr::dot4_grouped_w8a8<GroupSize,4,2,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else rdna2_nr::dot4_grouped_w8a8<GroupSize,4,3,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
    }
    template<unsigned GroupSize>void launch_dot(GroupedInt8Matrix&entry,const __half*skip,const __half*cosine,__half*output,unsigned m){
        if constexpr(GroupSize==64){
            if(entry.hardwarePacked){
                const bool usePrefetch=prefetch64&&!(m<256&&entry.k==1024&&entry.n==1024);
                if(m>=32&&usePrefetch){
                    const dim3 grid(entry.n/32,(m+31)/32),threads(32,8);
                    if(entry.epilogue==0)rdna2_nr::dot4_grouped_w8a8_prefetch64<0,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
                    else if(entry.epilogue==1)rdna2_nr::dot4_grouped_w8a8_prefetch64<1,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
                    else if(entry.epilogue==2)rdna2_nr::dot4_grouped_w8a8_prefetch64<2,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
                    else rdna2_nr::dot4_grouped_w8a8_prefetch64<3,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
                }else if(m>=32)launch_dot_tiled<64,32,64,2,4>(entry,skip,cosine,output,m);
                else launch_dot_tiled<64,16,64,2,4>(entry,skip,cosine,output,m);
                return;
            }
        }
        if constexpr(GroupSize==64||GroupSize==128){
            if(tile64&&entry.k>=1024){
                if(row32&&m>=32){launch_dot_tiled<GroupSize,32,64>(entry,skip,cosine,output,m);return;}
                launch_dot_tiled<GroupSize,16,64>(entry,skip,cosine,output,m);return;
            }
        }
        launch_dot_tiled<GroupSize,16,32>(entry,skip,cosine,output,m);
    }
    template<unsigned GroupSize,unsigned TileK=32,int PackedWeights=0,int DotUnroll=TileK/4>void launch_qkv_dot_tiled(
        GroupedInt8Matrix&entry,const __half*scale,__half*output,unsigned m,
        unsigned heads,unsigned tokensPerWindow){
        const dim3 grid(3*heads,(m+15)/16),threads(32,4);
        rdna2_nr::dot4_grouped_w8a8_qkv_normalize<GroupSize,4,TileK,PackedWeights,DotUnroll><<<grid,threads,0,stream>>>(
            activation->data,entry.weight->data,activationScale->data,entry.scale->data,
            scale,output,m,entry.k,heads,tokensPerWindow);
    }
    template<unsigned GroupSize>void launch_qkv_dot(GroupedInt8Matrix&entry,
        const __half*scale,__half*output,unsigned m,unsigned heads,unsigned tokensPerWindow){
        if constexpr(GroupSize==64){
            if(entry.hardwarePacked){
                if(tile64&&entry.k>=1024)launch_qkv_dot_tiled<64,64,2,4>(entry,scale,output,m,heads,tokensPerWindow);
                else launch_qkv_dot_tiled<64,32,2,4>(entry,scale,output,m,heads,tokensPerWindow);
                return;
            }
            if(tile64&&entry.k>=1024){
                launch_qkv_dot_tiled<GroupSize,64>(entry,scale,output,m,heads,tokensPerWindow);return;
            }
        }
        launch_qkv_dot_tiled<GroupSize,32>(entry,scale,output,m,heads,tokensPerWindow);
    }
    template<unsigned GroupSize>void pack_rows(const __half*input,unsigned rows,unsigned k){
        stage_event("pack",true);
        if(wavePack)rdna2_nr::quantize_groups_wave_w8a8<GroupSize><<<(rows*(k/GroupSize)+3)/4,128,0,stream>>>(input,activation->data,activationScale->data,nullptr,rows,k,status->data);
        else rdna2_nr::quantize_row_groups_w8a8<GroupSize><<<rows,256,0,stream>>>(input,activation->data,activationScale->data,nullptr,rows,k,status->data);
        stage_event("pack",false);
    }
    template<unsigned GroupSize>void pack_window(const __half*input,unsigned rows,unsigned k,unsigned h,unsigned w,unsigned top,unsigned left,unsigned paddedWidth){
        stage_event("pack",true);
        if(wavePack)rdna2_nr::quantize_groups_wave_w8a8<GroupSize,true><<<(rows*(k/GroupSize)+3)/4,128,0,stream>>>(input,activation->data,activationScale->data,nullptr,rows,k,status->data,h,w,top,left,paddedWidth);
        else rdna2_nr::quantize_window_groups_w8a8<GroupSize><<<rows,256,0,stream>>>(input,activation->data,activationScale->data,rows,k,h,w,top,left,paddedWidth,status->data);
        stage_event("pack",false);
    }
    template<unsigned GroupSize>void launch_grouped(GroupedInt8Matrix&entry,const __half*input,const __half*skip,const __half*cosine,__half*output,unsigned m){
        pack_rows<GroupSize>(input,m,entry.k);
        launch_dot<GroupSize>(entry,skip,cosine,output,m);
    }
    GroupedInt8Matrix& prepare_with_group(unsigned block,Int8Role role,const __half*matrix,unsigned k,unsigned n,
                                           unsigned epilogue,bool publish,unsigned matrixGroup,bool force=false){
        if((!force&&!enabled(block,role))||!matrix||!k||!n)throw std::runtime_error("grouped INT8 matrix arguments");
        const unsigned id=key(block,role);auto&owner=matrices[id];
        if(!owner)owner=std::make_unique<GroupedInt8Matrix>(k,n,matrixGroup,epilogue,publish,stream);
        auto&entry=*owner;if(entry.k!=k||entry.n!=n||entry.groupSize!=matrixGroup||entry.epilogue!=epilogue||entry.publish!=publish)throw std::runtime_error("grouped INT8 matrix identity changed");
        if(entry.source&&entry.source!=matrix)throw std::runtime_error("grouped INT8 weight pointer changed");
        if(!entry.packed){
            entry.hardwarePacked=hardwareLayout&&matrixGroup==64&&
                ((block>=31&&block<=38&&(role==Int8Role::FfnExpand||role==Int8Role::FfnContract||role==Int8Role::AttentionProjection))||
                 (qkvHardwareLayout&&role==Int8Role::Qkv));
            if(matrixGroup==1)rdna2_nr::quantize_columns_w8a8<<<n,128,0,stream>>>(matrix,entry.weight->data,entry.scale->data,nullptr,k,n,status->data);
            else if(matrixGroup==16)rdna2_nr::quantize_weight_groups_w8a8<16><<<dim3((n+63)/64,k/16),64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,nullptr,k,n,status->data);
            else if(matrixGroup==32)rdna2_nr::quantize_weight_groups_w8a8<32><<<dim3((n+63)/64,k/32),64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,nullptr,k,n,status->data);
            else if(matrixGroup==64){
                if(entry.hardwarePacked)rdna2_nr::quantize_weight_groups_w8a8<64,true><<<dim3((n+63)/64,k/64),64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,nullptr,k,n,status->data);
                else rdna2_nr::quantize_weight_groups_w8a8<64><<<dim3((n+63)/64,k/64),64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,nullptr,k,n,status->data);
            }
            else if(matrixGroup==128)rdna2_nr::quantize_weight_groups_w8a8<128><<<dim3((n+63)/64,k/128),64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,nullptr,k,n,status->data);
            else throw std::runtime_error("unsupported grouped INT8 matrix group");
            entry.source=matrix;entry.packed=true;
        }
        return entry;
    }
    GroupedInt8Matrix& prepare(unsigned block,Int8Role role,const __half*matrix,unsigned k,unsigned n,unsigned epilogue,bool publish){
        const bool wider=group128Global&&block>=31&&block<=38&&
            (role==Int8Role::FfnExpand||role==Int8Role::FfnContract||role==Int8Role::AttentionProjection);
        return prepare_with_group(block,role,matrix,k,n,epilogue,publish,wider?128:groupSize);
    }
    GroupedInt8Matrix& prepare_batched(unsigned block,Int8Role role,const __half*matrix,unsigned batches,unsigned k,unsigned n,unsigned epilogue){
        if(!enabled(block,role)||!matrix||!batches||!k||!n||groupSize==1)throw std::runtime_error("batched grouped INT8 matrix arguments");
        const unsigned id=key(block,role);auto&owner=matrices[id];
        if(!owner)owner=std::make_unique<GroupedInt8Matrix>(k,n,groupSize,epilogue,false,stream,batches);
        auto&entry=*owner;
        if(entry.k!=k||entry.n!=n||entry.batches!=batches||entry.epilogue!=epilogue||entry.publish)throw std::runtime_error("batched grouped INT8 matrix identity changed");
        if(entry.source&&entry.source!=matrix)throw std::runtime_error("batched grouped INT8 weight pointer changed");
        if(!entry.packed){
            const dim3 grid((n+63)/64,k/groupSize,batches);
            if(groupSize==32)rdna2_nr::quantize_batched_weight_groups_w8a8<32><<<grid,64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,batches,k,n,status->data);
            else rdna2_nr::quantize_batched_weight_groups_w8a8<64><<<grid,64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,batches,k,n,status->data);
            entry.source=matrix;entry.packed=true;
        }
        return entry;
    }
    void finish(GroupedInt8Matrix&entry){
        if(validateGuards)validate_int8_guards<<<1,32,0,stream>>>(entry.weight->allocation,entry.weight->payload_bytes(),activation->allocation,activation->payload_bytes(),
            entry.scale->allocation,entry.scale->payload_bytes(),nullptr,0,activationScale->allocation,activationScale->payload_bytes(),
            nullptr,0,status->allocation,status->payload_bytes(),status->data);
        sync_gpu();++launches;
    }
    void launch_c32_ffn(unsigned block,const __half*input,const __half*expand,const __half*contract,
                        const __half*cosine,__half*output,unsigned rows,unsigned matrixGroup){
        if(matrixGroup!=16&&matrixGroup!=32)throw std::runtime_error("unsupported C32 INT8 group");
        auto&first=prepare_with_group(block,Int8Role::FfnExpand,expand,32,128,1,true,matrixGroup,true);
        auto&second=prepare_with_group(block,Int8Role::FfnContract,contract,128,32,3,false,matrixGroup,true);
        const dim3 grid((rows+15)/16),threads(32,4);
        if(matrixGroup==16)fused_c32_ffn_w8a8<16><<<grid,threads,0,stream>>>(input,first.weight->data,
            first.scale->data,second.weight->data,second.scale->data,cosine,output,rows,status->data);
        else fused_c32_ffn_w8a8<32><<<grid,threads,0,stream>>>(input,first.weight->data,
            first.scale->data,second.weight->data,second.scale->data,cosine,output,rows,status->data);
        if(validateGuards)validate_int8_guards<<<1,32,0,stream>>>(first.weight->allocation,first.weight->payload_bytes(),
            second.weight->allocation,second.weight->payload_bytes(),first.scale->allocation,first.scale->payload_bytes(),
            second.scale->allocation,second.scale->payload_bytes(),status->allocation,status->payload_bytes(),nullptr,0,
            status->allocation,status->payload_bytes(),status->data);
        sync_gpu();++launches;
    }
    void launch(unsigned block,Int8Role role,const __half*input,const __half*matrix,const __half*skip,const __half*cosine,
                __half*output,unsigned m,unsigned k,unsigned n,unsigned epilogue,bool publish=false){
        if(!input||!output||!m)throw std::runtime_error("grouped INT8 launch arguments");
        auto&entry=prepare(block,role,matrix,k,n,epilogue,publish);ensure_workspace(m,k);
        if(groupSize==1){
            if(epilogue!=1)throw std::runtime_error("row/column W8A8 epilogue unsupported");
            rdna2_nr::quantize_rows_w8a8<<<m,128,0,stream>>>(input,activation->data,activationScale->data,nullptr,m,k,status->data);
            rdna2_nr::dot4_gate_publish_w8a8<4><<<dim3(n/32,(m+15)/16),dim3(32,4),0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,m,k,n);
        }else if(entry.groupSize==32)launch_grouped<32>(entry,input,skip,cosine,output,m);
        else if(entry.groupSize==64)launch_grouped<64>(entry,input,skip,cosine,output,m);
        else if(entry.groupSize==128)launch_grouped<128>(entry,input,skip,cosine,output,m);
        else throw std::runtime_error("unsupported grouped INT8 launch group");
        finish(entry);
    }
    void launch_batched(unsigned block,Int8Role role,const __half*input,const __half*matrix,__half*output,
                        unsigned rows,unsigned batches,unsigned k,unsigned n,unsigned epilogue){
        if(!input||!output||!rows||!batches||(epilogue!=2&&epilogue!=4))throw std::runtime_error("batched grouped INT8 launch arguments");
        auto&entry=prepare_batched(block,role,matrix,batches,k,n,epilogue);ensure_workspace(rows*batches,k);
        const unsigned batchRows=rows*batches;
        const dim3 grid(n/32,(rows+15)/16,batches),threads(32,4);
        if(groupSize==32){
            pack_rows<32>(input,batchRows,k);
            if(epilogue==2)rdna2_nr::dot4_batched_grouped_w8a8<32,4,2><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
            else rdna2_nr::dot4_batched_grouped_w8a8<32,4,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
        }else{
            pack_rows<64>(input,batchRows,k);
            if(epilogue==2)rdna2_nr::dot4_batched_grouped_w8a8<64,4,2><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
            else rdna2_nr::dot4_batched_grouped_w8a8<64,4,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
        }
        finish(entry);
    }
    bool launch_split_resident(unsigned block,const __half*input,const __half*expand,
                               const __half*project,__half*output,unsigned rows){
        if(!splitResident||groupSize!=64||!input||!expand||!project||!output||!rows||
           !((block>=23&&block<=30)||(block>=40&&block<=47))||
           !enabled(block,Int8Role::SplitExpand)||!enabled(block,Int8Role::SplitProject))return false;
        auto&first=prepare_batched(block,Int8Role::SplitExpand,expand,8,64,256,4);
        auto&second=prepare_batched(block,Int8Role::SplitProject,project,8,256,64,2);
        ensure_workspace(rows*8,64);
        pack_rows<64>(input,rows*8,64);
        if(splitResident==1)rdna2_nr::split_ffn_resident_w8a8<16,4,true><<<dim3((rows+15)/16,8),dim3(32,4),0,stream>>>(
            activation->data,activationScale->data,first.weight->data,first.scale->data,
            second.weight->data,second.scale->data,output,rows,status->data);
        else if(splitResident==2)rdna2_nr::split_ffn_resident_w8a8<16,4,false><<<dim3((rows+15)/16,8),dim3(32,4),0,stream>>>(
            activation->data,activationScale->data,first.weight->data,first.scale->data,
            second.weight->data,second.scale->data,output,rows,status->data);
        else if(splitResident==3)rdna2_nr::split_ffn_resident_w8a8<8,4,true><<<dim3((rows+7)/8,8),dim3(32,4),0,stream>>>(
            activation->data,activationScale->data,first.weight->data,first.scale->data,
            second.weight->data,second.scale->data,output,rows,status->data);
        else rdna2_nr::split_ffn_resident_w8a8<16,8,true><<<dim3((rows+15)/16,8),dim3(32,8),0,stream>>>(
            activation->data,activationScale->data,first.weight->data,first.scale->data,
            second.weight->data,second.scale->data,output,rows,status->data);
        if(validateGuards)validate_int8_guards<<<1,32,0,stream>>>(
            first.weight->allocation,first.weight->payload_bytes(),first.scale->allocation,first.scale->payload_bytes(),
            second.weight->allocation,second.weight->payload_bytes(),second.scale->allocation,second.scale->payload_bytes(),
            activation->allocation,activation->payload_bytes(),activationScale->allocation,activationScale->payload_bytes(),
            status->allocation,status->payload_bytes(),status->data);
        sync_gpu();++launches;
        return true;
    }
    void launch_broadcast_batched(unsigned block,const __half*input,const __half*matrix,__half*output,
                                  unsigned rows,unsigned batches,unsigned k,unsigned n){
        if(!input||!output||!rows||!batches||groupSize==1)throw std::runtime_error("broadcast batched grouped INT8 launch arguments");
        auto&entry=prepare_batched(block,Int8Role::BranchedExpand,matrix,batches,k,n,1);ensure_workspace(rows,k);
        const dim3 grid(n/32,(rows+15)/16,batches),threads(32,4);
        if(groupSize==32){
            pack_rows<32>(input,rows,k);
            rdna2_nr::dot4_broadcast_batched_gate_w8a8<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
        }else{
            pack_rows<64>(input,rows,k);
            rdna2_nr::dot4_broadcast_batched_gate_w8a8<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
        }
        finish(entry);
    }
    void launch_branched_resident(unsigned block,const __half*input,const __half*expand,
                                  const __half*project,__half*output,unsigned rows,unsigned channels){
        if(!input||!expand||!project||!output||!rows||groupSize!=64||(channels!=64&&channels!=128&&channels!=256))
            throw std::runtime_error("resident branched W8A8 arguments");
        const unsigned batches=4*channels/32;
        auto&entry=prepare_batched(block,Int8Role::BranchedExpand,expand,batches,channels,32,1);
        ensure_workspace(rows,channels);pack_rows<64>(input,rows,channels);
        if(branchFp16Packed){
            auto&packed=branchProjects[block];
            if(!packed){
                packed=std::make_unique<PackedBranchProjection>(channels,stream);
                rdna2_nr::pack_branched_project_half2<<<(packed->weight->count+255)/256,256,0,stream>>>(
                    project,packed->weight->data,packed->batches);
                packed->source=project;
            }
            if(packed->channels!=channels||packed->source!=project)throw std::runtime_error("packed branch projection identity changed");
            rdna2_nr::dot4_branched_staged_w8a8<64,4,true><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,
                reinterpret_cast<const __half*>(packed->weight->data),output,rows,channels);
            if(validateGuards)validate_one_int8_guard<<<1,32,0,stream>>>(packed->weight->allocation,
                packed->weight->payload_bytes(),status->data);
        }else rdna2_nr::dot4_branched_resident_w8a8<64,4,false><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,project,output,rows,channels);
        finish(entry);
    }
    void launch_window_qkv(unsigned block,const __half*input,const __half*matrix,__half*output,
                           unsigned m,unsigned k,unsigned n,unsigned h,unsigned w,unsigned top,unsigned left,unsigned paddedWidth){
        if(!input||!output||!m||groupSize==1)throw std::runtime_error("grouped window QKV arguments");
        auto&entry=prepare(block,Int8Role::Qkv,matrix,k,n,0,false);ensure_workspace(m,k);
        if(groupSize==32){
            pack_window<32>(input,m,k,h,w,top,left,paddedWidth);
            launch_dot<32>(entry,nullptr,nullptr,output,m);
        }else{
            pack_window<64>(input,m,k,h,w,top,left,paddedWidth);
            launch_dot<64>(entry,nullptr,nullptr,output,m);
        }
        finish(entry);
    }
    std::pair<const __half*,const __half*>prepare_c32_attention(
        unsigned block,const __half*qkvWeight,const __half*projectionWeight){
        if(!c32AttentionWave||!qkvWeight||!projectionWeight)throw std::runtime_error("packed C32 attention arguments");
        auto&packed=c32Attention[block];
        if(!packed){
            packed=std::make_unique<PackedC32Attention>(stream);
            pack_c32_attention_half2<<<6,256,0,stream>>>(qkvWeight,packed->qkv->data,96);
            pack_c32_attention_half2<<<2,256,0,stream>>>(projectionWeight,packed->projection->data,32);
            packed->qkvSource=qkvWeight;packed->projectionSource=projectionWeight;
            if(validateGuards){
                validate_one_int8_guard<<<1,32,0,stream>>>(packed->qkv->allocation,packed->qkv->payload_bytes(),status->data);
                validate_one_int8_guard<<<1,32,0,stream>>>(packed->projection->allocation,packed->projection->payload_bytes(),status->data);
            }
        }
        if(packed->qkvSource!=qkvWeight||packed->projectionSource!=projectionWeight)
            throw std::runtime_error("packed C32 attention identity changed");
        return {reinterpret_cast<const __half*>(packed->qkv->data),
                reinterpret_cast<const __half*>(packed->projection->data)};
    }
    void launch_qkv_normalized(unsigned block,const __half*input,const __half*matrix,
                               const __half*scale,__half*output,unsigned m,unsigned k,
                               unsigned heads,unsigned tokensPerWindow){
        if(!input||!scale||!output||!m||groupSize==1)throw std::runtime_error("grouped normalized QKV arguments");
        auto&entry=prepare(block,Int8Role::Qkv,matrix,k,3*heads*32,0,false);ensure_workspace(m,k);
        if(groupSize==32){pack_rows<32>(input,m,k);launch_qkv_dot<32>(entry,scale,output,m,heads,tokensPerWindow);}
        else{pack_rows<64>(input,m,k);launch_qkv_dot<64>(entry,scale,output,m,heads,tokensPerWindow);}
        finish(entry);
    }
    void launch_window_qkv_normalized(unsigned block,const __half*input,const __half*matrix,
                                      const __half*scale,__half*output,unsigned m,unsigned k,
                                      unsigned heads,unsigned h,unsigned w,unsigned top,
                                      unsigned left,unsigned paddedWidth){
        if(!input||!scale||!output||!m||groupSize==1)throw std::runtime_error("grouped window normalized QKV arguments");
        auto&entry=prepare(block,Int8Role::Qkv,matrix,k,3*heads*32,0,false);ensure_workspace(m,k);
        if(groupSize==32){pack_window<32>(input,m,k,h,w,top,left,paddedWidth);launch_qkv_dot<32>(entry,scale,output,m,heads,64);}
        else{pack_window<64>(input,m,k,h,w,top,left,paddedWidth);launch_qkv_dot<64>(entry,scale,output,m,heads,64);}
        finish(entry);
    }
    void launch_window_projection(unsigned block,const __half*input,const __half*matrix,const __half*skip,const __half*cosine,__half*output,
                                  unsigned m,unsigned k,unsigned n,bool publish,unsigned h,unsigned w,unsigned top,unsigned left,unsigned paddedWidth){
        if(!input||!skip||!cosine||!output||!m||groupSize==1)throw std::runtime_error("grouped window projection arguments");
        auto&entry=prepare(block,Int8Role::AttentionProjection,matrix,k,n,3,publish);ensure_workspace(m,k);
        const dim3 grid(n/32,(m+15)/16),threads(32,4);
        if(groupSize==32){
            pack_rows<32>(input,m,k);
            rdna2_nr::dot4_grouped_w8a8_to_spatial<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);
        }else{
            pack_rows<64>(input,m,k);
            rdna2_nr::dot4_grouped_w8a8_to_spatial<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);
        }
        finish(entry);
    }

    // Two non-overlapping code/scale workspaces survive until the owning stream
    // has consumed them. Diagnostic observers use the original FP16 path below.
    bool launch_ffn_handoff(unsigned block,const __half*input,const __half*expand,
            const __half*contract,const __half*cosine,__half*hidden,__half*out,
            unsigned m,unsigned k,unsigned n,bool chain){
        if(groupSize!=64||group128Global||prefetch64||!hardwareLayout||!row32||!tile64||
           block<31||block>38||k<1024||n%64||
           !enabled(block,Int8Role::FfnExpand)||!enabled(block,Int8Role::FfnContract))return false;
        auto&first=prepare(block,Int8Role::FfnExpand,expand,k,n,1,true);
        auto&second=prepare(block,Int8Role::FfnContract,contract,n,k,3,false);
        ensure_workspace(m,k);
        const size_t count=size_t(m)*n;
        if(!handoffCodes||handoffCodes->count<count){
            handoffCodes=std::make_unique<AlignedInt8Buffer<std::int8_t>>(count,stream);
            handoffScales=std::make_unique<AlignedInt8Buffer<float>>(count/64,stream);
        }
        pack_rows<64>(input,m,k);
        rdna2_nr::dot4_global_expand_handoff<32,true,false><<<dim3(n/64,(m+31)/32),dim3(32,8),0,stream>>>(
            activation->data,first.weight->data,activationScale->data,first.scale->data,
            hidden,handoffCodes->data,handoffScales->data,m,k,n,status->data);

        finish(first);
        if(chain){
            rdna2_nr::dot4_global_residual_handoff<32,true><<<dim3(k/64,(m+31)/32),dim3(32,8),0,stream>>>(
                handoffCodes->data,second.weight->data,handoffScales->data,second.scale->data,
                out,activation->data,activationScale->data,m,n,k,status->data,input,cosine);
        }else{
            rdna2_nr::dot4_grouped_w8a8<64,4,3,32,32,64,2,4><<<dim3(k/32,(m+31)/32),dim3(32,8),0,stream>>>(
                handoffCodes->data,second.weight->data,handoffScales->data,second.scale->data,
                input,cosine,out,m,n,k,false);
        }
        finish(second);
        if(validateGuards){
            validate_one_int8_guard<<<1,32,0,stream>>>(handoffCodes->allocation,handoffCodes->payload_bytes(),status->data);
            validate_one_int8_guard<<<1,32,0,stream>>>(handoffScales->allocation,handoffScales->payload_bytes(),status->data);
        }
        sync_gpu();return true;
    }

    void launch_handoff_qkv(unsigned block,const __half*weight,const __half*scale,__half*out,unsigned m,unsigned k){
        auto&entry=prepare(block,Int8Role::Qkv,weight,k,3*k,0,false);
        launch_qkv_dot<64>(entry,scale,out,m,32,m);finish(entry);
    }
    void validate()const{unsigned flags=0;HIP_CHECK(hipMemcpy(&flags,status->data,sizeof(flags),hipMemcpyDeviceToHost));if(flags&1)throw std::runtime_error("grouped INT8 saw nonfinite input");if(flags&2)throw std::runtime_error("grouped INT8 buffer guard overwritten");}
    size_t reserved_bytes()const{size_t bytes=status->payload_bytes()+2*Int8GuardBytes;for(const auto&item:matrices)bytes+=item.second->reserved_bytes();for(const auto&item:branchProjects)bytes+=item.second->reserved_bytes();for(const auto&item:c32Attention)bytes+=item.second->reserved_bytes();if(activation)bytes+=activation->payload_bytes()+2*Int8GuardBytes;if(activationScale)bytes+=activationScale->payload_bytes()+2*Int8GuardBytes;if(handoffCodes)bytes+=handoffCodes->payload_bytes()+2*Int8GuardBytes;if(handoffScales)bytes+=handoffScales->payload_bytes()+2*Int8GuardBytes;return bytes;}
    size_t invocations()const{return launches;}
    bool release()noexcept{bool ok=true;for(auto&item:c32Attention)ok=item.second->release()&&ok;c32Attention.clear();for(auto&item:branchProjects)ok=item.second->release()&&ok;branchProjects.clear();for(auto&item:matrices)ok=item.second->release()&&ok;matrices.clear();if(handoffScales)ok=handoffScales->release()&&ok;if(handoffCodes)ok=handoffCodes->release()&&ok;if(activationScale)ok=activationScale->release()&&ok;if(activation)ok=activation->release()&&ok;ok=status->release()&&ok;return ok;}
};

static unsigned global_int8_group_size(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_GLOBAL_INT8"))throw std::runtime_error("Invalid global INT8 switch");if(!size||!value[0]||!std::strcmp(value,"0"))return 0;if(!std::strcmp(value,"1"))return 1;if(!std::strcmp(value,"32"))return 32;if(!std::strcmp(value,"64"))return 64;throw std::runtime_error("RDNA2_GLOBAL_INT8 must be 0, 1, 32, or 64");}
static bool parallel_window_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_WINDOW_PARALLEL"))throw std::runtime_error("Invalid parallel window switch");if(!size||!value[0]||!std::strcmp(value,"1"))return true;if(!std::strcmp(value,"0"))return false;throw std::runtime_error("RDNA2_WINDOW_PARALLEL must be 0 or 1");}
static bool fused_qkv_normalize_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_QKV_NORM_FUSED"))throw std::runtime_error("Invalid QKV normalization fusion switch");if(!size||!value[0]||!std::strcmp(value,"1"))return true;if(!std::strcmp(value,"0"))return false;throw std::runtime_error("RDNA2_QKV_NORM_FUSED must be 0 or 1");}
static bool fused_branch_project_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_BRANCH_PROJECT_FUSED"))throw std::runtime_error("Invalid branch fusion switch");if(!size||!value[0]||!std::strcmp(value,"1"))return true;if(!std::strcmp(value,"0"))return false;throw std::runtime_error("RDNA2_BRANCH_PROJECT_FUSED must be 0 or 1");}
static bool resident_branch_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_BRANCH_RESIDENT"))throw std::runtime_error("Invalid resident branch switch");if(!size||!value[0]||!std::strcmp(value,"0"))return false;if(!std::strcmp(value,"1"))return true;throw std::runtime_error("RDNA2_BRANCH_RESIDENT must be 0 or 1");}
static bool fused_c32_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_C32_FUSED"))throw std::runtime_error("Invalid fused C32 switch");if(!size||!value[0]||!std::strcmp(value,"1"))return true;if(!std::strcmp(value,"0"))return false;throw std::runtime_error("RDNA2_C32_FUSED must be 0 or 1");}
static bool packed_c32_fp16_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_C32_FP16_PACKED"))throw std::runtime_error("Invalid packed C32 FP16 switch");if(!size||!value[0]||!std::strcmp(value,"1"))return true;if(!std::strcmp(value,"0"))return false;throw std::runtime_error("RDNA2_C32_FP16_PACKED must be 0 or 1");}
static unsigned c32_int8_group_size(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_C32_INT8"))throw std::runtime_error("Invalid C32 INT8 switch");if(!size||!value[0]||!std::strcmp(value,"0"))return 0;if(!std::strcmp(value,"1")||!std::strcmp(value,"32"))return 32;if(!std::strcmp(value,"16"))return 16;throw std::runtime_error("RDNA2_C32_INT8 must be 0, 16, or 32");}
static bool fused_c32_attention_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_C32_ATTENTION_FUSED"))throw std::runtime_error("Invalid fused C32 attention switch");if(!size||!value[0]||!std::strcmp(value,"1"))return true;if(!std::strcmp(value,"0"))return false;throw std::runtime_error("RDNA2_C32_ATTENTION_FUSED must be 0 or 1");}
static bool attention_lds_padding_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_ATTENTION_LDS_PAD"))throw std::runtime_error("Invalid attention LDS switch");if(!size||!value[0]||!std::strcmp(value,"0"))return false;if(!std::strcmp(value,"1"))return true;throw std::runtime_error("RDNA2_ATTENTION_LDS_PAD must be 0 or 1");}
static bool graph_buffer_guards_enabled(){char value[16];size_t size=0;if(NrExecution::ReadSetting(&size,value,sizeof(value),"RDNA2_BUFFER_GUARDS"))throw std::runtime_error("Invalid buffer guard switch");if(!size||!value[0]||!std::strcmp(value,"0"))return false;if(!std::strcmp(value,"1"))return true;throw std::runtime_error("RDNA2_BUFFER_GUARDS must be 0 or 1");}
static Int8Scope grouped_int8_scope(){
    char scope[32];size_t size=0;if(NrExecution::ReadSetting(&size,scope,sizeof(scope),"RDNA2_GROUPED_INT8_SCOPE"))throw std::runtime_error("Invalid grouped INT8 scope");
    if(size&&scope[0]){if(!std::strcmp(scope,"block31"))return Int8Scope::Block31;if(!std::strcmp(scope,"global-expand"))return Int8Scope::GlobalExpand;if(!std::strcmp(scope,"global-dense"))return Int8Scope::GlobalDense;if(!std::strcmp(scope,"mixed-v1"))return Int8Scope::MixedV1;if(!std::strcmp(scope,"screened"))return Int8Scope::Screened;if(!std::strcmp(scope,"window512-attention"))return Int8Scope::Window512Attention;if(!std::strcmp(scope,"window512-dense"))return Int8Scope::Window512Dense;if(!std::strcmp(scope,"mixed-v2"))return Int8Scope::MixedV2;if(!std::strcmp(scope,"window512-split"))return Int8Scope::Window512Split;if(!std::strcmp(scope,"mixed-v3"))return Int8Scope::MixedV3;if(!std::strcmp(scope,"branched-expand"))return Int8Scope::BranchedExpand;if(!std::strcmp(scope,"mixed-v4"))return Int8Scope::MixedV4;if(!std::strcmp(scope,"junior-output"))return Int8Scope::JuniorOutput;if(!std::strcmp(scope,"junior-attention"))return Int8Scope::JuniorAttention;if(!std::strcmp(scope,"junior-dense"))return Int8Scope::JuniorDense;if(!std::strcmp(scope,"mixed-v5"))return Int8Scope::MixedV5;throw std::runtime_error("RDNA2_GROUPED_INT8_SCOPE invalid");}
    char legacy[16];size=0;if(NrExecution::ReadSetting(&size,legacy,sizeof(legacy),"RDNA2_GLOBAL_INT8_ALL"))throw std::runtime_error("Invalid global INT8 scope switch");if(!size||!legacy[0]||!std::strcmp(legacy,"0"))return Int8Scope::Block31;if(!std::strcmp(legacy,"1"))return Int8Scope::GlobalExpand;throw std::runtime_error("RDNA2_GLOBAL_INT8_ALL must be 0 or 1");
}
static const char* grouped_scope_name(Int8Scope scope){if(scope==Int8Scope::Block31)return"block31";if(scope==Int8Scope::GlobalExpand)return"global-expand";if(scope==Int8Scope::GlobalDense)return"global-dense";if(scope==Int8Scope::MixedV1)return"mixed-v1";if(scope==Int8Scope::Screened)return"screened";if(scope==Int8Scope::Window512Attention)return"window512-attention";if(scope==Int8Scope::Window512Dense)return"window512-dense";if(scope==Int8Scope::MixedV2)return"mixed-v2";if(scope==Int8Scope::Window512Split)return"window512-split";if(scope==Int8Scope::MixedV3)return"mixed-v3";if(scope==Int8Scope::BranchedExpand)return"branched-expand";if(scope==Int8Scope::MixedV4)return"mixed-v4";if(scope==Int8Scope::JuniorOutput)return"junior-output";if(scope==Int8Scope::JuniorAttention)return"junior-attention";if(scope==Int8Scope::JuniorDense)return"junior-dense";return"mixed-v5";}

struct WeightData{std::vector<__half>host;const __half*device=nullptr;size_t count=0;};
#ifndef NR_SPATIAL_GRAPH_LIBRARY
static const NrV2::WeightPackage* runtime_package=nullptr;
#endif
static thread_local const NrV2::GpuModel* runtime_model=nullptr;
static thread_local GroupedInt8Plan* active_grouped_int8=nullptr;
struct LinkedW128Plan{
    struct Entry{size_t offset=0,count=0;const __half*device=nullptr;};
    std::map<std::pair<unsigned,std::string>,Entry> entries;
    __half*allocation=nullptr;
    unsigned residualGroups[4]{},attentionHeads[4]{};
    static bool covered(unsigned block){return (block>=15&&block<=22)||(block>=48&&block<=55);}
    static std::vector<unsigned> expected(unsigned block,const std::string&role){
        if(covered(block)){
            if(role=="ffn_expand")return{4,4,4,32,32};
            if(role=="ffn_branch")return{4,4,32,32};
            if(role=="ffn_output"||role=="attention_projection")return{128,128};
            if(role=="ffn_cos"||role=="attention_cos")return{128};
            if(role=="qkv")return{128,384};
            if(role=="bias")return{4,64,64};
            if(role=="scale")return{4};
        }
        if(block==14&&role=="downsample")return{128,128};
        if(block==22&&role=="downsample")return{128,512};
        if(block==48&&role=="upsample")return{512,128};
        if(block==48&&role=="upsample_sin")return{128};
        if(block==56&&role=="upsample")return{128,128};
        if(block==56&&role=="upsample_sin")return{128};
        return{};
    }
    LinkedW128Plan(const NrV2::GpuModel&model,hipStream_t stream){
        char path[4096],hash[128];size_t ps=0,hs=0;
        if(NrExecution::ReadSetting(&ps,path,sizeof(path),"RDNA2_LINKED_W128_BUNDLE")||
           NrExecution::ReadSetting(&hs,hash,sizeof(hash),"RDNA2_LINKED_W128_SHA256"))throw std::runtime_error("Invalid linked W128 environment");
        if(!ps||!path[0])return;
        if(!hs||!hash[0])throw std::runtime_error("RDNA2_LINKED_W128_SHA256 is required");
        std::ifstream input(path,std::ios::binary|std::ios::ate);
        if(!input)throw std::runtime_error("Unable to open linked W128 artifact");
        const auto end=input.tellg();if(end<112||end>32*1024*1024)throw std::runtime_error("Linked W128 size mismatch");
        std::vector<unsigned char>bytes(static_cast<size_t>(end));input.seekg(0);
        if(!input.read(reinterpret_cast<char*>(bytes.data()),end))throw std::runtime_error("Unable to read linked W128 artifact");
        if(NrV2::Sha256(bytes.data(),bytes.size())!=NrV2::ParseSha256(hash))throw std::runtime_error("Linked W128 SHA-256 mismatch");
        if(std::memcmp(bytes.data(),"NRW12801",8))throw std::runtime_error("Linked W128 magic");
        auto u32=[&](size_t at){unsigned v=0;if(at+4>bytes.size())throw std::runtime_error("Linked W128 truncated u32");std::memcpy(&v,bytes.data()+at,4);return v;};
        auto u64=[&](size_t at){std::uint64_t v=0;if(at+8>bytes.size())throw std::runtime_error("Linked W128 truncated u64");std::memcpy(&v,bytes.data()+at,8);return v;};
        if(u32(8)!=1||u32(12)!=150||
           std::memcmp(bytes.data()+16,model.PackageHash().data(),32)||
           std::memcmp(bytes.data()+48,model.SourceHash().data(),32))throw std::runtime_error("Linked W128 parent contract");
        for(unsigned i=0;i<4;++i){residualGroups[i]=u32(80+4*i);attentionHeads[i]=u32(96+4*i);
            if(residualGroups[i]>=8||attentionHeads[i]>=8||
               (i&&(residualGroups[i]<=residualGroups[i-1]||attentionHeads[i]<=attentionHeads[i-1])))
                throw std::runtime_error("Linked W128 channel map");}
        std::vector<__half>host;size_t cursor=112;
        for(unsigned record=0;record<150;++record){
            if(cursor+92>bytes.size())throw std::runtime_error("Linked W128 descriptor truncated");
            const unsigned block=u32(cursor),rank=u32(cursor+28);
            const char*roleStart=reinterpret_cast<const char*>(bytes.data()+cursor+4);
            const void*zero=std::memchr(roleStart,0,24);
            if(!zero||rank==0||rank>5)throw std::runtime_error("Linked W128 descriptor name/rank");
            const std::string role(roleStart,static_cast<const char*>(zero));
            for(const char*p=static_cast<const char*>(zero);p<roleStart+24;++p)if(*p)throw std::runtime_error("Linked W128 role padding");
            const auto shape=expected(block,role);
            if(shape.size()!=rank)throw std::runtime_error("Linked W128 unexpected tensor");
            size_t count=1;for(unsigned i=0;i<5;++i){const unsigned dim=u32(cursor+32+4*i);
                if(dim!=(i<rank?shape[i]:0))throw std::runtime_error("Linked W128 tensor shape");
                if(i<rank)count*=dim;}
            const auto length=u64(cursor+52);if(length!=count*2||cursor+92+length>bytes.size())throw std::runtime_error("Linked W128 tensor length");
            const auto checksum=NrV2::Sha256(bytes.data()+cursor+92,size_t(length));
            if(std::memcmp(bytes.data()+cursor+60,checksum.data(),32))throw std::runtime_error("Linked W128 tensor SHA-256");
            const auto key=std::make_pair(block,role);
            if(entries.count(key))throw std::runtime_error("Linked W128 duplicate tensor");
            Entry entry;entry.offset=host.size();entry.count=count;entries.emplace(key,entry);
            const size_t old=host.size();host.resize(old+count);
            std::memcpy(host.data()+old,bytes.data()+cursor+92,size_t(length));
            for(size_t i=old;i<host.size();++i){unsigned bits=0;std::memcpy(&bits,&host[i],2);if((bits&0x7c00)==0x7c00)throw std::runtime_error("Linked W128 nonfinite weight");}
            cursor+=92+size_t(length);
        }
        if(cursor!=bytes.size())throw std::runtime_error("Linked W128 trailing bytes");
        HIP_CHECK(hipMalloc(&allocation,host.size()*sizeof(__half)));
        try{HIP_CHECK(hipMemcpyAsync(allocation,host.data(),host.size()*sizeof(__half),hipMemcpyHostToDevice,stream));}
        catch(...){static_cast<void>(hipFree(allocation));allocation=nullptr;throw;}
        for(auto&item:entries)item.second.device=allocation+item.second.offset;
    }
    ~LinkedW128Plan(){if(allocation&&hipFree(allocation)!=hipSuccess)cleanup_failed=true;}
    const Entry*find(unsigned block,const char*role)const{auto it=entries.find({block,role});return it==entries.end()?nullptr:&it->second;}
    bool enabled()const noexcept{return allocation!=nullptr;}
    bool release()noexcept{if(!allocation)return true;if(hipFree(allocation)!=hipSuccess)return false;allocation=nullptr;return true;}
};
static thread_local LinkedW128Plan*active_linked_w128=nullptr;
struct CompactC32Plan{
    struct Entry{unsigned hidden=0;size_t offset=0;const __half*expand=nullptr;const __half*contract=nullptr;const __half*cosine=nullptr;};
    __half*allocation=nullptr;Entry entries[71]{};unsigned blocks=0;
    CompactC32Plan(const NrV2::GpuModel&model,hipStream_t stream){
        char path[4096],legacy[4096],hash[128];size_t pathSize=0,legacySize=0,hashSize=0;
        if(NrExecution::ReadSetting(&pathSize,path,sizeof(path),"RDNA2_COMPACT_C32_BUNDLE")||NrExecution::ReadSetting(&legacySize,legacy,sizeof(legacy),"RDNA2_COMPACT_C32_BLOCK2")||NrExecution::ReadSetting(&hashSize,hash,sizeof(hash),"RDNA2_COMPACT_C32_SHA256"))
            throw std::runtime_error("Invalid compact C32 environment");
        if((!pathSize||!path[0])&&legacySize&&legacy[0]){if(strcpy_s(path,sizeof(path),legacy))throw std::runtime_error("Compact C32 legacy path");pathSize=legacySize;}
        if(!pathSize||!path[0])return;
        if(!hashSize||!hash[0])throw std::runtime_error("RDNA2_COMPACT_C32_SHA256 is required");
        std::ifstream input(path,std::ios::binary|std::ios::ate);if(!input)throw std::runtime_error("Unable to open compact C32 artifact");
        const auto end=input.tellg();if(end<80||end>1024*1024)throw std::runtime_error("Compact C32 artifact size mismatch");
        std::vector<unsigned char> bytes(static_cast<size_t>(end));input.seekg(0);if(!input.read(reinterpret_cast<char*>(bytes.data()),end))throw std::runtime_error("Unable to read compact C32 artifact");
        if(NrV2::Sha256(bytes.data(),bytes.size())!=NrV2::ParseSha256(hash))throw std::runtime_error("Compact C32 artifact SHA-256 mismatch");
        auto u32=[&](size_t offset){unsigned value=0;std::memcpy(&value,bytes.data()+offset,4);return value;};
        std::vector<__half> host;
        auto append=[&](unsigned block,unsigned hidden,size_t indicesOffset,size_t payloadOffset){
            if(block>70||entries[block].hidden||(hidden!=32&&hidden!=64&&hidden!=96))throw std::runtime_error("Compact C32 record identity");
            if(indicesOffset+size_t(hidden)*4>bytes.size())throw std::runtime_error("Compact C32 indices truncated");
            unsigned previous=0;for(unsigned i=0;i<hidden;++i){const unsigned current=u32(indicesOffset+i*4);if(current>=128||(i&&current<=previous))throw std::runtime_error("Compact C32 selection invalid");previous=current;}
            const size_t halfCount=size_t(64)*hidden+32,byteCount=halfCount*2;if(payloadOffset+byteCount>bytes.size())throw std::runtime_error("Compact C32 payload truncated");
            Entry&entry=entries[block];entry.hidden=hidden;entry.offset=host.size();const size_t old=host.size();host.resize(old+halfCount);std::memcpy(host.data()+old,bytes.data()+payloadOffset,byteCount);++blocks;
        };
        if(!std::memcmp(bytes.data(),"NRCFFN01",8)){
            if(bytes.size()!=8608||u32(8)!=1||u32(12)!=2||u32(16)!=32||u32(20)!=64||u32(24)!=32||u32(28)!=64)throw std::runtime_error("Compact C32 legacy contract mismatch");
            if(std::memcmp(bytes.data()+32,model.PackageHash().data(),32)||std::memcmp(bytes.data()+64,model.SourceHash().data(),32))throw std::runtime_error("Compact C32 parent identity mismatch");
            append(2,64,96,352);
        }else if(!std::memcmp(bytes.data(),"NRCFNB01",8)){
            const unsigned count=u32(12);if(u32(8)!=1||!count||count>10)throw std::runtime_error("Compact C32 bundle contract mismatch");
            if(std::memcmp(bytes.data()+16,model.PackageHash().data(),32)||std::memcmp(bytes.data()+48,model.SourceHash().data(),32))throw std::runtime_error("Compact C32 parent identity mismatch");
            size_t cursor=80;for(unsigned record=0;record<count;++record){if(cursor+20>bytes.size())throw std::runtime_error("Compact C32 record truncated");const unsigned block=u32(cursor),inputColumns=u32(cursor+4),hidden=u32(cursor+8),outputColumns=u32(cursor+12),selected=u32(cursor+16);cursor+=20;if(inputColumns!=32||outputColumns!=32||selected!=hidden)throw std::runtime_error("Compact C32 record shape");const size_t indices=cursor;cursor+=size_t(hidden)*4;const size_t payload=cursor;append(block,hidden,indices,payload);cursor+=size_t(64)*hidden*2+64;}if(cursor!=bytes.size())throw std::runtime_error("Compact C32 trailing bytes");
        }else throw std::runtime_error("Compact C32 artifact magic");
        HIP_CHECK(hipMalloc(&allocation,host.size()*sizeof(__half)));try{HIP_CHECK(hipMemcpyAsync(allocation,host.data(),host.size()*sizeof(__half),hipMemcpyHostToDevice,stream));}catch(...){static_cast<void>(hipFree(allocation));allocation=nullptr;throw;}
        for(auto&entry:entries)if(entry.hidden){entry.expand=allocation+entry.offset;entry.contract=entry.expand+32*entry.hidden;entry.cosine=entry.contract+entry.hidden*32;}
    }
    ~CompactC32Plan(){if(allocation&&hipFree(allocation)!=hipSuccess)cleanup_failed=true;}
    const Entry*find(unsigned block)const noexcept{return block<71&&entries[block].hidden?&entries[block]:nullptr;}
    bool enabled()const noexcept{return allocation!=nullptr;}
    bool release()noexcept{if(!allocation)return true;if(hipFree(allocation)!=hipSuccess)return false;allocation=nullptr;for(auto&entry:entries)entry=Entry{};blocks=0;return true;}
};
static thread_local CompactC32Plan*active_compact_c32=nullptr;
struct CompactBranchedPlan{
    struct Entry{unsigned channels=0,branches=0;size_t offset=0;const __half*expand=nullptr;const __half*branch=nullptr;const __half*output=nullptr;const __half*cosine=nullptr;};
    __half*allocation=nullptr;Entry entries[71]{};unsigned blocks=0;
    CompactBranchedPlan(const NrV2::GpuModel&model,hipStream_t stream){
        char path[4096],hash[128];size_t pathSize=0,hashSize=0;
        if(NrExecution::ReadSetting(&pathSize,path,sizeof(path),"RDNA2_COMPACT_BRANCHED_BUNDLE")||NrExecution::ReadSetting(&hashSize,hash,sizeof(hash),"RDNA2_COMPACT_BRANCHED_SHA256"))
            throw std::runtime_error("Invalid compact branched environment");
        if(!pathSize||!path[0])return;if(!hashSize||!hash[0])throw std::runtime_error("RDNA2_COMPACT_BRANCHED_SHA256 is required");
        std::ifstream input(path,std::ios::binary|std::ios::ate);if(!input)throw std::runtime_error("Unable to open compact branched artifact");
        const auto end=input.tellg();if(end<80||end>32*1024*1024)throw std::runtime_error("Compact branched artifact size mismatch");
        std::vector<unsigned char>bytes(static_cast<size_t>(end));input.seekg(0);if(!input.read(reinterpret_cast<char*>(bytes.data()),end))throw std::runtime_error("Unable to read compact branched artifact");
        if(NrV2::Sha256(bytes.data(),bytes.size())!=NrV2::ParseSha256(hash))throw std::runtime_error("Compact branched artifact SHA-256 mismatch");
        if(std::memcmp(bytes.data(),"NRCBRN01",8))throw std::runtime_error("Compact branched artifact magic");
        auto u32=[&](size_t offset){unsigned value=0;if(offset+4>bytes.size())throw std::runtime_error("Compact branched artifact truncated");std::memcpy(&value,bytes.data()+offset,4);return value;};
        const unsigned count=u32(12);if(u32(8)!=1||!count||count>36)throw std::runtime_error("Compact branched bundle contract mismatch");
        if(std::memcmp(bytes.data()+16,model.PackageHash().data(),32)||std::memcmp(bytes.data()+48,model.SourceHash().data(),32))throw std::runtime_error("Compact branched parent identity mismatch");
        std::vector<__half>host;size_t cursor=80;
        for(unsigned record=0;record<count;++record){
            if(cursor+20>bytes.size())throw std::runtime_error("Compact branched record truncated");
            const unsigned block=u32(cursor),channels=u32(cursor+4),branches=u32(cursor+8),groups=u32(cursor+12),selected=u32(cursor+16);cursor+=20;
            const bool c64=(block>=5&&block<=8)||(block>=62&&block<=65),c128=(block>=9&&block<=14)||(block>=56&&block<=61),c256=(block>=15&&block<=22)||(block>=48&&block<=55);
            if(block>70||entries[block].channels||(!c64&&!c128&&!c256)||channels!=(c64?64u:c128?128u:256u)||groups!=channels/32||branches<1||branches>3||selected!=groups*branches)
                throw std::runtime_error("Compact branched record identity");
            if(cursor+size_t(selected)*4>bytes.size())throw std::runtime_error("Compact branched selection truncated");
            for(unsigned head=0;head<groups;++head){unsigned previous=0;for(unsigned branch=0;branch<branches;++branch){const unsigned current=u32(cursor+size_t(head*branches+branch)*4);if(current>=4||(branch&&current<=previous))throw std::runtime_error("Compact branched selection invalid");previous=current;}}
            cursor+=size_t(selected)*4;
            const size_t expandCount=size_t(groups)*branches*groups*32*32,branchCount=size_t(groups)*branches*32*32,outputCount=size_t(channels)*channels;
            const size_t halfCount=expandCount+branchCount+outputCount+channels,byteCount=halfCount*2;if(cursor+byteCount>bytes.size())throw std::runtime_error("Compact branched payload truncated");
            Entry&entry=entries[block];entry.channels=channels;entry.branches=branches;entry.offset=host.size();const size_t old=host.size();host.resize(old+halfCount);std::memcpy(host.data()+old,bytes.data()+cursor,byteCount);cursor+=byteCount;++blocks;
        }
        if(cursor!=bytes.size())throw std::runtime_error("Compact branched trailing bytes");
        HIP_CHECK(hipMalloc(&allocation,host.size()*sizeof(__half)));try{HIP_CHECK(hipMemcpyAsync(allocation,host.data(),host.size()*sizeof(__half),hipMemcpyHostToDevice,stream));}catch(...){static_cast<void>(hipFree(allocation));allocation=nullptr;throw;}
        for(auto&entry:entries)if(entry.channels){const size_t groups=entry.channels/32,expandCount=groups*entry.branches*groups*32*32,branchCount=groups*entry.branches*32*32,outputCount=size_t(entry.channels)*entry.channels;entry.expand=allocation+entry.offset;entry.branch=entry.expand+expandCount;entry.output=entry.branch+branchCount;entry.cosine=entry.output+outputCount;}
    }
    ~CompactBranchedPlan(){if(allocation&&hipFree(allocation)!=hipSuccess)cleanup_failed=true;}
    const Entry*find(unsigned block)const noexcept{return block<71&&entries[block].channels?&entries[block]:nullptr;}
    bool enabled()const noexcept{return allocation!=nullptr;}
    bool release()noexcept{if(!allocation)return true;if(hipFree(allocation)!=hipSuccess)return false;allocation=nullptr;for(auto&entry:entries)entry=Entry{};blocks=0;return true;}
};
static thread_local CompactBranchedPlan*active_compact_branched=nullptr;
static void observe_activation(int block,const char*role,const __half*values,size_t rows,size_t columns){
    if(active_graph_profile&&active_graph_profile->activationObserver)
        active_graph_profile->activationObserver(active_graph_profile->activationObserverContext,
            unsigned(block),role,values,rows,columns,current_stream());
}
static WeightData weight(Reader*r,int block,const char*role,size_t count){if(r){auto host=r->read<__half>(count);return{std::move(host),nullptr,count};}if(active_linked_w128){if(const auto*entry=active_linked_w128->find(unsigned(block),role)){if(entry->count!=count)throw std::runtime_error("Linked W128 runtime tensor count");return{{},entry->device,count};}if(LinkedW128Plan::covered(unsigned(block)))throw std::runtime_error("Linked W128 missing covered tensor");}if(!runtime_model)throw std::runtime_error("missing GPU model");return{{},static_cast<const __half*>(runtime_model->TensorData(block,role,count)),count};}
static HalfPtr upload_half(const std::vector<__half>&v){auto p=std::make_unique<HalfBuffer>(v.size());p->upload(v);return p;}
static const __half* weight_ptr(const WeightData&w,HalfPtr&owner){if(w.device)return w.device;if(w.host.size()!=w.count)throw std::runtime_error("host weight size mismatch");owner=upload_half(w.host);return owner->ptr();}
static HalfPtr clone_half(const HalfBuffer&v){auto p=std::make_unique<HalfBuffer>(v.count);if(current_stream())HIP_CHECK(hipMemcpyAsync(p->ptr(),const_cast<HalfBuffer&>(v).ptr(),v.count*sizeof(__half),hipMemcpyDeviceToDevice,current_stream()));else HIP_CHECK(hipMemcpy(p->ptr(),const_cast<HalfBuffer&>(v).ptr(),v.count*sizeof(__half),hipMemcpyDeviceToDevice));sync_gpu();return p;}

static HalfPtr publish_device(HalfBuffer&v){auto p=std::make_unique<HalfBuffer>(v.count);publish_half_kernel<<<(v.count+255)/256,256,0,current_stream()>>>(v.ptr(),p->ptr(),v.count);sync_gpu();return p;}
static HalfPtr selective_linear_device(HalfBuffer&input,int m,int k,int n,const WeightData&weight,int block,Int8Role role,unsigned epilogue,bool publish=false){HalfPtr owner;auto w=weight_ptr(weight,owner);auto out=std::make_unique<HalfBuffer>(size_t(m)*n);if(active_grouped_int8&&active_grouped_int8->enabled(block,role))active_grouped_int8->launch(block,role,input.ptr(),w,nullptr,nullptr,out->ptr(),m,k,n,epilogue,publish);else launch_gemm_half(input.ptr(),w,out->ptr(),m,k,n,publish);return out;}
static HalfPtr pool_device(HalfBuffer&input,int h,int w,int c){auto p=std::make_unique<HalfBuffer>(size_t(h/2)*(w/2)*c);pool2_kernel<<<(p->count+255)/256,256,0,current_stream()>>>(input.ptr(),p->ptr(),h,w,c);sync_gpu();return p;}
static HalfPtr pad_device(HalfBuffer&input,int h,int w,int ph,int pw,int c){auto p=std::make_unique<HalfBuffer>(size_t(ph)*pw*c);pad_end_kernel<<<(p->count+255)/256,256,0,current_stream()>>>(input.ptr(),p->ptr(),h,w,ph,pw,c);sync_gpu();return p;}
static unsigned long long checkpoint_differences=0,checkpoint_values=0;static unsigned checkpoint_worst_step=0;static bool rebase_checkpoints=false;
static unsigned e4_order(unsigned char code){return(code&0x80)?0x80-(code&0x7f):0x80+(code&0x7f);}
static unsigned half_order(unsigned short bits){return(bits&0x8000)?0x8000-(bits&0x7fff):0x8000+(bits&0x7fff);}
static void check_codes(Reader*r,HalfBuffer&actual,const char*label){if(!r)return;auto expected=r->read<unsigned char>(actual.count);Buffer<unsigned char>codes(actual.count);final_codes_kernel<<<(actual.count+255)/256,256,0,current_stream()>>>(actual.ptr(),codes.ptr(),actual.count);sync_gpu();auto got=codes.read();auto halves=actual.read();unsigned errors=0,noncanonical=0,max_step=0;for(size_t i=0;i<got.size();++i){unsigned short bits;std::memcpy(&bits,&halves[i],2);if((bits&0x7c00)==0x7c00||(expected[i]&0x7f)==0x7f)throw std::runtime_error("Nonfinite checkpoint");if(bits!=rdna2_nr::e4m3fn_to_half_bits(got[i]))++noncanonical;if(got[i]!=expected[i]){unsigned a=e4_order(got[i]),b=e4_order(expected[i]);max_step=std::max(max_step,a>b?a-b:b-a);if(errors++<3)std::printf("%s mismatch i=%zu code=%02x/%02x half=%04x\n",label,i,got[i],expected[i],bits);}}if(noncanonical)throw std::runtime_error(std::string(label)+" noncanonical E4M3 state");checkpoint_differences+=errors;checkpoint_values+=got.size();checkpoint_worst_step=std::max(checkpoint_worst_step,max_step);if(errors)std::printf("CHECKPOINT %s %s differences=%u/%zu max_e4m3_step=%u canonical=PASS\n",rebase_checkpoints?"BOUNDED":"DIVERGED",label,errors,got.size(),max_step);else std::printf("CHECKPOINT PASS %s e4m3=%zu/%zu\n",label,got.size(),got.size());if(rebase_checkpoints){std::vector<__half>decoded(expected.size());for(size_t i=0;i<expected.size();++i){unsigned short bits=rdna2_nr::e4m3fn_to_half_bits(expected[i]);std::memcpy(&decoded[i],&bits,2);}actual.upload(decoded);}}
static void check_half_output(Reader*r,HalfBuffer&actual,const char*label){if(!r)return;auto expected=r->read<unsigned short>(actual.count);auto got_half=actual.read();unsigned errors=0,max_ulp=0;double max_abs=0;for(size_t i=0;i<got_half.size();++i){unsigned short got;std::memcpy(&got,&got_half[i],2);if((got&0x7c00)==0x7c00||(expected[i]&0x7c00)==0x7c00)throw std::runtime_error("Nonfinite output");if(got!=expected[i]){unsigned a=half_order(got),b=half_order(expected[i]);max_ulp=std::max(max_ulp,a>b?a-b:b-a);__half expected_half;std::memcpy(&expected_half,&expected[i],2);max_abs=std::max(max_abs,std::abs(double(__half2float(got_half[i]))-double(__half2float(expected_half))));++errors;}}std::printf("CHECKPOINT %s %s differences=%u/%zu max_half_ulp=%u max_abs=%.9g\n",errors?(rebase_checkpoints?"BOUNDED":"DIVERGED"):"PASS",label,errors,got_half.size(),max_ulp,max_abs);}

struct WindowWeights{int attentionChannels=0;WeightData adapter,up,upsin,a,b,d,outw,fcos,qkv,bias,scale,proj,acos,down,merge_sin,merge_cos,gain,conv;};
static WindowWeights read_window_weights(Reader*r,int c,int family,int kind,int block){WindowWeights w;w.attentionChannels=active_linked_w128&&LinkedW128Plan::covered(unsigned(block))?128:c;if(kind==3)w.adapter=weight(r,block,"adapter",16*c);if(kind==2){const int inputChannels=active_linked_w128&&block==48?512:active_linked_w128&&block==56?128:2*c;w.up=weight(r,block,"upsample",size_t(inputChannels)*c);w.upsin=weight(r,block,"upsample_sin",c);}if(family==0){w.a=weight(r,block,"ffn_expand",c*128);w.b=weight(r,block,"ffn_contract",128*c);}else if(family==1){int g=c/32;w.a=weight(r,block,"ffn_expand",g*4*g*32*32);w.b=weight(r,block,"ffn_branch",g*4*32*32);w.d=weight(r,block,"ffn_output",c*c);}else{w.a=weight(r,block,"ffn_first",c*c);w.b=weight(r,block,"ffn_expand",8*64*256);w.d=weight(r,block,"ffn_group_project",8*256*64);w.outw=weight(r,block,"ffn_output",c*c);}w.fcos=weight(r,block,"ffn_cos",c);w.qkv=weight(r,block,"qkv",size_t(c)*3*w.attentionChannels);w.bias=weight(r,block,"bias",(w.attentionChannels/32)*4096);w.scale=weight(r,block,"scale",w.attentionChannels/32);w.proj=weight(r,block,"attention_projection",size_t(w.attentionChannels)*c);w.acos=weight(r,block,"attention_cos",c);if(kind==1){const int outputChannels=active_linked_w128&&block==14?128:active_linked_w128&&block==22?512:2*c;w.down=weight(r,block,"downsample",size_t(c)*outputChannels);}if(kind==4){w.merge_sin=weight(r,block,"merge_sin",c);w.merge_cos=weight(r,block,"merge_cos",c);w.gain=weight(r,block,"out_gain",16*4);w.conv=weight(r,block,"out_conv",16*4);}return w;}

static HalfPtr ffn_device(HalfBuffer&input,int tokens,int c,int family,const WindowWeights&w,int trace_block){int padded=(tokens+63)/64*64;HalfPtr pin;if(padded!=tokens){pin=std::make_unique<HalfBuffer>(size_t(padded)*c);if(current_stream()){HIP_CHECK(hipMemsetAsync(pin->ptr(),0,pin->count*sizeof(__half),current_stream()));HIP_CHECK(hipMemcpyAsync(pin->ptr(),input.ptr(),size_t(tokens)*c*sizeof(__half),hipMemcpyDeviceToDevice,current_stream()));}else{HIP_CHECK(hipMemset(pin->ptr(),0,pin->count*sizeof(__half)));HIP_CHECK(hipMemcpy(pin->ptr(),input.ptr(),size_t(tokens)*c*sizeof(__half),hipMemcpyDeviceToDevice));}sync_gpu();}const __half*x=padded==tokens?input.ptr():pin->ptr();HalfPtr cosine_owner;auto cosine=weight_ptr(w.fcos,cosine_owner);auto padded_out=std::make_unique<HalfBuffer>(size_t(padded)*c);
    if(family==0){HalfPtr wa_owner,wb_owner;auto wa=weight_ptr(w.a,wa_owner),wb=weight_ptr(w.b,wb_owner);const bool inspectHidden=trace_requested(trace_block)||(active_graph_profile&&active_graph_profile->activationObserver)||int8_capture_requested(trace_block);const bool compactDisabled=active_graph_profile&&active_graph_profile->disableCompactC32;const auto*compact=active_compact_c32&&!compactDisabled?active_compact_c32->find(unsigned(trace_block)):nullptr;const unsigned c32Group=c32_int8_group_size();if(compact&&!inspectHidden){launch_fused_c32_ffn_compact(x,compact->expand,compact->contract,compact->cosine,padded_out->ptr(),padded,compact->hidden);}else if(c32Group&&active_grouped_int8&&!inspectHidden){active_grouped_int8->launch_c32_ffn(unsigned(trace_block),x,wa,wb,cosine,padded_out->ptr(),unsigned(padded),c32Group);}else if(fused_c32_enabled()&&!inspectHidden){launch_fused_c32_ffn(x,wa,wb,cosine,padded_out->ptr(),padded,packed_c32_fp16_enabled());}else{Buffer<__half>hidden(size_t(padded)*128);if(trace_requested(trace_block)){Buffer<float>expand(size_t(padded)*128),contract(size_t(padded)*c);launch_gemm(x,wa,expand.ptr(),padded,c,128);gate_publish<<<(hidden.count+255)/256,256,0,current_stream()>>>(expand.ptr(),hidden.ptr(),hidden.count);sync_gpu();observe_activation(trace_block,"ffn_contract_input",hidden.ptr(),padded,128);launch_gemm(hidden.ptr(),wb,contract.ptr(),padded,128,c);residual_h<<<(padded_out->count+255)/256,256,0,current_stream()>>>(contract.ptr(),x,cosine,padded_out->ptr(),padded_out->count,c,false);sync_gpu();trace_stage(trace_block,"ffn_expand",expand);trace_stage(trace_block,"ffn_hidden",hidden);trace_stage(trace_block,"ffn_contract",contract);}else{launch_gemm_gate_publish(x,wa,hidden.ptr(),padded,c,128);observe_activation(trace_block,"ffn_contract_input",hidden.ptr(),padded,128);launch_gemm_residual(hidden.ptr(),wb,x,cosine,padded_out->ptr(),padded,128,c,false);}capture_int8_activation(trace_block,"ffn_hidden",hidden,padded,128);}}
    else if(family==1){
        HalfPtr we_owner,wb_owner,wo_owner;auto we=weight_ptr(w.a,we_owner),wb=weight_ptr(w.b,wb_owner),wo=weight_ptr(w.d,wo_owner);
        const bool inspect=trace_requested(trace_block)||(active_graph_profile&&active_graph_profile->activationObserver)||int8_capture_requested(trace_block);
        const bool compactDisabled=active_graph_profile&&active_graph_profile->disableCompactBranched;
        const auto*compact=active_compact_branched&&!compactDisabled?active_compact_branched->find(unsigned(trace_block)):nullptr;
        if(compact&&!inspect){
            const size_t hiddenCount=size_t(padded)*(c/32)*compact->branches*32;
            Buffer<__half>hidden(hiddenCount),combined(size_t(padded)*c);
            launch_compact_branched_expand(x,compact->expand,hidden.ptr(),c,padded,compact->branches);
            launch_compact_branched_project_merge(hidden.ptr(),compact->branch,combined.ptr(),c,padded,compact->branches);
            launch_gemm_residual(combined.ptr(),compact->output,x,compact->cosine,padded_out->ptr(),padded,c,c,true);
        }else{
            Buffer<__half>combined(size_t(padded)*c);
            const bool resident=!inspect&&resident_branch_enabled()&&active_grouped_int8&&
                active_grouped_int8->enabled(trace_block,Int8Role::BranchedExpand)&&active_grouped_int8->groupSize==64;
            if(resident){
                stage_event("branched_resident",true);
                active_grouped_int8->launch_branched_resident(
                    unsigned(trace_block),x,we,wb,combined.ptr(),unsigned(padded),unsigned(c));
                stage_event("branched_resident",false);
            }
            else{
                Buffer<__half>hidden(size_t(padded)*4*c);
                if(active_grouped_int8&&active_grouped_int8->enabled(trace_block,Int8Role::BranchedExpand))
                    active_grouped_int8->launch_broadcast_batched(trace_block,x,we,hidden.ptr(),padded,4*c/32,c,32);
                else branched_expand_rows<<<(hidden.count+255)/256,256,0,current_stream()>>>(x,we,hidden.ptr(),c,padded);
                observe_activation(trace_block,"branch_project_input",hidden.ptr(),tokens,4*c);
                if(!inspect&&padded>=64&&fused_branch_project_enabled()){
                    branched_project_merge_tiled<<<dim3(c/32,(padded+15)/16),dim3(32,4),0,current_stream()>>>(
                        hidden.ptr(),wb,combined.ptr(),c,padded);sync_gpu();
                }else{
                    Buffer<__half>branches(size_t(padded)*4*c);
                    branched_project_rows<<<(branches.count+255)/256,256,0,current_stream()>>>(hidden.ptr(),wb,branches.ptr(),c,padded);
                    observe_activation(trace_block,"branch_merge_input",branches.ptr(),tokens,4*c);
                    branched_merge_rows<<<(combined.count+255)/256,256,0,current_stream()>>>(branches.ptr(),combined.ptr(),c,padded);sync_gpu();
                }
            }
            observe_activation(trace_block,"ffn_output_input",combined.ptr(),tokens,c);
            stage_event("branched_output",true);
            if(active_grouped_int8&&active_grouped_int8->enabled(trace_block,Int8Role::BranchedOutput))
                active_grouped_int8->launch(trace_block,Int8Role::BranchedOutput,combined.ptr(),wo,x,cosine,padded_out->ptr(),padded,c,c,3,true);
            else if(c==256&&active_graph_profile&&active_graph_profile->activationObserver){
                Buffer<__half>raw(size_t(padded)*c);launch_gemm_residual(combined.ptr(),wo,x,cosine,padded_out->ptr(),padded,c,c,true,raw.ptr());
                observe_activation(trace_block,"ffn_projection_raw",raw.ptr(),tokens,c);
            }else launch_gemm_residual(combined.ptr(),wo,x,cosine,padded_out->ptr(),padded,c,c,true);
            stage_event("branched_output",false);
        }
    }
    else{
        HalfPtr wf_owner,we_owner,wg_owner,wo_owner;
        auto wf=weight_ptr(w.a,wf_owner),we=weight_ptr(w.b,we_owner),
             wg=weight_ptr(w.d,wg_owner),wo=weight_ptr(w.outw,wo_owner);
        Buffer<__half>first(size_t(padded)*c),grouped(size_t(padded)*c);
        if(active_grouped_int8&&active_grouped_int8->enabled(trace_block,Int8Role::SplitFirst))
            active_grouped_int8->launch(trace_block,Int8Role::SplitFirst,x,wf,nullptr,nullptr,first.ptr(),padded,c,c,2,true);
        else launch_gemm_publish(x,wf,first.ptr(),padded,c,c);
        observe_activation(trace_block,"split_expand_input",first.ptr(),padded,c);
        const bool inspectSplit=trace_requested(trace_block)||int8_capture_requested(trace_block)||
            (active_graph_profile&&active_graph_profile->activationObserver);
        const bool resident=!inspectSplit&&active_grouped_int8&&
            active_grouped_int8->launch_split_resident(unsigned(trace_block),first.ptr(),we,wg,grouped.ptr(),unsigned(padded));
        if(!resident){
            Buffer<__half>hidden(size_t(padded)*4*c);
            if(active_grouped_int8&&active_grouped_int8->enabled(trace_block,Int8Role::SplitExpand))
                active_grouped_int8->launch_batched(trace_block,Int8Role::SplitExpand,first.ptr(),we,hidden.ptr(),padded,8,64,256,4);
            else split_expand_rows<<<(hidden.count+255)/256,256,0,current_stream()>>>(first.ptr(),we,hidden.ptr(),padded);
            observe_activation(trace_block,"group_project_input",hidden.ptr(),padded,4*c);
            if(active_grouped_int8&&active_grouped_int8->enabled(trace_block,Int8Role::SplitProject))
                active_grouped_int8->launch_batched(trace_block,Int8Role::SplitProject,hidden.ptr(),wg,grouped.ptr(),padded,8,256,64,2);
            else split_project_rows<<<(grouped.count+255)/256,256,0,current_stream()>>>(hidden.ptr(),wg,grouped.ptr(),padded);
        }
        sync_gpu();observe_activation(trace_block,"ffn_output_input",grouped.ptr(),padded,c);
        if(active_grouped_int8&&active_grouped_int8->enabled(trace_block,Int8Role::SplitOutput))
            active_grouped_int8->launch(trace_block,Int8Role::SplitOutput,grouped.ptr(),wo,x,cosine,padded_out->ptr(),padded,c,c,3,false);
        else launch_gemm_residual(grouped.ptr(),wo,x,cosine,padded_out->ptr(),padded,c,c,false);
    }
    if(padded==tokens)return padded_out;auto out=std::make_unique<HalfBuffer>(size_t(tokens)*c);if(current_stream())HIP_CHECK(hipMemcpyAsync(out->ptr(),padded_out->ptr(),out->count*sizeof(__half),hipMemcpyDeviceToDevice,current_stream()));else HIP_CHECK(hipMemcpy(out->ptr(),padded_out->ptr(),out->count*sizeof(__half),hipMemcpyDeviceToDevice));sync_gpu();return out;}

static std::pair<int,int> origin_for(int b){int phase=0;if(b==70)phase=1;else if(b!=0){struct R{int lo,hi,base;};R ranges[]={{1,4,1},{5,8,5},{9,14,9},{15,22,15},{23,30,23},{40,47,40},{48,55,48},{56,61,54},{62,65,62},{66,69,66}};for(auto r:ranges)if(b>=r.lo&&b<=r.hi){phase=b-r.base;break;}}std::pair<int,int>v[]={{0,0},{-4,-4},{0,-4},{-4,0}};return v[(phase%4+4)%4];}
static HalfPtr window_device(HalfBuffer&input,int h,int w,int c,int family,int block,const WindowWeights&weights,bool publish_output){
    observe_activation(block,"ffn_expand_input",input.ptr(),h*w,c);capture_int8_activation(block,"input",input,h*w,c);trace_stage(block,"input",input);
    const char*ffnStage=family==0?"c32_ffn":"window_ffn";
    const char*ffnFamilyStage=family==0?"c32_ffn":family==1?(c==64?"c64_ffn":c==128?"c128_ffn":"c256_ffn"):"c512_ffn";
    stage_event(ffnStage,true);stage_event(ffnFamilyStage,true);auto ffn=ffn_device(input,h*w,c,family,weights,block);stage_event(ffnFamilyStage,false);stage_event(ffnStage,false);
    observe_activation(block,"qkv_input",ffn->ptr(),h*w,c);trace_stage(block,"ffn",*ffn);
    auto origin=origin_for(block);int top=-origin.first,left=-origin.second,ph=(h+top+7)/8*8,pw=(w+left+7)/8*8,nwin=(ph/8)*(pw/8),all=nwin*64;
    const bool keepTrace=trace_requested(block);
    HalfPtr windows;
    if(keepTrace){windows=std::make_unique<HalfBuffer>(size_t(all)*c);partition_features<<<(windows->count+255)/256,256,0,current_stream()>>>(ffn->ptr(),windows->ptr(),h,w,c,top,left,ph,pw);sync_gpu();}
    HalfPtr q_owner,bias_owner,scale_owner,proj_owner,acos_owner;auto q=weight_ptr(weights.qkv,q_owner),bias=weight_ptr(weights.bias,bias_owner),scale=weight_ptr(weights.scale,scale_owner),proj=weight_ptr(weights.proj,proj_owner),acos=weight_ptr(weights.acos,acos_owner);
    const bool inspectAttention=active_graph_profile&&active_graph_profile->activationObserver;
    if(family==0&&!keepTrace&&!inspectAttention&&fused_c32_attention_enabled()){
        auto spatial=std::make_unique<HalfBuffer>(size_t(h)*w*c);
        const bool waveProjection=active_grouped_int8&&active_grouped_int8->c32AttentionWave;
        if(waveProjection){auto packed=active_grouped_int8->prepare_c32_attention(unsigned(block),q,proj);q=packed.first;proj=packed.second;}
        stage_event("c32_attention",true);
        launch_fused_c32_attention(ffn->ptr(),q,bias,scale,proj,acos,spatial->ptr(),
                                   nwin,h,w,top,left,pw,publish_output,parallel_window_enabled(),attention_lds_padding_enabled(),waveProjection);
        stage_event("c32_attention",false);
        return spatial;
    }
    const int attentionChannels=weights.attentionChannels;int heads=attentionChannels/32;
    if(attentionChannels<=0||attentionChannels%32)throw std::runtime_error("Invalid window attention width");
    std::unique_ptr<Buffer<float>> projected=keepTrace?std::make_unique<Buffer<float>>(size_t(all)*3*attentionChannels):nullptr;
    HalfPtr norms=keepTrace?std::make_unique<HalfBuffer>(size_t(nwin)*2*heads*64):nullptr;
    HalfPtr rawweights=keepTrace?std::make_unique<HalfBuffer>(size_t(nwin)*heads*4096):nullptr;
    HalfPtr scores=keepTrace?std::make_unique<HalfBuffer>(size_t(nwin)*heads*4096):nullptr;
    HalfPtr probs=keepTrace?std::make_unique<HalfBuffer>(size_t(nwin)*heads*4096):nullptr;
    Buffer<__half>published(size_t(nwin)*3*attentionChannels*64),attended(size_t(all)*attentionChannels);
    stage_event("window_qkv",true);
    if(keepTrace){launch_gemm(windows->ptr(),q,projected->ptr(),all,c,3*attentionChannels);normalize_windows<<<(nwin*3*heads*64+63)/64,64,0,current_stream()>>>(projected->ptr(),scale,norms->ptr(),published.ptr(),attentionChannels,nwin);}
    else if(active_grouped_int8&&active_grouped_int8->enabled(block,Int8Role::Qkv)){
        if(fused_qkv_normalize_enabled())active_grouped_int8->launch_window_qkv_normalized(
            block,ffn->ptr(),q,scale,published.ptr(),all,c,heads,h,w,top,left,pw);
        else{
            Buffer<__half>qkv(size_t(all)*3*attentionChannels);
            active_grouped_int8->launch_window_qkv(block,ffn->ptr(),q,qkv.ptr(),all,c,3*attentionChannels,h,w,top,left,pw);
            launch_qkv_normalize_half(qkv.ptr(),scale,published.ptr(),all,heads,64);
        }
    }else launch_gemm_qkv_normalize_window(ffn->ptr(),q,scale,published.ptr(),all,c,heads,h,w,top,left,pw);
    stage_event("window_qkv",false);
    stage_event("window_attention",true);
    if(keepTrace){
        score_windows<<<(nwin*heads*4096+255)/256,256,0,current_stream()>>>(published.ptr(),bias,scores->ptr(),heads,nwin);
        probability_windows<<<(nwin*heads*64+63)/64,64,0,current_stream()>>>(scores->ptr(),rawweights->ptr(),probs->ptr(),nwin*heads*64);
        attend_windows<<<(all*attentionChannels+255)/256,256,0,current_stream()>>>(probs->ptr(),published.ptr(),attended.ptr(),attentionChannels,nwin);
    }else if(parallel_window_enabled()&&attention_lds_padding_enabled())attention_windows_fused<true,66><<<nwin*heads,256,0,current_stream()>>>(published.ptr(),bias,attended.ptr(),attentionChannels,nwin);
    else if(parallel_window_enabled())attention_windows_fused<true><<<nwin*heads,256,0,current_stream()>>>(published.ptr(),bias,attended.ptr(),attentionChannels,nwin);
    else attention_windows_fused<false><<<nwin*heads,256,0,current_stream()>>>(published.ptr(),bias,attended.ptr(),attentionChannels,nwin);
    sync_gpu();stage_event("window_attention",false);observe_activation(block,"attention_projection_input",attended.ptr(),all,attentionChannels);
    stage_event("window_projection",true);
    auto spatial=std::make_unique<HalfBuffer>(size_t(h)*w*c);
    if(keepTrace){Buffer<__half>window_out(size_t(all)*c);Buffer<float>branch(size_t(all)*c);launch_gemm(attended.ptr(),proj,branch.ptr(),all,attentionChannels,c);residual_h<<<(window_out.count+255)/256,256,0,current_stream()>>>(branch.ptr(),windows->ptr(),acos,window_out.ptr(),window_out.count,c,false);sync_gpu();trace_stage(block,"branch",branch);trace_stage(block,"windows",*windows);trace_stage(block,"projected",*projected);trace_stage(block,"norms",*norms);trace_stage(block,"published",published);trace_stage(block,"scores",*scores);trace_stage(block,"rawweights",*rawweights);trace_stage(block,"probs",*probs);trace_stage(block,"attended",attended);trace_stage(block,"window_out",window_out);reverse_features<<<(spatial->count+255)/256,256,0,current_stream()>>>(window_out.ptr(),spatial->ptr(),h,w,c,top,left,pw);sync_gpu();if(publish_output){stage_event("window_projection",false);return publish_device(*spatial);}}
    else if(active_grouped_int8&&active_grouped_int8->enabled(block,Int8Role::AttentionProjection))active_grouped_int8->launch_window_projection(block,attended.ptr(),proj,ffn->ptr(),acos,spatial->ptr(),all,attentionChannels,c,publish_output,h,w,top,left,pw);
    else launch_gemm_residual_to_spatial(attended.ptr(),proj,ffn->ptr(),acos,spatial->ptr(),all,attentionChannels,c,publish_output,h,w,top,left,pw);
    stage_event("window_projection",false);
    return spatial;
}

static HalfPtr global_device(HalfBuffer&input,Reader*r,int m,int block){
    constexpr int k=1024,h=4096;if(m<1||(m&1))throw std::runtime_error("unsupported global token count");
    int device=0;hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device));HIP_CHECK(hipGetDeviceProperties(&properties,device));
    if(std::size_t(m)*sizeof(__half)>properties.sharedMemPerBlock)
        throw std::runtime_error("global attention row exceeds device shared memory");
    observe_activation(block,"ffn_expand_input",input.ptr(),m,k);capture_int8_activation(block,"input",input,m,k);
    auto w1d=weight(r,block,"ffn_expand",k*h),w2d=weight(r,block,"ffn_contract",h*k),fcosd=weight(r,block,"ffn_cos",k),
        wqd=weight(r,block,"qkv",k*3*k),scaled=weight(r,block,"effective_attention_scale",32),
        wpd=weight(r,block,"attention_projection",k*k),acosd=weight(r,block,"attention_cos",k);
    HalfPtr w1o,w2o,fcoso,wqo,scaleo,wpo,akoso;auto w1=weight_ptr(w1d,w1o),w2=weight_ptr(w2d,w2o),fcos=weight_ptr(fcosd,fcoso),
        wq=weight_ptr(wqd,wqo),scale=weight_ptr(scaled,scaleo),wp=weight_ptr(wpd,wpo),acos=weight_ptr(acosd,akoso);
    Buffer<__half>hidden(size_t(m)*h),ffn(size_t(m)*k),published(size_t(m)*3*k),attended(size_t(m)*k),output(size_t(m)*k);
    stage_event("global_ffn",true);
    const bool inspectHandoff=trace_requested(block)||int8_capture_requested(block)||
        (active_graph_profile&&active_graph_profile->activationObserver);
    const bool chain=m>=256&&fused_qkv_normalize_enabled()&&active_grouped_int8&&active_grouped_int8->qkvHardwareLayout;
    const bool handoff=!inspectHandoff&&active_grouped_int8&&
        active_grouped_int8->launch_ffn_handoff(block,input.ptr(),w1,w2,fcos,hidden.ptr(),ffn.ptr(),m,k,h,chain);
    if(!handoff){
        if(active_grouped_int8&&active_grouped_int8->enabled(block,Int8Role::FfnExpand))
            active_grouped_int8->launch(block,Int8Role::FfnExpand,input.ptr(),w1,nullptr,nullptr,hidden.ptr(),m,k,h,1,true);
        else launch_gemm_gate_publish(input.ptr(),w1,hidden.ptr(),m,k,h);
        observe_activation(block,"ffn_contract_input",hidden.ptr(),m,h);capture_int8_activation(block,"ffn_hidden",hidden,m,h);
        if(active_grouped_int8&&active_grouped_int8->enabled(block,Int8Role::FfnContract))
            active_grouped_int8->launch(block,Int8Role::FfnContract,hidden.ptr(),w2,input.ptr(),fcos,ffn.ptr(),m,h,k,3,false);
        else launch_gemm_residual(hidden.ptr(),w2,input.ptr(),fcos,ffn.ptr(),m,h,k,false);
    }
    stage_event("global_ffn",false);
    observe_activation(block,"qkv_input",ffn.ptr(),m,k);
    stage_event("global_qkv",true);
    if(handoff&&chain)active_grouped_int8->launch_handoff_qkv(block,wq,scale,published.ptr(),m,k);
    else if(active_grouped_int8&&active_grouped_int8->enabled(block,Int8Role::Qkv)){
        if(fused_qkv_normalize_enabled())active_grouped_int8->launch_qkv_normalized(
            block,ffn.ptr(),wq,scale,published.ptr(),m,k,32,m);
        else{
            Buffer<__half>qkv(size_t(m)*3*k);
            active_grouped_int8->launch(block,Int8Role::Qkv,ffn.ptr(),wq,nullptr,nullptr,qkv.ptr(),m,k,3*k,0,false);
            launch_qkv_normalize_half(qkv.ptr(),scale,published.ptr(),m,32,m);
        }
    }else launch_gemm_qkv_normalize(ffn.ptr(),wq,scale,published.ptr(),m,k,32,m);
    stage_event("global_qkv",false);
    stage_event("global_attention",true);
    global_attention_fused<<<32*m,64,std::size_t(m)*sizeof(__half),current_stream()>>>(published.ptr(),attended.ptr(),m);sync_gpu();
    stage_event("global_attention",false);
    observe_activation(block,"attention_projection_input",attended.ptr(),m,k);
    stage_event("global_projection",true);
    if(active_grouped_int8&&active_grouped_int8->enabled(block,Int8Role::AttentionProjection))
        active_grouped_int8->launch(block,Int8Role::AttentionProjection,attended.ptr(),wp,ffn.ptr(),acos,output.ptr(),m,k,k,3,false);
    else launch_gemm_residual(attended.ptr(),wp,ffn.ptr(),acos,output.ptr(),m,k,k,false);
    auto result=publish_device(output);
    stage_event("global_projection",false);
    return result;
}

static std::vector<unsigned> runtime_block_header(int block){
    int c=32,family=0;
    if((block>=5&&block<=8)||(block>=62&&block<=65)){c=64;family=1;}
    else if((block>=9&&block<=14)||(block>=56&&block<=61)){c=128;family=1;}
    else if((block>=15&&block<=22)||(block>=48&&block<=55)){c=256;family=1;}
    else if((block>=23&&block<=30)||(block>=40&&block<=47)){c=512;family=2;}
    else if(block>=31&&block<=38){c=1024;family=3;}
    else if(block==39){c=512;family=4;}
    int kind=0;
    if(block==0)kind=3;else if(block==70)kind=4;
    else if(block==4||block==8||block==14||block==22)kind=1;
    else if(block==48||block==56||block==62||block==66)kind=2;
    return{unsigned(block),unsigned(c),unsigned(family),unsigned(kind)};
}

static HalfPtr execute_spatial_graph(HalfPtr current,int input_h,int input_w,Reader*reader,const NrV2::SpatialGraphProfile* profile=nullptr){
    const auto blockEvents=profile?profile->events:nullptr;
    int h=input_h,w=input_w,c=16;HalfPtr fullskip,split_skip;std::vector<HalfPtr>skips(4);std::vector<std::pair<int,int>>skip_dims(4);int split_h=0,split_w=0;
    for(int block=0;block<71;++block){if(blockEvents){HIP_CHECK(hipEventRecord(blockEvents[block],current_stream()));if(profile->isolateBlocks)HIP_CHECK(hipEventSynchronize(blockEvents[block]));}const auto blockStart=profile&&profile->isolateBlocks?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};auto bh=reader?reader->read<unsigned>(4):runtime_block_header(block);if(int(bh[0])!=block)throw std::runtime_error("block order");int bc=active_linked_w128&&LinkedW128Plan::covered(unsigned(block))?128:int(bh[1]),family=bh[2],kind=bh[3];if(family<=2){auto ww=read_window_weights(reader,bc,family,kind,block);if(block==0){observe_activation(block,"adapter_input",current->ptr(),size_t(h)*w,16);HalfPtr owner;auto adapter=weight_ptr(ww.adapter,owner);auto adapted=std::make_unique<HalfBuffer>(size_t(h)*w*32);stage_event("adapter_projection",true);launch_adapter_16x32(current->ptr(),adapter,adapted->ptr(),h*w);stage_event("adapter_projection",false);current=std::move(adapted);c=32;}else if(kind==2){observe_activation(block,"upsample_projection_input",current->ptr(),size_t(h)*w,c);stage_event("upsample_projection",true);auto projected=selective_linear_device(*current,h*w,c,bc,ww.up,block,Int8Role::UpsampleProjection,0,false);stage_event("upsample_projection",false);int index=block==48?3:block==56?2:block==62?1:0,targeth=skip_dims[index].first,targetw=skip_dims[index].second;if(!skips[index]||targeth>h*2||targetw>w*2)throw std::runtime_error("invalid decoder crop");HalfPtr sine_owner;auto sine=weight_ptr(ww.upsin,sine_owner);auto merged=std::make_unique<HalfBuffer>(size_t(targeth)*targetw*bc);upsample_merge_kernel<<<(merged->count+255)/256,256,0,current_stream()>>>(projected->ptr(),skips[index]->ptr(),sine,merged->ptr(),w,targeth,targetw,bc);sync_gpu();skips[index].reset();current=std::move(merged);h=targeth;w=targetw;c=bc;std::printf("GEOMETRY decoder block=%d extent=%dx%d\n",block,h,w);}else if(block==70){if(!fullskip||input_h>h*2||input_w>w*2)throw std::runtime_error("invalid final crop");HalfPtr sine_owner,cosine_owner;auto sine=weight_ptr(ww.merge_sin,sine_owner),cosine=weight_ptr(ww.merge_cos,cosine_owner);auto merged=std::make_unique<HalfBuffer>(size_t(input_h)*input_w*32);final_merge_kernel<<<(merged->count+255)/256,256,0,current_stream()>>>(current->ptr(),fullskip->ptr(),sine,cosine,merged->ptr(),w,input_h,input_w,32);sync_gpu();fullskip.reset();current=std::move(merged);h=input_h;w=input_w;c=32;}
        auto transformed=window_device(*current,h,w,bc,family,block,ww,kind!=1&&block!=0&&block!=70);c=bc;if(block==0){fullskip=publish_device(*transformed);observe_activation(block,"full_skip",fullskip->ptr(),size_t(h)*w,c);check_codes(reader,*fullskip,"block0.full_skip");current=publish_device(*pool_device(*transformed,h,w,c));h/=2;w/=2;check_codes(reader,*current,"block0.down");}
        else if(kind==1){if(block==22&&(h%8||w%8)){int ph=(h+7)/8*8,pw=(w+7)/8*8;std::printf("GEOMETRY block22 transform=%dx%d padded=%dx%d\n",h,w,ph,pw);transformed=pad_device(*transformed,h,w,ph,pw,c);h=ph;w=pw;}auto pooled=pool_device(*transformed,h,w,c);auto pooledpub=publish_device(*pooled);observe_activation(block,"downsample_input",pooledpub->ptr(),size_t(h/2)*(w/2),c);const int nextChannels=active_linked_w128&&block==14?128:active_linked_w128&&block==22?512:2*c;stage_event("downsample_projection",true);current=selective_linear_device(*pooledpub,(h/2)*(w/2),c,nextChannels,ww.down,block,Int8Role::Downsample,2,true);stage_event("downsample_projection",false);h/=2;w/=2;c=nextChannels;std::string label="block"+std::to_string(block)+".down";check_codes(reader,*current,label.c_str());}
        else if(block==70){observe_activation(block,"head_input",transformed->ptr(),size_t(h)*w,32);HalfPtr gain_owner,conv_owner;auto gain=weight_ptr(ww.gain,gain_owner),conv=weight_ptr(ww.conv,conv_owner);auto output=std::make_unique<HalfBuffer>(size_t(h)*w*4);head_kernel<<<(output->count+255)/256,256,0,current_stream()>>>(transformed->ptr(),gain,conv,output->ptr(),h*w);sync_gpu();observe_activation(block,"head_output",output->ptr(),size_t(h)*w,4);trace_stage(70,"head",*output);check_half_output(reader,*output,"block70.output_half");current=std::move(output);c=4;}
        else{current=std::move(transformed);std::string label="block"+std::to_string(block);check_codes(reader,*current,label.c_str());int index=block==3?0:block==7?1:block==13?2:block==21?3:-1;if(index>=0){skips[index]=clone_half(*current);skip_dims[index]={h,w};}}}
      else if(family==3){current=global_device(*current,reader,h*w,block);std::string label="block"+std::to_string(block);check_codes(reader,*current,label.c_str());}
      else if(family==4){auto projection=weight(reader,block,"bridge_projection",1024*512),sinev=weight(reader,block,"bridge_sin",512);observe_activation(block,"bridge_projection_input",current->ptr(),size_t(h)*w,1024);stage_event("bridge_projection",true);auto projected=selective_linear_device(*current,h*w,1024,512,projection,block,Int8Role::BridgeProjection,0,false);stage_event("bridge_projection",false);HalfPtr sine_owner;auto sine=weight_ptr(sinev,sine_owner);if(!split_skip||split_h>h*2||split_w>w*2)throw std::runtime_error("invalid bridge crop");auto merged=std::make_unique<HalfBuffer>(split_skip->count);upsample_merge_kernel<<<(merged->count+255)/256,256,0,current_stream()>>>(projected->ptr(),split_skip->ptr(),sine,merged->ptr(),w,split_h,split_w,512);sync_gpu();split_skip.reset();current=std::move(merged);h=split_h;w=split_w;c=512;std::printf("GEOMETRY bridge extent=%dx%d\n",h,w);check_codes(reader,*current,"block39");}
      if(block==30){split_skip=clone_half(*current);split_h=h;split_w=w;auto gp=weight(reader,block,"global_projection",512*1024);int ph=(h+7)/8*8,pw=(w+7)/8*8;auto padded=pad_device(*current,h,w,ph,pw,512),pooled=pool_device(*padded,ph,pw,512);observe_activation(block,"global_projection_input",pooled->ptr(),size_t(ph/2)*(pw/2),512);stage_event("global_input_projection",true);current=selective_linear_device(*pooled,(ph/2)*(pw/2),512,1024,gp,block,Int8Role::GlobalProjection,2,true);stage_event("global_input_projection",false);h=ph/2;w=pw/2;c=1024;std::printf("GEOMETRY global input=%dx%d tokens=%d\n",h,w,h*w);check_codes(reader,*current,"block30.global_input");}
      if(profile&&profile->isolateBlocks){HIP_CHECK(hipStreamSynchronize(current_stream()));profile->blockWallMs[block]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-blockStart).count();}}
    if(blockEvents)HIP_CHECK(hipEventRecord(blockEvents[71],current_stream()));
    if(current->count!=size_t(input_h)*input_w*4)throw std::runtime_error("final shape");
    return current;
}

namespace NrV2 {
struct SpatialGraphRuntime::Impl {
    Impl(const GpuModel& value,hipStream_t valueStream):model(value),stream(valueStream){
        if(!stream)throw std::runtime_error("spatial runtime requires an owning stream");
        const unsigned groupSize=global_int8_group_size();const Int8Scope scope=grouped_int8_scope();
        if(c32_int8_group_size()&&groupSize!=32&&groupSize!=64)
            throw std::runtime_error("C32 INT8 requires grouped INT8 group 32 or 64");
        if(resident_branch_enabled()&&groupSize!=64)
            throw std::runtime_error("resident branched W8A8 requires grouped INT8 group 64");
        linkedW128=std::make_unique<LinkedW128Plan>(model,stream);
        if(linkedW128->enabled()){if(groupSize)throw std::runtime_error("Linked W128 and grouped INT8 cannot share the current weight contract");std::printf("EXPERIMENT linked_w128 blocks=16 branches=4 physical_bundle=1\n");}
        if(groupSize){int8=std::make_unique<GroupedInt8Plan>(stream,groupSize,scope);std::printf("EXPERIMENT grouped_int8=W8A8_DOT4 group=%u scope=%s\n",groupSize,grouped_scope_name(scope));}
        compact=std::make_unique<CompactC32Plan>(model,stream);if(compact->enabled())std::printf("EXPERIMENT compact_c32 blocks=%u physical_bundle=1\n",compact->blocks);
        compactBranched=std::make_unique<CompactBranchedPlan>(model,stream);if(compactBranched->enabled())std::printf("EXPERIMENT compact_branched blocks=%u physical_bundle=1\n",compactBranched->blocks);
        if(linkedW128->enabled())for(unsigned block=0;block<71;++block)if(LinkedW128Plan::covered(block)&&compactBranched->find(block))throw std::runtime_error("Linked W128 conflicts with C256 compact branched entry");
    }
    const GpuModel& model;
    hipStream_t stream=nullptr;
    DeviceBufferArena workspace;
    std::unique_ptr<LinkedW128Plan> linkedW128;
    std::unique_ptr<GroupedInt8Plan> int8;
    std::unique_ptr<CompactC32Plan> compact;
    std::unique_ptr<CompactBranchedPlan> compactBranched;
    std::size_t frames=0;
    std::size_t preparations=0;
    std::uint32_t checkedWidth=0,checkedHeight=0;
    bool active=false;
};

SpatialGraphRuntime::SpatialGraphRuntime(const GpuModel& model,hipStream_t stream):impl_(new Impl(model,stream)){}
SpatialGraphRuntime::~SpatialGraphRuntime(){delete impl_;}
bool SpatialGraphRuntime::Release() noexcept {
    if(!impl_)return true;
    if(impl_->active||(impl_->int8&&!impl_->int8->release())||(impl_->compact&&!impl_->compact->release())||(impl_->compactBranched&&!impl_->compactBranched->release())||(impl_->linkedW128&&!impl_->linkedW128->release())||!impl_->workspace.release_all())return false;
    delete impl_;impl_=nullptr;return true;
}
void SpatialGraphRuntime::Enqueue(const void* preparedDevice,std::uint32_t height,
                                  std::uint32_t width,void* outputDevice,const SpatialGraphProfile* profile){
    if(!impl_||!preparedDevice||!outputDevice||!SupportedGraphExtent(width,height))
        throw std::runtime_error("spatial runtime arguments");
    if(impl_->active||active_buffer_arena||runtime_model||active_graph_stream||active_grouped_int8||active_compact_c32||active_compact_branched||active_linked_w128)
        throw std::runtime_error("spatial runtime is already active");
    if(impl_->checkedWidth!=width||impl_->checkedHeight!=height){
        // Retain the conservative envelope through 1440p. Beyond it, use
        // the measured full-frame workspace slope plus bounded headroom for
        // a diagnostic 2160p run; HIP allocations still enforce real VRAM.
        const std::size_t pixels=std::size_t(width)*height;
        const std::size_t workspaceBytesPerPixel=
            pixels>std::size_t(2560)*1472?1664:2048;
        constexpr std::size_t safetyHeadroom=512ull*1024*1024;
        if(pixels>(std::numeric_limits<std::size_t>::max()-64ull*1024*1024)/workspaceBytesPerPixel)
            throw std::runtime_error("spatial workspace estimate overflow");
        const std::size_t estimate=pixels*workspaceBytesPerPixel+64ull*1024*1024;
        std::size_t freeBytes=0,totalBytes=0;HIP_CHECK(hipMemGetInfo(&freeBytes,&totalBytes));
        const std::size_t reusable=impl_->workspace.reserved_bytes;
        if(estimate>freeBytes+reusable||safetyHeadroom>freeBytes+reusable-estimate)
            throw std::runtime_error("insufficient VRAM for spatial graph extent");
        std::printf("MEMORY_PREFLIGHT extent=%ux%u estimate_bytes=%zu available_bytes=%zu total_bytes=%zu\n",
                    width,height,estimate,freeBytes+reusable,totalBytes);
        impl_->checkedWidth=width;impl_->checkedHeight=height;
    }
    impl_->active=true;active_buffer_arena=&impl_->workspace;active_buffer_guards=graph_buffer_guards_enabled();runtime_model=&impl_->model;active_grouped_int8=impl_->int8.get();active_compact_c32=impl_->compact.get();active_compact_branched=impl_->compactBranched.get();active_linked_w128=profile&&profile->disableLinkedW128?nullptr:impl_->linkedW128.get()->enabled()?impl_->linkedW128.get():nullptr;active_graph_profile=profile;
    active_graph_stream=impl_->stream;
    try{
        auto input=std::make_unique<HalfBuffer>(size_t(height)*width*16);
        HIP_CHECK(hipMemcpyAsync(input->ptr(),preparedDevice,input->count*sizeof(__half),
                                 hipMemcpyDeviceToDevice,impl_->stream));
        auto output=execute_spatial_graph(std::move(input),int(height),int(width),nullptr,profile);
        HIP_CHECK(hipMemcpyAsync(outputDevice,output->ptr(),output->count*sizeof(__half),
                                 hipMemcpyDeviceToDevice,impl_->stream));
        sync_gpu();++impl_->frames;runtime_model=nullptr;active_buffer_arena=nullptr;active_buffer_guards=true;active_grouped_int8=nullptr;active_compact_c32=nullptr;active_compact_branched=nullptr;active_linked_w128=nullptr;active_graph_profile=nullptr;
        active_graph_stream=nullptr;impl_->active=false;
    }catch(...){runtime_model=nullptr;active_buffer_arena=nullptr;active_buffer_guards=true;active_graph_stream=nullptr;active_grouped_int8=nullptr;active_compact_c32=nullptr;active_compact_branched=nullptr;active_linked_w128=nullptr;active_graph_profile=nullptr;
        impl_->active=false;throw;}
}
void SpatialGraphRuntime::Execute(const void* preparedDevice,std::uint32_t height,
                                  std::uint32_t width,void* outputDevice){
    Enqueue(preparedDevice,height,width,outputDevice);
    HIP_CHECK(hipStreamSynchronize(impl_->stream));
    ValidateExperimental();
}
static void prepare_observer(void*,std::uint32_t,const char*,const void*,std::size_t,
                             std::size_t,hipStream_t){}
void SpatialGraphRuntime::Prepare(const void* preparedDevice,std::uint32_t height,
                                  std::uint32_t width,void* outputDevice){
    const auto frames=impl_->frames;
    Execute(preparedDevice,height,width,outputDevice);
    // Diagnostic activation capture materializes the C32 FFN hidden boundary,
    // whereas production keeps it in LDS. Warm both arena lifetimes so enabling
    // profiling after Prepare cannot allocate during a frame.
    SpatialGraphProfile diagnostic{};
    diagnostic.activationObserver=&prepare_observer;
    Enqueue(preparedDevice,height,width,outputDevice,&diagnostic);
    HIP_CHECK(hipStreamSynchronize(impl_->stream));
    ValidateExperimental();
    impl_->frames=frames;++impl_->preparations;
}
SpatialGraphStats SpatialGraphRuntime::Stats()const noexcept{
    if(!impl_)return{};const auto&w=impl_->workspace;
    return{w.allocation_count,w.reuse_count,w.reserved_bytes,w.peak_in_use_bytes,
           impl_->frames,impl_->preparations,impl_->int8?impl_->int8->reserved_bytes():0,
           impl_->int8?impl_->int8->invocations():0,bool(impl_->int8)};
}
void SpatialGraphRuntime::ValidateExperimental()const{if(!impl_)throw std::runtime_error("spatial runtime is released");if(impl_->active)throw std::runtime_error("cannot validate active spatial runtime");if(impl_->int8)impl_->int8->validate();}
} // namespace NrV2

#ifndef NR_SPATIAL_GRAPH_LIBRARY
int main(int argc,char**argv){try{
    if(argc<2)throw std::runtime_error("usage: spatial_graph_probe fixture [options] | --runtime package package_sha source_sha height width --input raw_half_file [options]");
    bool runtime=!std::strcmp(argv[1],"--runtime");
    int option_start=2,input_h=0,input_w=0;
    std::unique_ptr<Reader> reader;
    if(runtime){
        if(argc<9)throw std::runtime_error("incomplete runtime arguments");
        runtime_package=new NrV2::WeightPackage(NrV2::WeightPackage::LoadFile(argv[2],NrV2::ParseSha256(argv[3]),NrV2::ParseSha256(argv[4])));
        input_h=std::atoi(argv[5]);input_w=std::atoi(argv[6]);option_start=7;
        if(!NrV2::SupportedGraphExtent(input_w,input_h))throw std::runtime_error("runtime extent");
    }else reader=std::make_unique<Reader>(argv[1],512ll*1024*1024);
    const char*output_path=nullptr,*input_path=nullptr,*input2_path=nullptr,*output2_path=nullptr;
    for(int i=option_start;i<argc;++i){
        if(!std::strcmp(argv[i],"--rebase"))rebase_checkpoints=true;
        else if(!std::strcmp(argv[i],"--input")&&i+1<argc)input_path=argv[++i];
        else if(!std::strcmp(argv[i],"--output")&&i+1<argc)output_path=argv[++i];
        else if(!std::strcmp(argv[i],"--input2")&&i+1<argc)input2_path=argv[++i];
        else if(!std::strcmp(argv[i],"--output2")&&i+1<argc)output2_path=argv[++i];
        else throw std::runtime_error("unknown or incomplete option");
    }
    if((input2_path||output2_path)&&(!runtime||!input2_path||!output2_path))throw std::runtime_error("second frame requires runtime input and output");
    bool v1=false,v2=false;std::vector<__half>prepared;
    if(runtime){
        if(!input_path)throw std::runtime_error("runtime input is required");
        prepared.resize(size_t(input_h)*input_w*16);
    }else{
        auto magic=reader->read<char>(8);auto header=reader->read<unsigned>(4);
        v1=!std::memcmp(magic.data(),"NRSPT001",8);v2=!std::memcmp(magic.data(),"NRSPT002",8);
        input_h=header[0];input_w=header[1];
        if((!v1&&!v2)||header[2]!=16||header[3]!=71||!NrV2::SupportedGraphExtent(input_w,input_h)||(v1&&(input_h!=128||input_w!=128)))throw std::runtime_error("fixture header");
        prepared=reader->read<__half>(size_t(input_h)*input_w*16);
    }
    if(input_path){std::ifstream input(input_path,std::ios::binary);if(!input.read(reinterpret_cast<char*>(prepared.data()),prepared.size()*sizeof(__half)))throw std::runtime_error("input override truncated");char trailing;if(input.read(&trailing,1))throw std::runtime_error("input override trailing bytes");}
    int devices=0,selected=-1;HIP_CHECK(hipGetDeviceCount(&devices));for(int i=0;i<devices;++i){hipDeviceProp_t p{};HIP_CHECK(hipGetDeviceProperties(&p,i));if(!std::strncmp(p.gcnArchName,"gfx1030",7)){selected=i;std::printf("device=%s arch=%s mode=%s source=%s\n",p.name,p.gcnArchName,rebase_checkpoints?"rebased":"connected",runtime?"NRWGT001":(v2?"NRSPT002":"NRSPT001"));break;}}if(selected<0)throw std::runtime_error("gfx1030 required");HIP_CHECK(hipSetDevice(selected));std::unique_ptr<NrV2::GpuModel>model;if(runtime){model=std::make_unique<NrV2::GpuModel>(*runtime_package);runtime_model=model.get();std::printf("MODEL tensors=%zu bytes=%zu uploads=%zu\n",model->TensorCount(),model->ByteLength(),model->UploadOperations());}
    DeviceBufferArena workspace;if(runtime)active_buffer_arena=&workspace;size_t first_frame_allocations=0;int frame_count=input2_path?2:1;
    for(int frame=0;frame<frame_count;++frame){if(frame==1){std::ifstream input(input2_path,std::ios::binary);if(!input.read(reinterpret_cast<char*>(prepared.data()),prepared.size()*sizeof(__half)))throw std::runtime_error("second input truncated");char trailing;if(input.read(&trailing,1))throw std::runtime_error("second input trailing bytes");}
    {auto current=execute_spatial_graph(upload_half(prepared),input_h,input_w,reader.get());auto final=current->read();const char*frame_output=frame?output2_path:output_path;
    if(frame_output){std::ofstream output(frame_output,std::ios::binary);if(!output.write(reinterpret_cast<const char*>(final.data()),final.size()*sizeof(__half)))throw std::runtime_error("unable to write output");}}
    if(frame==0)first_frame_allocations=workspace.allocation_count;}
    if(reader)reader->end();active_buffer_arena=nullptr;
    if(runtime){if(workspace.allocation_count!=first_frame_allocations)throw std::runtime_error("workspace grew after first frame");std::printf("WORKSPACE frames=%d allocations_first=%zu allocations_final=%zu reuses=%zu reserved_bytes=%zu peak_in_use_bytes=%zu\n",frame_count,first_frame_allocations,workspace.allocation_count,workspace.reuse_count,workspace.reserved_bytes,workspace.peak_in_use_bytes);}
    if(cleanup_failed)throw std::runtime_error("cleanup");if(runtime)std::printf("PASS runtime_graph source=NRWGT001 blocks=71 frames=%d prepared=%dx%dx16 output=%dx%dx4 package=%s guards=PASS\n",frame_count,input_h,input_w,input_h,input_w,NrV2::FormatSha256(runtime_package->PackageHash()).c_str());else std::printf("PASS spatial_graph mode=%s blocks=71 checkpoints=73 prepared=%dx%dx16 output=%dx%dx4 checkpoint_differences=%llu/%llu worst_e4m3_step=%u guards=PASS\n",rebase_checkpoints?"rebased":"connected",input_h,input_w,input_h,input_w,checkpoint_differences,checkpoint_values,checkpoint_worst_step);runtime_model=nullptr;model.reset();delete runtime_package;runtime_package=nullptr;return 0;}catch(const std::exception&e){active_buffer_arena=nullptr;runtime_model=nullptr;delete runtime_package;runtime_package=nullptr;std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
#endif
