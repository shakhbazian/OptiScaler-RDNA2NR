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
    // Independent pair transforms run together; the dependent half sum stays ordered.
    for(int col=tid*2;col<m;col+=blockDim.x*2){
        unsigned bits[2];
        for(int pair=0;pair<2;++pair){
            __half value=__hfma(probabilities[col+pair],__float2half(.044921875f),__float2half(1.30078125f));
            value=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(value))));
            bits[pair]=__half_as_ushort(value);
        }
        const unsigned transformed=((bits[0]|bits[1]<<16)<<5)+0x7ff88000u;
        probabilities[col]=__ushort_as_half(transformed&65535);
        probabilities[col+1]=__ushort_as_half(transformed>>16);
    }
    __syncthreads();
    __shared__ __half sharedInverse;
    if(tid==0){
        __half total=__float2half(0);
        for(int col=0;col<m;++col)total=__hadd(total,probabilities[col]);
        sharedInverse=__float2half(1.f/__half2float(total));
    }
    __syncthreads();
    for(int col=tid;col<m;col+=blockDim.x)probabilities[col]=pubh(__hmul(probabilities[col],sharedInverse));
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
    unsigned k=0,n=0,groupSize=0,groups=0,batches=1,epilogue=0;bool publish=false,packed=false,hardwarePacked=false,researchFolded=false,researchLong=false;
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

// Weight preparation runs once per graph context, before measured frames.
// A is divided by D in the resident kernel; these weights are multiplied by D.
__global__ void research_balance_weights(const __half* source,__half* output,
    unsigned block,unsigned k,unsigned n,unsigned offset,unsigned* status) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=k*n)return;
    const __half value=__float2half(__half2float(source[i])/research_inverse(block,offset+i/n));
    if(!isfinite(__half2float(value)))atomicOr(status,1u);
    output[i]=value;
}

// Fixed ranges selected on eight fit scenes; independent of image dimensions.
__device__ __constant__ float research_large_scale[1320]={3.464567065e-01f,5.039370060e-01f,3.464567065e-01f,2.362204790e-01f,4.625984132e-01f,2.834645808e-01f,6.299212575e-01f,2.834645808e-01f,5.039370060e-01f,1.732283533e-01f,4.094488323e-01f,2.834645808e-01f,4.409448802e-01f,2.362204790e-01f,3.779527545e-01f,3.779527545e-01f,3.464567065e-01f,2.834645808e-01f,4.724409580e-01f,3.779527545e-01f,3.779527545e-01f,3.464567065e-01f,2.362204790e-01f,2.519685030e-01f,3.149606287e-01f,2.834645808e-01f,3.464567065e-01f,3.464567065e-01f,3.149606287e-01f,3.681102395e-01f,2.047244161e-01f,1.574803144e-01f,2.519685030e-01f,3.779527545e-01f,2.519685030e-01f,1.889763772e-01f,3.149606287e-01f,3.149606287e-01f,2.362204790e-01f,2.519685030e-01f,2.519685030e-01f,3.149606287e-01f,2.047244161e-01f,1.732283533e-01f,2.834645808e-01f,3.464567065e-01f,1.574803144e-01f,1.574803144e-01f,2.519685030e-01f,3.051181138e-01f,1.259842515e-01f,1.181102395e-01f,2.834645808e-01f,2.834645808e-01f,1.732283533e-01f,1.417322904e-01f,2.204724401e-01f,2.519685030e-01f,1.417322904e-01f,1.574803144e-01f,2.519685030e-01f,2.834645808e-01f,1.417322904e-01f,1.574803144e-01f,3.149606287e-01f,2.519685030e-01f,2.362204790e-01f,2.834645808e-01f,1.889763772e-01f,1.998031437e-01f,1.732283533e-01f,2.519685030e-01f,3.464567065e-01f,2.519685030e-01f,2.519685030e-01f,2.834645808e-01f,1.889763772e-01f,2.519685030e-01f,1.417322904e-01f,1.732283533e-01f,3.779527545e-01f,2.519685030e-01f,2.519685030e-01f,2.519685030e-01f,1.417322904e-01f,1.732283533e-01f,2.519685030e-01f,1.574803144e-01f,3.149606287e-01f,2.047244161e-01f,2.519685030e-01f,2.204724401e-01f,2.047244161e-01f,1.181102395e-01f,1.574803144e-01f,1.259842515e-01f,3.779527545e-01f,2.362204790e-01f,2.834645808e-01f,2.362204790e-01f,2.519685030e-01f,1.889763772e-01f,1.732283533e-01f,2.362204790e-01f,3.149606287e-01f,2.362204790e-01f,2.362204790e-01f,2.204724401e-01f,1.181102395e-01f,2.519685030e-01f,2.362204790e-01f,3.149606287e-01f,3.464567065e-01f,2.834645808e-01f,2.834645808e-01f,2.047244161e-01f,1.417322904e-01f,1.574803144e-01f,1.732283533e-01f,1.574803144e-01f,3.149606287e-01f,3.464567065e-01f,3.779527545e-01f,2.834645808e-01f,1.259842515e-01f,3.149606287e-01f,1.417322904e-01f,1.574803144e-01f,3.149606287e-01f,2.047244161e-01f,2.834645808e-01f,2.047244161e-01f,1.023622081e-01f,1.259842515e-01f,1.259842515e-01f,1.181102395e-01f,3.149606287e-01f,2.519685030e-01f,2.834645808e-01f,2.519685030e-01f,1.368110180e-01f,1.181102395e-01f,1.889763772e-01f,1.574803144e-01f,3.779527545e-01f,1.732283533e-01f,2.312992066e-01f,2.519685030e-01f,2.204724401e-01f,9.448818862e-02f,1.181102395e-01f,1.181102395e-01f,3.149606287e-01f,1.683070809e-01f,2.362204790e-01f,2.519685030e-01f,1.181102395e-01f,1.102362201e-01f,1.368110180e-01f,1.102362201e-01f,3.779527545e-01f,1.574803144e-01f,2.155511826e-01f,1.889763772e-01f,2.519685030e-01f,1.102362201e-01f,8.661417663e-02f,1.417322904e-01f,3.149606287e-01f,1.574803144e-01f,2.204724401e-01f,2.519685030e-01f,1.102362201e-01f,1.259842515e-01f,1.102362201e-01f,1.102362201e-01f,3.464567065e-01f,2.047244161e-01f,2.834645808e-01f,2.312992066e-01f,1.102362201e-01f,1.181102395e-01f,1.102362201e-01f,1.417322904e-01f,3.149606287e-01f,1.574803144e-01f,3.464567065e-01f,2.519685030e-01f,1.574803144e-01f,1.732283533e-01f,1.574803144e-01f,2.362204790e-01f,1.059301198e-01f,1.544737369e-01f,1.032464951e-01f,9.651821107e-02f,1.563192010e-01f,1.312746108e-01f,1.376722455e-01f,1.560039371e-01f,1.968503930e-02f,1.476377994e-02f,1.279527601e-02f,1.574803144e-02f,1.771653630e-02f,1.377952751e-02f,1.476377994e-02f,1.574803144e-02f,1.417322904e-01f,2.519685030e-01f,1.574803144e-01f,1.417322904e-01f,1.732283533e-01f,1.889763772e-01f,2.204724401e-01f,2.047244161e-01f,4.693279099e-10f,1.771653630e-02f,1.574803144e-02f,9.842519648e-03f,1.181102395e-01f,1.107283519e-03f,4.693279099e-10f,1.377952751e-02f,2.519685030e-01f,8.058562875e-02f,1.732283533e-01f,7.837106287e-02f,7.726377994e-02f,1.090674177e-01f,1.229084656e-01f,1.147883832e-01f,1.142347455e-01f,9.842519648e-03f,9.842519648e-03f,1.279527601e-02f,1.181102358e-02f,8.858268149e-03f,1.082677208e-02f,1.082677208e-02f,1.082677208e-02f,1.259842515e-01f,9.448818862e-02f,9.448818862e-02f,9.448818862e-02f,1.102362201e-01f,1.417322904e-01f,1.417322904e-01f,1.574803144e-01f,2.559055202e-02f,3.149606287e-02f,1.181102358e-02f,1.181102358e-02f,1.574803144e-02f,1.181102358e-02f,1.082677208e-02f,1.574803144e-02f,1.417322904e-01f,7.078540325e-02f,7.381889969e-02f,7.853638381e-02f,6.709446013e-02f,7.185039669e-02f,9.718334675e-02f,1.162032485e-01f,1.419783533e-01f,5.118110403e-02f,5.511811003e-02f,3.149606287e-02f,2.165354416e-02f,7.874015719e-02f,2.755905502e-02f,3.149606287e-02f,3.543307260e-02f,7.874015719e-02f,7.874015719e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,1.023622081e-01f,1.181102395e-01f,1.417322904e-01f,5.511811003e-02f,7.086614519e-02f,7.874015719e-02f,5.905511975e-02f,5.511811003e-02f,7.086614519e-02f,5.905511975e-02f,9.990157187e-02f,9.448818862e-02f,7.564514875e-02f,7.363434881e-02f,7.552211732e-02f,6.014317647e-02f,7.202725112e-02f,9.072034806e-02f,9.343472868e-02f,1.037155539e-01f,4.330708832e-02f,4.330708832e-02f,3.937007859e-02f,4.330708832e-02f,9.448818862e-02f,4.724409431e-02f,3.149606287e-02f,3.149606287e-02f,7.874015719e-02f,7.874015719e-02f,7.874015719e-02f,7.086614519e-02f,7.874015719e-02f,1.023622081e-01f,1.181102395e-01f,1.417322904e-01f,5.905511975e-02f,6.299212575e-02f,1.102362201e-01f,1.417322904e-01f,1.102362201e-01f,1.181102395e-01f,5.511811003e-02f,1.023622081e-01f,8.661417663e-02f,7.078540325e-02f,7.076618075e-02f,7.564514875e-02f,6.019315869e-02f,7.156972587e-02f,9.116633981e-02f,9.142393619e-02f,1.158764437e-01f,2.362204716e-02f,2.165354416e-02f,2.559055202e-02f,2.559055202e-02f,6.299212575e-02f,5.511811003e-02f,5.905511975e-02f,6.299212575e-02f,7.874015719e-02f,7.086614519e-02f,7.874015719e-02f,5.905511975e-02f,7.086614519e-02f,9.448818862e-02f,1.023622081e-01f,1.181102395e-01f,1.181102395e-01f,9.448818862e-02f,5.118110403e-02f,3.543307260e-02f,6.299212575e-02f,2.755905502e-02f,4.330708832e-02f,5.511811003e-02f,7.086614519e-02f,7.161202282e-02f,6.984344125e-02f,7.677165419e-02f,5.945689231e-02f,6.975885481e-02f,9.381536394e-02f,1.028620228e-01f,1.396407485e-01f,1.377952751e-02f,1.574803144e-02f,6.299212575e-02f,1.968503930e-02f,2.362204716e-02f,2.165354416e-02f,6.299212575e-02f,1.574803144e-02f,7.086614519e-02f,7.086614519e-02f,7.874015719e-02f,6.299212575e-02f,7.086614519e-02f,9.448818862e-02f,1.023622081e-01f,1.181102395e-01f,3.543307260e-02f,3.937007859e-02f,1.181102395e-01f,4.330708832e-02f,3.937007859e-02f,3.543307260e-02f,3.543307260e-02f,6.299212575e-02f,7.086614519e-02f,6.214628369e-02f,6.932824850e-02f,7.197342813e-02f,5.911663547e-02f,6.730975956e-02f,8.579524606e-02f,1.182947829e-01f,1.405019611e-01f,4.330708832e-02f,3.543307260e-02f,5.905511975e-02f,4.724409431e-02f,5.118110403e-02f,1.181102358e-02f,4.330708832e-02f,5.118110403e-02f,7.086614519e-02f,7.086614519e-02f,7.874015719e-02f,5.905511975e-02f,7.086614519e-02f,9.448818862e-02f,1.023622081e-01f,1.181102395e-01f,5.118110403e-02f,3.543307260e-02f,5.118110403e-02f,5.118110403e-02f,3.149606287e-02f,3.937007859e-02f,3.149606287e-02f,6.299212575e-02f,6.299212575e-02f,7.416877151e-02f,7.416877151e-02f,7.779819518e-02f,1.267224401e-01f,7.152359188e-02f,9.348471463e-02f,9.095103294e-02f,1.012356952e-01f,4.724409431e-02f,9.448818862e-02f,2.559055202e-02f,3.937007859e-02f,2.362204716e-02f,1.574803144e-02f,1.574803144e-02f,4.724409431e-02f,7.086614519e-02f,7.086614519e-02f,7.874015719e-02f,5.905511975e-02f,7.086614519e-02f,9.448818862e-02f,1.023622081e-01f,1.102362201e-01f,5.905511975e-02f,5.511811003e-02f,5.905511975e-02f,4.330708832e-02f,4.330708832e-02f,1.102362201e-01f,1.259842515e-01f,5.905511975e-02f,6.299212575e-02f,7.874015719e-02f,1.259842515e-01f,9.448818862e-02f,1.259842515e-01f,7.874015719e-02f,9.448818862e-02f,9.448818862e-02f,9.448818862e-02f,7.874015719e-02f,1.259842515e-01f,1.023622081e-01f,1.023622081e-01f,1.259842515e-01f,1.259842515e-01f,7.086614519e-02f,1.181102395e-01f,7.454171032e-02f,1.360728294e-01f,9.178149700e-02f,1.022391766e-01f,7.230407000e-02f,9.183147550e-02f,1.017470509e-01f,8.735236526e-02f,8.378444612e-02f,1.015432775e-01f,1.174950823e-01f,8.645269275e-02f,1.265994161e-01f,1.011780277e-01f,7.378044724e-02f,1.266916841e-01f,4.330708832e-02f,7.086614519e-02f,5.511811003e-02f,6.299212575e-02f,6.299212575e-02f,7.874015719e-02f,4.724409431e-02f,3.149606287e-02f,3.543307260e-02f,4.724409431e-02f,3.937007859e-02f,7.874015719e-02f,6.299212575e-02f,3.543307260e-02f,3.543307260e-02f,5.118110403e-02f,7.086614519e-02f,1.417322904e-01f,7.874015719e-02f,8.661417663e-02f,6.299212575e-02f,7.627952844e-02f,7.874015719e-02f,9.448818862e-02f,8.661417663e-02f,7.874015719e-02f,1.259842515e-01f,9.448818862e-02f,7.874015719e-02f,7.086614519e-02f,7.086614519e-02f,1.259842515e-01f,6.566037238e-02f,1.325895041e-01f,7.995124906e-02f,8.027805388e-02f,7.308071107e-02f,7.743294537e-02f,7.513379306e-02f,7.884396613e-02f,8.083553612e-02f,7.468011975e-02f,9.608759731e-02f,7.925534993e-02f,7.135827094e-02f,6.920906156e-02f,7.012794912e-02f,1.031042412e-01f,3.149606287e-02f,7.874015719e-02f,3.937007859e-02f,4.330708832e-02f,3.543307260e-02f,7.874015719e-02f,3.149606287e-02f,3.543307260e-02f,4.724409431e-02f,3.149606287e-02f,3.937007859e-02f,5.118110403e-02f,3.543307260e-02f,5.905511975e-02f,3.149606287e-02f,3.543307260e-02f,7.086614519e-02f,1.417322904e-01f,7.086614519e-02f,7.086614519e-02f,6.299212575e-02f,7.874015719e-02f,7.086614519e-02f,8.661417663e-02f,7.874015719e-02f,7.086614519e-02f,8.661417663e-02f,8.661417663e-02f,6.840550900e-02f,5.905511975e-02f,7.086614519e-02f,1.023622081e-01f,8.169291168e-02f,1.017085984e-01f,7.560285181e-02f,7.504921407e-02f,8.464566618e-02f,7.361897081e-02f,6.705216318e-02f,7.144284993e-02f,7.068159431e-02f,8.218503743e-02f,9.514947981e-02f,9.251968563e-02f,6.766732037e-02f,6.063530222e-02f,8.591827750e-02f,1.036348119e-01f,1.259842515e-01f,3.937007859e-02f,7.874015719e-02f,5.511811003e-02f,3.543307260e-02f,3.937007859e-02f,9.448818862e-02f,4.330708832e-02f,3.543307260e-02f,4.724409431e-02f,4.724409431e-02f,3.543307260e-02f,3.543307260e-02f,4.724409431e-02f,5.511811003e-02f,3.149606287e-02f,8.661417663e-02f,1.259842515e-01f,7.086614519e-02f,7.086614519e-02f,7.874015719e-02f,9.448818862e-02f,6.299212575e-02f,8.661417663e-02f,9.448818862e-02f,6.299212575e-02f,8.661417663e-02f,1.023622081e-01f,7.086614519e-02f,7.874015719e-02f,9.448818862e-02f,1.077755913e-01f,9.725639969e-02f,1.066490859e-01f,6.945512444e-02f,7.031634450e-02f,7.989357412e-02f,6.441083550e-02f,7.747139782e-02f,7.890932262e-02f,7.197342813e-02f,6.617172062e-02f,9.461122006e-02f,8.468796313e-02f,6.543353200e-02f,6.902451068e-02f,8.460722119e-02f,1.046805829e-01f,4.330708832e-02f,3.937007859e-02f,3.937007859e-02f,6.299212575e-02f,5.511811003e-02f,1.259842515e-01f,6.299212575e-02f,7.086614519e-02f,3.543307260e-02f,5.118110403e-02f,6.299212575e-02f,5.905511975e-02f,5.118110403e-02f,4.724409431e-02f,7.086614519e-02f,3.543307260e-02f,1.259842515e-01f,1.259842515e-01f,6.299212575e-02f,7.086614519e-02f,7.874015719e-02f,9.448818862e-02f,7.874015719e-02f,7.086614519e-02f,7.874015719e-02f,7.086614519e-02f,9.448818862e-02f,1.181102395e-01f,9.202755988e-02f,8.661417663e-02f,1.259842515e-01f,1.259842515e-01f,1.065721884e-01f,1.052380651e-01f,6.282488257e-02f,7.402651012e-02f,7.640255988e-02f,6.914369762e-02f,8.221194893e-02f,6.647930294e-02f,7.487619668e-02f,7.591043413e-02f,7.627952844e-02f,6.635627151e-02f,6.480299681e-02f,6.770961732e-02f,7.525683194e-02f,9.608759731e-02f,3.543307260e-02f,6.299212575e-02f,7.086614519e-02f,6.299212575e-02f,4.724409431e-02f,4.330708832e-02f,5.118110403e-02f,5.905511975e-02f,8.661417663e-02f,3.543307260e-02f,3.937007859e-02f,6.299212575e-02f,4.724409431e-02f,8.661417663e-02f,4.724409431e-02f,1.102362201e-01f,1.259842515e-01f,1.417322904e-01f,6.299212575e-02f,1.102362201e-01f,7.874015719e-02f,9.448818862e-02f,7.874015719e-02f,7.874015719e-02f,9.448818862e-02f,7.874015719e-02f,7.086614519e-02f,1.181102395e-01f,1.181102395e-01f,7.874015719e-02f,9.448818862e-02f,8.661417663e-02f,7.587198913e-02f,8.297705650e-02f,6.234620884e-02f,7.092765719e-02f,6.494140625e-02f,6.643700600e-02f,6.820558757e-02f,6.623708457e-02f,7.767516375e-02f,6.586414576e-02f,6.243656203e-02f,7.820958644e-02f,7.636411488e-02f,6.140040606e-02f,7.738681138e-02f,6.674843282e-02f,6.299212575e-02f,4.330708832e-02f,5.511811003e-02f,6.299212575e-02f,4.330708832e-02f,4.330708832e-02f,6.299212575e-02f,4.330708832e-02f,6.299212575e-02f,3.937007859e-02f,4.724409431e-02f,5.511811003e-02f,6.299212575e-02f,4.724409431e-02f,4.330708832e-02f,4.724409431e-02f,6.299212575e-02f,8.661417663e-02f,6.299212575e-02f,6.299212575e-02f,5.905511975e-02f,7.874015719e-02f,6.299212575e-02f,9.448818862e-02f,1.417322904e-01f,6.299212575e-02f,5.511811003e-02f,1.417322904e-01f,1.574803144e-01f,6.299212575e-02f,7.086614519e-02f,7.874015719e-02f,6.102362275e-02f,5.610236153e-02f,5.752491206e-02f,5.263825506e-02f,5.400313810e-02f,4.992002994e-02f,5.321304500e-02f,5.588898063e-02f,7.424950600e-02f,5.223071575e-02f,5.583707616e-02f,7.649867982e-02f,8.214659244e-02f,5.133489147e-02f,6.584491581e-02f,5.507773906e-02f,2.559055202e-02f,5.118110403e-02f,3.937007859e-02f,3.543307260e-02f,3.937007859e-02f,3.149606287e-02f,3.937007859e-02f,3.543307260e-02f,4.330708832e-02f,3.149606287e-02f,3.937007859e-02f,6.299212575e-02f,3.543307260e-02f,2.891240083e-02f,3.149606287e-02f,3.937007859e-02f,7.086614519e-02f,5.905511975e-02f,6.299212575e-02f,5.905511975e-02f,6.299212575e-02f,6.299212575e-02f,6.299212575e-02f,5.118110403e-02f,1.023622081e-01f,6.299212575e-02f,6.299212575e-02f,7.086614519e-02f,1.417322904e-01f,9.448818862e-02f,7.086614519e-02f,5.905511975e-02f,4.624062032e-02f,4.506028444e-02f,4.617910460e-02f,4.476616159e-02f,5.092542619e-02f,4.611758888e-02f,4.627329856e-02f,4.636365175e-02f,7.078540325e-02f,4.945866019e-02f,4.647706822e-02f,4.207677022e-02f,5.247485638e-02f,4.836291075e-02f,4.995270818e-02f,4.948941991e-02f,3.937007859e-02f,4.330708832e-02f,2.559055202e-02f,3.149606287e-02f,3.149606287e-02f,3.149606287e-02f,3.937007859e-02f,3.149606287e-02f,1.968503930e-02f,2.559055202e-02f,2.952755988e-02f,3.149606287e-02f,1.968503930e-02f,2.362204716e-02f,3.937007859e-02f,2.165354416e-02f,4.330708832e-02f,3.937007859e-02f,3.937007859e-02f,3.937007859e-02f,3.937007859e-02f,4.724409431e-02f,4.330708832e-02f,3.937007859e-02f,4.330708832e-02f,6.299212575e-02f,6.299212575e-02f,4.330708832e-02f,3.937007859e-02f,6.299212575e-02f,4.330708832e-02f,6.299212575e-02f,6.328047812e-02f,7.347287238e-02f,7.860174775e-02f,7.886703312e-02f,6.317667663e-02f,6.187523156e-02f,6.420321763e-02f,7.721764594e-02f,5.905511975e-02f,8.661417663e-02f,5.905511975e-02f,8.661417663e-02f,4.330708832e-02f,5.905511975e-02f,5.782480165e-02f,3.543307260e-02f,6.299212575e-02f,7.874015719e-02f,7.874015719e-02f,7.874015719e-02f,6.299212575e-02f,6.299212575e-02f,7.086614519e-02f,7.874015719e-02f,4.330708832e-02f,3.543307260e-02f,6.299212575e-02f,4.724409431e-02f,5.511811003e-02f,6.299212575e-02f,5.118110403e-02f,4.724409431e-02f,6.299212575e-02f,6.900144368e-02f,7.571434975e-02f,9.285033494e-02f,8.136611432e-02f,1.067298204e-01f,6.527205557e-02f,6.779804081e-02f,8.028958738e-02f,3.937007859e-02f,3.937007859e-02f,5.782480165e-02f,6.299212575e-02f,5.511811003e-02f,5.118110403e-02f,5.905511975e-02f,3.937007859e-02f,7.086614519e-02f,7.874015719e-02f,1.259842515e-01f,8.661417663e-02f,1.259842515e-01f,6.299212575e-02f,7.086614519e-02f,8.661417663e-02f,6.299212575e-02f,1.259842515e-01f,4.330708832e-02f,5.511811003e-02f,3.937007859e-02f,4.724409431e-02f,3.543307260e-02f,5.511811003e-02f,7.874015719e-02f,6.251153350e-02f,7.098917663e-02f,7.779819518e-02f,7.837106287e-02f,6.418399513e-02f,6.177911162e-02f,6.371109188e-02f,6.949356943e-02f,4.724409431e-02f,1.968503930e-02f,2.362204716e-02f,1.968503930e-02f,2.165354416e-02f,1.968503930e-02f,4.330708832e-02f,2.165354416e-02f,6.299212575e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,7.086614519e-02f,6.299212575e-02f,7.086614519e-02f,7.086614519e-02f,3.543307260e-02f,3.149606287e-02f,4.330708832e-02f,5.118110403e-02f,9.448818862e-02f,5.118110403e-02f,3.543307260e-02f,4.330708832e-02f,7.874015719e-02f,6.319974363e-02f,7.511072606e-02f,7.888625562e-02f,7.861712575e-02f,6.360728294e-02f,6.222702563e-02f,6.503368169e-02f,6.834399700e-02f,4.724409431e-02f,1.968503930e-02f,2.165354416e-02f,2.362204716e-02f,2.165354416e-02f,2.165354416e-02f,1.968503930e-02f,2.165354416e-02f,6.299212575e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,7.086614519e-02f,6.299212575e-02f,7.086614519e-02f,7.086614519e-02f,3.937007859e-02f,3.937007859e-02f,4.724409431e-02f,5.118110403e-02f,5.118110403e-02f,4.330708832e-02f,3.149606287e-02f,4.724409431e-02f,7.086614519e-02f,6.278066337e-02f,7.685624063e-02f,7.964366674e-02f,7.910925150e-02f,6.216166168e-02f,6.146961078e-02f,6.340350956e-02f,6.619863212e-02f,2.559055202e-02f,2.165354416e-02f,2.165354416e-02f,3.937007859e-02f,3.937007859e-02f,1.968503930e-02f,2.362204716e-02f,4.724409431e-02f,6.299212575e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,6.299212575e-02f,6.299212575e-02f,7.086614519e-02f,7.086614519e-02f,3.543307260e-02f,5.118110403e-02f,6.299212575e-02f,4.330708832e-02f,4.330708832e-02f,6.176181138e-02f,3.937007859e-02f,5.511811003e-02f,6.299212575e-02f,6.306133419e-02f,7.232329249e-02f,7.987050712e-02f,7.841335237e-02f,5.914931372e-02f,6.003937125e-02f,6.013164297e-02f,6.251153350e-02f,3.149606287e-02f,6.299212575e-02f,4.330708832e-02f,6.299212575e-02f,7.086614519e-02f,7.874015719e-02f,5.118110403e-02f,3.937007859e-02f,6.299212575e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,6.299212575e-02f,6.299212575e-02f,6.840550900e-02f,7.086614519e-02f,4.995078593e-02f,3.937007859e-02f,3.149606287e-02f,5.118110403e-02f,3.937007859e-02f,4.330708832e-02f,7.874015719e-02f,3.543307260e-02f,6.840550900e-02f,5.832846463e-02f,7.086614519e-02f,7.843258232e-02f,7.837106287e-02f,5.854376778e-02f,5.903589353e-02f,6.155804172e-02f,6.007781625e-02f,2.165354416e-02f,2.165354416e-02f,6.299212575e-02f,3.543307260e-02f,2.559055202e-02f,2.362204716e-02f,2.165354416e-02f,1.968503930e-02f,6.299212575e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,6.299212575e-02f,6.299212575e-02f,6.299212575e-02f,7.086614519e-02f,3.149606287e-02f,3.149606287e-02f,3.937007859e-02f,4.330708832e-02f,3.543307260e-02f,3.543307260e-02f,5.905511975e-02f,4.330708832e-02f,7.086614519e-02f,5.902436003e-02f,7.127752900e-02f,7.890547812e-02f,7.867864519e-02f,5.916853622e-02f,6.078140438e-02f,6.290177256e-02f,6.180602685e-02f,2.559055202e-02f,2.755905502e-02f,2.362204716e-02f,2.952755988e-02f,3.543307260e-02f,2.755905502e-02f,3.543307260e-02f,2.755905502e-02f,5.905511975e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,5.905511975e-02f,6.299212575e-02f,6.299212575e-02f,6.299212575e-02f,3.543307260e-02f,3.937007859e-02f,2.559055202e-02f,5.511811003e-02f,3.543307260e-02f,3.543307260e-02f,3.149606287e-02f,4.330708832e-02f,5.905511975e-02f,1.889763772e-01f,9.448818862e-02f,1.259842515e-01f,1.259842515e-01f,7.086614519e-02f,1.259842515e-01f,6.299212575e-02f,8.661417663e-02f,5.905511975e-02f,7.874015719e-02f,8.661417663e-02f,7.874015719e-02f,5.905511975e-02f,6.176181138e-02f,6.299212575e-02f,6.299212575e-02f,2.204724401e-01f,9.448818862e-02f,1.259842515e-01f,1.417322904e-01f,7.086614519e-02f,7.874015719e-02f,1.259842515e-01f,9.448818862e-02f,1.574803144e-01f,1.259842515e-01f,1.181102395e-01f,1.259842515e-01f,7.874015719e-02f,7.086614519e-02f,1.181102395e-01f,7.086614519e-02f,1.732283533e-01f,9.448818862e-02f,1.259842515e-01f,1.259842515e-01f,8.661417663e-02f,7.086614519e-02f,7.874015719e-02f,7.086614519e-02f,1.732283533e-01f,1.023622081e-01f,1.181102395e-01f,1.181102395e-01f,7.086614519e-02f,7.874015719e-02f,7.086614519e-02f,1.181102395e-01f,1.732283533e-01f,1.023622081e-01f,1.259842515e-01f,1.259842515e-01f,6.299212575e-02f,7.086614519e-02f,6.299212575e-02f,8.661417663e-02f,2.519685030e-01f,1.102362201e-01f,1.259842515e-01f,1.235236228e-01f,1.417322904e-01f,8.661417663e-02f,7.874015719e-02f,1.259842515e-01f,1.732283533e-01f,1.102362201e-01f,1.259842515e-01f,1.259842515e-01f,5.511811003e-02f,6.299212575e-02f,7.086614519e-02f,6.299212575e-02f,1.574803144e-01f,9.448818862e-02f,1.181102395e-01f,1.259842515e-01f,7.086614519e-02f,5.905511975e-02f,6.299212575e-02f,7.874015719e-02f,1.732283533e-01f,1.259842515e-01f,1.259842515e-01f,1.259842515e-01f,6.299212575e-02f,7.086614519e-02f,6.299212575e-02f,1.102362201e-01f,1.417322904e-01f,9.448818862e-02f,1.102362201e-01f,1.181102395e-01f,6.299212575e-02f,7.627952844e-02f,6.299212575e-02f,7.086614519e-02f,1.574803144e-01f,9.448818862e-02f,1.235236228e-01f,1.259842515e-01f,7.086614519e-02f,7.874015719e-02f,6.299212575e-02f,8.661417663e-02f,1.417322904e-01f,9.448818862e-02f,1.181102395e-01f,1.156496033e-01f,7.086614519e-02f,8.661417663e-02f,6.299212575e-02f,1.023622081e-01f,1.417322904e-01f,9.448818862e-02f,1.181102395e-01f,1.259842515e-01f,7.874015719e-02f,7.086614519e-02f,6.299212575e-02f,7.874015719e-02f,1.417322904e-01f,1.181102395e-01f,1.259842515e-01f,1.417322904e-01f,1.259842515e-01f,1.181102395e-01f,9.448818862e-02f,1.102362201e-01f,1.417322904e-01f,1.259842515e-01f,1.181102395e-01f,1.259842515e-01f,1.417322904e-01f,8.661417663e-02f,1.023622081e-01f,9.448818862e-02f,2.834645808e-01f,2.834645808e-01f,1.889763772e-01f,2.047244161e-01f,2.834645808e-01f,2.834645808e-01f,2.047244161e-01f,1.732283533e-01f,2.834645808e-01f,3.149606287e-01f,2.519685030e-01f,2.155511826e-01f,2.834645808e-01f,2.834645808e-01f,1.889763772e-01f,1.732283533e-01f,5.039370060e-01f,3.149606287e-01f,2.204724401e-01f,2.047244161e-01f,2.834645808e-01f,3.149606287e-01f,3.779527545e-01f,1.732283533e-01f,2.834645808e-01f,3.149606287e-01f,2.519685030e-01f,1.889763772e-01f,3.149606287e-01f,3.149606287e-01f,2.519685030e-01f,2.047244161e-01f,3.779527545e-01f,1.007874012e+00f,2.362204790e-01f,2.204724401e-01f,3.464567065e-01f,3.464567065e-01f,2.362204790e-01f,2.519685030e-01f,3.779527545e-01f,5.039370060e-01f,3.464567065e-01f,3.149606287e-01f,3.779527545e-01f,4.724409580e-01f,2.519685030e-01f,2.834645808e-01f,1.637795329e+00f,6.929134130e-01f,8.818897605e-01f,3.149606287e-01f,9.448819160e-01f,7.559055090e-01f,9.448819160e-01f,5.669291615e-01f,1.007874012e+00f,1.637795329e+00f,1.259842515e+00f,5.039370060e-01f,1.094488144e+00f,1.133858323e+00f,1.007874012e+00f,3.464567065e-01f};
__device__ __constant__ float research_large_inverse[1320]={2.886363745e+00f,1.984375000e+00f,2.886363745e+00f,4.233333111e+00f,2.161702156e+00f,3.527777672e+00f,1.587499976e+00f,3.527777672e+00f,1.984375000e+00f,5.772727489e+00f,2.442307711e+00f,3.527777672e+00f,2.267857075e+00f,4.233333111e+00f,2.645833254e+00f,2.645833254e+00f,2.886363745e+00f,3.527777672e+00f,2.116666555e+00f,2.645833254e+00f,2.645833254e+00f,2.886363745e+00f,4.233333111e+00f,3.968750000e+00f,3.174999952e+00f,3.527777672e+00f,2.886363745e+00f,2.886363745e+00f,3.174999952e+00f,2.716577530e+00f,4.884615421e+00f,6.349999905e+00f,3.968750000e+00f,2.645833254e+00f,3.968750000e+00f,5.291666508e+00f,3.174999952e+00f,3.174999952e+00f,4.233333111e+00f,3.968750000e+00f,3.968750000e+00f,3.174999952e+00f,4.884615421e+00f,5.772727489e+00f,3.527777672e+00f,2.886363745e+00f,6.349999905e+00f,6.349999905e+00f,3.968750000e+00f,3.277419329e+00f,7.937500000e+00f,8.466666222e+00f,3.527777672e+00f,3.527777672e+00f,5.772727489e+00f,7.055555344e+00f,4.535714149e+00f,3.968750000e+00f,7.055555344e+00f,6.349999905e+00f,3.968750000e+00f,3.527777672e+00f,7.055555344e+00f,6.349999905e+00f,3.174999952e+00f,3.968750000e+00f,4.233333111e+00f,3.527777672e+00f,5.291666508e+00f,5.004926205e+00f,5.772727489e+00f,3.968750000e+00f,2.886363745e+00f,3.968750000e+00f,3.968750000e+00f,3.527777672e+00f,5.291666508e+00f,3.968750000e+00f,7.055555344e+00f,5.772727489e+00f,2.645833254e+00f,3.968750000e+00f,3.968750000e+00f,3.968750000e+00f,7.055555344e+00f,5.772727489e+00f,3.968750000e+00f,6.349999905e+00f,3.174999952e+00f,4.884615421e+00f,3.968750000e+00f,4.535714149e+00f,4.884615421e+00f,8.466666222e+00f,6.349999905e+00f,7.937500000e+00f,2.645833254e+00f,4.233333111e+00f,3.527777672e+00f,4.233333111e+00f,3.968750000e+00f,5.291666508e+00f,5.772727489e+00f,4.233333111e+00f,3.174999952e+00f,4.233333111e+00f,4.233333111e+00f,4.535714149e+00f,8.466666222e+00f,3.968750000e+00f,4.233333111e+00f,3.174999952e+00f,2.886363745e+00f,3.527777672e+00f,3.527777672e+00f,4.884615421e+00f,7.055555344e+00f,6.349999905e+00f,5.772727489e+00f,6.349999905e+00f,3.174999952e+00f,2.886363745e+00f,2.645833254e+00f,3.527777672e+00f,7.937500000e+00f,3.174999952e+00f,7.055555344e+00f,6.349999905e+00f,3.174999952e+00f,4.884615421e+00f,3.527777672e+00f,4.884615421e+00f,9.769230843e+00f,7.937500000e+00f,7.937500000e+00f,8.466666222e+00f,3.174999952e+00f,3.968750000e+00f,3.527777672e+00f,3.968750000e+00f,7.309352398e+00f,8.466666222e+00f,5.291666508e+00f,6.349999905e+00f,2.645833254e+00f,5.772727489e+00f,4.323404312e+00f,3.968750000e+00f,4.535714149e+00f,1.058333302e+01f,8.466666222e+00f,8.466666222e+00f,3.174999952e+00f,5.941520691e+00f,4.233333111e+00f,3.968750000e+00f,8.466666222e+00f,9.071428299e+00f,7.309352398e+00f,9.071428299e+00f,2.645833254e+00f,6.349999905e+00f,4.639269352e+00f,5.291666508e+00f,3.968750000e+00f,9.071428299e+00f,1.154545498e+01f,7.055555344e+00f,3.174999952e+00f,6.349999905e+00f,4.535714149e+00f,3.968750000e+00f,9.071428299e+00f,7.937500000e+00f,9.071428299e+00f,9.071428299e+00f,2.886363745e+00f,4.884615421e+00f,3.527777672e+00f,4.323404312e+00f,9.071428299e+00f,8.466666222e+00f,9.071428299e+00f,7.055555344e+00f,3.174999952e+00f,6.349999905e+00f,2.886363745e+00f,3.968750000e+00f,6.349999905e+00f,5.772727489e+00f,6.349999905e+00f,4.233333111e+00f,9.440185547e+00f,6.473592281e+00f,9.685559273e+00f,1.036073971e+01f,6.397166729e+00f,7.617619514e+00f,7.263628006e+00f,6.410094738e+00f,5.079999924e+01f,6.773332977e+01f,7.815384674e+01f,6.350000000e+01f,5.644444275e+01f,7.257142639e+01f,6.773332977e+01f,6.350000000e+01f,7.055555344e+00f,3.968750000e+00f,6.349999905e+00f,7.055555344e+00f,5.772727489e+00f,5.291666508e+00f,4.535714149e+00f,4.884615421e+00f,2.130706432e+09f,5.644444275e+01f,6.350000000e+01f,1.015999985e+02f,8.466666222e+00f,9.031110840e+02f,2.130706432e+09f,7.257142639e+01f,3.968750000e+00f,1.240916061e+01f,5.772727489e+00f,1.275981140e+01f,1.294267559e+01f,9.168641090e+00f,8.136136055e+00f,8.711682320e+00f,8.753904343e+00f,1.015999985e+02f,1.015999985e+02f,7.815384674e+01f,8.466666412e+01f,1.128888855e+02f,9.236363983e+01f,9.236363983e+01f,9.236363983e+01f,7.937500000e+00f,1.058333302e+01f,1.058333302e+01f,1.058333302e+01f,9.071428299e+00f,7.055555344e+00f,7.055555344e+00f,6.349999905e+00f,3.907692337e+01f,3.175000000e+01f,8.466666412e+01f,8.466666412e+01f,6.350000000e+01f,8.466666412e+01f,9.236363983e+01f,6.350000000e+01f,7.055555344e+00f,1.412720680e+01f,1.354666710e+01f,1.273295116e+01f,1.490436077e+01f,1.391780853e+01f,1.028982830e+01f,8.605611801e+00f,7.043327332e+00f,1.953846169e+01f,1.814285660e+01f,3.175000000e+01f,4.618181992e+01f,1.269999981e+01f,3.628571320e+01f,3.175000000e+01f,2.822222137e+01f,1.269999981e+01f,1.269999981e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,9.769230843e+00f,8.466666222e+00f,7.055555344e+00f,1.814285660e+01f,1.411111069e+01f,1.269999981e+01f,1.693333244e+01f,1.814285660e+01f,1.411111069e+01f,1.693333244e+01f,1.000985241e+01f,1.058333302e+01f,1.321961880e+01f,1.358061790e+01f,1.324115467e+01f,1.662698936e+01f,1.388363361e+01f,1.102288532e+01f,1.070265865e+01f,9.641756058e+00f,2.309090996e+01f,2.309090996e+01f,2.539999962e+01f,2.309090996e+01f,1.058333302e+01f,2.116666603e+01f,3.175000000e+01f,3.175000000e+01f,1.269999981e+01f,1.269999981e+01f,1.269999981e+01f,1.411111069e+01f,1.269999981e+01f,9.769230843e+00f,8.466666222e+00f,7.055555344e+00f,1.693333244e+01f,1.587500000e+01f,9.071428299e+00f,7.055555344e+00f,9.071428299e+00f,8.466666222e+00f,1.814285660e+01f,9.769230843e+00f,1.154545498e+01f,1.412720680e+01f,1.413104439e+01f,1.321961880e+01f,1.661318398e+01f,1.397238827e+01f,1.096896076e+01f,1.093805504e+01f,8.629881859e+00f,4.233333206e+01f,4.618181992e+01f,3.907692337e+01f,3.907692337e+01f,1.587500000e+01f,1.814285660e+01f,1.693333244e+01f,1.587500000e+01f,1.269999981e+01f,1.411111069e+01f,1.269999981e+01f,1.693333244e+01f,1.411111069e+01f,1.058333302e+01f,9.769230843e+00f,8.466666222e+00f,8.466666222e+00f,1.058333302e+01f,1.953846169e+01f,2.822222137e+01f,1.587500000e+01f,3.628571320e+01f,2.309090996e+01f,1.814285660e+01f,1.411111069e+01f,1.396413612e+01f,1.431773663e+01f,1.302564144e+01f,1.681890869e+01f,1.433509731e+01f,1.065923500e+01f,9.721761703e+00f,7.161233425e+00f,7.257142639e+01f,6.350000000e+01f,1.587500000e+01f,5.079999924e+01f,4.233333206e+01f,4.618181992e+01f,1.587500000e+01f,6.350000000e+01f,1.411111069e+01f,1.411111069e+01f,1.269999981e+01f,1.587500000e+01f,1.411111069e+01f,1.058333302e+01f,9.769230843e+00f,8.466666222e+00f,2.822222137e+01f,2.539999962e+01f,8.466666222e+00f,2.309090996e+01f,2.539999962e+01f,2.822222137e+01f,2.822222137e+01f,1.587500000e+01f,1.411111069e+01f,1.609106636e+01f,1.442413521e+01f,1.389401722e+01f,1.691571236e+01f,1.485668564e+01f,1.165565777e+01f,8.453457832e+00f,7.117338181e+00f,2.309090996e+01f,2.822222137e+01f,1.693333244e+01f,2.116666603e+01f,1.953846169e+01f,8.466666412e+01f,2.309090996e+01f,1.953846169e+01f,1.411111069e+01f,1.411111069e+01f,1.269999981e+01f,1.693333244e+01f,1.411111069e+01f,1.058333302e+01f,9.769230843e+00f,8.466666222e+00f,1.953846169e+01f,2.822222137e+01f,1.953846169e+01f,1.953846169e+01f,3.175000000e+01f,2.539999962e+01f,3.175000000e+01f,1.587500000e+01f,1.587500000e+01f,1.348276424e+01f,1.348276424e+01f,1.285376835e+01f,7.891262054e+00f,1.398140049e+01f,1.069693565e+01f,1.099492741e+01f,9.877938271e+00f,2.116666603e+01f,1.058333302e+01f,3.907692337e+01f,2.539999962e+01f,4.233333206e+01f,6.350000000e+01f,6.350000000e+01f,2.116666603e+01f,1.411111069e+01f,1.411111069e+01f,1.269999981e+01f,1.693333244e+01f,1.411111069e+01f,1.058333302e+01f,9.769230843e+00f,9.071428299e+00f,1.693333244e+01f,1.814285660e+01f,1.693333244e+01f,2.309090996e+01f,2.309090996e+01f,9.071428299e+00f,7.937500000e+00f,1.693333244e+01f,1.587500000e+01f,1.269999981e+01f,7.937500000e+00f,1.058333302e+01f,7.937500000e+00f,1.269999981e+01f,1.058333302e+01f,1.058333302e+01f,1.058333302e+01f,1.269999981e+01f,7.937500000e+00f,9.769230843e+00f,9.769230843e+00f,7.937500000e+00f,7.937500000e+00f,1.411111069e+01f,8.466666222e+00f,1.341530800e+01f,7.349005222e+00f,1.089544201e+01f,9.780986786e+00f,1.383047962e+01f,1.088951206e+01f,9.828294754e+00f,1.144788742e+01f,1.193538952e+01f,9.848017693e+00f,8.510994911e+00f,1.156701946e+01f,7.898931026e+00f,9.883568764e+00f,1.355372620e+01f,7.893177986e+00f,2.309090996e+01f,1.411111069e+01f,1.814285660e+01f,1.587500000e+01f,1.587500000e+01f,1.269999981e+01f,2.116666603e+01f,3.175000000e+01f,2.822222137e+01f,2.116666603e+01f,2.539999962e+01f,1.269999981e+01f,1.587500000e+01f,2.822222137e+01f,2.822222137e+01f,1.953846169e+01f,1.411111069e+01f,7.055555344e+00f,1.269999981e+01f,1.154545498e+01f,1.587500000e+01f,1.310967731e+01f,1.269999981e+01f,1.058333302e+01f,1.154545498e+01f,1.269999981e+01f,7.937500000e+00f,1.058333302e+01f,1.269999981e+01f,1.411111069e+01f,1.411111069e+01f,7.937500000e+00f,1.522988605e+01f,7.542075157e+00f,1.250762177e+01f,1.245670509e+01f,1.368350124e+01f,1.291439915e+01f,1.330958939e+01f,1.268327904e+01f,1.237079620e+01f,1.339044476e+01f,1.040717030e+01f,1.261744404e+01f,1.401379299e+01f,1.444897461e+01f,1.425964928e+01f,9.698922157e+00f,3.175000000e+01f,1.269999981e+01f,2.539999962e+01f,2.309090996e+01f,2.822222137e+01f,1.269999981e+01f,3.175000000e+01f,2.822222137e+01f,2.116666603e+01f,3.175000000e+01f,2.539999962e+01f,1.953846169e+01f,2.822222137e+01f,1.693333244e+01f,3.175000000e+01f,2.822222137e+01f,1.411111069e+01f,7.055555344e+00f,1.411111069e+01f,1.411111069e+01f,1.587500000e+01f,1.269999981e+01f,1.411111069e+01f,1.154545498e+01f,1.269999981e+01f,1.411111069e+01f,1.154545498e+01f,1.154545498e+01f,1.461870480e+01f,1.693333244e+01f,1.411111069e+01f,9.769230843e+00f,1.224096394e+01f,9.832010269e+00f,1.322701359e+01f,1.332458973e+01f,1.181395340e+01f,1.358345509e+01f,1.491376114e+01f,1.399720192e+01f,1.414795494e+01f,1.216766453e+01f,1.050977898e+01f,1.080851078e+01f,1.477818203e+01f,1.649204254e+01f,1.163896751e+01f,9.649267197e+00f,7.937500000e+00f,2.539999962e+01f,1.269999981e+01f,1.814285660e+01f,2.822222137e+01f,2.539999962e+01f,1.058333302e+01f,2.309090996e+01f,2.822222137e+01f,2.116666603e+01f,2.116666603e+01f,2.822222137e+01f,2.822222137e+01f,2.116666603e+01f,1.814285660e+01f,3.175000000e+01f,1.154545498e+01f,7.937500000e+00f,1.411111069e+01f,1.411111069e+01f,1.269999981e+01f,1.058333302e+01f,1.587500000e+01f,1.154545498e+01f,1.058333302e+01f,1.587500000e+01f,1.154545498e+01f,9.769230843e+00f,1.411111069e+01f,1.269999981e+01f,1.058333302e+01f,9.278538704e+00f,1.028209972e+01f,9.376545906e+00f,1.439778614e+01f,1.422144413e+01f,1.251665020e+01f,1.552533913e+01f,1.290799046e+01f,1.267277336e+01f,1.389401722e+01f,1.511219597e+01f,1.056957054e+01f,1.180805397e+01f,1.528268433e+01f,1.448760700e+01f,1.181932163e+01f,9.552870750e+00f,2.309090996e+01f,2.539999962e+01f,2.539999962e+01f,1.587500000e+01f,1.814285660e+01f,7.937500000e+00f,1.587500000e+01f,1.411111069e+01f,2.822222137e+01f,1.953846169e+01f,1.587500000e+01f,1.693333244e+01f,1.953846169e+01f,2.116666603e+01f,1.411111069e+01f,2.822222137e+01f,7.937500000e+00f,7.937500000e+00f,1.587500000e+01f,1.411111069e+01f,1.269999981e+01f,1.058333302e+01f,1.269999981e+01f,1.411111069e+01f,1.269999981e+01f,1.411111069e+01f,1.058333302e+01f,8.466666222e+00f,1.086631012e+01f,1.154545498e+01f,7.937500000e+00f,7.937500000e+00f,9.383311272e+00f,9.502264977e+00f,1.591726112e+01f,1.350867367e+01f,1.308856678e+01f,1.446263313e+01f,1.216368103e+01f,1.504227638e+01f,1.335537910e+01f,1.317341995e+01f,1.310967731e+01f,1.507016659e+01f,1.543138504e+01f,1.476895142e+01f,1.328783035e+01f,1.040717030e+01f,2.822222137e+01f,1.587500000e+01f,1.411111069e+01f,1.587500000e+01f,2.116666603e+01f,2.309090996e+01f,1.953846169e+01f,1.693333244e+01f,1.154545498e+01f,2.822222137e+01f,2.539999962e+01f,1.587500000e+01f,2.116666603e+01f,1.154545498e+01f,2.116666603e+01f,9.071428299e+00f,7.937500000e+00f,7.055555344e+00f,1.587500000e+01f,9.071428299e+00f,1.269999981e+01f,1.058333302e+01f,1.269999981e+01f,1.269999981e+01f,1.058333302e+01f,1.269999981e+01f,1.411111069e+01f,8.466666222e+00f,8.466666222e+00f,1.269999981e+01f,1.058333302e+01f,1.154545498e+01f,1.318009567e+01f,1.205152416e+01f,1.603946686e+01f,1.409887218e+01f,1.539849663e+01f,1.505185223e+01f,1.466155624e+01f,1.509728336e+01f,1.287412739e+01f,1.518276787e+01f,1.601625633e+01f,1.278615665e+01f,1.309515667e+01f,1.628653717e+01f,1.292209816e+01f,1.498162556e+01f,1.587500000e+01f,2.309090996e+01f,1.814285660e+01f,1.587500000e+01f,2.309090996e+01f,2.309090996e+01f,1.587500000e+01f,2.309090996e+01f,1.587500000e+01f,2.539999962e+01f,2.116666603e+01f,1.814285660e+01f,1.587500000e+01f,2.116666603e+01f,2.309090996e+01f,2.116666603e+01f,1.587500000e+01f,1.154545498e+01f,1.587500000e+01f,1.587500000e+01f,1.693333244e+01f,1.269999981e+01f,1.587500000e+01f,1.058333302e+01f,7.055555344e+00f,1.587500000e+01f,1.814285660e+01f,7.055555344e+00f,6.349999905e+00f,1.587500000e+01f,1.411111069e+01f,1.269999981e+01f,1.638709641e+01f,1.782456207e+01f,1.738377190e+01f,1.899758911e+01f,1.851744270e+01f,2.003203964e+01f,1.879238510e+01f,1.789261436e+01f,1.346810246e+01f,1.914582253e+01f,1.790924835e+01f,1.307212162e+01f,1.217335987e+01f,1.947992897e+01f,1.518720055e+01f,1.815615463e+01f,3.907692337e+01f,1.953846169e+01f,2.539999962e+01f,2.822222137e+01f,2.539999962e+01f,3.175000000e+01f,2.539999962e+01f,2.822222137e+01f,2.309090996e+01f,3.175000000e+01f,2.539999962e+01f,1.587500000e+01f,2.822222137e+01f,3.458723450e+01f,3.175000000e+01f,2.539999962e+01f,1.411111069e+01f,1.693333244e+01f,1.587500000e+01f,1.693333244e+01f,1.587500000e+01f,1.587500000e+01f,1.587500000e+01f,1.953846169e+01f,9.769230843e+00f,1.587500000e+01f,1.587500000e+01f,1.411111069e+01f,7.055555344e+00f,1.058333302e+01f,1.411111069e+01f,1.693333244e+01f,2.162600899e+01f,2.219249153e+01f,2.165481567e+01f,2.233830070e+01f,1.963655663e+01f,2.168370247e+01f,2.161073494e+01f,2.156862068e+01f,1.412720680e+01f,2.021890640e+01f,2.151598549e+01f,2.376608276e+01f,1.905674553e+01f,2.067700195e+01f,2.001893425e+01f,2.020633888e+01f,2.539999962e+01f,2.309090996e+01f,3.907692337e+01f,3.175000000e+01f,3.175000000e+01f,3.175000000e+01f,2.539999962e+01f,3.175000000e+01f,5.079999924e+01f,3.907692337e+01f,3.386666489e+01f,3.175000000e+01f,5.079999924e+01f,4.233333206e+01f,2.539999962e+01f,4.618181992e+01f,2.309090996e+01f,2.539999962e+01f,2.539999962e+01f,2.539999962e+01f,2.539999962e+01f,2.116666603e+01f,2.309090996e+01f,2.539999962e+01f,2.309090996e+01f,1.587500000e+01f,1.587500000e+01f,2.309090996e+01f,2.539999962e+01f,1.587500000e+01f,2.309090996e+01f,1.587500000e+01f,1.580266094e+01f,1.361046600e+01f,1.272236347e+01f,1.267956924e+01f,1.582862663e+01f,1.616155624e+01f,1.557554340e+01f,1.295040798e+01f,1.693333244e+01f,1.154545498e+01f,1.693333244e+01f,1.154545498e+01f,2.309090996e+01f,1.693333244e+01f,1.729361725e+01f,2.822222137e+01f,1.587500000e+01f,1.269999981e+01f,1.269999981e+01f,1.269999981e+01f,1.587500000e+01f,1.587500000e+01f,1.411111069e+01f,1.269999981e+01f,2.309090996e+01f,2.822222137e+01f,1.587500000e+01f,2.116666603e+01f,1.814285660e+01f,1.587500000e+01f,1.953846169e+01f,2.116666603e+01f,1.587500000e+01f,1.449244976e+01f,1.320753574e+01f,1.077002048e+01f,1.229012871e+01f,9.369452477e+00f,1.532049274e+01f,1.474968815e+01f,1.245491505e+01f,2.539999962e+01f,2.539999962e+01f,1.729361725e+01f,1.587500000e+01f,1.814285660e+01f,1.953846169e+01f,1.693333244e+01f,2.539999962e+01f,1.411111069e+01f,1.269999981e+01f,7.937500000e+00f,1.154545498e+01f,7.937500000e+00f,1.587500000e+01f,1.411111069e+01f,1.154545498e+01f,1.587500000e+01f,7.937500000e+00f,2.309090996e+01f,1.814285660e+01f,2.539999962e+01f,2.116666603e+01f,2.822222137e+01f,1.814285660e+01f,1.269999981e+01f,1.599704742e+01f,1.408665466e+01f,1.285376835e+01f,1.275981140e+01f,1.558020878e+01f,1.618670082e+01f,1.569585419e+01f,1.438982010e+01f,2.116666603e+01f,5.079999924e+01f,4.233333206e+01f,5.079999924e+01f,4.618181992e+01f,5.079999924e+01f,2.309090996e+01f,4.618181992e+01f,1.587500000e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.411111069e+01f,1.587500000e+01f,1.411111069e+01f,1.411111069e+01f,2.822222137e+01f,3.175000000e+01f,2.309090996e+01f,1.953846169e+01f,1.058333302e+01f,1.953846169e+01f,2.822222137e+01f,2.309090996e+01f,1.269999981e+01f,1.582284927e+01f,1.331367779e+01f,1.267647934e+01f,1.271987438e+01f,1.572146988e+01f,1.607018852e+01f,1.537664795e+01f,1.463186359e+01f,2.116666603e+01f,5.079999924e+01f,4.618181992e+01f,4.233333206e+01f,4.618181992e+01f,4.618181992e+01f,5.079999924e+01f,4.618181992e+01f,1.587500000e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.411111069e+01f,1.587500000e+01f,1.411111069e+01f,1.411111069e+01f,2.539999962e+01f,2.539999962e+01f,2.116666603e+01f,1.953846169e+01f,1.953846169e+01f,2.309090996e+01f,3.175000000e+01f,2.116666603e+01f,1.411111069e+01f,1.592847061e+01f,1.301130581e+01f,1.255592537e+01f,1.264074612e+01f,1.608708572e+01f,1.626820183e+01f,1.577199650e+01f,1.510605145e+01f,3.907692337e+01f,4.618181992e+01f,4.618181992e+01f,2.539999962e+01f,2.539999962e+01f,5.079999924e+01f,4.233333206e+01f,2.116666603e+01f,1.587500000e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.587500000e+01f,1.587500000e+01f,1.411111069e+01f,1.411111069e+01f,2.822222137e+01f,1.953846169e+01f,1.587500000e+01f,2.309090996e+01f,2.309090996e+01f,1.619123459e+01f,2.539999962e+01f,1.814285660e+01f,1.587500000e+01f,1.585757828e+01f,1.382680321e+01f,1.252026558e+01f,1.275292969e+01f,1.690636635e+01f,1.665573692e+01f,1.663017845e+01f,1.599704742e+01f,3.175000000e+01f,1.587500000e+01f,2.309090996e+01f,1.587500000e+01f,1.411111069e+01f,1.269999981e+01f,1.953846169e+01f,2.539999962e+01f,1.587500000e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.587500000e+01f,1.587500000e+01f,1.461870480e+01f,1.411111069e+01f,2.001970482e+01f,2.539999962e+01f,3.175000000e+01f,1.953846169e+01f,2.539999962e+01f,2.309090996e+01f,1.269999981e+01f,2.822222137e+01f,1.461870480e+01f,1.714428902e+01f,1.411111069e+01f,1.274980354e+01f,1.275981140e+01f,1.708123779e+01f,1.693884659e+01f,1.624483109e+01f,1.664507866e+01f,4.618181992e+01f,4.618181992e+01f,1.587500000e+01f,2.822222137e+01f,3.907692337e+01f,4.233333206e+01f,4.618181992e+01f,5.079999924e+01f,1.587500000e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.587500000e+01f,1.587500000e+01f,1.587500000e+01f,1.411111069e+01f,3.175000000e+01f,3.175000000e+01f,2.539999962e+01f,2.309090996e+01f,2.822222137e+01f,2.822222137e+01f,1.693333244e+01f,2.309090996e+01f,1.411111069e+01f,1.694215775e+01f,1.402966690e+01f,1.267339039e+01f,1.270992947e+01f,1.690087318e+01f,1.645240021e+01f,1.589780235e+01f,1.617965317e+01f,3.907692337e+01f,3.628571320e+01f,4.233333206e+01f,3.386666489e+01f,2.822222137e+01f,3.628571320e+01f,2.822222137e+01f,3.628571320e+01f,1.693333244e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.693333244e+01f,1.587500000e+01f,1.587500000e+01f,1.587500000e+01f,2.822222137e+01f,2.539999962e+01f,3.907692337e+01f,1.814285660e+01f,2.822222137e+01f,2.822222137e+01f,3.175000000e+01f,2.309090996e+01f,1.693333244e+01f,5.291666508e+00f,1.058333302e+01f,7.937500000e+00f,7.937500000e+00f,1.411111069e+01f,7.937500000e+00f,1.587500000e+01f,1.154545498e+01f,1.693333244e+01f,1.269999981e+01f,1.154545498e+01f,1.269999981e+01f,1.693333244e+01f,1.619123459e+01f,1.587500000e+01f,1.587500000e+01f,4.535714149e+00f,1.058333302e+01f,7.937500000e+00f,7.055555344e+00f,1.411111069e+01f,1.269999981e+01f,7.937500000e+00f,1.058333302e+01f,6.349999905e+00f,7.937500000e+00f,8.466666222e+00f,7.937500000e+00f,1.269999981e+01f,1.411111069e+01f,8.466666222e+00f,1.411111069e+01f,5.772727489e+00f,1.058333302e+01f,7.937500000e+00f,7.937500000e+00f,1.154545498e+01f,1.411111069e+01f,1.269999981e+01f,1.411111069e+01f,5.772727489e+00f,9.769230843e+00f,8.466666222e+00f,8.466666222e+00f,1.411111069e+01f,1.269999981e+01f,1.411111069e+01f,8.466666222e+00f,5.772727489e+00f,9.769230843e+00f,7.937500000e+00f,7.937500000e+00f,1.587500000e+01f,1.411111069e+01f,1.587500000e+01f,1.154545498e+01f,3.968750000e+00f,9.071428299e+00f,7.937500000e+00f,8.095617294e+00f,7.055555344e+00f,1.154545498e+01f,1.269999981e+01f,7.937500000e+00f,5.772727489e+00f,9.071428299e+00f,7.937500000e+00f,7.937500000e+00f,1.814285660e+01f,1.587500000e+01f,1.411111069e+01f,1.587500000e+01f,6.349999905e+00f,1.058333302e+01f,8.466666222e+00f,7.937500000e+00f,1.411111069e+01f,1.693333244e+01f,1.587500000e+01f,1.269999981e+01f,5.772727489e+00f,7.937500000e+00f,7.937500000e+00f,7.937500000e+00f,1.587500000e+01f,1.411111069e+01f,1.587500000e+01f,9.071428299e+00f,7.055555344e+00f,1.058333302e+01f,9.071428299e+00f,8.466666222e+00f,1.587500000e+01f,1.310967731e+01f,1.587500000e+01f,1.411111069e+01f,6.349999905e+00f,1.058333302e+01f,8.095617294e+00f,7.937500000e+00f,1.411111069e+01f,1.269999981e+01f,1.587500000e+01f,1.154545498e+01f,7.055555344e+00f,1.058333302e+01f,8.466666222e+00f,8.646808624e+00f,1.411111069e+01f,1.154545498e+01f,1.587500000e+01f,9.769230843e+00f,7.055555344e+00f,1.058333302e+01f,8.466666222e+00f,7.937500000e+00f,1.269999981e+01f,1.411111069e+01f,1.587500000e+01f,1.269999981e+01f,7.055555344e+00f,8.466666222e+00f,7.937500000e+00f,7.055555344e+00f,7.937500000e+00f,8.466666222e+00f,1.058333302e+01f,9.071428299e+00f,7.055555344e+00f,7.937500000e+00f,8.466666222e+00f,7.937500000e+00f,7.055555344e+00f,1.154545498e+01f,9.769230843e+00f,1.058333302e+01f,3.527777672e+00f,3.527777672e+00f,5.291666508e+00f,4.884615421e+00f,3.527777672e+00f,3.527777672e+00f,4.884615421e+00f,5.772727489e+00f,3.527777672e+00f,3.174999952e+00f,3.968750000e+00f,4.639269352e+00f,3.527777672e+00f,3.527777672e+00f,5.291666508e+00f,5.772727489e+00f,1.984375000e+00f,3.174999952e+00f,4.535714149e+00f,4.884615421e+00f,3.527777672e+00f,3.174999952e+00f,2.645833254e+00f,5.772727489e+00f,3.527777672e+00f,3.174999952e+00f,3.968750000e+00f,5.291666508e+00f,3.174999952e+00f,3.174999952e+00f,3.968750000e+00f,4.884615421e+00f,2.645833254e+00f,9.921875000e-01f,4.233333111e+00f,4.535714149e+00f,2.886363745e+00f,2.886363745e+00f,4.233333111e+00f,3.968750000e+00f,2.645833254e+00f,1.984375000e+00f,2.886363745e+00f,3.174999952e+00f,2.645833254e+00f,2.116666555e+00f,3.968750000e+00f,3.527777672e+00f,6.105769277e-01f,1.443181872e+00f,1.133928537e+00f,3.174999952e+00f,1.058333278e+00f,1.322916627e+00f,1.058333278e+00f,1.763888836e+00f,9.921875000e-01f,6.105769277e-01f,7.937499881e-01f,1.984375000e+00f,9.136690497e-01f,8.819444180e-01f,9.921875000e-01f,2.886363745e+00f};
static int research_range_offset(unsigned block,unsigned role,unsigned k,unsigned group){switch(block*16+role){case 82:if(k==64&&group==64)return 0;break;case 83:if(k==64&&group==64)return 1;break;case 92:if(k==64&&group==64)return 2;break;case 93:if(k==64&&group==64)return 3;break;case 98:if(k==64&&group==64)return 4;break;case 99:if(k==64&&group==64)return 5;break;case 108:if(k==64&&group==64)return 6;break;case 109:if(k==64&&group==64)return 7;break;case 114:if(k==64&&group==64)return 8;break;case 115:if(k==64&&group==64)return 9;break;case 124:if(k==64&&group==64)return 10;break;case 125:if(k==64&&group==64)return 11;break;case 130:if(k==64&&group==64)return 12;break;case 131:if(k==64&&group==64)return 13;break;case 140:if(k==64&&group==64)return 14;break;case 141:if(k==64&&group==64)return 15;break;case 146:if(k==128&&group==64)return 16;break;case 147:if(k==128&&group==64)return 18;break;case 156:if(k==128&&group==64)return 20;break;case 157:if(k==128&&group==64)return 22;break;case 162:if(k==128&&group==64)return 24;break;case 163:if(k==128&&group==64)return 26;break;case 172:if(k==128&&group==64)return 28;break;case 173:if(k==128&&group==64)return 30;break;case 178:if(k==128&&group==64)return 32;break;case 179:if(k==128&&group==64)return 34;break;case 188:if(k==128&&group==64)return 36;break;case 189:if(k==128&&group==64)return 38;break;case 194:if(k==128&&group==64)return 40;break;case 195:if(k==128&&group==64)return 42;break;case 204:if(k==128&&group==64)return 44;break;case 205:if(k==128&&group==64)return 46;break;case 210:if(k==128&&group==64)return 48;break;case 211:if(k==128&&group==64)return 50;break;case 220:if(k==128&&group==64)return 52;break;case 221:if(k==128&&group==64)return 54;break;case 226:if(k==128&&group==64)return 56;break;case 227:if(k==128&&group==64)return 58;break;case 236:if(k==128&&group==64)return 60;break;case 237:if(k==128&&group==64)return 62;break;case 242:if(k==256&&group==64)return 64;break;case 243:if(k==256&&group==64)return 68;break;case 252:if(k==256&&group==64)return 72;break;case 253:if(k==256&&group==64)return 76;break;case 258:if(k==256&&group==64)return 80;break;case 259:if(k==256&&group==64)return 84;break;case 268:if(k==256&&group==64)return 88;break;case 269:if(k==256&&group==64)return 92;break;case 274:if(k==256&&group==64)return 96;break;case 275:if(k==256&&group==64)return 100;break;case 284:if(k==256&&group==64)return 104;break;case 285:if(k==256&&group==64)return 108;break;case 290:if(k==256&&group==64)return 112;break;case 291:if(k==256&&group==64)return 116;break;case 300:if(k==256&&group==64)return 120;break;case 301:if(k==256&&group==64)return 124;break;case 306:if(k==256&&group==64)return 128;break;case 307:if(k==256&&group==64)return 132;break;case 316:if(k==256&&group==64)return 136;break;case 317:if(k==256&&group==64)return 140;break;case 322:if(k==256&&group==64)return 144;break;case 323:if(k==256&&group==64)return 148;break;case 332:if(k==256&&group==64)return 152;break;case 333:if(k==256&&group==64)return 156;break;case 338:if(k==256&&group==64)return 160;break;case 339:if(k==256&&group==64)return 164;break;case 348:if(k==256&&group==64)return 168;break;case 349:if(k==256&&group==64)return 172;break;case 354:if(k==256&&group==64)return 176;break;case 355:if(k==256&&group==64)return 180;break;case 364:if(k==256&&group==64)return 184;break;case 365:if(k==256&&group==64)return 188;break;case 370:if(k==512&&group==64)return 192;break;case 371:if(k==512&&group==64)return 200;break;case 376:if(k==512&&group==64)return 208;break;case 377:if(k==512&&group==64)return 216;break;case 378:if(k==64&&group==64)return 224;break;case 386:if(k==512&&group==64)return 225;break;case 387:if(k==512&&group==64)return 233;break;case 392:if(k==512&&group==64)return 241;break;case 393:if(k==512&&group==64)return 249;break;case 394:if(k==64&&group==64)return 257;break;case 402:if(k==512&&group==64)return 258;break;case 403:if(k==512&&group==64)return 266;break;case 408:if(k==512&&group==64)return 274;break;case 409:if(k==512&&group==64)return 282;break;case 410:if(k==64&&group==64)return 290;break;case 418:if(k==512&&group==64)return 291;break;case 419:if(k==512&&group==64)return 299;break;case 424:if(k==512&&group==64)return 307;break;case 425:if(k==512&&group==64)return 315;break;case 426:if(k==64&&group==64)return 323;break;case 434:if(k==512&&group==64)return 324;break;case 435:if(k==512&&group==64)return 332;break;case 440:if(k==512&&group==64)return 340;break;case 441:if(k==512&&group==64)return 348;break;case 442:if(k==64&&group==64)return 356;break;case 450:if(k==512&&group==64)return 357;break;case 451:if(k==512&&group==64)return 365;break;case 456:if(k==512&&group==64)return 373;break;case 457:if(k==512&&group==64)return 381;break;case 458:if(k==64&&group==64)return 389;break;case 466:if(k==512&&group==64)return 390;break;case 467:if(k==512&&group==64)return 398;break;case 472:if(k==512&&group==64)return 406;break;case 473:if(k==512&&group==64)return 414;break;case 474:if(k==64&&group==64)return 422;break;case 482:if(k==512&&group==64)return 423;break;case 483:if(k==512&&group==64)return 431;break;case 488:if(k==512&&group==64)return 439;break;case 489:if(k==512&&group==64)return 447;break;case 490:if(k==64&&group==64)return 455;break;case 496:if(k==1024&&group==64)return 456;break;case 498:if(k==1024&&group==64)return 472;break;case 499:if(k==1024&&group==64)return 488;break;case 512:if(k==1024&&group==64)return 504;break;case 514:if(k==1024&&group==64)return 520;break;case 515:if(k==1024&&group==64)return 536;break;case 528:if(k==1024&&group==64)return 552;break;case 530:if(k==1024&&group==64)return 568;break;case 531:if(k==1024&&group==64)return 584;break;case 544:if(k==1024&&group==64)return 600;break;case 546:if(k==1024&&group==64)return 616;break;case 547:if(k==1024&&group==64)return 632;break;case 560:if(k==1024&&group==64)return 648;break;case 562:if(k==1024&&group==64)return 664;break;case 563:if(k==1024&&group==64)return 680;break;case 576:if(k==1024&&group==64)return 696;break;case 578:if(k==1024&&group==64)return 712;break;case 579:if(k==1024&&group==64)return 728;break;case 592:if(k==1024&&group==64)return 744;break;case 594:if(k==1024&&group==64)return 760;break;case 595:if(k==1024&&group==64)return 776;break;case 608:if(k==1024&&group==64)return 792;break;case 610:if(k==1024&&group==64)return 808;break;case 611:if(k==1024&&group==64)return 824;break;case 630:if(k==1024&&group==64)return 840;break;case 642:if(k==512&&group==64)return 856;break;case 643:if(k==512&&group==64)return 864;break;case 648:if(k==512&&group==64)return 872;break;case 649:if(k==512&&group==64)return 880;break;case 650:if(k==64&&group==64)return 888;break;case 658:if(k==512&&group==64)return 889;break;case 659:if(k==512&&group==64)return 897;break;case 664:if(k==512&&group==64)return 905;break;case 665:if(k==512&&group==64)return 913;break;case 666:if(k==64&&group==64)return 921;break;case 674:if(k==512&&group==64)return 922;break;case 675:if(k==512&&group==64)return 930;break;case 680:if(k==512&&group==64)return 938;break;case 681:if(k==512&&group==64)return 946;break;case 682:if(k==64&&group==64)return 954;break;case 690:if(k==512&&group==64)return 955;break;case 691:if(k==512&&group==64)return 963;break;case 696:if(k==512&&group==64)return 971;break;case 697:if(k==512&&group==64)return 979;break;case 698:if(k==64&&group==64)return 987;break;case 706:if(k==512&&group==64)return 988;break;case 707:if(k==512&&group==64)return 996;break;case 712:if(k==512&&group==64)return 1004;break;case 713:if(k==512&&group==64)return 1012;break;case 714:if(k==64&&group==64)return 1020;break;case 722:if(k==512&&group==64)return 1021;break;case 723:if(k==512&&group==64)return 1029;break;case 728:if(k==512&&group==64)return 1037;break;case 729:if(k==512&&group==64)return 1045;break;case 730:if(k==64&&group==64)return 1053;break;case 738:if(k==512&&group==64)return 1054;break;case 739:if(k==512&&group==64)return 1062;break;case 744:if(k==512&&group==64)return 1070;break;case 745:if(k==512&&group==64)return 1078;break;case 746:if(k==64&&group==64)return 1086;break;case 754:if(k==512&&group==64)return 1087;break;case 755:if(k==512&&group==64)return 1095;break;case 760:if(k==512&&group==64)return 1103;break;case 761:if(k==512&&group==64)return 1111;break;case 762:if(k==64&&group==64)return 1119;break;case 770:if(k==256&&group==64)return 1120;break;case 771:if(k==256&&group==64)return 1124;break;case 775:if(k==512&&group==64)return 1128;break;case 780:if(k==256&&group==64)return 1136;break;case 781:if(k==256&&group==64)return 1140;break;case 786:if(k==256&&group==64)return 1144;break;case 787:if(k==256&&group==64)return 1148;break;case 796:if(k==256&&group==64)return 1152;break;case 797:if(k==256&&group==64)return 1156;break;case 802:if(k==256&&group==64)return 1160;break;case 803:if(k==256&&group==64)return 1164;break;case 812:if(k==256&&group==64)return 1168;break;case 813:if(k==256&&group==64)return 1172;break;case 818:if(k==256&&group==64)return 1176;break;case 819:if(k==256&&group==64)return 1180;break;case 828:if(k==256&&group==64)return 1184;break;case 829:if(k==256&&group==64)return 1188;break;case 834:if(k==256&&group==64)return 1192;break;case 835:if(k==256&&group==64)return 1196;break;case 844:if(k==256&&group==64)return 1200;break;case 845:if(k==256&&group==64)return 1204;break;case 850:if(k==256&&group==64)return 1208;break;case 851:if(k==256&&group==64)return 1212;break;case 860:if(k==256&&group==64)return 1216;break;case 861:if(k==256&&group==64)return 1220;break;case 866:if(k==256&&group==64)return 1224;break;case 867:if(k==256&&group==64)return 1228;break;case 876:if(k==256&&group==64)return 1232;break;case 877:if(k==256&&group==64)return 1236;break;case 882:if(k==256&&group==64)return 1240;break;case 883:if(k==256&&group==64)return 1244;break;case 892:if(k==256&&group==64)return 1248;break;case 893:if(k==256&&group==64)return 1252;break;case 898:if(k==128&&group==64)return 1256;break;case 899:if(k==128&&group==64)return 1258;break;case 908:if(k==128&&group==64)return 1260;break;case 909:if(k==128&&group==64)return 1262;break;case 914:if(k==128&&group==64)return 1264;break;case 915:if(k==128&&group==64)return 1266;break;case 924:if(k==128&&group==64)return 1268;break;case 925:if(k==128&&group==64)return 1270;break;case 930:if(k==128&&group==64)return 1272;break;case 931:if(k==128&&group==64)return 1274;break;case 940:if(k==128&&group==64)return 1276;break;case 941:if(k==128&&group==64)return 1278;break;case 946:if(k==128&&group==64)return 1280;break;case 947:if(k==128&&group==64)return 1282;break;case 956:if(k==128&&group==64)return 1284;break;case 957:if(k==128&&group==64)return 1286;break;case 962:if(k==128&&group==64)return 1288;break;case 963:if(k==128&&group==64)return 1290;break;case 972:if(k==128&&group==64)return 1292;break;case 973:if(k==128&&group==64)return 1294;break;case 978:if(k==128&&group==64)return 1296;break;case 979:if(k==128&&group==64)return 1298;break;case 988:if(k==128&&group==64)return 1300;break;case 989:if(k==128&&group==64)return 1302;break;case 994:if(k==64&&group==64)return 1304;break;case 995:if(k==64&&group==64)return 1305;break;case 1004:if(k==64&&group==64)return 1306;break;case 1005:if(k==64&&group==64)return 1307;break;case 1010:if(k==64&&group==64)return 1308;break;case 1011:if(k==64&&group==64)return 1309;break;case 1020:if(k==64&&group==64)return 1310;break;case 1021:if(k==64&&group==64)return 1311;break;case 1026:if(k==64&&group==64)return 1312;break;case 1027:if(k==64&&group==64)return 1313;break;case 1036:if(k==64&&group==64)return 1314;break;case 1037:if(k==64&&group==64)return 1315;break;case 1042:if(k==64&&group==64)return 1316;break;case 1043:if(k==64&&group==64)return 1317;break;case 1052:if(k==64&&group==64)return 1318;break;case 1053:if(k==64&&group==64)return 1319;break;}return -1;}

// The matrix kernels still consume their usual per-row scale layout. Only the
// separate maximum reduction disappears; clipping and rounding remain explicit.
template<unsigned GroupSize,bool Window=false>
__global__ void research_fixed_pack(const __half* input,std::int8_t* output,
    float* scales,unsigned rows,unsigned k,unsigned* status,unsigned offset,
    unsigned h=0,unsigned w=0,unsigned top=0,unsigned left=0,unsigned pw=0){
    const size_t pair=size_t(blockIdx.x)*blockDim.x+threadIdx.x,i=pair*2;
    if(i>=size_t(rows)*k)return;
    const unsigned row=i/k,column=i%k,group=column/GroupSize;
    size_t source=row;bool valid=true;
    if constexpr(Window){
        const unsigned token=row%64,window=row/64;
        const int y=int(window/(pw/8)*8+token/8)-int(top);
        const int x=int(window%(pw/8)*8+token%8)-int(left);
        valid=y>=0&&y<int(h)&&x>=0&&x<int(w);
        if(valid)source=size_t(y)*w+x;
    }
    const __half2 values=valid?*reinterpret_cast<const __half2*>(input+source*k+column):__float2half2_rn(0.f);
    const float a=__low2float(values),b=__high2float(values),inverse=research_large_inverse[offset+group];
    if(!isfinite(a)||!isfinite(b))atomicOr(status,1u);
    // Clip before conversion: tiny fitted ranges can overflow an int32.
    const unsigned lo=unsigned(std::uint8_t(__float2int_rn(fminf(127.f,fmaxf(-127.f,a*inverse)))));
    const unsigned hi=unsigned(std::uint8_t(__float2int_rn(fminf(127.f,fmaxf(-127.f,b*inverse)))));
    *reinterpret_cast<std::uint16_t*>(output+i)=std::uint16_t(lo|(hi<<8));
    if(column%GroupSize==0)scales[size_t(row)*(k/GroupSize)+group]=research_large_scale[offset+group];
}
static bool research_range_enabled(unsigned mask,unsigned block,unsigned role){
    if(mask==0)return true;
    if(mask==1)return role==12||role==13;
    if(mask==2)return (block>=23&&block<=30)||(block>=40&&block<=47);
    if(mask==3)return block>=31&&block<=38;
    if(mask==4)return role==2||role==3;
    if(mask==5)return role>=4&&role<=7;
    if(mask==6)return role!=2&&role!=3;
    return false;
}

// Once an activation scale is fixed, its product with each weight-group scale
// is also fixed. Compute that product once, in FP32, preserving its rounding.
__global__ void research_fold_weight_scales(float* scales,unsigned groups,unsigned n,
    unsigned batches,unsigned offset){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(batches)*groups*n)return;
    scales[i]=scales[i]*research_large_scale[offset+(i/n)%groups];
}

// Fixed input scales can be absorbed into the original FP16 weights. Use one
// scale per output column, so an integer dot can run across the whole K axis.
// Reading the original weights avoids rounding the old INT8 codes a second time.
__global__ void research_long_weights(const __half* input,std::int8_t* output,
    float* scales,unsigned k,unsigned n,unsigned group,unsigned offset,
    bool packed,unsigned* status){
    __shared__ float maxima[128];
    const unsigned col=blockIdx.x,batch=blockIdx.y,lane=threadIdx.x,groups=k/group;
    float maximum=0;
    for(unsigned row=lane;row<k;row+=128){
        const float value=__half2float(input[(size_t(batch)*k+row)*n+col])*
                          research_large_scale[offset+row/group];
        if(!isfinite(value))atomicOr(status,1u);
        maximum=fmaxf(maximum,fabsf(value));
    }
    maxima[lane]=maximum;__syncthreads();
    for(unsigned stride=64;stride;stride>>=1){
        if(lane<stride)maxima[lane]=fmaxf(maxima[lane],maxima[lane+stride]);
        __syncthreads();
    }
    const float scale=maxima[0]==0?1.f:maxima[0]/127.f;
    const float inverse=maxima[0]==0?1.f:127.f/maxima[0];
    // Keep the old allocation layout for its existing bounds/guard checks.
    // The long-dot kernels read only group zero; the grouped control reads all.
    for(unsigned g=lane;g<groups;g+=128)scales[(size_t(batch)*groups+g)*n+col]=scale;
    for(unsigned row=lane;row<k;row+=128){
        const float value=__half2float(input[(size_t(batch)*k+row)*n+col])*
                          research_large_scale[offset+row/group];
        const int code=__float2int_rn(fminf(127.f,fmaxf(-127.f,value*inverse)));
        const size_t index=packed?
            size_t(batch)*k*n+(size_t(col/32)*(k/4)*32+size_t(row/4)*32+col%32)*4+row%4:
            (size_t(batch)*n+col)*k+row;
        output[index]=std::int8_t(code);
    }
}

static bool research_long_enabled(unsigned mask,unsigned block,unsigned role,unsigned k){
    if(k<=64||mask==0)return false; // A single group has nothing to amortize.
    if(mask==1)return true;
    if(mask==2)return role==2;
    if(mask==3)return role==3;
    if(mask==4)return role==12||role==13;
    if(mask==5)return role==8||role==9;
    if(mask==6)return (block>=23&&block<=30)||(block>=40&&block<=47);
    if(mask==7)return block>=31&&block<=38;
    return false;
}

namespace rdna2_nr {
template<unsigned GroupSize,int RowsPerThread,int Epilogue,unsigned TileRows=16,unsigned TileColumns=32,unsigned TileK=32,
         int PackedWeights=0,int DotUnroll=TileK/4,bool DirectWeights=false>
__global__ void research_ffn_to_qkv_codes(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,
    const __half* skip,const __half* cosine,__half* out,
    unsigned m,unsigned k,unsigned n,bool publish,
    std::int8_t* nextCodes,unsigned nextOffset,unsigned imageH,unsigned imageW,
    unsigned top,unsigned left,unsigned paddedWidth,unsigned* status) {
    static_cast<void>(scaleA);
    static_assert(Epilogue==3,"only final FFN residuals are handed off");
    static_assert(GroupSize==32||GroupSize==64||GroupSize==128,"supported W8A8 group size");
    static_assert(GroupSize%TileK==0&&TileK%4==0,"tile must divide scale group");
    static_assert(TileRows%RowsPerThread==0&&TileColumns%32==0,"invalid output tile");
    static_assert(!DirectWeights||PackedWeights,"direct loads require packed weights");
    constexpr unsigned RowThreads=TileRows/RowsPerThread,DotGroups=TileK/4;
    constexpr unsigned Threads=TileColumns*RowThreads;
    __shared__ std::int32_t tileA[TileRows][DotGroups];
    // Transposed with padding: lanes read adjacent banks instead of stride-8.
    __shared__ std::int32_t tileW[DotGroups][TileColumns+1];
    const unsigned local=threadIdx.y*TileColumns+threadIdx.x;
    const unsigned column=blockIdx.x*TileColumns+threadIdx.x;
    float totals[RowsPerThread]={};
    std::int32_t partial[RowsPerThread]={};
    for(unsigned base=0;base<k;base+=TileK){

            for(unsigned index=local;index<TileRows*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*TileRows+r;
                tileA[r][g]=(row<m)?
                    *reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+base+g*4):0;
            }
            if constexpr(!DirectWeights){
                for(unsigned index=local;index<TileColumns*DotGroups;index+=Threads){
                    const unsigned c=PackedWeights?index%TileColumns:index/DotGroups;
                    const unsigned g=PackedWeights?index/TileColumns:index%DotGroups;
                    const unsigned globalColumn=blockIdx.x*TileColumns+c;
                    const size_t offset=PackedWeights==2?(size_t(globalColumn/32)*(k/4)*32+size_t(base/4+g)*32+globalColumn%32)*4:
                        PackedWeights?(size_t(base/4+g)*n+globalColumn)*4:size_t(globalColumn)*k+base+g*4;
                    tileW[g][c]=(globalColumn<n)?*reinterpret_cast<const std::int32_t*>(w+offset):0;
                }
            }
            __syncthreads();
            #pragma unroll DotUnroll
            for(unsigned g=0;g<DotGroups;++g){
                char4 bv;
                if constexpr(DirectWeights){
                    const size_t offset=PackedWeights==2?size_t(column/32)*(k/4)*32+size_t(base/4+g)*32+column%32:size_t(base/4+g)*n+column;
                    const std::int32_t bits=column<n?reinterpret_cast<const std::int32_t*>(w)[offset]:0;
                    bv=*reinterpret_cast<const char4*>(&bits);
                }else bv=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
                #pragma unroll
                for(unsigned r=0;r<RowsPerThread;++r){
                    const unsigned tileRow=threadIdx.y+r*RowThreads;
                    const char4 av=*reinterpret_cast<const char4*>(&tileA[tileRow][g]);
                    partial[r]=amd_mixed_dot(av,bv,partial[r],false);
                }
            }
            __syncthreads();

    }
        #pragma unroll
        for(unsigned r=0;r<RowsPerThread;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<m&&column<n){
                const float combined=scaleW[column];
                totals[r]=float(partial[r])*combined;
            }
        }

    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
        if(row<m&&column<n){
            __half value=__float2half(totals[r]);
            if constexpr(Epilogue==1)value=pubh(ffn_gate(value));
            if constexpr(Epilogue==2)value=pubh(value);
            if constexpr(Epilogue==3){
                value=__hadd(value,__hmul(skip[size_t(row)*n+column],cosine[column]));
                if(publish)value=pubh(value);
            }
            if constexpr(Epilogue==4)value=ffn_gate(value);
            out[size_t(row)*n+column]=value;
            if(row<imageH*imageW){
                const unsigned y=row/imageW+top,x=row%imageW+left;
                const size_t windowRow=(size_t(y/8)*(paddedWidth/8)+x/8)*64+(y%8)*8+x%8;
                const float v=__half2float(value);
                if(!isfinite(v))atomicOr(status,1u);
                const float scaled=v*research_large_inverse[nextOffset+column/64];
                nextCodes[windowRow*n+column]=std::int8_t(__float2int_rn(fminf(127.f,fmaxf(-127.f,scaled))));
            }
        }
    }
}
}
struct GroupedInt8Plan{
    // Separate storage is essential: the FFN output kernel is still reading
    // its own packed input while it produces the next QKV's input codes.
    std::unique_ptr<AlignedInt8Buffer<std::int8_t>>windowHandoffCodes;
    size_t windowHandoffCapacity=0;
    bool windowHandoffArmed=false,windowHandoffReady=false;
    unsigned windowHandoffBlock=0,windowHandoffOffset=0,windowHandoffRows=0,
             windowHandoffChannels=0,windowHandoffHeight=0,windowHandoffWidth=0,
             windowHandoffTop=0,windowHandoffLeft=0,windowHandoffPaddedWidth=0;
    void arm_window_handoff(unsigned block,const __half*weight,unsigned c,
                            unsigned h,unsigned w,unsigned top,unsigned left,unsigned pw,
                            unsigned rows,bool allowed){
        windowHandoffArmed=windowHandoffReady=false;
        if(!allowed||c<128||groupSize!=64||!enabled(block,Int8Role::Qkv))return;
        auto& entry=prepare_with_group(block,Int8Role::Qkv,weight,c,3*c,0,false,groupSize);
        if(!entry.researchLong)return;
        const int offset=research_range_offset(block,unsigned(Int8Role::Qkv),c,groupSize);
        if(offset<0)throw std::runtime_error("missing fixed QKV handoff range");
        const size_t count=size_t(rows)*c;
        if(count>windowHandoffCapacity){
            windowHandoffCodes=std::make_unique<AlignedInt8Buffer<std::int8_t>>(count,stream);
            windowHandoffCapacity=count;
        }
        // FFN covers real pixels. The window halo must remain quantized zero.
        HIP_CHECK(hipMemsetAsync(windowHandoffCodes->data,0,count,stream));
        windowHandoffBlock=block;windowHandoffOffset=unsigned(offset);
        windowHandoffRows=rows;windowHandoffChannels=c;windowHandoffHeight=h;
        windowHandoffWidth=w;windowHandoffTop=top;windowHandoffLeft=left;windowHandoffPaddedWidth=pw;
        windowHandoffArmed=true;
    }

    unsigned packBlock=0,packRole=0;
    unsigned researchMask=0,researchLongMask=1;
    hipStream_t stream=nullptr;unsigned groupSize=0;Int8Scope scope=Int8Scope::Block31;
    bool validateGuards=false,wavePack=false,tile64=true,row32=true,group128Global=false,hardwareLayout=false,qkvHardwareLayout=false,branchFp16Packed=false,c32AttentionWave=false,prefetch64=false;
    unsigned splitResident=4;
    std::unordered_map<unsigned,std::unique_ptr<GroupedInt8Matrix>>matrices;
    std::unordered_map<unsigned,std::unique_ptr<AlignedInt8Buffer<__half>>> researchScaledWeights;
    const __half* researchWeightSources[20]={};
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
        char longText[4]={};size_t longSize=0;
        if(NrExecution::ReadSetting(&longSize,longText,sizeof(longText),"NR_RESEARCH_LONG_MASK"))throw std::runtime_error("long mask read");
        if(longSize&&longText[0]){
            if(longText[1]||longText[0]<'0'||longText[0]>'7')throw std::runtime_error("long mask must be 0..7");
            researchLongMask=unsigned(longText[0]-'0');
        }
        char maskText[4]={};size_t maskSize=0;
        if(NrExecution::ReadSetting(&maskSize,maskText,sizeof(maskText),"NR_RESEARCH_STATIC_MASK"))throw std::runtime_error("static mask read");
        if(maskSize&&maskText[0]){
            if(maskText[1]||maskText[0]<'0'||maskText[0]>'7')throw std::runtime_error("static mask must be 0..7");
            researchMask=unsigned(maskText[0]-'0');
        }
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
    template<unsigned GroupSize,unsigned TileRows=16,unsigned TileK=32,int Layout=0,int Unroll=TileK/4>void research_long_launch_dot_tiled(GroupedInt8Matrix&entry,const __half*skip,const __half*cosine,__half*output,unsigned m){
        if(entry.epilogue==3&&windowHandoffArmed&&
           (packRole==unsigned(Int8Role::BranchedOutput)||packRole==unsigned(Int8Role::SplitOutput))){
            if(packBlock!=windowHandoffBlock||entry.n!=windowHandoffChannels||m<windowHandoffHeight*windowHandoffWidth)
                throw std::runtime_error("FFN/QKV handoff identity mismatch");
            const dim3 tiles(entry.n/32,(m+TileRows-1)/TileRows),workers(32,TileRows/4);
            rdna2_nr::research_ffn_to_qkv_codes<GroupSize,4,3,TileRows,32,TileK,Layout,Unroll><<<tiles,workers,0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish,
                windowHandoffCodes->data,windowHandoffOffset,windowHandoffHeight,windowHandoffWidth,
                windowHandoffTop,windowHandoffLeft,windowHandoffPaddedWidth,status->data);
            windowHandoffReady=true;return;
        }

        const dim3 grid(entry.n/32,(m+TileRows-1)/TileRows),threads(32,TileRows/4);
        if(entry.epilogue==0)rdna2_nr::research_long_dot4_grouped_w8a8<GroupSize,4,0,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else if(entry.epilogue==1)rdna2_nr::research_long_dot4_grouped_w8a8<GroupSize,4,1,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else if(entry.epilogue==2)rdna2_nr::research_long_dot4_grouped_w8a8<GroupSize,4,2,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else rdna2_nr::research_long_dot4_grouped_w8a8<GroupSize,4,3,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
    }
    template<unsigned GroupSize,unsigned TileRows=16,unsigned TileK=32,int Layout=0,int Unroll=TileK/4>void research_folded_launch_dot_tiled(GroupedInt8Matrix&entry,const __half*skip,const __half*cosine,__half*output,unsigned m){
        if(entry.researchLong){research_long_launch_dot_tiled<GroupSize,TileRows,TileK,Layout,Unroll>(entry,skip,cosine,output,m);return;}
        const dim3 grid(entry.n/32,(m+TileRows-1)/TileRows),threads(32,TileRows/4);
        if(entry.epilogue==0)rdna2_nr::research_folded_dot4_grouped_w8a8<GroupSize,4,0,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else if(entry.epilogue==1)rdna2_nr::research_folded_dot4_grouped_w8a8<GroupSize,4,1,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else if(entry.epilogue==2)rdna2_nr::research_folded_dot4_grouped_w8a8<GroupSize,4,2,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
        else rdna2_nr::research_folded_dot4_grouped_w8a8<GroupSize,4,3,TileRows,32,TileK,Layout,Unroll><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,entry.k,entry.n,entry.publish);
    }
    template<unsigned GroupSize,unsigned TileRows=16,unsigned TileK=32,int Layout=0,int Unroll=TileK/4>void launch_dot_tiled(GroupedInt8Matrix&entry,const __half*skip,const __half*cosine,__half*output,unsigned m){
        if(entry.researchFolded){research_folded_launch_dot_tiled<GroupSize,TileRows,TileK,Layout,Unroll>(entry,skip,cosine,output,m);return;}
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
    template<unsigned GroupSize,unsigned TileK=32,int PackedWeights=0,int DotUnroll=TileK/4>void research_long_launch_qkv_dot_tiled(
        GroupedInt8Matrix&entry,const __half*scale,__half*output,unsigned m,
        unsigned heads,unsigned tokensPerWindow){
        const dim3 grid(3*heads,(m+63)/64),threads(32,8);
        rdna2_nr::research_long_dot4_grouped_w8a8_qkv_normalize<GroupSize,8,TileK,PackedWeights,DotUnroll><<<grid,threads,0,stream>>>(
            (windowHandoffReady?windowHandoffCodes->data:activation->data),entry.weight->data,activationScale->data,entry.scale->data,
            scale,output,m,entry.k,heads,tokensPerWindow);
    }
    template<unsigned GroupSize,unsigned TileK=32,int PackedWeights=0,int DotUnroll=TileK/4>void research_folded_launch_qkv_dot_tiled(
        GroupedInt8Matrix&entry,const __half*scale,__half*output,unsigned m,
        unsigned heads,unsigned tokensPerWindow){
        if(entry.researchLong){research_long_launch_qkv_dot_tiled<GroupSize,TileK,PackedWeights,DotUnroll>(entry,scale,output,m,heads,tokensPerWindow);return;}
        const dim3 grid(3*heads,(m+15)/16),threads(32,4);
        rdna2_nr::research_folded_dot4_grouped_w8a8_qkv_normalize<GroupSize,4,TileK,PackedWeights,DotUnroll><<<grid,threads,0,stream>>>(
            activation->data,entry.weight->data,activationScale->data,entry.scale->data,
            scale,output,m,entry.k,heads,tokensPerWindow);
    }
    template<unsigned GroupSize,unsigned TileK=32,int PackedWeights=0,int DotUnroll=TileK/4>void launch_qkv_dot_tiled(
        GroupedInt8Matrix&entry,const __half*scale,__half*output,unsigned m,
        unsigned heads,unsigned tokensPerWindow){
        if(entry.researchFolded){research_folded_launch_qkv_dot_tiled<GroupSize,TileK,PackedWeights,DotUnroll>(entry,scale,output,m,heads,tokensPerWindow);return;}
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
        const int offset=research_range_offset(packBlock,packRole,k,GroupSize);
        if(offset>=0&&research_range_enabled(researchMask,packBlock,packRole)){
            research_fixed_pack<GroupSize,false><<<(size_t(rows)*k/2+255)/256,256,0,stream>>>(
                input,activation->data,activationScale->data,rows,k,status->data,offset);
            stage_event("pack",false);return;
        }
        if(wavePack)rdna2_nr::quantize_groups_wave_w8a8<GroupSize><<<(rows*(k/GroupSize)+3)/4,128,0,stream>>>(input,activation->data,activationScale->data,nullptr,rows,k,status->data);
        else rdna2_nr::quantize_row_groups_w8a8<GroupSize><<<rows,256,0,stream>>>(input,activation->data,activationScale->data,nullptr,rows,k,status->data);
        stage_event("pack",false);
    }
    template<unsigned GroupSize>void pack_window(const __half*input,unsigned rows,unsigned k,unsigned h,unsigned w,unsigned top,unsigned left,unsigned paddedWidth){
        stage_event("pack",true);
        const int offset=research_range_offset(packBlock,packRole,k,GroupSize);
        if(offset>=0&&research_range_enabled(researchMask,packBlock,packRole)){
            research_fixed_pack<GroupSize,true><<<(size_t(rows)*k/2+255)/256,256,0,stream>>>(
                input,activation->data,activationScale->data,rows,k,status->data,offset,h,w,top,left,paddedWidth);
            stage_event("pack",false);return;
        }
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
        packBlock=block;packRole=unsigned(role);
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
            const int range=research_range_offset(block,unsigned(role),k,entry.groupSize);
            // Global QKV can consume dynamic residual-handoff scales at larger
            // token counts. Its cached weights must work for both input paths.
            const bool eligible=(role==Int8Role::Qkv&&!(block>=31&&block<=38))||role==Int8Role::AttentionProjection||
                role==Int8Role::BranchedExpand||role==Int8Role::BranchedOutput||
                role==Int8Role::SplitFirst||role==Int8Role::SplitOutput;
            // Prefetch kernels retain their ordinary dynamic-scale epilogues.
            if(eligible&&!prefetch64&&range>=0&&research_range_enabled(researchMask,block,unsigned(role))){
                research_fold_weight_scales<<<(size_t(entry.batches)*entry.groups*entry.n+255)/256,256,0,stream>>>(
                    entry.scale->data,entry.groups,entry.n,entry.batches,range);
                entry.researchFolded=true;
                if(research_long_enabled(researchLongMask,block,unsigned(role),k)){
                    // Codes are bounded by 127. This check is independent of
                    // calibration or input data, and covers every cached matrix.
                    if(uint64_t(k)*127u*127u>2147483647u)throw std::runtime_error("long INT32 overflow bound");
                    research_long_weights<<<dim3(n,entry.batches),128,0,stream>>>(matrix,entry.weight->data,
                        entry.scale->data,k,n,entry.groupSize,range,entry.hardwarePacked,status->data);
                    entry.researchLong=true;
                }
            }
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
        packBlock=block;packRole=unsigned(role);
        const unsigned id=key(block,role);auto&owner=matrices[id];
        if(!owner)owner=std::make_unique<GroupedInt8Matrix>(k,n,groupSize,epilogue,false,stream,batches);
        auto&entry=*owner;
        if(entry.k!=k||entry.n!=n||entry.batches!=batches||entry.epilogue!=epilogue||entry.publish)throw std::runtime_error("batched grouped INT8 matrix identity changed");
        if(entry.source&&entry.source!=matrix)throw std::runtime_error("batched grouped INT8 weight pointer changed");
        if(!entry.packed){
            const dim3 grid((n+63)/64,k/groupSize,batches);
            if(groupSize==32)rdna2_nr::quantize_batched_weight_groups_w8a8<32><<<grid,64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,batches,k,n,status->data);
            else rdna2_nr::quantize_batched_weight_groups_w8a8<64><<<grid,64,0,stream>>>(matrix,entry.weight->data,entry.scale->data,batches,k,n,status->data);
            const int range=research_range_offset(block,unsigned(role),k,entry.groupSize);
            // Global QKV can consume dynamic residual-handoff scales at larger
            // token counts. Its cached weights must work for both input paths.
            const bool eligible=(role==Int8Role::Qkv&&!(block>=31&&block<=38))||role==Int8Role::AttentionProjection||
                role==Int8Role::BranchedExpand||role==Int8Role::BranchedOutput||
                role==Int8Role::SplitFirst||role==Int8Role::SplitOutput;
            // Prefetch kernels retain their ordinary dynamic-scale epilogues.
            if(eligible&&!prefetch64&&range>=0&&research_range_enabled(researchMask,block,unsigned(role))){
                research_fold_weight_scales<<<(size_t(entry.batches)*entry.groups*entry.n+255)/256,256,0,stream>>>(
                    entry.scale->data,entry.groups,entry.n,entry.batches,range);
                entry.researchFolded=true;
                if(research_long_enabled(researchLongMask,block,unsigned(role),k)){
                    // Codes are bounded by 127. This check is independent of
                    // calibration or input data, and covers every cached matrix.
                    if(uint64_t(k)*127u*127u>2147483647u)throw std::runtime_error("long INT32 overflow bound");
                    research_long_weights<<<dim3(n,entry.batches),128,0,stream>>>(matrix,entry.weight->data,
                        entry.scale->data,k,n,entry.groupSize,range,entry.hardwarePacked,status->data);
                    entry.researchLong=true;
                }
            }
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
        if(!(block<=4||(block>=66&&block<=70)))throw std::runtime_error("research C32 block");
        auto balanced=[&](const __half* source,unsigned k,unsigned n,unsigned offset){
            const unsigned key=(block<5?block:block-61)*2+(offset!=0);
            auto& buffer=researchScaledWeights[key];
            if(!buffer){
                buffer=std::make_unique<AlignedInt8Buffer<__half>>(size_t(k)*n,stream);
                researchWeightSources[key]=source;
                research_balance_weights<<<(k*n+255)/256,256,0,stream>>>(source,buffer->data,block,k,n,offset,status->data);
                HIP_CHECK(hipGetLastError());
            }
            if(researchWeightSources[key]!=source)throw std::runtime_error("research weight source changed");
            if(validateGuards)validate_one_int8_guard<<<1,32,0,stream>>>(buffer->allocation,buffer->payload_bytes(),status->data);
            return buffer->data;
        };
        const __half* balancedExpand=balanced(expand,32,128,0);
        const __half* balancedContract=balanced(contract,128,32,32);

        auto&first=prepare_with_group(block,Int8Role::FfnExpand,balancedExpand,32,128,1,true,matrixGroup,true);
        auto&second=prepare_with_group(block,Int8Role::FfnContract,balancedContract,128,32,3,false,matrixGroup,true);
        const dim3 grid((rows+15)/16),threads(32,4);
        if(matrixGroup==16)fused_c32_ffn_w8a8<16><<<grid,threads,0,stream>>>(input,first.weight->data,
            first.scale->data,second.weight->data,second.scale->data,cosine,output,rows,status->data,block);
        else research_c32_fixed_full<<<grid,dim3(32,8),0,stream>>>(input,first.weight->data,
            first.scale->data,second.weight->data,second.scale->data,cosine,output,rows,status->data,block);
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
        packBlock=block;packRole=unsigned(Int8Role::SplitExpand);
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
            if(entry.researchLong)rdna2_nr::research_long_dot4_broadcast_batched_gate_w8a8<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);else if(entry.researchFolded)rdna2_nr::research_folded_dot4_broadcast_batched_gate_w8a8<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);else rdna2_nr::dot4_broadcast_batched_gate_w8a8<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
        }else{
            pack_rows<64>(input,rows,k);
            if(entry.researchLong)rdna2_nr::research_long_dot4_broadcast_batched_gate_w8a8<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);else if(entry.researchFolded)rdna2_nr::research_folded_dot4_broadcast_batched_gate_w8a8<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);else rdna2_nr::dot4_broadcast_batched_gate_w8a8<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,output,rows,batches,k,n);
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
            if(entry.researchLong)rdna2_nr::research_long_dot4_branched_staged_w8a8<64,4,true><<<dim3(channels/32,(rows+31)/32),dim3(32,8),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,
                reinterpret_cast<const __half*>(packed->weight->data),output,rows,channels);else if(entry.researchFolded)rdna2_nr::research_folded_dot4_branched_staged_w8a8<64,4,true><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,
                reinterpret_cast<const __half*>(packed->weight->data),output,rows,channels);else rdna2_nr::dot4_branched_staged_w8a8<64,4,true><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,
                reinterpret_cast<const __half*>(packed->weight->data),output,rows,channels);
            if(validateGuards)validate_one_int8_guard<<<1,32,0,stream>>>(packed->weight->allocation,
                packed->weight->payload_bytes(),status->data);
        }else if(entry.researchLong)rdna2_nr::research_long_dot4_branched_resident_w8a8<64,4,false><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,project,output,rows,channels);else if(entry.researchFolded)rdna2_nr::research_folded_dot4_branched_resident_w8a8<64,4,false><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
                activation->data,entry.weight->data,activationScale->data,entry.scale->data,project,output,rows,channels);else rdna2_nr::dot4_branched_resident_w8a8<64,4,false><<<dim3(channels/32,(rows+15)/16),dim3(32,4),0,stream>>>(
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
        if(windowHandoffReady){
            if(block!=windowHandoffBlock||m!=windowHandoffRows||k!=windowHandoffChannels||!entry.researchLong)
                throw std::runtime_error("QKV handoff consume mismatch");
            launch_qkv_dot<64>(entry,scale,output,m,heads,64);
            if(validateGuards)validate_one_int8_guard<<<1,32,0,stream>>>(windowHandoffCodes->allocation,windowHandoffCodes->payload_bytes(),status->data);
            windowHandoffReady=windowHandoffArmed=false;finish(entry);return;
        }
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
            if(entry.researchLong)rdna2_nr::research_long_dot4_grouped_w8a8_to_spatial<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);else if(entry.researchFolded)rdna2_nr::research_folded_dot4_grouped_w8a8_to_spatial<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);else rdna2_nr::dot4_grouped_w8a8_to_spatial<32,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);
        }else{
            pack_rows<64>(input,m,k);
            if(entry.researchLong)rdna2_nr::research_long_dot4_grouped_w8a8_to_spatial<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);else if(entry.researchFolded)rdna2_nr::research_folded_dot4_grouped_w8a8_to_spatial<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);else rdna2_nr::dot4_grouped_w8a8_to_spatial<64,4><<<grid,threads,0,stream>>>(activation->data,entry.weight->data,activationScale->data,entry.scale->data,skip,cosine,output,m,k,n,publish,h,w,top,left,paddedWidth);
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
        packBlock=block;packRole=unsigned(Int8Role::FfnExpand);
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
    size_t reserved_bytes()const{size_t bytes=status->payload_bytes()+2*Int8GuardBytes;for(const auto&item:researchScaledWeights)bytes+=item.second->payload_bytes()+2*Int8GuardBytes;for(const auto&item:matrices)bytes+=item.second->reserved_bytes();for(const auto&item:branchProjects)bytes+=item.second->reserved_bytes();for(const auto&item:c32Attention)bytes+=item.second->reserved_bytes();if(activation)bytes+=activation->payload_bytes()+2*Int8GuardBytes;if(activationScale)bytes+=activationScale->payload_bytes()+2*Int8GuardBytes;if(handoffCodes)bytes+=handoffCodes->payload_bytes()+2*Int8GuardBytes;if(handoffScales)bytes+=handoffScales->payload_bytes()+2*Int8GuardBytes;if(windowHandoffCodes)bytes+=windowHandoffCodes->payload_bytes()+2*Int8GuardBytes;return bytes;}
    size_t invocations()const{return launches;}
    bool release()noexcept{bool ok=true;for(auto&item:c32Attention)ok=item.second->release()&&ok;c32Attention.clear();for(auto&item:branchProjects)ok=item.second->release()&&ok;branchProjects.clear();for(auto&item:matrices)ok=item.second->release()&&ok;matrices.clear();for(auto&item:researchScaledWeights)ok=item.second->release()&&ok;researchScaledWeights.clear();if(windowHandoffCodes)ok=windowHandoffCodes->release()&&ok;if(handoffScales)ok=handoffScales->release()&&ok;if(handoffCodes)ok=handoffCodes->release()&&ok;if(activationScale)ok=activationScale->release()&&ok;if(activation)ok=activation->release()&&ok;ok=status->release()&&ok;return ok;}
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
    if(family==0){HalfPtr wa_owner,wb_owner;auto wa=weight_ptr(w.a,wa_owner),wb=weight_ptr(w.b,wb_owner);const bool inspectHidden=trace_requested(trace_block)||(active_graph_profile&&active_graph_profile->activationObserver)||int8_capture_requested(trace_block);const bool compactDisabled=active_graph_profile&&active_graph_profile->disableCompactC32;const auto*compact=active_compact_c32&&!compactDisabled?active_compact_c32->find(unsigned(trace_block)):nullptr;const unsigned c32Group=active_grouped_int8&&(true)?32:0;if(compact&&!inspectHidden){launch_fused_c32_ffn_compact(x,compact->expand,compact->contract,compact->cosine,padded_out->ptr(),padded,compact->hidden);}else if(c32Group&&active_grouped_int8&&!inspectHidden){active_grouped_int8->launch_c32_ffn(unsigned(trace_block),x,wa,wb,cosine,padded_out->ptr(),unsigned(padded),c32Group);}else if(fused_c32_enabled()&&!inspectHidden){launch_fused_c32_ffn(x,wa,wb,cosine,padded_out->ptr(),padded,packed_c32_fp16_enabled());}else{Buffer<__half>hidden(size_t(padded)*128);if(trace_requested(trace_block)){Buffer<float>expand(size_t(padded)*128),contract(size_t(padded)*c);launch_gemm(x,wa,expand.ptr(),padded,c,128);gate_publish<<<(hidden.count+255)/256,256,0,current_stream()>>>(expand.ptr(),hidden.ptr(),hidden.count);sync_gpu();observe_activation(trace_block,"ffn_contract_input",hidden.ptr(),padded,128);launch_gemm(hidden.ptr(),wb,contract.ptr(),padded,128,c);residual_h<<<(padded_out->count+255)/256,256,0,current_stream()>>>(contract.ptr(),x,cosine,padded_out->ptr(),padded_out->count,c,false);sync_gpu();trace_stage(trace_block,"ffn_expand",expand);trace_stage(trace_block,"ffn_hidden",hidden);trace_stage(trace_block,"ffn_contract",contract);}else{launch_gemm_gate_publish(x,wa,hidden.ptr(),padded,c,128);observe_activation(trace_block,"ffn_contract_input",hidden.ptr(),padded,128);launch_gemm_residual(hidden.ptr(),wb,x,cosine,padded_out->ptr(),padded,128,c,false);}capture_int8_activation(trace_block,"ffn_hidden",hidden,padded,128);}}
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
    auto origin=origin_for(block);int top=-origin.first,left=-origin.second,ph=(h+top+7)/8*8,pw=(w+left+7)/8*8,nwin=(ph/8)*(pw/8),all=nwin*64;
    HalfPtr handoffWeightOwner;
    if(active_grouped_int8){
        auto handoffWeight=weight_ptr(weights.qkv,handoffWeightOwner);
        const bool allowed=family!=0&&weights.attentionChannels==c&&fused_qkv_normalize_enabled()&&!trace_requested(block)&&
            !(active_graph_profile&&active_graph_profile->activationObserver);
        active_grouped_int8->arm_window_handoff(unsigned(block),handoffWeight,unsigned(c),unsigned(h),unsigned(w),
            unsigned(top),unsigned(left),unsigned(pw),unsigned(all),allowed);
    }
    const char*ffnStage=family==0?"c32_ffn":"window_ffn";
    const char*ffnFamilyStage=family==0?"c32_ffn":family==1?(c==64?"c64_ffn":c==128?"c128_ffn":"c256_ffn"):"c512_ffn";
    stage_event(ffnStage,true);stage_event(ffnFamilyStage,true);auto ffn=ffn_device(input,h*w,c,family,weights,block);stage_event(ffnFamilyStage,false);stage_event(ffnStage,false);
    observe_activation(block,"qkv_input",ffn->ptr(),h*w,c);trace_stage(block,"ffn",*ffn);

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
