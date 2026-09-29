#pragma once

#include "nr_window_runtime.h"

#include <cstdint>

namespace rdna2_nr {

__global__ void quantize_rows_w8a8(const __half* input, std::int8_t* output,
                                   float* scale, float* inverse,
                                   unsigned rows, unsigned columns,
                                   unsigned* status) {
    __shared__ float maximum[128];
    const unsigned row=blockIdx.x,lane=threadIdx.x;
    if(row>=rows)return;
    float value=0;
    for(unsigned column=lane;column<columns;column+=128){
        const float x=__half2float(input[size_t(row)*columns+column]);
        if(!isfinite(x))atomicOr(status,1u);
        value=fmaxf(value,fabsf(x));
    }
    maximum[lane]=value;__syncthreads();
    for(unsigned stride=64;stride;stride>>=1){
        if(lane<stride)maximum[lane]=fmaxf(maximum[lane],maximum[lane+stride]);
        __syncthreads();
    }
    const float s=maximum[0]==0?1.f:maximum[0]/127.f;
    const float inv=maximum[0]==0?1.f:127.f/maximum[0];
    if(lane==0){scale[row]=s;if(inverse)inverse[row]=inv;}
    for(unsigned column=lane;column<columns;column+=128){
        int q=__float2int_rn(__half2float(input[size_t(row)*columns+column])*inv);
        q=max(-127,min(127,q));
        output[size_t(row)*columns+column]=std::int8_t(q);
    }
}

__global__ void quantize_columns_w8a8(const __half* input, std::int8_t* output,
                                      float* scale, float* inverse,
                                      unsigned k, unsigned n,
                                      unsigned* status) {
    __shared__ float maximum[128];
    const unsigned column=blockIdx.x,lane=threadIdx.x;
    if(column>=n)return;
    float value=0;
    for(unsigned row=lane;row<k;row+=128){
        const float x=__half2float(input[size_t(row)*n+column]);
        if(!isfinite(x))atomicOr(status,1u);
        value=fmaxf(value,fabsf(x));
    }
    maximum[lane]=value;__syncthreads();
    for(unsigned stride=64;stride;stride>>=1){
        if(lane<stride)maximum[lane]=fmaxf(maximum[lane],maximum[lane+stride]);
        __syncthreads();
    }
    const float s=maximum[0]==0?1.f:maximum[0]/127.f;
    const float inv=maximum[0]==0?1.f:127.f/maximum[0];
    if(lane==0){scale[column]=s;if(inverse)inverse[column]=inv;}
    for(unsigned row=lane;row<k;row+=128){
        int q=__float2int_rn(__half2float(input[size_t(row)*n+column])*inv);
        q=max(-127,min(127,q));
        output[size_t(column)*k+row]=std::int8_t(q);
    }
}

// Grouped activation scales are laid out as [row, K / GroupSize]. One
// 256-thread block handles a row in batches of contiguous groups, keeping
// source loads and packed output stores coalesced.
template<unsigned GroupSize>
__global__ void quantize_row_groups_w8a8(const __half* input,
                                          std::int8_t* output,
                                          float* scale, float* inverse,
                                          unsigned rows, unsigned k,
                                          unsigned* status) {
    static_assert(GroupSize==32||GroupSize==64||GroupSize==128,"supported W8A8 group size");
    constexpr unsigned GroupsPerBatch=256/GroupSize;
    __shared__ float values[256];
    __shared__ float groupScale[GroupsPerBatch];
    __shared__ float groupInverse[GroupsPerBatch];
    const unsigned row=blockIdx.x,tid=threadIdx.x,lane=tid%GroupSize,
        localGroup=tid/GroupSize,groups=k/GroupSize;
    if(row>=rows)return;
    for(unsigned baseGroup=0;baseGroup<groups;baseGroup+=GroupsPerBatch){
        const unsigned group=baseGroup+localGroup,column=group*GroupSize+lane;
        float value=0;
        if(group<groups){
            const float x=__half2float(input[size_t(row)*k+column]);
            if(!isfinite(x))atomicOr(status,1u);
            value=fabsf(x);
        }
        values[tid]=value;__syncthreads();
        for(unsigned stride=GroupSize/2;stride;stride>>=1){
            if(lane<stride)values[tid]=fmaxf(values[tid],values[tid+stride]);
            __syncthreads();
        }
        if(lane==0&&group<groups){
            const float maximum=values[tid];
            groupScale[localGroup]=maximum==0?1.f:maximum/127.f;
            groupInverse[localGroup]=maximum==0?1.f:127.f/maximum;
            scale[size_t(row)*groups+group]=groupScale[localGroup];
            if(inverse)inverse[size_t(row)*groups+group]=groupInverse[localGroup];
        }
        __syncthreads();
        if(group<groups){
            int q=__float2int_rn(__half2float(input[size_t(row)*k+column])*
                                 groupInverse[localGroup]);
            q=max(-127,min(127,q));
            output[size_t(row)*k+column]=std::int8_t(q);
        }
        __syncthreads();
    }
}

// Quantize rows in 8x8-window order while reading the source directly from
// spatial HWC storage. Padded window tokens are exact zero rows. This avoids a
// full FP16 partition buffer before window QKV.
template<unsigned GroupSize>
__global__ void quantize_window_groups_w8a8(const __half* input,
                                             std::int8_t* output,
                                             float* scale,
                                             unsigned rows,unsigned k,
                                             unsigned h,unsigned w,
                                             unsigned top,unsigned left,
                                             unsigned paddedWidth,
                                             unsigned* status) {
    static_assert(GroupSize==32||GroupSize==64,"supported W8A8 group size");
    constexpr unsigned GroupsPerBatch=256/GroupSize;
    __shared__ float values[256];
    __shared__ float groupScale[GroupsPerBatch];
    __shared__ float groupInverse[GroupsPerBatch];
    const unsigned row=blockIdx.x,tid=threadIdx.x,lane=tid%GroupSize,
        localGroup=tid/GroupSize,groups=k/GroupSize;
    if(row>=rows)return;
    unsigned element=row,ix=element%8;element/=8;const unsigned iy=element%8;
    element/=8;const unsigned windowsPerRow=paddedWidth/8;
    const unsigned wx=element%windowsPerRow,wy=element/windowsPerRow;
    const int y=int(wy*8+iy)-int(top),x=int(wx*8+ix)-int(left);
    const bool valid=y>=0&&y<int(h)&&x>=0&&x<int(w);
    for(unsigned baseGroup=0;baseGroup<groups;baseGroup+=GroupsPerBatch){
        const unsigned group=baseGroup+localGroup,column=group*GroupSize+lane;
        float source=0;
        if(group<groups&&valid){
            source=__half2float(input[(size_t(y)*w+x)*k+column]);
            if(!isfinite(source))atomicOr(status,1u);
        }
        values[tid]=fabsf(source);__syncthreads();
        for(unsigned stride=GroupSize/2;stride;stride>>=1){
            if(lane<stride)values[tid]=fmaxf(values[tid],values[tid+stride]);
            __syncthreads();
        }
        if(lane==0&&group<groups){
            const float maximum=values[tid];
            groupScale[localGroup]=maximum==0?1.f:maximum/127.f;
            groupInverse[localGroup]=maximum==0?1.f:127.f/maximum;
            scale[size_t(row)*groups+group]=groupScale[localGroup];
        }
        __syncthreads();
        if(group<groups){
            int q=__float2int_rn(source*groupInverse[localGroup]);
            q=max(-127,min(127,q));
            output[size_t(row)*k+column]=std::int8_t(q);
        }
        __syncthreads();
    }
}

// One 32-lane subgroup per scale group, four groups per workgroup. Keeping
// the reduction within a subgroup avoids row-wide barriers and idle lanes
// for K=64/128. Group64 reads adjacent half2 and stores paired INT8 codes;
// its maximum and rounding match the original scale domain exactly.
template<unsigned GroupSize,bool Window=false>
__global__ void quantize_groups_wave_w8a8(const __half* input,
    std::int8_t* output,float* scale,float* inverse,unsigned rows,unsigned k,
    unsigned* status,unsigned h=0,unsigned w=0,unsigned top=0,unsigned left=0,
    unsigned paddedWidth=0) {
    static_assert(GroupSize==32||GroupSize==64||GroupSize==128,"supported W8A8 group size");
    const unsigned groupIndex=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    const unsigned groups=k/GroupSize;
    if(groupIndex>=rows*groups)return;
    const unsigned row=groupIndex/groups,group=groupIndex%groups;
    size_t sourceRow=row;
    bool valid=true;
    if constexpr(Window){
        const unsigned token=row%64,window=row/64;
        const int y=int(window/(paddedWidth/8)*8+token/8)-int(top);
        const int x=int(window%(paddedWidth/8)*8+token%8)-int(left);
        valid=y>=0&&y<int(h)&&x>=0&&x<int(w);
        if(valid)sourceRow=size_t(y)*w+x;
    }
    const unsigned column=group*GroupSize+(GroupSize==64?2*lane:lane);
    float a=0.f,b=0.f,c=0.f,d=0.f;
    if constexpr(GroupSize==64){
        const __half2 pair=valid?*reinterpret_cast<const __half2*>(input+sourceRow*k+column):__float2half2_rn(0.f);
        a=__low2float(pair);b=__high2float(pair);
    }else{
        a=valid?__half2float(input[sourceRow*k+column]):0.f;
        if constexpr(GroupSize>=64)b=valid?__half2float(input[sourceRow*k+column+32]):0.f;
    }
    if constexpr(GroupSize==128){
        c=valid?__half2float(input[sourceRow*k+column+64]):0.f;
        d=valid?__half2float(input[sourceRow*k+column+96]):0.f;
    }
    if(!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(d))atomicOr(status,1u);
    // Old row reducer starts with zero for inactive lanes; fmax also preserves
    // its NaN handling while status reports the invalid input.
    float maximum=fmaxf(fmaxf(fabsf(a),fabsf(b)),fmaxf(fabsf(c),fabsf(d)));
    for(unsigned stride=16;stride;stride>>=1)
        maximum=fmaxf(maximum,__shfl_down(maximum,stride,32));
    maximum=__shfl(maximum,0,32);
    const float s=maximum==0?1.f:maximum/127.f;
    const float inv=maximum==0?1.f:127.f/maximum;
    if(lane==0){scale[groupIndex]=s;if(inverse)inverse[groupIndex]=inv;}
    if constexpr(GroupSize==64){
        const unsigned lo=unsigned(std::uint8_t(max(-127,min(127,__float2int_rn(a*inv)))));
        const unsigned hi=unsigned(std::uint8_t(max(-127,min(127,__float2int_rn(b*inv)))));
        *reinterpret_cast<std::uint16_t*>(output+size_t(row)*k+column)=std::uint16_t(lo|(hi<<8));
    }else{
        output[size_t(row)*k+column]=std::int8_t(max(-127,min(127,__float2int_rn(a*inv))));
        if constexpr(GroupSize>=64)
            output[size_t(row)*k+column+32]=std::int8_t(max(-127,min(127,__float2int_rn(b*inv))));
    }
    if constexpr(GroupSize==128){
        output[size_t(row)*k+column+64]=std::int8_t(max(-127,min(127,__float2int_rn(c*inv))));
        output[size_t(row)*k+column+96]=std::int8_t(max(-127,min(127,__float2int_rn(d*inv))));
    }
}

// Input weights use [K,N]; packed weights use [N,K]. Scales use [group,N],
// making the epilogue's adjacent-column scale loads contiguous.
template<unsigned GroupSize,bool BlockedLayout=false>
__global__ void quantize_weight_groups_w8a8(const __half* input,
                                             std::int8_t* output,
                                             float* scale, float* inverse,
                                             unsigned k, unsigned n,
                                             unsigned* status) {
    static_assert(GroupSize==16||GroupSize==32||GroupSize==64||GroupSize==128,"supported W8A8 weight group size");
    const unsigned column=blockIdx.x*blockDim.x+threadIdx.x,
        group=blockIdx.y,groups=k/GroupSize,base=group*GroupSize;
    if(column>=n||group>=groups)return;
    float maximum=0;
    for(unsigned q=0;q<GroupSize;++q){
        const float x=__half2float(input[size_t(base+q)*n+column]);
        if(!isfinite(x))atomicOr(status,1u);
        maximum=fmaxf(maximum,fabsf(x));
    }
    const float s=maximum==0?1.f:maximum/127.f;
    const float inv=maximum==0?1.f:127.f/maximum;
    scale[size_t(group)*n+column]=s;
    if(inverse)inverse[size_t(group)*n+column]=inv;
    for(unsigned q=0;q<GroupSize;++q){
        int value=__float2int_rn(__half2float(input[size_t(base+q)*n+column])*inv);
        value=max(-127,min(127,value));
        const unsigned d=base+q;
        const size_t offset=BlockedLayout?(size_t(column/32)*(k/4)*32+size_t(d/4)*32+column%32)*4+d%4:size_t(column)*k+d;
        output[offset]=std::int8_t(value);
    }
}

// Pack a small family of independent [K,N] matrices without expanding it into
// a block-sparse dense matrix. Input is [batch,K,N], packed output is
// [batch,N,K], and scales are [batch,K/GroupSize,N].
template<unsigned GroupSize>
__global__ void quantize_batched_weight_groups_w8a8(const __half* input,
                                                     std::int8_t* output,
                                                     float* scale,
                                                     unsigned batches,
                                                     unsigned k,unsigned n,
                                                     unsigned* status) {
    static_assert(GroupSize==32||GroupSize==64,"supported W8A8 group size");
    const unsigned column=blockIdx.x*blockDim.x+threadIdx.x,
        group=blockIdx.y,batch=blockIdx.z,groups=k/GroupSize,base=group*GroupSize;
    if(column>=n||group>=groups||batch>=batches)return;
    const size_t inputBase=size_t(batch)*k*n,outputBase=size_t(batch)*n*k,
        scaleBase=size_t(batch)*groups*n;
    float maximum=0;
    for(unsigned q=0;q<GroupSize;++q){
        const float x=__half2float(input[inputBase+size_t(base+q)*n+column]);
        if(!isfinite(x))atomicOr(status,1u);
        maximum=fmaxf(maximum,fabsf(x));
    }
    const float s=maximum==0?1.f:maximum/127.f;
    const float inv=maximum==0?1.f:127.f/maximum;
    scale[scaleBase+size_t(group)*n+column]=s;
    for(unsigned q=0;q<GroupSize;++q){
        int value=__float2int_rn(__half2float(input[inputBase+size_t(base+q)*n+column])*inv);
        value=max(-127,min(127,value));
        output[outputBase+size_t(column)*k+base+q]=std::int8_t(value);
    }
}

__global__ void dot4_i32_w8a8(const std::int8_t* a,const std::int8_t* w,
                              std::int32_t* out,unsigned m,unsigned k,unsigned n){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=m*n)return;
    const unsigned row=i/n,column=i%n;
    int sum=0;
    for(unsigned p=0;p<k;p+=4){
        const char4 av=*reinterpret_cast<const char4*>(a+size_t(row)*k+p);
        const char4 bv=*reinterpret_cast<const char4*>(w+size_t(column)*k+p);
        sum=amd_mixed_dot(av,bv,sum,false);
    }
    out[i]=sum;
}

template<int RowsPerThread>
__global__ void dot4_gate_publish_w8a8(const std::int8_t* a,const std::int8_t* w,
                                      const float* scaleA,const float* scaleW,
                                      __half* out,unsigned m,unsigned k,unsigned n){
    constexpr unsigned RowThreads=16/RowsPerThread,TileK=32,Groups=TileK/4;
    constexpr unsigned Threads=32*RowThreads;
    __shared__ std::int32_t tileA[16][Groups];
    __shared__ std::int32_t tileW[32][Groups];
    const unsigned local=threadIdx.y*32+threadIdx.x;
    const unsigned column=blockIdx.x*32+threadIdx.x;
    std::int32_t sums[RowsPerThread]={};
    for(unsigned base=0;base<k;base+=TileK){
        for(unsigned index=local;index<16*Groups;index+=Threads){
            const unsigned r=index/Groups,g=index%Groups,row=blockIdx.y*16+r;
            tileA[r][g]=(row<m&&base+g*4<k)?
                *reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+base+g*4):0;
        }
        for(unsigned index=local;index<32*Groups;index+=Threads){
            const unsigned c=index/Groups,g=index%Groups,globalColumn=blockIdx.x*32+c;
            tileW[c][g]=(globalColumn<n&&base+g*4<k)?
                *reinterpret_cast<const std::int32_t*>(w+size_t(globalColumn)*k+base+g*4):0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned g=0;g<Groups;++g){
            const char4 bv=*reinterpret_cast<const char4*>(&tileW[threadIdx.x][g]);
            #pragma unroll
            for(unsigned r=0;r<RowsPerThread;++r){
                const unsigned tileRow=threadIdx.y+r*RowThreads;
                const char4 av=*reinterpret_cast<const char4*>(&tileA[tileRow][g]);
                sums[r]=amd_mixed_dot(av,bv,sums[r],false);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned row=blockIdx.y*16+threadIdx.y+r*RowThreads;
        if(row<m&&column<n){
            const float combined=scaleA[row]*scaleW[column];
            out[size_t(row)*n+column]=pubh(ffn_gate(__float2half(float(sums[r])*combined)));
        }
    }
}

// Epilogue: 0 = FP16, 1 = gate/publication, 2 = publication,
// 3 = residual (optionally published), 4 = gate without publication. The half
// boundaries match the FP16 path.
template<unsigned GroupSize,int RowsPerThread,int Epilogue,unsigned TileRows=16,unsigned TileColumns=32,unsigned TileK=32,
         int PackedWeights=0,int DotUnroll=TileK/4,bool DirectWeights=false>
__global__ void dot4_grouped_w8a8(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,
    const __half* skip,const __half* cosine,__half* out,
    unsigned m,unsigned k,unsigned n,bool publish=false) {
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
    const unsigned scaleGroups=k/GroupSize;
    float totals[RowsPerThread]={};
    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[RowsPerThread]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=group*GroupSize+tile;
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
                const float combined=scaleA[size_t(row)*scaleGroups+group]*
                                     scaleW[size_t(group)*n+column];
                totals[r]+=float(partial[r])*combined;
            }
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
        }
    }
}

// Experimental group-64 pipeline for the 32x32x64 blocked-weight path. Each
// thread fetches its part of the next activation/weight tile into registers
// before consuming the current LDS tile. This overlaps global-memory latency
// with DOT4 work without changing the scale-group reduction order. Keep this
// out of production dispatch until the full-chain audit accepts it.
template<int Epilogue,int DotUnroll=4>
__global__ void dot4_grouped_w8a8_prefetch64(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,
    const __half* skip,const __half* cosine,__half* out,
    unsigned m,unsigned k,unsigned n,bool publish=false) {
    constexpr unsigned GroupSize=64,TileRows=32,TileColumns=32,RowsPerThread=4;
    constexpr unsigned RowThreads=TileRows/RowsPerThread,DotGroups=GroupSize/4;
    constexpr unsigned Threads=TileColumns*RowThreads,TileValues=TileRows*DotGroups;
    static_assert(TileValues==2*Threads,"prefetch mapping assumes two values per thread");
    __shared__ std::int32_t tileA[TileRows][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][TileColumns+1];
    const unsigned local=threadIdx.y*TileColumns+threadIdx.x;
    const unsigned column=blockIdx.x*TileColumns+threadIdx.x;
    const unsigned scaleGroups=k/GroupSize;
    std::int32_t nextA[2],nextW[2];
    float totals[RowsPerThread]={};

    // The blocked layout is [N/32,K/4,32,4]. Loads for both operands are
    // distributed exactly as in the accepted kernel, but remain in registers
    // until the LDS buffer is available.
    #pragma unroll
    for(unsigned slot=0;slot<2;++slot){
        const unsigned index=local+slot*Threads;
        const unsigned r=index/DotGroups,g=index%DotGroups;
        const unsigned row=blockIdx.y*TileRows+r;
        nextA[slot]=(row<m)?*reinterpret_cast<const std::int32_t*>(
            a+size_t(row)*k+g*4):0;
        const unsigned c=index%TileColumns,wg=index/TileColumns;
        const unsigned globalColumn=blockIdx.x*TileColumns+c;
        const size_t offset=(size_t(globalColumn/32)*(k/4)*32+size_t(wg)*32+globalColumn%32)*4;
        nextW[slot]=(globalColumn<n)?*reinterpret_cast<const std::int32_t*>(w+offset):0;
    }
    #pragma unroll
    for(unsigned slot=0;slot<2;++slot){
        const unsigned index=local+slot*Threads;
        tileA[index/DotGroups][index%DotGroups]=nextA[slot];
        tileW[index/TileColumns][index%TileColumns]=nextW[slot];
    }
    __syncthreads();

    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[RowsPerThread]={};
        if(group+1<scaleGroups){
            const unsigned base=(group+1)*GroupSize;
            #pragma unroll
            for(unsigned slot=0;slot<2;++slot){
                const unsigned index=local+slot*Threads;
                const unsigned r=index/DotGroups,g=index%DotGroups;
                const unsigned row=blockIdx.y*TileRows+r;
                nextA[slot]=(row<m)?*reinterpret_cast<const std::int32_t*>(
                    a+size_t(row)*k+base+g*4):0;
                const unsigned c=index%TileColumns,wg=index/TileColumns;
                const unsigned globalColumn=blockIdx.x*TileColumns+c;
                const size_t offset=(size_t(globalColumn/32)*(k/4)*32+
                    size_t(base/4+wg)*32+globalColumn%32)*4;
                nextW[slot]=(globalColumn<n)?*reinterpret_cast<const std::int32_t*>(w+offset):0;
            }
        }
        #pragma unroll DotUnroll
        for(unsigned g=0;g<DotGroups;++g){
            const char4 bv=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
            #pragma unroll
            for(unsigned r=0;r<RowsPerThread;++r){
                const unsigned tileRow=threadIdx.y+r*RowThreads;
                const char4 av=*reinterpret_cast<const char4*>(&tileA[tileRow][g]);
                partial[r]=amd_mixed_dot(av,bv,partial[r],false);
            }
        }
        #pragma unroll
        for(unsigned r=0;r<RowsPerThread;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<m&&column<n){
                const float combined=scaleA[size_t(row)*scaleGroups+group]*
                                     scaleW[size_t(group)*n+column];
                totals[r]+=float(partial[r])*combined;
            }
        }
        __syncthreads();
        if(group+1<scaleGroups){
            #pragma unroll
            for(unsigned slot=0;slot<2;++slot){
                const unsigned index=local+slot*Threads;
                tileA[index/DotGroups][index%DotGroups]=nextA[slot];
                tileW[index/TileColumns][index%TileColumns]=nextW[slot];
            }
        }
        __syncthreads();
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
        }
    }
}

// Two 32-column output panels per lane.  Compared with a 256-thread 64-column
// workgroup this keeps the proven wave32 row mapping and spends registers,
// rather than another wave, to reuse the same activation tile.
template<unsigned GroupSize,int RowsPerThread,int Epilogue,unsigned TileRows=16,unsigned TileK=64>
__global__ void dot4_grouped_w8a8_2col(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,
    const __half* skip,const __half* cosine,__half* out,
    unsigned m,unsigned k,unsigned n,bool publish=false) {
    static_assert(GroupSize==64,"two-column kernel is screened for group64");
    static_assert(GroupSize%TileK==0&&TileK%4==0,"tile must divide scale group");
    static_assert(TileRows%RowsPerThread==0,"invalid output tile");
    constexpr unsigned RowThreads=TileRows/RowsPerThread,DotGroups=TileK/4;
    constexpr unsigned Threads=32*RowThreads,TileColumns=64;
    __shared__ std::int32_t tileA[TileRows][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][TileColumns+1];
    const unsigned local=threadIdx.y*32+threadIdx.x;
    const unsigned firstColumn=blockIdx.x*TileColumns+threadIdx.x;
    const unsigned scaleGroups=k/GroupSize;
    float totals[2][RowsPerThread]={};
    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[2][RowsPerThread]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=group*GroupSize+tile;
            for(unsigned index=local;index<TileRows*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*TileRows+r;
                tileA[r][g]=(row<m)?
                    *reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+base+g*4):0;
            }
            for(unsigned index=local;index<TileColumns*DotGroups;index+=Threads){
                const unsigned c=index/DotGroups,g=index%DotGroups,column=blockIdx.x*TileColumns+c;
                tileW[g][c]=(column<n)?
                    *reinterpret_cast<const std::int32_t*>(w+size_t(column)*k+base+g*4):0;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned g=0;g<DotGroups;++g){
                const char4 bv0=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
                const char4 bv1=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x+32]);
                #pragma unroll
                for(unsigned r=0;r<RowsPerThread;++r){
                    const unsigned tileRow=threadIdx.y+r*RowThreads;
                    const char4 av=*reinterpret_cast<const char4*>(&tileA[tileRow][g]);
                    partial[0][r]=amd_mixed_dot(av,bv0,partial[0][r],false);
                    partial[1][r]=amd_mixed_dot(av,bv1,partial[1][r],false);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(unsigned panel=0;panel<2;++panel){
            const unsigned column=firstColumn+panel*32;
            #pragma unroll
            for(unsigned r=0;r<RowsPerThread;++r){
                const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
                if(row<m&&column<n){
                    const float combined=scaleA[size_t(row)*scaleGroups+group]*
                                         scaleW[size_t(group)*n+column];
                    totals[panel][r]+=float(partial[panel][r])*combined;
                }
            }
        }
    }
    #pragma unroll
    for(unsigned panel=0;panel<2;++panel){
        const unsigned column=firstColumn+panel*32;
        #pragma unroll
        for(unsigned r=0;r<RowsPerThread;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<m&&column<n){
                __half value=__float2half(totals[panel][r]);
                if constexpr(Epilogue==1)value=pubh(ffn_gate(value));
                if constexpr(Epilogue==2)value=pubh(value);
                if constexpr(Epilogue==3){
                    value=__hadd(value,__hmul(skip[size_t(row)*n+column],cosine[column]));
                    if(publish)value=pubh(value);
                }
                if constexpr(Epilogue==4)value=ffn_gate(value);
                out[size_t(row)*n+column]=value;
            }
        }
    }
}

// Grouped INT8 QKV with the original FP32 -> FP16 boundary and exact half
// normalization tree. One output tile is one complete 32-channel head, so the
// row-major Mx3C intermediate never has to leave the workgroup.
template<unsigned GroupSize,int RowsPerThread,unsigned TileK=32,int PackedWeights=0,int DotUnroll=TileK/4>
__global__ void dot4_grouped_w8a8_qkv_normalize(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,const __half* scale,__half* out,
    unsigned m,unsigned k,unsigned heads,unsigned tokensPerWindow) {
    static_assert(GroupSize==32||GroupSize==64||GroupSize==128,"supported W8A8 group size");
    static_assert(GroupSize%TileK==0&&TileK%4==0,"tile must divide scale group");
    constexpr unsigned TileRows=16,TileColumns=32;
    constexpr unsigned RowThreads=TileRows/RowsPerThread,DotGroups=TileK/4;
    constexpr unsigned Threads=TileColumns*RowThreads;
    __shared__ std::int32_t tileA[TileRows][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][TileColumns+1];
    __shared__ __half values[TileRows][TileColumns];
    __shared__ __half inverse[TileRows];
    const unsigned local=threadIdx.y*TileColumns+threadIdx.x;
    const unsigned chunk=blockIdx.x,column=chunk*TileColumns+threadIdx.x;
    const unsigned n=3*heads*TileColumns,scaleGroups=k/GroupSize;
    float totals[RowsPerThread]={};
    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[RowsPerThread]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=group*GroupSize+tile;
            for(unsigned index=local;index<TileRows*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*TileRows+r;
                tileA[r][g]=(row<m)?
                    *reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+base+g*4):0;
            }
            for(unsigned index=local;index<TileColumns*DotGroups;index+=Threads){
                const unsigned c=PackedWeights?index%TileColumns:index/DotGroups;
                const unsigned g=PackedWeights?index/TileColumns:index%DotGroups;
                const unsigned globalColumn=chunk*TileColumns+c;
                const size_t offset=PackedWeights==2?
                    (size_t(globalColumn/32)*(k/4)*32+size_t(base/4+g)*32+globalColumn%32)*4:
                    size_t(globalColumn)*k+base+g*4;
                tileW[g][c]=*reinterpret_cast<const std::int32_t*>(
                    w+offset);
            }
            __syncthreads();
            #pragma unroll DotUnroll
            for(unsigned g=0;g<DotGroups;++g){
                const char4 bv=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
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
            if(row<m){
                const float combined=scaleA[size_t(row)*scaleGroups+group]*
                                     scaleW[size_t(group)*n+column];
                totals[r]+=float(partial[r])*combined;
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
        values[tileRow][threadIdx.x]=row<m?__float2half(totals[r]):__float2half(0);
    }
    __syncthreads();
    const unsigned kind=chunk/heads,head=chunk%heads;
    if(threadIdx.x==0){
        #pragma unroll
        for(unsigned r=0;r<RowsPerThread;++r){
            const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
            __half inv=__float2half(1);
            if(row<m&&kind<2){
                __half part[4][2],pair[4][2];
                #pragma unroll
                for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){
                    const int q=lane*2+p;
                    const __half x0=values[tileRow][q],x1=values[tileRow][q+8];
                    const __half x2=values[tileRow][q+16],x3=values[tileRow][q+24];
                    const __half first=__hfma(x1,x1,__hmul(x0,x0));
                    const __half second=__hfma(x3,x3,__hmul(x2,x2));
                    part[lane][p]=__hadd(first,second);
                }
                #pragma unroll
                for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)
                    pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);
                __half norm=__hadd(__hadd(pair[0][0],pair[1][0]),
                                   __hadd(pair[0][1],pair[1][1]));
                norm=__float2half(fmaxf(__half2float(norm),0.00006198883056640625f));
                inv=__float2half(rsqrtf(__half2float(norm)));
            }
            inverse[tileRow]=inv;
        }
    }
    __syncthreads();
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
        if(row<m){
            __half value=values[tileRow][threadIdx.x];
            if(kind<2)value=__hmul(value,inverse[tileRow]);
            if(kind==0)value=__hmul(value,scale[head]);
            const unsigned window=row/tokensPerWindow,token=row%tokensPerWindow;
            const size_t target=(((size_t(window)*3+kind)*heads+head)*
                                 tokensPerWindow+token)*32+threadIdx.x;
            out[target]=pubh(value);
        }
    }
}

// Batched grouped GEMM for physical [row,batch,K] activations and a family of
// independent [batch,K,N] weights. blockIdx.z selects the matrix; no zeros or
// duplicated activations are introduced as a dense block-diagonal surrogate.
template<unsigned GroupSize,int RowsPerThread,int Epilogue>
__global__ void dot4_batched_grouped_w8a8(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,__half* out,
    unsigned rows,unsigned batches,unsigned k,unsigned n) {
    static_assert(GroupSize==32||GroupSize==64,"supported W8A8 group size");
    static_assert(Epilogue==2||Epilogue==4,"batched epilogue must match split FFN boundary");
    constexpr unsigned RowThreads=16/RowsPerThread,TileK=32,DotGroups=TileK/4;
    constexpr unsigned Threads=32*RowThreads;
    __shared__ std::int32_t tileA[16][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][33];
    const unsigned local=threadIdx.y*32+threadIdx.x,column=blockIdx.x*32+threadIdx.x,
        batch=blockIdx.z,scaleGroups=k/GroupSize;
    float totals[RowsPerThread]={};
    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[RowsPerThread]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=group*GroupSize+tile;
            for(unsigned index=local;index<16*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*16+r,
                    batchRow=row*batches+batch;
                tileA[r][g]=(row<rows)?
                    *reinterpret_cast<const std::int32_t*>(a+size_t(batchRow)*k+base+g*4):0;
            }
            for(unsigned index=local;index<32*DotGroups;index+=Threads){
                const unsigned c=index/DotGroups,g=index%DotGroups,globalColumn=blockIdx.x*32+c;
                tileW[g][c]=(globalColumn<n)?*reinterpret_cast<const std::int32_t*>(
                    w+size_t(batch)*n*k+size_t(globalColumn)*k+base+g*4):0;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned g=0;g<DotGroups;++g){
                const char4 bv=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
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
            const unsigned row=blockIdx.y*16+threadIdx.y+r*RowThreads,batchRow=row*batches+batch;
            if(row<rows&&column<n){
                const float combined=scaleA[size_t(batchRow)*scaleGroups+group]*
                    scaleW[(size_t(batch)*scaleGroups+group)*n+column];
                totals[r]+=float(partial[r])*combined;
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned row=blockIdx.y*16+threadIdx.y+r*RowThreads,batchRow=row*batches+batch;
        if(row<rows&&column<n){
            __half value=__float2half(totals[r]);
            if constexpr(Epilogue==2)value=pubh(value);
            if constexpr(Epilogue==4)value=ffn_gate(value);
            out[size_t(batchRow)*n+column]=value;
        }
    }
}

// The branched C64/C128/C256 expansion shares one [row,K] activation across a
// family of [K,N] matrices and emits [row,batch,N]. Activations and their scales
// are therefore read once per row rather than duplicated for every branch.
template<unsigned GroupSize,int RowsPerThread>
__global__ void dot4_broadcast_batched_gate_w8a8(
    const std::int8_t* a,const std::int8_t* w,
    const float* scaleA,const float* scaleW,__half* out,
    unsigned rows,unsigned batches,unsigned k,unsigned n) {
    static_assert(GroupSize==32||GroupSize==64,"supported W8A8 group size");
    constexpr unsigned RowThreads=16/RowsPerThread,TileK=32,DotGroups=TileK/4;
    constexpr unsigned Threads=32*RowThreads;
    __shared__ std::int32_t tileA[16][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][33];
    const unsigned local=threadIdx.y*32+threadIdx.x,column=blockIdx.x*32+threadIdx.x,
        batch=blockIdx.z,scaleGroups=k/GroupSize;
    float totals[RowsPerThread]={};
    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[RowsPerThread]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=group*GroupSize+tile;
            for(unsigned index=local;index<16*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*16+r;
                tileA[r][g]=(row<rows)?*reinterpret_cast<const std::int32_t*>(
                    a+size_t(row)*k+base+g*4):0;
            }
            for(unsigned index=local;index<32*DotGroups;index+=Threads){
                const unsigned c=index/DotGroups,g=index%DotGroups,globalColumn=blockIdx.x*32+c;
                tileW[g][c]=(globalColumn<n)?*reinterpret_cast<const std::int32_t*>(
                    w+size_t(batch)*n*k+size_t(globalColumn)*k+base+g*4):0;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned g=0;g<DotGroups;++g){
                const char4 bv=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
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
            const unsigned row=blockIdx.y*16+threadIdx.y+r*RowThreads;
            if(row<rows&&column<n){
                const float combined=scaleA[size_t(row)*scaleGroups+group]*
                    scaleW[(size_t(batch)*scaleGroups+group)*n+column];
                totals[r]+=float(partial[r])*combined;
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned row=blockIdx.y*16+threadIdx.y+r*RowThreads;
        if(row<rows&&column<n)
            out[(size_t(row)*batches+batch)*n+column]=pubh(ffn_gate(__float2half(totals[r])));
    }
}

// Resident producer-consumer path for the physical C64/C128/C256 FFN. The
// grouped W8A8 expansion is reconstructed in the same order as the standalone
// broadcast kernel, but its four gated 32-channel branches stay in LDS and are
// consumed immediately by the original FP16 branch projections and ordered
// merge. Only the merged C-channel tensor is published for the final dense
// projection.
__global__ void pack_branched_project_half2(const __half* input,__half2* output,
                                             unsigned batches) {
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=batches*16*32)return;
    const unsigned column=index%32,pair=(index/32)%16,batch=index/(16*32);
    const size_t source=(size_t(batch)*32+pair*2)*32+column;
    output[index]=__halves2half2(input[source],input[source+32]);
}

// Stage all four intact branches together, retaining group64 reconstruction
// and ordered FP16 branch merging. The projection reuses the weight tile storage.
template<unsigned GroupSize,int RowsPerThread,bool PackedProject=false>
__global__ void dot4_branched_staged_w8a8(
    const std::int8_t* a,const std::int8_t* expandWeight,
    const float* scaleA,const float* expandScale,
    const __half* projectWeight,__half* combined,
    unsigned rows,unsigned channels) {
    static_assert(GroupSize==64,"supported resident branch group size");
    constexpr unsigned TileRows=16,RowThreads=TileRows/RowsPerThread,
        TileK=64,DotGroups=TileK/4,Threads=32*RowThreads;
    __shared__ std::int32_t tileA[TileRows][DotGroups];
    __shared__ std::int32_t weightScratch[4*DotGroups*33];
    auto& tileW=*reinterpret_cast<std::int32_t (*)[4][DotGroups][33]>(weightScratch);
    __shared__ __half hidden[TileRows][4][32];
    using ProjectValue=typename std::conditional<PackedProject,__half2,__half>::type;
    constexpr unsigned ProjectRows=PackedProject?16:32;
    static_assert(PackedProject,"shared reuse expects packed projection");
    // The final matrix barrier ends all tileW reads before projection overwrites it.
    static_assert(4*ProjectRows*33*sizeof(ProjectValue)<=sizeof(weightScratch),"projection scratch size");
    auto& projection=*reinterpret_cast<ProjectValue (*)[4][ProjectRows][33]>(weightScratch);
    const unsigned local=threadIdx.y*32+threadIdx.x,outGroup=blockIdx.x,
        column=threadIdx.x,scaleGroups=channels/GroupSize,batchBase=outGroup*4;
    float totals[RowsPerThread][4]={};
    for(unsigned scaleGroup=0;scaleGroup<scaleGroups;++scaleGroup){
        std::int32_t partial[RowsPerThread][4]={};

        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=scaleGroup*GroupSize+tile;
            for(unsigned index=local;index<TileRows*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*TileRows+r;
                tileA[r][g]=(row<rows)?*reinterpret_cast<const std::int32_t*>(
                    a+size_t(row)*channels+base+g*4):0;
            }
            for(unsigned index=local;index<4*32*DotGroups;index+=Threads){
                const unsigned branch=index/(32*DotGroups),c=(index/DotGroups)%32,g=index%DotGroups;
                tileW[branch][g][c]=*reinterpret_cast<const std::int32_t*>(
                    expandWeight+size_t(batchBase+branch)*32*channels+size_t(c)*channels+base+g*4);
            }
            __syncthreads();
            #pragma unroll 4
            for(unsigned g=0;g<DotGroups;++g){
                char4 bv[4];
                #pragma unroll
                for(unsigned branch=0;branch<4;++branch)
                    bv[branch]=*reinterpret_cast<const char4*>(&tileW[branch][g][column]);
                #pragma unroll
                for(unsigned r=0;r<RowsPerThread;++r){
                    const unsigned tileRow=threadIdx.y+r*RowThreads;
                    const char4 av=*reinterpret_cast<const char4*>(&tileA[tileRow][g]);
                    #pragma unroll
                    for(unsigned branch=0;branch<4;++branch)
                        partial[r][branch]=amd_mixed_dot(av,bv[branch],partial[r][branch],false);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(unsigned r=0;r<RowsPerThread;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<rows){
                const float activationScale=scaleA[size_t(row)*scaleGroups+scaleGroup];
                #pragma unroll
                for(unsigned branch=0;branch<4;++branch){
                    const unsigned batch=batchBase+branch;
                    const float combinedScale=activationScale*
                        expandScale[(size_t(batch)*scaleGroups+scaleGroup)*32+column];
                    totals[r][branch]+=float(partial[r][branch])*combinedScale;
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
        #pragma unroll
        for(unsigned branch=0;branch<4;++branch)
            hidden[tileRow][branch][column]=row<rows?
                pubh(ffn_gate(__float2half(totals[r][branch]))):__float2half(0);
    }
    for(unsigned index=local;index<4*ProjectRows*32;index+=Threads){
        const unsigned branch=index/(ProjectRows*32),k=(index/32)%ProjectRows,c=index%32;
        if constexpr(PackedProject)
            projection[branch][k][c]=reinterpret_cast<const __half2*>(projectWeight)[
                (size_t(batchBase+branch)*16+k)*32+c];
        else projection[branch][k][c]=projectWeight[(size_t(batchBase+branch)*32+k)*32+c];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
        if(row<rows){
            __half merged=__float2half(0);
            #pragma unroll
            for(unsigned branch=0;branch<4;++branch){
                float sum=0;
                #pragma unroll
                for(unsigned k=0;k<32;k+=2){
                    __half2 weightPair;
                    if constexpr(PackedProject)weightPair=projection[branch][k/2][column];
                    else weightPair=__halves2half2(projection[branch][k][column],projection[branch][k+1][column]);
                    sum=amd_mixed_dot(__halves2half2(hidden[tileRow][branch][k],hidden[tileRow][branch][k+1]),
                        weightPair,sum,false);
                }
                merged=__hadd(merged,__float2half(sum));
            }
            combined[size_t(row)*channels+outGroup*32+column]=pubh(merged);
        }
    }
}


template<unsigned GroupSize,int RowsPerThread,bool PackedProject=false>
__global__ void dot4_branched_resident_w8a8(
    const std::int8_t* a,const std::int8_t* expandWeight,
    const float* scaleA,const float* expandScale,
    const __half* projectWeight,__half* combined,
    unsigned rows,unsigned channels) {
    static_assert(GroupSize==32||GroupSize==64,"supported resident branch group size");
    constexpr unsigned TileRows=16,RowThreads=TileRows/RowsPerThread,
        TileK=32,DotGroups=TileK/4,Threads=32*RowThreads;
    __shared__ std::int32_t tileA[TileRows][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][33];
    __shared__ __half hidden[TileRows][4][32];
    using ProjectValue=typename std::conditional<PackedProject,__half2,__half>::type;
    constexpr unsigned ProjectRows=PackedProject?16:32;
    __shared__ ProjectValue projection[4][ProjectRows][33];
    const unsigned local=threadIdx.y*32+threadIdx.x,outGroup=blockIdx.x,
        column=threadIdx.x,scaleGroups=channels/GroupSize,batchBase=outGroup*4;
    float totals[RowsPerThread][4]={};
    for(unsigned scaleGroup=0;scaleGroup<scaleGroups;++scaleGroup){
        std::int32_t partial[RowsPerThread][4]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=scaleGroup*GroupSize+tile;
            for(unsigned index=local;index<TileRows*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*TileRows+r;
                tileA[r][g]=(row<rows)?*reinterpret_cast<const std::int32_t*>(
                    a+size_t(row)*channels+base+g*4):0;
            }
            for(unsigned branch=0;branch<4;++branch){
                const unsigned batch=batchBase+branch;
                for(unsigned index=local;index<32*DotGroups;index+=Threads){
                    const unsigned c=index/DotGroups,g=index%DotGroups;
                    tileW[g][c]=*reinterpret_cast<const std::int32_t*>(
                        expandWeight+size_t(batch)*32*channels+size_t(c)*channels+base+g*4);
                }
                __syncthreads();
                #pragma unroll
                for(unsigned g=0;g<DotGroups;++g){
                    const char4 bv=*reinterpret_cast<const char4*>(&tileW[g][column]);
                    #pragma unroll
                    for(unsigned r=0;r<RowsPerThread;++r){
                        const unsigned tileRow=threadIdx.y+r*RowThreads;
                        const char4 av=*reinterpret_cast<const char4*>(&tileA[tileRow][g]);
                        partial[r][branch]=amd_mixed_dot(av,bv,partial[r][branch],false);
                    }
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for(unsigned r=0;r<RowsPerThread;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<rows){
                const float activationScale=scaleA[size_t(row)*scaleGroups+scaleGroup];
                #pragma unroll
                for(unsigned branch=0;branch<4;++branch){
                    const unsigned batch=batchBase+branch;
                    const float combinedScale=activationScale*
                        expandScale[(size_t(batch)*scaleGroups+scaleGroup)*32+column];
                    totals[r][branch]+=float(partial[r][branch])*combinedScale;
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
        #pragma unroll
        for(unsigned branch=0;branch<4;++branch)
            hidden[tileRow][branch][column]=row<rows?
                pubh(ffn_gate(__float2half(totals[r][branch]))):__float2half(0);
    }
    for(unsigned index=local;index<4*ProjectRows*32;index+=Threads){
        const unsigned branch=index/(ProjectRows*32),k=(index/32)%ProjectRows,c=index%32;
        if constexpr(PackedProject)
            projection[branch][k][c]=reinterpret_cast<const __half2*>(projectWeight)[
                (size_t(batchBase+branch)*16+k)*32+c];
        else projection[branch][k][c]=projectWeight[(size_t(batchBase+branch)*32+k)*32+c];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        const unsigned tileRow=threadIdx.y+r*RowThreads,row=blockIdx.y*TileRows+tileRow;
        if(row<rows){
            __half merged=__float2half(0);
            #pragma unroll
            for(unsigned branch=0;branch<4;++branch){
                float sum=0;
                #pragma unroll
                for(unsigned k=0;k<32;k+=2){
                    __half2 weightPair;
                    if constexpr(PackedProject)weightPair=projection[branch][k/2][column];
                    else weightPair=__halves2half2(projection[branch][k][column],projection[branch][k+1][column]);
                    sum=amd_mixed_dot(__halves2half2(hidden[tileRow][branch][k],hidden[tileRow][branch][k+1]),
                        weightPair,sum,false);
                }
                merged=__hadd(merged,__float2half(sum));
            }
            combined[size_t(row)*channels+outGroup*32+column]=pubh(merged);
        }
    }
}

// Window attention projection with the same grouped reconstruction order as
// dot4_grouped_w8a8, but its residual and destination remain in spatial HWC
// order. Invalid padded window tokens produce no store.
template<unsigned GroupSize,int RowsPerThread>
__global__ void dot4_grouped_w8a8_to_spatial(
    const std::int8_t* a,const std::int8_t* weights,
    const float* scaleA,const float* scaleW,
    const __half* skip,const __half* cosine,__half* out,
    unsigned m,unsigned k,unsigned n,bool publish,
    unsigned h,unsigned w,unsigned top,unsigned left,unsigned paddedWidth) {
    static_assert(GroupSize==32||GroupSize==64,"supported W8A8 group size");
    constexpr unsigned RowThreads=16/RowsPerThread,TileK=32,DotGroups=TileK/4;
    constexpr unsigned Threads=32*RowThreads;
    __shared__ std::int32_t tileA[16][DotGroups];
    __shared__ std::int32_t tileW[DotGroups][33];
    const unsigned local=threadIdx.y*32+threadIdx.x;
    const unsigned column=blockIdx.x*32+threadIdx.x;
    const unsigned scaleGroups=k/GroupSize;
    float totals[RowsPerThread]={};
    for(unsigned group=0;group<scaleGroups;++group){
        std::int32_t partial[RowsPerThread]={};
        for(unsigned tile=0;tile<GroupSize;tile+=TileK){
            const unsigned base=group*GroupSize+tile;
            for(unsigned index=local;index<16*DotGroups;index+=Threads){
                const unsigned r=index/DotGroups,g=index%DotGroups,row=blockIdx.y*16+r;
                tileA[r][g]=(row<m)?
                    *reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+base+g*4):0;
            }
            for(unsigned index=local;index<32*DotGroups;index+=Threads){
                const unsigned c=index/DotGroups,g=index%DotGroups,globalColumn=blockIdx.x*32+c;
                tileW[g][c]=(globalColumn<n)?
                    *reinterpret_cast<const std::int32_t*>(weights+size_t(globalColumn)*k+base+g*4):0;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned g=0;g<DotGroups;++g){
                const char4 bv=*reinterpret_cast<const char4*>(&tileW[g][threadIdx.x]);
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
            const unsigned row=blockIdx.y*16+threadIdx.y+r*RowThreads;
            if(row<m&&column<n){
                const float combined=scaleA[size_t(row)*scaleGroups+group]*
                                     scaleW[size_t(group)*n+column];
                totals[r]+=float(partial[r])*combined;
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<RowsPerThread;++r){
        unsigned element=blockIdx.y*16+threadIdx.y+r*RowThreads;
        if(element<m&&column<n){
            const unsigned ix=element%8;element/=8;const unsigned iy=element%8;
            element/=8;const unsigned wx=element%(paddedWidth/8),wy=element/(paddedWidth/8);
            const int y=int(wy*8+iy)-int(top),x=int(wx*8+ix)-int(left);
            if(y>=0&&y<int(h)&&x>=0&&x<int(w)){
                const size_t target=(size_t(y)*w+x)*n+column;
                __half value=__hadd(__float2half(totals[r]),__hmul(skip[target],cosine[column]));
                out[target]=publish?pubh(value):value;
            }
        }
    }
}

} // namespace rdna2_nr
