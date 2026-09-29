#pragma once

#include "../include/attention_probe_support.h"
#include "../include/ffn_candidate.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <type_traits>

constexpr int M=64;
__device__ __half pubh(__half x) {
#if !defined(NR_DIRECT_PUBLICATION) || NR_DIRECT_PUBLICATION
    return __ushort_as_half(rdna2_nr::publish_e4m3_half_bits(__half_as_ushort(x)));
#else
    return __ushort_as_half(rdna2_nr::e4m3fn_to_half_bits(rdna2_nr::half_bits_to_e4m3fn(__half_as_ushort(x))));
#endif
}
__global__ void gemm_hf(const __half* a,const __half* b,float* out,int m,int k,int n) {
    int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y*blockDim.y+threadIdx.y;
    if (row>=m || col>=n) return; float sum=0;
    for (int i=0;i<k;i+=2) sum=amd_mixed_dot(__halves2half2(a[row*k+i],a[row*k+i+1]),__halves2half2(b[i*n+col],b[(i+1)*n+col]),sum,false);
    out[row*n+col]=sum;
}
// Same 16-row by 32-column output tile and the same ascending DOT2 reduction
// as gemm_hf, with a 32-wide K tile shared by the 512 threads.  Reusing A
// across columns and B across rows removes the dominant redundant global loads
// without changing the arithmetic contract of an output element.
template<class Output,bool Publish=false>
__global__ void gemm_hf_tiled(const __half* a,const __half* b,Output* out,int m,int k,int n) {
    constexpr int TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*32+threadIdx.x;
    float sum=0;
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=512){
            const int r=index/TileK,q=index%TileK,globalRow=blockIdx.y*16+r;
            tileA[r][q]=(globalRow<m&&base+q<k)?a[globalRow*k+base+q]:__float2half(0);
        }
        for(int index=local;index<TileK*32;index+=512){
            const int q=index/32,c=index%32,globalCol=blockIdx.x*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        if(row<m&&col<n){
            const int span=min(TileK,k-base);
            for(int q=0;q<span;q+=2)
                sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                                  __halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]),sum,false);
        }
        __syncthreads();
    }
    if(row<m&&col<n){
        if constexpr(std::is_same_v<Output,float>)out[row*n+col]=sum;
        else {const __half value=__float2half(sum);out[row*n+col]=Publish?pubh(value):value;}
    }
}
// Block 0 has a fixed 16 -> 32 adapter.  Publish its FP32 accumulator directly
// to FP16 so the large-frame path does not materialize and reread an Mx32 FP32
// intermediate.  DOT2 visits K in the same ascending order as gemm_hf_tiled.
__global__ void adapter_16x32_half(const __half* a,const __half* b,__half* out,int m) {
    __shared__ __half tileA[16][16];
    __shared__ __half tileB[16][32];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y,col=threadIdx.x;
    if(local<16*16){
        const int r=local/16,q=local%16,globalRow=blockIdx.y*16+r;
        tileA[r][q]=globalRow<m?a[globalRow*16+q]:__float2half(0);
    }
    tileB[local/32][local%32]=b[(local/32)*32+local%32];
    __syncthreads();
    if(row<m){
        float sum=0;
        for(int q=0;q<16;q+=2)
            sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                              __halves2half2(tileB[q][col],tileB[q+1][col]),sum,false);
        out[row*32+col]=__float2half(sum);
    }
}
__global__ void gemm_hf_tiled_gate_publish(const __half* a,const __half* b,__half* out,
                                            int m,int k,int n) {
    constexpr int TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*32+threadIdx.x;
    float sum=0;
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=512){
            const int r=index/TileK,q=index%TileK,globalRow=blockIdx.y*16+r;
            tileA[r][q]=(globalRow<m&&base+q<k)?a[globalRow*k+base+q]:__float2half(0);
        }
        for(int index=local;index<TileK*32;index+=512){
            const int q=index/32,c=index%32,globalCol=blockIdx.x*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        if(row<m&&col<n){
            const int span=min(TileK,k-base);
            for(int q=0;q<span;q+=2)
                sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                                  __halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]),sum,false);
        }
        __syncthreads();
    }
    if(row<m&&col<n)out[row*n+col]=pubh(rdna2_nr::ffn_gate(__float2half(sum)));
}
__global__ void gemm_hf_tiled_publish(const __half* a,const __half* b,__half* out,
                                      int m,int k,int n) {
    constexpr int TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*32+threadIdx.x;
    float sum=0;
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=512){
            const int r=index/TileK,q=index%TileK,globalRow=blockIdx.y*16+r;
            tileA[r][q]=(globalRow<m&&base+q<k)?a[globalRow*k+base+q]:__float2half(0);
        }
        for(int index=local;index<TileK*32;index+=512){
            const int q=index/32,c=index%32,globalCol=blockIdx.x*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        if(row<m&&col<n){
            const int span=min(TileK,k-base);
            for(int q=0;q<span;q+=2)
                sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                                  __halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]),sum,false);
        }
        __syncthreads();
    }
    if(row<m&&col<n)out[row*n+col]=pubh(__float2half(sum));
}
// Four (or two) rows per lane reuse the B operand in registers. The output
// tile stays 16x32 and every accumulator still visits DOT2 pairs in ascending
// K order. Epilogue: 0 = FP16, 1 = gate/publication, 2 = publication,
// 3 = residual (optionally published). No half arithmetic boundary is removed.
template<int RowsPerThread,int Epilogue>
__global__ void gemm_hf_register(const __half* a,const __half* b,const __half* skip,
                                const __half* cosine,__half* out,int m,int k,int n,bool publish) {
    constexpr int RowThreads=16/RowsPerThread, Threads=32*RowThreads, TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    const int local=threadIdx.y*32+threadIdx.x,col=blockIdx.x*32+threadIdx.x;
    float sums[RowsPerThread]={};
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=Threads){
            const int r=index/TileK,q=index%TileK,row=blockIdx.y*16+r;
            tileA[r][q]=(row<m&&base+q<k)?a[row*k+base+q]:__float2half(0);
        }
        for(int index=local;index<TileK*32;index+=Threads){
            const int q=index/32,c=index%32,globalCol=blockIdx.x*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        const int span=min(TileK,k-base);
        for(int q=0;q<span;q+=2){
            const __half2 bv=__halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]);
            #pragma unroll
            for(int r=0;r<RowsPerThread;++r){
                const int tileRow=threadIdx.y+r*RowThreads;
                sums[r]=amd_mixed_dot(__halves2half2(tileA[tileRow][q],tileA[tileRow][q+1]),bv,sums[r],false);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RowsPerThread;++r){
        const int row=blockIdx.y*16+threadIdx.y+r*RowThreads;
        if(row<m&&col<n){
            __half value=__float2half(sums[r]);
            if constexpr(Epilogue==1)value=pubh(rdna2_nr::ffn_gate(value));
            if constexpr(Epilogue==2)value=pubh(value);
            if constexpr(Epilogue==3){
                value=__hadd(value,__hmul(skip[row*n+col],cosine[col]));
                if(publish)value=pubh(value);
            }
            out[row*n+col]=value;
        }
    }
}
// C32 is dominated by moving the 128-channel FFN hidden tensor rather than by
// arithmetic.  One workgroup keeps a 16-row hidden tile in LDS and performs
// expand -> original gate/publication -> contract -> residual without a global
// hidden allocation.  Every dot product visits K in the same ascending DOT2
// order as the two register-tiled reference kernels.
template<int TileRows=16,int RowsPerThread=4>
__global__ void fused_c32_ffn(const __half* input,const __half* expandWeight,
                              const __half* contractWeight,const __half* cosine,
                              __half* output,int rows) {
    constexpr int Channels=32,Hidden=128;
    static_assert(TileRows%RowsPerThread==0,"invalid C32 row tile");
    constexpr int RowThreads=TileRows/RowsPerThread,Threads=Channels*RowThreads;
    __shared__ __half inputTile[TileRows][Channels];
    __shared__ __half hiddenTile[TileRows][Hidden];
    __shared__ __half weightTile[Channels][Channels];
    const int local=threadIdx.y*Channels+threadIdx.x;
    const int firstRow=blockIdx.x*TileRows+threadIdx.y;
    for(int index=local;index<TileRows*Channels;index+=Threads){
        const int tileRow=index/Channels,channel=index%Channels;
        const int row=blockIdx.x*TileRows+tileRow;
        inputTile[tileRow][channel]=row<rows?input[size_t(row)*Channels+channel]:__float2half(0);
    }
    __syncthreads();
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Channels*Channels;index+=Threads){
            const int k=index/Channels,column=index%Channels;
            weightTile[k][column]=expandWeight[size_t(k)*Hidden+chunk*Channels+column];
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<RowsPerThread;++r){
            const int tileRow=threadIdx.y+r*RowThreads;
            float sum=0;
            #pragma unroll
            for(int k=0;k<Channels;k+=2)
                sum=amd_mixed_dot(__halves2half2(inputTile[tileRow][k],inputTile[tileRow][k+1]),
                                  __halves2half2(weightTile[k][threadIdx.x],weightTile[k+1][threadIdx.x]),sum,false);
            hiddenTile[tileRow][chunk*Channels+threadIdx.x]=
                pubh(rdna2_nr::ffn_gate(__float2half(sum)));
        }
        __syncthreads();
    }
    float sums[RowsPerThread]={};
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Channels*Channels;index+=Threads){
            const int k=index/Channels,column=index%Channels;
            weightTile[k][column]=contractWeight[size_t(chunk*Channels+k)*Channels+column];
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<Channels;k+=2){
            const __half2 weight=__halves2half2(weightTile[k][threadIdx.x],weightTile[k+1][threadIdx.x]);
            #pragma unroll
            for(int r=0;r<RowsPerThread;++r){
                const int tileRow=threadIdx.y+r*RowThreads;
                sums[r]=amd_mixed_dot(__halves2half2(hiddenTile[tileRow][chunk*Channels+k],
                                                     hiddenTile[tileRow][chunk*Channels+k+1]),
                                      weight,sums[r],false);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RowsPerThread;++r){
        const int row=firstRow+r*RowThreads;
        if(row<rows){
            const __half value=__hadd(__float2half(sums[r]),
                                      __hmul(input[size_t(row)*Channels+threadIdx.x],cosine[threadIdx.x]));
            output[size_t(row)*Channels+threadIdx.x]=value;
        }
    }
}

// The same C32 FP16 contract with DOT2 operands stored as native 32-bit LDS
// words. The original kernel stores each operand as two separate half values;
// on gfx1030 that makes adjacent lanes share an LDS bank and requires two LDS
// reads before every DOT2. Packing only the local representation removes that
// traffic without changing weight storage, reduction order, or FP16/E4M3
// boundaries.
template<int TileRows=16,int RowsPerThread=4>
__global__ void fused_c32_ffn_packed(const __half* input,const __half* expandWeight,
                                     const __half* contractWeight,const __half* cosine,
                                     __half* output,int rows) {
    constexpr int Channels=32,Hidden=128,Pairs=Channels/2;
    static_assert(TileRows%RowsPerThread==0,"invalid packed C32 row tile");
    constexpr int RowThreads=TileRows/RowsPerThread,Threads=Channels*RowThreads;
    __shared__ __half2 inputTile[TileRows][Pairs];
    __shared__ __half hiddenTile[TileRows][Hidden];
    __shared__ __half2 weightTile[Pairs][Channels];
    const int local=threadIdx.y*Channels+threadIdx.x;
    const int firstRow=blockIdx.x*TileRows+threadIdx.y;
    for(int index=local;index<TileRows*Pairs;index+=Threads){
        const int tileRow=index/Pairs,pair=index%Pairs;
        const int row=blockIdx.x*TileRows+tileRow;
        inputTile[tileRow][pair]=row<rows?
            *reinterpret_cast<const __half2*>(input+size_t(row)*Channels+pair*2):
            __float2half2_rn(0.f);
    }
    __syncthreads();
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Pairs*Channels;index+=Threads){
            const int pair=index/Channels,column=index%Channels,k=pair*2;
            weightTile[pair][column]=__halves2half2(
                expandWeight[size_t(k)*Hidden+chunk*Channels+column],
                expandWeight[size_t(k+1)*Hidden+chunk*Channels+column]);
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<RowsPerThread;++r){
            const int tileRow=threadIdx.y+r*RowThreads;
            float sum=0;
            #pragma unroll
            for(int pair=0;pair<Pairs;++pair)
                sum=amd_mixed_dot(inputTile[tileRow][pair],weightTile[pair][threadIdx.x],sum,false);
            hiddenTile[tileRow][chunk*Channels+threadIdx.x]=
                pubh(rdna2_nr::ffn_gate(__float2half(sum)));
        }
        __syncthreads();
    }
    float sums[RowsPerThread]={};
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Pairs*Channels;index+=Threads){
            const int pair=index/Channels,column=index%Channels,k=pair*2;
            weightTile[pair][column]=__halves2half2(
                contractWeight[size_t(chunk*Channels+k)*Channels+column],
                contractWeight[size_t(chunk*Channels+k+1)*Channels+column]);
        }
        __syncthreads();
        #pragma unroll
        for(int pair=0;pair<Pairs;++pair){
            const __half2 weight=weightTile[pair][threadIdx.x];
            #pragma unroll
            for(int r=0;r<RowsPerThread;++r){
                const int tileRow=threadIdx.y+r*RowThreads;
                sums[r]=amd_mixed_dot(*reinterpret_cast<const __half2*>(
                    hiddenTile[tileRow]+chunk*Channels+pair*2),weight,sums[r],false);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RowsPerThread;++r){
        const int row=firstRow+r*RowThreads;
        if(row<rows){
            const __half value=__hadd(__float2half(sums[r]),
                                      __hmul(input[size_t(row)*Channels+threadIdx.x],cosine[threadIdx.x]));
            output[size_t(row)*Channels+threadIdx.x]=value;
        }
    }
}

// W8A8 version of the resident C32 FFN. Input and gated-hidden quantization
// stay inside the 16-row workgroup; the 128-channel intermediate never reaches
// global memory. Scales use the selected group-16/group-32 contract for both
// matrices. This is a deliberately separate light-runtime experiment.
template<unsigned GroupSize>
__global__ void fused_c32_ffn_w8a8(const __half* input,
                                   const std::int8_t* expandWeight,const float* expandScale,
                                   const std::int8_t* contractWeight,const float* contractScale,
                                   const __half* cosine,__half* output,int rows,unsigned* status) {
    static_assert(GroupSize==16||GroupSize==32,"supported resident C32 group size");
    constexpr int TileRows=16,Channels=32,Hidden=128,RowsPerThread=4,
        RowThreads=TileRows/RowsPerThread,DotGroups=Channels/4,ScaleGroups=Channels/GroupSize;
    __shared__ std::int8_t inputTile[TileRows][Channels];
    __shared__ std::int8_t hiddenTile[TileRows][Hidden];
    __shared__ float inputScale[TileRows][Channels/16];
    __shared__ float hiddenScale[TileRows][Hidden/16];
    __shared__ std::int32_t weightTile[DotGroups][Channels+1];
    const int lane=threadIdx.x,local=threadIdx.y*Channels+lane;
    const int firstRow=blockIdx.x*TileRows+threadIdx.y;
    #pragma unroll
    for(int r=0;r<RowsPerThread;++r){
        const int tileRow=threadIdx.y+r*RowThreads,row=firstRow+r*RowThreads;
        const float value=row<rows?__half2float(input[size_t(row)*Channels+lane]):0.f;
        if(!isfinite(value))atomicOr(status,1u);
        const int scaleGroup=lane/GroupSize,scaleLane=lane%GroupSize;
        float maximum=fabsf(value);
        #pragma unroll
        for(int stride=GroupSize/2;stride;stride>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,stride,GroupSize));
        maximum=__shfl(maximum,0,GroupSize);
        const float scale=maximum==0?1.f:maximum/127.f,inv=maximum==0?1.f:127.f/maximum;
        if(scaleLane==0)inputScale[tileRow][scaleGroup]=scale;
        inputTile[tileRow][lane]=std::int8_t(max(-127,min(127,__float2int_rn(value*inv))));
    }
    __syncthreads();
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Channels*DotGroups;index+=Channels*RowThreads){
            const int column=index/DotGroups,g=index%DotGroups;
            weightTile[g][column]=*reinterpret_cast<const std::int32_t*>(
                expandWeight+size_t(chunk*Channels+column)*Channels+g*4);
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<RowsPerThread;++r){
            const int tileRow=threadIdx.y+r*RowThreads;
            float sum=0;
            #pragma unroll
            for(int scaleGroup=0;scaleGroup<ScaleGroups;++scaleGroup){
                int partial=0;
                #pragma unroll
                for(unsigned g=0;g<GroupSize/4;++g){
                    const int dotGroup=scaleGroup*(GroupSize/4)+g;
                    const char4 av=*reinterpret_cast<const char4*>(inputTile[tileRow]+dotGroup*4);
                    const char4 bv=*reinterpret_cast<const char4*>(&weightTile[dotGroup][lane]);
                    partial=amd_mixed_dot(av,bv,partial,false);
                }
                const int weightColumn=chunk*Channels+lane;
                const float combined=inputScale[tileRow][scaleGroup]*
                    expandScale[scaleGroup*Hidden+weightColumn];
                sum+=float(partial)*combined;
            }
            const __half gated=pubh(rdna2_nr::ffn_gate(__float2half(sum)));
            const float value=__half2float(gated);
            if(!isfinite(value))atomicOr(status,1u);
            const int scaleGroup=lane/GroupSize,scaleLane=lane%GroupSize;
            float maximum=fabsf(value);
            #pragma unroll
            for(int stride=GroupSize/2;stride;stride>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,stride,GroupSize));
            maximum=__shfl(maximum,0,GroupSize);
            const float scale=maximum==0?1.f:maximum/127.f,inv=maximum==0?1.f:127.f/maximum;
            if(scaleLane==0)hiddenScale[tileRow][chunk*ScaleGroups+scaleGroup]=scale;
            hiddenTile[tileRow][chunk*Channels+lane]=
                std::int8_t(max(-127,min(127,__float2int_rn(value*inv))));
        }
        __syncthreads();
    }
    float sums[RowsPerThread]={};
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Channels*DotGroups;index+=Channels*RowThreads){
            const int column=index/DotGroups,g=index%DotGroups;
            weightTile[g][column]=*reinterpret_cast<const std::int32_t*>(
                contractWeight+size_t(column)*Hidden+chunk*Channels+g*4);
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<RowsPerThread;++r){
            const int tileRow=threadIdx.y+r*RowThreads;
            #pragma unroll
            for(int scaleGroup=0;scaleGroup<ScaleGroups;++scaleGroup){
                int partial=0;
                #pragma unroll
                for(unsigned g=0;g<GroupSize/4;++g){
                    const int dotGroup=scaleGroup*(GroupSize/4)+g;
                    const char4 av=*reinterpret_cast<const char4*>(
                        hiddenTile[tileRow]+chunk*Channels+dotGroup*4);
                    const char4 bv=*reinterpret_cast<const char4*>(&weightTile[dotGroup][lane]);
                    partial=amd_mixed_dot(av,bv,partial,false);
                }
                const int globalGroup=chunk*ScaleGroups+scaleGroup;
                const float combined=hiddenScale[tileRow][globalGroup]*
                    contractScale[globalGroup*Channels+lane];
                sums[r]+=float(partial)*combined;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RowsPerThread;++r){
        const int row=firstRow+r*RowThreads;
        if(row<rows)output[size_t(row)*Channels+lane]=__hadd(
            __float2half(sums[r]),__hmul(input[size_t(row)*Channels+lane],cosine[lane]));
    }
}

// First compact-student primitive: the same numerical contract and layout as
// fused_c32_ffn, with physical 32x64 and 64x32 matrices.
template<int Hidden>
__global__ void fused_c32_ffn_compact(const __half* input,const __half* expandWeight,
                                      const __half* contractWeight,const __half* cosine,
                                      __half* output,int rows) {
    static_assert(Hidden==32||Hidden==64||Hidden==96,"unsupported compact C32 width");
    constexpr int TileRows=16,Channels=32,RowsPerThread=4;
    constexpr int RowThreads=TileRows/RowsPerThread,Threads=Channels*RowThreads;
    __shared__ __half inputTile[TileRows][Channels];
    __shared__ __half hiddenTile[TileRows][Hidden];
    __shared__ __half weightTile[Channels][Channels];
    const int local=threadIdx.y*Channels+threadIdx.x;
    const int firstRow=blockIdx.x*TileRows+threadIdx.y;
    for(int index=local;index<TileRows*Channels;index+=Threads){
        const int tileRow=index/Channels,channel=index%Channels;
        const int row=blockIdx.x*TileRows+tileRow;
        inputTile[tileRow][channel]=row<rows?input[size_t(row)*Channels+channel]:__float2half(0);
    }
    __syncthreads();
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Channels*Channels;index+=Threads){
            const int k=index/Channels,column=index%Channels;
            weightTile[k][column]=expandWeight[size_t(k)*Hidden+chunk*Channels+column];
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<RowsPerThread;++r){
            const int tileRow=threadIdx.y+r*RowThreads;float sum=0;
            #pragma unroll
            for(int k=0;k<Channels;k+=2)
                sum=amd_mixed_dot(__halves2half2(inputTile[tileRow][k],inputTile[tileRow][k+1]),
                                  __halves2half2(weightTile[k][threadIdx.x],weightTile[k+1][threadIdx.x]),sum,false);
            hiddenTile[tileRow][chunk*Channels+threadIdx.x]=pubh(rdna2_nr::ffn_gate(__float2half(sum)));
        }
        __syncthreads();
    }
    float sums[RowsPerThread]={};
    for(int chunk=0;chunk<Hidden/Channels;++chunk){
        for(int index=local;index<Channels*Channels;index+=Threads){
            const int k=index/Channels,column=index%Channels;
            weightTile[k][column]=contractWeight[size_t(chunk*Channels+k)*Channels+column];
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<Channels;k+=2){
            const __half2 weight=__halves2half2(weightTile[k][threadIdx.x],weightTile[k+1][threadIdx.x]);
            #pragma unroll
            for(int r=0;r<RowsPerThread;++r){
                const int tileRow=threadIdx.y+r*RowThreads;
                sums[r]=amd_mixed_dot(__halves2half2(hiddenTile[tileRow][chunk*Channels+k],
                                                     hiddenTile[tileRow][chunk*Channels+k+1]),weight,sums[r],false);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RowsPerThread;++r){
        const int row=firstRow+r*RowThreads;
        if(row<rows)output[size_t(row)*Channels+threadIdx.x]=__hadd(
            __float2half(sums[r]),__hmul(input[size_t(row)*Channels+threadIdx.x],cosine[threadIdx.x]));
    }
}
// QKV consumes the GEMM result only after an FP32 -> FP16 boundary.  Keep that
// boundary in LDS, reproduce the original half-precision norm tree, and write
// the published head-major layout directly instead of materializing Mx3C FP32.
__global__ void gemm_hf_tiled_qkv_normalize(const __half* a,const __half* b,
                                             const __half* scale,__half* out,
                                             int m,int k,int heads,int tokensPerWindow,
                                             int spatialH,int spatialW,int top,int left,int paddedWidth) {
    constexpr int TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    __shared__ __half inverse[16];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y;
    const int chunk=blockIdx.x,n=3*heads*32;
    float sum=0;
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=512){
            const int r=index/TileK,q=index%TileK,globalRow=blockIdx.y*16+r;
            __half value=__float2half(0);
            if(globalRow<m&&base+q<k){
                if(spatialH){
                    int element=globalRow,ix=element%8;element/=8;int iy=element%8;element/=8;
                    const int wx=element%(paddedWidth/8),wy=element/(paddedWidth/8);
                    const int y=wy*8+iy-top,x=wx*8+ix-left;
                    if(y>=0&&y<spatialH&&x>=0&&x<spatialW)value=a[(size_t(y)*spatialW+x)*k+base+q];
                }else value=a[globalRow*k+base+q];
            }
            tileA[r][q]=value;
        }
        for(int index=local;index<TileK*32;index+=512){
            const int q=index/32,c=index%32,globalCol=chunk*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        if(row<m){
            const int span=min(TileK,k-base);
            for(int q=0;q<span;q+=2)
                sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                                  __halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]),sum,false);
        }
        __syncthreads();
    }
    tileA[threadIdx.y][threadIdx.x]=row<m?__float2half(sum):__float2half(0);
    __syncthreads();
    const int kind=chunk/heads,head=chunk%heads;
    if(threadIdx.x==0){
        __half inv=__float2half(1);
        if(row<m&&kind<2){
            __half part[4][2],pair[4][2];
            for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){
                const int q=lane*2+p;
                const __half x0=tileA[threadIdx.y][q],x1=tileA[threadIdx.y][q+8];
                const __half x2=tileA[threadIdx.y][q+16],x3=tileA[threadIdx.y][q+24];
                const __half first=__hfma(x1,x1,__hmul(x0,x0));
                const __half second=__hfma(x3,x3,__hmul(x2,x2));
                part[lane][p]=__hadd(first,second);
            }
            for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)
                pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);
            __half norm=__hadd(__hadd(pair[0][0],pair[1][0]),
                               __hadd(pair[0][1],pair[1][1]));
            norm=__float2half(fmaxf(__half2float(norm),0.00006198883056640625f));
            inv=__float2half(rsqrtf(__half2float(norm)));
        }
        inverse[threadIdx.y]=inv;
    }
    __syncthreads();
    if(row<m){
        __half value=tileA[threadIdx.y][threadIdx.x];
        if(kind<2)value=__hmul(value,inverse[threadIdx.y]);
        if(kind==0)value=__hmul(value,scale[head]);
        const int window=row/tokensPerWindow,token=row%tokensPerWindow;
        const size_t target=(((size_t(window)*3+kind)*heads+head)*tokensPerWindow+token)*32+threadIdx.x;
        out[target]=pubh(value);
    }
}
// Normalize a materialized row-major half QKV projection with the exact tree
// used by the fused FP16 kernel, then publish its head-major layout. Grouped
// INT8 uses this boundary so only the GEMM changes precision.
__global__ void normalize_qkv_half(const __half* source,const __half* scale,__half* out,
                                   int m,int heads,int tokensPerWindow){
    __shared__ __half values[16][32];
    __shared__ __half inverse[16];
    const int row=blockIdx.y*16+threadIdx.y,chunk=blockIdx.x,n=3*heads*32;
    values[threadIdx.y][threadIdx.x]=row<m?source[size_t(row)*n+chunk*32+threadIdx.x]:__float2half(0);
    __syncthreads();
    const int kind=chunk/heads,head=chunk%heads;
    if(threadIdx.x==0){
        __half inv=__float2half(1);
        if(row<m&&kind<2){
            __half part[4][2],pair[4][2];
            for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){
                const int q=lane*2+p;
                const __half x0=values[threadIdx.y][q],x1=values[threadIdx.y][q+8];
                const __half x2=values[threadIdx.y][q+16],x3=values[threadIdx.y][q+24];
                const __half first=__hfma(x1,x1,__hmul(x0,x0));
                const __half second=__hfma(x3,x3,__hmul(x2,x2));
                part[lane][p]=__hadd(first,second);
            }
            for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);
            __half norm=__hadd(__hadd(pair[0][0],pair[1][0]),__hadd(pair[0][1],pair[1][1]));
            norm=__float2half(fmaxf(__half2float(norm),0.00006198883056640625f));
            inv=__float2half(rsqrtf(__half2float(norm)));
        }
        inverse[threadIdx.y]=inv;
    }
    __syncthreads();
    if(row<m){
        __half value=values[threadIdx.y][threadIdx.x];
        if(kind<2)value=__hmul(value,inverse[threadIdx.y]);
        if(kind==0)value=__hmul(value,scale[head]);
        const int window=row/tokensPerWindow,token=row%tokensPerWindow;
        const size_t target=(((size_t(window)*3+kind)*heads+head)*tokensPerWindow+token)*32+threadIdx.x;
        out[target]=pubh(value);
    }
}
__global__ void gemm_hf_tiled_residual(const __half* a,const __half* b,
                                       const __half* skip,const __half* cosine,
                                       __half* out,int m,int k,int n,bool publish,__half* raw=nullptr) {
    constexpr int TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*32+threadIdx.x;
    float sum=0;
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=512){
            const int r=index/TileK,q=index%TileK,globalRow=blockIdx.y*16+r;
            tileA[r][q]=(globalRow<m&&base+q<k)?a[globalRow*k+base+q]:__float2half(0);
        }
        for(int index=local;index<TileK*32;index+=512){
            const int q=index/32,c=index%32,globalCol=blockIdx.x*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        if(row<m&&col<n){
            const int span=min(TileK,k-base);
            for(int q=0;q<span;q+=2)
                sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                                  __halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]),sum,false);
        }
        __syncthreads();
    }
    if(row<m&&col<n){
        if(raw)raw[row*n+col]=__float2half(sum);
        __half value=__hadd(__float2half(sum),__hmul(skip[row*n+col],cosine[col]));
        out[row*n+col]=publish?pubh(value):value;
    }
}
__global__ void gemm_hf_tiled_residual_to_spatial(const __half* a,const __half* b,
                                                   const __half* skip,const __half* cosine,
                                                   __half* out,int m,int k,int n,bool publish,
                                                   int h,int w,int top,int left,int paddedWidth) {
    constexpr int TileK=32;
    __shared__ __half tileA[16][TileK];
    __shared__ __half tileB[TileK][32];
    const int local=threadIdx.y*32+threadIdx.x;
    const int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*32+threadIdx.x;
    float sum=0;
    for(int base=0;base<k;base+=TileK){
        for(int index=local;index<16*TileK;index+=512){
            const int r=index/TileK,q=index%TileK,globalRow=blockIdx.y*16+r;
            tileA[r][q]=(globalRow<m&&base+q<k)?a[globalRow*k+base+q]:__float2half(0);
        }
        for(int index=local;index<TileK*32;index+=512){
            const int q=index/32,c=index%32,globalCol=blockIdx.x*32+c;
            tileB[q][c]=(base+q<k&&globalCol<n)?b[(base+q)*n+globalCol]:__float2half(0);
        }
        __syncthreads();
        if(row<m&&col<n){
            const int span=min(TileK,k-base);
            for(int q=0;q<span;q+=2)
                sum=amd_mixed_dot(__halves2half2(tileA[threadIdx.y][q],tileA[threadIdx.y][q+1]),
                                  __halves2half2(tileB[q][threadIdx.x],tileB[q+1][threadIdx.x]),sum,false);
        }
        __syncthreads();
    }
    if(row<m&&col<n){
        int element=row,ix=element%8;element/=8;int iy=element%8;element/=8;
        const int wx=element%(paddedWidth/8),wy=element/(paddedWidth/8);
        const int y=wy*8+iy-top,x=wx*8+ix-left;
        if(y>=0&&y<h&&x>=0&&x<w){
            const size_t target=(size_t(y)*w+x)*n+col;
            __half value=__hadd(__float2half(sum),__hmul(skip[target],cosine[col]));
            out[target]=publish?pubh(value):value;
        }
    }
}
__global__ void gate_publish(const float* in,__half* out,unsigned n) { unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=pubh(rdna2_nr::ffn_gate(__float2half(in[i]))); }
__global__ void gate_half(const float* in,__half* out,unsigned n) { unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=rdna2_nr::ffn_gate(__float2half(in[i])); }
__global__ void publish_float(const float* in,__half* out,unsigned n) { unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=pubh(__float2half(in[i])); }
__global__ void residual_h(const float* branch,const __half* skip,const __half* cosine,__half* out,unsigned n,int c,bool publish) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) { __half v=__hadd(__float2half(branch[i]),__hmul(skip[i],cosine[i%c])); out[i]=publish?pubh(v):v; }
}
__global__ void branched_expand(const __half* x,const __half* w,__half* hidden,int c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x,g=c/32,total=M*g*4*32; if(idx>=total)return;
    int col=idx%32,t=idx/32,b=t%4;t/=4;int o=t%g,row=t/g; __half sum=__float2half(0);
    for(int ih=0;ih<g;++ih){float part=0;size_t base=(((size_t(o)*4+b)*g+ih)*32)*32+col;
        for(int k=0;k<32;k+=2) part=amd_mixed_dot(__halves2half2(x[row*c+ih*32+k],x[row*c+ih*32+k+1]),__halves2half2(w[base+k*32],w[base+(k+1)*32]),part,false);
        sum=__hadd(sum,__float2half(part));}
    hidden[idx]=pubh(rdna2_nr::ffn_gate(sum));
}
__global__ void branched_project(const __half* hidden,const __half* w,__half* branches,int c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x,g=c/32,total=M*g*4*32;if(idx>=total)return;
    int col=idx%32,t=idx/32,b=t%4;t/=4;int o=t%g,row=t/g;float sum=0;size_t wb=((size_t(o)*4+b)*32)*32+col;
    size_t hi=(((size_t(row)*g+o)*4+b)*32);
    for(int k=0;k<32;k+=2)sum=amd_mixed_dot(__halves2half2(hidden[hi+k],hidden[hi+k+1]),__halves2half2(w[wb+k*32],w[wb+(k+1)*32]),sum,false);
    branches[idx]=__float2half(sum);
}
__global__ void branched_merge(const __half* branches,__half* combined,int c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M*c)return;int g=c/32,col=idx%32,o=(idx/32)%g,row=idx/c;__half sum=__float2half(0);
    size_t base=((size_t(row)*g+o)*4)*32+col;for(int b=0;b<4;++b)sum=__hadd(sum,branches[base+b*32]);combined[idx]=pubh(sum);
}
// Production graph variants process every 64-token group in one launch.  The
// index mapping inside each row is identical to the diagnostic M=64 kernels;
// only the launch boundary changes.
__global__ void branched_expand_rows(const __half* x,const __half* w,__half* hidden,int c,int rows) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x,g=c/32,total=rows*g*4*32;if(idx>=total)return;
    int col=idx%32,t=idx/32,b=t%4;t/=4;int o=t%g,row=t/g;__half sum=__float2half(0);
    for(int ih=0;ih<g;++ih){float part=0;size_t base=(((size_t(o)*4+b)*g+ih)*32)*32+col;
        for(int k=0;k<32;k+=2)part=amd_mixed_dot(__halves2half2(x[row*c+ih*32+k],x[row*c+ih*32+k+1]),__halves2half2(w[base+k*32],w[base+(k+1)*32]),part,false);
        sum=__hadd(sum,__float2half(part));}
    hidden[idx]=pubh(rdna2_nr::ffn_gate(sum));
}
__global__ void branched_project_rows(const __half* hidden,const __half* w,__half* branches,int c,int rows) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x,g=c/32,total=rows*g*4*32;if(idx>=total)return;
    int col=idx%32,t=idx/32,b=t%4;t/=4;int o=t%g,row=t/g;float sum=0;size_t wb=((size_t(o)*4+b)*32)*32+col;
    size_t hi=(((size_t(row)*g+o)*4+b)*32);
    for(int k=0;k<32;k+=2)sum=amd_mixed_dot(__halves2half2(hidden[hi+k],hidden[hi+k+1]),__halves2half2(w[wb+k*32],w[wb+(k+1)*32]),sum,false);
    branches[idx]=__float2half(sum);
}
__global__ void branched_merge_rows(const __half* branches,__half* combined,int c,int rows) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=rows*c)return;int g=c/32,col=idx%32,o=(idx/32)%g,row=idx/c;__half sum=__float2half(0);
    size_t base=((size_t(row)*g+o)*4)*32+col;for(int b=0;b<4;++b)sum=__hadd(sum,branches[base+b*32]);combined[idx]=pubh(sum);
}
// Full four-branch projection, with the original ordered DOT2 and half merge.
// Reuse one weight tile across 16 rows; keep branch outputs in registers.
__global__ void branched_project_merge_tiled(const __half* hidden,const __half* weight,
                                            __half* combined,int c,int rows) {
    __shared__ __half weights[4][32][33];
    __shared__ __half inputs[16][4][32];
    const unsigned tid=threadIdx.y*32+threadIdx.x,group=blockIdx.x,groups=c/32;
    for(unsigned i=tid;i<4*32*32;i+=128){
        const unsigned b=i/1024,k=(i/32)%32,col=i%32;
        weights[b][k][col]=weight[(size_t(group)*4+b)*1024+k*32+col];
    }
    for(unsigned i=tid;i<16*4*32;i+=128){
        const unsigned r=i/128,b=(i/32)%4,col=i%32,row=blockIdx.y*16+r;
        inputs[r][b][col]=row<unsigned(rows)?hidden[((size_t(row)*groups+group)*4+b)*32+col]:__float2half(0);
    }
    __syncthreads();
    for(unsigned r=threadIdx.y;r<16;r+=4){
        const unsigned row=blockIdx.y*16+r;
        if(row<unsigned(rows)){
            __half merged=__float2half(0);
            for(unsigned b=0;b<4;++b){
                float sum=0;
                for(unsigned k=0;k<32;k+=2)
                    sum=amd_mixed_dot(__halves2half2(inputs[r][b][k],inputs[r][b][k+1]),
                        __halves2half2(weights[b][k][threadIdx.x],weights[b][k+1][threadIdx.x]),sum,false);
                merged=__hadd(merged,__float2half(sum));
            }
            combined[size_t(row)*c+group*32+threadIdx.x]=pubh(merged);
        }
    }
}
template<int Branches>
__global__ void compact_branched_expand_rows(const __half* x,const __half* w,__half* hidden,int c,int rows) {
    static_assert(Branches>=1&&Branches<=3,"unsupported compact branch count");
    int idx=blockIdx.x*blockDim.x+threadIdx.x,g=c/32,total=rows*g*Branches*32;if(idx>=total)return;
    int col=idx%32,t=idx/32,b=t%Branches;t/=Branches;int o=t%g,row=t/g;__half sum=__float2half(0);
    for(int ih=0;ih<g;++ih){float part=0;size_t base=(((size_t(o)*Branches+b)*g+ih)*32)*32+col;
        for(int k=0;k<32;k+=2)part=amd_mixed_dot(__halves2half2(x[row*c+ih*32+k],x[row*c+ih*32+k+1]),__halves2half2(w[base+k*32],w[base+(k+1)*32]),part,false);
        sum=__hadd(sum,__float2half(part));}
    hidden[idx]=pubh(rdna2_nr::ffn_gate(sum));
}
template<int Branches>
__global__ void compact_branched_project_merge_rows(const __half* hidden,const __half* w,__half* combined,int c,int rows) {
    static_assert(Branches>=1&&Branches<=3,"unsupported compact branch count");
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=rows*c)return;int g=c/32,col=idx%32,o=(idx/32)%g,row=idx/c;__half merged=__float2half(0);
    for(int b=0;b<Branches;++b){float sum=0;size_t wb=((size_t(o)*Branches+b)*32)*32+col;
        size_t hi=(((size_t(row)*g+o)*Branches+b)*32);
        for(int k=0;k<32;k+=2)sum=amd_mixed_dot(__halves2half2(hidden[hi+k],hidden[hi+k+1]),__halves2half2(w[wb+k*32],w[wb+(k+1)*32]),sum,false);
        merged=__hadd(merged,__float2half(sum));}
    combined[idx]=pubh(merged);
}
__global__ void split_expand(const __half* first,const __half* w,__half* hidden) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M*8*256)return;int col=idx%256,t=idx/256,g=t%8,row=t/8;float sum=0;size_t wb=(size_t(g)*64)*256+col;
    for(int k=0;k<64;k+=2)sum=amd_mixed_dot(__halves2half2(first[row*512+g*64+k],first[row*512+g*64+k+1]),__halves2half2(w[wb+k*256],w[wb+(k+1)*256]),sum,false);hidden[idx]=rdna2_nr::ffn_gate(__float2half(sum));
}
__global__ void split_project(const __half* hidden,const __half* w,__half* grouped) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M*512)return;int col=idx%64,g=(idx/64)%8,row=idx/512;float sum=0;size_t hi=(size_t(row)*8+g)*256,wb=(size_t(g)*256)*64+col;
    for(int k=0;k<256;k+=2)sum=amd_mixed_dot(__halves2half2(hidden[hi+k],hidden[hi+k+1]),__halves2half2(w[wb+k*64],w[wb+(k+1)*64]),sum,false);grouped[idx]=pubh(__float2half(sum));
}
__global__ void split_expand_rows(const __half* first,const __half* w,__half* hidden,int rows) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=rows*8*256)return;int col=idx%256,t=idx/256,g=t%8,row=t/8;float sum=0;size_t wb=(size_t(g)*64)*256+col;
    for(int k=0;k<64;k+=2)sum=amd_mixed_dot(__halves2half2(first[row*512+g*64+k],first[row*512+g*64+k+1]),__halves2half2(w[wb+k*256],w[wb+(k+1)*256]),sum,false);hidden[idx]=rdna2_nr::ffn_gate(__float2half(sum));
}
__global__ void split_project_rows(const __half* hidden,const __half* w,__half* grouped,int rows) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=rows*512)return;int col=idx%64,g=(idx/64)%8,row=idx/512;float sum=0;size_t hi=(size_t(row)*8+g)*256,wb=(size_t(g)*256)*64+col;
    for(int k=0;k<256;k+=2)sum=amd_mixed_dot(__halves2half2(hidden[hi+k],hidden[hi+k+1]),__halves2half2(w[wb+k*64],w[wb+(k+1)*64]),sum,false);grouped[idx]=pubh(__float2half(sum));
}
__global__ void normalize_window(const float* projected,const __half* scale,__half* norms,__half* published,int c) {
    int heads=c/32,vector=blockIdx.x*blockDim.x+threadIdx.x;if(vector>=3*heads*M)return;int kind=vector/(heads*M),head=(vector/M)%heads,token=vector%M;__half x[32];
    for(int i=0;i<32;++i)x[i]=__float2half(projected[token*3*c+kind*c+head*32+i]);__half inverse=__float2half(1);
    if(kind<2){__half part[4][2],pair[4][2];for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){int q=lane*2+p;__half a=__hfma(x[q+8],x[q+8],__hmul(x[q],x[q]));__half b=__hfma(x[q+24],x[q+24],__hmul(x[q+16],x[q+16]));part[lane][p]=__hadd(a,b);}for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);__half norm=__hadd(__hadd(pair[0][0],pair[1][0]),__hadd(pair[0][1],pair[1][1]));norm=__float2half(fmaxf(__half2float(norm),0.00006198883056640625f));norms[vector]=norm;inverse=__float2half(rsqrtf(__half2float(norm)));}
    for(int i=0;i<32;++i){__half z=kind<2?__hmul(x[i],inverse):x[i];if(kind==0)z=__hmul(z,scale[head]);published[vector*32+i]=pubh(z);}
}
__global__ void window_scores(const __half* qkv,const __half* bias,__half* scores,int heads) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=heads*4096)return;int head=idx/4096,row=(idx/64)%64,col=idx%64;const __half* q=qkv+(head*64+row)*32;const __half* k=qkv+heads*64*32+(head*64+col)*32;float sum=0;
    for(int i=0;i<32;i+=2)sum=amd_mixed_dot(__halves2half2(q[i],q[i+1]),__halves2half2(k[i],k[i+1]),sum,false);scores[idx]=__hadd(__float2half(sum),bias[idx]);
}
__global__ void window_probabilities(const __half* scores,__half* weights,__half* probs,int rows) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;__half total=__float2half(0),w[64];
    for(int j=0;j<64;j+=2){unsigned bits[2];for(int i=0;i<2;++i){__half a=__hfma(scores[row*64+j+i],__float2half(.044921875f),__float2half(1.30078125f));a=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(a))));bits[i]=__half_as_ushort(a);}unsigned transformed=((bits[0]|(bits[1]<<16))<<5)+0x7ff88000u;w[j]=__ushort_as_half(transformed&65535);w[j+1]=__ushort_as_half(transformed>>16);total=__hadd(total,w[j]);total=__hadd(total,w[j+1]);}
    __half inv=__float2half(1.f/__half2float(total));for(int j=0;j<64;++j){weights[row*64+j]=w[j];probs[row*64+j]=pubh(__hmul(w[j],inv));}
}
__global__ void window_attend(const __half* probs,const __half* qkv,__half* attended,int c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=M*c)return;int row=idx/c,head=(idx%c)/32,ch=idx%32,heads=c/32;float sum=0;
    for(int j=0;j<64;++j)sum=fmaf(__half2float(probs[(head*64+row)*64+j]),__half2float(qkv[((2*heads+head)*64+j)*32+ch]),sum);attended[idx]=pubh(__float2half(sum));
}
__global__ void normalize_windows(const float* projected,const __half* scale,__half* norms,__half* published,int c,int windows) {
    int heads=c/32,global=blockIdx.x*blockDim.x+threadIdx.x,per=3*heads*M;if(global>=windows*per)return;
    int window=global/per,vector=global%per,kind=vector/(heads*M),head=(vector/M)%heads,token=vector%M;const float* source=projected+size_t(window)*M*3*c;__half x[32];
    for(int i=0;i<32;++i)x[i]=__float2half(source[token*3*c+kind*c+head*32+i]);__half inverse=__float2half(1);
    if(kind<2){__half part[4][2],pair[4][2];for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){int q=lane*2+p;__half a=__hfma(x[q+8],x[q+8],__hmul(x[q],x[q]));__half b=__hfma(x[q+24],x[q+24],__hmul(x[q+16],x[q+16]));part[lane][p]=__hadd(a,b);}for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);__half norm=__hadd(__hadd(pair[0][0],pair[1][0]),__hadd(pair[0][1],pair[1][1]));norm=__float2half(fmaxf(__half2float(norm),0.00006198883056640625f));if(norms)norms[size_t(window)*2*heads*M+vector]=norm;inverse=__float2half(rsqrtf(__half2float(norm)));}
    for(int i=0;i<32;++i){__half z=kind<2?__hmul(x[i],inverse):x[i];if(kind==0)z=__hmul(z,scale[head]);published[size_t(global)*32+i]=pubh(z);}
}
__global__ void score_windows(const __half* qkv,const __half* bias,__half* scores,int heads,int windows) {
    int global=blockIdx.x*blockDim.x+threadIdx.x,per=heads*4096;if(global>=windows*per)return;int window=global/per,idx=global%per,head=idx/4096,row=(idx/64)%64,col=idx%64;
    const __half* base=qkv+size_t(window)*M*heads*3*32;const __half* q=base+(head*64+row)*32,*k=base+heads*64*32+(head*64+col)*32;float sum=0;
    for(int i=0;i<32;i+=2)sum=amd_mixed_dot(__halves2half2(q[i],q[i+1]),__halves2half2(k[i],k[i+1]),sum,false);scores[global]=__hadd(__float2half(sum),bias[idx]);
}
__global__ void probability_windows(const __half* scores,__half* weights,__half* probs,int rows) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;__half total=__float2half(0),w[64];
    for(int j=0;j<64;j+=2){unsigned bits[2];for(int i=0;i<2;++i){__half a=__hfma(scores[size_t(row)*64+j+i],__float2half(.044921875f),__float2half(1.30078125f));a=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(a))));bits[i]=__half_as_ushort(a);}unsigned transformed=((bits[0]|(bits[1]<<16))<<5)+0x7ff88000u;w[j]=__ushort_as_half(transformed&65535);w[j+1]=__ushort_as_half(transformed>>16);total=__hadd(total,w[j]);total=__hadd(total,w[j+1]);}
    __half inv=__float2half(1.f/__half2float(total));for(int j=0;j<64;++j){if(weights)weights[size_t(row)*64+j]=w[j];probs[size_t(row)*64+j]=pubh(__hmul(w[j],inv));}
}
__global__ void attend_windows(const __half* probs,const __half* qkv,__half* attended,int c,int windows) {
    int global=blockIdx.x*blockDim.x+threadIdx.x,per=M*c;if(global>=windows*per)return;int window=global/per,idx=global%per,row=idx/c,head=(idx%c)/32,ch=idx%32,heads=c/32;float sum=0;
    const __half* probability=probs+size_t(window)*heads*M*M;const __half* base=qkv+size_t(window)*M*3*c;
    for(int j=0;j<64;++j)sum=fmaf(__half2float(probability[(head*64+row)*64+j]),__half2float(base[((2*heads+head)*64+j)*32+ch]),sum);attended[global]=pubh(__float2half(sum));
}
// Compute independent score pairs in parallel, then retain the exact ordered
// half sum for each row. Transposed packed keys avoid strided LDS reads.
template<unsigned ProbabilityStride=64,unsigned ChannelStride=32>
__device__ inline void window_probabilities_parallel(const __half*q,const __half*k,
    const __half*bias,__half*probabilities,unsigned*keys,__half*inverses) {
    const unsigned tid=threadIdx.x;
    for(unsigned i=tid;i<16*64;i+=blockDim.x){
        const unsigned pair=i/64,token=i%64;
        keys[i]=unsigned(__half_as_ushort(k[token*ChannelStride+pair*2]))|
                (unsigned(__half_as_ushort(k[token*ChannelStride+pair*2+1]))<<16);
    }
    __syncthreads();
    for(unsigned i=tid;i<64*32;i+=blockDim.x){
        const unsigned row=i/32,column=(i%32)*2;
        unsigned bits[2];
        for(unsigned pair=0;pair<2;++pair){
            float sum=0;
            for(unsigned j=0;j<16;++j){
                const unsigned packed=keys[j*64+column+pair];
                sum=amd_mixed_dot(__halves2half2(q[row*ChannelStride+j*2],q[row*ChannelStride+j*2+1]),
                    __halves2half2(__ushort_as_half(packed&65535),__ushort_as_half(packed>>16)),sum,false);
            }
            const __half score=__hadd(__float2half(sum),bias[row*64+column+pair]);
            __half weight=__hfma(score,__float2half(.044921875f),__float2half(1.30078125f));
            weight=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(weight))));
            bits[pair]=__half_as_ushort(weight);
        }
        const unsigned transformed=((bits[0]|(bits[1]<<16))<<5)+0x7ff88000u;
        probabilities[row*ProbabilityStride+column]=__ushort_as_half(transformed&65535);
        probabilities[row*ProbabilityStride+column+1]=__ushort_as_half(transformed>>16);
    }
    __syncthreads();
    if(tid<64){
        __half total=__float2half(0);
        for(unsigned col=0;col<64;++col)total=__hadd(total,probabilities[tid*ProbabilityStride+col]);
        inverses[tid]=__float2half(1.f/__half2float(total));
    }
    __syncthreads();
    for(unsigned i=tid;i<64*64;i+=blockDim.x)
        probabilities[(i/64)*ProbabilityStride+i%64]=pubh(__hmul(probabilities[(i/64)*ProbabilityStride+i%64],inverses[i/64]));
    __syncthreads();
}

// One workgroup owns one window/head. Scores and published probabilities never
// leave LDS; each row keeps the original sequential half reduction and each
// attended output keeps the original ascending float FMA reduction.
template<bool Parallel=false,unsigned ProbabilityStride=64>
__global__ void attention_windows_fused(const __half* qkv,const __half* bias,__half* attended,int c,int windows) {
    __shared__ __half q[64*32],k[64*32],v[64*32],probabilities[64*ProbabilityStride];
    const int heads=c/32,group=blockIdx.x;
    if(group>=windows*heads)return;
    const int window=group/heads,head=group%heads,tid=threadIdx.x;
    const __half* base=qkv+size_t(window)*64*3*c;
    for(int i=tid;i<64*32;i+=blockDim.x){
        const int token=i/32,ch=i%32;
        q[i]=base[(head*64+token)*32+ch];
        k[i]=base[((heads+head)*64+token)*32+ch];
        v[i]=base[((2*heads+head)*64+token)*32+ch];
    }
    __syncthreads();
    if constexpr(Parallel){
        __shared__ unsigned keys[16*64];
        __shared__ __half inverses[64];
        window_probabilities_parallel<ProbabilityStride>(q,k,bias+head*64*64,probabilities,keys,inverses);
    }else if(tid<64){
        __half total=__float2half(0);
        for(int col=0;col<64;col+=2){
            unsigned bits[2];
            for(int pair=0;pair<2;++pair){
                float sum=0;const int target=col+pair;
                for(int channel=0;channel<32;channel+=2)
                    sum=amd_mixed_dot(__halves2half2(q[tid*32+channel],q[tid*32+channel+1]),
                                      __halves2half2(k[target*32+channel],k[target*32+channel+1]),sum,false);
                __half score=__hadd(__float2half(sum),bias[(head*64+tid)*64+target]);
                __half weight=__hfma(score,__float2half(.044921875f),__float2half(1.30078125f));
                weight=__float2half(fminf(1.5693359375f,fmaxf(1.03125f,__half2float(weight))));
                bits[pair]=__half_as_ushort(weight);
            }
            unsigned transformed=((bits[0]|(bits[1]<<16))<<5)+0x7ff88000u;
            probabilities[tid*ProbabilityStride+col]=__ushort_as_half(transformed&65535);
            probabilities[tid*ProbabilityStride+col+1]=__ushort_as_half(transformed>>16);
            total=__hadd(total,probabilities[tid*ProbabilityStride+col]);
            total=__hadd(total,probabilities[tid*ProbabilityStride+col+1]);
        }
        const __half inverse=__float2half(1.f/__half2float(total));
        for(int col=0;col<64;++col)
            probabilities[tid*ProbabilityStride+col]=pubh(__hmul(probabilities[tid*ProbabilityStride+col],inverse));
    }
    __syncthreads();
    for(int index=tid;index<64*32;index+=blockDim.x){
        const int row=index/32,ch=index%32;float sum=0;
#if defined(NR_ATTENTION_DOT2) && NR_ATTENTION_DOT2
        for(int col=0;col<64;col+=2)
            sum=amd_mixed_dot(__halves2half2(probabilities[row*ProbabilityStride+col],probabilities[row*ProbabilityStride+col+1]),
                __halves2half2(v[col*32+ch],v[(col+1)*32+ch]),sum,false);
#else
        for(int col=0;col<64;++col)
            sum=fmaf(__half2float(probabilities[row*ProbabilityStride+col]),__half2float(v[col*32+ch]),sum);
#endif
        attended[size_t(window)*64*c+row*c+head*32+ch]=pubh(__float2half(sum));
    }
}

// C32 has one attention head.  Keep its QKV publication, probabilities, and
// attended values in LDS, then project directly back to the spatial layout.
// This preserves the original ascending DOT2/FMA reductions and the published
// E4M3 boundaries while removing both full-frame intermediate tensors.
// PackedWeights stores every adjacent K pair as one native half2 word with the
// output column contiguous.  It is an initialization-time layout only: the
// arithmetic and all publication boundaries below remain unchanged.
template<bool Parallel=false,unsigned ProbabilityStride=64,unsigned ChannelStride=32,
         bool PackedWeights=false,bool WaveProjection=false>
__global__ void fused_c32_attention(const __half* input,const __half* qkvWeight,
                                    const __half* bias,const __half* scale,
                                    const __half* projectionWeight,
                                    const __half* cosine,__half* output,
                                    int h,int w,int top,int left,int paddedWidth,
                                    bool publish) {
    __shared__ __half qkv[3*64*ChannelStride];
    __shared__ __half probabilities[64*ProbabilityStride];
    __shared__ __half attended[WaveProjection?1:64*32];
    const int window=blockIdx.x,tid=threadIdx.x;
    const int windowColumns=paddedWidth/8;
    const int wx=window%windowColumns,wy=window/windowColumns;
    for(int index=tid;index<3*64*32;index+=blockDim.x){
        const int kind=index/(64*32),within=index%(64*32);
        const int token=within/32,column=within%32;
        const int x=wx*8+token%8-left,y=wy*8+token/8-top;
        float sum=0;
        if(y>=0&&y<h&&x>=0&&x<w){
            const __half* row=input+(size_t(y)*w+x)*32;
            for(int channel=0;channel<32;channel+=2){
                __half2 weight;
                if constexpr(PackedWeights)
                    weight=reinterpret_cast<const __half2*>(qkvWeight)[
                        (channel/2)*96+kind*32+column];
                else weight=__halves2half2(qkvWeight[channel*96+kind*32+column],
                                           qkvWeight[(channel+1)*96+kind*32+column]);
                sum=amd_mixed_dot(__halves2half2(row[channel],row[channel+1]),weight,sum,false);
            }
        }
        qkv[(index/32)*ChannelStride+index%32]=__float2half(sum);
    }
    __syncthreads();
    if(tid<3*64){
        const int kind=tid/64,token=tid%64;
        __half* values=qkv+(kind*64+token)*ChannelStride;
        __half inverse=__float2half(1);
        if(kind<2){
            __half part[4][2],pair[4][2];
            for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p){
                const int q=lane*2+p;
                const __half first=__hfma(values[q+8],values[q+8],
                                          __hmul(values[q],values[q]));
                const __half second=__hfma(values[q+24],values[q+24],
                                           __hmul(values[q+16],values[q+16]));
                part[lane][p]=__hadd(first,second);
            }
            for(int lane=0;lane<4;++lane)for(int p=0;p<2;++p)
                pair[lane][p]=__hadd(part[lane][p],part[lane^2][p]);
            __half norm=__hadd(__hadd(pair[0][0],pair[1][0]),
                               __hadd(pair[0][1],pair[1][1]));
            norm=__float2half(fmaxf(__half2float(norm),0.00006198883056640625f));
            inverse=__float2half(rsqrtf(__half2float(norm)));
        }
        for(int channel=0;channel<32;++channel){
            __half value=kind<2?__hmul(values[channel],inverse):values[channel];
            if(kind==0)value=__hmul(value,scale[0]);
            values[channel]=pubh(value);
        }
    }
    __syncthreads();
    if constexpr(Parallel){
        __shared__ unsigned keys[16*64];
        __shared__ __half inverses[64];
        window_probabilities_parallel<ProbabilityStride,ChannelStride>(qkv,qkv+64*ChannelStride,bias,probabilities,keys,inverses);
    }else if(tid<64){
        __half total=__float2half(0);
        const __half* query=qkv+tid*ChannelStride;
        const __half* keys=qkv+64*ChannelStride;
        for(int column=0;column<64;column+=2){
            unsigned bits[2];
            for(int pair=0;pair<2;++pair){
                const int target=column+pair;
                float sum=0;
                for(int channel=0;channel<32;channel+=2)
                    sum=amd_mixed_dot(__halves2half2(query[channel],query[channel+1]),
                        __halves2half2(keys[target*ChannelStride+channel],keys[target*ChannelStride+channel+1]),sum,false);
                __half score=__hadd(__float2half(sum),bias[tid*64+target]);
                __half weight=__hfma(score,__float2half(.044921875f),
                                     __float2half(1.30078125f));
                weight=__float2half(fminf(1.5693359375f,
                                          fmaxf(1.03125f,__half2float(weight))));
                bits[pair]=__half_as_ushort(weight);
            }
            const unsigned transformed=((bits[0]|(bits[1]<<16))<<5)+0x7ff88000u;
            probabilities[tid*ProbabilityStride+column]=__ushort_as_half(transformed&65535);
            probabilities[tid*ProbabilityStride+column+1]=__ushort_as_half(transformed>>16);
            total=__hadd(total,probabilities[tid*ProbabilityStride+column]);
            total=__hadd(total,probabilities[tid*ProbabilityStride+column+1]);
        }
        const __half inverse=__float2half(1.f/__half2float(total));
        for(int column=0;column<64;++column)
            probabilities[tid*ProbabilityStride+column]=pubh(__hmul(probabilities[tid*ProbabilityStride+column],inverse));
    }
    __syncthreads();
    const __half* values=qkv+2*64*ChannelStride;
    if constexpr(WaveProjection){
        const int lane=tid%32,wave=tid/32;
        for(int token=wave;token<64;token+=8){
            float attendedSum=0;
#if defined(NR_ATTENTION_DOT2) && NR_ATTENTION_DOT2
            for(int column=0;column<64;column+=2)
                attendedSum=amd_mixed_dot(__halves2half2(probabilities[token*ProbabilityStride+column],probabilities[token*ProbabilityStride+column+1]),
                    __halves2half2(values[column*ChannelStride+lane],values[(column+1)*ChannelStride+lane]),attendedSum,false);
#else
            for(int column=0;column<64;++column)
                attendedSum=fmaf(__half2float(probabilities[token*ProbabilityStride+column]),
                                 __half2float(values[column*ChannelStride+lane]),attendedSum);
#endif
            const unsigned attendedBits=__half_as_ushort(pubh(__float2half(attendedSum)));
            float projectionSum=0;
            for(int channel=0;channel<32;channel+=2){
                const __half2 av=__halves2half2(
                    __ushort_as_half(__shfl(attendedBits,channel,32)),
                    __ushort_as_half(__shfl(attendedBits,channel+1,32)));
                __half2 weight;
                if constexpr(PackedWeights)
                    weight=reinterpret_cast<const __half2*>(projectionWeight)[(channel/2)*32+lane];
                else weight=__halves2half2(projectionWeight[channel*32+lane],
                                           projectionWeight[(channel+1)*32+lane]);
                projectionSum=amd_mixed_dot(av,weight,projectionSum,false);
            }
            const int x=wx*8+token%8-left,y=wy*8+token/8-top;
            if(y>=0&&y<h&&x>=0&&x<w){
                const size_t target=(size_t(y)*w+x)*32+lane;
                __half value=__hadd(__float2half(projectionSum),__hmul(input[target],cosine[lane]));
                output[target]=publish?pubh(value):value;
            }
        }
    }else{
        for(int index=tid;index<64*32;index+=blockDim.x){
            const int token=index/32,channel=index%32;
            float sum=0;
#if defined(NR_ATTENTION_DOT2) && NR_ATTENTION_DOT2
            for(int column=0;column<64;column+=2)
                sum=amd_mixed_dot(__halves2half2(probabilities[token*ProbabilityStride+column],probabilities[token*ProbabilityStride+column+1]),
                    __halves2half2(values[column*ChannelStride+channel],values[(column+1)*ChannelStride+channel]),sum,false);
#else
            for(int column=0;column<64;++column)
                sum=fmaf(__half2float(probabilities[token*ProbabilityStride+column]),
                         __half2float(values[column*ChannelStride+channel]),sum);
#endif
            attended[index]=pubh(__float2half(sum));
        }
        __syncthreads();
        for(int index=tid;index<64*32;index+=blockDim.x){
            const int token=index/32,column=index%32;
            const int x=wx*8+token%8-left,y=wy*8+token/8-top;
            if(y>=0&&y<h&&x>=0&&x<w){
                float sum=0;
                const __half* row=attended+token*32;
                for(int channel=0;channel<32;channel+=2){
                    __half2 weight;
                    if constexpr(PackedWeights)
                        weight=reinterpret_cast<const __half2*>(projectionWeight)[
                            (channel/2)*32+column];
                    else weight=__halves2half2(projectionWeight[channel*32+column],
                                               projectionWeight[(channel+1)*32+column]);
                    sum=amd_mixed_dot(__halves2half2(row[channel],row[channel+1]),weight,sum,false);
                }
                const size_t target=(size_t(y)*w+x)*32+column;
                __half value=__hadd(__float2half(sum),__hmul(input[target],cosine[column]));
                output[target]=publish?pubh(value):value;
            }
        }
    }
}
__global__ void final_codes_kernel(const __half* in,unsigned char* out,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=rdna2_nr::half_bits_to_e4m3fn(__half_as_ushort(in[i]));}
__global__ void partition_markers(int* windows,int h,int w,int top,int left,int ph,int pw){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ph*pw)return;int ix=i%8,t=i/8,iy=t%8;t/=8;int wx=t%(pw/8),wy=t/(pw/8);int y=wy*8+iy-top,x=wx*8+ix-left;windows[i]=(y>=0&&y<h&&x>=0&&x<w)?y*w+x:-1;}
__global__ void reverse_markers(const int* windows,int* image,int h,int w,int top,int left,int pw){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h*w)return;int y=i/w+top,x=i%w+left;int wi=((y/8)*(pw/8)+x/8)*64+(y%8)*8+x%8;image[i]=windows[wi];}

static void launch_gemm(const __half*a,const __half*b,float*c,int m,int k,int n){gemm_hf_tiled<<<dim3((n+31)/32,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,b,c,m,k,n);sync_gpu();}
inline void launch_adapter_16x32(const __half*a,const __half*b,__half*c,int m){adapter_16x32_half<<<dim3(1,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,b,c,m);sync_gpu();}
inline void launch_fused_c32_ffn(const __half*input,const __half*expandWeight,
                                 const __half*contractWeight,const __half*cosine,
                                 __half*output,int rows,bool packed=false){
    if(packed)fused_c32_ffn_packed<><<<(rows+15)/16,dim3(32,4),0,current_stream()>>>(
        input,expandWeight,contractWeight,cosine,output,rows);
    else fused_c32_ffn<><<<(rows+15)/16,dim3(32,4),0,current_stream()>>>(
        input,expandWeight,contractWeight,cosine,output,rows);
    sync_gpu();
}
__global__ void pack_c32_attention_half2(const __half* input,__half2* output,
                                          unsigned columns) {
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=16*columns)return;
    const unsigned pair=index/columns,column=index%columns;
    output[index]=__halves2half2(input[(pair*2)*columns+column],
                                 input[(pair*2+1)*columns+column]);
}
inline void launch_fused_c32_ffn_compact(const __half*input,const __half*expandWeight,
                                         const __half*contractWeight,const __half*cosine,
                                         __half*output,int rows,int hidden) {
    const dim3 grid((rows+15)/16),threads(32,4);
    if(hidden==32)fused_c32_ffn_compact<32><<<grid,threads,0,current_stream()>>>(input,expandWeight,contractWeight,cosine,output,rows);
    else if(hidden==64)fused_c32_ffn_compact<64><<<grid,threads,0,current_stream()>>>(input,expandWeight,contractWeight,cosine,output,rows);
    else if(hidden==96)fused_c32_ffn_compact<96><<<grid,threads,0,current_stream()>>>(input,expandWeight,contractWeight,cosine,output,rows);
    else throw std::runtime_error("unsupported compact C32 width");
    sync_gpu();
}
inline void launch_compact_branched_expand(const __half*input,const __half*weight,
                                            __half*hidden,int channels,int rows,int branches){
    const unsigned count=unsigned(size_t(rows)*(channels/32)*branches*32),blocks=(count+255)/256;
    if(branches==1)compact_branched_expand_rows<1><<<blocks,256,0,current_stream()>>>(input,weight,hidden,channels,rows);
    else if(branches==2)compact_branched_expand_rows<2><<<blocks,256,0,current_stream()>>>(input,weight,hidden,channels,rows);
    else if(branches==3)compact_branched_expand_rows<3><<<blocks,256,0,current_stream()>>>(input,weight,hidden,channels,rows);
    else throw std::runtime_error("unsupported compact branch count");
}
inline void launch_compact_branched_project_merge(const __half*hidden,const __half*weight,
                                                   __half*combined,int channels,int rows,int branches){
    const unsigned count=unsigned(size_t(rows)*channels),blocks=(count+255)/256;
    if(branches==1)compact_branched_project_merge_rows<1><<<blocks,256,0,current_stream()>>>(hidden,weight,combined,channels,rows);
    else if(branches==2)compact_branched_project_merge_rows<2><<<blocks,256,0,current_stream()>>>(hidden,weight,combined,channels,rows);
    else if(branches==3)compact_branched_project_merge_rows<3><<<blocks,256,0,current_stream()>>>(hidden,weight,combined,channels,rows);
    else throw std::runtime_error("unsupported compact branch count");
}
inline void launch_gemm_half(const __half*a,const __half*b,__half*c,int m,int k,int n,bool publish){const dim3 grid((n+31)/32,(m+15)/16);if(publish)gemm_hf_tiled<__half,true><<<grid,dim3(32,16),0,current_stream()>>>(a,b,c,m,k,n);else gemm_hf_tiled<__half,false><<<grid,dim3(32,16),0,current_stream()>>>(a,b,c,m,k,n);sync_gpu();}
inline void launch_gemm_gate_publish(const __half*a,const __half*b,__half*c,int m,int k,int n){const dim3 grid((n+31)/32,(m+15)/16);if(k==32&&n==128&&m>=8192)gemm_hf_register<4,1><<<grid,dim3(32,4),0,current_stream()>>>(a,b,nullptr,nullptr,c,m,k,n,false);else gemm_hf_tiled_gate_publish<<<grid,dim3(32,16),0,current_stream()>>>(a,b,c,m,k,n);sync_gpu();}
inline void launch_gemm_publish(const __half*a,const __half*b,__half*c,int m,int k,int n){gemm_hf_tiled_publish<<<dim3((n+31)/32,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,b,c,m,k,n);sync_gpu();}
inline void launch_gemm_qkv_normalize(const __half*a,const __half*b,const __half*scale,__half*c,int m,int k,int heads,int tokensPerWindow){gemm_hf_tiled_qkv_normalize<<<dim3(3*heads,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,b,scale,c,m,k,heads,tokensPerWindow,0,0,0,0,0);sync_gpu();}
inline void launch_qkv_normalize_half(const __half*a,const __half*scale,__half*c,int m,int heads,int tokensPerWindow){normalize_qkv_half<<<dim3(3*heads,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,scale,c,m,heads,tokensPerWindow);sync_gpu();}
inline void launch_gemm_qkv_normalize_window(const __half*a,const __half*b,const __half*scale,__half*c,int m,int k,int heads,int h,int w,int top,int left,int paddedWidth){gemm_hf_tiled_qkv_normalize<<<dim3(3*heads,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,b,scale,c,m,k,heads,64,h,w,top,left,paddedWidth);sync_gpu();}
inline void launch_gemm_residual(const __half*a,const __half*b,const __half*skip,const __half*cosine,__half*out,int m,int k,int n,bool publish,__half* raw=nullptr){const dim3 grid((n+31)/32,(m+15)/16);if(!raw&&k==128&&n==32&&m>=8192)gemm_hf_register<4,3><<<grid,dim3(32,4),0,current_stream()>>>(a,b,skip,cosine,out,m,k,n,publish);else gemm_hf_tiled_residual<<<grid,dim3(32,16),0,current_stream()>>>(a,b,skip,cosine,out,m,k,n,publish,raw);sync_gpu();}
inline void launch_gemm_residual_to_spatial(const __half*a,const __half*b,const __half*skip,const __half*cosine,__half*out,int m,int k,int n,bool publish,int h,int w,int top,int left,int paddedWidth){gemm_hf_tiled_residual_to_spatial<<<dim3((n+31)/32,(m+15)/16),dim3(32,16),0,current_stream()>>>(a,b,skip,cosine,out,m,k,n,publish,h,w,top,left,paddedWidth);sync_gpu();}
inline void launch_fused_c32_attention(const __half*input,const __half*qkvWeight,
                                       const __half*bias,const __half*scale,
                                       const __half*projectionWeight,const __half*cosine,
                                       __half*output,int windows,int h,int w,int top,
                                       int left,int paddedWidth,bool publish,bool parallel=false,
                                       bool padded=false,bool waveProjection=false){
    if(waveProjection&&parallel&&padded)fused_c32_attention<true,66,34,true,true><<<windows,256,0,current_stream()>>>(input,qkvWeight,bias,scale,projectionWeight,cosine,output,h,w,top,left,paddedWidth,publish);
    else if(waveProjection&&parallel)fused_c32_attention<true,64,32,true,true><<<windows,256,0,current_stream()>>>(input,qkvWeight,bias,scale,projectionWeight,cosine,output,h,w,top,left,paddedWidth,publish);
    else if(waveProjection)fused_c32_attention<false,64,32,true,true><<<windows,256,0,current_stream()>>>(input,qkvWeight,bias,scale,projectionWeight,cosine,output,h,w,top,left,paddedWidth,publish);
    else if(parallel&&padded)fused_c32_attention<true,66,34><<<windows,256,0,current_stream()>>>(input,qkvWeight,bias,scale,projectionWeight,cosine,output,h,w,top,left,paddedWidth,publish);
    else if(parallel)fused_c32_attention<true><<<windows,256,0,current_stream()>>>(input,qkvWeight,bias,scale,projectionWeight,cosine,output,h,w,top,left,paddedWidth,publish);
    else fused_c32_attention<false><<<windows,256,0,current_stream()>>>(input,qkvWeight,bias,scale,
        projectionWeight,cosine,output,h,w,top,left,paddedWidth,publish);
    sync_gpu();
}
