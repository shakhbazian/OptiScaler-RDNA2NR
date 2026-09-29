#pragma once
#include "nr_int8_runtime.h"

namespace rdna2_nr {
// Full global FFN: retain half/gate/publication or half residual boundaries,
// then produce exactly the same group64 codes/scales for the next operation.
// Two columns per lane cover the entire 64-channel scale domain.
template<unsigned TileRows,bool Quantize,bool StoreHalf=true>
__global__ void dot4_global_expand_handoff(const std::int8_t* a,const std::int8_t* w,
    const float* sa,const float* sw,__half* out,std::int8_t* codes,float* scales,
    unsigned m,unsigned k,unsigned n,unsigned* status){
    constexpr unsigned Rows=4,RowThreads=TileRows/Rows,Threads=32*RowThreads;
    __shared__ std::int32_t tileA[TileRows][16],tileW[2][16][33];
    const unsigned lane=threadIdx.x,local=threadIdx.y*32+lane,baseColumn=blockIdx.x*64;
    float total[Rows][2]={};
    for(unsigned group=0;group<k/64;++group){
        for(unsigned i=local;i<TileRows*16;i+=Threads){
            const unsigned r=i/16,g=i%16,row=blockIdx.y*TileRows+r;
            tileA[r][g]=row<m?reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+group*64)[g]:0;
        }
        for(unsigned i=local;i<2*16*32;i+=Threads){
            const unsigned pair=i/(16*32),g=(i/32)%16,c=i%32,column=baseColumn+pair*32+c;
            tileW[pair][g][c]=reinterpret_cast<const std::int32_t*>(w)[
                size_t(column/32)*(k/4)*32+size_t(group*16+g)*32+column%32];
        }
        __syncthreads();
        int partial[Rows][2]={};
        #pragma unroll 4
        for(unsigned g=0;g<16;++g){
            const char4 b0=*reinterpret_cast<const char4*>(&tileW[0][g][lane]);
            const char4 b1=*reinterpret_cast<const char4*>(&tileW[1][g][lane]);
            #pragma unroll
            for(unsigned r=0;r<Rows;++r){
                const char4 av=*reinterpret_cast<const char4*>(&tileA[threadIdx.y+r*RowThreads][g]);
                partial[r][0]=amd_mixed_dot(av,b0,partial[r][0],false);
                partial[r][1]=amd_mixed_dot(av,b1,partial[r][1],false);
            }
        }
        __syncthreads();
        #pragma unroll
        for(unsigned r=0;r<Rows;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<m){
                #pragma unroll
                for(unsigned p=0;p<2;++p){
                    const float combined=sa[size_t(row)*(k/64)+group]*sw[size_t(group)*n+baseColumn+p*32+lane];
                    total[r][p]+=float(partial[r][p])*combined;
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<Rows;++r){
        const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
        if(row<m){
            const __half h0=pubh(ffn_gate(__float2half(total[r][0])));
            const __half h1=pubh(ffn_gate(__float2half(total[r][1])));
            const size_t offset=size_t(row)*n+baseColumn+lane;
            if constexpr(StoreHalf){out[offset]=h0;out[offset+32]=h1;}
            if constexpr(Quantize){
                const float x=__half2float(h0),y=__half2float(h1);
                if(!isfinite(x)||!isfinite(y))atomicOr(status,1u);
                float maximum=fmaxf(fmaxf(fabsf(x),fabsf(y)),0.f);
                for(unsigned stride=16;stride;stride>>=1)
                    maximum=fmaxf(maximum,__shfl_down(maximum,stride,32));
                maximum=__shfl(maximum,0,32);
                const float s=maximum==0?1.f:maximum/127.f,inv=maximum==0?1.f:127.f/maximum;
                if(lane==0)scales[size_t(row)*(n/64)+blockIdx.x]=s;
                codes[offset]=std::int8_t(max(-127,min(127,__float2int_rn(x*inv))));
                codes[offset+32]=std::int8_t(max(-127,min(127,__float2int_rn(y*inv))));
            }
        }
    }
}
template<unsigned TileRows,bool Quantize,bool StoreHalf=true>
__global__ void dot4_global_residual_handoff(const std::int8_t* a,const std::int8_t* w,
    const float* sa,const float* sw,__half* out,std::int8_t* codes,float* scales,
    unsigned m,unsigned k,unsigned n,unsigned* status,const __half*skip,const __half*cosine){
    constexpr unsigned Rows=4,RowThreads=TileRows/Rows,Threads=32*RowThreads;
    __shared__ std::int32_t tileA[TileRows][16],tileW[2][16][33];
    const unsigned lane=threadIdx.x,local=threadIdx.y*32+lane,baseColumn=blockIdx.x*64;
    float total[Rows][2]={};
    for(unsigned group=0;group<k/64;++group){
        for(unsigned i=local;i<TileRows*16;i+=Threads){
            const unsigned r=i/16,g=i%16,row=blockIdx.y*TileRows+r;
            tileA[r][g]=row<m?reinterpret_cast<const std::int32_t*>(a+size_t(row)*k+group*64)[g]:0;
        }
        for(unsigned i=local;i<2*16*32;i+=Threads){
            const unsigned pair=i/(16*32),g=(i/32)%16,c=i%32,column=baseColumn+pair*32+c;
            tileW[pair][g][c]=reinterpret_cast<const std::int32_t*>(w)[
                size_t(column/32)*(k/4)*32+size_t(group*16+g)*32+column%32];
        }
        __syncthreads();
        int partial[Rows][2]={};
        #pragma unroll 4
        for(unsigned g=0;g<16;++g){
            const char4 b0=*reinterpret_cast<const char4*>(&tileW[0][g][lane]);
            const char4 b1=*reinterpret_cast<const char4*>(&tileW[1][g][lane]);
            #pragma unroll
            for(unsigned r=0;r<Rows;++r){
                const char4 av=*reinterpret_cast<const char4*>(&tileA[threadIdx.y+r*RowThreads][g]);
                partial[r][0]=amd_mixed_dot(av,b0,partial[r][0],false);
                partial[r][1]=amd_mixed_dot(av,b1,partial[r][1],false);
            }
        }
        __syncthreads();
        #pragma unroll
        for(unsigned r=0;r<Rows;++r){
            const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
            if(row<m){
                #pragma unroll
                for(unsigned p=0;p<2;++p){
                    const float combined=sa[size_t(row)*(k/64)+group]*sw[size_t(group)*n+baseColumn+p*32+lane];
                    total[r][p]+=float(partial[r][p])*combined;
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<Rows;++r){
        const unsigned row=blockIdx.y*TileRows+threadIdx.y+r*RowThreads;
        if(row<m){
            const size_t offset=size_t(row)*n+baseColumn+lane;
            const __half h0=__hadd(__float2half(total[r][0]),__hmul(skip[offset],cosine[baseColumn+lane]));
            const __half h1=__hadd(__float2half(total[r][1]),__hmul(skip[offset+32],cosine[baseColumn+lane+32]));
            if constexpr(StoreHalf){out[offset]=h0;out[offset+32]=h1;}
            if constexpr(Quantize){
                const float x=__half2float(h0),y=__half2float(h1);
                if(!isfinite(x)||!isfinite(y))atomicOr(status,1u);
                float maximum=fmaxf(fmaxf(fabsf(x),fabsf(y)),0.f);
                for(unsigned stride=16;stride;stride>>=1)
                    maximum=fmaxf(maximum,__shfl_down(maximum,stride,32));
                maximum=__shfl(maximum,0,32);
                const float s=maximum==0?1.f:maximum/127.f,inv=maximum==0?1.f:127.f/maximum;
                if(lane==0)scales[size_t(row)*(n/64)+blockIdx.x]=s;
                codes[offset]=std::int8_t(max(-127,min(127,__float2int_rn(x*inv))));
                codes[offset+32]=std::int8_t(max(-127,min(127,__float2int_rn(y*inv))));
            }
        }
    }
}
}
