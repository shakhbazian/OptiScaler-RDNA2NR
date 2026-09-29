#pragma once
#include "nr_int8_runtime.h"

namespace rdna2_nr {

// The four 64-channel scale domains of the physical C512 split FFN are
// consumed in order. No intermediate half/INT8 tensor leaves the workgroup.
template<unsigned TileRows,unsigned WaveCount,bool TwoWeightTiles>
__global__ void split_ffn_resident_w8a8(
    const std::int8_t* input,const float* inputScale,
    const std::int8_t* expand,const float* expandScale,
    const std::int8_t* project,const float* projectScale,
    __half* output,unsigned rows,unsigned* status) {
    static_assert((TileRows==8||TileRows==16)&&(WaveCount==4||WaveCount==8)&&TileRows%WaveCount==0,
                  "supported resident row tile");
    constexpr unsigned LocalRows=TileRows/WaveCount,WeightTiles=TwoWeightTiles?2:1;
    __shared__ std::int32_t tileA[TileRows][16],tileH[TileRows][16];
    __shared__ float scaleA[TileRows],scaleH[TileRows];
    __shared__ std::int32_t tileW[WeightTiles][16][2][33];
    __shared__ float scaleW[WeightTiles][2][32];
    const unsigned lane=threadIdx.x,wave=threadIdx.y,tid=wave*32+lane;
    const unsigned batch=blockIdx.y,firstRow=blockIdx.x*TileRows;
    constexpr unsigned ProjectTile=TwoWeightTiles?1:0;
    for(unsigned index=tid;index<TileRows*16;index+=WaveCount*32){
        const unsigned localRow=index/16,d=index%16,row=firstRow+localRow;
        tileA[localRow][d]=row<rows?*reinterpret_cast<const std::int32_t*>(
            input+(size_t(row)*8+batch)*64+d*4):0;
    }
    for(unsigned localRow=tid;localRow<TileRows;localRow+=WaveCount*32){
        const unsigned row=firstRow+localRow;
        scaleA[localRow]=row<rows?inputScale[size_t(row)*8+batch]:1.f;
    }
    float totals[LocalRows][2]={};
    for(unsigned group=0;group<4;++group){
        for(unsigned index=tid;index<1024;index+=WaveCount*32){
            const unsigned d=index%16,column=index/16,parity=column&1,halfColumn=column/2;
            tileW[0][d][parity][halfColumn]=*reinterpret_cast<const std::int32_t*>(
                expand+(size_t(batch)*256+group*64+column)*64+d*4);
            if(d==0)scaleW[0][parity][halfColumn]=expandScale[size_t(batch)*256+group*64+column];
            if constexpr(TwoWeightTiles){
                tileW[1][d][parity][halfColumn]=*reinterpret_cast<const std::int32_t*>(
                    project+(size_t(batch)*64+column)*256+group*64+d*4);
                if(d==0)scaleW[1][parity][halfColumn]=projectScale[(size_t(batch)*4+group)*64+column];
            }
        }
        __syncthreads();
        #pragma unroll
        for(unsigned t=0;t<LocalRows;++t){
            const unsigned localRow=wave+WaveCount*t;
            std::int32_t partial0=0,partial1=0;
            #pragma unroll
            for(unsigned d=0;d<16;++d){
                const char4 a=*reinterpret_cast<const char4*>(&tileA[localRow][d]);
                const char4 w0=*reinterpret_cast<const char4*>(&tileW[0][d][0][lane]);
                const char4 w1=*reinterpret_cast<const char4*>(&tileW[0][d][1][lane]);
                partial0=amd_mixed_dot(a,w0,partial0,false);
                partial1=amd_mixed_dot(a,w1,partial1,false);
            }
            const float combined0=scaleA[localRow]*scaleW[0][0][lane];
            const float combined1=scaleA[localRow]*scaleW[0][1][lane];
            float expanded0=0.f,expanded1=0.f;
            expanded0+=float(partial0)*combined0;
            expanded1+=float(partial1)*combined1;
            const __half h0=ffn_gate(__float2half(expanded0));
            const __half h1=ffn_gate(__float2half(expanded1));
            const float v0=__half2float(h0),v1=__half2float(h1);
            if(!isfinite(v0)||!isfinite(v1))atomicOr(status,1u);
            float maximum=fmaxf(fabsf(v0),fabsf(v1));
            for(unsigned stride=16;stride;stride>>=1)
                maximum=fmaxf(maximum,__shfl_down(maximum,stride,32));
            maximum=__shfl(maximum,0,32);
            const float s=maximum==0?1.f:maximum/127.f;
            const float inv=maximum==0?1.f:127.f/maximum;
            const unsigned c0=unsigned(std::uint8_t(max(-127,min(127,__float2int_rn(v0*inv)))));
            const unsigned c1=unsigned(std::uint8_t(max(-127,min(127,__float2int_rn(v1*inv)))));
            const unsigned pair=c0|(c1<<8);
            const unsigned other=__shfl_xor(pair,1,32);
            if(!(lane&1))tileH[localRow][lane/2]=std::int32_t(pair|(other<<16));
            if(lane==0)scaleH[localRow]=s;
        }
        __syncthreads();
        if constexpr(!TwoWeightTiles){
            for(unsigned index=tid;index<1024;index+=WaveCount*32){
                const unsigned d=index%16,column=index/16,parity=column&1,halfColumn=column/2;
                tileW[0][d][parity][halfColumn]=*reinterpret_cast<const std::int32_t*>(
                    project+(size_t(batch)*64+column)*256+group*64+d*4);
                if(d==0)scaleW[0][parity][halfColumn]=projectScale[(size_t(batch)*4+group)*64+column];
            }
            __syncthreads();
        }
        #pragma unroll
        for(unsigned t=0;t<LocalRows;++t){
            const unsigned localRow=wave+WaveCount*t;
            std::int32_t partial0=0,partial1=0;
            #pragma unroll
            for(unsigned d=0;d<16;++d){
                const char4 a=*reinterpret_cast<const char4*>(&tileH[localRow][d]);
                const char4 w0=*reinterpret_cast<const char4*>(&tileW[ProjectTile][d][0][lane]);
                const char4 w1=*reinterpret_cast<const char4*>(&tileW[ProjectTile][d][1][lane]);
                partial0=amd_mixed_dot(a,w0,partial0,false);
                partial1=amd_mixed_dot(a,w1,partial1,false);
            }
            const float combined0=scaleH[localRow]*scaleW[ProjectTile][0][lane];
            const float combined1=scaleH[localRow]*scaleW[ProjectTile][1][lane];
            totals[t][0]+=float(partial0)*combined0;
            totals[t][1]+=float(partial1)*combined1;
        }
        if(group<3)__syncthreads();
    }
    #pragma unroll
    for(unsigned t=0;t<LocalRows;++t){
        const unsigned row=firstRow+wave+WaveCount*t;
        if(row<rows){
            const size_t base=(size_t(row)*8+batch)*64+2*lane;
            output[base]=pubh(__float2half(totals[t][0]));
            output[base+1]=pubh(__float2half(totals[t][1]));
        }
    }
}

} // namespace rdna2_nr
