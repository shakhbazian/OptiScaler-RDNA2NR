#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../include/nr_gpu_scheduled_state.h"

namespace {
using namespace NrScheduled;
__global__ void Begin(State* state, Frame frame, Decision* decision) {
    if (threadIdx.x || blockIdx.x) return;
    const bool consecutive=state->valid && state->committedStream==frame.streamId &&
        state->committedFrame!=UINT64_MAX && frame.frameIndex==state->committedFrame+1;
    if (!consecutive || frame.reset || !frame.hasMotion) {
        state->valid=0; state->noise=0;
    }
    // Keep the seed cadence and transactional commits while SR owns temporal
    // accumulation. A policy switch supplies reset once, not every frame.
    decision->useHistory=state->valid && !frame.disableAccumulation;
    decision->noise=state->noise;
}
__global__ void Resolve(const unsigned char* color,std::size_t colorPitch,
    unsigned colorWidth,unsigned colorHeight,unsigned char* output,std::size_t outputPitch,
    const float* candidate,float* committed,unsigned width,unsigned height,
    const std::uint32_t* invalid,State* state,Frame frame) {
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=width*height)return;
    const bool good=*invalid==0;
    const unsigned x=index%width,y=index/width;
    if(good) {
        for(unsigned c=0;c<3;++c)committed[index*3+c]=candidate[index*3+c];
    } else {
        const unsigned cx=min(colorWidth-1,static_cast<unsigned>((2ull*x+1)*colorWidth/(2ull*width)));
        const unsigned cy=min(colorHeight-1,static_cast<unsigned>((2ull*y+1)*colorHeight/(2ull*height)));
        const auto* src=reinterpret_cast<const __half*>(color+std::size_t(cy)*colorPitch)+cx*4;
        auto* dst=reinterpret_cast<__half*>(output+std::size_t(y)*outputPitch)+x*4;
        for(unsigned c=0;c<3;++c)dst[c]=src[c];
        dst[3]=__float2half(1.f);
    }
    if(index==0) {
        state->lastInvalid=*invalid;
        if(good) {
            state->committedStream=frame.streamId;
            state->committedFrame=frame.frameIndex;
            state->valid=1; ++state->noise; ++state->commits;
        } else {
            state->valid=0; state->noise=0;
        }
    }
}
} // namespace

extern "C" hipError_t launch_scheduled_begin(NrScheduled::State* state,
    NrScheduled::Frame frame,NrScheduled::Decision* decision,hipStream_t stream) {
    Begin<<<1,1,0,stream>>>(state,frame,decision);
    return hipGetLastError();
}
extern "C" hipError_t launch_scheduled_resolve(const void* color,std::size_t colorPitch,
    unsigned colorWidth,unsigned colorHeight,void* output,std::size_t outputPitch,
    const float* candidate,float* committed,unsigned width,unsigned height,
    const std::uint32_t* invalid,NrScheduled::State* state,NrScheduled::Frame frame,hipStream_t stream) {
    const unsigned pixels=width*height;
    Resolve<<<(pixels+255)/256,256,0,stream>>>(static_cast<const unsigned char*>(color),colorPitch,
        colorWidth,colorHeight,static_cast<unsigned char*>(output),outputPitch,candidate,committed,
        width,height,invalid,state,frame);
    return hipGetLastError();
}
