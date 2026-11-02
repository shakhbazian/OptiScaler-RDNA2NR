#pragma once
#include <cstdint>
#include <stdexcept>

namespace NrModel {
struct Block {
    int channels, family, kind, skipStore, skipLoad, originY, originX;
};
// Model B phases, edges and extent policy adapted from DLSSNR-AMD 118451d6.
// Copyright (c) 2026 mochizuki0323. MIT; see Licenses/DLSSNR-AMD_LICENSE.txt.
// Origins use our NHWC height/width order, in pixels, not the descriptor axes.
inline constexpr Block OptimizedBlocks[] = {
    {32, 0, 3, -1, -1, 0, 0}, // 0
    {32, 0, 0, -1, -1, 0, 0}, // 1
    {32, 0, 1, 0, -1, -4, -4}, // 2
    {64, 1, 0, -1, -1, 0, 0}, // 3
    {64, 1, 1, 1, -1, -4, -4}, // 4
    {128, 1, 0, -1, -1, 0, 0}, // 5
    {128, 1, 1, 2, -1, -4, -4}, // 6
    {256, 1, 0, -1, -1, 0, 0}, // 7
    {256, 1, 1, 3, -1, -4, -4}, // 8
    {512, 2, 0, -1, -1, 0, 0}, // 9
    {512, 2, 0, -1, -1, -4, -4}, // 10
    {1024, 3, 0, -1, -1, 0, 0}, // 11
    {1024, 3, 0, -1, -1, 0, 0}, // 12
    {1024, 3, 0, -1, -1, 0, 0}, // 13
    {1024, 3, 0, -1, -1, 0, 0}, // 14
    {1024, 3, 0, -1, -1, 0, 0}, // 15
    {1024, 3, 0, -1, -1, 0, 0}, // 16
    {1024, 3, 0, -1, -1, 0, 0}, // 17
    {1024, 3, 0, -1, -1, 0, 0}, // 18
    {512, 4, 0, -1, -1, 0, 0}, // 19
    {512, 2, 0, -1, -1, 0, -4}, // 20
    {512, 2, 0, -1, -1, -4, 0}, // 21
    {256, 1, 2, -1, 3, 0, -4}, // 22
    {256, 1, 0, -1, -1, -4, 0}, // 23
    {128, 1, 2, -1, 2, 0, -4}, // 24
    {128, 1, 0, -1, -1, -4, 0}, // 25
    {64, 1, 2, -1, 1, 0, -4}, // 26
    {64, 1, 0, -1, -1, -4, 0}, // 27
    {32, 0, 2, -1, 0, 0, -4}, // 28
    {32, 0, 0, -1, -1, -4, 0}, // 29
    {32, 0, 4, -1, -1, -4, -4}, // 30
};
struct Spec {
    unsigned version, blocks, lastDown, globalInput, post;
    Block At(unsigned b) const {
        if(b>=blocks)throw std::out_of_range("model block");
        if(version==2)return OptimizedBlocks[b];
        int c=32,family=0,kind=0;
        if((b>=5&&b<=8)||(b>=62&&b<=65)){c=64;family=1;}
        else if((b>=9&&b<=14)||(b>=56&&b<=61)){c=128;family=1;}
        else if((b>=15&&b<=22)||(b>=48&&b<=55)){c=256;family=1;}
        else if((b>=23&&b<=30)||(b>=40&&b<=47)){c=512;family=2;}
        else if(b>=31&&b<=38){c=1024;family=3;}
        else if(b==39){c=512;family=4;}
        if(b==0)kind=3;else if(b==70)kind=4;
        else if(b==4||b==8||b==14||b==22)kind=1;
        else if(b==48||b==56||b==62||b==66)kind=2;
        int phase=0;
        if(b==70)phase=1;
        else if(b!=0){struct R{unsigned lo,hi,base;};
            const R ranges[]={{1,4,1},{5,8,5},{9,14,9},{15,22,15},{23,30,23},
                {40,47,40},{48,55,48},{56,61,54},{62,65,62},{66,69,66}};
            for(auto r:ranges)if(b>=r.lo&&b<=r.hi){phase=int(b-r.base);break;}}
        const int origins[][2]={{0,0},{-4,-4},{0,-4},{-4,0}};
        const int s=b==3?0:b==7?1:b==13?2:b==21?3:-1;
        const int l=b==66?0:b==62?1:b==56?2:b==48?3:-1;
        return {c,family,kind,s,l,origins[phase%4][0],origins[phase%4][1]};
    }
};
inline Spec Get(unsigned version) {
    if(version==1)return {1,71,22,30,70};
    if(version==2)return {2,31,8,10,30};
    throw std::invalid_argument("unknown model graph");
}
struct Extent { unsigned width,height; };
inline Extent WorkingExtent(unsigned version,unsigned width,unsigned height) {
    if(!width||!height||width>16384||height>16384)throw std::invalid_argument("model extent");
    auto up=[](unsigned n,unsigned a){return (n+a-1)/a*a;};
    if(version==1)return {up(width<320?320:width,64),up(height<320?320:height,64)};
    if(version!=2)throw std::invalid_argument("unknown model graph");
    const unsigned ax=up((width+1)/2,4)%8==4?128:64;
    const unsigned ay=up((height+1)/2,4)%8==4?128:64;
    auto w=up(width,ax),h=up(height,ay);w=w<320?320:w;h=h<320?320:h;
    if(!(h%(ay*4))&&!(w%(ax*4)))w+=ax;
    if(w>16384||h>16384)throw std::invalid_argument("padded model extent");
    return {w,h};
}
} // namespace NrModel
