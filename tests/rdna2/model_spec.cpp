#include "../../rdna2_nr/include/nr_model_spec.h"
#include <cstdio>
#include <stdexcept>

static void Require(bool value) { if(!value) throw std::runtime_error("model specification regression"); }
int main() {
    const auto b=NrModel::Get(2),a=NrModel::Get(1);
    Require(a.blocks==71&&b.blocks==31&&a.post==70&&b.post==30);
    int available[4]={-1,-1,-1,-1};
    for(unsigned i=0;i<b.blocks;++i){
        const auto block=b.At(i);
        if(block.skipStore>=0){
            Require(block.kind==1); // Store the down-block result, before pooling.
            Require(available[block.skipStore]==-1);
            available[block.skipStore]=block.channels;
        }
        if(block.skipLoad>=0){
            Require(block.kind==2&&available[block.skipLoad]==block.channels);
            available[block.skipLoad]=-1;
        }
    }
    for(int value:available)Require(value==-1);
    Require(a.At(3).skipStore==0&&a.At(4).skipStore==-1); // Preserve release A.
    Require(b.At(20).originY==0&&b.At(20).originX==-4);
    Require(b.At(21).originY==-4&&b.At(21).originX==0);
    struct Case {unsigned w,h,aw,ah,bw,bh;};
    for(auto c:{Case{854,1080,896,1088,896,1152},Case{1920,1080,1920,1088,1920,1152},
                Case{1280,720,1280,768,1344,768},Case{960,540,960,576,960,576},
                Case{2560,1440,2560,1472,2560,1472},Case{3840,2160,3840,2176,3840,2176},
                Case{253,320,320,320,320,320}}){
        const auto x=NrModel::WorkingExtent(1,c.w,c.h),y=NrModel::WorkingExtent(2,c.w,c.h);
        Require(x.width==c.aw&&x.height==c.ah&&y.width==c.bw&&y.height==c.bh);
    }
    bool caught=false;try{NrModel::WorkingExtent(2,16384,16384);}catch(const std::invalid_argument&){caught=true;}
    Require(caught);caught=false;try{b.At(31);}catch(const std::out_of_range&){caught=true;}
    Require(caught);std::puts("PASS model graph edges, phases, extents and bounds");
}
