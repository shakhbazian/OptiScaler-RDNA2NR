#include "../../rdna2_nr/include/nr_model_catalog.h"
#include "../../rdna2_nr/include/nr_runtime_v3_validation.h"
#include <cstdio>
#include <stdexcept>

static void Require(bool value){if(!value)throw std::runtime_error("model integration metadata regression");}
int main(){
    for(const auto& identity:NrModel::Packages){
        unsigned char header[128]{};std::memcpy(header,"NRWGT001",8);std::memcpy(header+8,&identity.graph,4);
        auto nibble=[](char c){return c<='9'?c-'0':c-'A'+10;};
        for(unsigned i=0;i<32;++i)header[32+i]=static_cast<unsigned char>((nibble(identity.sourceHash[i*2])<<4)|nibble(identity.sourceHash[i*2+1]));
        Require(NrModel::Identify(header,identity.bytes)==&identity);
        Require(!NrModel::Identify(header,identity.bytes-1));
        header[32]^=1;Require(!NrModel::Identify(header,identity.bytes));header[32]^=1;
        NrV3::Config c{};c.prefix={sizeof(c),NrV3::Version};c.sourceHash.bytes[0]=c.packageHash.bytes[0]=1;
        c.graphVersion=identity.graph;c.mathContract=NrV2::MathContract;
        for(auto extent:{NrModel::Extent{960,540},NrModel::Extent{1280,720},NrModel::Extent{854,1080},NrModel::Extent{3840,2160}}){
            c.logicalWidth=extent.width;c.logicalHeight=extent.height;
            const auto work=NrModel::WorkingExtent(c.graphVersion,c.logicalWidth,c.logicalHeight);
            c.networkWidth=work.width;c.networkHeight=work.height;Require(NrV3::Validate(c)==NrV3::Status::Ok);
            ++c.networkWidth;Require(NrV3::Validate(c)==NrV3::Status::InvalidArgument);--c.networkWidth;
            if(identity.graph==2&&(work.width!=NrV2::AlignNetworkExtent(extent.width)||work.height!=NrV2::AlignNetworkExtent(extent.height))){
                c.networkWidth=NrV2::AlignNetworkExtent(extent.width);c.networkHeight=NrV2::AlignNetworkExtent(extent.height);
                Require(NrV3::Validate(c)==NrV3::Status::InvalidArgument);
            }
        }
        c.graphVersion=3;Require(NrV3::Validate(c)==NrV3::Status::Unsupported);
    }
    std::puts("PASS A/B catalog identities, graph-specific padding and rejection of mismatched configurations");
}
