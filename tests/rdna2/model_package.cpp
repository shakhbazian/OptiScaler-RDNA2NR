#include "../../rdna2_nr/include/nr_weight_package.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <cstring>

static void Reject(const std::function<void()>& test,const char* message) {
    try {test();}catch(const std::exception& e){
        if(std::string(e.what()).find(message)==std::string::npos)throw;
        std::cout<<"PASS reject "<<message<<'\n';return;
    }
    throw std::runtime_error(std::string("accepted invalid package: ")+message);
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("usage: model_package model-a.nrwgt model-b.nrwgt");
    const auto source=NrV2::ParseSha256("2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F");
    for(unsigned model=1;model<=2;++model){
        std::ifstream input(argv[model],std::ios::binary);
        std::vector<std::uint8_t> raw((std::istreambuf_iterator<char>(input)),{});
        auto hash=NrV2::Sha256(raw.data(),raw.size());
        auto package=NrV2::WeightPackage::LoadMemory(raw,hash,source);
        if(package.GraphVersion()!=model)throw std::runtime_error("wrong graph identity");
        Reject([&]{auto wrong=hash;wrong[0]^=1;NrV2::WeightPackage::LoadMemory(raw,wrong,source);},"package SHA-256");
        Reject([&]{auto wrong=source;wrong[0]^=1;NrV2::WeightPackage::LoadMemory(raw,hash,wrong);},"source SHA-256");
        Reject([&]{NrV2::WeightPackage::LoadMemory(std::vector<std::uint8_t>(127),hash,source);},"Truncated");
        Reject([&]{auto copy=raw;copy[12]=2;NrV2::WeightPackage::LoadMemory(copy,NrV2::Sha256(copy.data(),copy.size()),source);},"graph or math");
        Reject([&]{auto copy=raw;copy[24]^=1;NrV2::WeightPackage::LoadMemory(copy,NrV2::Sha256(copy.data(),copy.size()),source);},"manifest size");
        Reject([&]{auto copy=raw;copy.back()^=1;NrV2::WeightPackage::LoadMemory(copy,NrV2::Sha256(copy.data(),copy.size()),source);},"payload SHA-256");
        if(model==1){
            // Re-sign the table so the common parser, rather than its checksum,
            // rejects a corrupted dtype, rank, shape and role.
            for(auto item:{std::pair<unsigned,const char*>{132,"tensor type"},{136,"tensor type"},
                           {144,"tensor shape"},{168,"role termination"}}){
                auto copy=raw;std::memset(copy.data()+item.first,0,4);
                if(item.first==132)copy[item.first]=2;
                auto table=NrV2::Sha256(copy.data()+128,641*128);
                std::memcpy(copy.data()+64,table.data(),32);
                Reject([&]{NrV2::WeightPackage::LoadMemory(copy,NrV2::Sha256(copy.data(),copy.size()),source);},item.second);
            }
        }
    }
    std::cout<<"PASS A/B package validation\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
