#pragma once
// Generated from tools/model_converter/model_catalog.json by model_profiles.py.
#include <cstdint>
#include <cstring>
namespace NrModel {
struct PackageIdentity {const char* sourceHash; const char* packageHash; const char* relative; const char* legacyRelative; unsigned graph; std::uint64_t bytes;};
inline constexpr PackageIdentity Packages[] = {
    {"E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E","A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB","1/E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E/nr-model-a-gfx1030.nrwgt","1/E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E/mixed-v5-gfx1030.nrwgt",1u,291595458ull},
    {"2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F","5C2B4272542C4CAF3B2E16A1242F21484B3BE2F8DC31337ECCCEA93143BFAE2C","1/2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F/nr-model-a-gfx1030.nrwgt","1/2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F/mixed-v5-gfx1030.nrwgt",1u,291595458ull},
    {"2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F","2CE6F8BB07F18399A58120F9FB2F1152DD39D6B872370DA7EE61B104E4165AE7","2/2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F/nr-model-b-gfx1030.nrwgt","2/2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F/balanced-b-gfx1030.nrwgt",2u,226720962ull},
};
// Header identity only; WeightPackage verifies the complete package SHA-256.
inline const PackageIdentity* Identify(const void* header, std::uint64_t bytes) {
    if(!header||bytes<128)return nullptr; const auto* p=static_cast<const unsigned char*>(header);
    if(std::memcmp(p,"NRWGT001",8))return nullptr; std::uint32_t graph=0;std::memcpy(&graph,p+8,4);
    static constexpr char hex[]="0123456789ABCDEF";char source[65]{};
    for(unsigned i=0;i<32;++i){source[i*2]=hex[p[32+i]>>4];source[i*2+1]=hex[p[32+i]&15];}
    for(const auto& item:Packages)if(item.graph==graph&&item.bytes==bytes&&!std::strcmp(item.sourceHash,source))return &item;
    return nullptr;
}
} // namespace NrModel
