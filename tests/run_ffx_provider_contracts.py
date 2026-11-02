"""Compile the production provider-selection function against controlled SDK lists.

Run in the MSVC developer shell. No vendor DLL or GPU is loaded.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def body(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


config = (ROOT / 'OptiScaler/Config.h').read_text(encoding='utf-8-sig')
optional = config[config.index('enum HasDefaultValue'):config.index('constexpr inline int UnboundKey')]
state = (ROOT / 'OptiScaler/State.h').read_text(encoding='utf-8-sig')
versions = body(state, 'struct FfxVersionList') + ';'
source = (ROOT / 'OptiScaler/upscalers/ffx/FFXFeature.cpp').read_text(encoding='utf-8-sig')
query = body(source, 'bool FFXFeature::QueryVersionsDx12')
code = r'''
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <upscalers/ffx/FfxProviderSelection.h>
@OPTIONAL@
struct State {
@VERSIONS@
 FfxVersionList ffxUpscalerVersionsDx12;
 static State& Instance(){static State state;return state;}
};
struct Config {
 CustomOptional<int,SoftDefault> FfxUpscalerIndex{0};
 CustomOptional<std::string,NoDefault> FfxProviderId;
 std::map<std::string,CustomOptional<int,SoftDefault>> indices;
 auto& FfxUpscalerIndexFor(const std::string& name){return name.empty()?FfxUpscalerIndex:indices.try_emplace(name,0).first->second;}
 static Config* Instance(){static Config config;return &config;}
};
using ID3D12Device = int;
constexpr int FFX_API_QUERY_DESC_TYPE_GET_VERSIONS=1,FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE=2,FFX_API_RETURN_OK=0;
struct Header{int type;};
struct ffxQueryDescGetVersions{Header header;int createDescType;void* device;uint64_t* outputCount;uint64_t* versionIds=nullptr;const char** versionNames=nullptr;};
struct Item{uint64_t id;std::string name;int module;};
std::vector<Item> items;
int Query(void*,Header* raw){
 auto q=reinterpret_cast<ffxQueryDescGetVersions*>(raw);
 auto n=q->versionIds||q->versionNames?std::min<uint64_t>(*q->outputCount,items.size()):items.size();
 for(size_t i=0;i<n;++i){if(q->versionIds)q->versionIds[i]=items[i].id;if(q->versionNames)q->versionNames[i]=items[i].name.c_str();}
 *q->outputCount=n;return 0;
}
struct CustomFfxModule {std::string name;struct{int(*Query)(void*,Header*)=::Query;}module;State::FfxVersionList versions;};
struct FSR4DriverProvider{enum class Listing{Always};static int LoadedModule(){return 9;}};
struct FfxApiProxy {
 static int D3D12_QueryUpscalerVersions(ffxQueryDescGetVersions* q,FSR4DriverProvider::Listing){return Query(nullptr,&q->header);}
 static int UpscalerModule(uint64_t id){for(auto& i:items)if(i.id==id)return i.module;return 0;}
 static int Dx12Module_SR(){return 1;}
};
struct FFXFeature{static bool QueryVersionsDx12(ID3D12Device*,CustomFfxModule* = nullptr);};
@QUERY@
int main(){
 auto cfg=Config::Instance();auto& list=State::Instance().ffxUpscalerVersionsDx12;
 items={{900,"4.1.0",9},{100,"4.0.2",1},{200,"3.1.5",1}};
 for(int i=0;i<3;++i){assert(FFXFeature::QueryVersionsDx12(nullptr));assert(cfg->FfxUpscalerIndex==1);assert(!cfg->FfxProviderId.has_value());}
 cfg->FfxUpscalerIndex=0; // saved index in the pre-driver SDK list
 assert(FFXFeature::QueryVersionsDx12(nullptr));assert(cfg->FfxUpscalerIndex==1);assert(cfg->FfxProviderId=="100");
 assert(FFXFeature::QueryVersionsDx12(nullptr));assert(cfg->FfxUpscalerIndex==1);
 std::rotate(items.begin(),items.begin()+1,items.end());
 assert(FFXFeature::QueryVersionsDx12(nullptr));assert(cfg->FfxUpscalerIndex==0);
 cfg->FfxProviderId="900";assert(FFXFeature::QueryVersionsDx12(nullptr));assert(cfg->FfxUpscalerIndex==2);
 items[0].name="mutated";assert(std::string(list.names[0])=="4.0.2");
 CustomFfxModule a{"A"},b{"B"};cfg->FfxUpscalerIndexFor("A")=1;cfg->FfxUpscalerIndexFor("B")=0;
 assert(FFXFeature::QueryVersionsDx12(nullptr,&a));assert(FFXFeature::QueryVersionsDx12(nullptr,&b));
 assert(cfg->FfxUpscalerIndexFor("A")==1&&cfg->FfxUpscalerIndexFor("B")==0&&cfg->FfxProviderId=="900");
 items.clear();assert(FFXFeature::QueryVersionsDx12(nullptr));assert(list.ids.empty()&&list.names.empty());
 std::puts("PASS FFX provider selection: repeat auto query, legacy index migration, stable id reorder, owned names, independent custom choices, empty list");
}
'''.replace('@OPTIONAL@', optional).replace('@VERSIONS@', versions).replace('@QUERY@', query)
out = Path(tempfile.mkdtemp(prefix='nr-ffx-contracts-'))
(out / 'test.cpp').write_text(code, encoding='utf-8')
subprocess.run(['cl', '/nologo', '/std:c++20', '/EHsc', '/O2', '/MD', '/I'+str(ROOT/'OptiScaler'),
                str(out/'test.cpp'), '/Fo:'+str(out/'test.obj'), '/Fe:'+str(out/'test.exe')], check=True)
subprocess.run([str(out/'test.exe')], check=True, timeout=20)
