#pragma once
#include "../../../rdna2_nr/include/nr_model_catalog.h"
#include <Config.h>
#include <Util.h>
#include <filesystem>

namespace DlssNr::Native {
// Paths are resolved separately from backend selection: NVIDIA NGX keeps its
// original DLL, while HIP reads one of the locally prepared A/B packages.
inline std::filesystem::path ModelCacheRoot(){
    const DWORD length=GetEnvironmentVariableW(L"LOCALAPPDATA",nullptr,0);
    if(length<=1)return {};
    std::wstring local(length,L'\0');
    if(GetEnvironmentVariableW(L"LOCALAPPDATA",local.data(),length)!=length-1)return {};
    local.resize(length-1);return std::filesystem::path(local)/L"OptiScaler-RDNA2NR"/L"models";
}
inline bool PackagePresent(const std::filesystem::path& path){
    std::error_code error;return !path.empty()&&std::filesystem::is_regular_file(path,error);
}
inline std::filesystem::path InstalledPackage(const std::filesystem::path& cache,const NrModel::PackageIdentity& identity){
    auto path=cache/identity.relative;
    if(!PackagePresent(path)&&identity.legacyRelative[0]){
        auto legacy=cache/identity.legacyRelative;
        if(PackagePresent(legacy))return legacy;
    }
    return path;
}
inline std::filesystem::path ModelPath(unsigned graph){
    const auto& cfg=*Config::Instance();
    const auto overridePath=graph==2?cfg.DlssNrModelBPath.value_or_default():cfg.DlssNrModelPath.value_or_default();
    if(!overridePath.empty())return std::filesystem::path(string_to_wstring(overridePath));
    const auto cache=ModelCacheRoot();if(cache.empty())return {};
    // Prefer A from the current source, retaining the older installed cache.
    const auto& latest=NrModel::Packages[graph==2?2:1];
    auto path=InstalledPackage(cache,latest);
    if(graph==1&&!PackagePresent(path))path=InstalledPackage(cache,NrModel::Packages[0]);
    return path;
}
inline unsigned SelectedModelGraph(){
    const auto model=Config::Instance()->DlssNrHipModel.value_or_default();
    return model=="b"?2:model=="a"?1:PackagePresent(ModelPath(2))?2:1;
}
inline const char* SelectedModelName(){return SelectedModelGraph()==2?"Model B":"Model A";}
} // namespace DlssNr::Native
