#pragma once
#include "nr_runtime_v3_contract.h"

// Transport API v1 deliberately reuses the checked v3 frame/config payloads
// (their prefixes remain v3). Its table has independent versioning/semantics:
// enqueue and recordOutput never require CPU-observed input/output completion.
namespace NrScheduledApi {
constexpr std::uint32_t Version=1;
using NrV3::Status;
using NrV3::Config;
using NrV3::Input;
using NrV3::Output;
using NrV3::Token;
using NrV3::TokenRequest;
using NrV3::RecordResult;
using NrV3::Snapshot;
using NrV3::Prefix;

#pragma pack(push,8)
struct Api {
    Prefix prefix;
    Status (__cdecl* create)(ID3D12Device*,ID3D12CommandQueue*,void**);
    Status (__cdecl* configure)(void*,const Config*,std::uint32_t,const void*,std::uint64_t);
    Status (__cdecl* recordInput)(void*,const Input*,std::uint32_t,RecordResult*,std::uint32_t);
    Status (__cdecl* notifyInputSubmitted)(void*,const TokenRequest*,std::uint32_t);
    Status (__cdecl* poll)(void*,Snapshot*,std::uint32_t);
    Status (__cdecl* enqueue)(void*,const TokenRequest*,std::uint32_t);
    Status (__cdecl* recordOutput)(void*,const Output*,std::uint32_t,RecordResult*,std::uint32_t);
    Status (__cdecl* notifyOutputSubmitted)(void*,const TokenRequest*,std::uint32_t);
    // The host must prove a recorded list was discarded before calling drop.
    Status (__cdecl* drop)(void*,const TokenRequest*,std::uint32_t);
    Status (__cdecl* acknowledgeTerminal)(void*,const TokenRequest*,std::uint32_t);
    Status (__cdecl* beginDrain)(void*);
    Status (__cdecl* shutdown)(void*);
    Status (__cdecl* quarantine)(void*);
    Status (__cdecl* destroy)(void**);
};
using GetApi=Status (__cdecl*)(std::uint32_t profileId,std::uint32_t requested,Api*,std::uint32_t);
#pragma pack(pop)
static_assert(sizeof(Api)==sizeof(NrV3::Api));
static_assert(offsetof(Api,destroy)==offsetof(NrV3::Api,destroy));
} // namespace NrScheduledApi
