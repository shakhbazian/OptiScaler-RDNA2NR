#pragma once
#include "nr_scheduled_runtime_contract.h"

// V2 preserves V1 fields and v3 payloads. The host records a continuation
// before submitting its prefix; armOutput installs the wait at real submission.
namespace NrSubmissionApi {
constexpr std::uint32_t Version=2;
using NrV3::Status;
using NrV3::Output;
using NrV3::TokenRequest;
using NrV3::RecordResult;
#pragma pack(push,8)
struct Api : NrScheduledApi::Api {
    Status (__cdecl* recordOutputDeferred)(void*,const Output*,std::uint32_t,RecordResult*,std::uint32_t);
    Status (__cdecl* armOutput)(void*,const TokenRequest*,std::uint32_t);
};
using GetApi=Status (__cdecl*)(std::uint32_t,std::uint32_t,Api*,std::uint32_t);
#pragma pack(pop)
static_assert(sizeof(Api)==sizeof(NrScheduledApi::Api)+2*sizeof(void*));
} // namespace NrSubmissionApi
