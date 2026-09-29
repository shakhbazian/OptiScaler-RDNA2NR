#pragma once
#include "nr_runtime_v3_contract.h"
namespace NrV3Test {
constexpr std::uint32_t Version=1;
#pragma pack(push,8)
struct State {
    std::uint32_t structSize,version;
    NrV3::Snapshot snapshot;
    std::uint64_t modelUploads,graphAllocations,graphPreparations,graphDispatches;
    std::uint64_t historySwaps,lastRecordMicroseconds,lastDispatchMicroseconds,lastPollMicroseconds;
};
#pragma pack(pop)
static_assert(sizeof(State)==312);
using Arm=NrV3::Status(__cdecl*)(void*);
using GetState=NrV3::Status(__cdecl*)(void*,State*,std::uint32_t);
}
