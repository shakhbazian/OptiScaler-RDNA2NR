#pragma once
#include "nr_runtime_v3_contract.h"
#include "nr_execution_profile.h"

namespace NrPinned {
// The existing v3 table/lifetime semantics are reused, but a distinct export
// requires explicit numerical-profile consent. No silent FP16 -> mixed switch.
using GetApi = NrV3::Status (__cdecl*)(std::uint32_t profileId, std::uint32_t abiVersion,
                                      NrV3::Api* table, std::uint32_t tableBytes);
using GetName = const char* (__cdecl*)(); // static UTF-8, valid while module loaded
}
