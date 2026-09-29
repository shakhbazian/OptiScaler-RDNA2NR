#pragma once

// Frozen host contract only. The v3 DLL implementation is a separate work item.
// Windows x64, pack 8; v2 and its synchronous semantics remain unchanged.
#include "nr_runtime_contract.h"

namespace NrV3 {
constexpr std::uint32_t Version = 3;
constexpr std::uint32_t Capacity = 3;
// Scheduled ABI only: Input.reserved is a checked per-frame policy field.
// Legacy v3 rejects nonzero reserved values as before.
constexpr std::uint32_t DisableTemporalAccumulation = 1u;
using NrV2::Prefix;
using NrV2::FrameKey;
using NrV2::Hash256;
using NrV2::Texture;
using NrV2::Controls;
using NrV2::Motion;
enum class Status : std::uint32_t {
    Ok, InvalidSize, Unsupported, InvalidArgument, InvalidResource,
    NotConfigured, Busy, NotSubmitted, StaleFrame, Failed, Quarantined, Closed
};
enum class Lifecycle : std::uint32_t { Created, Live, Draining, Faulted, Closed, Quarantined };
enum class Phase : std::uint32_t {
    Free, Recorded, Queued, Running, Ready, CopyRecorded, Retiring, Cancelling,
    Terminal, Quarantined
};
enum class Outcome : std::uint32_t { None, OutputRetired, Dropped, Cancelled, Failed, Quarantined };

#pragma pack(push, 8)
struct Config {
    Prefix prefix;
    Hash256 sourceHash, packageHash;
    std::uint32_t graphVersion, mathContract;
    std::uint32_t logicalWidth, logicalHeight, networkWidth, networkHeight;
    std::uint32_t reserved[2];
};
struct Token { FrameKey key; std::uint64_t generation; };
struct TokenRequest { Prefix prefix; Token token; };
struct Input {
    Prefix prefix;
    FrameKey key;
    ID3D12GraphicsCommandList* commands;
    ID3D12CommandQueue* queue;
    Texture color, motion, controlMask;
    Controls controls;
    Motion motionParameters;
    std::uint32_t resetReasons, reserved;
    std::uint64_t generation;
};
struct Output {
    Prefix prefix;
    Token token;
    ID3D12GraphicsCommandList* commands;
    ID3D12CommandQueue* queue;
    Texture target;
    std::uint32_t reserved[2];
};
// Check recorded even if the return status is a failure: commands cannot be
// rolled back. recorded=0 guarantees no list mutation or retained GPU reference.
struct RecordResult {
    Prefix prefix;
    Status status;
    std::uint32_t recorded;
    Token token;
};
struct Slot {
    Token token;
    Phase phase;
    Outcome outcome;
    std::uint32_t noiseIndex, usesHistory;
    std::uint64_t inputSerial, outputSerial;
};
struct Snapshot {
    Prefix prefix;
    Status status;
    Lifecycle lifecycle;
    std::uint64_t generation, commits;
    std::uint32_t occupied, capacity;
    std::uint64_t lastAcceptedEpoch;
    Slot slots[Capacity];
};
struct Api {
    Prefix prefix;
    Status (__cdecl* create)(ID3D12Device*, ID3D12CommandQueue*, void**);
    Status (__cdecl* configure)(void*, const Config*, std::uint32_t, const void*, std::uint64_t);
    Status (__cdecl* recordInput)(void*, const Input*, std::uint32_t, RecordResult*, std::uint32_t);
    Status (__cdecl* notifyInputSubmitted)(void*, const TokenRequest*, std::uint32_t);
    // Observe at most three slots and one event. Never dispatch or wait for work.
    Status (__cdecl* poll)(void*, Snapshot*, std::uint32_t);
    // May enqueue ONE complete graph. CPU submission cost is not a poll budget.
    Status (__cdecl* dispatchNext)(void*);
    Status (__cdecl* recordOutput)(void*, const Output*, std::uint32_t, RecordResult*, std::uint32_t);
    Status (__cdecl* notifyOutputSubmitted)(void*, const TokenRequest*, std::uint32_t);
    Status (__cdecl* dropOutput)(void*, const TokenRequest*, std::uint32_t);
    Status (__cdecl* acknowledgeTerminal)(void*, const TokenRequest*, std::uint32_t);
    Status (__cdecl* beginDrain)(void*);
    Status (__cdecl* shutdown)(void*); // Poll retirement; never dispatch or loop-wait.
    Status (__cdecl* quarantine)(void*); // Irreversible; keep owner/module until process exit.
    Status (__cdecl* destroy)(void**); // Only Closed; checked cleanup, null only on success.
};
// Planned export: DlssNrHipBackendGetApiV3(requestedVersion, table, tableBytes).
using GetApi = Status (__cdecl*)(std::uint32_t, Api*, std::uint32_t);
#pragma pack(pop)

static_assert(sizeof(Config) == 104 && sizeof(Token) == 32 && sizeof(TokenRequest) == 40);
static_assert(sizeof(Input) == 248 && sizeof(Output) == 104 && sizeof(RecordResult) == 48);
static_assert(sizeof(Slot) == 64 && sizeof(Snapshot) == 240 && sizeof(Api) == 120);
static_assert(offsetof(Input, color) == 48 && offsetof(Input, controls) == 168);
static_assert(offsetof(Input, generation) == 240 && offsetof(Output, target) == 56);
static_assert(offsetof(Snapshot, slots) == 48 && offsetof(Api, destroy) == 112);
static_assert(std::is_standard_layout<Input>::value && std::is_trivially_copyable<Input>::value);
static_assert(std::is_standard_layout<Snapshot>::value && std::is_trivially_copyable<Snapshot>::value);
} // namespace NrV3
