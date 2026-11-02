#pragma once
#include <atomic>
#include <cstdio>

// Optional host logging keeps HIP initialization errors visible in games,
// which normally have no stderr console. The host must outlive the companion.
namespace NrRuntimeLog {
using Sink = void (__cdecl*)(const char*);
inline std::atomic<Sink> sink{nullptr};
inline void Write(const char* text) {
    if (const auto callback = sink.load(std::memory_order_acquire)) callback(text);
    else std::fprintf(stderr,"NR HIP: %s\n",text);
}
}
