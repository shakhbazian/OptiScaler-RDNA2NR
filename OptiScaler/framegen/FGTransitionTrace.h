#pragma once

#ifdef NR_FG_TRANSITION_TRACE
#include <Config.h>
#include <State.h>
#include <Util.h>
#include <framegen/IFGFeature.h>
#include <atomic>

// Opt-in trace for distinguishing game requests from backend and swapchain resets.
// Keep it out of normal builds: transition bursts are useful evidence, not telemetry.
inline void TraceFGTransition(const char* source, IFGFeature* fg, bool requested,
                              void* caller, const void* context)
{
    const auto module = Util::GetCallerModule(caller);
    const auto offset = reinterpret_cast<uintptr_t>(caller) - reinterpret_cast<uintptr_t>(module);
    const auto& state = State::Instance();
    LOG_INFO("FGTRACE source={} requested={} active={} menu={} input={} inputEnabled={} "
             "frame={} dispatched={} paused={} fgChanged={} scChanged={} context={:X} "
             "thread={} caller={}+{:X}",
             source, requested, fg->IsActive(), Config::Instance()->FGEnabled.value_or_default(),
             static_cast<int>(state.activeFgInput), state.fsrfgInputActive,
             fg->FrameCount(), fg->LastDispatchedFrame(), fg->IsPaused(), state.fgChanged,
             state.scChanged, reinterpret_cast<uintptr_t>(context), GetCurrentThreadId(),
             Util::WhoIsTheCaller(caller), offset);
}

// Capture a few requests in each direction, including the caller above the FSR wrapper.
// This is deliberately bounded and does not unwind the stack on every frame.
inline void TraceFGRequestOrigin(IFGFeature* fg, bool requested, bool async, uint32_t flags,
                                 bool onlyInterpolated, const void* hudless,
                                 uintptr_t frameCallback, uintptr_t presentCallback)
{
    static std::atomic<unsigned> captured[2] {};
    const auto sample = captured[requested ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
    if (sample >= 8)
        return;

    const auto frame = fg->FrameCount();
    LOG_INFO("FGORIGIN requested={} sample={} frame={} async={} flags={:X} onlyInterpolated={} "
             "hudless={:X} frameCallback={:X} presentCallback={:X}",
             requested, sample, frame, async, flags, onlyInterpolated,
             reinterpret_cast<uintptr_t>(hudless), frameCallback, presentCallback);

    void* frames[12] {};
    const auto count = CaptureStackBackTrace(0, 12, frames, nullptr);
    for (USHORT i = 0; i < count; ++i)
    {
        const auto module = Util::GetCallerModule(frames[i]);
        const auto offset = reinterpret_cast<uintptr_t>(frames[i]) - reinterpret_cast<uintptr_t>(module);
        LOG_INFO("FGORIGIN stack requested={} sample={} frame={} depth={} caller={}+{:X}",
                 requested, sample, frame, i,
                 Util::WhoIsTheCaller(frames[i]), offset);
    }
}
#endif
