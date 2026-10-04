#pragma once
#include <cstdint>
#include <mutex>

// Cyberpunk sends disable/enable pairs while preparing a single frame. Keep the
// backend alive through those pairs, but honor a remaining disable at Present.
// No timeout or frame-rate assumption: the app's Present is the commit boundary.
class Fsr3FrameGenerationRequests
{
  public:
    struct PresentResult
    {
        bool disabled = false;
        uint64_t cancelledDisables = 0;
    };

    template <class Apply> void Request(const void* owner, bool enabled, Apply&& apply)
    {
        std::lock_guard lock(_mutex);
        SetOwner(owner);
        if (!enabled)
        {
            _disablePending = true;
            return;
        }

        if (_disablePending)
            ++_cancelledDisables;
        _disablePending = false;
        apply(true);
    }

    template <class Apply> PresentResult Present(const void* owner, Apply&& apply)
    {
        std::lock_guard lock(_mutex);
        SetOwner(owner);
        PresentResult result { _disablePending, _cancelledDisables };
        if (_disablePending)
            apply(false);
        _disablePending = false;
        _cancelledDisables = 0;
        return result;
    }

    void Reset()
    {
        std::lock_guard lock(_mutex);
        _owner = nullptr;
        _disablePending = false;
        _cancelledDisables = 0;
    }

  private:
    void SetOwner(const void* owner)
    {
        if (_owner == owner)
            return;
        _owner = owner;
        _disablePending = false;
        _cancelledDisables = 0;
    }

    std::mutex _mutex;
    const void* _owner = nullptr;
    bool _disablePending = false;
    uint64_t _cancelledDisables = 0;
};
