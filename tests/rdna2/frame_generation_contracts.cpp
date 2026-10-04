#include "../../OptiScaler/inputs/FG/Fsr3FrameGenerationRequests.h"
#include "../../OptiScaler/upscalers/ffx/FfxProviderSelection.h"
#include <cassert>
#include <future>
#include <iostream>
#include <thread>
#ifdef _WIN32
#include "../../OptiScaler/fsr4/WatermarkEnvironment.h"
#endif

int main()
{
#ifdef _WIN32
    // Exercise the real setter with both readers used by Windows providers.
    for (const auto name : { L"MLSR-WATERMARK", L"MLFI-WATERMARK" })
    {
        const char* narrow = name[2] == L'S' ? "MLSR-WATERMARK" : "MLFI-WATERMARK";
        size_t size = 0;
        assert(getenv_s(&size, nullptr, 0, narrow) == 0);
        SetFidelityFxWatermark(name, true);
        assert(getenv_s(&size, nullptr, 0, narrow) == 0 && size == 2);
        assert(_wgetenv_s(&size, nullptr, 0, name) == 0 && size == 2);
        assert(GetEnvironmentVariableW(name, nullptr, 0) == 2);
        SetFidelityFxWatermark(name, false);
        assert(getenv_s(&size, nullptr, 0, narrow) == 0 && size == 0);
        assert(_wgetenv_s(&size, nullptr, 0, name) == 0 && size == 0);
        assert(GetEnvironmentVariableW(name, nullptr, 0) == 0);
        assert(GetLastError() == ERROR_ENVVAR_NOT_FOUND);
    }
    std::cout << "PASS: SR/FG watermark absent from CRT and Win32 environments after disable\n";
#endif
    const char* mixed[] = { "3.1.5", nullptr, "4.0.2", "2.3.4" };
    const char* fallback[] = { "2.3.4", "3.1.5" };
    const char* legacy[] = { nullptr, "2.3.4" };
    assert(PreferredFsrProvider(mixed) == 2);
    assert(PreferredFsrProvider(fallback) == 1);
    assert(PreferredFsrProvider(legacy) == 0);
    assert(PreferredFsrProvider({}) == 0);
    Fsr3FrameGenerationRequests requests;
    int contextA = 0, contextB = 0;
    bool active = false, allowed = true, paused = false;
    unsigned starts = 0, stops = 0;
    const auto apply = [&](bool value) {
        if (value && allowed && !paused && !active) { active = true; ++starts; }
        if (!value && active) { active = false; ++stops; }
    };
    requests.Request(&contextA, true, apply);
    assert(active && starts == 1);
    // Repeated per-frame pairs must preserve backend state and its history.
    for (unsigned i = 0; i < 10000; ++i)
    {
        requests.Request(&contextA, false, apply);
        assert(active);
        requests.Request(&contextA, true, apply);
        const auto result = requests.Present(&contextA, apply);
        assert(!result.disabled && result.cancelledDisables == 1 && active);
    }
    assert(starts == 1 && stops == 0);
    // A final disable must be applied before the presentation can generate.
    requests.Request(&contextA, false, apply);
    requests.Request(&contextA, false, apply);
    auto result = requests.Present(&contextA, apply);
    assert(result.disabled && !active && stops == 1);
    result = requests.Present(&contextA, apply);
    assert(!result.disabled && stops == 1);
    // A pause may defer activation. The next enable must still retry it.
    paused = true;
    requests.Request(&contextA, true, apply);
    assert(!active);
    paused = false;
    requests.Request(&contextA, true, apply);
    assert(active && starts == 2);
    // The user's Active switch remains authoritative even with game requests.
    allowed = false; apply(false);
    requests.Request(&contextA, false, apply);
    requests.Request(&contextA, true, apply);
    requests.Present(&contextA, apply);
    assert(!active);
    allowed = true;
    // Replacing a backend or resetting its context must drop stale disables.
    requests.Request(&contextA, false, apply);
    requests.Request(&contextB, true, apply);
    result = requests.Present(&contextB, apply);
    assert(!result.disabled && result.cancelledDisables == 0 && active);
    requests.Request(&contextB, false, apply);
    requests.Reset();
    result = requests.Present(&contextB, apply);
    assert(!result.disabled && active);
    // Exercise actual cross-thread ordering, not a guessed debounce duration.
    std::promise<void> disabled;
    auto ready = disabled.get_future();
    std::thread worker([&] { requests.Request(&contextB, false, apply); disabled.set_value(); });
    ready.wait();
    requests.Request(&contextB, true, apply);
    worker.join();
    result = requests.Present(&contextB, apply);
    assert(!result.disabled && result.cancelledDisables == 1 && active);
    std::promise<void> entered, resume;
    auto gate = resume.get_future();
    requests.Request(&contextB, false, apply);
    std::thread presenter([&] {
        requests.Present(&contextB, [&](bool value) { entered.set_value(); gate.wait(); apply(value); });
    });
    entered.get_future().wait();
    std::thread enabler([&] { requests.Request(&contextB, true, apply); });
    resume.set_value();
    presenter.join(); enabler.join();
    assert(active);
    std::cout << "PASS: 10000 frame pairs, final disable, pause, Active, context lifetime, worker/present ordering\n";
}
