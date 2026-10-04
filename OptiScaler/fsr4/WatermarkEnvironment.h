#pragma once
#include <cstdlib>
#include <Windows.h>

inline void SetFidelityFxWatermark(const wchar_t* name, bool enabled)
{
    // FidelityFX checks presence: a value of "0" still enables the watermark.
    // Clear both environment APIs because the provider may use either one.
    _wputenv_s(name, enabled ? L"1" : L"");
    SetEnvironmentVariableW(name, enabled ? L"1" : nullptr);
}
