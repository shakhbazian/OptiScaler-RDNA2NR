#pragma once

#include <initializer_list>
#include <span>
#include <string_view>

inline int PreferredFsrProvider(std::span<const char* const> names)
{
    // The runtime supplies a device-specific list; indices are not fixed versions.
    for (const auto prefix : { std::string_view("4."), std::string_view("3.") })
    {
        for (size_t i = 0; i < names.size(); ++i)
        {
            if (names[i] != nullptr && std::string_view(names[i]).starts_with(prefix))
                return static_cast<int>(i);
        }
    }
    return 0;
}
