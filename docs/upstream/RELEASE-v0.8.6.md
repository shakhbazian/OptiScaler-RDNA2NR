> Inherited parent-fork reference. This document describes NVIDIA paths or historical work; see the [RDNA2NR documentation](../README.md) for current AMD support.

# v0.8.6 prerelease — shutdown and GPU lifetime fixes

This release addresses shutdown defects found while investigating an exit-time GPU hang followed by NVIDIA `nvlddmkm` Event 153 reports. The supplied log reached `Unloading OptiScaler`, which previously preceded logger shutdown and C++ static destruction. It did not prove that cleanup had finished.

The code defects below are corrected. Their role in the reported GPU reset remains unconfirmed; testing on the affected game and driver is still needed.

## Fixes

- **One NGX shutdown:** the DX12 shutdown variants share one implementation and select exactly one NVIDIA entry point. Backend initialization state is reset after success. Repeat calls are harmless, and NVIDIA failures are returned so shutdown can be retried.
- **NR before runtime teardown:** retire NR while its parent upscalers and local FG resources remain alive. Give tracked GPU completion up to one second of polling, releasing the registry lock between polls. Completed final submissions can retire even if the game does not reset its last command list.
- **Unresolved work stays protected:** unsubmitted recordings, failed fence signals, removed-device fences and unfinished DX11 presentation transfers do not count as completion. If NR owners remain, return failure and keep the NGX runtime and parent contexts alive. This can retain memory until process exit; it avoids freeing resources whose GPU use is unknown.
- **No NR destruction during DLL unloading:** early-return guards now explicitly relinquish smart-pointer ownership. DX11/DX12/Vulkan context registries and the shared shutdown flag survive CRT teardown. Late NR submission, reset, presentation and feature-release callbacks stop before accessing runtime state.
- **Device-removal handling:** finished-picture drains recheck completion after event wakeup and reject the all-ones removed-device sentinel. Private-upscaler cleanup checks that NGX release entry points remain available.

## Additional exit safeguards

These changes address plausible failure paths; they are not claimed as independently reproduced causes of the reported driver reset:

- On process termination, `DllMain` sets the shutdown flag and returns without unloading companion DLLs, logging, flushing or joining workers. Ordinary explicit shutdown remains outside this path.
- Keep the active logger and its worker pool alive through CRT teardown, so static destruction does not join threads already stopped by `ExitProcess`. Streamline and NGX logging callbacks stop during unloading.
- Stop NR's Streamline swapchain queries as soon as unloading begins, while still forwarding the original presentation callback.
- Apply equivalent unload guards to Vulkan NR and DX11 bridge owners. Normal Vulkan shutdown still follows its existing device-idle behavior.

No rendering shader, model, default setting, TDR registry setting or RTX40 binary patch was changed. The half-rate/cadence feature remains excluded. The separate reported RTX40 MFG black screen has not been established as fixed by these shutdown changes.

## Validation

- Production shutdown/destructor bodies compiled against counting dependencies: cleanup ordering; one-call shutdown; repeated calls; single-export runtimes; failed/deferred shutdown and retry; process-detach guards.
- WARP tests using the production GPU lifetime tracker: final completed submissions, unsent work, blocked queues, generation/replay ownership and device removal.
- Existing NR prerelease suite: proxy readiness, MFG patch recognition, Streamline routing, captures, composition, sharpness, exposure, active-region/multi-mip copies, finished-queue ordering and production Vulkan shaders.
- Standard and optional RTX40 MFG Release x64 builds; release archive integrity and SHA-256 checksums.

The D3D12 debug layer was unavailable in the test environment. No affected-game exit reproduction or RTX40 hardware validation is claimed. The optional MFG package retains the same MFG implementation as v0.8.5.

For follow-up logs, normal DX12 shutdown now reports retirement, runtime shutdown and completion separately. Process termination deliberately avoids the old final DLL log writes. If the freeze persists, preserve `OptiScaler.log` and the INI, plus the first Event 153's full message and timestamp and any Windows LiveKernelReports dump.
