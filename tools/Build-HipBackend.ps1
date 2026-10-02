param([string]$HipRoot='C:\Program Files\AMD\ROCm\6.4',
      [string]$VsRoot='C:\Program Files\Microsoft Visual Studio\18\Community',
      [string]$OutputRoot='build/hip-gfx1030')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/Enter-Toolchain.ps1" -HipRoot $HipRoot -VsRoot $VsRoot `
    -PlatformToolset v145 -MsvcToolsVersion 14.44
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $out=[IO.Path]::GetFullPath($OutputRoot)
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $flags=@('--offload-arch=gfx1030','-D__HIP_PLATFORM_AMD__',
        '-DNR_DIRECT_PUBLICATION=1','-DNR_ATTENTION_DOT2=1',
        '-DNR_FIXED_ROW_THREADS=8','-DNR_OPTIMIZED_CHECKPOINT=1',
        '-DNR_SPATIAL_GRAPH_LIBRARY','-DNR_PINNED_MIXED=1','-DNR_SCHEDULED_MIXED=1',
        '-O2','-ffp-contract=off','-std=c++17','-Wall','-Wextra','-Werror',
        '-I',"$HipRoot/include")
    $hostSources=@('rdna2_nr/src/scheduled_runtime_plugin.cpp','rdna2_nr/src/nr_runtime_v3_dx12.cpp',
        'rdna2_nr/src/nr_gpu_model.cpp','rdna2_nr/src/nr_weight_package.cpp')
    $hipSources=@('rdna2_nr/src/spatial_graph.cpp','rdna2_nr/src/runtime_v2_neural_kernel.hip',
        'rdna2_nr/src/scheduled_state_kernel.hip')
    & "$HipRoot/bin/clang++.exe" @flags -shared @hostSources -x hip @hipSources `
        -ld3d12 -ldxgi -lbcrypt -o "$out/dlssnr_hip_scheduled_bridge.dll" 2>&1 |
        Tee-Object "$out/dll-build.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Scheduled companion build failed' }
    & dumpbin.exe /nologo /exports "$out/dlssnr_hip_scheduled_bridge.dll" |
        Set-Content "$out/exports.txt"
    $exports=Get-Content "$out/exports.txt" -Raw
    if ($exports -notmatch 'DlssNrHipBackendGetScheduledApiV1' -or
        $exports -notmatch 'DlssNrHipBackendGetScheduledApiV2' -or
        $exports -match 'DlssNrHipBackendGetPinnedApiV1|DlssNrHipBackendGetApiV3|TestArm') {
        throw 'Scheduled companion export isolation failed'
    }
    & "$HipRoot/bin/llvm-objcopy.exe" --dump-section `
        ".hip_fat=$out/scheduled.bundle" "$out/dlssnr_hip_scheduled_bridge.dll"
    if ($LASTEXITCODE -ne 0) { throw 'Scheduled HIP bundle extraction failed' }
    & "$HipRoot/bin/clang-offload-bundler.exe" --type=o `
        "--input=$out/scheduled.bundle" "--output=$out/scheduled.gfx1030.o" `
        --targets=hipv4-amdgcn-amd-amdhsa--gfx1030 --unbundle
    if ($LASTEXITCODE -ne 0) { throw 'Scheduled gfx1030 extraction failed' }
    & "$HipRoot/bin/llvm-readobj.exe" --file-headers "$out/scheduled.gfx1030.o" |
        Set-Content "$out/code-object.txt"
    if ($LASTEXITCODE -ne 0 -or (Get-Content "$out/code-object.txt" -Raw) -notmatch
        'EF_AMDGPU_MACH_AMDGCN_GFX1030') { throw 'Wrong scheduled code object' }
    $sourceHashes=[ordered]@{}
    foreach($path in @($hostSources)+@($hipSources)+@(Get-ChildItem rdna2_nr/include -Filter '*.h' | ForEach-Object {$_.FullName})) {
        $sourceHashes[$path]=(Get-FileHash -LiteralPath $path).Hash
    }
    [ordered]@{date=(Get-Date).ToString('o');target='gfx1030';toolset=$env:RDNA2_PLATFORM_TOOLSET
        msvc=$env:VCTOOLSVERSION;dllSha256=(Get-FileHash "$out/dlssnr_hip_scheduled_bridge.dll").Hash
        executionProfile='static-c32-long-int32-wave-qkv64-20261002';profileId='0x20261002'
        acceptedProbeSha256='489EAD0377304D4B3630B100EB1A3CEBBB921423AB7E7A5B2C217FD05DAF7469'
        flags=$flags;sourceHashes=$sourceHashes
        exports=@('DlssNrHipBackendGetScheduledApiV1','DlssNrHipBackendGetScheduledApiV2')} | ConvertTo-Json |
        Set-Content "$out/build-receipt.json"
    Write-Output 'PASS gfx1030 HIP backend build and export isolation'
} finally {Pop-Location}
