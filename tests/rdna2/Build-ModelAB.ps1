param([string]$OutputRoot='build/model-ab', [switch]$Mixed, [string]$BaselineSource='', [ValidateRange(-1,3)][int]$BMode=-1, [ValidateRange(0,63)][int]$BC32Mask=63)
$ErrorActionPreference='Stop'
if($BMode -ge 0 -and !$Mixed){throw 'BMode requires -Mixed; omit BMode for the FP16 reference'}
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. "$root/tools/Enter-Toolchain.ps1" -PlatformToolset v145 -MsvcToolsVersion 14.44
Push-Location $root
try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    $flags=@('--offload-arch=gfx1030','-D__HIP_PLATFORM_AMD__',
        '-DNR_DIRECT_PUBLICATION=1','-DNR_ATTENTION_DOT2=1',
        '-DNR_FIXED_ROW_THREADS=8','-DNR_OPTIMIZED_CHECKPOINT=1',
        '-DNR_SPATIAL_GRAPH_LIBRARY','-O2','-ffp-contract=off',
        '-std=c++17','-Wall','-Wextra','-Werror','-I',"$env:HIP_PATH/include")
    if($Mixed){$flags+='-DNR_PINNED_MIXED=1'}
    if($BMode -ge 0){$flags+=@("-DNR_MODEL_B_MIXED=$BMode","-DNR_MODEL_B_C32_MASK=$BC32Mask")}
    $hostSources=@('tests/rdna2/model_ab.cpp','rdna2_nr/src/nr_gpu_model.cpp','rdna2_nr/src/nr_weight_package.cpp')
    $hipSources=@('rdna2_nr/src/spatial_graph.cpp','rdna2_nr/src/runtime_v2_neural_kernel.hip')
    $name=if($Mixed){'model_ab_mixed.exe'}else{'model_ab_fp16.exe'}
    if($BaselineSource){
        $hipSources[0]=$BaselineSource
        $flags+=@('-I',"$root/rdna2_nr/src")
        $name=$name.Replace('.exe','_baseline.exe')
    }
    & "$env:HIP_PATH/bin/clang++.exe" @flags @hostSources -x hip @hipSources -lbcrypt -o "$OutputRoot/$name" 2>&1 |
        Tee-Object "$OutputRoot/$name.build.txt"
    if($LASTEXITCODE -ne 0){throw 'Model comparison host build failed'}
    Get-FileHash "$OutputRoot/$name" -Algorithm SHA256
} finally {Pop-Location}
