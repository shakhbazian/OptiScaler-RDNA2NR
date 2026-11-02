param([string]$OutputRoot='build/tests/rdna2', [string]$AgilitySdkDirectory='', [uint32]$AgilitySdkVersion=0)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. (Join-Path $root 'tools/Enter-Toolchain.ps1') -PlatformToolset v145 -MsvcToolsVersion 14.44
Push-Location $root
try {
    $output=$OutputRoot
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $common=@('/nologo','/std:c++17','/EHsc','/O2','/MD','/W4','/WX',
        '/Iexternal/nvngx_dlss_sdk',"/Fo:$output/")
    if($AgilitySdkDirectory){
        if(-not $AgilitySdkVersion){throw 'An explicit Agility version is required.'}
        $sdkStage=Join-Path $output 'AgilitySDK'
        New-Item -ItemType Directory -Force -Path $sdkStage|Out-Null
        foreach($file in @('D3D12Core.dll','d3d12SDKLayers.dll')){
            Copy-Item -LiteralPath (Join-Path $AgilitySdkDirectory $file) -Destination $sdkStage -Force
        }
        $common+="/DNR_HOST_AGILITY_SDK_VERSION=$AgilitySdkVersion"
    }
    & cl.exe @common 'rdna2_nr/src/optiscaler_dx12_evaluate_host.cpp' /link `
        d3d12.lib dxgi.lib bcrypt.lib dbghelp.lib d3dcompiler.lib "/OUT:$output/optiscaler_dx12_evaluate_host.exe"
    if($LASTEXITCODE -ne 0){throw 'D3D12 host build failed'}
    & cl.exe @common 'rdna2_nr/src/optiscaler_dx11_evaluate_host.cpp' /link `
        d3d11.lib d3d12.lib dxgi.lib "/OUT:$output/optiscaler_dx11_evaluate_host.exe"
    if($LASTEXITCODE -ne 0){throw 'D3D11 host build failed'}
    & cl.exe @common /IOptiScaler /IOptiScaler/include 'tests/rdna2/dynamic_raster.cpp' /link `
        d3d12.lib dxgi.lib bcrypt.lib d3dcompiler.lib /LIBPATH:OptiScaler/library/detours detours.lib "/OUT:$output/dynamic_raster.exe"
    if($LASTEXITCODE -ne 0){throw 'Dynamic raster test build failed'}
    & cl.exe @common 'tests/rdna2/codec_identity.cpp' /link `
        d3d12.lib dxgi.lib bcrypt.lib "/OUT:$output/codec_identity.exe"
    if($LASTEXITCODE -ne 0){throw 'Codec identity test build failed'}
    Write-Output 'PASS source-built D3D12 and D3D11-on-12 hosts'
} finally {Pop-Location}
