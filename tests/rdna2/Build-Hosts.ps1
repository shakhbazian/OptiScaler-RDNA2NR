$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. (Join-Path $root 'tools/Enter-Toolchain.ps1') -PlatformToolset v145 -MsvcToolsVersion 14.44
Push-Location $root
try {
    $output='build/tests/rdna2'
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $common=@('/nologo','/std:c++17','/EHsc','/O2','/MD','/W4','/WX',
        '/Iexternal/nvngx_dlss_sdk',"/Fo:$output/")
    & cl.exe @common 'rdna2_nr/src/optiscaler_dx12_evaluate_host.cpp' /link `
        d3d12.lib dxgi.lib bcrypt.lib dbghelp.lib "/OUT:$output/optiscaler_dx12_evaluate_host.exe"
    if($LASTEXITCODE -ne 0){throw 'D3D12 host build failed'}
    & cl.exe @common 'rdna2_nr/src/optiscaler_dx11_evaluate_host.cpp' /link `
        d3d11.lib d3d12.lib dxgi.lib "/OUT:$output/optiscaler_dx11_evaluate_host.exe"
    if($LASTEXITCODE -ne 0){throw 'D3D11 host build failed'}
    Write-Output 'PASS source-built D3D12 and D3D11-on-12 hosts'
} finally {Pop-Location}
