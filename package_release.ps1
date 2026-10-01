# Assemble only allowlisted files. User models and NVIDIA runtimes never enter the build.
param([ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$Version=(Get-Content -LiteralPath (Join-Path $PSScriptRoot 'VERSION.txt') -Raw).Trim(),
      [switch]$SkipBuild,
      [switch]$NoZip,
      [string]$PortablePythonHome)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSCommandPath
if(-not $PortablePythonHome){throw 'PortablePythonHome is required for an installable release.'}
. (Join-Path $root 'tools/Enter-Toolchain.ps1') -PlatformToolset v145 -MsvcToolsVersion 14.44
$stage=Join-Path $root "release/OptiScaler-RDNA2NR-$Version"
$zip=Join-Path $root "release/OptiScaler-RDNA2NR-$Version.zip"
if((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)){
    throw 'Choose a new version: an existing release is never overwritten.'
}
if(-not $SkipBuild){
    & (Join-Path $root 'tools/Build-HipBackend.ps1')
    & (Join-Path $root 'tools/Build-OptiScaler.ps1')
}
$files=[ordered]@{
    'OptiScaler.dll'='x64/Release/OptiScaler.dll'
    'dlssnr_hip_scheduled_bridge.dll'='build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll'
    'OptiScaler.ini'='OptiScaler.ini'
    'README.md'='README.md'
    'Changelog.md'='Changelog.md'
    'VERSION.txt'='VERSION.txt'
    'LICENSE'='LICENSE'
    'INSTALL-DLSSNR.md'='INSTALL-DLSSNR.md'
    'docs/README.md'='docs/README.md'
    'docs/INSTALLATION.md'='docs/INSTALLATION.md'
    'docs/SETTINGS.md'='docs/SETTINGS.md'
    'docs/MODEL.md'='docs/MODEL.md'
    'docs/COMPATIBILITY.md'='docs/COMPATIBILITY.md'
    'docs/NVIDIA.md'='docs/NVIDIA.md'
    'docs/TROUBLESHOOTING.md'='docs/TROUBLESHOOTING.md'
    'docs/ARCHITECTURE.md'='docs/ARCHITECTURE.md'
    'docs/BUILDING.md'='docs/BUILDING.md'
    'docs/CREDITS.md'='docs/CREDITS.md'
    'docs/upstream/README.md'='docs/upstream/README.md'
    'docs/release-notes/r1-0.8.91.md'='docs/release-notes/r1-0.8.91.md'
    'setup_windows.bat'='setup_windows.bat'
    'Install-RDNA2NR.ps1'='Install-RDNA2NR.ps1'
    'Install-RDNA2NR.cmd'='Install-RDNA2NR.cmd'
    'OptiScaler/libxess.dll'='external/xess/bin/libxess.dll'
    'OptiScaler/libxess_dx11.dll'='external/xess/bin/libxess_dx11.dll'
    'OptiScaler/libxell.dll'='external/xess/bin/libxell.dll'
    'OptiScaler/libxess_fg.dll'='external/xess/bin/libxess_fg.dll'
    'OptiScaler/amd_fidelityfx_vk.dll'='external/FidelityFX-SDK/PrebuiltSignedDLL/amd_fidelityfx_vk.dll'
    'OptiScaler/amd_fidelityfx_loader_dx12.dll'='external/FidelityFX-SDK-v2/Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll'
    'OptiScaler/amd_fidelityfx_upscaler_dx12.dll'='external/FidelityFX-SDK-v2/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll'
    'OptiScaler/amd_fidelityfx_framegeneration_dx12.dll'='external/FidelityFX-SDK-v2/Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_dx12.dll'
    'OptiScaler/D3D12_OptiScaler/D3D12Core.dll'='external/directx_agility_sdk/lib/D3D12Core.dll'
    'Licenses/XeSS_LICENSE.txt'='external/xess/LICENSE.txt'
    'Licenses/FidelityFX_v1_LICENSE.md'='external/FidelityFX-SDK/docs/license.md'
    'Licenses/FidelityFX_v2_LICENSE.md'='external/FidelityFX-SDK-v2/docs/license.md'
    'Licenses/DirectX_LICENSE.txt'='external/directx_agility_sdk/LICENSE.txt'
    'Licenses/FreeType_FTL.txt'='Licenses/FreeType_FTL.txt'
    'Licenses/RenoDX_ATTRIBUTION.txt'='Licenses/RenoDX_ATTRIBUTION.txt'
    'Licenses/MFGUnlock_LICENSE.txt'='Licenses/MFGUnlock_LICENSE.txt'
    'Licenses/PeripheralWarp_LICENSE.txt'='external/peripheral_warp/LICENSE'
}
foreach($entry in $files.GetEnumerator()){
    $source=Join-Path $root $entry.Value
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){throw "Missing release input: $source"}
}
$converterFiles=@('write_runtime_package.py','model_weight_provider.py','weights_ht.py',
    'model_schemas.py','qmma_layout.py','research_layouts.py')
foreach($name in $converterFiles){
    $source=Join-Path $root "tools/model_converter/$name"
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){throw "Missing converter source: $source"}
}
$frontendExports=& dumpbin.exe /nologo /exports (Join-Path $root $files['OptiScaler.dll'])
$hipExports=& dumpbin.exe /nologo /exports (Join-Path $root $files['dlssnr_hip_scheduled_bridge.dll'])
$frontendText=$frontendExports -join "`n"
$hipText=$hipExports -join "`n"
if($LASTEXITCODE -ne 0 -or $frontendText -match 'DlssNrNativeTest|TestArm' -or
   $hipText -notmatch 'DlssNrHipBackendGetScheduledApiV2' -or
   $hipText -match 'TestArm|DlssNrHipBackendGetPinnedApiV1'){
    throw 'Production export audit failed.'
}
$ini=Get-Content -LiteralPath (Join-Path $root $files['OptiScaler.ini']) -Raw
if($ini -notmatch '(?mi)^TargetProcessName=auto\s*$' -or
   $ini -match '(?mi)^Enabled=true\s*$' -or $ini -match '(?mi)^ModelPath=\S+' -or
   $ini -match '(?mi)^(FinishedPicture|DeferredDLSS|AdaMfgUnlock)=true\s*$'){
    throw 'Default INI enables an experimental or machine-specific setting.'
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null
foreach($entry in $files.GetEnumerator()){
    $destination=Join-Path $stage $entry.Key
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $root $entry.Value) -Destination $destination
}
[IO.File]::WriteAllText((Join-Path $stage 'VERSION.txt'),"$Version`n",[Text.UTF8Encoding]::new($false))
$converterStage=Join-Path $stage 'tools/model_converter'
New-Item -ItemType Directory -Path $converterStage -Force | Out-Null
foreach($name in $converterFiles){
    Copy-Item -LiteralPath (Join-Path $root "tools/model_converter/$name") -Destination (Join-Path $converterStage $name)
}
if($PortablePythonHome){
    & (Join-Path $root 'tools/Stage-PortablePython.ps1') -PythonHome $PortablePythonHome -Destination (Join-Path $stage 'tools/python')
    if($LASTEXITCODE -ne 0){throw 'Portable Python staging failed.'}
}
$ini=$ini -replace '(?m)^; Experimental built-in RTX 40 MFG unlock[^\r\n]*\r?\n',''
$ini=$ini -replace '(?m)^AdaMfgUnlock=[^\r\n]*\r?\n',''
$ini=$ini -replace '(?ms)^; Frame timing fix for the extra frames.*?^AdaFlipMeteringPatch=[^\r\n]*\r?\n',''
[IO.File]::WriteAllText((Join-Path $stage 'OptiScaler.ini'),$ini,[Text.UTF8Encoding]::new($false))
@('Run Install-RDNA2NR.cmd to select a game folder and your own original nvngx_dlssnr.dll.',
  'The installer parses the DLL as data and creates the private model package locally.',
  'No NVIDIA model or converted weights are included.',
  'The gfx1030 HIP backend requires a compatible AMD driver runtime.') |
    Set-Content -LiteralPath (Join-Path $stage 'MODEL-SETUP.txt') -Encoding utf8
$unexpected=@(Get-ChildItem -LiteralPath $stage -Recurse -File | Where-Object {
    $_.Name -match '(?i)\.nrwgt$|^nvngx_dlssnr\.dll$|^amdhip64_6\.dll$|^amd_comgr_2\.dll$'
})
if($unexpected.Count){throw 'A model or driver runtime entered the release stage.'}
$checksums=Get-ChildItem -LiteralPath $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative=$_.FullName.Substring($stage.TrimEnd('\','/').Length).TrimStart('\','/').Replace('\','/')
    '{0} *{1}' -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash,$relative
}
[IO.File]::WriteAllLines((Join-Path $stage 'SHA256SUMS.txt'),$checksums,[Text.UTF8Encoding]::new($false))
if(-not $NoZip){
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Output "Created $zip"
}else{Write-Output "Created stage $stage"}
