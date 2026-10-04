# Assemble only allowlisted files. User models and NVIDIA runtimes never enter the build.
param([ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$Version=(Get-Content -LiteralPath (Join-Path $PSScriptRoot 'VERSION.txt') -Raw).Trim(),
      [switch]$SkipBuild,
      [switch]$NoZip,
      [string]$PortablePythonHome,
      [string]$OutputRoot='release',
      [string]$FrontendPath='x64/Release/OptiScaler.dll',
      [string]$HipBackendPath='build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSCommandPath
if(-not $PortablePythonHome){throw 'PortablePythonHome is required for an installable release.'}
. (Join-Path $root 'tools/Enter-Toolchain.ps1') -PlatformToolset v145 -MsvcToolsVersion 14.44
$destinationRoot=if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path $root $OutputRoot}
$stage=Join-Path $destinationRoot "OptiScaler-RDNA2NR-$Version"
$zip=Join-Path $destinationRoot "OptiScaler-RDNA2NR-$Version.zip"
if((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)){
    throw 'Choose a new version: an existing release is never overwritten.'
}
if(-not $SkipBuild){
    if($FrontendPath -ne 'x64/Release/OptiScaler.dll' -or
       $HipBackendPath -ne 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll'){
        throw 'Custom binary paths require -SkipBuild and separately built production DLLs.'
    }
    & (Join-Path $root 'tools/Build-HipBackend.ps1')
    & (Join-Path $root 'tools/Build-OptiScaler.ps1')
}
$files=[ordered]@{
    'OptiScaler.dll'=$FrontendPath
    'dlssnr_hip_scheduled_bridge.dll'=$HipBackendPath
    'OptiScaler.ini'='OptiScaler.ini'
    'README.md'='README.md'
    'Changelog.md'='Changelog.md'
    'VERSION.txt'='VERSION.txt'
    'LICENSE'='LICENSE'
    'docs/README.md'='docs/README.md'
    'docs/INSTALLATION.md'='docs/INSTALLATION.md'
    'docs/SETTINGS.md'='docs/SETTINGS.md'
    'docs/MODEL.md'='docs/MODEL.md'
    'docs/COMPATIBILITY.md'='docs/COMPATIBILITY.md'
    'docs/NVIDIA.md'='docs/NVIDIA.md'
    'docs/TROUBLESHOOTING.md'='docs/TROUBLESHOOTING.md'
    'docs/ARCHITECTURE.md'='docs/ARCHITECTURE.md'
    'docs/BUILDING.md'='docs/BUILDING.md'
    'tools/README.md'='tools/README.md'
    'docs/CREDITS.md'='docs/CREDITS.md'
    'docs/upstream/README.md'='docs/upstream/README.md'
    'docs/release-notes/r1-0.8.91.md'='docs/release-notes/r1-0.8.91.md'
    'docs/release-notes/r2-0.8.91.md'='docs/release-notes/r2-0.8.91.md'
    'docs/release-notes/r3-0.8.91.md'='docs/release-notes/r3-0.8.91.md'
    'docs/release-notes/r3-0.8.91_fix-1.md'='docs/release-notes/r3-0.8.91_fix-1.md'
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
    $source=if([IO.Path]::IsPathRooted($entry.Value)){$entry.Value}else{Join-Path $root $entry.Value}
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){throw "Missing release input: $source"}
}
$converterFiles=@('write_runtime_package.py','model_weight_provider.py','weights_ht.py',
    'model_schemas.py','qmma_layout.py','research_layouts.py')
foreach($name in $converterFiles){
    $source=Join-Path $root "tools/model_converter/$name"
    if(-not(Test-Path -LiteralPath $source -PathType Leaf)){throw "Missing converter source: $source"}
}
$frontendBinary=if([IO.Path]::IsPathRooted($FrontendPath)){$FrontendPath}else{Join-Path $root $FrontendPath}
$hipBinary=if([IO.Path]::IsPathRooted($HipBackendPath)){$HipBackendPath}else{Join-Path $root $HipBackendPath}
$frontendExports=& dumpbin.exe /nologo /exports $frontendBinary
if($LASTEXITCODE -ne 0){throw 'Frontend export inspection failed.'}
$hipExports=& dumpbin.exe /nologo /exports $hipBinary
$frontendText=$frontendExports -join "`n"
$hipText=$hipExports -join "`n"
if($LASTEXITCODE -ne 0 -or $frontendText -match 'DlssNrNativeTest|TestArm' -or
   $hipText -notmatch 'DlssNrHipBackendGetScheduledApiV2' -or
   $hipText -match 'TestArm|DlssNrHipBackendGetPinnedApiV1'){
    throw 'Production export audit failed.'
}
# Trace builds can have production exports, so inspect their diagnostic markers too.
$binaryText=[Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($frontendBinary))
if($binaryText -match 'FGTRACE|FGORIGIN|FGCOMMIT'){
    throw 'A private FG trace build cannot be packaged as a release.'
}
$ini=Get-Content -LiteralPath (Join-Path $root $files['OptiScaler.ini']) -Raw
$nrSection='(?ms)^\[DlssNr\]\r?\n(?<settings>.*?)(?=^\[|\z)'
$nrDefaults=[regex]::Match($ini,$nrSection).Groups['settings'].Value
$fgSection='(?ms)^\[FrameGen\]\r?\n(?<settings>.*?)(?=^\[|\z)'
$fgDefaults=[regex]::Match($ini,$fgSection).Groups['settings'].Value
$otherDefaults=[regex]::Replace([regex]::Replace($ini,$nrSection,''),$fgSection,'')
if($ini -notmatch '(?mi)^TargetProcessName=auto\s*$' -or
   $nrDefaults -notmatch '(?mi)^Enabled=true\s*$' -or
   $fgDefaults -notmatch '(?mi)^Enabled=true\s*$' -or
   $otherDefaults -match '(?mi)^Enabled=true\s*$' -or $ini -match '(?mi)^ModelPath=\S+' -or
   $fgDefaults -notmatch '(?mi)^FGInput=FSRFG30\s*$' -or $fgDefaults -notmatch '(?mi)^FGOutput=XeFG\s*$' -or
   $ini -match '(?mi)^(FinishedPicture|DeferredDLSS|AdaMfgUnlock|Fsr4EnableWatermark|EnableWatermark)=true\s*$' -or
   $ini -match '(?mi)^DebugView=(true|[1-9][0-9]*)\s*$' -or
   $ini -notmatch '(?mi)^FfxDx12SRPath=auto\s*$'){
    throw 'Default INI enables an experimental or machine-specific setting.'
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null
foreach($entry in $files.GetEnumerator()){
    $destination=Join-Path $stage $entry.Key
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    $source=if([IO.Path]::IsPathRooted($entry.Value)){$entry.Value}else{Join-Path $root $entry.Value}
    Copy-Item -LiteralPath $source -Destination $destination
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
