param([Parameter(Mandatory=$true)][string]$ModelPath,
      [string]$CompanionPath='',
      [string]$FrontendPath='',
      [ValidateSet('fsr22_12','ffx')][string]$Upscaler='fsr22_12',
      [string]$RuntimeDirectory='',
      [string]$HostDirectory='',
      [string]$OutputRoot='',
      [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Label='candidate',
      [ValidateSet('standard','draw','draw_reuse','draw_reuse_restore','draw_reuse_postcopy','pan_jitter','notifier','queue_switch','resize','drs','wrapper','early_reset','hdr',
                   'exposure','rgba8','rgba32','r11','apply_off','transfer_zero',
                   'recreate','batch','reuse','stream','paced_stream','subrect_bypass',
                   'bundle_bypass','unknown_motion_state','wrapper_bypass','motion32_bypass')]
      [string]$Scenario='standard',
      [ValidateRange(3,12)][int]$Frames=4,
      [ValidateRange(64,3840)][int]$Width=160,
      [ValidateRange(64,2160)][int]$Height=90,
      [ValidateRange(0.25,2.0)][float]$WorkingScale=1.0,
      [switch]$NativeOutput,
      [switch]$FunctionalOnly,
      [switch]$SkipBuild)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$model=(Resolve-Path -LiteralPath $ModelPath).Path
$inputHeader = [IO.File]::OpenRead($model)
try { $header = New-Object byte[] 128; if ($inputHeader.Read($header,0,128) -ne 128) { throw 'Model header truncated.' } }
finally { $inputHeader.Dispose() }
$graph = [BitConverter]::ToUInt32($header,8)
if ($graph -notin @(1,2)) { throw 'Unsupported model graph.' }
$modelSetting = if ($graph -eq 2) { 'ModelBPath' } else { 'ModelPath' }
$modelChoice = if ($graph -eq 2) { 'b' } else { 'a' }

$companion=if($CompanionPath){(Resolve-Path -LiteralPath $CompanionPath).Path}
           else{Join-Path $root 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll'}
if($HostDirectory -and -not $SkipBuild){throw 'Build custom hosts separately and pass -SkipBuild.'}
if(-not $SkipBuild){
    & (Join-Path $root 'tools/Build-HipBackend.ps1')
    & (Join-Path $root 'tools/Build-OptiScaler.ps1')
    & (Join-Path $root 'tests/rdna2/Build-Hosts.ps1')
}
$hostRoot=if($HostDirectory){(Resolve-Path -LiteralPath $HostDirectory).Path}else{Join-Path $root 'build/tests/rdna2'}
$out=if($OutputRoot){[IO.Path]::GetFullPath($OutputRoot)}else{$hostRoot}
New-Item -ItemType Directory -Force -Path $out | Out-Null
$stage=Join-Path $out 'stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
if($RuntimeDirectory){
    $runtimeStage=Join-Path $stage 'OptiScaler'
    New-Item -ItemType Directory -Force -Path $runtimeStage | Out-Null
    foreach($name in @('amd_fidelityfx_loader_dx12.dll','amd_fidelityfx_upscaler_dx12.dll')){
        Copy-Item -LiteralPath (Join-Path $RuntimeDirectory $name) -Destination $runtimeStage -Force
    }
}
$inputs=@{
    'OptiScaler.dll'=$(if($FrontendPath){(Resolve-Path -LiteralPath $FrontendPath).Path}else{Join-Path $root 'x64/Release/OptiScaler.dll'})
    'dlssnr_hip_scheduled_bridge.dll'=$companion
    'dlssnr_gfx1030_v1.nrwgt'=$model
}
foreach($entry in $inputs.GetEnumerator()){
    $destination=Join-Path $stage $entry.Key
    if($entry.Key -eq 'dlssnr_gfx1030_v1.nrwgt'){
        if(Test-Path -LiteralPath $destination){Remove-Item -LiteralPath $destination -Force}
        try { New-Item -ItemType HardLink -Path $destination -Target $entry.Value | Out-Null }
        catch { Copy-Item -LiteralPath $entry.Value -Destination $destination }
    } else {
        Copy-Item -LiteralPath $entry.Value -Destination $destination -Force
    }
}
$raw=Join-Path $out "synthetic_${Width}x${Height}.rgba.f16"
if(-not ('SyntheticFrame' -as [type])){
    Add-Type -Path (Join-Path $PSScriptRoot 'SyntheticFrame.cs')
}
[SyntheticFrame]::Write($raw,$Width,$Height)
function Invoke-Mode([string]$name,[int]$mode,[bool]$before){
    $config=@('[Upscalers]',("Dx12Upscaler=$Upscaler"),'[DLSS]','Enabled=false',
        '[DlssNr]',"Enabled=$((($mode -eq 1) -or ($mode -eq 2)).ToString().ToLowerInvariant())",'Backend=4',
        ($modelSetting+'='+(Join-Path $stage 'dlssnr_gfx1030_v1.nrwgt')),("HipModel=$modelChoice"),
        "RunBeforeSR=$($before.ToString().ToLowerInvariant())",'Style=2','LocalTone=1',
        'LocalStructure=1','SkinStructure=-1','AutoMask=false','Intensity=1',
        "WhitePointSource=$(if($Scenario -eq 'exposure'){'1'}else{'0'})",
        'TemporalAccumulation=true',('WorkingScale='+$WorkingScale.ToString([Globalization.CultureInfo]::InvariantCulture)),
        "ApplyModel=$(($Scenario -ne 'apply_off').ToString().ToLowerInvariant())",
        "TransferStrength=$(if($Scenario -eq 'transfer_zero'){'0'}else{'1'})",
        '[Hotfix]','ColorResourceBarrier=64','OutputResourceBarrier=8',
        'MotionVectorResourceBarrier=64','ExposureResourceBarrier=64','[ProcessFilter]',
        'ProcessExclusionList=optiscaler_dx12_evaluate_host.exe')
    if($RuntimeDirectory){
        $config += @('[Libraries]',"FfxDx12Path=$runtimeStage","FfxDx12SRPath=$runtimeStage")
    }
    $config | Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=Join-Path $hostRoot 'optiscaler_dx12_evaluate_host.exe'
    $start.WorkingDirectory=$root
    $start.UseShellExecute=$false
    $start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true
    $start.RedirectStandardError=$true
    # The AMD driver supplies HIP. Keep the SDK out of the child process PATH.
    $start.Environment['PATH']="${env:SystemRoot}\System32;${env:SystemRoot}"
    if($NativeOutput){$start.Environment['NR_HOST_NATIVE_OUTPUT']='1'}
    $start.Environment['NR_HOST_REFERENCE_SCALE']=$WorkingScale.ToString([Globalization.CultureInfo]::InvariantCulture)
    foreach($arg in @((Join-Path $stage 'OptiScaler.dll'),$raw,"$Width","$Height","$mode",$Scenario,"$Frames")){
        [void]$start.ArgumentList.Add($arg)
    }
    $process=[Diagnostics.Process]::Start($start)
    try {
        $stdout=$process.StandardOutput.ReadToEndAsync()
        $stderr=$process.StandardError.ReadToEndAsync()
        if(-not $process.WaitForExit(180000)){$process.Kill($true);throw "$name timed out"}
        $log=$stdout.Result+"`n"+$stderr.Result
        $log | Set-Content -LiteralPath (Join-Path $out "$Label-$Scenario-${Width}x${Height}-$name.log")
        if($process.ExitCode -ne 0 -or $log -notmatch 'debug_clean=1'){
            throw "$name failed: $log"
        }
        if($Upscaler -eq 'ffx' -and $log -notmatch 'FfxApiProxy::InitFfxDx12_SR LoadResult: true'){
            throw "$name did not load the requested FFX runtime; see its log"
        }
        if($Upscaler -eq 'ffx' -and $RuntimeDirectory){
            $expectedRuntime=Join-Path $runtimeStage 'amd_fidelityfx_upscaler_dx12.dll'
            if(-not $log.Contains("FfxApiProxy::InitFfxDx12_SR Loaded from $expectedRuntime")){
                throw "$name loaded a different FFX runtime; see its log"
            }
        }
        $hash=[regex]::Match($log,'PASS OptiScaler native DX12 Evaluate[^\r\n]*sha256=([0-9A-F]+)')
        if(-not $hash.Success){throw "$name output hash missing"}
        Write-Host "$name $($hash.Groups[1].Value)"
        return $hash.Groups[1].Value
    } finally {$process.Dispose()}
}
$bypass=$Scenario -in @('subrect_bypass','bundle_bypass','unknown_motion_state',
                        'wrapper_bypass','motion32_bypass')
$off=if($FunctionalOnly){$null}else{Invoke-Mode 'off' 0 $true}
$pre=Invoke-Mode 'pre' 1 $true
$preReference=if($FunctionalOnly -or $bypass -or $WorkingScale -gt 1.0){$null}else{Invoke-Mode 'pre-reference' 3 $true}
$post=Invoke-Mode 'post' 2 $false
$postReference=if($FunctionalOnly -or $bypass -or $WorkingScale -gt 1.0){$null}else{Invoke-Mode 'post-reference' 4 $false}
if($FunctionalOnly -or $WorkingScale -gt 1.0){
    Write-Output "PASS $Scenario ${Width}x${Height} pre/post functional-only debug-clean"
    return
}
if($bypass){
    if($pre -ne $off -or $post -ne $off){throw 'Rejected input changed the game frame'}
    Write-Output "PASS $Scenario raw fallback debug-clean"
    return
}
if($Scenario -notin @('apply_off','transfer_zero') -and
   ($pre -eq $off -or $post -eq $off)){throw 'NR did not change the frame'}
if($pre -ne $preReference -or $post -ne $postReference){
    throw 'Migrated common frontend differs from independent codec oracle'
}
Write-Output "PASS $Scenario source-built DX12, HIP on driver PATH, common-codec parity"
