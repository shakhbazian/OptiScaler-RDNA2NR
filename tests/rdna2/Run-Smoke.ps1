param([Parameter(Mandatory=$true)][string]$ModelPath,
      [string]$CompanionPath='',
      [string]$FrontendPath='',
      [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Label='candidate',
      [ValidateSet('standard','notifier','queue_switch','resize','drs','wrapper','early_reset','hdr',
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
$companion=if($CompanionPath){(Resolve-Path -LiteralPath $CompanionPath).Path}
           else{Join-Path $root 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll'}
if(-not $SkipBuild){
    & (Join-Path $root 'tools/Build-HipBackend.ps1')
    & (Join-Path $root 'tools/Build-OptiScaler.ps1')
    & (Join-Path $root 'tests/rdna2/Build-Hosts.ps1')
}
$out=Join-Path $root 'build/tests/rdna2'
$stage=Join-Path $out 'stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
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
    $config=@('[Upscalers]','Dx12Upscaler=fsr22_12','[DLSS]','Enabled=false',
        '[DlssNr]',"Enabled=$((($mode -eq 1) -or ($mode -eq 2)).ToString().ToLowerInvariant())",'Backend=4',
        ('ModelPath='+(Join-Path $stage 'dlssnr_gfx1030_v1.nrwgt')),
        "RunBeforeSR=$($before.ToString().ToLowerInvariant())",'Style=2','LocalTone=1',
        'LocalStructure=1','SkinStructure=-1','AutoMask=false','Intensity=1',
        "WhitePointSource=$(if($Scenario -eq 'exposure'){'1'}else{'0'})",
        'TemporalAccumulation=true',('WorkingScale='+$WorkingScale.ToString([Globalization.CultureInfo]::InvariantCulture)),
        "ApplyModel=$(($Scenario -ne 'apply_off').ToString().ToLowerInvariant())",
        "TransferStrength=$(if($Scenario -eq 'transfer_zero'){'0'}else{'1'})",
        '[Hotfix]','ColorResourceBarrier=64','OutputResourceBarrier=8',
        'MotionVectorResourceBarrier=64','ExposureResourceBarrier=64','[ProcessFilter]',
        'ProcessExclusionList=optiscaler_dx12_evaluate_host.exe')
    $config | Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=Join-Path $out 'optiscaler_dx12_evaluate_host.exe'
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
