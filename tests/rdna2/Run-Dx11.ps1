param([Parameter(Mandatory=$true)][string]$ModelPath,
      [string]$FrontendPath='',
      [string]$CompanionPath='',
      [ValidateRange(0.25,2.0)][float]$WorkingScale=1.0,
      [ValidateSet('static','pan','pan_reset','occlusion','occlusion_reset','resize')]
      [string]$Scenario='static',
      [ValidateSet(0,3,4)][uint32]$Backend=4,
      [switch]$ExpectUnavailable,
      [switch]$Post)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=Join-Path $root 'build/tests/rdna2'
$raw=Join-Path $out 'synthetic_160x90.rgba.f16'
if(-not(Test-Path -LiteralPath $raw)){throw 'Run-Smoke.ps1 once to create the deterministic input'}
$stage=Join-Path $out 'dx11-stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$frontend=if($FrontendPath){(Resolve-Path -LiteralPath $FrontendPath).Path}else{Join-Path $root 'x64/Release/OptiScaler.dll'}
$companion=if($CompanionPath){(Resolve-Path -LiteralPath $CompanionPath).Path}else{Join-Path $root 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll'}
Copy-Item -LiteralPath $frontend -Destination $stage -Force
Copy-Item -LiteralPath $companion -Destination $stage -Force
$model=(Resolve-Path -LiteralPath $ModelPath).Path
$stagedModel=Join-Path $stage 'model.nrwgt'
if(Test-Path -LiteralPath $stagedModel){Remove-Item -LiteralPath $stagedModel -Force}
try { New-Item -ItemType HardLink -Path $stagedModel -Target $model | Out-Null }
catch { Copy-Item -LiteralPath $model -Destination $stagedModel }
$hashes=@{}
foreach($enabled in @($false,$true)){
    $label=if($enabled){'nr'}else{'raw'}
    @('[Upscalers]','Dx11Upscaler=fsr22_12','[DLSS]','Enabled=false',
      '[DlssNr]',"Enabled=$($enabled.ToString().ToLowerInvariant())","Backend=$Backend",
      "ModelPath=$stagedModel","RunBeforeSR=$((!$Post).ToString().ToLowerInvariant())",
      'Style=2','LocalTone=1','LocalStructure=1','SkinStructure=-1',
      'AutoMask=false','Intensity=1','TemporalAccumulation=true',('WorkingScale='+$WorkingScale.ToString([Globalization.CultureInfo]::InvariantCulture)),
      'ApplyModel=true','[InitFlags]','AutoExposure=true',
      '[ProcessFilter]','ProcessExclusionList=optiscaler_dx11_evaluate_host.exe') |
        Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
    $results=Join-Path $out "dx11-$Scenario-$($Post.IsPresent)-$label"
    New-Item -ItemType Directory -Force -Path $results | Out-Null
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=Join-Path $out 'optiscaler_dx11_evaluate_host.exe'
    $start.WorkingDirectory=$root
    $start.UseShellExecute=$false
    $start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true
    $start.RedirectStandardError=$true
    $start.Environment['PATH']="${env:SystemRoot}\System32;${env:SystemRoot}"
    $start.Environment['NR_HOST_DEBUG']='1'
    foreach($arg in @((Join-Path $stage 'OptiScaler.dll'),$raw,'160','90',$results,$Scenario,'4')){
        [void]$start.ArgumentList.Add($arg)
    }
    $process=[Diagnostics.Process]::Start($start)
    try {
        $stdout=$process.StandardOutput.ReadToEndAsync()
        $stderr=$process.StandardError.ReadToEndAsync()
        if(-not $process.WaitForExit(120000)){$process.Kill($true);throw "DX11 $label timed out"}
        $log=$stdout.Result+"`n"+$stderr.Result
        $log | Set-Content -LiteralPath (Join-Path $out "dx11-$Scenario-$($Post.IsPresent)-$label.log")
        if($process.ExitCode -ne 0 -or $log -notmatch 'bridge_debug_clean=1' -or
           $log -notmatch 'PASS real OptiScaler DX11 Evaluate'){
            throw "DX11 $label failed: $log"
        }
        $finalFile=if($Scenario -eq 'resize'){'resize_return.rgba.f16'}else{'evaluate_3.rgba.f16'}
        $hashes[$label]=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $results $finalFile)).Hash
    } finally {$process.Dispose()}
}
if($ExpectUnavailable){
    if($hashes['nr'] -ne $hashes['raw']){throw 'Forced mismatched backend changed the frame'}
} elseif($hashes['nr'] -eq $hashes['raw']){throw 'DX11 NR did not change the frame'}
Write-Output "PASS DX11-on-12 $Scenario post=$($Post.IsPresent) raw=$($hashes['raw']) NR=$($hashes['nr'])"
