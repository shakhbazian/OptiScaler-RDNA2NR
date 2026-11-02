# Run the A/B product qualification on source-built hosts, outside a game.
param([Parameter(Mandatory=$true)][string]$ModelA,
      [Parameter(Mandatory=$true)][string]$ModelB,
      [Parameter(Mandatory=$true)][string]$Frontend,
      [Parameter(Mandatory=$true)][string]$DiagnosticFrontend,
      [Parameter(Mandatory=$true)][string]$Companion,
      [Parameter(Mandatory=$true)][string]$OutputRoot)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=[IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $out | Out-Null
$checks=@()
function Smoke([string]$Model,[string]$Label,[string]$Scenario='standard',[int]$Width=160,[int]$Height=90,[float]$Scale=1,[bool]$Functional=$false){
    & "$PSScriptRoot/Run-Smoke.ps1" -SkipBuild -ModelPath $Model -FrontendPath $Frontend -CompanionPath $Companion `
        -OutputRoot $out -Label $Label -Scenario $Scenario -Width $Width -Height $Height -WorkingScale $Scale `
        -NativeOutput -Frames 4 -FunctionalOnly:$Functional | Out-Host
    $script:checks+=@{name="$Label/$Scenario";passed=$true}
}
Smoke $ModelA 'A'
Smoke $ModelB 'B'
Smoke $ModelB 'B720' 'standard' 1280 720
Smoke $ModelB 'B75' 'standard' 1280 720 0.75
foreach($scenario in @('pan_jitter','queue_switch','resize','wrapper','rgba8','rgba32','r11','hdr','exposure','apply_off','subrect_bypass','paced_stream')){Smoke $ModelB 'B' $scenario}
Smoke $ModelB 'B4K' 'standard' 3840 2160 1 $true
foreach($post in @($false,$true)){
    & "$PSScriptRoot/Run-Dx11.ps1" -ModelPath $ModelB -FrontendPath $Frontend -CompanionPath $Companion `
        -OutputRoot $out -WorkingScale 0.75 -Post:$post | Out-Host
    $checks+=@{name="B/DX11/post=$post";passed=$true}
}
& "$PSScriptRoot/Run-Failures.ps1" -ModelPath $ModelB -FrontendPath $Frontend -CompanionPath $Companion -OutputRoot $out | Out-Host
$checks+=@{name='B/failures';passed=$true}
# This stage uses private menu hooks. The packaged product DLL is audited separately.
$stage=Join-Path $out 'model-switch'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -LiteralPath $DiagnosticFrontend -Destination (Join-Path $stage 'OptiScaler.dll') -Force
Copy-Item -LiteralPath $Companion -Destination (Join-Path $stage 'dlssnr_hip_scheduled_bridge.dll') -Force
@('[Upscalers]','Dx12Upscaler=fsr22_12','[DLSS]','Enabled=false',
  '[DlssNr]','Enabled=true','Backend=4','HipModel=b',"ModelPath=$ModelA","ModelBPath=$ModelB",
  'RunBeforeSR=true','WorkingScale=0.75','Style=2','LocalTone=1','LocalStructure=1',
  'SkinStructure=-1','AutoMask=false','Intensity=1','ApplyModel=true','TemporalAccumulation=true',
  '[Log]','LogToFile=true','LogLevel=2','[Hotfix]','ColorResourceBarrier=64','OutputResourceBarrier=8','MotionVectorResourceBarrier=64',
  '[ProcessFilter]','ProcessExclusionList=optiscaler_dx12_evaluate_host.exe') |
    Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
foreach($scenario in @('model_cycle','controls','toggle','temporal_toggle','scale_cycle')){
    $caseRoot=Join-Path $stage $scenario
    New-Item -ItemType Directory -Force -Path $caseRoot | Out-Null
    $ini=Join-Path $stage 'OptiScaler.ini'
    $config=[IO.File]::ReadAllText($ini)
    $config=[regex]::Replace($config,'(?m)^WorkingScale=.*$',$(if($scenario -eq 'scale_cycle'){'WorkingScale=1'}else{'WorkingScale=0.75'}))
    [IO.File]::WriteAllText($ini,$config,[Text.UTF8Encoding]::new($false))
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=Join-Path $root 'build/tests/rdna2/optiscaler_dx12_evaluate_host.exe'
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    $start.Environment['PATH']="${env:SystemRoot}/System32;${env:SystemRoot}"
    $start.Environment['NR_HOST_NATIVE_OUTPUT']='1'
    $start.Environment['NR_HOST_OUTPUT_DIR']=Join-Path $caseRoot 'frames'
    $frames=if($scenario -eq 'scale_cycle'){'9'}else{'6'}
    foreach($arg in @((Join-Path $stage 'OptiScaler.dll'),(Join-Path $out 'synthetic_160x90.rgba.f16'),'160','90','1',$scenario,$frames)){[void]$start.ArgumentList.Add($arg)}
    $process=[Diagnostics.Process]::Start($start)
    try{
        $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
        if(-not $process.WaitForExit(180000)){$process.Kill($true);throw "$scenario test timed out"}
        $text=$stdout.Result+"`n"+$stderr.Result
        $text | Set-Content -LiteralPath (Join-Path $caseRoot 'run.log')
        if($process.ExitCode -ne 0 -or $text -notmatch 'PASS OptiScaler native DX12 Evaluate' -or $text -notmatch 'debug_clean=1' -or
           ($scenario -eq 'model_cycle' -and $text -notmatch 'NativeModelCycle frame=4 selected=0 ready=1')){
            throw "$scenario test failed: $text"
        }
        if($scenario -eq 'model_cycle'){
            $first=(Get-FileHash (Join-Path $caseRoot 'frames/0.rgba.f16')).Hash
            $returned=(Get-FileHash (Join-Path $caseRoot 'frames/4.rgba.f16')).Hash
            $other=(Get-FileHash (Join-Path $caseRoot 'frames/2.rgba.f16')).Hash
            if($first -ne $returned -or $first -eq $other){throw 'Live model switch did not restore B exactly or did not change to A'}
        }
        $checks+=@{name="B/private/$scenario";passed=$true}
        Write-Output "PASS B private $scenario"
    }finally{$process.Dispose()}
}
[ordered]@{date=(Get-Date).ToString('o');checks=$checks;
    frontendSha256=(Get-FileHash $Frontend).Hash;companionSha256=(Get-FileHash $Companion).Hash} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'integration-results.json')
Write-Output "PASS A/B product integration: $($checks.Count) groups, product/diagnostic DLLs isolated."
