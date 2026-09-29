param([Parameter(Mandatory=$true)][string]$ModelPath,
      [Parameter(Mandatory=$true)][string]$CompetitorExe,
      [ValidateRange(16,120)][int]$Frames=60)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=Join-Path $root 'build/tests/rdna2/coload'
$stage=Join-Path $out 'stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$input=Join-Path $out 'input_960x540.rgba.f16'
if(-not ('SyntheticFrame' -as [type])){Add-Type -Path (Join-Path $PSScriptRoot 'SyntheticFrame.cs')}
[SyntheticFrame]::Write($input,960,540)
Copy-Item (Join-Path $root 'x64/Release/OptiScaler.dll') $stage -Force
Copy-Item (Join-Path $root 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll') $stage -Force
$model=(Resolve-Path -LiteralPath $ModelPath).Path
$stagedModel=Join-Path $stage 'model.nrwgt'
if(Test-Path -LiteralPath $stagedModel){Remove-Item -LiteralPath $stagedModel -Force}
try{New-Item -ItemType HardLink -Path $stagedModel -Target $model | Out-Null}
catch{Copy-Item -LiteralPath $model -Destination $stagedModel}
$competitor=(Resolve-Path -LiteralPath $CompetitorExe).Path

function Start-Probe([string]$exe,[string[]]$arguments){
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=$exe
    $start.WorkingDirectory=$root
    $start.UseShellExecute=$false
    $start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true
    $start.RedirectStandardError=$true
    $start.Environment['PATH']="${env:SystemRoot}\System32;${env:SystemRoot}"
    foreach($argument in $arguments){[void]$start.ArgumentList.Add($argument)}
    $process=[Diagnostics.Process]::Start($start)
    return [pscustomobject]@{Process=$process;Out=$process.StandardOutput.ReadToEndAsync();Err=$process.StandardError.ReadToEndAsync()}
}
function Finish-Probe($running,[string]$label){
    try{
        if(-not $running.Process.WaitForExit(120000)){
            $running.Process.Kill($true)
            throw "$label timed out"
        }
        $log=$running.Out.Result+"`n"+$running.Err.Result
        $log | Set-Content -LiteralPath (Join-Path $out "$label.log")
        if($running.Process.ExitCode -ne 0){throw "$label exited $($running.Process.ExitCode): $log"}
        return $log
    }finally{$running.Process.Dispose()}
}

$runs=@(
    @{name='idle-a';enabled=$true;loaded=$false},
    @{name='loaded-a';enabled=$true;loaded=$true},
    @{name='loaded-b';enabled=$true;loaded=$true},
    @{name='idle-b';enabled=$true;loaded=$false},
    @{name='raw-idle';enabled=$false;loaded=$false},
    @{name='raw-loaded';enabled=$false;loaded=$true}
)
$records=@()
foreach($run in $runs){
    $name=$run.name
    @('[Upscalers]','Dx11Upscaler=fsr22_12','[DLSS]','Enabled=false',
      '[DlssNr]',"Enabled=$($run.enabled.ToString().ToLowerInvariant())",'Backend=4',
      "ModelPath=$stagedModel",'RunBeforeSR=true','Style=2','LocalTone=1',
      'LocalStructure=1','SkinStructure=-1','AutoMask=false','Intensity=1',
      'TemporalAccumulation=true','WorkingScale=1','ApplyModel=true',
      '[InitFlags]','AutoExposure=true','[ProcessFilter]',
      'ProcessExclusionList=optiscaler_dx11_evaluate_host.exe') |
        Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
    $results=Join-Path $out $name
    New-Item -ItemType Directory -Force -Path $results | Out-Null
    $load=$null
    try{
        if($run.loaded){
            $load=Start-Probe $competitor @('background','60')
            Start-Sleep -Milliseconds 500
            if($load.Process.HasExited){throw 'GPU competitor exited before host run'}
        }
        $probe=Start-Probe (Join-Path $root 'build/tests/rdna2/optiscaler_dx11_evaluate_host.exe') `
            @((Join-Path $stage 'OptiScaler.dll'),$input,'960','540',$results,'occlusion',"$Frames",'timed')
        $log=Finish-Probe $probe $name
        if($run.loaded -and $load.Process.HasExited){throw 'GPU competitor stopped during host run'}
        if($log -notmatch 'PASS real OptiScaler DX11 Evaluate' -or
           $log -notmatch 'pending_after_shutdown=\d+ cleanup_wait_ms=[0-9.]+ faulted=0'){
            throw "$name did not complete the full path"
        }
        $appliedMatch=[regex]::Match($log,'TimedNr applied=(\d+)')
        if(-not $appliedMatch.Success){throw "$name lacks applied count"}
        $applied=[int]$appliedMatch.Groups[1].Value
        if($run.enabled -and $applied -ne $Frames){throw "$name applied $applied of $Frames frames"}
        $summary=Get-Content (Join-Path $results 'host-summary.json') -Raw | ConvertFrom-Json
        if($summary.frames -ne $Frames -or -not $summary.finite){throw "$name incomplete summary"}
        $frameMs=@([regex]::Matches($log,'HostTiming frame=\d+ evaluate_ms=([0-9.]+)') |
            ForEach-Object {[double]::Parse($_.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture)})
        $ordered=@($frameMs | Select-Object -Skip 5 | Sort-Object)
        $records += [ordered]@{
            name=$name;enabled=$run.enabled;loaded=$run.loaded;frames=$Frames;applied=$applied
            loopMs=$summary.evaluateLoopMs;msPerFrame=$summary.evaluateLoopMs/$Frames
            medianEvaluateMs=$ordered[[int][math]::Floor($ordered.Count/2)]
            finalReadbackMs=$summary.finalReadbackMs
            finalSha256=(Get-FileHash (Join-Path $results 'evaluate_last.rgba.f16')).Hash
        }
        Write-Output "$name $([math]::Round($summary.evaluateLoopMs/$Frames,2)) ms/frame, applied $applied/$Frames"
    }finally{
        if($load){
            if(-not $load.Process.HasExited){$load.Process.Kill($true)}
            $load.Process.WaitForExit()
            $load.Process.Dispose()
        }
    }
}
if(@($records[0..3] | ForEach-Object {$_['finalSha256']} | Sort-Object -Unique).Count -ne 1){
    throw 'NR output changed across paired timing runs'
}
if($records[4]['finalSha256'] -ne $records[5]['finalSha256']){throw 'Raw output changed under load'}
$receipt=[ordered]@{contract='source-built-dx11-paired-coload-v1';date=(Get-Date).ToString('o')
    input='960x540';output='1920x1080';frames=$Frames
    frontendSha256=(Get-FileHash (Join-Path $stage 'OptiScaler.dll')).Hash
    hipSha256=(Get-FileHash (Join-Path $stage 'dlssnr_hip_scheduled_bridge.dll')).Hash
    competitorSha256=(Get-FileHash $competitor).Hash;records=$records}
$receipt | ConvertTo-Json -Depth 7 | Set-Content (Join-Path $out 'receipt.json') -Encoding utf8
Write-Output 'PASS paired co-load: same output, all requested NR frames, clean retirement'
