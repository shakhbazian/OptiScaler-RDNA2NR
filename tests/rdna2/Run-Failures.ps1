param([Parameter(Mandatory=$true)][string]$ModelPath)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=Join-Path $root 'build/tests/rdna2'
$model=(Resolve-Path -LiteralPath $ModelPath).Path
$hostExe=Join-Path $out 'optiscaler_dx12_evaluate_host.exe'
$raw=Join-Path $out 'synthetic_160x90.rgba.f16'
if(-not(Test-Path -LiteralPath $raw)){throw 'Run-Smoke.ps1 once to create the deterministic input'}

foreach($case in @('missing-model','corrupt-model','missing-companion')){
    $stage=Join-Path $out "failure-$case"
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item -LiteralPath (Join-Path $root 'x64/Release/OptiScaler.dll') -Destination $stage -Force
    if($case -ne 'missing-companion'){
        Copy-Item -LiteralPath (Join-Path $root 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll') -Destination $stage -Force
    }
    $package=Join-Path $stage 'model.nrwgt'
    if($case -eq 'corrupt-model'){
        [IO.File]::WriteAllBytes($package,[byte[]](1..32))
    } elseif($case -eq 'missing-companion') {
        if(Test-Path -LiteralPath $package){Remove-Item -LiteralPath $package -Force}
        try { New-Item -ItemType HardLink -Path $package -Target $model | Out-Null }
        catch { Copy-Item -LiteralPath $model -Destination $package }
    } else {
        if(Test-Path -LiteralPath $package){Remove-Item -LiteralPath $package -Force}
    }
    @('[Upscalers]','Dx12Upscaler=fsr22_12','[DLSS]','Enabled=false','[DlssNr]',
      'Enabled=true','Backend=4',"ModelPath=$package",'RunBeforeSR=true',
      'Style=2','LocalTone=1','LocalStructure=1','SkinStructure=-1',
      'Intensity=1','TemporalAccumulation=true','WorkingScale=1','ApplyModel=true',
      '[Hotfix]','ColorResourceBarrier=64','OutputResourceBarrier=8',
      'MotionVectorResourceBarrier=64','[ProcessFilter]',
      'ProcessExclusionList=optiscaler_dx12_evaluate_host.exe') |
        Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=$hostExe
    $start.WorkingDirectory=$root
    $start.UseShellExecute=$false
    $start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true
    $start.RedirectStandardError=$true
    $start.Environment['PATH']="${env:SystemRoot}\System32;${env:SystemRoot}"
    foreach($arg in @((Join-Path $stage 'OptiScaler.dll'),$raw,'160','90','1','standard','4')){
        [void]$start.ArgumentList.Add($arg)
    }
    $process=[Diagnostics.Process]::Start($start)
    try {
        $stdout=$process.StandardOutput.ReadToEndAsync()
        $stderr=$process.StandardError.ReadToEndAsync()
        if(-not $process.WaitForExit(45000)){$process.Kill($true);throw "$case timed out"}
        $log=$stdout.Result+"`n"+$stderr.Result
        $log | Set-Content -LiteralPath (Join-Path $out "$case.log")
        $expected=if($case -eq 'missing-companion'){'companion could not load'}
                  elseif($case -eq 'corrupt-model'){'model package (size invalid|rejected)'}
                  else{'model package missing or unreadable'}
        if($process.ExitCode -eq 0 -or $log -notmatch $expected){
            throw "$case did not fail with its own cause (exit $($process.ExitCode))"
        }
        Write-Output "PASS $case rejected with its cause"
    } finally {$process.Dispose()}
}
