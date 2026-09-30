param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [Parameter(Mandatory=$true)][string]$InputRaw,
    [string]$DllPath,
    [ValidateSet('present_queue','standard','wrapper','queue_switch')][string]$Scenario='present_queue',
    [ValidateSet('pre','post')][string]$Placement='pre',
    [ValidatePattern('^[A-Za-z0-9_-]*$')][string]$Trial='',
    [switch]$ExpectBypass,
    [switch]$DebugLayer,
    [switch]$SkipBuild
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/Enter-Toolchain.ps1" -PlatformToolset v145 -MsvcToolsVersion 14.44
$out=Join-Path $root 'build/queue-regression'
$stage=Join-Path $out 'stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
if(-not $DllPath){$DllPath=Join-Path $root 'x64/Release/OptiScaler.dll'}
$exe=Join-Path $out 'optiscaler_dx12_evaluate_host.exe'
if(-not $SkipBuild){
    & cl.exe /nologo /std:c++17 /EHsc /O2 /MD /W4 /WX "/I$root/external/nvngx_dlss_sdk" "/Fo:$out/" `
        "$root/rdna2_nr/src/optiscaler_dx12_evaluate_host.cpp" /link d3d12.lib dxgi.lib bcrypt.lib dbghelp.lib user32.lib "/OUT:$exe"
    if($LASTEXITCODE){throw 'Queue regression host build failed'}
}
Copy-Item -LiteralPath $DllPath -Destination (Join-Path $stage 'dxgi.dll') -Force
Copy-Item -LiteralPath (Join-Path $root 'build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll') -Destination $stage -Force
$model=(Resolve-Path -LiteralPath $ModelPath).Path
$input=(Resolve-Path -LiteralPath $InputRaw).Path
@('[Upscalers]','Dx12Upscaler=fsr22_12','[DLSS]','Enabled=false',
    '[DlssNr]','Enabled=true','Backend=3',"ModelPath=$model",("RunBeforeSR="+($Placement -eq 'pre').ToString().ToLowerInvariant()),'Style=2',
    'LocalTone=1','LocalStructure=1','SkinStructure=-1','AutoMask=false','Intensity=1',
    'WorkingScale=1','ApplyModel=true','AutoCapture=false',
    '[Hotfix]','ExposureResourceBarrier=64','ColorResourceBarrier=64',
    'OutputResourceBarrier=8','MotionVectorResourceBarrier=64',
    '[ProcessFilter]','ProcessExclusionList=optiscaler_dx12_evaluate_host.exe') |
    Set-Content -LiteralPath (Join-Path $stage 'OptiScaler.ini') -Encoding utf8
# Only the diagnostic frontend exposes the queue assignment used by this
# regression. Standard/wrapper runs can use the production DLL directly.
$label=if($ExpectBypass){'before'}else{'after'}
$label+="-$Placement"
if($Trial){$label+="-$Trial"}
$mode=if($Placement -eq 'pre'){1}else{2}
$stdout=Join-Path $out "$label-$Scenario.stdout.txt"
$stderr=Join-Path $out "$label-$Scenario.stderr.txt"
$previousDebug=$env:NR_NATIVE_NO_DEBUG
try{
    $env:NR_NATIVE_NO_DEBUG=if($DebugLayer){$null}else{'1'}
    $process=Start-Process -FilePath $exe -WindowStyle Hidden -PassThru -WorkingDirectory $root `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -ArgumentList @(('"'+(Join-Path $stage 'dxgi.dll')+'"'),('"'+$input+'"'),'160','90',"$mode",$Scenario,'4')
    if(-not $process.WaitForExit(120000)){$process.Kill();throw 'Queue regression timed out'}
    $process.Refresh()
    $log=[IO.File]::ReadAllText($stdout)
    $errors=[IO.File]::ReadAllText($stderr)
    if($ExpectBypass){
        if($process.ExitCode -eq 0 -or $log -notmatch 'warmup-only frame' -or $errors -notmatch 'native processing coverage'){
            throw "Did not reproduce the expected perpetual warmup: $log $errors"
        }
    }elseif($process.ExitCode -ne 0 -or $log -notmatch "NativeCoverage mode=$mode applied=4 debug_clean=1"){
        throw "NR did not apply all four frames: $log $errors"
    }
    if($Scenario -eq 'present_queue' -and $log -notmatch 'NativePresentQueue distinct=1'){
        throw 'Presentation queue was not distinct from the Evaluate queue'
    }
    [ordered]@{scenario=$Scenario;placement=$Placement;expectedBypass=[bool]$ExpectBypass;exitCode=$process.ExitCode;
        dllSha256=(Get-FileHash -LiteralPath $DllPath).Hash;output=$stdout;errors=$stderr} |
        ConvertTo-Json | Set-Content (Join-Path $out "$label-$Scenario.json")
    Write-Output "PASS queue regression: $label $Scenario"
}finally{$env:NR_NATIVE_NO_DEBUG=$previousDebug}
