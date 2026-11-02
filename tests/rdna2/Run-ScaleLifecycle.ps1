# Reproduce the game's 75 -> 80 -> 75 percent transition on source-built hosts.
param([Parameter(Mandatory=$true)][string]$DiagnosticFrontend,
      [Parameter(Mandatory=$true)][string]$Companion,
      [Parameter(Mandatory=$true)][string]$ModelB,
      [Parameter(Mandatory=$true)][string]$OutputRoot)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$out=[IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $out | Out-Null
Copy-Item -LiteralPath $DiagnosticFrontend -Destination (Join-Path $out 'OptiScaler.dll') -Force
Copy-Item -LiteralPath $Companion -Destination (Join-Path $out 'dlssnr_hip_scheduled_bridge.dll') -Force
if(-not ('SyntheticFrame' -as [type])){Add-Type -Path (Join-Path $PSScriptRoot 'SyntheticFrame.cs')}
$inputPath=Join-Path $out 'synthetic_1280x720.rgba.f16'
[SyntheticFrame]::Write($inputPath,1280,720)
$checks=@()
foreach($mode in @(1,2)){
    foreach($scenario in @('scale_cycle','scale_stream')){
        $name="$scenario-mode$mode"
        $runtimeLog=Join-Path $out "$name-runtime.log"
        @('[Upscalers]','Dx12Upscaler=fsr22_12','[DLSS]','Enabled=false',
          '[DlssNr]','Enabled=true','Backend=4','HipModel=b',"ModelBPath=$ModelB",
          "RunBeforeSR=$(($mode -eq 1).ToString().ToLowerInvariant())",'WorkingScale=0.75',
          'Style=2','LocalTone=1','LocalStructure=1','SkinStructure=-1','AutoMask=false',
          'Intensity=1','ApplyModel=true','TemporalAccumulation=true',
          '[Log]','LogToFile=true','LogLevel=2',"LogFileName=$runtimeLog",
          '[Hotfix]','ColorResourceBarrier=64','OutputResourceBarrier=8','MotionVectorResourceBarrier=64',
          '[ProcessFilter]','ProcessExclusionList=optiscaler_dx12_evaluate_host.exe') |
            Set-Content -LiteralPath (Join-Path $out 'OptiScaler.ini') -Encoding utf8
        $start=[Diagnostics.ProcessStartInfo]::new()
        $start.FileName=Join-Path $root 'build/tests/rdna2/optiscaler_dx12_evaluate_host.exe'
        $start.UseShellExecute=$false;$start.CreateNoWindow=$true
        $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
        $start.Environment['PATH']="${env:SystemRoot}/System32;${env:SystemRoot}"
        $start.Environment.Remove('NR_HOST_NATIVE_OUTPUT') | Out-Null
        $start.Environment['NR_HOST_TEST_SCALES']='0.75,0.80,0.75'
        $start.Environment['NR_HOST_EXPECT_COMPANION_RETAINED']='1'
        foreach($arg in @((Join-Path $out 'OptiScaler.dll'),$inputPath,'1280','720',"$mode",$scenario,'9')){
            [void]$start.ArgumentList.Add($arg)
        }
        $process=[Diagnostics.Process]::Start($start)
        try{
            $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
            if(-not $process.WaitForExit(120000)){
                # Stop the suite after a hang; never start another GPU workload.
                $process.Kill($true)
                throw "$name timed out; inspect $runtimeLog before running anything else."
            }
            $text=$stdout.Result+"`n"+$stderr.Result
            $text | Set-Content -LiteralPath (Join-Path $out "$name.log")
            if($process.ExitCode -ne 0 -or $text -notmatch 'PASS OptiScaler native DX12 Evaluate' -or
               $text -notmatch 'debug_clean=1' -or $text -notmatch 'retained_after_drain=1'){
                throw "$name failed; see $out"
            }
            $returnedExactly=$null
            if($scenario -eq 'scale_cycle'){
                $first=[regex]::Match($text,'NativeFrame frame=0 sha256=([0-9A-F]+)').Groups[1].Value
                $returned=[regex]::Match($text,'NativeFrame frame=6 sha256=([0-9A-F]+)').Groups[1].Value
                $returnedExactly=$first -and $first -eq $returned
                if(-not $returnedExactly){throw "$name did not restore the initial 75 percent output exactly"}
            }
            $checks+=@{name=$name;passed=$true;returnedOutputBitexact=$returnedExactly;
                coverage=$(if($scenario -eq 'scale_cycle'){'all frames'}else{'safe bypass during preparation'})}
        }finally{$process.Dispose()}
    }
}
[ordered]@{checks=$checks;frontendSha256=(Get-FileHash $DiagnosticFrontend).Hash;
    companionSha256=(Get-FileHash $Companion).Hash} | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $out 'results.json')
Write-Output 'PASS 75/80/75 percent lifecycle, pre/post upscale, clean drain and retained companion'
