param([switch]$NativeTestHooks)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath (Split-Path $PSScriptRoot -Parent)).Path
. "$PSScriptRoot/Enter-Toolchain.ps1" -PlatformToolset v145 -MsvcToolsVersion 14.44
& "$PSScriptRoot/Build-FreeType.ps1"
$environment = @{}
foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
    $key = ([string]$entry.Key).ToUpperInvariant()
    $environment[$key] = [string]$entry.Value
}
$environment['MSBUILDDISABLENODEREUSE'] = '1'
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = (Get-Command MSBuild.exe).Source
$start.WorkingDirectory = $source
$start.UseShellExecute = $false
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.Environment.Clear()
foreach ($entry in $environment.GetEnumerator()) { $start.Environment[$entry.Key] = $entry.Value }
$argsList = @((Join-Path $source 'OptiScaler/OptiScaler.vcxproj'), '/t:Build',
    '/p:Configuration=Release', '/p:Platform=x64', ('/p:SolutionDir=' + $source.Replace('\','/') + '/'),
    '/p:PlatformToolset=v145', "/p:VCToolsVersion=$env:VCTOOLSVERSION", '/p:OptiScalerRtx40Mfg=false',
    '/p:PostBuildEventUseInBuild=false', '/p:PreBuildEventUseInBuild=false',
    '/p:TrackFileAccess=false', '/m:2', '/nr:false', '/nologo', '/v:minimal')
# Isolate compilation from upstream's legacy unquoted post-build file operations.
if($NativeTestHooks){$argsList += '/p:NrNativeTestHooks=true'}
# Packaging is tested separately through its manifest-based package_release.ps1.
$commit = (& git -C $source rev-parse HEAD).Trim()
$date = (& git -C $source show -s --format=%cs HEAD).Trim().Replace('-','')
[IO.File]::WriteAllText((Join-Path $source 'OptiScaler/resource_build_commit.h'), "#define VER_BUILD_COMMIT `"$($commit.Substring(0,8))`"`n")
[IO.File]::WriteAllText((Join-Path $source 'OptiScaler/resource_build_date.h'), "#define VER_BUILD_DATE `"${date}_000000`"`n")
$start.Arguments = ($argsList | ForEach-Object { '"' + $_ + '"' }) -join ' '
$started = Get-Date
$process = [Diagnostics.Process]::Start($start)
$stdout = $process.StandardOutput.ReadToEndAsync()
$stderr = $process.StandardError.ReadToEndAsync()
$process.WaitForExit()
$reportDir = Join-Path $source 'build/reports'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$log = Join-Path $reportDir 'frontend-build.txt'
[IO.File]::WriteAllText($log, $stdout.Result + $stderr.Result)
$dll = Join-Path $source 'x64/Release/OptiScaler.dll'
$receipt = [ordered]@{source=$source;commit=$commit;toolset='v145';msvc=$env:VCTOOLSVERSION;
    standardBuild=(!$NativeTestHooks);nativeTestHooks=[bool]$NativeTestHooks;buildEvents=$false;exitCode=$process.ExitCode;
    elapsedSeconds=((Get-Date)-$started).TotalSeconds;log=$log;arguments=$argsList}
if (Test-Path -LiteralPath $dll) { $receipt.dllSha256=(Get-FileHash -LiteralPath $dll).Hash; $receipt.dllBytes=(Get-Item $dll).Length }
$receipt | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $reportDir 'frontend-build.json')
Get-Content -LiteralPath $log -Tail 12
if ($process.ExitCode) { throw "Base build failed: $($process.ExitCode)" }
Write-Output $(if($NativeTestHooks){'PASS RDNA2 diagnostic source build'}else{'PASS RDNA2 product source build'})
