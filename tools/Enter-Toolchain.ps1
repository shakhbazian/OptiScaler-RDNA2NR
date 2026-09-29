param(
    [string]$HipRoot = 'C:\Program Files\AMD\ROCm\6.4',
    [string]$VsRoot = 'C:\Program Files\Microsoft Visual Studio\18\Community',
    [string]$PlatformToolset = 'v145',
    [string]$MsvcToolsVersion = '14.44'
)
$ErrorActionPreference = 'Stop'
# Large qualification batches dot-source this file many times in one process.
# Re-entering VsDevCmd grows its inherited INCLUDE/LIB bookkeeping until cmd's
# line limit is reached. Reuse only our fully initialized, matching selection.
$toolchainSignature = "$VsRoot|$HipRoot|$PlatformToolset|$MsvcToolsVersion"
if ($env:RDNA2_TOOLCHAIN_SIGNATURE -eq $toolchainSignature -and $env:INCLUDE -and $env:LIB -and
    $env:HIP_PATH -eq $HipRoot -and (!$MsvcToolsVersion -or $env:VCTOOLSVERSION -like "$MsvcToolsVersion*")) {
    $selectedCompiler = Get-Command cl.exe -ErrorAction SilentlyContinue
    $expectedCompiler = Join-Path $env:VCTOOLSINSTALLDIR 'bin/Hostx64/x64/cl.exe'
    if ($selectedCompiler -and [IO.Path]::GetFullPath($selectedCompiler.Source) -eq
        [IO.Path]::GetFullPath($expectedCompiler) -and (Test-Path -LiteralPath "$HipRoot/bin/clang++.exe")) {
        return
    }
}
$baseProcessPath = if ($env:RDNA2_BASE_PROCESS_PATH) {
    $env:RDNA2_BASE_PROCESS_PATH
} else {
    $env:PATH
}
[Environment]::SetEnvironmentVariable('RDNA2_BASE_PROCESS_PATH', $baseProcessPath, 'Process')
$devCmd = Join-Path $VsRoot 'Common7\Tools\VsDevCmd.bat'
if (!(Test-Path -LiteralPath $devCmd)) { throw "Missing VS developer environment: $devCmd" }
# Import only this process's environment. No machine-wide PATH changes.
$toolsetRoot = Join-Path $VsRoot "MSBuild\Microsoft\VC\v180\Platforms\x64\PlatformToolsets\$PlatformToolset"
if (!(Test-Path -LiteralPath $toolsetRoot)) { throw "Missing MSBuild platform toolset: $toolsetRoot" }
$vcvarsVersionArgument = if ($MsvcToolsVersion) { " -vcvars_ver=$MsvcToolsVersion" } else { '' }
$envLines = & $env:ComSpec /d /c "call `"$devCmd`" -no_logo -arch=x64 -host_arch=x64$vcvarsVersionArgument >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'VS environment initialization failed' }
$importedEnvironment = @{}
foreach ($line in $envLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        $varName = $Matches[1]
        $varValue = $Matches[2]
        $canonicalName = $varName.ToUpperInvariant()
        # cmd.exe can expose both Path and PATH. Prefer its canonical uppercase
        # value because VsDevCmd updates that entry on this host.
        if (!$importedEnvironment.ContainsKey($canonicalName) -or $varName -ceq $canonicalName) {
            $importedEnvironment[$canonicalName] = $varValue
        }
    }
}
foreach ($entry in $importedEnvironment.GetEnumerator()) {
    $variants = @([Environment]::GetEnvironmentVariables('Process').Keys | Where-Object {
        [string]::Equals([string]$_, $entry.Key, [StringComparison]::OrdinalIgnoreCase)
    })
    foreach ($variant in $variants) {
        [Environment]::SetEnvironmentVariable([string]$variant, $null, 'Process')
    }
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
[Environment]::SetEnvironmentVariable('HIP_PATH', $HipRoot, 'Process')
[Environment]::SetEnvironmentVariable('HIP_CLANG_PATH', (Join-Path $HipRoot 'bin'), 'Process')
$normalizedPath = "$HipRoot\bin;$env:VCTOOLSINSTALLDIR\bin\Hostx64\x64;$VsRoot\MSBuild\Current\Bin;$env:WINDOWSSDKVERBINPATH\x64;C:\Program Files\Git\cmd;C:\Program Files\Git\usr\bin;C:\Program Files\Git\mingw64\bin;$baseProcessPath"
[Environment]::SetEnvironmentVariable('PATH', $normalizedPath, 'Process')
if (!(Test-Path -LiteralPath "$HipRoot\bin\clang++.exe")) { throw 'HIP compiler missing' }
$cl = Get-Command cl.exe -ErrorAction Stop
$clVersion = $cl.FileVersionInfo.ProductVersion
if ($MsvcToolsVersion -and $env:VCTOOLSVERSION -notlike "$MsvcToolsVersion*") {
    throw "MSVC $MsvcToolsVersion requested, but developer environment selected $env:VCTOOLSVERSION"
}
$env:RDNA2_PLATFORM_TOOLSET = $PlatformToolset
$env:RDNA2_REQUESTED_MSVC_VERSION = $MsvcToolsVersion
$env:RDNA2_MSVC_VERSION = $clVersion
$env:RDNA2_TOOLCHAIN_SIGNATURE = $toolchainSignature
