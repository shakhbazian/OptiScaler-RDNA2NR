param([string]$SourceCache = (Join-Path $env:TEMP 'OptiScaler-RDNA2NR/freetype-2.13.3-src'),
      [string]$BuildCache = (Join-Path $env:TEMP 'OptiScaler-RDNA2NR/freetype-md'),
      [string]$VsRoot = 'C:\Program Files\Microsoft Visual Studio\18\Community')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$revision = '42608f77f20749dd6ddc9e0536788eaad70ea4b5'
$output = Join-Path $root 'build/freetype-md/freetype.lib'
. (Join-Path $PSScriptRoot 'Enter-Toolchain.ps1') -VsRoot $VsRoot -PlatformToolset v145 -MsvcToolsVersion 14.44
$vsCmake = Join-Path $VsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
$cmakeExe = if ($cmakeCommand) { $cmakeCommand.Source } elseif (Test-Path -LiteralPath $vsCmake) { $vsCmake } else { throw 'CMake was not found in PATH or the selected Visual Studio installation.' }
function Assert-SharedCrt([string]$library) {
    $directives = (& dumpbin.exe /nologo /directives $library) -join "`n"
    if ($LASTEXITCODE -or $directives -match '/DEFAULTLIB:LIBCMT' -or
        $directives -notmatch '/DEFAULTLIB:MSVCRT') {
        throw "FreeType is not linked against the shared CRT: $library"
    }
}
if (Test-Path -LiteralPath $output -PathType Leaf) {
    Assert-SharedCrt $output
    return
}
if (-not (Test-Path -LiteralPath (Join-Path $SourceCache '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $SourceCache -Parent) | Out-Null
    & git clone --depth 1 --branch VER-2-13-3 https://github.com/freetype/freetype.git $SourceCache
    if ($LASTEXITCODE) { throw 'FreeType source fetch failed' }
}
$actual = (& git -c "safe.directory=$($SourceCache.Replace('\','/'))" -C $SourceCache rev-parse HEAD).Trim()
if ($actual -ne $revision) { throw "FreeType 2.13.3 revision mismatch: $actual" }

# Keep the NMake build in an ASCII temp directory: its response-file handling
# cannot reliably write beneath a Unicode OneDrive checkout on Windows.
& $cmakeExe -S $SourceCache -B $BuildCache -G 'NMake Makefiles' -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DFT_DISABLE_ZLIB=ON -DFT_DISABLE_BZIP2=ON `
    -DFT_DISABLE_PNG=ON -DFT_DISABLE_HARFBUZZ=ON -DFT_DISABLE_BROTLI=ON
if ($LASTEXITCODE) { throw 'FreeType configure failed' }
$buildLog = Join-Path $BuildCache 'build.log'
& $cmakeExe --build $BuildCache --config Release -- /NOLOGO *> $buildLog
if ($LASTEXITCODE) {
    Get-Content -LiteralPath $buildLog -Tail 40
    throw 'FreeType build failed'
}
$built = Join-Path $BuildCache 'freetype.lib'
if (-not (Test-Path -LiteralPath $built -PathType Leaf)) { throw 'FreeType library missing' }
Assert-SharedCrt $built
New-Item -ItemType Directory -Force -Path (Split-Path $output -Parent) | Out-Null
Copy-Item -LiteralPath $built -Destination $output
Write-Output 'PASS FreeType 2.13.3 /MD'
