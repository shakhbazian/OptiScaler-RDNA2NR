# Build-time only: copy a licensed CPython + NumPy installation into the release.
param([Parameter(Mandatory)][string]$PythonHome,
      [Parameter(Mandatory)][string]$Destination)
$ErrorActionPreference='Stop'
$homePath=[IO.Path]::GetFullPath($PythonHome)
foreach($file in @('python.exe','python312.dll','LICENSE.txt')){
    if(-not(Test-Path -LiteralPath (Join-Path $homePath $file) -PathType Leaf)){
        throw "Python runtime is missing $file"
    }
}
$site=Join-Path $homePath 'Lib/site-packages'
$numpy=Join-Path $site 'numpy'
$numpyLibs=Join-Path $site 'numpy.libs'
$numpyInfo=Get-ChildItem -LiteralPath $site -Directory -Filter 'numpy-*.dist-info' | Select-Object -First 1
if(-not(Test-Path -LiteralPath $numpy -PathType Container) -or
   -not(Test-Path -LiteralPath $numpyLibs -PathType Container) -or
   -not $numpyInfo){throw 'NumPy package or distribution metadata is missing.'}
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
foreach($file in @('python.exe','python312.dll','python3.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt')){
    $source=Join-Path $homePath $file
    if(Test-Path -LiteralPath $source -PathType Leaf){Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $file)}
}
$libDest=Join-Path $Destination 'Lib'
New-Item -ItemType Directory -Path $libDest -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $homePath 'Lib') | Where-Object Name -ne 'site-packages' |
    ForEach-Object {Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $libDest $_.Name) -Recurse}
Copy-Item -LiteralPath (Join-Path $homePath 'DLLs') -Destination (Join-Path $Destination 'DLLs') -Recurse
$siteDest=Join-Path $libDest 'site-packages'
New-Item -ItemType Directory -Path $siteDest -Force | Out-Null
foreach($source in @($numpy,$numpyLibs,$numpyInfo.FullName)){
    Copy-Item -LiteralPath $source -Destination (Join-Path $siteDest (Split-Path -Leaf $source)) -Recurse
}
$python=Join-Path $Destination 'python.exe'
& $python -I -c 'import sys,numpy; assert sys.version_info[:2] == (3,12); print("Portable Python",sys.version.split()[0],"NumPy",numpy.__version__)'
if($LASTEXITCODE -ne 0){throw 'Portable Python self-test failed.'}
