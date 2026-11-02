param([Parameter(Mandatory)][string]$CorpusRoot,
      [Parameter(Mandatory)][string]$PackageRoot,
      [ValidateSet('a','b','release','baseline','release-baseline')][string[]]$Models=@('a','b'),
      [switch]$SaveAllFrames, [switch]$ImagesOnly)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. "$root/tools/Enter-Toolchain.ps1" -PlatformToolset v145 -MsvcToolsVersion 14.44
$source='2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F'
foreach($variant in $Models){
    $model=if($variant -eq 'b'){'b'}else{'a'}
    $hash=if($model -eq 'b'){'2CE6F8BB07F18399A58120F9FB2F1152DD39D6B872370DA7EE61B104E4165AE7'}else{'5C2B4272542C4CAF3B2E16A1242F21484B3BE2F8DC31337ECCCEA93143BFAE2C'}
    $executable=switch($variant){
        'release' {'model_ab_mixed.exe'}
        'baseline' {'model_ab_fp16_baseline.exe'}
        'release-baseline' {'model_ab_mixed_baseline.exe'}
        default {'model_ab_fp16.exe'}
    }
    $destination=Join-Path $CorpusRoot $variant
    New-Item -ItemType Directory -Force $destination | Out-Null
    $arguments=@("$PackageRoot/model-$model.nrwgt",$hash,$source,"$CorpusRoot/$model.bin",$destination)
    if(!$SaveAllFrames){$arguments+='--first-frame-only'}
    if($ImagesOnly){$arguments+='--images-only'}
    & "$root/build/model-ab/$executable" @arguments 2>&1 | Tee-Object "$destination/run.log"
    if($LASTEXITCODE -ne 0){throw "Model $variant failed; see $destination/run.log"}
}
