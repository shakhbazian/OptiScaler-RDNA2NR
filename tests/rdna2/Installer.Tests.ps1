# Data-only installer regression tests. No game, GPU or vendor executable is run.
param([string]$OutputRoot)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $OutputRoot) { $OutputRoot = Join-Path ([IO.Path]::GetTempPath()) ('rdna2nr-test-' + [guid]::NewGuid().ToString('N').Substring(0,8)) }
$work = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $work) { throw 'Choose a fresh test output directory.' }
New-Item -ItemType Directory -Path $work | Out-Null

# Load definitions without launching the GUI or executing the CLI entry point.
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'Install-RDNA2NR.ps1'),
                                                         [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
foreach ($statement in $ast.EndBlock.Statements) {
    if ($statement -is [Management.Automation.Language.FunctionDefinitionAst] -or
        $statement -is [Management.Automation.Language.AssignmentStatementAst]) {
        . ([scriptblock]::Create($statement.Extent.Text))
    }
}
function Write-Data([string]$Path, [string]$Text) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}
function Assert([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Expect-Failure([scriptblock]$Action, [string]$Pattern) {
    $caught = $null
    try { & $Action | Out-Null } catch { $caught = $_.Exception.Message }
    Assert ($null -ne $caught -and $caught -match $Pattern) "Expected failure: $Pattern; received: $caught"
}
function Assert-Setting([string]$Section, [string]$Key, [string]$Value) {
    Assert ((Get-IniValue (Join-Path $game 'OptiScaler.ini') $Section $Key) -eq $Value) "Wrong setting: [$Section] $Key"
}

$release = Join-Path $work 'release'; $game = Join-Path $work 'game'
$models = Join-Path $work 'models'; $source = Join-Path $work 'source.dll'
$fsr = Join-Path $work 'download/fsr.dll'
Write-Data $source 'test NR source, not executable'
Write-Data $fsr 'test INT8 runtime, not executable'
# Substitute tiny fixture hashes in the isolated test scope. Production checks
# and their real hashes are never changed in the installer or source checkout.
$SourceHash = Get-Hash $source
$Fsr4Int8Hash = Get-Hash $fsr
$ModelRelative = "1\$SourceHash\mixed-v5-gfx1030.nrwgt"
$model = Join-Path $models $ModelRelative
Write-Data $model 'test prepared model'
$PackageHash = Get-Hash $model; $PackageBytes = (Get-Item -LiteralPath $model).Length
Write-Data (Join-Path $release 'VERSION.txt') 'candidate-test'
Write-Data (Join-Path $release 'OptiScaler.dll') 'test frontend v1'
foreach ($relative in $Payload) { Write-Data (Join-Path $release $relative) "bundled fixture $relative" }
Copy-Item -LiteralPath (Join-Path $root 'OptiScaler.ini') -Destination (Join-Path $release 'OptiScaler.ini') -Force
Write-Data (Join-Path $game 'fixture.exe') 'not executable'
$slot = Join-Path $game 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll'
Write-Data $slot 'original game runtime'
$originalHash = Get-Hash $slot
Write-Data (Join-Path $game 'OptiScaler.ini') @'
; user settings must survive the update
[FrameGen]
FGInput=Upscaler
FGOutput=XeFG
Enabled=true
DebugView=true
[DlssNr]
Enabled=false
WorkingScale=0.65
Preset=2
DebugView=3
[FSR]
UpscalerIndex=auto
Fsr4EnableWatermark=true
DebugView=true
[FSRFG]
EnableWatermark=true
[XeFG]
DebugView=true
[Log]
LogToFile=true
LogLevel=0
OpenConsole=true
'@
# OptiScaler's saved INI uses spaces around '=', unlike the shipped template.
$savedIni = Join-Path $game 'OptiScaler.ini'
$savedText = [regex]::Replace([IO.File]::ReadAllText($savedIni), '(?m)^([A-Za-z0-9]+)=', '$1 = ')
[IO.File]::WriteAllText($savedIni, $savedText, [Text.UTF8Encoding]::new($true))
Install-Product $game $source 'dxgi.dll' $release 'unused' $models $fsr | Out-Host
Assert ((Get-Hash $slot) -eq $Fsr4Int8Hash) 'Selected FSR DLL was not copied.'
Assert-Setting 'Libraries' 'FfxDx12SRPath' 'auto'
Assert-Setting 'FSR' 'UpscalerIndex' '0'
Assert-Setting 'FrameGen' 'FGInput' 'Upscaler'
Assert-Setting 'FrameGen' 'Enabled' 'true'
Assert-Setting 'DlssNr' 'WorkingScale' '0.65'
Assert-Setting 'DlssNr' 'Preset' '2'
Assert-Setting 'DlssNr' 'Enabled' 'false'
Assert-Setting 'DlssNr' 'DebugView' '0'
foreach ($pair in @(@('FrameGen','DebugView'),@('FSR','DebugView'),@('FSR','Fsr4EnableWatermark'),
                    @('FSRFG','EnableWatermark'),@('XeFG','DebugView'),@('Log','OpenConsole'))) {
    Assert-Setting $pair[0] $pair[1] 'false'
}
Assert-Setting 'Log' 'LogLevel' '2'
Assert-Setting 'Log' 'LogToFile' 'true'
Remove-Item -LiteralPath $fsr
Verify-Product $game | Out-Host
Set-IniValues (Join-Path $game 'OptiScaler.ini') (,@('FSR','UpscalerIndex','1'))
Install-Product $game $source 'dxgi.dll' $release 'unused' $models | Out-Host
Assert ((Get-Hash $slot) -eq $Fsr4Int8Hash) 'Empty optional field replaced an installed INT8 runtime.'
Assert-Setting 'FSR' 'UpscalerIndex' '1'

# Old external-path installs migrate to the owned slot and survive source removal.
Write-Data $fsr 'test INT8 runtime, not executable'
Set-IniValues (Join-Path $game 'OptiScaler.ini') (,@('Libraries','FfxDx12SRPath',$fsr))
Install-Product $game $source 'dxgi.dll' $release 'unused' $models | Out-Host
Remove-Item -LiteralPath $fsr
Assert-Setting 'Libraries' 'FfxDx12SRPath' 'auto'
Assert-Setting 'FSR' 'UpscalerIndex' '1'
Verify-Product $game | Out-Host

$proxy = Join-Path $game 'dxgi.dll'; $manifest = Join-Path $game $ManifestName
$before = Get-Hash $proxy; $manifestBefore = Get-Hash $manifest
$invalid = Join-Path $work 'wrong.dll'; Write-Data $invalid 'unsupported'
Expect-Failure { Install-Product $game $source 'dxgi.dll' $release 'unused' $models $invalid } 'SHA-256'
Assert ((Get-Hash $proxy) -eq $before -and (Get-Hash $manifest) -eq $manifestBefore) 'Rejected source changed the install.'

# Fail after copying the new proxy; rollback must restore its exact old bytes.
Write-Data (Join-Path $release 'OptiScaler.dll') 'test frontend v2'
$ini = Join-Path $game 'OptiScaler.ini'
[IO.File]::AppendAllText($ini, "`n[Log]`nLogLevel=0`n")
$iniBefore = Get-Hash $ini
Expect-Failure { Install-Product $game $source 'dxgi.dll' $release 'unused' $models } 'Duplicate INI'
Assert ((Get-Hash $proxy) -eq $before -and (Get-Hash $ini) -eq $iniBefore -and
        (Get-Hash $manifest) -eq $manifestBefore) 'Rollback changed the old installation.'
$text = [IO.File]::ReadAllText($ini).Replace("`n[Log]`nLogLevel=0`n", '')
[IO.File]::WriteAllText($ini, $text, [Text.UTF8Encoding]::new($false))
Write-Data $proxy 'manual replacement'
Expect-Failure { Install-Product $game $source 'dxgi.dll' $release 'unused' $models } 'Installed files differ'
Install-Product $game $source 'dxgi.dll' $release 'unused' $models '' $true | Out-Host
$record = Read-Manifest $game
$retained = @($record.retainedChangedFiles | Where-Object relative -eq 'dxgi.dll')
Assert ($retained.Count -eq 1 -and (Get-Hash (Join-Path $game $retained[0].backup)) -eq
        $retained[0].sha256) 'Manual replacement backup is missing.'
Verify-Product $game | Out-Host
Uninstall-Product $game | Out-Host
Assert ((Get-Hash $slot) -eq $originalHash) 'Uninstall did not restore the original runtime.'
Assert (Test-Path -LiteralPath $ini) 'Uninstall deleted the user INI.'

# Fresh installation without an optional runtime uses the bundled slot.
$game = Join-Path $work 'fresh-game'
Write-Data (Join-Path $game 'fixture.exe') 'not executable'
Install-Product $game $source 'dxgi.dll' $release 'unused' $models | Out-Host
Assert-Setting 'FrameGen' 'FGInput' 'FSRFG30'
Assert-Setting 'FrameGen' 'FGOutput' 'XeFG'
Assert-Setting 'DlssNr' 'Enabled' 'true'
Assert-Setting 'DlssNr' 'RunBeforeSR' 'true'
Assert-Setting 'DlssNr' 'WorkingScale' '0.75'
Assert ((Get-Hash (Join-Path $game 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll')) -eq
        (Get-Hash (Join-Path $release 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll'))) 'Bundled runtime differs.'
Verify-Product $game | Out-Host
Write-Output 'PASS installer: copying, migration, settings cleanup, update, rollback, verification and uninstall.'
