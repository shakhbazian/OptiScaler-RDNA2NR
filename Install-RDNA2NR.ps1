# Installs the OptiScaler frontend and converts a user-provided NR DLL locally.
# The NVIDIA source is read as data; an optional FSR runtime is copied separately.
[CmdletBinding()]
param(
    [ValidateSet('Install','Verify','Uninstall')][string]$Mode = 'Install',
    [string]$GameDirectory,
    [string]$SourceDll,
    [ValidateSet('dxgi.dll','winmm.dll','version.dll','dbghelp.dll','d3d12.dll','wininet.dll','winhttp.dll')]
    [string]$ProxyName = 'dxgi.dll',
    [string]$PythonPath,
    [string]$ModelRoot,
    [string]$ReleaseRoot,
    [string]$Fsr4Int8Dll,
    [switch]$ReplaceChangedFiles,
    [switch]$Gui
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $ReleaseRoot = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($MyInvocation.MyCommand.Path))
}
$SourceHash = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
$PackageHash = 'A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB'
$PackageBytes = 291595458
$Fsr4Int8Hash = 'C7720BC16BEDE334F59A1A32CD22EDBCBBB159685ED5240E61350A5FB0BC8A94'
$ModelRelative = "1\$SourceHash\mixed-v5-gfx1030.nrwgt"
$ManifestName = '.optiscaler-rdna2nr-install.json'
$BackupName = '.optiscaler-rdna2nr-backup'
$Payload = @(
    'dlssnr_hip_scheduled_bridge.dll', 'OptiScaler.ini',
    'OptiScaler/libxess.dll', 'OptiScaler/libxess_dx11.dll',
    'OptiScaler/libxell.dll', 'OptiScaler/libxess_fg.dll',
    'OptiScaler/amd_fidelityfx_vk.dll',
    'OptiScaler/amd_fidelityfx_loader_dx12.dll',
    'OptiScaler/amd_fidelityfx_upscaler_dx12.dll',
    'OptiScaler/amd_fidelityfx_framegeneration_dx12.dll',
    'OptiScaler/D3D12_OptiScaler/D3D12Core.dll'
)

function Get-Hash([string]$Path) {
    $stream = [IO.File]::OpenRead([IO.Path]::GetFullPath($Path))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '')
    } finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}
function Get-ReleaseVersion([string]$Root) {
    $path = Join-Path $Root 'VERSION.txt'
    # Older packages have no version file; their install/update path still works.
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        return [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8).Trim()
    }
    return $null
}
function Assert-Within([string]$Root, [string]$Path) {
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $pathFull = [IO.Path]::GetFullPath($Path)
    if (-not $pathFull.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the selected game directory: $pathFull"
    }
}
function Read-Manifest([string]$Game) {
    $path = Join-Path $Game $ManifestName
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    $record = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8) | ConvertFrom-Json
    if ($record.schemaVersion -ne 1 -or $record.product -ne 'OptiScaler-RDNA2NR') {
        throw 'Unknown installation manifest; refusing to modify it.'
    }
    $allowedProxies = @('dxgi.dll','winmm.dll','version.dll','dbghelp.dll','d3d12.dll','wininet.dll','winhttp.dll')
    if ($allowedProxies -notcontains $record.proxyName -or $record.files.Count -ne ($Payload.Count + 1)) {
        throw 'Installation manifest has an unsupported file list.'
    }
    $expectedFiles = @($record.proxyName) + $Payload
    $actualFiles = @($record.files | ForEach-Object { $_.relative })
    if (@(Compare-Object $expectedFiles $actualFiles).Count -ne 0 -or
        @($actualFiles | Select-Object -Unique).Count -ne $expectedFiles.Count) {
        throw 'Installation manifest file list differs from this installer.'
    }
    foreach ($file in $record.files) {
        $target = Join-Path $Game $file.relative
        Assert-Within $Game $target
        if ($file.backup) {
            $backup = Join-Path $Game $file.backup
            Assert-Within (Join-Path $Game $BackupName) $backup
        }
    }
    return $record
}
function Get-ChangedInstallFiles([string]$Game, $Record) {
    if (-not $Record) { return }
    foreach ($item in $Record.files) {
        if ($item.relative -eq 'OptiScaler.ini') { continue }
        $path = Join-Path $Game $item.relative
        $hash = if (Test-Path -LiteralPath $path -PathType Leaf) { Get-Hash $path } else { $null }
        if ($hash -ne $item.sha256) {
            $version = if ($hash) { (Get-Item -LiteralPath $path).VersionInfo.ProductVersion } else { $null }
            [pscustomobject]@{ relative = $item.relative; sha256 = $hash; version = $version }
        }
    }
}
function Find-Python([string]$Explicit, [string]$Root) {
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit -PathType Leaf)) { throw "Python not found: $Explicit" }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $bundled = Join-Path $Root 'tools/python/python.exe'
    if (Test-Path -LiteralPath $bundled -PathType Leaf) { return $bundled }
    foreach ($command in @('python.exe', 'python3.exe')) {
        $found = Get-Command $command -ErrorAction SilentlyContinue
        if ($found -and $found.Source) { return $found.Source }
    }
    throw 'Python 3.10+ with NumPy is required for model conversion. Select python.exe with -PythonPath.'
}
function Resolve-ModelRoot([string]$Override) {
    if ($Override) { return [IO.Path]::GetFullPath($Override) }
    if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is unavailable.' }
    return (Join-Path $env:LOCALAPPDATA 'OptiScaler-RDNA2NR/models')
}
function Prepare-Model([string]$Dll, [string]$Root, [string]$Python, [string]$Models) {
    if (-not (Test-Path -LiteralPath $Dll -PathType Leaf)) { throw "Source DLL not found: $Dll" }
    if ((Get-Hash $Dll) -ne $SourceHash) {
        throw 'The selected DLL is not the supported original NR version (SHA-256 mismatch).'
    }
    $target = Join-Path $Models $ModelRelative
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $existing = Get-Item -LiteralPath $target
        if ($existing.Length -eq $PackageBytes -and (Get-Hash $target) -eq $PackageHash) {
            Write-Host "Model already verified: $target"
            return $target
        }
        throw "The model cache contains a different file: $target"
    }
    $converter = Join-Path $Root 'tools/model_converter/write_runtime_package.py'
    if (-not (Test-Path -LiteralPath $converter -PathType Leaf)) {
        throw "Converter is missing: $converter"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    $temporary = Join-Path (Split-Path -Parent $target) "nr-$([guid]::NewGuid().ToString('N')).pending"
    try {
        & $Python $converter --source $Dll --output $temporary | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Model conversion failed (exit $LASTEXITCODE)." }
        if ((Get-Item -LiteralPath $temporary).Length -ne $PackageBytes -or
            (Get-Hash $temporary) -ne $PackageHash) { throw 'Converted package verification failed.' }
        Move-Item -LiteralPath $temporary -Destination $target
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
    Write-Host "Model prepared: $target"
    return $target
}
function Get-InstallFiles([string]$Root, [string]$Proxy) {
    $result = @([pscustomobject]@{ source = (Join-Path $Root 'OptiScaler.dll'); relative = $Proxy })
    foreach ($relative in $Payload) {
        $result += [pscustomobject]@{ source = (Join-Path $Root $relative); relative = $relative }
    }
    foreach ($file in $result) {
        if (-not (Test-Path -LiteralPath $file.source -PathType Leaf)) {
            throw "Release payload is missing: $($file.source)"
        }
    }
    return $result
}
function Resolve-Fsr4Int8([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "FSR 4 INT8 DLL not found: $Path" }
    if ((Get-Hash $Path) -ne $Fsr4Int8Hash) {
        throw 'Select the supported FSR 4.0.2c INT8 DLL. Its SHA-256 differs from the verified build.'
    }
    return (Resolve-Path -LiteralPath $Path).Path
}
function Get-IniValue([string]$Ini, [string]$Section, [string]$Key) {
    if (-not (Test-Path -LiteralPath $Ini -PathType Leaf)) { return $null }
    $inSection = $false
    $values = @()
    foreach ($line in [IO.File]::ReadAllLines($Ini, [Text.Encoding]::UTF8)) {
        if ($line -match '^\s*\[([^]]+)\]\s*$') { $inSection = $Matches[1] -eq $Section }
        elseif ($inSection -and $line -match ('^\s*' + [regex]::Escape($Key) + '\s*=(.*)$')) {
            $values += $Matches[1].Trim()
        }
    }
    if ($values.Count -gt 1) { throw "Duplicate INI setting: [$Section] $Key" }
    if ($values.Count) { return $values[0] }
    return $null
}
function Resolve-InstalledFsr4([string]$Game, [string]$Explicit) {
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) { return (Resolve-Fsr4Int8 $Explicit) }
    $saved = Get-IniValue (Join-Path $Game 'OptiScaler.ini') 'Libraries' 'FfxDx12SRPath'
    if ($saved -and $saved -ne 'auto') {
        $path = if ([IO.Path]::IsPathRooted($saved)) { $saved } else { Join-Path $Game $saved }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "The configured FSR runtime is missing: $path. Select your FSR 4 DLL, or correct [Libraries] FfxDx12SRPath before updating."
        }
        # Import the supported external runtime from older test installations.
        # Other custom libraries and their settings remain user-owned.
        if ((Get-Hash $path) -eq $Fsr4Int8Hash) { return [IO.Path]::GetFullPath($path) }
        return $null
    }
    $managed = Join-Path $Game 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll'
    if ((Test-Path -LiteralPath $managed -PathType Leaf) -and (Get-Hash $managed) -eq $Fsr4Int8Hash) {
        return $managed
    }
    return $null
}
function Set-IniValues([string]$Ini, [object[]]$Settings) {
    $lines = [Collections.Generic.List[string]]::new()
    $lines.AddRange([string[]][IO.File]::ReadAllLines($Ini, [Text.Encoding]::UTF8))
    # Update only named settings, retaining comments and unrelated user choices.
    foreach ($setting in $Settings) {
        if ($setting -is [string] -or $setting.Count -ne 3) { throw 'Invalid INI update tuple.' }
        $section = $setting[0]; $key = $setting[1]; $value = $setting[2]
        $inSection = $false; $insert = -1; $keyIndices = @()
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^\s*\[([^]]+)\]\s*$') {
                if ($inSection) { $insert = $i }
                $inSection = $Matches[1] -eq $section
                if ($inSection) { $insert = $i + 1 }
            } elseif ($inSection -and $lines[$i] -match ('^\s*' + [regex]::Escape($key) + '\s*=')) {
                $keyIndices += $i
            }
        }
        if ($keyIndices.Count -gt 1) { throw "Duplicate INI setting: [$section] $key" }
        if ($keyIndices.Count -eq 1) { $lines[$keyIndices[0]] = "$key=$value" }
        elseif ($insert -ge 0) { $lines.Insert($insert, "$key=$value") }
        else { $lines.Add("`r`n[$section]"); $lines.Add("$key=$value") }
    }
    [IO.File]::WriteAllLines($Ini, $lines, [Text.UTF8Encoding]::new($true))
}
function Install-Product([string]$Game, [string]$Dll, [string]$Proxy,
                         [string]$Root, [string]$Python, [string]$Models, [string]$Fsr4Dll = '',
                         [bool]$AllowChangedFiles = $false) {
    $gameFull = [IO.Path]::GetFullPath($Game)
    if (-not (Test-Path -LiteralPath $gameFull -PathType Container)) { throw 'Game directory does not exist.' }
    if (-not @(Get-ChildItem -LiteralPath $gameFull -File -Filter '*.exe').Count) {
        throw 'Select the folder containing the game executable.'
    }
    $rootFull = [IO.Path]::GetFullPath($Root)
    $selectFsr4 = -not [string]::IsNullOrWhiteSpace($Fsr4Dll)
    $fsr4Path = Resolve-InstalledFsr4 $gameFull $Fsr4Dll
    $releaseVersion = Get-ReleaseVersion $rootFull
    $files = @(Get-InstallFiles $rootFull $Proxy)
    if ($fsr4Path) {
        ($files | Where-Object relative -eq 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll').source = $fsr4Path
    }
    $old = Read-Manifest $gameFull
    if ($old -and $old.proxyName -ne $Proxy) {
        throw "This game already uses $($old.proxyName). Uninstall before changing the proxy name."
    }
    $changed = @(Get-ChangedInstallFiles $gameFull $old)
    if ($changed.Count -and -not $AllowChangedFiles) {
        $paths = ($changed | ForEach-Object { Join-Path $gameFull $_.relative }) -join "`n"
        throw "Installed files differ from the previous installation:`n$paths`nUse the GUI to confirm replacement, or pass -ReplaceChangedFiles to retain copies and update."
    }
    $model = Prepare-Model $Dll $rootFull $Python $Models
    $installId = [guid]::NewGuid().ToString('N')
    $backupRoot = Join-Path $gameFull $BackupName
    $transaction = Join-Path $backupRoot "transaction-$installId"
    $baseline = Join-Path $backupRoot "original-$installId"
    $changedRoot = Join-Path $backupRoot "changed-$installId"
    Assert-Within $gameFull $transaction
    Assert-Within $gameFull $baseline
    Assert-Within $backupRoot $changedRoot
    $manifestPath = Join-Path $gameFull $ManifestName
    $receipt = @()
    $touched = @()
    $originals = @{}
    $oldEntries = @{}
    $changedEntries = @{}
    $retained = @()
    foreach ($item in $changed) { $changedEntries[$item.relative] = $item }
    if ($old) { foreach ($item in $old.files) {
        $originals[$item.relative] = $item.backup
        $oldEntries[$item.relative] = $item
    } }
    try {
        foreach ($file in $files) {
            $destination = Join-Path $gameFull $file.relative
            Assert-Within $gameFull $destination
            if ($oldEntries.ContainsKey($file.relative) -and $file.relative -ne 'OptiScaler.ini') {
                $expected = if ($changedEntries.ContainsKey($file.relative)) {
                    $changedEntries[$file.relative].sha256
                } else { $oldEntries[$file.relative].sha256 }
                $actual = if (Test-Path -LiteralPath $destination -PathType Leaf) { Get-Hash $destination } else { $null }
                if ($actual -ne $expected) { throw "File changed during update; close the game and retry: $destination" }
            }
            if ($changedEntries.ContainsKey($file.relative) -and $changedEntries[$file.relative].sha256) {
                # Keep manual replacements separately from the original uninstall backup.
                $saved = Join-Path $changedRoot $file.relative
                Assert-Within $changedRoot $saved
                New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
                Copy-Item -LiteralPath $destination -Destination $saved
                if ((Get-Hash $saved) -ne $changedEntries[$file.relative].sha256) {
                    throw "Changed-file backup verification failed: $saved"
                }
                $retained += [pscustomobject]@{ relative = $file.relative;
                    sha256 = $changedEntries[$file.relative].sha256;
                    backup = $saved.Substring($gameFull.TrimEnd('\').Length).TrimStart('\') }
            }
            if ($file.relative -eq 'OptiScaler.ini' -and (Test-Path -LiteralPath $destination)) {
                # The INI belongs to the user, including on the first install.
                $source = $destination
            } else { $source = $file.source }
            $entry = [ordered]@{ relative = $file.relative; sha256 = (Get-Hash $source);
                backup = $null; userOwned = $false }
            if ($originals.ContainsKey($file.relative)) { $entry.backup = $originals[$file.relative] }
            elseif (Test-Path -LiteralPath $destination) {
                $backup = Join-Path $baseline $file.relative
                Assert-Within $gameFull $backup
                New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
                Copy-Item -LiteralPath $destination -Destination $backup
                $entry.backup = $backup.Substring($gameFull.TrimEnd('\').Length).TrimStart('\')
            }
            if ($file.relative -eq 'OptiScaler.ini' -and $oldEntries.ContainsKey($file.relative)) {
                $previous = $oldEntries[$file.relative]
                $entry.userOwned = [bool]$previous.userOwned -or $entry.sha256 -ne $previous.sha256
            }
            if (Test-Path -LiteralPath $destination) {
                $preimage = Join-Path $transaction $file.relative
                Assert-Within $gameFull $preimage
                New-Item -ItemType Directory -Path (Split-Path -Parent $preimage) -Force | Out-Null
                Copy-Item -LiteralPath $destination -Destination $preimage
            }
            $touched += $file.relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            if ($source -ne $destination) { Copy-Item -LiteralPath $source -Destination $destination -Force }
            if ($file.relative -eq 'OptiScaler.ini') {
                # Retire diagnostic settings used by private test builds on update.
                $settings = @(
                    @('FrameGen', 'DebugView', 'false'),
                    @('FSRFG', 'EnableWatermark', 'false'),
                    @('XeFG', 'DebugView', 'false'),
                    @('FSR', 'DebugView', 'false'),
                    @('FSR', 'Fsr4EnableWatermark', 'false'),
                    @('DLSSNR', 'DebugView', '0'),
                    @('Log', 'LogLevel', '2'),
                    @('Log', 'LogToConsole', 'false'),
                    @('Log', 'LogToDebug', 'false'),
                    @('Log', 'LogToNGX', 'false'),
                    @('Log', 'OpenConsole', 'false')
                )
                if ($fsr4Path) {
                    $settings += ,@('Libraries', 'FfxDx12SRPath', 'auto')
                }
                if ($selectFsr4) {
                    $settings += @(
                        @('Upscalers', 'Dx12Upscaler', 'ffx'),
                        @('Upscalers', 'Dx11Upscaler', 'ffx_12'),
                        @('FSR', 'UpscalerIndex', '0'),
                        @('FSR', 'Fsr4ForceModel', 'auto')
                    )
                }
                Set-IniValues $destination $settings
                $entry.sha256 = Get-Hash $destination
            }
            if ((Get-Hash $destination) -ne $entry.sha256) { throw "Installed file verification failed: $destination" }
            $receipt += [pscustomobject]$entry
        }
        $record = [ordered]@{ schemaVersion = 1; product = 'OptiScaler-RDNA2NR';
            releaseVersion = $releaseVersion;
            installId = $installId; proxyName = $Proxy; modelSha256 = $PackageHash;
            modelPath = $model; files = $receipt }
        $retainedHistory = @()
        if ($old -and $old.PSObject.Properties['retainedChangedFiles']) {
            $retainedHistory += @($old.retainedChangedFiles)
        }
        $retainedHistory += $retained
        if ($retainedHistory.Count) { $record.retainedChangedFiles = $retainedHistory }
        $pendingManifest = "$manifestPath.pending"
        [IO.File]::WriteAllText($pendingManifest, ($record | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $pendingManifest -Destination $manifestPath -Force
    } catch {
        foreach ($relative in $touched) {
            $destination = Join-Path $gameFull $relative
            $preimage = Join-Path $transaction $relative
            if (Test-Path -LiteralPath $preimage -PathType Leaf) {
                Copy-Item -LiteralPath $preimage -Destination $destination -Force
            } elseif (Test-Path -LiteralPath $destination) {
                Remove-Item -LiteralPath $destination -Force
            }
        }
        throw
    } finally {
        if (Test-Path -LiteralPath $transaction) { Remove-Item -LiteralPath $transaction -Recurse -Force }
    }
    Write-Output "Installed and verified $($receipt.Count) files in $gameFull"
    if ($releaseVersion) { Write-Output "Release: $releaseVersion" }
    Write-Output "Proxy: $Proxy; model: $model"
    if ($fsr4Path) { Write-Output "FSR 4 INT8 installed: $(Join-Path $gameFull 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll')" }
    if ($retained.Count) { Write-Output "Previous changed files retained in: $changedRoot" }
}
function Verify-Product([string]$Game) {
    $gameFull = [IO.Path]::GetFullPath($Game)
    $record = Read-Manifest $gameFull
    if (-not $record) { throw 'No OptiScaler-RDNA2NR installation manifest found.' }
    foreach ($item in $record.files) {
        $target = Join-Path $gameFull $item.relative
        if ($item.relative -eq 'OptiScaler.ini' -and (Test-Path -LiteralPath $target -PathType Leaf)) { continue }
        if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or
            (Get-Hash $target) -ne $item.sha256) { throw "Installed file differs: $target" }
    }
    if (-not (Test-Path -LiteralPath $record.modelPath -PathType Leaf) -or
        (Get-Item -LiteralPath $record.modelPath).Length -ne $PackageBytes -or
        (Get-Hash $record.modelPath) -ne $PackageHash) { throw 'Installed model cache differs.' }
    Write-Output "Verified $($record.files.Count) files and model package."
    if ($record.releaseVersion) { Write-Output "Release: $($record.releaseVersion)" }
}
function Uninstall-Product([string]$Game) {
    $gameFull = [IO.Path]::GetFullPath($Game)
    $record = Read-Manifest $gameFull
    if (-not $record) { throw 'No OptiScaler-RDNA2NR installation manifest found.' }
    foreach ($item in $record.files) {
        $target = Join-Path $gameFull $item.relative
        if ($item.relative -eq 'OptiScaler.ini' -and (Test-Path -LiteralPath $target -PathType Leaf)) { continue }
        if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or
            (Get-Hash $target) -ne $item.sha256) {
            throw "Installed file differs; uninstall stopped without changes: $target"
        }
        if ($item.backup -and -not (Test-Path -LiteralPath (Join-Path $gameFull $item.backup) -PathType Leaf)) {
            throw "Original backup is missing; uninstall stopped without changes: $($item.backup)"
        }
    }
    foreach ($item in $record.files) {
        $target = Join-Path $gameFull $item.relative
        if ($item.relative -eq 'OptiScaler.ini' -and (Test-Path -LiteralPath $target -PathType Leaf)) {
            if ($item.backup -or $item.userOwned -or (Get-Hash $target) -ne $item.sha256) { continue }
        }
        if ($item.backup) { Copy-Item -LiteralPath (Join-Path $gameFull $item.backup) -Destination $target -Force }
        else { Remove-Item -LiteralPath $target -Force }
    }
    Remove-Item -LiteralPath (Join-Path $gameFull $ManifestName) -Force
    Write-Output 'OptiScaler-RDNA2NR removed; pre-existing files restored. Shared model cache retained.'
}
function Show-Installer {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()
    $form = New-Object System.Windows.Forms.Form
    $form.Text = 'OptiScaler-RDNA2NR setup'
    $releaseVersion = Get-ReleaseVersion $ReleaseRoot
    if ($releaseVersion) { $form.Text += " - $releaseVersion" }
    $form.Size = New-Object System.Drawing.Size(660,420)
    $form.StartPosition = 'CenterScreen'
    $form.MinimumSize = $form.Size
    $labels = @('Game executable folder', 'Your original nvngx_dlssnr.dll', 'Proxy DLL name')
    $values = @($GameDirectory, $SourceDll, $ProxyName)
    $boxes = @()
    for ($i = 0; $i -lt 3; $i++) {
        $label = New-Object System.Windows.Forms.Label
        $label.Text = $labels[$i]; $label.SetBounds(20, (20 + 70*$i), 580, 22)
        $form.Controls.Add($label)
        if ($i -lt 2) {
            $box = New-Object System.Windows.Forms.TextBox
            $box.SetBounds(20, (43 + 70*$i), 535, 25)
            $box.Text = $values[$i]
            $button = New-Object System.Windows.Forms.Button
            $button.Text = 'Browse'; $button.SetBounds(565, (42 + 70*$i), 70, 27)
            $button.Tag = $box
            if ($i -eq 0) {
                $button.Add_Click({ $dialog = New-Object System.Windows.Forms.FolderBrowserDialog;
                    if ($dialog.ShowDialog() -eq 'OK') { $this.Tag.Text = $dialog.SelectedPath } })
            } else {
                $button.Add_Click({ $dialog = New-Object System.Windows.Forms.OpenFileDialog;
                    $dialog.Filter = 'NVIDIA NR DLL|nvngx_dlssnr.dll|DLL files|*.dll';
                    if ($dialog.ShowDialog() -eq 'OK') { $this.Tag.Text = $dialog.FileName } })
            }
            $form.Controls.Add($button)
        } else {
            $box = New-Object System.Windows.Forms.ComboBox
            $box.DropDownStyle = 'DropDownList'
            [void]$box.Items.AddRange(@('dxgi.dll','winmm.dll','version.dll','dbghelp.dll','d3d12.dll','wininet.dll','winhttp.dll'))
            $box.SelectedItem = $ProxyName
            $box.SetBounds(20, (43 + 70*$i), 190, 25)
        }
        $form.Controls.Add($box); $boxes += $box
    }
    $fsrLabel = New-Object System.Windows.Forms.Label
    $fsrLabel.Text = 'FSR 4.0.2c INT8 DLL (optional; copied into the game folder)'
    $fsrLabel.SetBounds(20,230,610,22); $form.Controls.Add($fsrLabel)
    $fsrBox = New-Object System.Windows.Forms.TextBox
    $fsrBox.Text = $Fsr4Int8Dll; $fsrBox.SetBounds(20,253,535,25); $form.Controls.Add($fsrBox)
    $fsrBrowse = New-Object System.Windows.Forms.Button
    $fsrBrowse.Text = 'Browse'; $fsrBrowse.SetBounds(565,252,70,27); $fsrBrowse.Tag = $fsrBox
    $fsrBrowse.Add_Click({ $dialog = New-Object System.Windows.Forms.OpenFileDialog;
        $dialog.Filter = 'FSR 4 INT8 DLL|amd_fidelityfx_upscaler_dx12.dll|DLL files|*.dll';
        if ($dialog.ShowDialog() -eq 'OK') { $this.Tag.Text = $dialog.FileName } })
    $form.Controls.Add($fsrBrowse)
    $status = New-Object System.Windows.Forms.Label
    $status.Text = 'The source DLL stays where it is. Conversion runs locally.'
    $status.AutoEllipsis = $true
    $status.SetBounds(20,304,610,25); $form.Controls.Add($status)
    $ui = [pscustomobject]@{ Game = $boxes[0]; Dll = $boxes[1]; Proxy = $boxes[2];
        Status = $status; Form = $form; Payload = $Payload; Python = $PythonPath;
        Models = $ModelRoot; Release = $ReleaseRoot; Fsr4 = $fsrBox }
    $install = New-Object System.Windows.Forms.Button
    $install.Text = 'Install / update'; $install.SetBounds(20,336,130,30)
    $install.Tag = $ui
    $install.Add_Click({
        try {
            $context = $this.Tag
            $game = $context.Game.Text.Trim()
            $dll = $context.Dll.Text.Trim()
            $proxy = [string]$context.Proxy.SelectedItem
            if ([string]::IsNullOrWhiteSpace($game)) { throw 'Select the folder containing the game executable.' }
            if ([string]::IsNullOrWhiteSpace($dll)) { throw 'Select your original nvngx_dlssnr.dll.' }
            if ([string]::IsNullOrWhiteSpace($proxy)) { throw 'Choose a proxy DLL name.' }
            if (-not (Test-Path -LiteralPath $game -PathType Container)) { throw "Game folder not found: $game" }
            if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw "Source DLL not found: $dll" }
            $existing = @()
            $changed = @(Get-ChangedInstallFiles $game (Read-Manifest $game))
            foreach ($relative in (@($proxy) + $context.Payload)) {
                if (Test-Path -LiteralPath (Join-Path $game $relative) -PathType Leaf) {
                    $existing += $relative
                }
            }
            if ($existing.Count -or $changed.Count) {
                $message = "Existing game files will be backed up before replacement:`n" +
                    ($existing -join "`n") + "`n`nContinue?"
                if ($changed.Count) {
                    $selectedVersion = Get-ReleaseVersion $context.Release
                    $message = "Selected build: $selectedVersion`n`nThese files differ from the previous installation:`n" +
                        (($changed | ForEach-Object {
                            if ($_.version) { "$($_.relative) ($($_.version))" }
                            elseif (-not $_.sha256) { "$($_.relative) (missing)" }
                            else { $_.relative }
                        }) -join "`n") +
                        "`n`nReplace them with this build? Current copies will be retained in the backup folder. Your INI settings will be preserved."
                }
                $answer = [System.Windows.Forms.MessageBox]::Show($message,
                    'OptiScaler-RDNA2NR', 'YesNo', 'Warning')
                if ($answer -ne 'Yes') { return }
            }
            $context.Status.Text = 'Preparing model and installing...'; $context.Form.Refresh()
            $py = Find-Python $context.Python $context.Release
            Install-Product $game $dll $proxy $context.Release $py (Resolve-ModelRoot $context.Models) $context.Fsr4.Text.Trim() ($changed.Count -gt 0) | Out-Null
            Verify-Product $game | Out-Null
            $context.Status.Text = 'Installation verified.'
            [System.Windows.Forms.MessageBox]::Show('Installation verified. Enable NR in the OptiScaler menu.','OptiScaler-RDNA2NR') | Out-Null
        } catch {
            $this.Tag.Status.Text = $_.Exception.Message
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message,'OptiScaler-RDNA2NR',0,16) | Out-Null
        }
    })
    $form.Controls.Add($install)
    $uninstall = New-Object System.Windows.Forms.Button
    $uninstall.Text = 'Uninstall'; $uninstall.SetBounds(165,336,110,30)
    $uninstall.Tag = $ui
    $uninstall.Add_Click({
        $context = $this.Tag
        $game = $context.Game.Text.Trim()
        if ([string]::IsNullOrWhiteSpace($game)) {
            [System.Windows.Forms.MessageBox]::Show('Select the game folder first.','OptiScaler-RDNA2NR',0,16) | Out-Null
            return
        }
        $answer = [System.Windows.Forms.MessageBox]::Show(
            'Remove this installation and restore any original files?',
            'OptiScaler-RDNA2NR', 'YesNo', 'Question')
        if ($answer -ne 'Yes') { return }
        try { Uninstall-Product $game | Out-Null; $context.Status.Text = 'Removed; original files restored.' }
        catch {
            $context.Status.Text = $_.Exception.Message
            [System.Windows.Forms.MessageBox]::Show($_.Exception.Message,'OptiScaler-RDNA2NR',0,16) | Out-Null
        }
    })
    $form.Controls.Add($uninstall)
    [void]$form.ShowDialog()
}

if ($Gui -or ($Mode -eq 'Install' -and (-not $GameDirectory -or -not $SourceDll))) {
    Show-Installer
} elseif ($Mode -eq 'Install') {
    Install-Product $GameDirectory $SourceDll $ProxyName $ReleaseRoot (Find-Python $PythonPath $ReleaseRoot) (Resolve-ModelRoot $ModelRoot) $Fsr4Int8Dll $ReplaceChangedFiles.IsPresent
    Verify-Product $GameDirectory
} elseif ($Mode -eq 'Verify') {
    Verify-Product $GameDirectory
} else {
    Uninstall-Product $GameDirectory
}
