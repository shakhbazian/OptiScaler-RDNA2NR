# Installs the OptiScaler frontend and converts a user-provided NR DLL locally.
# No vendor DLL is loaded or copied to the game directory.
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
    [switch]$Gui
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $ReleaseRoot = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($MyInvocation.MyCommand.Path))
}
$SourceHash = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
$PackageHash = 'A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB'
$PackageBytes = 291595458
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
function Install-Product([string]$Game, [string]$Dll, [string]$Proxy,
                         [string]$Root, [string]$Python, [string]$Models) {
    $gameFull = [IO.Path]::GetFullPath($Game)
    if (-not (Test-Path -LiteralPath $gameFull -PathType Container)) { throw 'Game directory does not exist.' }
    if (-not @(Get-ChildItem -LiteralPath $gameFull -File -Filter '*.exe').Count) {
        throw 'Select the folder containing the game executable.'
    }
    $rootFull = [IO.Path]::GetFullPath($Root)
    $releaseVersion = Get-ReleaseVersion $rootFull
    $files = @(Get-InstallFiles $rootFull $Proxy)
    $old = Read-Manifest $gameFull
    if ($old -and $old.proxyName -ne $Proxy) {
        throw "This game already uses $($old.proxyName). Uninstall before changing the proxy name."
    }
    if ($old) {
        foreach ($item in $old.files) {
            $installed = Join-Path $gameFull $item.relative
            if ($item.relative -eq 'OptiScaler.ini') { continue }
            if (-not (Test-Path -LiteralPath $installed -PathType Leaf) -or
                (Get-Hash $installed) -ne $item.sha256) {
                throw "An installed file has changed; refusing to overwrite: $installed"
            }
        }
    }
    $model = Prepare-Model $Dll $rootFull $Python $Models
    $installId = [guid]::NewGuid().ToString('N')
    $backupRoot = Join-Path $gameFull $BackupName
    $transaction = Join-Path $backupRoot "transaction-$installId"
    $baseline = Join-Path $backupRoot "original-$installId"
    Assert-Within $gameFull $transaction
    Assert-Within $gameFull $baseline
    $manifestPath = Join-Path $gameFull $ManifestName
    $receipt = @()
    $touched = @()
    $originals = @{}
    $oldEntries = @{}
    if ($old) { foreach ($item in $old.files) {
        $originals[$item.relative] = $item.backup
        $oldEntries[$item.relative] = $item
    } }
    try {
        foreach ($file in $files) {
            $destination = Join-Path $gameFull $file.relative
            Assert-Within $gameFull $destination
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
            if ((Get-Hash $destination) -ne $entry.sha256) { throw "Installed file verification failed: $destination" }
            $receipt += [pscustomobject]$entry
        }
        $record = [ordered]@{ schemaVersion = 1; product = 'OptiScaler-RDNA2NR';
            releaseVersion = $releaseVersion;
            installId = $installId; proxyName = $Proxy; modelSha256 = $PackageHash;
            modelPath = $model; files = $receipt }
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
    $form.Size = New-Object System.Drawing.Size(660,350)
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
    $status = New-Object System.Windows.Forms.Label
    $status.Text = 'The source DLL stays where it is. Conversion runs locally.'
    $status.AutoEllipsis = $true
    $status.SetBounds(20,234,610,25); $form.Controls.Add($status)
    $ui = [pscustomobject]@{ Game = $boxes[0]; Dll = $boxes[1]; Proxy = $boxes[2];
        Status = $status; Form = $form; Payload = $Payload; Python = $PythonPath;
        Models = $ModelRoot; Release = $ReleaseRoot }
    $install = New-Object System.Windows.Forms.Button
    $install.Text = 'Install / update'; $install.SetBounds(20,266,130,30)
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
            foreach ($relative in (@($proxy) + $context.Payload)) {
                if (Test-Path -LiteralPath (Join-Path $game $relative) -PathType Leaf) {
                    $existing += $relative
                }
            }
            if ($existing.Count) {
                $message = "Existing game files will be backed up before replacement:`n" +
                    ($existing -join "`n") + "`n`nContinue?"
                $answer = [System.Windows.Forms.MessageBox]::Show($message,
                    'OptiScaler-RDNA2NR', 'YesNo', 'Warning')
                if ($answer -ne 'Yes') { return }
            }
            $context.Status.Text = 'Preparing model and installing...'; $context.Form.Refresh()
            $py = Find-Python $context.Python $context.Release
            Install-Product $game $dll $proxy $context.Release $py (Resolve-ModelRoot $context.Models) | Out-Null
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
    $uninstall.Text = 'Uninstall'; $uninstall.SetBounds(165,266,110,30)
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
    Install-Product $GameDirectory $SourceDll $ProxyName $ReleaseRoot (Find-Python $PythonPath $ReleaseRoot) (Resolve-ModelRoot $ModelRoot)
    Verify-Product $GameDirectory
} elseif ($Mode -eq 'Verify') {
    Verify-Product $GameDirectory
} else {
    Uninstall-Product $GameDirectory
}
