param([string]$OutputRoot='build/tests/hooks', [switch]$Run, [switch]$NativeTestHooks)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. (Join-Path $root 'tools/Enter-Toolchain.ps1') -PlatformToolset v145 -MsvcToolsVersion 14.44
Push-Location $root
try {
    [xml]$project=Get-Content (Join-Path $root 'OptiScaler/OptiScaler.vcxproj') -Raw
    $globalDiagnostic=@($project.Project.ItemDefinitionGroup | Where-Object {
        $_.Condition -eq "'`$(NrNativeTestHooks)'=='true'" -and
        $_.ClCompile.PreprocessorDefinitions -match 'NR_NATIVE_TEST_HOOKS'
    })
    if($globalDiagnostic.Count -ne 1){throw 'Inline hook diagnostic definitions must be project-wide.'}
    [string[]]$diagnosticFlags=@()
    if($NativeTestHooks){$diagnosticFlags+= '/DNR_NATIVE_TEST_HOOKS'}
    # These call sites were once omitted from a migration despite copying the
    # hub itself. Raw detours here bypass its recorder/Private/Restore policy.
    foreach($file in @('hooks/D3D12_Hooks.cpp','resource_tracking/ResTrack_dx12.cpp')){
        $text=Get-Content -LiteralPath (Join-Path $root "OptiScaler/$file") -Raw
        if($text -notmatch 'NativeHookHub.h' -or
           $text -match 'Detour(?:Attach|Detach)\(\s*&\(PVOID&\)\s*(?:s_Set\w+\.o_(?:early|late)Hook|o_(?:LateReset|Reset|ClearState|OMSetRenderTargets|Set(?:Graphics|Compute)RootDescriptorTable|Draw(?:Indexed)?Instanced|Dispatch))\s*,'){
            throw "Command-list hooks bypass the shared recorder: $file"
        }
    }
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    foreach($name in @('hook_order','hook_hub','runtime_dispatch')){
        & cl.exe /nologo /std:c++20 /EHsc /O2 /MD /W4 /WX /IOptiScaler /IOptiScaler/include @diagnosticFlags `
            "tests/rdna2/$name.cpp" "/Fo:$OutputRoot/$name.obj" /link `
            /LIBPATH:OptiScaler/library/detours detours.lib "/OUT:$OutputRoot/$name.exe"
        if($LASTEXITCODE){throw "$name build failed"}
        if($Run){
            & (Join-Path (Resolve-Path $OutputRoot) "$name.exe") | Tee-Object (Join-Path $OutputRoot "$name.log")
            if($LASTEXITCODE){throw "$name failed"}
        }
    }
}finally{Pop-Location}
