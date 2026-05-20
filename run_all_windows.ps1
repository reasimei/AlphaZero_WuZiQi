param(
    [string]$CondaEnv = "base"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

& (Join-Path $RepoRoot "run_human_play_windows.ps1") -CondaEnv $CondaEnv
& (Join-Path $RepoRoot "run_train_windows.ps1") -CondaEnv $CondaEnv
