param(
    [string]$CondaEnv = "base",
    [string]$Backend = "numpy",
    [string]$ModelFile = "best_policy_8_8_5.model",
    [int]$Width = 8,
    [int]$Height = 8,
    [int]$NInRow = 5,
    [int]$NPlayout = 400,
    [int]$StartPlayer = 1,
    [string]$MovesFile = "demo_human_moves.txt",
    [string]$LogFile = "artifacts/logs/human_play.log",
    [string]$ScreenshotFile = "artifacts/screenshots/human_play.png"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$CondaHome = Join-Path $RepoRoot ".codex_conda_home"
$XdgHome = Join-Path $RepoRoot ".codex_xdg"
New-Item -ItemType Directory -Force -Path $CondaHome, $XdgHome | Out-Null
$env:HOME = $CondaHome
$env:USERPROFILE = $CondaHome
$env:XDG_CONFIG_HOME = $XdgHome

$LogPath = Join-Path $RepoRoot $LogFile
$ScreenshotPath = Join-Path $RepoRoot $ScreenshotFile
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent), (Split-Path $ScreenshotPath -Parent) | Out-Null

$CommandArgs = @(
    "run", "-n", $CondaEnv, "--no-capture-output",
    "python", "human_play.py",
    "--backend", $Backend,
    "--model-file", $ModelFile,
    "--width", $Width,
    "--height", $Height,
    "--n-in-row", $NInRow,
    "--n-playout", $NPlayout,
    "--start-player", $StartPlayer,
    "--moves-file", $MovesFile
)

& conda @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) {
    throw "human_play.py failed with exit code $LASTEXITCODE"
}

& conda run -n $CondaEnv --no-capture-output python render_terminal_screenshot.py $LogPath $ScreenshotPath
if ($LASTEXITCODE -ne 0) {
    throw "render_terminal_screenshot.py failed with exit code $LASTEXITCODE"
}

Write-Host "Human play log saved to $LogPath"
Write-Host "Human play screenshot saved to $ScreenshotPath"
