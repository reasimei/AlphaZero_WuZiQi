param(
    [string]$CondaEnv = "base",
    [string]$Backend = "pytorch",
    [int]$BoardWidth = 6,
    [int]$BoardHeight = 6,
    [int]$NInRow = 4,
    [int]$NPlayout = 64,
    [float]$CPuct = 5,
    [int]$BatchSize = 32,
    [int]$BufferSize = 10000,
    [int]$PlayBatchSize = 1,
    [int]$Epochs = 5,
    [int]$CheckFreq = 2,
    [int]$GameBatchNum = 2,
    [int]$PureMctsPlayoutNum = 64,
    [string]$SaveDir = "artifacts/train_run",
    [string]$LogFile = "artifacts/logs/train.log",
    [string]$ScreenshotFile = "artifacts/screenshots/train.png",
    [switch]$UseGpu
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
$SavePath = Join-Path $RepoRoot $SaveDir
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent), (Split-Path $ScreenshotPath -Parent), $SavePath | Out-Null

$CommandArgs = @(
    "run", "-n", $CondaEnv, "--no-capture-output",
    "python", "train.py",
    "--backend", $Backend,
    "--board-width", $BoardWidth,
    "--board-height", $BoardHeight,
    "--n-in-row", $NInRow,
    "--n-playout", $NPlayout,
    "--c-puct", $CPuct,
    "--batch-size", $BatchSize,
    "--buffer-size", $BufferSize,
    "--play-batch-size", $PlayBatchSize,
    "--epochs", $Epochs,
    "--check-freq", $CheckFreq,
    "--game-batch-num", $GameBatchNum,
    "--pure-mcts-playout-num", $PureMctsPlayoutNum,
    "--save-dir", $SaveDir
)

if ($UseGpu) {
    $CommandArgs += "--use-gpu"
}

& conda @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) {
    throw "train.py failed with exit code $LASTEXITCODE"
}

& conda run -n $CondaEnv --no-capture-output python render_terminal_screenshot.py $LogPath $ScreenshotPath
if ($LASTEXITCODE -ne 0) {
    throw "render_terminal_screenshot.py failed with exit code $LASTEXITCODE"
}

Write-Host "Training log saved to $LogPath"
Write-Host "Training screenshot saved to $ScreenshotPath"
Write-Host "Training outputs saved to $SavePath"
