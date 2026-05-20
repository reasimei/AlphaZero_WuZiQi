param(
    [string]$PythonExe = "C:\ProgramData\miniconda3\python.exe",
    [string]$InitModel = "artifacts\train_8x8_5_500_simple\current_policy.model",
    [string]$RunDir = "artifacts\train_8x8_5_longrun_v1",
    [string]$LogDir = "artifacts\logs",
    [int]$BoardWidth = 8,
    [int]$BoardHeight = 8,
    [int]$NInRow = 5,
    [int]$NPlayout = 32,
    [int]$BatchSize = 64,
    [int]$BufferSize = 30000,
    [int]$Epochs = 5,
    [int]$CheckFreq = 200,
    [int]$GameBatchNum = 20000,
    [int]$PureMctsPlayoutNum = 64,
    [int]$EvalGames = 8,
    [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
$RunPath = Join-Path $RepoRoot $RunDir
$StdoutPath = Join-Path $RepoRoot (Join-Path $LogDir "train_8x8_5_longrun_v1.log")
$StderrPath = Join-Path $RepoRoot (Join-Path $LogDir "train_8x8_5_longrun_v1.err.log")
$PidPath = Join-Path $RunPath "trainer.pid"

$args = @(
    "-u",
    "train.py",
    "--backend", "pytorch",
    "--board-width", $BoardWidth,
    "--board-height", $BoardHeight,
    "--n-in-row", $NInRow,
    "--n-playout", $NPlayout,
    "--batch-size", $BatchSize,
    "--buffer-size", $BufferSize,
    "--play-batch-size", 1,
    "--epochs", $Epochs,
    "--check-freq", $CheckFreq,
    "--game-batch-num", $GameBatchNum,
    "--pure-mcts-playout-num", $PureMctsPlayoutNum,
    "--eval-games", $EvalGames,
    "--seed", $Seed,
    "--use-gpu",
    "--save-dir", $RunDir,
    "--init-model", $InitModel
)

[System.Environment]::SetEnvironmentVariable('PATH', $env:Path, 'Process')
[System.Environment]::SetEnvironmentVariable('Path', $null, 'Process')

$proc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $args `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $PidPath -Value $proc.Id
Write-Host "Started PID $($proc.Id)"
Write-Host "Run dir: $RunPath"
Write-Host "stdout: $StdoutPath"
Write-Host "stderr: $StderrPath"
