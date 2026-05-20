param(
    [string]$RunDir = "artifacts\train_8x8_5_longrun_v1",
    [string]$PythonExe = "C:\ProgramData\miniconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$RunPath = Join-Path $RepoRoot $RunDir
$PidPath = Join-Path $RunPath "trainer.pid"
$MetricsPath = Join-Path $RunPath "training_metrics.jsonl"

if (Test-Path $PidPath) {
    $trainerPid = Get-Content $PidPath | Select-Object -First 1
    $proc = Get-Process -Id $trainerPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "status: running"
        Write-Host "pid: $trainerPid"
    } else {
        Write-Host "status: stopped"
        Write-Host "pid: $trainerPid"
    }
} else {
    Write-Host "status: no-pid-file"
}

if (Test-Path $MetricsPath) {
    & $PythonExe plot_training_metrics.py --run-dir $RunDir | Out-Null
    Get-Content $MetricsPath -Tail 5
} else {
    Write-Host "No metrics file yet."
}
