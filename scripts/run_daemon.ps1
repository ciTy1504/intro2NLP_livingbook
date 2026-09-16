# Run the Living Book daemon continuously on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_daemon.ps1
#
# Restarts the process if it dies, because an unattended system that stops on a
# transient failure is not unattended. All state is in SQLite, so a restart resumes
# every pipeline where it left off rather than replaying it.

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
# winget installs MiKTeX per-user; a service or scheduled task may not inherit it.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

$Python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null

Write-Host "Living Book daemon — $Root"
Write-Host "Checking the environment first..."
& $Python -m livingbook.cli doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "doctor reported problems; fix them before running unattended." -ForegroundColor Red
    exit 1
}

$restarts = 0
while ($true) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] starting daemon (restart #$restarts)"

    # --reset-schedule on restart clears any lock left by a hard kill, so a job that
    # was interrupted becomes due again instead of appearing permanently running.
    & $Python -u -m livingbook.cli daemon --reset-schedule 2>&1 |
        Tee-Object -Append -FilePath "$Root\logs\daemon.log"

    $code = $LASTEXITCODE
    if ($code -eq 130) {
        Write-Host "interrupted by user; exiting"
        break
    }

    $restarts++
    $delay = [Math]::Min(60 * $restarts, 900)
    Write-Host "daemon exited with code $code; restarting in ${delay}s" -ForegroundColor Yellow
    Start-Sleep -Seconds $delay
}
