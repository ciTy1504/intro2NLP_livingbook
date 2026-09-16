# Register the Living Book daemon as a Windows scheduled task, so it survives reboots
# and keeps running after you log out.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_scheduled_task.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_scheduled_task.ps1 -Remove
#
# Runs as the current user rather than SYSTEM: the daemon reads secrets/.env and the
# per-user MiKTeX install, neither of which SYSTEM can see.

param(
    [switch]$Remove,
    [string]$TaskName = "LivingBookDaemon"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot "run_daemon.ps1"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'"
    } else {
        Write-Host "No scheduled task named '$TaskName'"
    }
    exit 0
}

if (-not (Test-Path $Script)) { throw "run_daemon.ps1 not found at $Script" }

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`"" `
    -WorkingDirectory $Root

# At logon and at startup: the daemon is meant to be always-on, and its own scheduler
# decides what is actually due, so starting it more often than necessary is harmless.
$triggers = @(
    (New-ScheduledTaskTrigger -AtLogOn),
    (New-ScheduledTaskTrigger -AtStartup)
)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $triggers `
    -Settings $settings `
    -Principal $principal `
    -Description "Living Book: continuous research-to-publication pipeline for intro2NLP." | Out-Null

Write-Host "Registered scheduled task '$TaskName'."
Write-Host ""
Write-Host "  start now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  stop      : Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  status    : Get-ScheduledTask   -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  logs      : Get-Content '$Root\logs\daemon.log' -Tail 50 -Wait"
Write-Host "  remove    : .\scripts\install_scheduled_task.ps1 -Remove"
