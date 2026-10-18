<#
================================================================================
schedule_daily.ps1  --  run the daily fetch with Windows Task Scheduler
================================================================================
Registers a task that runs, every day:

    uv run python Scripts/pipeline.py daily

which starts Open-Meteo if it is down, brings the history up to yesterday,
fetches the 7-day forecast, predicts, and prunes old forecast runs. Output
goes to logs/daily_<date>.log.

The task runs as you, only while you are logged on (Docker Desktop runs in
your session). A run missed because the PC was off or asleep starts as soon
as it is back; a run still going when the next one is due is not doubled.

Default time 11:30: just after midnight UTC in Sydney (AEDT and AEST), so
yesterday's UTC day is complete and today's forecast starts fresh.

USAGE (from the repo root)
-----
    powershell -ExecutionPolicy Bypass -File Scripts\schedule_daily.ps1
    powershell -ExecutionPolicy Bypass -File Scripts\schedule_daily.ps1 -At 07:00
    powershell -ExecutionPolicy Bypass -File Scripts\schedule_daily.ps1 -Remove
    Start-ScheduledTask -TaskName WildfireDailyFetch      # run it now
================================================================================
#>
param(
    [string]$At = "11:30",
    [string]$TaskName = "WildfireDailyFetch",
    [switch]$Remove
)
$ErrorActionPreference = "Stop"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
    return
}

$repo = Split-Path -Parent $PSScriptRoot
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw "uv is not on PATH. Install it or open a shell where 'uv' works." }

$action = New-ScheduledTaskAction -Execute $uv `
    -Argument "run python Scripts\pipeline.py daily" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Wildfire prediction: daily weather fetch, predictions and prune ($repo)" `
    -Force | Out-Null

$next = (Get-ScheduledTaskInfo -TaskName $TaskName).NextRunTime
Write-Host "Scheduled task '$TaskName' runs daily at $At (next: $next)."
Write-Host "Command : $uv run python Scripts\pipeline.py daily"
Write-Host "Folder  : $repo"
Write-Host "Logs    : $repo\logs\daily_<date>.log"
