<#
================================================================================
setup.ps1  --  one command from a fresh clone to the running dashboard
================================================================================
    1. uv            installed if missing (it manages Python and the packages)
    2. Python 3.12   installed through uv if missing (.python-version)
    3. packages      uv sync
    4. data          git lfs pull if the history CSVs are still LFS pointers
    5. Docker        must be installed (you are asked to install it if not);
                     Docker Desktop is started if it is not running, then the
                     Open-Meteo container (docker compose up -d)
    6. pipeline      Scripts/pipeline.py setup: create the database, load the
                     weather history (first run only), fetch the history up to
                     yesterday, the 7-day forecast and the predictions
    7. app           FastAPI (http://127.0.0.1:8000/docs) and Streamlit
                     (http://localhost:8501), each in its own window

Every step is skipped when it is already done, so running it again just
brings the data up to date and starts the app.

USAGE (repo root; or double-click setup.cmd)
-----
    powershell -ExecutionPolicy Bypass -File setup.ps1
    powershell -ExecutionPolicy Bypass -File setup.ps1 -PublicApi      # no Docker: public Open-Meteo
    powershell -ExecutionPolicy Bypass -File setup.ps1 -SkipPipeline   # just start the app
================================================================================
#>
param(
    [switch]$PublicApi,      # use api.open-meteo.com instead of the Docker server
    [switch]$SkipPipeline,   # do not touch the data, only start the app
    [int]$DockerWaitSeconds = 180
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
Set-Location $repo

$ApiUrl = "http://127.0.0.1:8000"
$DashboardUrl = "http://localhost:8501"
$OpenMeteoUrl = "http://127.0.0.1:8080/v1/forecast"
$DockerDownload = "https://docs.docker.com/desktop/setup/install/windows-install/"
$LfsFiles = @("data\full_grid_daily_weather.csv", "data\full_grid_land_mask.csv")

function Step($n, $text) { Write-Host ""; Write-Host "[$n/7] $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "      OK  $text" -ForegroundColor Green }
function Info($text) { Write-Host "          $text" }
function Fail($text) { Write-Host ""; Write-Host "STOPPED: $text" -ForegroundColor Red; exit 1 }

# Native commands write progress to stderr; under "Stop" PowerShell 5.1 would
# treat redirected stderr as an error, so quiet checks go through cmd.
function Test-Command($cmdline) { cmd /c "$cmdline >nul 2>&1"; return ($LASTEXITCODE -eq 0) }

function Test-Url($url) {
    try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 | Out-Null; return $true }
    catch { return [bool]$_.Exception.Response }   # an HTTP error still means it is up
}

function Wait-Until($seconds, [scriptblock]$check) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $check) { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Test-LfsPointer($path) {
    if (-not (Test-Path $path)) { return $true }
    $head = Get-Content $path -TotalCount 1 -ErrorAction SilentlyContinue
    return ($head -like "version https://git-lfs*")
}

Write-Host "Wildfire risk prediction -- setup" -ForegroundColor Yellow
Write-Host "Repo: $repo"

# ------------------------------------------------------------------ 1. uv
Step 1 "uv (Python package manager)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Info "uv not found -- installing it from astral.sh..."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail "uv was installed but is not on PATH. Open a new terminal and run setup again."
    }
}
Ok (uv --version)

# -------------------------------------------------------------- 2. Python
Step 2 "Python $(Get-Content .python-version)"
if (Test-Command "uv python find") {
    Ok "found: $(uv python find)"
} else {
    Info "not found -- installing it with uv..."
    uv python install
    if ($LASTEXITCODE -ne 0) { Fail "uv could not install Python." }
    Ok "installed: $(uv python find)"
}

# ------------------------------------------------------------ 3. packages
Step 3 "Python packages (uv sync)"
uv sync
if ($LASTEXITCODE -ne 0) { Fail "uv sync failed (see above)." }
Ok "packages installed"

# ---------------------------------------------------------------- 4. data
Step 4 "Data files (Git LFS)"
$pointers = @($LfsFiles | Where-Object { Test-LfsPointer (Join-Path $repo $_) })
if ($pointers.Count -gt 0) {
    Info "still LFS pointers: $($pointers -join ', ')"
    if (-not (Test-Command "git lfs version")) {
        Fail "Git LFS is not installed. Install it from https://git-lfs.com, run 'git lfs install', then run setup again."
    }
    git lfs pull
    if ($LASTEXITCODE -ne 0) { Fail "git lfs pull failed (see above)." }
    $pointers = @($LfsFiles | Where-Object { Test-LfsPointer (Join-Path $repo $_) })
    if ($pointers.Count -gt 0) { Fail "still missing after git lfs pull: $($pointers -join ', ')" }
}
Ok "history and land-mask CSVs present"

# -------------------------------------------------------------- 5. Docker
Step 5 "Weather server"
if ($SkipPipeline) {
    Info "skipped (-SkipPipeline)"
} elseif ($PublicApi) {
    $env:OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    $env:OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    Ok "public Open-Meteo API (-PublicApi); requests are paced to its free limits"
} else {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "Docker Desktop is not installed. It runs the local Open-Meteo weather server." -ForegroundColor Yellow
        Write-Host "  1. Install it:  $DockerDownload"
        Write-Host "     (or: winget install -e --id Docker.DockerDesktop)"
        Write-Host "  2. Start Docker Desktop once and finish its first-run setup."
        Write-Host "  3. Run this setup again."
        Write-Host "Or run without Docker:  powershell -ExecutionPolicy Bypass -File setup.ps1 -PublicApi"
        exit 1
    }
    if (-not (Test-Command "docker info")) {
        $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
        if (-not (Test-Path $desktop)) {
            Fail "Docker is installed but its engine is not running, and Docker Desktop was not found at $desktop. Start Docker, then run setup again."
        }
        Info "starting Docker Desktop (up to $DockerWaitSeconds s)..."
        Start-Process $desktop
        if (-not (Wait-Until $DockerWaitSeconds { Test-Command "docker info" })) {
            Fail "Docker did not start within $DockerWaitSeconds s. Open Docker Desktop, wait until it says 'Engine running', then run setup again."
        }
    }
    Ok "Docker engine running"
    docker compose up -d
    if ($LASTEXITCODE -ne 0) { Fail "docker compose up -d failed (see above)." }
    Info "waiting for Open-Meteo at $OpenMeteoUrl..."
    if (-not (Wait-Until 120 { Test-Url $OpenMeteoUrl })) {
        Fail "the Open-Meteo container is running but not answering. Check: docker compose logs open-meteo"
    }
    Ok "Open-Meteo answering on port 8080"
}

# ------------------------------------------------------------ 6. pipeline
Step 6 "Database, weather history, forecast and predictions"
if ($SkipPipeline) {
    Info "skipped (-SkipPipeline)"
} else {
    Info "first run: loads ~3M history rows and builds features (several minutes);"
    Info "later runs only fetch the days since the last run."
    uv run python Scripts\pipeline.py setup
    switch ($LASTEXITCODE) {
        0 { Ok "data up to date" }
        1 { Write-Host "      WARN  some cells were not refreshed; their previous predictions stay current" -ForegroundColor Yellow }
        default { Fail "the pipeline failed (exit $LASTEXITCODE, see above and logs\daily_<date>.log)." }
    }
}

# ----------------------------------------------------------------- 7. app
Step 7 "Start FastAPI and Streamlit"
function Start-Window($title, $command) {
    $inner = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location '$repo'; $command"
    Start-Process powershell -ArgumentList "-NoExit", "-NoProfile", "-Command", $inner -WorkingDirectory $repo
}

if (Test-Url "$ApiUrl/health") {
    Ok "API already running at $ApiUrl"
} else {
    Start-Window "Wildfire API (FastAPI)" "uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000"
    if (-not (Wait-Until 90 { Test-Url "$ApiUrl/health" })) {
        Fail "the API did not start; see the 'Wildfire API (FastAPI)' window."
    }
    Ok "API running at $ApiUrl  (docs: $ApiUrl/docs)"
}

if (Test-Url $DashboardUrl) {
    Ok "dashboard already running at $DashboardUrl"
    Start-Process $DashboardUrl
} else {
    # Headless: otherwise Streamlit's first run stops to ask for an email.
    Start-Window "Wildfire dashboard (Streamlit)" `
        "uv run streamlit run frontend\main.py --server.port 8501 --server.headless true"
    if (-not (Wait-Until 90 { Test-Url $DashboardUrl })) {
        Fail "the dashboard did not start; see the 'Wildfire dashboard (Streamlit)' window."
    }
    Ok "dashboard running at $DashboardUrl"
    Start-Process $DashboardUrl
}

Write-Host ""
Write-Host "All set." -ForegroundColor Green
Write-Host "  Dashboard : $DashboardUrl"
Write-Host "  API docs  : $ApiUrl/docs"
Write-Host "  Stop      : close the two app windows (Open-Meteo: docker compose stop)"
Write-Host "  Daily data: powershell -ExecutionPolicy Bypass -File Scripts\schedule_daily.ps1"
