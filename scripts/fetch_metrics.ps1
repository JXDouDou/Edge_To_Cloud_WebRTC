# ============================================================
# Fetch latest metrics run from Pi to local Windows machine
# ============================================================
#
# Usage (project root):
#   .\scripts\fetch_metrics.ps1                 # latest run
#   .\scripts\fetch_metrics.ps1 -All            # all runs
#   .\scripts\fetch_metrics.ps1 -Run run_xxx    # specific run
#
# After fetch, opens the folder in Explorer.

param(
    [string]$Run = "",
    [switch]$All
)

# Edit if Pi's Tailscale IP/hostname changes
$PiUser = "mawatarilab"
$PiHost = "100.98.101.5"
$PiMetricsDir = "~/Project_Edge/metrics"

# Local destination
$LocalDest = Join-Path $PSScriptRoot "..\metrics"
$LocalDest = (Resolve-Path -Path $LocalDest -ErrorAction SilentlyContinue).Path
if (-not $LocalDest) {
    $LocalDest = Join-Path (Split-Path $PSScriptRoot -Parent) "metrics"
    New-Item -ItemType Directory -Path $LocalDest -Force | Out-Null
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Fetch metrics from Pi" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

if ($All) {
    Write-Host "Mode: ALL runs" -ForegroundColor Yellow
    scp -r "${PiUser}@${PiHost}:${PiMetricsDir}/*" $LocalDest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] scp exit=$LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] All runs fetched to: $LocalDest" -ForegroundColor Green
    explorer $LocalDest
    exit 0
}

if (-not $Run) {
    # Find latest run on Pi
    Write-Host "Finding latest run on Pi..." -ForegroundColor DarkGray
    $Run = ssh "${PiUser}@${PiHost}" "ls -1t ${PiMetricsDir} 2>/dev/null | head -1"
    if (-not $Run) {
        Write-Host "[FAIL] No metrics runs found on Pi at ${PiMetricsDir}" -ForegroundColor Red
        exit 1
    }
    $Run = $Run.Trim()
}

Write-Host "Target run: $Run" -ForegroundColor Yellow
$src = "${PiUser}@${PiHost}:${PiMetricsDir}/${Run}"
$dst = Join-Path $LocalDest $Run
Write-Host "  src: $src" -ForegroundColor DarkGray
Write-Host "  dst: $dst" -ForegroundColor DarkGray
Write-Host ""

scp -r $src $LocalDest
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] scp exit=$LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Fetched to: $dst" -ForegroundColor Green
Write-Host ""

# Show summary.txt content right here
$summaryPath = Join-Path $dst "summary.txt"
if (Test-Path $summaryPath) {
    Write-Host "--- summary.txt ---" -ForegroundColor Cyan
    Get-Content $summaryPath
    Write-Host "-------------------" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Opening folder..." -ForegroundColor DarkGray
explorer $dst
