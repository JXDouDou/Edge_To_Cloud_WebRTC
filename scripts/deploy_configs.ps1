# ============================================================
# Deploy env-specific configs (staging.yaml etc) to all machines
# ============================================================
#
# These configs are .gitignored, so they cannot be synced via git pull.
# This script scp's them to every target machine.
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts/deploy_configs.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/deploy_configs.ps1 -Only "EC2 Dispatcher 001"
#
# NOTE: Only ASCII characters in this script. Windows PowerShell 5.1 reads
#       .ps1 in system codepage (e.g. CP950) by default. UTF-8 multi-byte
#       chars in comments can corrupt parsing of nearby variable assignments.
#       Keep this file ASCII-only to avoid that.

param(
    [string]$Only = ""
)

# ── Force UTF-8 console output (so file paths with non-ASCII chars print ok) ──
chcp 65001 > $null 2>&1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# ────────────────────────────────────────────────────────────
# CONFIGURATION (edit these to match your environment)
# ────────────────────────────────────────────────────────────

# EC2 SSH private key path (used for ubuntu@... connections)
$EC2KeyPath = "C:\Users\Owner\.ssh\aws_key"

# Files to upload (relative to project root)
$Files = @(
    "config\staging.yaml",
    "config\staging_video.yaml"
)

# Deployment targets - EDIT VALUES HERE DIRECTLY (no variable indirection)
$Targets = @(
    @{
        Name       = "Pi (Edge)"
        SshUser    = "mawatarilab"
        # If 'mawatarilab' alone cannot be resolved by Windows DNS,
        # change to one of:
        #   "mawatarilab.local"   - mDNS (works on most LANs)
        #   "192.168.x.x"         - Pi LAN IP (run 'hostname -I' on Pi)
        #   "<tailscale-name>"    - if Pi has Tailscale installed
        SshHost    = "mawatarilab-edge"
        RemotePath = "~/Project_Edge/config/"
        KeyPath    = ""
    }
    @{
        Name       = "EC2 Signaling"
        SshUser    = "ubuntu"
        SshHost    = "18.178.31.155"
        RemotePath = "~/Edge_To_Cloud_WebRTC/config/"
        KeyPath    = $EC2KeyPath
    }
    @{
        Name       = "EC2 Dispatcher 001"
        SshUser    = "ubuntu"
        SshHost    = "57.181.45.231"
        # NOTE: Verify with: ssh ubuntu@57.181.45.231 "ls"
        # If you see "Edge_To_Cloud_WebRTC", change back to that
        RemotePath = "~/project/config/"
        KeyPath    = $EC2KeyPath
    }
    @{
        Name       = "EC2 Dispatcher 002"
        SshUser    = "ubuntu"
        SshHost    = "35.72.149.122"
        # NOTE: Verify with: ssh ubuntu@35.72.149.122 "ls"
        RemotePath = "~/project/config/"
        KeyPath    = $EC2KeyPath
    }
    @{
        Name       = "Inference (Windows)"
        SshUser    = "user"
        SshHost    = "mawatarilab-inference"
        RemotePath = "C:/Code_Inference/Edge_To_Cloud_WebRTC/config/"
        KeyPath    = ""
    }
)

# ────────────────────────────────────────────────────────────
# DO NOT EDIT BELOW
# ────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"
$totalSuccess = 0
$totalFail = 0

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Deploy configs to all machines" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Validate EC2 key exists upfront
if (-not (Test-Path $EC2KeyPath)) {
    Write-Host "[WARN] EC2 key not found at: $EC2KeyPath" -ForegroundColor Yellow
    Write-Host "       EC2 targets will likely fail with 'Permission denied (publickey)'" -ForegroundColor Yellow
    Write-Host ""
}

foreach ($t in $Targets) {
    if ($Only -and $t.Name -ne $Only) {
        continue
    }

    $userHost = "$($t.SshUser)@$($t.SshHost)"
    Write-Host "==> [$($t.Name)] $userHost" -ForegroundColor Yellow
    Write-Host "    target: $($t.RemotePath)" -ForegroundColor DarkGray
    if ($t.KeyPath) {
        Write-Host "    key:    $($t.KeyPath)" -ForegroundColor DarkGray
    }

    foreach ($f in $Files) {
        if (-not (Test-Path $f)) {
            Write-Host "    [SKIP] local file not found: $f" -ForegroundColor Red
            $totalFail++
            continue
        }

        $dest = "${userHost}:$($t.RemotePath)"
        Write-Host "    scp $f  ->  $dest" -ForegroundColor DarkGray

        # NOTE: do NOT redirect stderr/stdout - that breaks password prompts
        if ($t.KeyPath -and (Test-Path $t.KeyPath)) {
            scp -i $t.KeyPath -o StrictHostKeyChecking=accept-new $f $dest
        }
        elseif ($t.KeyPath) {
            Write-Host "    [SKIP] key path set but file missing: $($t.KeyPath)" -ForegroundColor Red
            $totalFail++
            continue
        }
        else {
            scp -o StrictHostKeyChecking=accept-new $f $dest
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Host "    [OK]   $f" -ForegroundColor Green
            $totalSuccess++
        }
        else {
            Write-Host "    [FAIL] $f  (scp exit=$LASTEXITCODE)" -ForegroundColor Red
            $totalFail++
        }
    }
    Write-Host ""
}

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Done: OK=$totalSuccess, FAIL=$totalFail" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Restart processes on each machine after deploy:" -ForegroundColor DarkGray
Write-Host "    EC2 Signaling:    Ctrl+C -> python signaling/server.py --config config/staging.yaml" -ForegroundColor DarkGray
Write-Host "    EC2 Dispatcher x2:Ctrl+C -> python dispatcher/main.py --config config/staging.yaml --id dispatcher-ec2-001" -ForegroundColor DarkGray
Write-Host "    Inference:        Ctrl+C -> python inference/main.py --config config/staging.yaml" -ForegroundColor DarkGray
Write-Host "    Pi Edge:          Ctrl+C -> python edge/main.py --config config/staging.yaml" -ForegroundColor DarkGray
Write-Host "===============================================" -ForegroundColor Cyan
