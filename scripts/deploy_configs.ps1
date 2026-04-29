# ============================================================
# 一鍵把含 credential 的 config 同步到所有部署機器
# ============================================================
#
# staging.yaml / staging_video.yaml / prod.yaml 在 .gitignore 中，
# 不能透過 git pull 同步，要用此腳本 scp 推到所有機器。
#
# 使用方式（在專案根目錄）：
#   powershell -ExecutionPolicy Bypass -File scripts/deploy_configs.ps1
#
# 預設會推下方 $Files 中的所有檔案到所有 $Targets。
# 如果只想推一台，可加參數 -Only "EC2 Dispatcher 001"

param(
    [string]$Only = ""    # 留空 = 全部；填名稱 = 只推那一台
)

# ── 環境變數 ────────────────────────────────────────────────
# EC2 SSH key 路徑（連 ubuntu@... 用）
$EC2KeyPath = "C:\Users\Owner\.ssh\my-ec2-key.pem"   # ← 改成你的 .pem 實際位置

# Pi SSH 設定
# 你平常從 Windows SSH 進 Pi 用的命令是什麼？把 @ 後面那串填到 $PiHost：
#   ssh mawatarilab@mawatarilab           → "mawatarilab"
#   ssh mawatarilab@mawatarilab.local     → "mawatarilab.local"  (mDNS)
#   ssh mawatarilab@192.168.x.x           → "192.168.x.x"        (LAN IP)
#   ssh mawatarilab@<tailscale-name>      → 那個 hostname        (Tailscale)
$PiUser = "mawatarilab"
$PiHost = "mawatarilab"           # ← 若連不上，依上方註解改
$PiKeyPath = ""                   # 用密碼登入留空；用 key 才填路徑

# Inference 桌機（Windows 上的 OpenSSH server，走 Tailscale）
$InferenceUser = "user"                          # ← 改成桌機 Windows 帳號名
$InferenceHost = "mawatarilab-inference"         # Tailscale MagicDNS 名稱
$InferenceKeyPath = ""                           # 用密碼登入留空

# ── 部署目標 ────────────────────────────────────────────────
$Targets = @(
    @{
        Name       = "Pi (Edge)"
        User       = $PiUser
        Host       = $PiHost
        RemotePath = "~/Project_Edge/config/"
        KeyPath    = $PiKeyPath
        IsWindows  = $false
    }
    @{
        Name       = "EC2 Signaling"
        User       = "ubuntu"
        Host       = "18.178.31.155"
        RemotePath = "~/Edge_To_Cloud_WebRTC/config/"
        KeyPath    = $EC2KeyPath
        IsWindows  = $false
    }
    @{
        Name       = "EC2 Dispatcher 001"
        User       = "ubuntu"
        Host       = "57.181.45.231"
        RemotePath = "~/Edge_To_Cloud_WebRTC/config/"
        KeyPath    = $EC2KeyPath
        IsWindows  = $false
    }
    @{
        Name       = "EC2 Dispatcher 002"
        User       = "ubuntu"
        Host       = "35.72.149.122"
        RemotePath = "~/Edge_To_Cloud_WebRTC/config/"
        KeyPath    = $EC2KeyPath
        IsWindows  = $false
    }
    @{
        Name       = "Inference (Windows 桌機)"
        User       = $InferenceUser
        Host       = $InferenceHost
        # Windows 路徑：scp 可吃正斜線，比較不會踩到引號跳脫的雷
        RemotePath = "C:/Code_Inference/Edge_To_Cloud_WebRTC/config/"
        KeyPath    = $InferenceKeyPath
        IsWindows  = $true
    }
)

# ── 要推送的檔案 ────────────────────────────────────────────
$Files = @(
    "config\staging.yaml",
    "config\staging_video.yaml"
)
# 之後若有 prod.yaml 含敏感資料，加進來:
#   "config\prod.yaml"

# ── 以下不用改 ──────────────────────────────────────────────
$ErrorActionPreference = "Continue"
$totalSuccess = 0
$totalFail = 0

Write-Host ""
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Config 部署 — 同步到所有機器" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

foreach ($t in $Targets) {
    if ($Only -and $t.Name -ne $Only) {
        continue
    }

    Write-Host "▶ [$($t.Name)] $($t.User)@$($t.Host)" -ForegroundColor Yellow
    Write-Host "  目標路徑: $($t.RemotePath)" -ForegroundColor DarkGray

    foreach ($f in $Files) {
        if (-not (Test-Path $f)) {
            Write-Host "  ✗ 跳過：本地找不到 $f" -ForegroundColor Red
            $totalFail++
            continue
        }

        $dest = "$($t.User)@$($t.Host):$($t.RemotePath)"

        if ($t.KeyPath -and (Test-Path $t.KeyPath)) {
            scp -i $t.KeyPath -o StrictHostKeyChecking=accept-new $f $dest 2>&1 | Out-Null
        }
        else {
            scp -o StrictHostKeyChecking=accept-new $f $dest 2>&1 | Out-Null
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $f" -ForegroundColor Green
            $totalSuccess++
        }
        else {
            Write-Host "  ✗ $f (scp exit=$LASTEXITCODE)" -ForegroundColor Red
            $totalFail++
        }
    }
    Write-Host ""
}

Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  完成: ✓ $totalSuccess 成功, ✗ $totalFail 失敗" -ForegroundColor Cyan
Write-Host ""
Write-Host "  記得到各機器重啟對應的 process（process 是啟動時讀 yaml）：" -ForegroundColor DarkGray
Write-Host "    EC2 Signaling:    Ctrl+C → python signaling/server.py --config config/staging.yaml" -ForegroundColor DarkGray
Write-Host "    EC2 Dispatcher×2: Ctrl+C → python dispatcher/main.py --config config/staging.yaml --id dispatcher-ec2-001" -ForegroundColor DarkGray
Write-Host "    Inference 桌機:    Ctrl+C → python inference/main.py --config config/staging.yaml" -ForegroundColor DarkGray
Write-Host "    Pi Edge:          Ctrl+C → python edge/main.py --config config/staging.yaml" -ForegroundColor DarkGray
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
