# 在 Signaling EC2 自建 coturn

> 目的：取代不穩定的免費 TURN（ExpressTURN / openrelay），讓 5G Pi 能可靠地穿透 NAT 連到 dispatcher EC2。
>
> 部署位置：**現有的 Signaling EC2（18.178.31.155）**，跟 signaling 共用一台不衝突。
> 預估時間：20–30 分鐘。

---

## 0. 為什麼自建

| 方案 | 結論 |
|---|---|
| ExpressTURN 免費 | 收到 `403 Forbidden IP` — 拒絕 relay 到 EC2 IP 段 |
| openrelay.metered.ca 免費 | 帳號失效，根本拿不到 relay candidate |
| Twilio TURN | 穩定但按用量收費 |
| **自建 coturn** | 一次設定永久受用、沒流量限制、可信賴 |

成本估算（EC2 t3.micro + 自家 coturn）：
- 額外 CPU/記憶體：可忽略（coturn 閒置時 < 20 MB）
- 出站流量：5 fps × 3 KB × 2 方向 = 30 KB/s ≈ 100 MB/小時 ≈ **每小時 $0.009 USD**
- 每天 4 小時 × 30 天 ≈ 12 GB ≈ **$1 USD/月**

---

## 1. 前置確認

```bash
# SSH 進 Signaling EC2
ssh ubuntu@18.178.31.155

# 確認你跑 signaling 的這台機器
hostname            # 應該是 ip-172-16-11-118
ip -4 addr show     # 找你的 internal IP（通常 172.16.x.x 或 172.31.x.x）

# 也記下 EIP（從你 AWS Console 看）
# 這個範例假設 EIP = 18.178.31.155，internal IP = 172.16.11.118
# 後面設定檔會用到這兩個
```

---

## 2. 安裝 coturn

```bash
sudo apt update
sudo apt install -y coturn

# 確認版本（4.5 以上都行）
turnserver --version
```

---

## 3. 啟用 daemon（apt 預設 disabled）

```bash
sudo nano /etc/default/coturn
```

把這行的註解拿掉，存檔：
```
TURNSERVER_ENABLED=1
```

---

## 4. 寫 coturn 設定

**先備份 apt 給的預設檔**：
```bash
sudo cp /etc/turnserver.conf /etc/turnserver.conf.bak
```

**重寫整個檔案：**
```bash
sudo nano /etc/turnserver.conf
```

把下面整段貼進去（取代全部內容）：

```conf
# ============================================================
# coturn 設定（給 Edge-to-Cloud WebRTC 用）
# ============================================================

# 監聽 port（標準 TURN port）
listening-port=3478

# 監聽介面
listening-ip=0.0.0.0

# 重要：external-ip 告訴 coturn 它的對外 IP 是什麼
# 格式：external-ip=PUBLIC_IP/PRIVATE_IP
# PUBLIC_IP  = EC2 EIP
# PRIVATE_IP = ip -4 addr show 看到的 EC2 internal IP
external-ip=18.178.31.155/172.16.11.118

# Relay port 範圍（要跟 AWS Security Group 開的範圍一致）
min-port=49152
max-port=65535

# 認證方式：long-term credential（username / password）
lt-cred-mech

# 使用者帳密（可加多個 user= 行）
# ⚠️ 改成你自己的強密碼，至少 16 字元
user=mawatari:CHANGE_ME_TO_STRONG_PASSWORD_12345

# realm 隨便取，跟 yaml 對應即可（其實 yaml 不用寫）
realm=mawatarilab

# 一些好習慣的安全設定
fingerprint
no-multicast-peers
no-cli
no-loopback-peers
no-tcp-relay

# 不對外提供純 STUN（我們用 Google STUN，這台只專心做 TURN）
no-stun

# Log：debug 階段開 verbose，穩定後改成 simple-log
verbose
log-file=/var/log/coturn/turn.log
# simple-log
# pidfile=/var/run/turnserver.pid

# 拒絕 relay 到敏感 IP 段（避免 coturn 被當開放代理）
denied-peer-ip=0.0.0.0-0.255.255.255
denied-peer-ip=10.0.0.0-10.255.255.255
denied-peer-ip=100.64.0.0-100.127.255.255
denied-peer-ip=127.0.0.0-127.255.255.255
denied-peer-ip=169.254.0.0-169.254.255.255
denied-peer-ip=172.16.0.0-172.31.255.255
denied-peer-ip=192.0.0.0-192.0.0.255
denied-peer-ip=192.0.2.0-192.0.2.255
denied-peer-ip=192.88.99.0-192.88.99.255
denied-peer-ip=192.168.0.0-192.168.255.255
denied-peer-ip=198.18.0.0-198.19.255.255
denied-peer-ip=198.51.100.0-198.51.100.255
denied-peer-ip=203.0.113.0-203.0.113.255
denied-peer-ip=240.0.0.0-255.255.255.255

# 允許 relay 到你自己的 EC2 段（dispatcher 那兩台）
# 用 EC2 internal IP 段；自己看 dispatcher 的 internal IP 在哪段
allowed-peer-ip=172.16.0.0-172.31.255.255
```

**改完存檔 (Ctrl+O Enter, Ctrl+X)。**

### ⚠️ 關鍵：要改的兩個欄位

| 欄位 | 改什麼 |
|---|---|
| `external-ip` | 你的 EIP/internal IP，例如 `18.178.31.155/172.16.11.118` |
| `user=mawatari:...` | 改成你自己的強密碼（用 `openssl rand -base64 24` 產一個） |

### ⚠️ allowed-peer-ip 段

我先寫了 `172.16.0.0-172.31.255.255` 涵蓋常見 AWS VPC 內網段。但你的兩台 dispatcher 是用**公網 IP（EIP）**通訊還是內網？

ICE 候選看 dispatcher 給的 candidate。如果 dispatcher 是 `srflx`（EC2 EIP，例如 57.181.45.231），那 TURN 要 relay 到的目標是「dispatcher 的 EIP」，不是 internal。

**安全做法：先暫時放寬，確定可以後再縮：**

把上面那一大段 `denied-peer-ip` 跟 `allowed-peer-ip` 全刪掉，先讓 coturn 不限制 relay 目標。等通了再加回去（先求會動）。

或者明確列你兩台 dispatcher 的 EIP：

```conf
# 只允許 relay 到我自己的 dispatcher 兩台
allowed-peer-ip=57.181.45.231-57.181.45.231
allowed-peer-ip=35.72.149.122-35.72.149.122
# 順便把 denied 全刪掉（不然會跟 allow 衝突）
```

---

## 5. 建立 log 目錄 + 啟動

```bash
sudo mkdir -p /var/log/coturn
sudo chown turnserver:turnserver /var/log/coturn

# 啟用 + 啟動
sudo systemctl enable coturn
sudo systemctl restart coturn

# 確認跑起來
sudo systemctl status coturn
# 應該看到 "Active: active (running)"

# 如果是 failed，看詳細錯誤：
sudo journalctl -u coturn -n 100
```

常見啟動失敗原因：
- `external-ip` 格式錯（用 / 不是 \）
- port 被佔用（用 `sudo ss -ulnp | grep 3478` 確認）
- log 目錄沒權限

---

## 6. AWS Security Group 開 port

到 AWS Console → EC2 → 找 Signaling EC2 → Security Group → Edit inbound rules → 加兩條：

| Type | Protocol | Port range | Source | Description |
|---|---|---|---|---|
| Custom UDP | UDP | 3478 | 0.0.0.0/0 | coturn STUN/TURN |
| Custom UDP | UDP | 49152-65535 | 0.0.0.0/0 | coturn relay |

**也可以加 TCP fallback（行動網路 UDP 受限時用 TCP TURN）：**

| Custom TCP | TCP | 3478 | 0.0.0.0/0 | coturn TCP |

---

## 7. 在 EC2 上自我測試 coturn

```bash
# 安裝 turnutils（測試工具）
sudo apt install -y coturn-utils

# 對本機 coturn 做 TURN allocation 測試
turnutils_uclient -u mawatari -w 你的密碼 -p 3478 18.178.31.155

# 看到類似以下輸出代表 OK：
# 0: IPv4. Connected from: 127.0.0.1:xxxxx
# 0: IPv4. Connected to: 18.178.31.155:3478
# 0: ...
# tot_send_msgs=5, tot_recv_msgs=5    ← 一收一回成功
```

如果看到：
- `401: Unauthorized` → 帳密寫錯，回去檢查 `/etc/turnserver.conf` 的 `user=` 行
- `connection refused` → coturn 沒跑或 port 不對
- `timeout` → SG 沒開 port

---

## 8. 從 Pi 測試 TURN allocation

```bash
# 在 Pi 上
sudo apt install -y coturn-utils
turnutils_uclient -u mawatari -w 你的密碼 -p 3478 18.178.31.155

# 同樣看到 tot_send_msgs / tot_recv_msgs > 0 才算成功
# 如果這個成功，dispatcher 也應該成功（同個網路類型）
```

---

## 9. 改 yaml 用自家 coturn

編輯 `config/staging.yaml` 跟 `config/staging_video.yaml`，把 ExpressTURN 那組換掉：

```yaml
# edge.ice_servers
ice_servers:
  - urls: "stun:stun.l.google.com:19302"
  - urls: "turn:18.178.31.155:3478"
    username: "mawatari"
    credential: "你的密碼"

# dispatchers[0].ice_servers 跟 dispatchers[1].ice_servers 一樣
```

**三個 ice_servers 區塊都要改**（edge + 2 個 dispatcher）。

---

## 10. 部署 yaml + 重啟 process

```powershell
# Windows 開發機
.\scripts\deploy_configs.ps1
```

各機器重啟（順序 dispatcher × 2 → edge）：

```bash
# Dispatcher 001 EC2
ssh ubuntu@57.181.45.231
screen -r dispatcher    # 或對應名稱
# Ctrl+C → 重跑
python dispatcher/main.py --config config/staging.yaml --id dispatcher-ec2-001

# Dispatcher 002 同理（換 EIP 跟 --id）

# Pi edge
ssh mawatarilab@100.98.101.5
cd ~/Project_Edge
python edge/main.py --config config/staging_video.yaml --metrics
```

---

## 11. 驗證成功

### Dispatcher log 應該看到：

```
ICE servers 設定 (2 個): ['stun:stun.l.google.com:19302', 'turn:18.178.31.155:3478']
本地 ICE candidates: host=N, srflx(STUN)=1, relay(TURN)=1   ← ★ relay=1 = 自家 coturn 通了
✓ Data Channel OPEN: edge=edge-rpi-001
← 收到 frame: edge=edge-rpi-001, count=0
→ 已轉發到 Inference: edge=edge-rpi-001, count=0
↑ 已回傳結果給 Edge: edge=edge-rpi-001, count=0
```

### Edge log 應該看到：

```
收到結果: frame=xxx, seq=N, prediction=29.xxxx
```

### coturn 自己的 log

```bash
sudo tail -f /var/log/coturn/turn.log
```

正常運作時會滾很多 `session 1: usage: ...` 跟 `relay: ...` 訊息。

---

## 12. 排錯 Cheat Sheet

| 症狀 | 可能原因 | 處理 |
|---|---|---|
| `systemctl status coturn` 顯示 failed | 設定檔語法錯 | `sudo turnserver -c /etc/turnserver.conf -v` 手動跑，看錯誤訊息 |
| `turnutils_uclient` 回 401 Unauthorized | `user=` 帳密錯 | 改 `/etc/turnserver.conf` 後 `sudo systemctl restart coturn` |
| `turnutils_uclient` 連線 timeout | SG 沒開 UDP 3478 | 回 AWS Console 加規則 |
| dispatcher 印 `relay(TURN)=0` + warning | yaml 沒讀到 TURN 或 TURN 連不到 | 看 dispatcher 啟動時 `ICE servers 設定` 那行印幾個 |
| relay=1 但還是沒 frame | coturn `denied-peer-ip` 把 dispatcher IP 擋了 | 改 `allowed-peer-ip` 或暫時拿掉所有 denied/allowed |
| 一段時間後突然不通 | EC2 OOM / coturn 自己掛 | `sudo systemctl restart coturn`；長期可加 systemd watchdog |

---

## 13. 定期維護（每個月看一次）

```bash
# 檢查 coturn 還在跑
sudo systemctl status coturn

# 檢查 log 大小（verbose 開很久 log 會肥）
ls -lh /var/log/coturn/

# 如果 log 太大：
sudo systemctl stop coturn
sudo truncate -s 0 /var/log/coturn/turn.log
sudo systemctl start coturn

# 穩定後把 /etc/turnserver.conf 的 verbose 註解掉，改成 simple-log
```

---

## 14. 如果你想換密碼

```bash
sudo nano /etc/turnserver.conf
# 修改 user= 那行
sudo systemctl restart coturn

# 同步改 staging*.yaml 的 credential
# Windows 開發機跑：
.\scripts\deploy_configs.ps1

# 重啟 edge + dispatcher
```

---

## 15. 進階：要做 SSL/TLS (TURNS) 的話

短期 demo 不需要。如果上正式環境要用 `turns://`：
1. 申請 domain（例如 `turn.yourdomain.com`）指向 EIP
2. 用 certbot 拿 Let's Encrypt 憑證
3. 在 `/etc/turnserver.conf` 加：
   ```
   tls-listening-port=5349
   cert=/etc/letsencrypt/live/turn.yourdomain.com/fullchain.pem
   pkey=/etc/letsencrypt/live/turn.yourdomain.com/privkey.pem
   ```
4. yaml 改用 `turns:turn.yourdomain.com:5349?transport=tcp`

但 5G 行動網路上純 UDP + IP-based TURN 已經夠用，先不折騰 TLS。

---

## 完成 Checklist

- [ ] `apt install coturn` 成功
- [ ] `/etc/default/coturn` 把 `TURNSERVER_ENABLED=1` 啟用
- [ ] `/etc/turnserver.conf` 寫好（external-ip + user 改成你的）
- [ ] `sudo systemctl restart coturn` 後 status 是 active (running)
- [ ] AWS SG 開了 UDP 3478 + UDP 49152-65535
- [ ] `turnutils_uclient` 在 EC2 上測試成功
- [ ] 從 Pi 用 `turnutils_uclient` 測試也成功
- [ ] staging.yaml 跟 staging_video.yaml 改成自家 coturn
- [ ] `deploy_configs.ps1` 推 yaml 到所有機器
- [ ] 重啟 dispatcher × 2 + edge
- [ ] dispatcher log 看到 `relay(TURN)=1`
- [ ] edge log 看到 prediction 回來

照這份做完，**永遠不用再看別人 TURN 服務臉色**。
