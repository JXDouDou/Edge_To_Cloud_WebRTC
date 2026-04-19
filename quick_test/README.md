# Quick Test — 快速測試

用自己的影片跑完整推論管線（Edge → Dispatcher → Inference → 結果回傳）。

## 使用前提

確認已安裝依賴套件：

```bash
pip install -r requirements.txt
```

## 一鍵執行

**在專案根目錄執行：**

```bash
# 預設使用 edge/video/30.mp4
python quick_test/run.py

# 指定其他影片
python quick_test/run.py --video edge/video/40.mp4
python quick_test/run.py --video edge/video/50.mp4

# 跑 30 秒後自動停止（不需要手動 Ctrl+C）
python quick_test/run.py --duration 30
```

## 流程說明

```
edge/video/30.mp4
      ↓
  Edge 讀取影片
  → 壓縮成 JPEG（5 fps）
  → WebRTC DataChannel
      ↓
  Dispatcher（EC2 模擬）
  → 注入 edge_id
  → WebSocket
      ↓
  Inference Server（dummy model）
  → 回傳假偵測結果 {"class":"dummy", "confidence":0.99}
      ↓
  Dispatcher → Edge（印出結果）
```

## 觀察重點

終端機會同時顯示五個元件的 log，觀察以下訊息確認管線正常：

| 元件 | 成功訊息 |
|------|---------|
| Signaling | `listening on port 8080` |
| Inference | `listening on port 8765` |
| Dispatcher | `connected to inference` |
| Edge | `WebRTC connected` |
| Edge | `result received` ← **這行代表完整管線通了** |

## 可用的影片

```
edge/video/20.mp4
edge/video/30.mp4   ← 預設
edge/video/40.mp4
edge/video/50.mp4
```

## .h5 模型（目前未使用）

以下模型檔已放在專案根目錄，待確認載入方式後再整合：

```
output_model_v1.h5
output_model_v1_0.25.h5
output_model_v1_0.25_ori.h5
```

目前測試統一使用 `dummy` model，確認管線通暢後再換成真實模型。

## 常見問題

**Port 已被佔用**：腳本會自動嘗試釋放 8080/8765，若仍失敗請重開 terminal。

**中文亂碼**：腳本已設定 `PYTHONUTF8=1`，若仍有問題可在執行前設定環境變數：
```bash
set PYTHONUTF8=1
python quick_test/run.py
```
