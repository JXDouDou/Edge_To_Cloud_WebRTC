"""一鍵本機測試：啟動全部元件，驗證完整資料流。

此腳本依序啟動所有元件（各為獨立 process），模擬完整的分散式架構：
  1. Signaling Server  (port 8080)
  2. Inference Server   (port 8765)
  3. Dispatcher 001     (WebRTC answerer)
  4. Dispatcher 002     (WebRTC answerer，備用)
  5. Edge               (WebRTC offerer，讀取影片)

啟動順序很重要：
  - Signaling 必須最先啟動（其他元件要向它註冊）
  - Inference 在 Dispatcher 之前啟動（Dispatcher 啟動時會連 Inference）
  - Dispatcher 在 Edge 之前啟動（Edge 啟動時會向 Signaling 請求 Dispatcher 列表）
  - Edge 最後啟動（需要前面的元件都就緒）

前置步驟：
  1. pip install -r requirements.txt
  2. python test_data/generate_test_video.py  （產生測試影片）
  3. python run_local_test.py

按 Ctrl+C 停止所有元件。

使用方式：
    python run_local_test.py                           # 使用預設 test.yaml
    python run_local_test.py --config config/test.yaml # 指定設定檔
"""

import argparse
import subprocess
import sys
import time
import os

# Windows 上 CP932 / CP936 terminal 無法顯示中文字元，強制 UTF-8 輸出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

# 專案根目錄（此腳本所在位置）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def start_component(name: str, args: list) -> subprocess.Popen:
    """啟動單一元件 process。

    每個元件都是一個獨立的 Python process，
    使用與父 process 相同的 Python 解釋器。

    Args:
        name: 元件名稱（用於日誌顯示）
        args: 傳給 Python 的命令列參數（如 ["edge/main.py", "--config", "..."]）

    Returns:
        啟動的 subprocess.Popen 物件
    """
    env = os.environ.copy()
    # 將專案根目錄加入 PYTHONPATH，確保 shared/ 等模組可被 import
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # 確保子 process 用 UTF-8 輸出（Windows 預設 cp932 無法顯示中文 log）
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [sys.executable] + args,
        cwd=PROJECT_ROOT,
        env=env,
    )
    print(f"  [{name}] PID={proc.pid}")
    return proc


def check_test_video(config_path: str):
    """檢查測試影片是否存在，不存在則自動產生。

    Args:
        config_path: 設定檔路徑（用於讀取影片路徑設定）
    """
    # 簡易檢查預設路徑
    video_path = os.path.join(PROJECT_ROOT, "test_data", "test_video.mp4")
    if not os.path.exists(video_path):
        print("[準備] 測試影片不存在，自動產生中...")
        gen_script = os.path.join(PROJECT_ROOT, "test_data", "generate_test_video.py")
        gen_env = os.environ.copy()
        gen_env["PYTHONUTF8"] = "1"
        subprocess.run([sys.executable, gen_script], cwd=PROJECT_ROOT, check=True, env=gen_env)
        print()


def free_ports(ports: list):
    """強制釋放指定 port（殺掉佔用的 process）。

    僅在 Windows 上執行；Linux/Mac 請手動 kill。
    避免上次測試殘留的 process 導致 port 衝突。

    Args:
        ports: 要釋放的 port 號列表
    """
    if sys.platform != "win32":
        return
    for port in ports:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                # 找出 LISTENING 狀態且符合 port 的行
                if f":{port} " in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5,
                        )
                        print(f"[準備] 釋放 port {port} (PID={pid})")
        except Exception:
            pass  # 釋放失敗不影響啟動（啟動時自然會報 bind 錯誤）


def main():
    """主函式：依序啟動所有元件，等待使用者中斷後優雅關閉。"""

    parser = argparse.ArgumentParser(description="本機測試：啟動全部元件")
    parser.add_argument("--config", default="config/test.yaml", help="YAML 設定檔路徑")
    args = parser.parse_args()
    cfg = args.config

    # ── 前置檢查 ──
    free_ports([8080, 8765])  # 清除可能殘留的上次測試 process
    check_test_video(cfg)

    processes = []

    print("=" * 60)
    print(" 分散式推論系統 — 本機測試")
    print("=" * 60)
    print()

    # ── 依序啟動各元件 ──
    # 每個元件之間等待一小段時間，確保前一個已經就緒

    # 1) Signaling Server — 必須第一個啟動
    print("[1/5] 啟動 Signaling Server (port 8080)")
    processes.append(("signaling", start_component("signaling", [
        "signaling/server.py", "--config", cfg,
    ])))
    time.sleep(1.5)  # 等待 aiohttp 啟動完成

    # 2) Inference Server — Dispatcher 連線前需要就緒
    print("[2/5] 啟動 Inference Server (port 8765)")
    processes.append(("inference", start_component("inference", [
        "inference/main.py", "--config", cfg,
    ])))
    time.sleep(1.5)

    # 3) Dispatcher 001 — 向 Signaling 註冊 + 連接 Inference
    print("[3/5] 啟動 Dispatcher 001")
    processes.append(("dispatcher-001", start_component("dispatcher-001", [
        "dispatcher/main.py", "--config", cfg, "--id", "dispatcher-001",
    ])))
    time.sleep(1.0)

    # 4) Dispatcher 002 — 備用 Dispatcher
    print("[4/5] 啟動 Dispatcher 002")
    processes.append(("dispatcher-002", start_component("dispatcher-002", [
        "dispatcher/main.py", "--config", cfg, "--id", "dispatcher-002",
    ])))
    time.sleep(1.0)

    # 5) Edge — 最後啟動，向 Signaling 請求 Dispatcher 列表並建立 WebRTC
    print("[5/5] 啟動 Edge")
    processes.append(("edge", start_component("edge", [
        "edge/main.py", "--config", cfg,
    ])))

    print()
    print("=" * 60)
    print(" 所有元件已啟動！按 Ctrl+C 停止")
    print("=" * 60)
    print()

    # ── 等待直到使用者中斷或某個 process 異常退出 ──
    try:
        while True:
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[WARN] [{name}] 已退出 (code={ret})")
                    # 關鍵元件死亡 → 停止整個系統
                    if name in ("signaling", "inference"):
                        print(f"[ERROR] 關鍵元件 [{name}] 已退出，停止所有元件...")
                        raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n正在關閉所有元件...")

        # 反向順序關閉（先關 Edge，最後關 Signaling）
        for name, proc in reversed(processes):
            print(f"  停止 [{name}]...")
            proc.terminate()

        # 等待所有 process 結束
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  [{name}] 未在 5 秒內結束，強制 kill")
                proc.kill()

        print("\n全部元件已關閉。")


if __name__ == "__main__":
    main()
