"""快速測試腳本：用 edge/video/ 裡的影片跑完整推論管線。

此腳本會啟動五個元件模擬完整的分散式架構：
  1. Signaling Server  (port 8080)
  2. Inference Server  (port 8765)
  3. Dispatcher 001
  4. Dispatcher 002
  5. Edge（讀取指定影片）

流程驗證：
  Edge 讀取影片 → 壓縮 JPEG → WebRTC → Dispatcher → WebSocket → Inference
  Inference → 回傳結果 → Dispatcher → Edge

使用方式（在專案根目錄執行）：
  python quick_test/run.py                              # dummy model + 30.mp4
  python quick_test/run.py --video edge/video/40.mp4    # 換影片
  python quick_test/run.py --model output_model_v1.h5   # 用 Keras .h5 模型
  python quick_test/run.py --model output_model_v1.h5 --video edge/video/40.mp4
  python quick_test/run.py --duration 30                # 30 秒後自動停止

按 Ctrl+C 可隨時停止所有元件。
"""

import argparse
import os
import subprocess
import sys
import time

# ── Windows UTF-8 修正 ──────────────────────────────────────
# Windows 預設 terminal 編碼（cp932/cp950）無法正確顯示 UTF-8 中文，
# 強制切換為 UTF-8 輸出。
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

# 專案根目錄（此腳本位於 quick_test/，向上一層）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_CONFIG = os.path.join(PROJECT_ROOT, "quick_test", "_runtime.yaml")

# 各模型類型對應的 YAML 模板
_TEMPLATE_DUMMY = os.path.join(PROJECT_ROOT, "quick_test", "video_test.yaml")
_TEMPLATE_KERAS = os.path.join(PROJECT_ROOT, "quick_test", "h5_test.yaml")


# ── 輔助函式 ────────────────────────────────────────────────

def log(tag: str, msg: str):
    """統一格式的日誌輸出。"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {msg}", flush=True)


def free_ports(ports: list):
    """釋放可能被上次測試殘留 process 佔用的 port（僅 Windows）。"""
    if sys.platform != "win32":
        return
    for port in ports:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != os.getpid():
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5,
                        )
                        log("準備", f"已釋放 port {port} (PID={pid})")
        except Exception:
            pass  # 釋放失敗不影響啟動


def make_runtime_config(video_path: str, model_path: str) -> str:
    """根據模板生成含有正確影片和模型路徑的暫存設定檔。

    Args:
        video_path: 影片檔路徑（相對於專案根目錄，或絕對路徑）
        model_path: 模型路徑，空字串代表使用 dummy；否則使用 Keras .h5

    Returns:
        暫存設定檔的路徑
    """
    # 選擇對應模板
    template = _TEMPLATE_KERAS if model_path else _TEMPLATE_DUMMY

    with open(template, "r", encoding="utf-8") as f:
        content = f.read()

    # 統一用正斜線（YAML 相容 Windows 路徑）
    norm_video = video_path.replace("\\", "/")
    norm_model = model_path.replace("\\", "/") if model_path else ""

    # 替換影片路徑（兩個模板都有這行）
    content = content.replace(
        'source: "edge/video/30.mp4"',
        f'source: "{norm_video}"',
    )

    # Keras 模板：替換模型路徑
    if norm_model:
        content = content.replace(
            'model_path: "output_model_v1.h5"',
            f'model_path: "{norm_model}"',
        )

    with open(TEMP_CONFIG, "w", encoding="utf-8") as f:
        f.write(content)

    return TEMP_CONFIG


def start_component(name: str, args: list) -> subprocess.Popen:
    """啟動單一元件 process。

    Args:
        name: 元件名稱（用於日誌顯示）
        args: 傳給 Python 的命令列參數

    Returns:
        subprocess.Popen 物件
    """
    env = os.environ.copy()
    # 將專案根目錄加入 PYTHONPATH，確保 shared/ 等模組可被 import
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # 強制 UTF-8，避免子 process 的中文 log 亂碼
    env["PYTHONUTF8"] = "1"
    # 抑制 TensorFlow 的 INFO log（只顯示 WARNING 以上）
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    proc = subprocess.Popen(
        [sys.executable] + args,
        cwd=PROJECT_ROOT,
        env=env,
        # 子 process 的 stdout/stderr 直接輸出到 terminal，讓用戶看到各元件 log
        stdout=None,
        stderr=None,
    )
    log(name, f"已啟動 (PID={proc.pid})")
    return proc


def check_video(video_path: str):
    """確認影片檔存在。

    Raises:
        SystemExit: 影片不存在時退出並提示可用清單
    """
    full_path = (
        os.path.join(PROJECT_ROOT, video_path)
        if not os.path.isabs(video_path)
        else video_path
    )
    if not os.path.exists(full_path):
        log("錯誤", f"找不到影片檔：{full_path}")
        video_dir = os.path.join(PROJECT_ROOT, "edge", "video")
        if os.path.isdir(video_dir):
            log("提示", "可用的影片：")
            for f in sorted(os.listdir(video_dir)):
                if f.endswith(".mp4"):
                    log("提示", f"  edge/video/{f}")
        sys.exit(1)
    log("準備", f"影片確認：{full_path}")


def check_model(model_path: str):
    """確認模型檔存在（model_path 為空時跳過）。

    Raises:
        SystemExit: 檔案不存在時退出並提示可用清單
    """
    if not model_path:
        return  # 使用 dummy，不需要檔案
    full_path = (
        os.path.join(PROJECT_ROOT, model_path)
        if not os.path.isabs(model_path)
        else model_path
    )
    if not os.path.exists(full_path):
        log("錯誤", f"找不到模型檔：{full_path}")
        log("提示", "可用的 .h5 模型：")
        for f in sorted(os.listdir(PROJECT_ROOT)):
            if f.endswith(".h5"):
                log("提示", f"  {f}")
        sys.exit(1)
    log("準備", f"模型確認：{full_path}")


# ── 主程式 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="快速測試：用自己的影片跑完整推論管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python quick_test/run.py
  python quick_test/run.py --video edge/video/40.mp4
  python quick_test/run.py --model output_model_v1.h5
  python quick_test/run.py --model output_model_v1_0.25.h5 --video edge/video/50.mp4
  python quick_test/run.py --duration 30
        """,
    )
    parser.add_argument(
        "--video",
        default="edge/video/30.mp4",
        help="影片檔路徑（相對於專案根目錄，預設：edge/video/30.mp4）",
    )
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Keras .h5 模型路徑（相對於專案根目錄）。"
            "留空時使用 dummy model（預設）。"
            "範例：output_model_v1.h5"
        ),
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="自動停止的秒數（0 = 不自動停止，需手動 Ctrl+C）",
    )
    args = parser.parse_args()

    # 決定顯示的模型名稱
    model_label = args.model if args.model else "dummy（假偵測結果，不需 GPU）"

    # ── 顯示啟動資訊 ──
    print()
    print("=" * 60)
    print("  分散式推論管線 — 快速測試")
    print("=" * 60)
    print(f"  影片來源: {args.video}")
    print(f"  推論模型: {model_label}")
    if args.duration > 0:
        print(f"  自動停止: {args.duration} 秒後")
    else:
        print(f"  停止方式: Ctrl+C")
    print("=" * 60)
    print()

    # ── 前置檢查 ──
    check_video(args.video)
    check_model(args.model)
    free_ports([8080, 8765])  # 清除可能殘留的上次測試 process

    # 生成含正確影片 / 模型路徑的暫存設定檔
    cfg = make_runtime_config(args.video, args.model)
    log("準備", f"設定檔已生成：{os.path.basename(cfg)}")

    # Keras 模型首次載入需要時間，給 Inference 更多等待時間
    inference_wait = 8.0 if args.model else 1.5

    if args.model:
        log("提示", f"Keras 模型首次載入需要約 {inference_wait:.0f} 秒，請稍候...")
    print()

    processes = []

    try:
        # ── 依序啟動各元件 ──
        # 啟動順序很重要：Signaling → Inference → Dispatcher × 2 → Edge

        print("[1/5] 啟動 Signaling Server（協調 WebRTC 連線，port 8080）")
        processes.append(("Signaling", start_component("signaling", [
            "signaling/server.py", "--config", cfg,
        ])))
        time.sleep(1.5)  # 等待 aiohttp 啟動

        print()
        print(f"[2/5] 啟動 Inference Server（{model_label}，port 8765）")
        processes.append(("Inference", start_component("inference", [
            "inference/main.py", "--config", cfg,
        ])))
        # Keras 模型載入慢（需要初始化 TF）；dummy 很快
        time.sleep(inference_wait)

        print()
        print("[3/5] 啟動 Dispatcher 001（主要轉發節點）")
        processes.append(("Disp-001", start_component("dispatcher-001", [
            "dispatcher/main.py", "--config", cfg, "--id", "dispatcher-001",
        ])))
        time.sleep(1.0)

        print()
        print("[4/5] 啟動 Dispatcher 002（備用轉發節點）")
        processes.append(("Disp-002", start_component("dispatcher-002", [
            "dispatcher/main.py", "--config", cfg, "--id", "dispatcher-002",
        ])))
        time.sleep(1.0)

        print()
        print("[5/5] 啟動 Edge（讀取影片，建立 WebRTC 連線）")
        processes.append(("Edge", start_component("edge", [
            "edge/main.py", "--config", cfg,
        ])))

        print()
        print("=" * 60)
        print("  所有元件已啟動！")
        print()
        print("  觀察重點：")
        print("  - Signaling log：「listening on port 8080」")
        print("  - Inference log：「推論伺服器啟動」或「Keras 模型已載入」")
        print("  - Dispatcher log：「connected to inference」")
        print("  - Edge log：「WebRTC connected」= 連線成功")
        print("  - Edge log：「result received」= ★ 完整管線通了！★")
        if args.model:
            print()
            print("  Keras 模型輸出格式：")
            print("    {\"prediction\": <float>}  ← 模型的回歸預測值")
        if args.duration > 0:
            print(f"  - 將於 {args.duration} 秒後自動停止")
        else:
            print("  - 按 Ctrl+C 停止")
        print("=" * 60)
        print()

        # ── 等待直到超時或用戶中斷 ──
        start_time = time.time()
        while True:
            # 檢查是否超過指定執行時間
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                log("系統", f"已達指定執行時間 {args.duration} 秒，自動停止")
                break

            # 檢查關鍵元件是否意外退出
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    log("警告", f"[{name}] 已退出 (code={ret})")
                    if name in ("Signaling", "Inference"):
                        log("錯誤", f"關鍵元件 [{name}] 異常退出，停止所有元件")
                        raise KeyboardInterrupt

            time.sleep(1)

    except KeyboardInterrupt:
        print()
        log("系統", "正在關閉所有元件...")

    finally:
        # ── 反向順序優雅關閉 ──
        for name, proc in reversed(processes):
            if proc.poll() is None:  # 仍在執行中
                log(name, "正在停止...")
                proc.terminate()

        # 等待所有 process 結束
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
                log(name, "已停止")
            except subprocess.TimeoutExpired:
                log(name, "未在 5 秒內結束，強制 kill")
                proc.kill()

        # 清除暫存設定檔
        if os.path.exists(TEMP_CONFIG):
            os.remove(TEMP_CONFIG)

        print()
        print("=" * 60)
        print("  全部元件已關閉。")
        print("=" * 60)


if __name__ == "__main__":
    main()
