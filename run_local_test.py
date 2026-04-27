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

模型切換邏輯（與 quick_test/run.py 一致）：
  - 不指定 --model：使用 config/test.yaml（dummy 假模型，ROI 關閉）
  - 指定 --model XXX.h5：使用 config/test_h5.yaml（Keras 真模型，ROI 開啟）
  動態產生的 _runtime.yaml 會置換影片 / 模型路徑

使用方式：
    python run_local_test.py                                              # dummy + 預設影片
    python run_local_test.py --video edge/video/30.mp4                    # dummy + 換影片
    python run_local_test.py --model output_model_v1_0.25_ori.h5          # Keras + 預設影片
    python run_local_test.py --model xxx.h5 --video edge/video/40.mp4     # Keras + 換影片
    python run_local_test.py --duration 30                                # 30 秒後自動停止
    python run_local_test.py --config config/test.yaml                    # 直接指定 config（跳過模板替換）

按 Ctrl+C 隨時停止所有元件。
"""

import argparse
import os
import subprocess
import sys
import time

# Windows 上 CP932 / CP936 terminal 無法顯示中文字元，強制 UTF-8 輸出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback

# 專案根目錄（此腳本所在位置）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模板與動態 config
TEMPLATE_DUMMY = os.path.join(PROJECT_ROOT, "config", "test.yaml")
TEMPLATE_KERAS = os.path.join(PROJECT_ROOT, "config", "test_h5.yaml")
TEMP_CONFIG = os.path.join(PROJECT_ROOT, "config", "_runtime.yaml")


# ── 輔助函式 ────────────────────────────────────────────────

def log(tag: str, msg: str):
    """統一格式的日誌輸出。"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {msg}", flush=True)


def start_component(name: str, args: list) -> subprocess.Popen:
    """啟動單一元件 process。

    Args:
        name: 元件名稱（用於日誌顯示）
        args: 傳給 Python 的命令列參數

    Returns:
        啟動的 subprocess.Popen 物件
    """
    env = os.environ.copy()
    # 將專案根目錄加入 PYTHONPATH，確保 shared/ 等模組可被 import
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # 強制 UTF-8，避免子 process 的中文 log 在 Windows 上亂碼
    env["PYTHONUTF8"] = "1"
    # 抑制 TensorFlow 啟動噪音（只顯示 WARNING 以上）
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    proc = subprocess.Popen(
        [sys.executable] + args,
        cwd=PROJECT_ROOT,
        env=env,
        # 子 process 的 stdout/stderr 直接輸出到 terminal
        stdout=None,
        stderr=None,
    )
    log(name, f"已啟動 (PID={proc.pid})")
    return proc


def free_ports(ports: list):
    """強制釋放指定 port（殺掉佔用的 process）。

    避免上次測試殘留的 process 導致 port 衝突。
    僅在 Windows 上執行；Linux/Mac 請手動 kill。
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
            pass  # 釋放失敗不影響啟動（啟動時自然會報 bind 錯誤）


def make_runtime_config(template_path: str, video_path: str, model_path: str) -> str:
    """根據模板生成含正確影片 / 模型路徑的暫存設定檔。

    Args:
        template_path: 模板 yaml 路徑
        video_path:    影片檔路徑（相對於專案根目錄，或絕對路徑）
        model_path:    模型路徑；空字串代表 dummy

    Returns:
        暫存設定檔的路徑
    """
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 統一用正斜線（YAML 在 Windows 路徑相容性更佳）
    norm_video = video_path.replace("\\", "/")
    norm_model = model_path.replace("\\", "/") if model_path else ""

    # 替換影片路徑（兩個模板的 capture.source 都有）
    # 為避免誤替換，用模板原本確切的字串做精準替換
    if 'source: "edge/video/30.mp4"' in content:
        content = content.replace(
            'source: "edge/video/30.mp4"',
            f'source: "{norm_video}"',
        )
    elif 'source: "Presentation.mp4"' in content:
        content = content.replace(
            'source: "Presentation.mp4"',
            f'source: "{norm_video}"',
        )

    # Keras 模板：替換模型路徑
    if norm_model and 'model_path: "output_model_v1_0.25_ori.h5"' in content:
        content = content.replace(
            'model_path: "output_model_v1_0.25_ori.h5"',
            f'model_path: "{norm_model}"',
        )

    with open(TEMP_CONFIG, "w", encoding="utf-8") as f:
        f.write(content)

    return TEMP_CONFIG


def check_video(video_path: str, allow_autogen: bool):
    """確認影片檔存在；若是預設測試影片且不存在，可自動產生。

    Args:
        video_path:    影片路徑（相對或絕對）
        allow_autogen: True 時，若是預設 test_data/test_video.mp4 缺失會自動產生

    Raises:
        SystemExit: 影片不存在且無法自動產生
    """
    full_path = (
        os.path.join(PROJECT_ROOT, video_path)
        if not os.path.isabs(video_path)
        else video_path
    )
    if os.path.exists(full_path):
        log("準備", f"影片確認：{video_path}")
        return

    # 預設測試影片缺失 → 嘗試自動產生
    if allow_autogen and video_path.endswith("test_video.mp4"):
        log("準備", "測試影片不存在，自動產生中...")
        gen_script = os.path.join(PROJECT_ROOT, "test_data", "generate_test_video.py")
        if os.path.exists(gen_script):
            gen_env = os.environ.copy()
            gen_env["PYTHONUTF8"] = "1"
            subprocess.run(
                [sys.executable, gen_script],
                cwd=PROJECT_ROOT, check=True, env=gen_env,
            )
            log("準備", f"影片已產生：{video_path}")
            return

    # 找不到也產生不出來 → 退出並提示可用清單
    log("錯誤", f"找不到影片檔：{full_path}")
    video_dir = os.path.join(PROJECT_ROOT, "edge", "video")
    if os.path.isdir(video_dir):
        log("提示", "edge/video/ 內可用影片：")
        for f in sorted(os.listdir(video_dir)):
            if f.endswith(".mp4"):
                log("提示", f"  edge/video/{f}")
    sys.exit(1)


def check_model(model_path: str):
    """確認模型檔存在（model_path 為空時跳過）。"""
    if not model_path:
        return
    full_path = (
        os.path.join(PROJECT_ROOT, model_path)
        if not os.path.isabs(model_path)
        else model_path
    )
    if not os.path.exists(full_path):
        log("錯誤", f"找不到模型檔：{full_path}")
        log("提示", "專案根目錄下可用的 .h5 模型：")
        for f in sorted(os.listdir(PROJECT_ROOT)):
            if f.endswith(".h5"):
                log("提示", f"  {f}")
        sys.exit(1)
    log("準備", f"模型確認：{model_path}")


# ── 主程式 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="本機測試：一鍵啟動全部元件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python run_local_test.py
  python run_local_test.py --video edge/video/30.mp4
  python run_local_test.py --model output_model_v1_0.25_ori.h5
  python run_local_test.py --model output_model_v1_0.25_ori.h5 --video edge/video/40.mp4
  python run_local_test.py --duration 30
  python run_local_test.py --config config/test.yaml             # 直接指定，不做動態替換
        """,
    )
    parser.add_argument(
        "--video",
        default="",
        help=(
            "影片檔路徑（相對於專案根目錄）。"
            "預設：dummy 模式用 test_data/test_video.mp4，keras 模式用 edge/video/30.mp4"
        ),
    )
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Keras .h5 模型路徑（相對於專案根目錄）。"
            "留空時使用 dummy 假模型（不需 GPU）。"
            "範例：output_model_v1_0.25_ori.h5"
        ),
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="自動停止的秒數（0 = 不自動停止，需 Ctrl+C）",
    )
    parser.add_argument(
        "--config",
        default="",
        help=(
            "直接指定 YAML 設定檔（指定後會跳過模板替換邏輯，"
            "也就不再使用 --video / --model）"
        ),
    )
    args = parser.parse_args()

    # ── 決定使用哪個 config ──
    if args.config:
        # 直接指定 → 跳過模板替換
        cfg = args.config
        model_label = "（依 config 內容）"
        video_label = "（依 config 內容）"
        is_keras = False  # 無法得知，保守用 dummy 等待時間
        skip_check = True
    else:
        # 用模板動態替換
        is_keras = bool(args.model)
        template = TEMPLATE_KERAS if is_keras else TEMPLATE_DUMMY
        # 預設影片：dummy → test_data/test_video.mp4 (向後相容)
        #          keras → edge/video/30.mp4
        if args.video:
            video_path = args.video
        else:
            video_path = "edge/video/30.mp4" if is_keras else "test_data/test_video.mp4"

        model_label = args.model if args.model else "dummy（假偵測結果，不需 GPU）"
        video_label = video_path
        skip_check = False

    # ── 顯示啟動資訊 ──
    print()
    print("=" * 60)
    print("  分散式推論系統 — 本機測試")
    print("=" * 60)
    print(f"  影片來源: {video_label}")
    print(f"  推論模型: {model_label}")
    if args.duration > 0:
        print(f"  自動停止: {args.duration} 秒後")
    else:
        print(f"  停止方式: Ctrl+C")
    print("=" * 60)
    print()

    # ── 前置檢查 ──
    free_ports([8080, 8765])

    if not skip_check:
        # dummy 模式 + 預設影片時，允許自動產生 test_video.mp4
        allow_autogen = (not is_keras) and (not args.video)
        check_video(video_path, allow_autogen=allow_autogen)
        check_model(args.model)

        # 動態產生 _runtime.yaml
        cfg = make_runtime_config(template, video_path, args.model)
        log("準備", f"設定檔已生成：{os.path.relpath(cfg, PROJECT_ROOT)}")

    # Keras 首次載入需要時間
    inference_wait = 8.0 if is_keras else 1.5
    if is_keras:
        log("提示", f"Keras 模型首次載入需要約 {inference_wait:.0f} 秒，請稍候...")
    print()

    processes = []

    try:
        # ── 依序啟動 ──
        print("[1/5] 啟動 Signaling Server（協調 WebRTC，port 8080）")
        processes.append(("Signaling", start_component("signaling", [
            "signaling/server.py", "--config", cfg,
        ])))
        time.sleep(1.5)

        print()
        print(f"[2/5] 啟動 Inference Server（{model_label}，port 8765）")
        processes.append(("Inference", start_component("inference", [
            "inference/main.py", "--config", cfg,
        ])))
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
        if is_keras:
            print()
            print("  Keras 模型輸出格式：")
            print("    {\"prediction\": <float>}  ← 模型的回歸預測值")
        if args.duration > 0:
            print(f"  - 將於 {args.duration} 秒後自動停止")
        else:
            print("  - 按 Ctrl+C 停止")
        print("=" * 60)
        print()

        # ── 等待中斷或 timeout ──
        start_time = time.time()
        while True:
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                log("系統", f"已達指定執行時間 {args.duration} 秒，自動停止")
                break

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
            if proc.poll() is None:
                log(name, "正在停止...")
                proc.terminate()

        for name, proc in processes:
            try:
                proc.wait(timeout=5)
                log(name, "已停止")
            except subprocess.TimeoutExpired:
                log(name, "未在 5 秒內結束，強制 kill")
                proc.kill()

        # 清除動態產生的 _runtime.yaml（保留使用者直接指定的 --config）
        if not skip_check and os.path.exists(TEMP_CONFIG):
            try:
                os.remove(TEMP_CONFIG)
            except OSError:
                pass

        print()
        print("=" * 60)
        print("  全部元件已關閉。")
        print("=" * 60)


if __name__ == "__main__":
    main()
