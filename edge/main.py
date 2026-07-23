"""Edge 裝置入口程式。

此程式是 Edge 端的主要入口，負責組裝並啟動所有元件：
  1. 讀取設定檔
  2. 初始化影像擷取器（capture）
  3. 初始化前處理器（preprocess）
  4. 初始化結果控制器（controller）
  5. 建立 WebRTC 連線到 Dispatcher
  6. 進入主迴圈：擷取 → 前處理 → 傳送 → 接收結果

使用方式：
    # 測試模式（本機跑全部元件）
    python edge/main.py --config config/test.yaml

    # 生產模式（在 Raspberry Pi 5 上，吃 Pi Camera）
    python edge/main.py --config config/prod.yaml

    # 在 Pi 上但先用影片驗證管線（不動 yaml，CLI override）
    python edge/main.py --config config/prod.yaml --mode video --source edge/video/30.mp4

    # 反過來：本機 config 是 video，但臨時想接 USB webcam
    python edge/main.py --config config/test.yaml --mode camera --source 0

    # 收集指標（延遲 / 丟包 / failover 切換），結束時輸出 CSV + summary + 圖表
    python edge/main.py --config config/staging_video.yaml --metrics

    # 影片播一次就退出（debug 用，不重播）；壓力測試全速灌
    python edge/main.py --config config/staging_video.yaml --loop no --sample-mode fast_replay

    # 注液到目標量自動停泵：連續 5 張 prediction < 10 就觸發 pump_stop_action()
    python edge/main.py --config config/staging.yaml --stop-target 10 --stop-frames 5
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

# Windows 主控台預設 cp932/cp950，無法輸出中文（含 argparse --help 的說明文字）。
# 強制 stdout/stderr 走 UTF-8，否則 `python edge/main.py --help` 會 UnicodeEncodeError。
# （Linux/Pi 本來就是 UTF-8，這段不影響。）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 沒有 reconfigure

# 確保專案根目錄在 Python 路徑中，讓 shared/ 等模組可被 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config
from edge.capture import FrameCapture
from edge.preprocess import Preprocessor
from edge.webrtc_client import WebRTCClient
from edge.controller import Controller
from edge.metrics import MetricsCollector
from edge.stop_monitor import StopMonitor
from edge.pump import get_pump

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
# 抑制 aioice/aiortc 的 ICE candidate pair 噪音 log
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
# TURN 個別 channel bind 偶爾失敗（403 Forbidden IP 等）會被 asyncio 印
# "Task exception was never retrieved" + traceback；它是背景 task，
# 失敗不影響整體運作，所以壓掉以免洗版。
# 若 debug 用，把這行改回 logging.WARNING 或 logging.ERROR 即可。
logging.getLogger("aioice.turn").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logger = logging.getLogger("edge")


async def run(
    config_path: str,
    mode_override: str = "",
    source_override: str = "",
    metrics_dir: str = "",
    loop_override: str = "",
    sample_mode_override: str = "",
    stop_target: float = 10.0,
    stop_frames: int = 5,
):
    """Edge 主要執行流程。

    完整流程：
    1. 載入設定 → 建立各元件
    2. 開啟影像擷取器
    3. 建立 WebRTC 連線（含 signaling + ICE）
    4. 進入無限迴圈：讀幀 → 前處理 → 傳送
    5. 推論結果由 WebRTC data channel 回傳，controller 非同步處理

    Args:
        config_path:          YAML 設定檔路徑
        mode_override:        覆寫 capture.mode（"video" / "camera"，空字串 = 不覆寫）
        source_override:      覆寫 capture.source（影片路徑或相機索引，空 = 不覆寫）
        metrics_dir:          指標輸出目錄；空字串 = 不收集指標
        loop_override:        覆寫 capture.loop（"yes" / "no"，空 = 不覆寫；僅 video 模式）
        sample_mode_override: 覆寫 capture.sample_mode（real_time / all_frames /
                              fast_replay，空 = 不覆寫；僅 video 模式）
        stop_target:          停止目標值；prediction 連續低於此值即觸發停止動作。
                              <= 0 表示停用停止監控。
        stop_frames:          連續幾張 prediction 低於 target 才觸發（去抖）。

    Note:
        capture.read() 是阻塞操作（會 sleep 做 FPS 節流），
        用 asyncio.to_thread() 包裝避免阻塞 event loop。
    """
    cfg = load_config(config_path)
    edge = cfg.edge

    # ── CLI override：允許不動 yaml 就切換影像來源 ──
    # 部署到 Pi 時很常見：prod.yaml 寫 camera，但想先用影片驗證管線
    if mode_override:
        if mode_override not in ("video", "camera"):
            raise ValueError(f"--mode 只能是 video 或 camera，收到: {mode_override}")
        logger.info("CLI override: mode %s → %s", edge.capture.mode, mode_override)
        edge.capture.mode = mode_override
    if source_override:
        logger.info("CLI override: source %s → %s", edge.capture.source, source_override)
        edge.capture.source = source_override
    if sample_mode_override:
        if sample_mode_override not in ("real_time", "all_frames", "fast_replay"):
            raise ValueError(
                f"--sample-mode 只能是 real_time / all_frames / fast_replay，"
                f"收到: {sample_mode_override}"
            )
        logger.info(
            "CLI override: sample_mode %s → %s",
            edge.capture.sample_mode, sample_mode_override,
        )
        edge.capture.sample_mode = sample_mode_override
    if loop_override:
        # loop_override 是字串 "yes" / "no"（CLI 進來的）
        new_loop = loop_override.lower() in ("yes", "true", "1", "on")
        logger.info("CLI override: loop %s → %s", edge.capture.loop, new_loop)
        edge.capture.loop = new_loop

    # ── 初始化各元件 ──

    # 影像擷取器：根據設定讀取影片或攝影機
    capture = FrameCapture(edge.capture)

    # 前處理器：ROI 裁切 + 壓縮
    preprocessor = Preprocessor(edge.preprocess)

    # 結果控制器：處理推論回傳結果
    controller = Controller()

    # ── 指標收集器（可選，--metrics 開啟）──
    # 在 send / receive 兩個 hot path 多加一行紀錄，overhead 極小，
    # 結束時 dump 出 raw CSV / summary / 圖表
    metrics = None
    if metrics_dir:
        metrics = MetricsCollector(metrics_dir)
        logger.info("Metrics 收集已啟用，輸出位置: %s", metrics_dir)

    # ── 泵控制 + 停止監控（依 prediction，可選）──
    # 啟用時：串流開始 → 開泵；連續 stop_frames 張 prediction < stop_target → 關泵。
    # 實際 GPIO 控制在 edge/pump_local.py（Pi 上才有）；其餘環境只 log。
    pump = None
    stop_monitor = None
    if stop_target and stop_target > 0:
        pump = get_pump()
        stop_monitor = StopMonitor(
            target=stop_target, consecutive=stop_frames,
            mode="below", on_trigger=lambda pred: pump.stop(),
        )
        logger.info(
            "停止監控已啟用: 連續 %d 張 prediction < %.2f 即關泵",
            stop_frames, stop_target,
        )

    # 包一層 controller.handle_result：metrics 記錄 + 停止監控 + 原始處理
    if metrics or stop_monitor:
        original_handle = controller.handle_result

        async def _handle(result):
            if metrics:
                metrics.record_received(result)
            if stop_monitor:
                pred = result.get("result", {}).get("prediction")
                stop_monitor.observe(pred)
            await original_handle(result)

        controller.handle_result = _handle
    # 範例：註冊偵測到 "person" 時的處理動作
    # async def on_person(det):
    #     logger.info("偵測到人！信心度: %.2f, bbox: %s", det["confidence"], det["bbox"])
    #     # 在這裡觸發 GPIO、警報、通知等控制動作
    # controller.register("person", on_person)

    # WebRTC 客戶端：連接 signaling → 建立 data channel → 含 failover
    client = WebRTCClient(
        edge_id=edge.id,
        signaling_url=edge.signaling.url,
        ice_servers=edge.ice_servers,
        failover=edge.failover,
        on_result=controller.handle_result,  # 推論結果的回呼函式
    )

    try:
        # 開啟影像來源
        capture.open()

        # 建立 WebRTC 連線（此步驟會阻塞直到 data channel open）
        await client.start()
        logger.info("Edge 串流已啟動 (id=%s, fps=%d)", edge.id, edge.capture.fps)

        # 注液開始 → 開泵（啟用停止監控時才管泵；其餘環境只 log）
        if pump:
            pump.start()

        # ── 主迴圈：擷取 → 前處理 → 傳送 ──
        seq = 0
        while True:
            # 在獨立 thread 中讀取幀（避免 sleep 阻塞 event loop）
            ok, frame = await asyncio.to_thread(capture.read)
            if not ok:
                logger.warning("影像擷取結束")
                break

            # 前處理：ROI + resize + JPEG 壓縮（量測耗時，前處理沒有節流 sleep，可直接量）
            _pp_start = time.time()
            jpeg = preprocessor.process(frame)
            preprocess_ms = (time.time() - _pp_start) * 1000.0

            # 組裝幀 header（中繼資訊，會隨幀一起傳送）
            header = {
                "frame_id": uuid.uuid4().hex[:8],
                "edge_id": edge.id,
                "seq": seq,
            }

            # 透過 WebRTC data channel 傳送
            sent = await client.send_frame(header, jpeg)

            # 記錄送出事件（給 metrics 用）
            # capture_ms 取自 capture.read() 內部量的「不含節流 sleep」的擷取成本
            if metrics:
                metrics.record_sent(
                    header["frame_id"], seq, len(jpeg), sent,
                    capture_ms=capture.last_capture_ms,
                    preprocess_ms=preprocess_ms,
                )

            # 每 30 幀印一次進度（方便確認串流是否正常）
            if seq % 30 == 0:
                logger.info(
                    "Streaming: seq=%d jpeg=%d bytes sent=%s",
                    seq, len(jpeg), sent,
                )
            seq += 1

    except KeyboardInterrupt:
        logger.info("收到中斷信號")
    except asyncio.TimeoutError:
        logger.error("連線逾時：請確認 Signaling / Dispatcher 已啟動")
    except ConnectionRefusedError as e:
        logger.error("無法連線: %s", e)
    except Exception:
        logger.exception("Edge 發生未預期錯誤")
    finally:
        # 退出時保險關泵（避免 edge 掛掉但泵還在跑）
        if pump:
            try:
                pump.stop()
            except Exception:
                logger.exception("關泵失敗")
        capture.release()
        await client.close()
        if metrics:
            metrics.finalize()
        logger.info("Edge 已關閉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge 裝置入口")
    parser.add_argument("--config", default="config/test.yaml", help="YAML 設定檔路徑")
    parser.add_argument(
        "--mode",
        default="",
        choices=["", "video", "camera"],
        help="覆寫 yaml 的 capture.mode（留空 = 用 yaml 設定）",
    )
    parser.add_argument(
        "--source",
        default="",
        help=(
            "覆寫 yaml 的 capture.source（留空 = 用 yaml 設定）。"
            "video 模式填影片路徑，例如 edge/video/30.mp4；"
            "camera 模式填裝置索引，例如 0 或 /dev/video0"
        ),
    )
    parser.add_argument(
        "--sample-mode",
        default="",
        choices=["", "real_time", "all_frames", "fast_replay"],
        help=(
            "覆寫 yaml 的 capture.sample_mode（僅 video 模式生效）："
            "real_time = 原速播 + 降取樣（預設）；"
            "all_frames = 每格都送 + 節流（慢播）；"
            "fast_replay = 每格都送 + 全速灌（壓測用）"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="?",
        const="auto",
        default="",
        help=(
            "啟用指標收集。不帶值 = 自動產生 metrics/run_<timestamp>/ 目錄；"
            "帶值 = 用該值當輸出目錄。"
            "結束時會寫入 raw_data.csv / summary.txt / *.png"
        ),
    )
    parser.add_argument(
        "--loop",
        default="",
        choices=["", "yes", "no"],
        help=(
            "覆寫 yaml 的 capture.loop（僅 video 模式生效，留空 = 用 yaml 設定）。"
            "yes = 影片播完從頭重播；no = 影片播完 edge 自動退出（適合 debug）"
        ),
    )
    parser.add_argument(
        "--stop-target",
        type=float,
        default=10.0,
        help=(
            "停止目標值（預設 10）。當 prediction 連續低於此值時，"
            "呼叫 pump_stop_action() 觸發停止/控泵動作。設 0 或負數 = 停用。"
        ),
    )
    parser.add_argument(
        "--stop-frames",
        type=int,
        default=5,
        help="連續幾張 prediction 低於 --stop-target 才觸發（去抖，預設 5）",
    )
    args = parser.parse_args()

    # --metrics auto → 自動產生帶時間戳的目錄名
    metrics_dir = args.metrics
    if metrics_dir == "auto":
        ts = time.strftime("%Y%m%d_%H%M%S")
        metrics_dir = f"metrics/run_{ts}"
    # aiortc 在 Windows 需要 SelectorEventLoop（PreactorEventLoop 不支援 UDP/DTLS）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(
        args.config, args.mode, args.source, metrics_dir,
        args.loop, args.sample_mode,
        args.stop_target, args.stop_frames,
    ))
