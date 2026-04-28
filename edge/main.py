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
"""

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

# 確保專案根目錄在 Python 路徑中，讓 shared/ 等模組可被 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config
from edge.capture import FrameCapture
from edge.preprocess import Preprocessor
from edge.webrtc_client import WebRTCClient
from edge.controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
# 抑制 aioice/aiortc 的 ICE candidate pair 噪音 log
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logger = logging.getLogger("edge")


async def run(config_path: str, mode_override: str = "", source_override: str = ""):
    """Edge 主要執行流程。

    完整流程：
    1. 載入設定 → 建立各元件
    2. 開啟影像擷取器
    3. 建立 WebRTC 連線（含 signaling + ICE）
    4. 進入無限迴圈：讀幀 → 前處理 → 傳送
    5. 推論結果由 WebRTC data channel 回傳，controller 非同步處理

    Args:
        config_path: YAML 設定檔路徑

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

    # ── 初始化各元件 ──

    # 影像擷取器：根據設定讀取影片或攝影機
    capture = FrameCapture(edge.capture)

    # 前處理器：ROI 裁切 + 壓縮
    preprocessor = Preprocessor(edge.preprocess)

    # 結果控制器：處理推論回傳結果
    controller = Controller()
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

        # ── 主迴圈：擷取 → 前處理 → 傳送 ──
        seq = 0
        while True:
            # 在獨立 thread 中讀取幀（避免 sleep 阻塞 event loop）
            ok, frame = await asyncio.to_thread(capture.read)
            if not ok:
                logger.warning("影像擷取結束")
                break

            # 前處理：ROI + resize + JPEG 壓縮
            jpeg = preprocessor.process(frame)

            # 組裝幀 header（中繼資訊，會隨幀一起傳送）
            header = {
                "frame_id": uuid.uuid4().hex[:8],
                "edge_id": edge.id,
                "seq": seq,
            }

            # 透過 WebRTC data channel 傳送
            sent = await client.send_frame(header, jpeg)

            # 每 30 幀印一次進度（方便確認串流是否正常）
            if seq % 30 == 0:
                logger.info(
                    "串流中: seq=%d, jpeg=%d bytes, sent=%s",
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
        capture.release()
        await client.close()
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
    args = parser.parse_args()
    # aiortc 在 Windows 需要 SelectorEventLoop（PreactorEventLoop 不支援 UDP/DTLS）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args.config, args.mode, args.source))
