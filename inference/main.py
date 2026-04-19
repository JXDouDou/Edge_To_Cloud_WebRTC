"""推論伺服器入口程式（部署於 5080 桌機）。

此伺服器接收 Dispatcher 透過 WebSocket 傳送的影像幀，
執行模型推論，並將結果回傳給 Dispatcher。

通訊協議：
  - 接收：WebSocket binary message（pack_frame 格式的二進位幀）
  - 回傳：WebSocket text message（JSON 格式的推論結果）

Tailscale 部署注意事項：
  - host 設為 "0.0.0.0" 讓 Tailscale 虛擬網卡的 IP 可達
  - Dispatcher 的 inference_ws_url 填 Tailscale hostname:
    例如 "ws://desktop-5080.tail12345.ts.net:8765/ws"
  - Tailscale 流量已加密，不需要額外用 wss://
  - 確保 Tailscale ACL / 防火牆有開放 port 8765

效能考量：
  - predict() 是 CPU/GPU 密集操作，用 asyncio.to_thread() 避免阻塞 event loop
  - 多台 Dispatcher 可以同時連入同一台 Inference Server
  - 如果推論速度跟不上幀率，舊幀會排隊等待（FIFO，無丟棄機制）
  - 如需丟幀策略，可在此模組加入佇列管理

使用方式：
    python inference/main.py --config config/test.yaml
    python inference/main.py --config config/prod.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from aiohttp import web, WSMsgType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config
from shared.protocol import Message, MsgType, unpack_frame
from inference.model_runner import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [inference] %(levelname)s: %(message)s",
)
logger = logging.getLogger("inference")


class InferenceServer:
    """推論伺服器核心邏輯。

    每個 WebSocket 連線代表一台 Dispatcher。
    多台 Dispatcher 可以同時連入。
    """

    def __init__(self, config):
        """初始化推論伺服器。

        Args:
            config: InferenceConfig 設定物件
        """
        self.config = config

        # 建立模型實例（這一步可能會載入數百 MB 的權重檔到 GPU）
        self.model = create_model(config.model_type, config.model_path, config.device)

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """處理 /ws 端點的 WebSocket 連線（來自 Dispatcher）。

        每台 Dispatcher 連入後，持續接收二進位幀並回傳推論結果。
        生命週期：連線 → 接收幀迴圈 → 斷線。

        Args:
            request: aiohttp HTTP request

        Returns:
            WebSocketResponse 物件
        """
        ws = web.WebSocketResponse(
            max_msg_size=4 * 1024 * 1024,  # 4 MB，配合大幀
        )
        await ws.prepare(request)
        logger.info("Dispatcher 已連入: %s", request.remote)

        async for ws_msg in ws:
            if ws_msg.type == WSMsgType.BINARY:
                # 二進位訊息 = 影像幀
                await self._process_frame(ws, ws_msg.data)
            elif ws_msg.type == WSMsgType.TEXT:
                # 文字訊息 = 控制命令（預留）
                logger.debug("收到文字訊息: %s", ws_msg.data[:100])
            elif ws_msg.type == WSMsgType.ERROR:
                logger.error("WS 錯誤: %s", ws.exception())

        logger.info("Dispatcher 已斷線: %s", request.remote)
        return ws

    async def _process_frame(self, ws, raw: bytes):
        """處理單一影像幀：解包 → 推論 → 回傳結果。

        流程：
        1. 解包二進位資料（header + JPEG）
        2. 在獨立 thread 中執行模型推論（避免阻塞 event loop）
        3. 組裝推論結果 Message
        4. 以 JSON text message 回傳給 Dispatcher

        Args:
            ws:  WebSocket 連線（用於回傳結果）
            raw: pack_frame() 格式的二進位資料

        Note:
            推論錯誤不會中斷連線，只記 log 並跳過該幀。
        """
        try:
            # 解包二進位幀
            header, jpeg = unpack_frame(raw)

            # 在獨立 thread 中執行推論
            # 這很重要：模型推論是 CPU/GPU 密集操作，
            # 如果在 event loop 中直接呼叫會阻塞所有其他 coroutine
            result = await asyncio.to_thread(self.model.predict, jpeg)

            # 組裝回傳訊息
            reply = Message(
                type=MsgType.RESULT,
                payload={
                    "edge_id": header.get("edge_id", ""),
                    "frame_id": header.get("frame_id", ""),
                    "seq": header.get("seq", -1),
                    "result": result,
                },
            )

            # 回傳給 Dispatcher（JSON text message）
            await ws.send_str(reply.serialize())

        except Exception:
            logger.exception("處理幀時發生錯誤")


def main(config_path: str):
    """啟動推論伺服器。

    讀取設定 → 載入模型 → 啟動 aiohttp WebSocket 伺服器。

    Args:
        config_path: YAML 設定檔路徑
    """
    cfg = load_config(config_path)
    inf = cfg.inference
    server = InferenceServer(inf)

    app = web.Application()
    app.router.add_get("/ws", server.handle_ws)

    logger.info(
        "推論伺服器啟動: %s:%d | model=%s | device=%s",
        inf.host, inf.port, inf.model_type, inf.device,
    )
    web.run_app(app, host=inf.host, port=inf.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推論伺服器")
    parser.add_argument("--config", default="config/test.yaml", help="YAML 設定檔路徑")
    args = parser.parse_args()
    # aiohttp web.run_app 在 Windows 需要 SelectorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main(args.config)
