"""WebRTC 信令伺服器（Signaling Server）。

此伺服器負責 WebRTC 連線建立階段的信令交換，本身不參與影像資料傳輸。
部署建議：獨立一台 EC2，綁定固定 domain name（如 signaling.yourdomain.com）。

為什麼需要獨立的 Signaling Server？
  - Edge 位於 5G NAT 後方，無法直接連到 EC2 Dispatcher
  - WebRTC 需要透過第三方交換 SDP offer/answer 才能建立 P2P 連線
  - Signaling Server 需要有穩定的位址讓 Edge 找得到
  - EC2 每次重啟 IP 會變，所以 Signaling 綁 domain 用 Route53 解析

功能清單：
  1. 追蹤已連線的 Dispatcher 和 Edge
  2. 提供 Dispatcher 列表給 Edge（用於選擇連線目標和 failover）
  3. 中繼 WebRTC SDP Offer（Edge → Dispatcher）
  4. 中繼 WebRTC SDP Answer（Dispatcher → Edge）
  5. 回覆 PING/PONG 健康檢查

使用方式：
    python signaling/server.py --config config/test.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from aiohttp import web, WSMsgType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config
from shared.protocol import Message, MsgType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [signaling] %(levelname)s: %(message)s",
)
logger = logging.getLogger("signaling")


class SignalingServer:
    """信令伺服器核心邏輯。

    維護兩個連線池：
    - _edges:       已連線的 Edge 裝置 {edge_id: WebSocketResponse}
    - _dispatchers: 已連線的 Dispatcher {dispatcher_id: WebSocketResponse}

    所有訊息都透過 /ws 端點的 WebSocket 連線收發。
    """

    def __init__(self):
        """初始化信令伺服器，建立空的連線池。"""
        self._edges: dict = {}        # edge_id → WebSocketResponse
        self._dispatchers: dict = {}  # dispatcher_id → WebSocketResponse

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """處理 /ws 端點的 WebSocket 連線。

        此函式是 aiohttp WebSocket handler，生命週期如下：
        1. WebSocket 握手（upgrade）
        2. 進入訊息迴圈，持續接收和處理訊息
        3. 連線斷開時，從連線池移除

        Args:
            request: aiohttp HTTP request 物件

        Returns:
            WebSocketResponse 物件（aiohttp 框架要求的回傳值）

        Note:
            每個 WebSocket 連線都是獨立的 coroutine，
            多個 Edge/Dispatcher 可以同時連線。
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_id = None    # 連線識別碼（註冊後填入）
        client_role = None  # 角色：'edge' 或 'dispatcher'

        try:
            async for ws_msg in ws:
                if ws_msg.type == WSMsgType.TEXT:
                    msg = Message.deserialize(ws_msg.data)
                    client_id, client_role = await self._route(
                        msg, ws, client_id, client_role
                    )
                elif ws_msg.type == WSMsgType.ERROR:
                    logger.error("WS 錯誤: %s", ws.exception())
        finally:
            # 無論正常關閉或異常，都要從連線池移除
            self._unregister(client_id, client_role)

        return ws

    async def _route(self, msg: Message, ws, client_id: str, client_role: str):
        """路由並處理單一信令訊息。

        根據訊息類型分派到對應的處理邏輯。
        這是信令伺服器的核心路由函式。

        Args:
            msg:         反序列化後的 Message 物件
            ws:          發送此訊息的 WebSocket 連線
            client_id:   當前連線的 client ID（可能尚未註冊，為 None）
            client_role: 當前連線的角色（可能尚未註冊，為 None）

        Returns:
            (client_id, client_role) 元組，可能在 REGISTER 時被更新
        """

        # ── REGISTER：元件註冊身份 ──
        if msg.type == MsgType.REGISTER:
            client_id = msg.source_id
            client_role = msg.payload.get("role")

            if client_role == "dispatcher":
                self._dispatchers[client_id] = ws
                logger.info(
                    "Dispatcher 已註冊: %s (目前共 %d 台)",
                    client_id, len(self._dispatchers),
                )
            elif client_role == "edge":
                self._edges[client_id] = ws
                logger.info(
                    "Edge 已註冊: %s (目前共 %d 台)",
                    client_id, len(self._edges),
                )
            return client_id, client_role

        # ── REQUEST_DISPATCHERS：Edge 請求 Dispatcher 列表 ──
        if msg.type == MsgType.REQUEST_DISPATCHERS:
            disp_list = [{"id": did} for did in self._dispatchers]
            reply = Message(
                type=MsgType.DISPATCHER_LIST,
                payload={"dispatchers": disp_list},
            )
            await ws.send_str(reply.serialize())
            logger.info(
                "回覆 dispatcher 列表 (%d 台) → %s",
                len(disp_list), msg.source_id,
            )

        # ── OFFER / ICE：Edge → Dispatcher 方向的信令 ──
        elif msg.type in (MsgType.OFFER, MsgType.ICE):
            # 從 target_id 找到目標 Dispatcher 的 WebSocket 連線
            target_ws = self._dispatchers.get(msg.target_id)
            if target_ws and not target_ws.closed:
                await target_ws.send_str(msg.serialize())
                logger.info(
                    "轉發 %s: %s → %s",
                    msg.type.value, msg.source_id, msg.target_id,
                )
            else:
                logger.warning(
                    "目標 Dispatcher 不存在或已斷線: %s", msg.target_id
                )

        # ── ANSWER：Dispatcher → Edge 方向的信令 ──
        elif msg.type == MsgType.ANSWER:
            # 從 target_id 找到目標 Edge 的 WebSocket 連線
            target_ws = self._edges.get(msg.target_id)
            if target_ws and not target_ws.closed:
                await target_ws.send_str(msg.serialize())
                logger.info(
                    "轉發 ANSWER: %s → %s",
                    msg.source_id, msg.target_id,
                )
            else:
                logger.warning(
                    "目標 Edge 不存在或已斷線: %s", msg.target_id
                )

        # ── PING：健康檢查 ──
        elif msg.type == MsgType.PING:
            pong = Message(
                type=MsgType.PONG,
                source_id="signaling",
                target_id=msg.source_id,
            )
            await ws.send_str(pong.serialize())

        return client_id, client_role

    def _unregister(self, client_id: str, client_role: str):
        """從連線池移除已斷線的元件。

        在 WebSocket 連線關閉（正常或異常）時由 handle_ws 的 finally 呼叫。

        Args:
            client_id:   要移除的元件 ID
            client_role: 元件角色 ('edge' 或 'dispatcher')
        """
        if client_role == "dispatcher" and client_id in self._dispatchers:
            del self._dispatchers[client_id]
            logger.info(
                "Dispatcher 已斷線: %s (剩餘 %d 台)",
                client_id, len(self._dispatchers),
            )
        elif client_role == "edge" and client_id in self._edges:
            del self._edges[client_id]
            logger.info(
                "Edge 已斷線: %s (剩餘 %d 台)",
                client_id, len(self._edges),
            )


def main(config_path: str):
    """啟動 Signaling Server。

    讀取設定檔取得監聽位址和埠號，建立 aiohttp web application。

    Args:
        config_path: YAML 設定檔路徑
    """
    cfg = load_config(config_path)
    server = SignalingServer()

    app = web.Application()
    app.router.add_get("/ws", server.handle_ws)

    logger.info(
        "Signaling Server 啟動: %s:%d",
        cfg.signaling_host, cfg.signaling_port,
    )
    web.run_app(app, host=cfg.signaling_host, port=cfg.signaling_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC 信令伺服器")
    parser.add_argument("--config", default="config/test.yaml", help="YAML 設定檔路徑")
    args = parser.parse_args()
    # aiohttp web.run_app 在 Windows 需要 SelectorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main(args.config)
