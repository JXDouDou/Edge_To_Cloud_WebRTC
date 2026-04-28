"""Dispatcher 入口程式（部署於 AWS EC2）。

Dispatcher 是 Edge 和 Inference Server 之間的中繼站，本身不做推論，
只負責轉發資料，讓算力集中在推論伺服器上。

資料流：
  Edge ──WebRTC DC──► Dispatcher ──WebSocket──► Inference Server
  Edge ◄──WebRTC DC── Dispatcher ◄──WebSocket── Inference Server

架構特點：
  - 每台 EC2 跑一個 Dispatcher process
  - 可同時服務多台 Edge（每台 Edge 有獨立的 PeerConnection）
  - 與 Inference Server 透過 Tailscale WebSocket 連接（非內網）
  - 如果 EC2 重啟 IP 變了，只需 Signaling Server 有固定 domain，
    Dispatcher 會重新向 Signaling 註冊

Tailscale 注意事項：
  - inference_ws_url 填 Tailscale hostname，例如:
    "ws://desktop-5080.tail12345.ts.net:8765/ws"
    或 Tailscale IP: "ws://100.x.x.x:8765/ws"
  - Tailscale 已加密，不需要額外用 wss://
  - 確保 Inference Server 的 Tailscale 防火牆有開放 8765 port

使用方式：
    python dispatcher/main.py --config config/test.yaml --id dispatcher-001
    python dispatcher/main.py --config config/prod.yaml --id dispatcher-ec2-001
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config, DispatcherConfig
from shared.protocol import Message, MsgType, pack_frame, unpack_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [dispatcher] %(levelname)s: %(message)s",
)
# 抑制 aioice/aiortc 的 ICE candidate pair 噪音 log
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logger = logging.getLogger("dispatcher")


class Dispatcher:
    """Dispatcher 核心邏輯：接收 Edge WebRTC 影像，轉發給 Inference Server。

    維護三種連線：
    1. ws_sig:    Signaling Server 的 WebSocket（信令交換用）
    2. ws_inf:    Inference Server 的 WebSocket（轉發影像 + 接收結果）
    3. peers/channels: 每台 Edge 的 WebRTC PeerConnection + Data Channel
    """

    def __init__(self, config: DispatcherConfig):
        """初始化 Dispatcher。

        Args:
            config: 此 Dispatcher 的設定物件（從 YAML dispatchers 列表中取得）
        """
        self.config = config
        self.id = config.id

        # ── 連線物件 ──
        self._session: aiohttp.ClientSession = None  # 共用 HTTP session
        self._ws_sig = None   # → Signaling Server WebSocket
        self._ws_inf = None   # → Inference Server WebSocket

        # ── Edge 連線追蹤 ──
        self._peers: dict = {}     # edge_id → RTCPeerConnection
        self._channels: dict = {}  # edge_id → RTCDataChannel

        # ── Debug 計數器：每 N 幀印一次 log，確認資料流動 ──
        self._frames_received: dict = {}   # edge_id → 收到的幀數
        self._frames_forwarded: dict = {}  # edge_id → 成功轉發到 inference 的幀數
        self._results_returned: dict = {}  # edge_id → 從 inference 收到並回傳給 edge 的結果數
        self._log_every = 30               # 每 30 幀 / 結果印一次進度

        self._running = True

    # ================================================================
    # 生命週期管理
    # ================================================================

    async def start(self):
        """啟動 Dispatcher：連接 Signaling + Inference，然後等待 Edge 連入。

        啟動順序很重要：
        1. 先連 Signaling（向 Edge 宣告自己的存在）
        2. 再連 Inference（準備好轉發路徑）
        3. 進入等待迴圈（接收 Edge WebRTC offer）

        Raises:
            aiohttp.ClientError: 無法連接 Signaling 或 Inference
        """
        self._session = aiohttp.ClientSession()
        await self._connect_signaling()
        await self._connect_inference()
        logger.info("Dispatcher %s 就緒，等待 Edge 連入...", self.id)

        # 持續運行直到被停止
        while self._running:
            await asyncio.sleep(1)

    async def stop(self):
        """優雅關閉所有連線。

        關閉順序：
        1. 所有 Edge 的 PeerConnection
        2. Signaling WebSocket
        3. Inference WebSocket
        4. HTTP session
        """
        self._running = False
        for edge_id, pc in self._peers.items():
            logger.info("關閉與 Edge %s 的連線", edge_id)
            await pc.close()
        if self._ws_sig and not self._ws_sig.closed:
            await self._ws_sig.close()
        if self._ws_inf and not self._ws_inf.closed:
            await self._ws_inf.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ================================================================
    # Signaling 信令層
    # ================================================================

    async def _connect_signaling(self):
        """連接到 Signaling Server 並註冊為 Dispatcher。

        註冊後，Edge 就能在 dispatcher list 中看到此 Dispatcher，
        並可能發送 WebRTC offer 過來。
        """
        url = self.config.signaling.url
        logger.info("連接 Signaling: %s", url)
        self._ws_sig = await self._session.ws_connect(url)

        # 註冊身份
        reg = Message(
            type=MsgType.REGISTER,
            source_id=self.id,
            payload={"role": "dispatcher"},
        )
        await self._ws_sig.send_str(reg.serialize())
        logger.info("已向 Signaling 註冊: %s", self.id)

        # 背景持續監聽信令訊息
        asyncio.create_task(self._signaling_loop())

    async def _signaling_loop(self):
        """持續監聽 Signaling Server 的訊息。

        主要處理：
        - OFFER: Edge 的 WebRTC SDP offer → 建立 PeerConnection 並回覆 answer
        - ICE:   ICE candidate（目前用 full ICE，預留）

        如果 Signaling 斷線，會嘗試重連。
        """
        try:
            async for ws_msg in self._ws_sig:
                if ws_msg.type == WSMsgType.TEXT:
                    msg = Message.deserialize(ws_msg.data)
                    if msg.type == MsgType.OFFER:
                        await self._handle_offer(msg)
                    elif msg.type == MsgType.PING:
                        # Dispatcher 也可能收到 signaling 的 ping
                        pong = Message(type=MsgType.PONG, source_id=self.id)
                        await self._ws_sig.send_str(pong.serialize())
                elif ws_msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except Exception:
            logger.exception("signaling_loop 異常")

        # 斷線重連
        if self._running:
            logger.warning("Signaling 連線斷開，5 秒後重連...")
            await asyncio.sleep(5)
            try:
                await self._connect_signaling()
            except Exception:
                logger.exception("Signaling 重連失敗")

    async def _handle_offer(self, msg: Message):
        """處理 Edge 的 WebRTC SDP Offer，建立 PeerConnection 並回覆 Answer。

        完整流程：
        1. 用 Edge 的 SDP Offer 建立 RTCPeerConnection
        2. 註冊 datachannel 事件（等待 Edge 建立的 data channel）
        3. 建立 SDP Answer → 等待 ICE gathering → 透過 Signaling 回傳

        Args:
            msg: 包含 SDP offer 的 Message（source_id = edge_id）
        """
        edge_id = msg.source_id
        logger.info("收到 WebRTC OFFER: %s", edge_id)

        # ── 建立 PeerConnection ──
        ice_cfg = RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=s.urls,
                    username=s.username or None,
                    credential=s.credential or None,
                )
                for s in self.config.ice_servers
            ]
        )
        pc = RTCPeerConnection(configuration=ice_cfg)
        self._peers[edge_id] = pc

        # ── 監聽 Data Channel ──
        # Edge 是 offerer，會建立 data channel；Dispatcher 是 answerer，要監聽 datachannel 事件
        @pc.on("datachannel")
        def on_datachannel(channel):
            """收到 Edge 建立的 Data Channel。

            注意：此回呼只代表 Edge 在 SDP 中宣告了 channel，
            channel 實際可用要等 on("open") 觸發。

            Args:
                channel: RTCDataChannel 物件，用於雙向傳輸
            """
            logger.info("Data Channel 已協商: edge=%s, label=%s, state=%s",
                        edge_id, channel.label, channel.readyState)
            self._channels[edge_id] = channel

            @channel.on("open")
            def on_open():
                """Data Channel 真正可用時觸發（雙向通了）。"""
                logger.info("✓ Data Channel OPEN: edge=%s（可開始收幀）", edge_id)

            @channel.on("close")
            def on_close():
                logger.warning("Data Channel CLOSE: edge=%s", edge_id)

            @channel.on("message")
            def on_message(data):
                """收到 Data Channel 訊息。

                二進位資料 → 影像幀，轉發到 Inference
                文字資料   → 控制訊息（如 PING/PONG）

                Args:
                    data: bytes（影像幀）或 str（JSON 控制訊息）
                """
                if isinstance(data, bytes):
                    # ── Debug：每 N 幀印一次，確認 frame 真的進到 dispatcher ──
                    n = self._frames_received.get(edge_id, 0)
                    if n % self._log_every == 0:
                        logger.info(
                            "← 收到 frame: edge=%s, count=%d, size=%d bytes",
                            edge_id, n, len(data),
                        )
                    self._frames_received[edge_id] = n + 1

                    # 二進位 = 影像幀 → 轉發到 Inference Server
                    asyncio.create_task(self._forward_to_inference(edge_id, data))
                elif isinstance(data, str):
                    # 文字 = 控制訊息（目前只有 PING）
                    asyncio.create_task(self._handle_dc_text(edge_id, channel, data))

        # ── PeerConnection 狀態監控 ──
        @pc.on("connectionstatechange")
        async def on_state():
            """監控與 Edge 的 WebRTC 連線狀態。

            當連線 failed/closed 時，清理該 Edge 的追蹤資料。
            """
            state = pc.connectionState
            logger.info("Edge %s PeerConnection 狀態: %s", edge_id, state)
            if state in ("failed", "closed"):
                self._peers.pop(edge_id, None)
                self._channels.pop(edge_id, None)

        # ── 設定 Remote Description（Edge 的 Offer）──
        offer = RTCSessionDescription(
            sdp=msg.payload["sdp"],
            type=msg.payload["type"],
        )
        await pc.setRemoteDescription(offer)

        # ── 建立 Answer ──
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # ── 等待 ICE Gathering 完成 ──
        gathering_done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def on_gather():
            """ICE candidate 收集狀態變更。"""
            if pc.iceGatheringState == "complete":
                gathering_done.set()

        if pc.iceGatheringState == "complete":
            gathering_done.set()

        await asyncio.wait_for(gathering_done.wait(), timeout=10)

        # ── 透過 Signaling 回傳 Answer ──
        answer_msg = Message(
            type=MsgType.ANSWER,
            source_id=self.id,
            target_id=edge_id,
            payload={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            },
        )
        await self._ws_sig.send_str(answer_msg.serialize())
        logger.info("已回傳 SDP Answer → %s", edge_id)

    async def _handle_dc_text(self, edge_id: str, channel, raw: str):
        """處理 Data Channel 上的文字訊息（如 PING/PONG）。

        Args:
            edge_id: 來源 Edge ID
            channel: Data Channel 物件（用於回覆）
            raw:     JSON 文字訊息
        """
        try:
            msg = Message.deserialize(raw)
            if msg.type == MsgType.PING:
                # 回覆 PONG，攜帶原始 ping_ts 讓 Edge 計算 RTT
                pong = Message(
                    type=MsgType.PONG,
                    source_id=self.id,
                    payload=msg.payload,  # 原封不動回傳 ping_ts
                )
                channel.send(pong.serialize())
        except Exception:
            logger.exception("處理 DC 文字訊息失敗: edge=%s", edge_id)

    # ================================================================
    # Inference Server 連線層
    # ================================================================

    async def _connect_inference(self):
        """連接到 Inference Server 的 WebSocket。

        連線目標由 config.inference_ws_url 指定。
        如果是 Tailscale 網路，URL 會是 Tailscale hostname。

        連線成功後啟動背景 task 持續接收推論結果。

        Raises:
            aiohttp.ClientError: 無法連接（Inference Server 未啟動或網路不通）
        """
        url = self.config.inference_ws_url
        logger.info("連接 Inference Server: %s", url)
        self._ws_inf = await self._session.ws_connect(
            url,
            max_msg_size=4 * 1024 * 1024,  # 4 MB，因為要接收大幀
        )
        logger.info("Inference Server 連線成功: %s", url)

        # 背景持續接收推論結果
        asyncio.create_task(self._inference_recv_loop())

    async def _forward_to_inference(self, edge_id: str, raw_frame: bytes):
        """將 Edge 的影像幀轉發到 Inference Server。

        流程：
        1. 解包二進位幀（取出 header 和 JPEG）
        2. 在 header 中注入 edge_id（Inference 回傳時需要知道送回給誰）
        3. 重新打包成二進位
        4. 透過 WebSocket binary message 傳送

        Args:
            edge_id:   來源 Edge 的 ID
            raw_frame: 由 pack_frame() 打包的二進位資料
        """
        try:
            if not self._ws_inf or self._ws_inf.closed:
                logger.warning("Inference WS 未連線，丟棄幀: edge=%s", edge_id)
                return

            # 解包 → 注入 edge_id → 重新打包
            header, jpeg = unpack_frame(raw_frame)
            header["edge_id"] = edge_id
            await self._ws_inf.send_bytes(pack_frame(header, jpeg))

            # ── Debug：每 N 幀印一次，確認轉發成功 ──
            n = self._frames_forwarded.get(edge_id, 0)
            if n % self._log_every == 0:
                logger.info(
                    "→ 已轉發到 Inference: edge=%s, count=%d", edge_id, n,
                )
            self._frames_forwarded[edge_id] = n + 1
        except Exception:
            logger.exception("轉發幀到 Inference 失敗: edge=%s", edge_id)

    async def _inference_recv_loop(self):
        """持續接收 Inference Server 的推論結果，路由回對應的 Edge。

        Inference Server 回傳的結果是 JSON 文字訊息，包含：
        - edge_id:  目標 Edge（由 _forward_to_inference 注入的）
        - frame_id: 對應的幀 ID
        - result:   推論結果

        路由邏輯：根據 edge_id 找到對應的 Data Channel 並傳送。

        如果 Inference 連線斷開，會自動嘗試重連。
        """
        try:
            async for ws_msg in self._ws_inf:
                if ws_msg.type == WSMsgType.TEXT:
                    msg = Message.deserialize(ws_msg.data)
                    if msg.type == MsgType.RESULT:
                        # 取出目標 Edge ID
                        edge_id = msg.payload.get("edge_id", "")
                        dc = self._channels.get(edge_id)
                        if dc and dc.readyState == "open":
                            # 透過 Data Channel 回傳結果給 Edge
                            dc.send(msg.serialize())
                            # ── Debug：每 N 筆印一次，確認結果有回傳 ──
                            n = self._results_returned.get(edge_id, 0)
                            if n % self._log_every == 0:
                                logger.info(
                                    "↑ 已回傳結果給 Edge: edge=%s, count=%d",
                                    edge_id, n,
                                )
                            self._results_returned[edge_id] = n + 1
                        else:
                            state = dc.readyState if dc else "no-channel"
                            logger.warning(
                                "Edge %s 的 Data Channel 不可用 (state=%s)，丟棄結果",
                                edge_id, state,
                            )
                elif ws_msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except Exception:
            logger.exception("inference_recv_loop 異常")

        # ── 斷線自動重連 ──
        if self._running:
            logger.warning("Inference 連線斷開，3 秒後重連...")
            await asyncio.sleep(3)
            try:
                await self._connect_inference()
            except Exception:
                logger.exception("Inference 重連失敗，10 秒後再試...")
                await asyncio.sleep(10)
                if self._running:
                    asyncio.create_task(self._connect_inference())


# ================================================================
# 入口
# ================================================================

def _find_dispatcher_config(cfg, disp_id: str) -> DispatcherConfig:
    """從設定檔中找到指定 ID 的 Dispatcher 設定。

    Args:
        cfg:     AppConfig 設定物件
        disp_id: 要尋找的 Dispatcher ID

    Returns:
        對應的 DispatcherConfig

    Raises:
        ValueError: 指定的 ID 在設定檔中不存在
    """
    for d in cfg.dispatchers:
        if d.id == disp_id:
            return d
    available = [d.id for d in cfg.dispatchers]
    raise ValueError(
        f"Dispatcher '{disp_id}' 不在設定檔中。可用的 ID: {available}"
    )


async def run(config_path: str, dispatcher_id: str):
    """啟動單一 Dispatcher 實例。

    Args:
        config_path:    YAML 設定檔路徑
        dispatcher_id:  此 Dispatcher 的 ID（需在設定檔的 dispatchers 列表中）
    """
    cfg = load_config(config_path)
    disp_cfg = _find_dispatcher_config(cfg, dispatcher_id)
    dispatcher = Dispatcher(disp_cfg)
    try:
        await dispatcher.start()
    except KeyboardInterrupt:
        pass
    finally:
        await dispatcher.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dispatcher（部署於 EC2）")
    parser.add_argument("--config", default="config/test.yaml", help="YAML 設定檔路徑")
    parser.add_argument("--id", default="dispatcher-001", help="此 Dispatcher 的 ID（需與設定檔一致）")
    args = parser.parse_args()
    # aiortc 在 Windows 需要 SelectorEventLoop（ProactorEventLoop 不支援 UDP/DTLS）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args.config, args.id))
