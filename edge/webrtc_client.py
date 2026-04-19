"""WebRTC 客戶端，含自動 Dispatcher failover 與健康檢查。

核心流程：
  1. 透過 WebSocket 連接到 Signaling Server
  2. 取得可用 Dispatcher 列表
  3. 與主要 Dispatcher 建立 WebRTC PeerConnection + Data Channel
  4. 透過 Data Channel 傳送二進位幀，接收推論結果
  5. 定期 PING/PONG 監控連線品質
  6. 連線失敗時自動切換到下一台 Dispatcher

為什麼 Edge→EC2 用 WebRTC 而非 WebSocket？
  - 5G 網路的 NAT 通常是 symmetric NAT，很難直接打洞
  - WebRTC 的 ICE 框架會自動嘗試 STUN/TURN 穿透
  - Data Channel 底層用 SCTP，支援 ordered/unordered 和可靠/不可靠模式
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, List

import aiohttp
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from shared.config import FailoverConfig, ICEServer
from shared.protocol import Message, MsgType, pack_frame

logger = logging.getLogger(__name__)

# Data Channel 單一訊息大小上限（安全值）
# SCTP 底層會自動分片，但超過此值的幀會被跳過
MAX_DC_MESSAGE = 256 * 1024  # 256 KB


class WebRTCClient:
    """WebRTC 客戶端，管理與 Dispatcher 的連線生命週期。

    此類別封裝了完整的 WebRTC 連線流程：
    - Signaling 信令交換
    - ICE 候選收集與穿透
    - Data Channel 建立與監控
    - 自動 failover 到備用 Dispatcher
    """

    def __init__(
        self,
        edge_id: str,
        signaling_url: str,
        ice_servers: List[ICEServer],
        failover: FailoverConfig,
        on_result: Callable[[dict], Awaitable[None]],
    ):
        """初始化 WebRTC 客戶端。

        Args:
            edge_id:       本 Edge 裝置的唯一識別碼（如 "edge-001"）
            signaling_url: Signaling Server 的 WebSocket URL
                           （如 "ws://localhost:8080/ws" 或 "wss://signaling.example.com/ws"）
            ice_servers:   STUN/TURN 伺服器設定列表
            failover:      故障轉移設定（檢查間隔、容忍次數、恢復延遲）
            on_result:     收到推論結果時的回呼函式（async），
                           參數為 Message.payload dict
        """
        self.edge_id = edge_id
        self.signaling_url = signaling_url
        self.ice_servers = ice_servers
        self.failover = failover
        self.on_result = on_result

        # ── 內部狀態 ──
        self._pc: RTCPeerConnection = None     # WebRTC PeerConnection
        self._dc = None                        # Data Channel 實例
        self._session: aiohttp.ClientSession = None  # HTTP session（管理 WS 連線）
        self._ws = None                        # Signaling WebSocket 連線
        self._dispatchers: list = []           # 可用 dispatcher 列表
        self._disp_idx = 0                     # 當前使用的 dispatcher 索引
        self._current_disp: str = ""           # 當前 dispatcher ID
        self._fail_count = 0                   # 連續失敗計數器
        self._connected = asyncio.Event()      # data channel 是否 open 的事件旗標
        self._failover_lock = asyncio.Lock()   # 防止並發觸發多次 failover
        self._health_task = None               # 健康檢查 task
        self._sig_task = None                  # signaling 監聽 task

    # ================================================================
    # 公開 API
    # ================================================================

    async def start(self):
        """啟動客戶端：連接 signaling → 取得 dispatcher 列表 → 建立 WebRTC。

        此方法完成後，data channel 已 open，可以開始 send_frame()。

        Raises:
            RuntimeError: 無法連接 signaling 或無可用 dispatcher
            asyncio.TimeoutError: WebRTC 連線建立超時
        """
        self._session = aiohttp.ClientSession()
        await self._connect_signaling()
        await self._fetch_dispatchers()
        await self._connect_to_dispatcher()

        # 啟動背景健康檢查
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def send_frame(self, header: dict, jpeg_data: bytes) -> bool:
        """透過 Data Channel 傳送一幀影像。

        Args:
            header:    幀中繼資訊 dict（frame_id, edge_id, seq）
            jpeg_data: JPEG 壓縮後的影像位元組

        Returns:
            True = 傳送成功，False = 傳送失敗（未連線或幀太大）

        Note:
            - 如果連續失敗次數達到閾值，會自動觸發 failover
            - 超過 256KB 的幀會被跳過（不傳送也不計失敗）
        """
        if not self._connected.is_set():
            self._fail_count += 1
            if self._fail_count >= self.failover.max_failures:
                asyncio.create_task(self._do_failover())
            return False

        try:
            data = pack_frame(header, jpeg_data)
            if len(data) > MAX_DC_MESSAGE:
                logger.warning("幀太大 (%d bytes)，跳過傳送", len(data))
                return False
            self._dc.send(data)  # data channel send 是同步方法
            return True
        except Exception as e:
            logger.error("send_frame 錯誤: %s", e)
            self._fail_count += 1
            if self._fail_count >= self.failover.max_failures:
                asyncio.create_task(self._do_failover())
            return False

    async def close(self):
        """關閉所有連線並釋放資源。

        會依序關閉：
        1. 健康檢查 task
        2. WebRTC PeerConnection（包含 data channel）
        3. Signaling WebSocket
        4. HTTP session
        """
        if self._health_task:
            self._health_task.cancel()
        if self._pc:
            await self._pc.close()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ================================================================
    # Signaling 信令層
    # ================================================================

    async def _connect_signaling(self):
        """連接到 Signaling Server 並註冊為 Edge。

        建立 WebSocket 連線後，發送 REGISTER 訊息告知身份，
        然後啟動背景 task 持續監聽信令訊息。
        """
        logger.info("連接 Signaling: %s", self.signaling_url)
        self._ws = await self._session.ws_connect(self.signaling_url)

        # 向 signaling 註冊身份
        reg = Message(
            type=MsgType.REGISTER,
            source_id=self.edge_id,
            payload={"role": "edge"},
        )
        await self._ws.send_str(reg.serialize())
        logger.info("已註冊為 Edge: %s", self.edge_id)

        # 背景持續監聽信令訊息
        self._sig_task = asyncio.create_task(self._signaling_loop())

    async def _signaling_loop(self):
        """持續監聽 Signaling Server 的 WebSocket 訊息。

        處理的訊息類型：
        - DISPATCHER_LIST: 更新可用 dispatcher 列表
        - ANSWER:          WebRTC SDP answer，完成連線建立
        - ICE:             ICE candidate（目前使用 full ICE，此處預留）
        """
        try:
            async for ws_msg in self._ws:
                if ws_msg.type == aiohttp.WSMsgType.TEXT:
                    msg = Message.deserialize(ws_msg.data)
                    await self._on_signaling(msg)
                elif ws_msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("Signaling WS 關閉")
                    break
        except Exception:
            logger.exception("signaling_loop 異常")

    async def _on_signaling(self, msg: Message):
        """處理單一信令訊息。

        Args:
            msg: 從 Signaling Server 收到的 Message
        """
        if msg.type == MsgType.DISPATCHER_LIST:
            # 更新 dispatcher 列表（可能在 failover 時重新請求）
            self._dispatchers = msg.payload.get("dispatchers", [])
            logger.info("收到 dispatcher 列表: %d 台", len(self._dispatchers))

        elif msg.type == MsgType.ANSWER:
            # 收到 Dispatcher 的 SDP answer → 設定到 PeerConnection
            answer = RTCSessionDescription(
                sdp=msg.payload["sdp"],
                type=msg.payload["type"],
            )
            await self._pc.setRemoteDescription(answer)
            logger.info("已設定 Remote SDP Answer")

        elif msg.type == MsgType.ICE:
            # ICE trickle candidate（目前用 full ICE 所以不一定會觸發）
            # aiortc 在 setLocalDescription 時就收集完 candidates
            pass

    # ================================================================
    # Dispatcher 列表管理
    # ================================================================

    async def _fetch_dispatchers(self):
        """向 Signaling Server 請求可用的 Dispatcher 列表。

        會重試最多 30 秒（每 0.5 秒檢查一次），
        因為 dispatcher 可能還在啟動中。

        Raises:
            RuntimeError: 超過 30 秒仍無可用 dispatcher
        """
        req = Message(
            type=MsgType.REQUEST_DISPATCHERS,
            source_id=self.edge_id,
        )
        await self._ws.send_str(req.serialize())

        # 等待 signaling_loop 收到 DISPATCHER_LIST 並更新 self._dispatchers
        for _ in range(60):  # 60 × 0.5s = 30s
            if self._dispatchers:
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("30 秒內未收到可用的 dispatcher 列表")

    # ================================================================
    # WebRTC 連線建立
    # ================================================================

    async def _connect_to_dispatcher(self):
        """與當前索引的 Dispatcher 建立 WebRTC PeerConnection。

        完整流程：
        1. 建立 RTCPeerConnection（含 ICE 伺服器設定）
        2. 建立 Data Channel（unordered 模式，降低延遲）
        3. 註冊 data channel 事件（open/close/message）
        4. 建立 SDP Offer → 等待 ICE gathering → 透過 signaling 送出
        5. 等待 SDP Answer → data channel open

        Raises:
            RuntimeError:         dispatcher 列表為空
            asyncio.TimeoutError: ICE gathering 或 data channel open 超時
        """
        if not self._dispatchers:
            raise RuntimeError("Dispatcher 列表為空")

        disp = self._dispatchers[self._disp_idx]
        self._current_disp = disp["id"]
        logger.info("正在連接 Dispatcher: %s", self._current_disp)

        # ── 建立 PeerConnection ──
        ice_cfg = RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=s.urls,
                    username=s.username or None,
                    credential=s.credential or None,
                )
                for s in self.ice_servers
            ]
        )
        self._pc = RTCPeerConnection(configuration=ice_cfg)
        pc = self._pc  # 固定參考，避免 closure 在 failover 後讀到新的 self._pc

        # ── 建立 Data Channel ──
        # ordered=False: 不保證順序，降低延遲（影像幀不需要嚴格順序）
        self._dc = pc.createDataChannel("frames", ordered=False)

        # ── Data Channel 事件 ──
        @self._dc.on("open")
        def _on_open():
            """Data Channel 連線建立成功。"""
            logger.info("✓ Data Channel OPEN → %s", self._current_disp)
            self._connected.set()
            self._fail_count = 0  # 重置失敗計數

        @self._dc.on("close")
        def _on_close():
            """Data Channel 連線關閉（可能是對方斷線或網路問題）。"""
            logger.warning("✗ Data Channel CLOSED → %s", self._current_disp)
            self._connected.clear()

        @self._dc.on("message")
        def _on_message(raw):
            """收到來自 Dispatcher 的訊息（通常是推論結果）。

            Args:
                raw: str 或 bytes。推論結果為 JSON 字串（str）。
            """
            if isinstance(raw, str):
                try:
                    msg = Message.deserialize(raw)
                    if msg.type == MsgType.RESULT:
                        asyncio.create_task(self.on_result(msg.payload))
                    elif msg.type == MsgType.PONG:
                        # 健康檢查 PONG 回覆
                        rtt = time.time() - msg.payload.get("ping_ts", 0)
                        logger.debug("PONG from %s, RTT=%.1fms", self._current_disp, rtt * 1000)
                except Exception:
                    logger.exception("解析訊息失敗")

        # ── PeerConnection 狀態監控 ──
        @pc.on("connectionstatechange")
        async def _on_state():
            """監控 PeerConnection 連線狀態變化。

            狀態流程: new → connecting → connected → (disconnected → failed)
            當狀態變為 failed/disconnected/closed 時，增加失敗計數並可能觸發 failover。
            """
            state = pc.connectionState  # 用 pc（固定參考），不用 self._pc
            logger.info("PeerConnection 狀態: %s", state)
            if state in ("failed", "disconnected", "closed"):
                self._connected.clear()
                self._fail_count += 1
                if self._fail_count >= self.failover.max_failures:
                    await self._do_failover()

        # ── 建立 SDP Offer ──
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # ── 等待 ICE Gathering 完成 ──
        # aiortc 在 setLocalDescription 後開始收集 ICE candidates
        # 等待收集完成後才送出 offer（full ICE 模式，非 trickle）
        gathering_done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def _on_gather():
            if pc.iceGatheringState == "complete":
                gathering_done.set()

        # 可能已經收集完了（例如只有 host candidate）
        if pc.iceGatheringState == "complete":
            gathering_done.set()

        await asyncio.wait_for(gathering_done.wait(), timeout=10)
        logger.info("ICE gathering 完成")

        # ── 透過 Signaling 送出 Offer ──
        offer_msg = Message(
            type=MsgType.OFFER,
            source_id=self.edge_id,
            target_id=self._current_disp,
            payload={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            },
        )
        await self._ws.send_str(offer_msg.serialize())
        logger.info("已送出 SDP Offer → %s", self._current_disp)

        # ── 等待 Data Channel Open ──
        # signaling_loop 會收到 ANSWER 並設定 remote description
        # 之後 ICE 連線建立，data channel 自動 open
        await asyncio.wait_for(self._connected.wait(), timeout=15)
        logger.info("✓ WebRTC 連線建立完成: %s", self._current_disp)

    # ================================================================
    # 健康檢查
    # ================================================================

    async def _health_check_loop(self):
        """定期透過 Data Channel 發送 PING，監控連線品質。

        如果 data channel 處於 open 狀態，每隔 health_check_interval 秒
        發送一個 PING 訊息。Dispatcher 收到後會回覆 PONG。
        Edge 在 _on_message 中接收 PONG 並計算 RTT。

        此 loop 會持續運行直到被 cancel。
        """
        while True:
            try:
                await asyncio.sleep(self.failover.health_check_interval)
                if self._connected.is_set() and self._dc and self._dc.readyState == "open":
                    ping = Message(
                        type=MsgType.PING,
                        source_id=self.edge_id,
                        payload={"ping_ts": time.time()},
                    )
                    self._dc.send(ping.serialize())
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("健康檢查錯誤")

    # ================================================================
    # Failover 故障轉移
    # ================================================================

    async def _do_failover(self):
        """執行 Dispatcher failover：關閉當前連線，切換到下一台。

        流程：
        1. 取得 failover lock（防止並發觸發）
        2. 關閉當前 PeerConnection
        3. 索引移到下一台 dispatcher（循環式）
        4. 等待 recovery_delay（避免快速輪轉）
        5. 嘗試建立新的 WebRTC 連線
        6. 如果新連線也失敗，遞迴重試下一台

        Note:
            使用 asyncio.Lock 確保同一時間只有一個 failover 在執行。
        """
        async with self._failover_lock:
            old = self._current_disp
            logger.warning("觸發 Failover: %s", old)
            self._connected.clear()

            # 關閉舊的 PeerConnection
            if self._pc:
                await self._pc.close()

            # 切換到下一台 dispatcher（循環式）
            self._disp_idx = (self._disp_idx + 1) % len(self._dispatchers)
            self._fail_count = 0

            # 等待恢復延遲
            logger.info(
                "等待 %.1f 秒後連接 %s...",
                self.failover.recovery_delay,
                self._dispatchers[self._disp_idx]["id"],
            )
            await asyncio.sleep(self.failover.recovery_delay)

            try:
                await self._connect_to_dispatcher()
                logger.info("Failover 成功: %s → %s", old, self._current_disp)
            except Exception:
                logger.exception("Failover 連線失敗，將重試下一台...")
                # 遞迴重試（非遞迴呼叫，透過 create_task 避免 stack 溢出）
                asyncio.create_task(self._do_failover())
