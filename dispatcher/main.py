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
# TURN 個別 channel_bind 偶爾失敗（ExpressTURN 拒絕私有 IP 等），
# asyncio 會印 "Task exception was never retrieved" 加 traceback。
# 由於資料實際走 Tailscale 不走 TURN，這些是純雜訊，壓掉。
# Debug 時改回 logging.WARNING 即可看到。
logging.getLogger("aioice.turn").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
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

        # ── Frame buffer 模式（fifo / drop_oldest / latest_only）──
        # 由 config.frame_buffer 控制。詳見 FrameBufferConfig docstring。
        self._buffer_mode = config.frame_buffer.mode
        self._buffer_max_size = config.frame_buffer.max_size
        if self._buffer_mode not in ("fifo", "drop_oldest", "latest_only"):
            raise ValueError(
                f"frame_buffer.mode 必須是 fifo / drop_oldest / latest_only，"
                f"收到: {self._buffer_mode}"
            )
        logger.info(
            "Frame buffer 模式: %s (max_size=%d)",
            self._buffer_mode,
            1 if self._buffer_mode == "latest_only" else self._buffer_max_size,
        )

        # drop_oldest / latest_only 用：per-edge queue + worker task
        self._frame_queues: dict = {}    # edge_id → asyncio.Queue
        self._worker_tasks: dict = {}    # edge_id → asyncio.Task

        # ── Debug 計數器：每 N 幀印一次 log，確認資料流動 ──
        self._frames_received: dict = {}   # edge_id → 從 edge 收到的幀數
        self._frames_forwarded: dict = {}  # edge_id → 成功轉發到 inference 的幀數
        self._frames_dropped: dict = {}    # edge_id → 因 buffer 滿被丟掉的幀數（drop_oldest/latest_only 才有）
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
        # 取消所有 frame worker
        for edge_id, task in self._worker_tasks.items():
            if not task.done():
                task.cancel()
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

        # ── 清理舊連線（同一個 edge_id 重連時）──
        # Edge Ctrl+C 後重連，會再送 OFFER 過來。
        # 必須先把舊的 PeerConnection 顯式關閉並從追蹤表移除，
        # 否則舊 pc 在轉換到 closed 狀態時的 handler 會把新 pc 的條目誤刪。
        old_pc = self._peers.pop(edge_id, None)
        self._channels.pop(edge_id, None)
        # 順便清掉舊的 debug 計數器（重新從 0 開始）
        self._frames_received.pop(edge_id, None)
        self._frames_forwarded.pop(edge_id, None)
        self._frames_dropped.pop(edge_id, None)
        self._results_returned.pop(edge_id, None)
        # 清掉舊的 queue 跟 worker（避免新 worker 從舊 queue 拿）
        old_worker = self._worker_tasks.pop(edge_id, None)
        if old_worker is not None and not old_worker.done():
            old_worker.cancel()
        self._frame_queues.pop(edge_id, None)
        if old_pc is not None:
            logger.info("清理舊的 PeerConnection: edge=%s", edge_id)
            try:
                await old_pc.close()
            except Exception:
                logger.exception("關閉舊 PeerConnection 失敗: edge=%s", edge_id)

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

        # ── Debug: 印出設定的 ICE servers ──
        logger.info(
            "ICE servers 設定 (%d 個): %s",
            len(self.config.ice_servers),
            [s.urls for s in self.config.ice_servers],
        )

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

                    # 二進位 = 影像幀 → 依 buffer 模式處理
                    self._enqueue_frame(edge_id, data)
                elif isinstance(data, str):
                    # 文字 = 控制訊息（目前只有 PING）
                    asyncio.create_task(self._handle_dc_text(edge_id, channel, data))

            # 為這個 edge 啟動 worker（drop_oldest / latest_only 模式才需要）
            self._ensure_worker(edge_id)

        # ── PeerConnection 狀態監控 ──
        @pc.on("connectionstatechange")
        async def on_state():
            """監控與 Edge 的 WebRTC 連線狀態。

            當連線 failed/closed 時，清理該 Edge 的追蹤資料——
            但只有當前條目仍指向此 pc 時才清理，避免「舊 pc 關閉時
            把新 pc 的條目誤刪」的 race condition（edge Ctrl+C 重連時會踩到）。
            """
            state = pc.connectionState
            logger.info("Edge %s PeerConnection 狀態: %s", edge_id, state)
            if state in ("failed", "closed"):
                # 只在此 pc 仍是當前條目時才清，避免覆蓋新連線
                if self._peers.get(edge_id) is pc:
                    self._peers.pop(edge_id, None)
                    self._channels.pop(edge_id, None)
                    logger.info("已清理 Edge 追蹤資料: %s", edge_id)
                else:
                    logger.info("舊 pc 關閉，當前條目已是新 pc，跳過清理: %s", edge_id)

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

        # ── Debug: 列出本地收集到的 ICE candidates ──
        # 從 SDP 中解析 candidate 行，看有沒有 typ relay（= TURN 成功）
        sdp = pc.localDescription.sdp
        host_count = sdp.count("typ host")
        srflx_count = sdp.count("typ srflx")
        relay_count = sdp.count("typ relay")
        logger.info(
            "本地 ICE candidates: host=%d, srflx(STUN)=%d, relay(TURN)=%d",
            host_count, srflx_count, relay_count,
        )
        if relay_count == 0 and len(self.config.ice_servers) > 1:
            logger.warning(
                "⚠️ 設定了 TURN 但沒收集到 relay candidate，TURN 可能失效或不可達"
            )

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

    # ================================================================
    # Frame buffer 三種模式
    # ================================================================

    def _enqueue_frame(self, edge_id: str, raw_frame: bytes):
        """依 frame_buffer.mode 把 frame 排進對應的處理路徑。

        - fifo:         直接 create_task 並行轉發（不過 queue）
        - drop_oldest:  put 進 bounded queue，滿了就先 get 掉最舊的
        - latest_only:  put 進 maxsize=1 queue，滿了就替換
        """
        if self._buffer_mode == "fifo":
            # 原本行為：每個 frame 一個 task，全部並行送
            asyncio.create_task(self._forward_to_inference(edge_id, raw_frame))
            return

        # 取或建 queue
        q = self._frame_queues.get(edge_id)
        if q is None:
            maxsize = 1 if self._buffer_mode == "latest_only" else self._buffer_max_size
            q = asyncio.Queue(maxsize=maxsize)
            self._frame_queues[edge_id] = q

        # Queue 滿了：丟最舊的
        if q.full():
            try:
                q.get_nowait()
                dropped = self._frames_dropped.get(edge_id, 0) + 1
                self._frames_dropped[edge_id] = dropped
                if dropped % self._log_every == 0:
                    logger.info(
                        "✂ buffer 滿丟舊 frame: edge=%s, dropped 累計=%d (mode=%s)",
                        edge_id, dropped, self._buffer_mode,
                    )
            except asyncio.QueueEmpty:
                pass

        # 不 await（_enqueue_frame 是 sync 回呼），直接 nowait
        try:
            q.put_nowait(raw_frame)
        except asyncio.QueueFull:
            # 理論上不會（前面已 get），但保險
            self._frames_dropped[edge_id] = self._frames_dropped.get(edge_id, 0) + 1

    def _ensure_worker(self, edge_id: str):
        """確保 edge 有一個 worker task 在處理它的 queue。

        FIFO 模式不需要 worker（直接 create_task），只在 queue 模式建立。
        """
        if self._buffer_mode == "fifo":
            return

        task = self._worker_tasks.get(edge_id)
        if task is not None and not task.done():
            return  # 已有 worker

        self._worker_tasks[edge_id] = asyncio.create_task(
            self._frame_worker(edge_id)
        )
        logger.info("啟動 frame worker: edge=%s, mode=%s",
                    edge_id, self._buffer_mode)

    async def _frame_worker(self, edge_id: str):
        """單 edge 的 worker，從 queue 拿 frame 序列轉發給 inference。

        為什麼用 worker 而不是 create_task 並行：
          drop_oldest / latest_only 的重點是「同一時間只有 1 個 frame in-flight」。
          並行轉發會讓 dropping 邏輯失效（多個 task 同時 await send_bytes，無法控制積壓）。
          single worker 確保 inference 一空閒就拿最新的 frame，達成「永遠處理最新狀態」。
        """
        q = self._frame_queues[edge_id]
        while self._running:
            try:
                raw_frame = await q.get()
            except asyncio.CancelledError:
                break

            # edge 已斷線就跳出
            if edge_id not in self._channels:
                logger.info("frame worker 結束: edge=%s 已斷線", edge_id)
                return

            try:
                await self._forward_to_inference(edge_id, raw_frame)
            except Exception:
                logger.exception("worker 轉發失敗: edge=%s", edge_id)

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

        # ── 斷線自動重連（無限重試，指數退避到上限 60s）──
        # 之前的版本只重試 2 次就放棄；若 Inference 停超過 13 秒，
        # dispatcher 會靜默死掉，必須手動重啟才能恢復。
        # 改成無限重試後：Inference 掛掉一天再回來，dispatcher 還是會自己接上。
        delay = 3
        max_delay = 60
        while self._running:
            logger.warning("Inference 連線斷開，%d 秒後重連...", delay)
            await asyncio.sleep(delay)
            try:
                await self._connect_inference()
                logger.info("Inference 重連成功")
                return  # _connect_inference 內部已啟動新的 recv_loop，此函式可結束
            except Exception as e:
                logger.warning(
                    "Inference 重連失敗 (%s)，%d 秒後再試...",
                    type(e).__name__, min(delay * 2, max_delay),
                )
                delay = min(delay * 2, max_delay)


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
