"""跨元件通訊協議定義。

本模組定義了系統中所有元件之間通訊的統一訊息格式：
  - Message: JSON 序列化的控制/信令訊息（signaling、結果回傳等）。
  - pack_frame / unpack_frame: 二進位幀打包格式，用於 WebRTC data channel
    和 WebSocket binary 傳輸影像資料，避免 base64 編碼的 33% 開銷。

二進位幀格式:
  ┌──────────────┬──────────────────┬──────────────┐
  │ 2 bytes (BE) │ N bytes          │ M bytes      │
  │ header_len   │ JSON header      │ JPEG 影像    │
  └──────────────┴──────────────────┴──────────────┘
  - header 包含 frame_id, edge_id, seq 等中繼資訊
  - JPEG 影像為前處理後的壓縮圖片
"""

import json
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple


class MsgType(str, Enum):
    """訊息類型列舉。

    分為三大類：
    1. 信令類 (Signaling): 用於 WebRTC 連線建立
    2. 資料類 (Data):       用於幀傳輸和推論結果
    3. 健康檢查 (Health):   用於連線品質監控
    """

    # ── 信令類：Edge ↔ Signaling Server ↔ Dispatcher ──
    REGISTER = "register"                      # 元件向 signaling 註冊身份
    REQUEST_DISPATCHERS = "request_dispatchers" # Edge 請求可用 dispatcher 列表
    DISPATCHER_LIST = "dispatcher_list"         # Signaling 回覆 dispatcher 列表
    OFFER = "offer"                            # WebRTC SDP offer（Edge → Dispatcher）
    ANSWER = "answer"                          # WebRTC SDP answer（Dispatcher → Edge）
    ICE = "ice"                                # ICE candidate 交換（預留，目前用 full ICE）

    # ── 資料類：沿資料路徑傳遞 ──
    FRAME = "frame"                            # 影像幀（目前用 binary 打包，此 type 備用）
    RESULT = "result"                          # 推論結果 JSON

    # ── 健康檢查：Edge ↔ Dispatcher 透過 data channel ──
    PING = "ping"                              # 健康檢查請求
    PONG = "pong"                              # 健康檢查回覆（攜帶原始 timestamp 以計算 RTT）


@dataclass
class Message:
    """統一訊息格式。

    所有元件之間的 JSON 訊息都使用此格式：
    - type:      訊息類型（MsgType 列舉）
    - payload:   訊息內容（依 type 不同而不同）
    - source_id: 發送者 ID（如 "edge-001", "dispatcher-002"）
    - target_id: 接收者 ID（信令轉發時使用，非點對點則為空）
    - msg_id:    訊息唯一 ID（用於追蹤和除錯）
    - ts:        發送時的 Unix timestamp
    """

    type: MsgType
    payload: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    target_id: str = ""
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def serialize(self) -> str:
        """將 Message 序列化為 JSON 字串。

        MsgType enum 會被轉為其 value 字串，確保跨語言相容性。
        用於 WebSocket text message 和 WebRTC data channel text message。

        Returns:
            JSON 格式字串
        """
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d)

    @classmethod
    def deserialize(cls, raw: str) -> "Message":
        """從 JSON 字串反序列化為 Message 物件。

        Args:
            raw: 由 serialize() 產生的 JSON 字串

        Returns:
            還原的 Message 物件

        Raises:
            json.JSONDecodeError: JSON 格式錯誤
            ValueError:           type 欄位不是合法的 MsgType
        """
        d = json.loads(raw)
        d["type"] = MsgType(d["type"])
        return cls(**d)


# ---------------------------------------------------------------------------
# 二進位幀打包 / 解包
# 用於 WebRTC data channel 和 WebSocket binary 傳輸影像
# ---------------------------------------------------------------------------

def pack_frame(header: dict, jpeg_data: bytes) -> bytes:
    """將幀的 metadata header 和 JPEG 影像打包成單一二進位訊息。

    打包格式：
      [2 bytes, big-endian] header JSON 的位元組長度
      [N bytes]             header JSON（UTF-8 編碼，無多餘空白）
      [M bytes]             JPEG 影像原始位元組

    這比 base64 編碼高效約 33%，且保持 header 可讀性。

    Args:
        header: 幀中繼資訊字典，通常包含:
                - frame_id: 幀唯一 ID
                - edge_id:  來源 Edge 裝置 ID
                - seq:      幀序號
        jpeg_data: JPEG 壓縮後的影像位元組

    Returns:
        打包後的二進位資料，可直接透過 data channel 或 WS binary 傳送

    Raises:
        struct.error: header JSON 超過 65535 bytes（不太可能發生）
    """
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("!H", len(header_bytes)) + header_bytes + jpeg_data


def unpack_frame(data: bytes) -> Tuple[dict, bytes]:
    """解包二進位幀訊息，還原 header 和 JPEG 影像。

    與 pack_frame() 對應，解析格式為：
      [2B header_len][header JSON][JPEG data]

    Args:
        data: 由 pack_frame() 產生的二進位資料

    Returns:
        (header_dict, jpeg_bytes) 元組
        - header_dict: 幀中繼資訊
        - jpeg_bytes:  JPEG 影像原始位元組

    Raises:
        struct.error: 資料長度不足 2 bytes
        json.JSONDecodeError: header 部分不是合法 JSON
    """
    header_len = struct.unpack("!H", data[:2])[0]
    header = json.loads(data[2 : 2 + header_len])
    jpeg_data = data[2 + header_len :]
    return header, jpeg_data
