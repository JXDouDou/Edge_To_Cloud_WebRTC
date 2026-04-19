"""設定檔載入與型別定義。

本模組將 YAML 設定檔映射為 typed dataclass 結構。
每個元件只讀取自己需要的區塊，降低耦合度。

設定檔切換：
  - config/test.yaml: 本機測試（影片檔 + dummy model + localhost）
  - config/prod.yaml: 正式部署（Pi camera + 真實模型 + Tailscale/domain）

使用方式：
    from shared.config import load_config
    cfg = load_config("config/test.yaml")
    print(cfg.edge.capture.fps)
"""

import typing
import yaml
from dataclasses import dataclass, field
from typing import List


# ====================================================================
# 葉節點設定（Leaf configs）—— 各功能區塊的參數
# ====================================================================

@dataclass
class CaptureConfig:
    """影像擷取設定。

    Attributes:
        mode:   "video" 使用影片檔（測試用），"camera" 使用攝影機（Pi 生產用）
        source: 影片檔路徑（mode=video）或攝影機索引/裝置路徑（mode=camera）
        fps:    輸出幀率上限，capture 模組會自動節流
        width:  攝影機解析度寬度（僅 camera 模式生效）
        height: 攝影機解析度高度（僅 camera 模式生效）
    """
    mode: str = "video"
    source: str = "test_data/test_video.mp4"
    fps: int = 5
    width: int = 640
    height: int = 480


@dataclass
class ROIConfig:
    """ROI（Region of Interest）感興趣區域裁切設定。

    Attributes:
        enabled: 是否啟用 ROI 裁切
        x:       ROI 左上角 x 座標（像素）
        y:       ROI 左上角 y 座標（像素）
        width:   ROI 寬度（像素）
        height:  ROI 高度（像素）
    """
    enabled: bool = False
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 480


@dataclass
class PreprocessConfig:
    """前處理設定：ROI 裁切 + 可選 resize + JPEG 壓縮。

    Attributes:
        roi:           ROI 裁切設定
        jpeg_quality:  JPEG 壓縮品質（1-100），越低檔案越小但品質越差
                       建議值：80（測試）/ 85（生產）
        resize_width:  resize 目標寬度，設為 0 表示不 resize
        resize_height: resize 目標高度，設為 0 表示不 resize
    """
    roi: ROIConfig = field(default_factory=ROIConfig)
    jpeg_quality: int = 80
    resize_width: int = 0
    resize_height: int = 0


@dataclass
class ICEServer:
    """WebRTC ICE 伺服器設定（STUN / TURN）。

    STUN: 僅用於發現自身公網 IP（免費，如 Google STUN）
    TURN: 中繼伺服器，當 P2P 打洞失敗時作為備援（需自建，如 coturn）

    ⚠️ 5G 的 symmetric NAT 很可能需要 TURN 才能穿透！
       測試時 localhost 不需要，但生產部署強烈建議配置 TURN。

    Attributes:
        urls:       STUN/TURN 伺服器 URL（如 "stun:stun.l.google.com:19302"）
        username:   TURN 認證使用者名稱（STUN 留空）
        credential: TURN 認證密碼（STUN 留空）
    """
    urls: str = "stun:stun.l.google.com:19302"
    username: str = ""
    credential: str = ""


@dataclass
class SignalingConfig:
    """信令伺服器連線設定。

    Attributes:
        url: 完整 WebSocket URL。
             測試: "ws://localhost:8080/ws"
             生產: "wss://signaling.yourdomain.com/ws"（需 TLS）
    """
    url: str = "ws://localhost:8080/ws"


@dataclass
class FailoverConfig:
    """Edge 端 dispatcher 故障轉移設定。

    Edge 會持續監控與當前 dispatcher 的連線品質，
    當連續失敗次數達到 max_failures 時，自動切換到下一台 dispatcher。

    Attributes:
        health_check_interval: 健康檢查間隔（秒），定期透過 data channel 發 PING
        max_failures:          允許的最大連續失敗次數，超過則觸發 failover
        recovery_delay:        failover 後等待多久再嘗試連線下一台（秒），
                               避免快速輪轉造成雪崩
    """
    health_check_interval: float = 2.0
    max_failures: int = 3
    recovery_delay: float = 3.0


# ====================================================================
# 元件設定（Component configs）
# ====================================================================

@dataclass
class EdgeConfig:
    """Edge 裝置完整設定。

    Attributes:
        id:          Edge 唯一識別碼（多台 Edge 時用於路由識別）
        capture:     影像擷取設定
        preprocess:  前處理設定
        signaling:   信令伺服器連線設定
        ice_servers: STUN/TURN 伺服器列表
        failover:    dispatcher 故障轉移設定
    """
    id: str = "edge-001"
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    signaling: SignalingConfig = field(default_factory=SignalingConfig)
    ice_servers: List[ICEServer] = field(default_factory=lambda: [ICEServer()])
    failover: FailoverConfig = field(default_factory=FailoverConfig)


@dataclass
class DispatcherConfig:
    """Dispatcher（EC2）設定。

    Attributes:
        id:               Dispatcher 唯一識別碼
        signaling:        信令伺服器連線設定
        inference_ws_url: 推論伺服器的 WebSocket URL。
                          ⚠️ 如果是走 Tailscale，請填 Tailscale hostname:
                          例如 "ws://desktop-5080:8765/ws"
                          或 Tailscale IP: "ws://100.x.x.x:8765/ws"
        ice_servers:      STUN/TURN 伺服器列表（與 Edge 側需一致）
    """
    id: str = "dispatcher-001"
    signaling: SignalingConfig = field(default_factory=SignalingConfig)
    inference_ws_url: str = "ws://localhost:8765/ws"
    ice_servers: List[ICEServer] = field(default_factory=lambda: [ICEServer()])


@dataclass
class InferenceConfig:
    """推論伺服器設定。

    Attributes:
        id:         推論伺服器唯一識別碼
        host:       監聽位址。Tailscale 環境下建議 "0.0.0.0" 讓 Tailscale IP 可達
        port:       監聽埠號
        model_type: 模型類型 "dummy"（測試）| "yolo"（YOLO）| "custom"（自訂）
        model_path: 模型檔案路徑（dummy 模式可留空）
        device:     推論裝置 "cpu" | "cuda" | "cuda:0"
    """
    id: str = "inference-001"
    host: str = "0.0.0.0"
    port: int = 8765
    model_type: str = "dummy"
    model_path: str = ""
    device: str = "cpu"


# ====================================================================
# 根設定（Root config）
# ====================================================================

@dataclass
class AppConfig:
    """應用程式根設定，包含所有元件的設定。

    YAML 設定檔的頂層結構直接對應此 dataclass。

    Attributes:
        mode:           "test" | "production"，目前用於日誌和人工識別
        signaling_host: signaling server 監聽位址
        signaling_port: signaling server 監聽埠號
        edge:           Edge 裝置設定
        dispatchers:    Dispatcher 列表（可配置多台）
        inference:      推論伺服器設定
    """
    mode: str = "test"
    signaling_host: str = "0.0.0.0"
    signaling_port: int = 8080
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    dispatchers: List[DispatcherConfig] = field(default_factory=list)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ====================================================================
# YAML 載入工具
# ====================================================================

def _dict_to_dataclass(cls, data: dict):
    """遞迴地將巢狀 dict 轉換為 dataclass 實例。

    處理邏輯：
    1. 遍歷 dataclass 的每個 field
    2. 如果 field 類型是 List[SomeDataclass]，對 list 中的每個 dict 遞迴轉換
    3. 如果 field 類型是另一個 dataclass，對該 dict 遞迴轉換
    4. 其他基本型別直接賦值

    Args:
        cls:  目標 dataclass 類別
        data: 從 YAML 解析出的 dict

    Returns:
        轉換後的 dataclass 實例
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    fields = cls.__dataclass_fields__
    kwargs = {}
    for name, f in fields.items():
        if name not in data:
            continue
        val = data[name]
        ft = f.type

        # 字串型別註解（使用了 from __future__ import annotations 時會出現）
        # 在此模組中我們沒用 future annotations，所以回退為直接賦值
        if isinstance(ft, str):
            ft = globals().get(ft, ft)
            if isinstance(ft, str):
                kwargs[name] = val
                continue

        # 處理 List[X] 型別
        origin = typing.get_origin(ft)
        args = typing.get_args(ft)
        if origin is list and args and isinstance(val, list):
            item_type = args[0]
            if hasattr(item_type, "__dataclass_fields__"):
                # List 中的每個 dict → dataclass
                val = [
                    _dict_to_dataclass(item_type, v) if isinstance(v, dict) else v
                    for v in val
                ]
        elif hasattr(ft, "__dataclass_fields__") and isinstance(val, dict):
            # 巢狀 dataclass
            val = _dict_to_dataclass(ft, val)

        kwargs[name] = val
    return cls(**kwargs)


def load_config(path: str) -> AppConfig:
    """從 YAML 檔案載入設定。

    完整流程：
    1. 讀取 YAML 檔案
    2. 解析為 Python dict
    3. 遞迴轉換為 typed dataclass 結構

    Args:
        path: YAML 設定檔路徑（相對或絕對路徑皆可）

    Returns:
        AppConfig 實例，所有欄位都有型別保證

    Raises:
        FileNotFoundError: 設定檔不存在
        yaml.YAMLError:    YAML 格式錯誤
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _dict_to_dataclass(AppConfig, data)
