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
        mode:        "video" 使用影片檔（測試用），"camera" 使用攝影機（Pi 生產用）
        source:      影片檔路徑（mode=video）或攝影機索引/裝置路徑（mode=camera）
        fps:         輸出幀率上限，capture 模組會自動節流
        width:       攝影機解析度寬度（僅 camera 模式生效）
        height:      攝影機解析度高度（僅 camera 模式生效）
        loop:        影片播完是否自動從頭重播（僅 video 模式生效）。
                     預設 True 維持原本行為；改 False 可讓影片播完就停止，
                     方便 debug「跑一輪要花多久」「播完後系統怎麼收尾」等問題。
        sample_mode: 影片取樣模式（僅 video 模式生效，camera 模式無視）：
                     - "real_time"   原速播放 + 降取樣到 target fps（預設，模擬真實相機）
                                     94 秒影片 + target=5 fps → 94 秒跑完，產生 ~470 幀
                     - "all_frames"  每格都送 + 節流到 target fps（影片變慢播）
                                     94 秒 30fps 影片 + target=5 fps → 564 秒跑完，全部 2820 幀
                     - "fast_replay" 每格都送 + 不節流（全速灌幀）
                                     用於壓力測試或快速倒帶看 prediction 走勢
    """
    mode: str = "video"
    source: str = "test_data/test_video.mp4"
    fps: int = 5
    width: int = 640
    height: int = 480
    loop: bool = True
    sample_mode: str = "real_time"


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
class ROIProfile:
    """具名 ROI + resize 預設組合，對應一種模型的輸入需求。

    Attributes:
        name:          識別名稱（如 "keras", "pytorch"），對應 active_profile
        roi:           該模型要用的 ROI 裁切設定
        resize_width:  resize 目標寬度，0 = 不 resize（直接送裁切後的尺寸）
        resize_height: resize 目標高度，0 = 不 resize
    """
    name: str = ""
    roi: ROIConfig = field(default_factory=ROIConfig)
    resize_width: int = 0
    resize_height: int = 0


@dataclass
class PreprocessConfig:
    """前處理設定：ROI 裁切 + 可選 resize + JPEG 壓縮。

    支援 ROI profile 切換：在 roi_profiles 定義多組 ROI 預設，
    用 active_profile 指定當前使用哪一組。切模型時只要改
    active_profile + inference.model_type 兩個值。

    向後相容：沒設 active_profile 時退回使用頂層 roi / resize_* 欄位。

    Attributes:
        roi:            ROI 裁切設定（未使用 profile 時的預設值）
        jpeg_quality:   JPEG 壓縮品質（1-100）
        resize_width:   resize 目標寬度，0 = 不 resize
        resize_height:  resize 目標高度，0 = 不 resize
        roi_profiles:   具名 ROI 預設組合列表
        active_profile: 當前啟用的 profile 名稱，空字串 = 不使用 profile
    """
    roi: ROIConfig = field(default_factory=ROIConfig)
    jpeg_quality: int = 80
    resize_width: int = 0
    resize_height: int = 0
    roi_profiles: List[ROIProfile] = field(default_factory=list)
    active_profile: str = ""


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

    Edge 會持續監控與當前 dispatcher 的連線品質：
      - 每 health_check_interval 秒透過 data channel 發 PING
      - 若超過 (max_failures × health_check_interval) 秒沒收到任何 PONG，視為失聯
      - 觸發 failover：關閉舊連線 → 等 recovery_delay → 連下一台 dispatcher

    Failover 三階段時間構成：
      A. 偵測          = max_failures × health_check_interval  秒
      B. 內部處理      = recovery_delay (主要) + ~0.5s 雜項
      C. 新連線建立    = 2-4 秒（Tailscale 直連時；其他環境更久）
      合計              = A + B + C

    各環境推薦值（在 staging*.yaml 直接覆寫）：

      ┌────────────────────────┬───────┬──────────────┬──────────────┬─────────┐
      │ 環境                   │ hci   │ max_failures │ recovery_dly │ 總切換  │
      ├────────────────────────┼───────┼──────────────┼──────────────┼─────────┤
      │ 本機 / LAN demo        │ 1.0   │ 2            │ 1.5          │ ~5s     │
      │ Tailscale 穩定網路     │ 1.5   │ 3            │ 2.0          │ ~7s     │
      │ 目前 5G + Tailscale ★  │ 2.0   │ 3            │ 3.0          │ ~11s    │
      │ 5G 直連（無 Tailscale）│ 3.0   │ 4            │ 5.0          │ ~18s    │
      │ 不穩定衛星 / 跨洲      │ 5.0   │ 5            │ 8.0          │ ~35s    │
      └────────────────────────┴───────┴──────────────┴──────────────┴─────────┘

    調整指南：
      - max_failures < 2  → 5G 抖動就誤判，會 pong-pong failover，不推薦
      - recovery_delay < 1.5 → aiortc/aioice 內部 cleanup 可能還沒做完，新連線 timeout 機率大幅上升
      - 想最激進 5-7s 切換：先確認網路真的穩，否則弄巧成拙

    Attributes:
        health_check_interval: 健康檢查間隔（秒），定期透過 data channel 發 PING。
                               同時是 PONG 缺席計時的單位。
        max_failures:          允許的最大連續無回應次數（PONG 漏接次數），超過則觸發 failover。
        recovery_delay:        failover 中等待 ICE / aiortc cleanup 的延遲（秒）。
                               太短 → 新連線 timeout；太長 → 切換變慢。
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
class FrameBufferConfig:
    """Dispatcher 每個 edge 的 frame buffer 策略。

    用來在 inference 處理速度跟不上 frame 進來速度時，決定要丟哪些 frame。
    三種模式跑同一段影片可做 A/B 比較，看哪個對 prediction 品質最有利。

    Modes:
      fifo (預設)
        不丟，每個 frame 都用 asyncio.create_task 並行轉發到 inference。
        WebSocket TCP buffer 滿了就 backpressure。
        過載時 latency 持續成長，但所有 frame 都會被處理。

      drop_oldest
        為每個 edge 建立 asyncio.Queue(maxsize=max_size)。
        Queue 滿時把最舊的丟掉再放新的。
        用 max_size 控制積壓上限（推薦 2-3）。
        ★ 對「即時取樣」場景（液面、溫度監控）最合理：永遠處理最新狀態。

      latest_only
        永遠只保留最新一張（單槽），等於 drop_oldest + max_size=1。
        Inference 一空閒就拿最新的，但會丟掉大量中間 frame。
        延遲最低，但取樣稀疏。

    比較這三種模式建議：
      1. edge 用 capture.sample_mode=fast_replay 灌滿 inference
      2. 三種模式各跑一次
      3. 比 metrics 的 latency / completion_rate / prediction 走勢

    Attributes:
        mode:      "fifo" | "drop_oldest" | "latest_only"
        max_size:  drop_oldest 模式的佇列上限（其他模式無視）。預設 3。
                   1 = 單槽（等同 latest_only）
                   3 = 容忍 600ms 抖動 (5fps × 3)
                   10 = 接近 fifo，drop 少但 latency 高
    """
    mode: str = "fifo"
    max_size: int = 3


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
        frame_buffer:     Frame buffer 策略（fifo / drop_oldest / latest_only）
    """
    id: str = "dispatcher-001"
    signaling: SignalingConfig = field(default_factory=SignalingConfig)
    inference_ws_url: str = "ws://localhost:8765/ws"
    ice_servers: List[ICEServer] = field(default_factory=lambda: [ICEServer()])
    frame_buffer: FrameBufferConfig = field(default_factory=FrameBufferConfig)


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
