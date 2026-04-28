"""影像擷取模組：支援影片檔（測試）和攝影機（生產）。

此模組負責：
1. 根據設定開啟影片來源（檔案或攝影機）
2. 以設定的 FPS 上限節流輸出幀率
3. 影片檔模式下自動迴圈播放（測試用）

測試模式：
    設定 capture.mode = "video"，將影片檔放在 test_data/ 目錄。
    Edge 會讀取影片並以設定的 fps 速率送出，影片播完自動從頭開始。

生產模式（Raspberry Pi 5）：
    設定 capture.mode = "camera"，source 填攝影機索引（通常 "0"）。
    OpenCV 會透過 V4L2 後端存取 Pi Camera Module。
    如果用 libcamera，source 可填裝置路徑如 "/dev/video0"。
"""

import time
import logging
import cv2
from shared.config import CaptureConfig

logger = logging.getLogger(__name__)


class FrameCapture:
    """影像擷取器，封裝 OpenCV VideoCapture 並加入 FPS 節流。

    使用方式：
        capture = FrameCapture(config)
        capture.open()
        while True:
            ok, frame = capture.read()  # 會自動按 fps 節流
            if not ok:
                break
        capture.release()
    """

    def __init__(self, config: CaptureConfig):
        """初始化擷取器。

        Args:
            config: CaptureConfig 設定物件，包含 mode、source、fps 等參數
        """
        self.config = config
        self._cap: cv2.VideoCapture = None

        # 計算每幀的最小間隔時間（秒），用於 FPS 節流
        # 例如 fps=5 → _frame_interval=0.2 秒
        self._frame_interval = 1.0 / max(config.fps, 1)
        self._last_frame_time = 0.0

        # 影片模式：播完自動迴圈；攝影機模式：不迴圈
        self._loop = config.mode == "video"

    def open(self):
        """開啟影像來源。

        根據 config.mode 決定來源：
        - "video": 直接用檔案路徑開啟
        - "camera": 將字串索引轉為整數（如 "0" → 0），並設定解析度

        Raises:
            RuntimeError: 無法開啟影像來源（檔案不存在、攝影機未連接等）
        """
        src = self.config.source

        if self.config.mode == "camera":
            # 攝影機索引通常是數字字串，轉為 int 讓 OpenCV 識別
            src = int(src) if src.isdigit() else src

        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"無法開啟影像來源: {src}\n"
                f"  模式: {self.config.mode}\n"
                f"  如果是測試模式，請確認影片檔存在於: {self.config.source}\n"
                f"  可用 python test_data/generate_test_video.py 產生測試影片"
            )

        # 攝影機模式下嘗試設定解析度與格式
        # （影片模式忽略，使用影片原生解析度）
        if self.config.mode == "camera":
            # ── 重要：指定 fourcc=MJPG ──────────────────────────
            # OpenCV 預設使用 YUYV（未壓縮），在 USB 2.0 攝影機上頻寬會
            # 卡住，例如 1920x1080 YUYV 通常被驅動限制到 5 fps。
            # 換成 MJPG（相機端壓 JPEG 後傳）可大幅降低 USB 頻寬，
            # 同樣 1920x1080 通常能跑到 30 fps。
            # OpenCV 會自動用 libjpeg 解 MJPG 為 BGR ndarray，
            # 對下游程式（preprocess、encode）完全透明。
            #
            # 不是所有相機都支援 MJPG，但若驅動支援，這個 set 會成功；
            # 不支援時 OpenCV 會 silently 退回 YUYV，不影響運作。
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            # 同時告訴驅動目標 fps，讓它選擇對應的 frame interval
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # 讀取實際解析度與格式作為日誌
        # （驅動可能拒絕我們要求的設定而退回別的，這裡看實際生效的值）
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        # FOURCC 是 32-bit 整數，要解回 4 個 ASCII 字元
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(
            chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)
        ) if fourcc_int else "?"
        logger.info(
            "擷取器開啟: source=%s, mode=%s, requested_fps=%d, "
            "actual=%dx%d @ %.1f fps, fourcc=%s",
            src, self.config.mode, self.config.fps,
            actual_w, actual_h, actual_fps, actual_fourcc,
        )

    def read(self):
        """讀取一幀影像，遵守 FPS 上限節流。

        此方法會在必要時 sleep，確保輸出幀率不超過設定的 fps。
        影片模式下播完會自動從頭開始。

        Returns:
            (ok, frame) 元組：
            - ok:    bool，是否成功讀取
            - frame: numpy.ndarray (H, W, 3) BGR 格式，或 None（讀取失敗時）

        Note:
            此方法會阻塞（sleep），不應在 async event loop 中直接呼叫。
            建議用 asyncio.to_thread(capture.read) 包裝。
        """
        # ── FPS 節流 ──
        elapsed = time.time() - self._last_frame_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        # ── 讀取幀 ──
        ret, frame = self._cap.read()

        # 影片模式：播完自動從頭迴圈
        if not ret and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if ret:
                logger.debug("影片迴圈重播")

        if ret:
            self._last_frame_time = time.time()
        return ret, frame

    def release(self):
        """釋放影像來源資源。

        應在程式結束或切換影像來源時呼叫，
        避免攝影機被長期佔用。
        """
        if self._cap:
            self._cap.release()
            logger.info("擷取器已釋放")
