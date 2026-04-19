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

        # 攝影機模式下嘗試設定解析度（影片模式忽略，使用影片原始解析度）
        if self.config.mode == "camera":
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        # 讀取實際解析度作為日誌
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "擷取器開啟: source=%s, mode=%s, fps=%d, resolution=%dx%d",
            src, self.config.mode, self.config.fps, actual_w, actual_h,
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
