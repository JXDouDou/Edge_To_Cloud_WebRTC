"""幀前處理模組：ROI 裁切、resize、JPEG 壓縮。

此模組將攝影機原始幀處理成適合網路傳輸的 JPEG 位元組：
1. ROI 裁切:    只保留感興趣的區域，減少無用資料
2. Resize:      縮放到目標尺寸，降低資料量
3. JPEG 壓縮:   有損壓縮，在品質和大小之間取得平衡

典型的 640×480 JPEG（quality=80）大約 30-80 KB，
遠小於 WebRTC data channel 的 256 KB 訊息上限。
"""

import cv2
import numpy as np
from shared.config import PreprocessConfig


class Preprocessor:
    """影像前處理器。

    使用方式：
        pp = Preprocessor(config)
        jpeg_bytes = pp.process(frame)  # numpy.ndarray → bytes
    """

    def __init__(self, config: PreprocessConfig):
        """初始化前處理器。

        Args:
            config: PreprocessConfig 設定物件，
                    包含 ROI、resize、JPEG 品質等參數
        """
        self.config = config

    def process(self, frame: np.ndarray) -> bytes:
        """對單幀影像進行前處理並壓縮為 JPEG。

        處理順序：ROI 裁切 → resize → JPEG 編碼。
        這個順序很重要：先裁切再縮放可以避免對無用區域做縮放運算。

        Args:
            frame: OpenCV BGR 格式的影像（numpy.ndarray, shape: H×W×3）

        Returns:
            JPEG 壓縮後的位元組（bytes），可直接用 pack_frame() 打包傳送

        Raises:
            RuntimeError: JPEG 編碼失敗（通常是因為 frame 為 None 或格式錯誤）
        """
        # ── 步驟 1: ROI 裁切 ──
        # 如果啟用 ROI，只保留指定區域
        # 使用 numpy array slicing，零拷貝操作，非常高效
        if self.config.roi.enabled:
            r = self.config.roi
            frame = frame[r.y : r.y + r.height, r.x : r.x + r.width]

        # ── 步驟 2: Resize ──
        # 只在 resize_width 和 resize_height 都 > 0 時才執行
        # 設為 0 表示保持原始尺寸（或 ROI 裁切後的尺寸）
        rw, rh = self.config.resize_width, self.config.resize_height
        if rw > 0 and rh > 0:
            frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # ── 步驟 3: JPEG 壓縮 ──
        # quality 範圍 1-100，建議值：
        #   - 80: 一般測試，檔案小
        #   - 85: 生產環境，品質與大小平衡
        #   - 90+: 高品質需求（但檔案顯著變大）
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG 編碼失敗，請確認輸入 frame 格式正確")

        return buf.tobytes()
