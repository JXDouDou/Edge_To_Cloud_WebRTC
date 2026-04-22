"""模型抽象層：統一的推論介面。

此模組定義了推論模型的抽象介面 (BaseModel)，
並提供了幾種內建實作：
  - DummyModel:  假模型，回傳固定結果（用於管線測試，不需要 GPU）
  - YOLOModel:   Ultralytics YOLO 包裝器（支援 YOLOv5/v8/v11 等）

擴展方式：
  1. 繼承 BaseModel
  2. 實作 predict(jpeg_data) → dict
  3. 在 create_model() 中註冊新的 model_type

設計理念：
  - 模型介面只接收 JPEG bytes，不依賴特定影像格式
  - 回傳標準化的 dict 結構，方便 controller 統一處理
  - predict() 是同步方法，由呼叫端決定是否放到 thread pool 執行
"""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """推論模型抽象基底類別。

    所有模型實作都必須繼承此類別並實作 predict() 方法。

    回傳格式約定::

        {
            "detections": [
                {
                    "class": "person",           # 類別名稱
                    "confidence": 0.92,           # 信心度 (0.0 ~ 1.0)
                    "bbox": [x1, y1, x2, y2],    # 邊界框 (左上, 右下) 像素座標
                },
                ...
            ],
            "image_size": [width, height],        # 可選：原圖尺寸
        }
    """

    @abstractmethod
    def predict(self, jpeg_data: bytes) -> dict:
        """對 JPEG 影像執行推論。

        Args:
            jpeg_data: JPEG 壓縮後的影像位元組

        Returns:
            推論結果 dict，至少包含 "detections" 列表
        """


class DummyModel(BaseModel):
    """假模型：回傳固定的偵測結果。

    用途：
    - 不需要 GPU 或模型檔案即可測試完整管線
    - 驗證 Edge → Dispatcher → Inference → Dispatcher → Edge 的資料流
    - 效能基準測試（測量純粹的網路延遲，排除推論時間）
    """

    def predict(self, jpeg_data: bytes) -> dict:
        """回傳一個假的偵測結果。

        會嘗試解碼 JPEG 以取得影像尺寸（驗證資料完整性），
        但不做實際推論。

        Args:
            jpeg_data: JPEG 影像位元組

        Returns:
            包含一個 "dummy" 類別偵測的結果 dict
        """
        # 解碼影像以驗證 JPEG 資料完整性並取得尺寸
        img = cv2.imdecode(
            np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
        )
        h, w = img.shape[:2] if img is not None else (0, 0)

        return {
            "detections": [
                {
                    "class": "dummy",
                    "confidence": 0.99,
                    "bbox": [10, 10, 100, 100],
                },
            ],
            "image_size": [w, h],
        }


class KerasModel(BaseModel):
    """TensorFlow / Keras .h5 模型包裝器。

    支援任何 Sequential 或 Functional API 的 Keras 模型。
    此專案的模型為回歸模型（輸出單一 float）：
      - 輸入：`(H, W, 3)` float32 影像（預設除以 255 正規化）
      - 輸出：`(1,)` 單一預測值

    已驗證可用的模型檔：
      - output_model_v1.h5           → 輸入 (88, 275, 3)
      - output_model_v1_0.25.h5      → 輸入 (88, 275, 3)
      - output_model_v1_0.25_ori.h5  → 輸入 (275, 88, 3)
    """

    def __init__(self, model_path: str, normalize: bool = True):
        """載入 Keras .h5 模型。

        Args:
            model_path: .h5 模型檔路徑
            normalize:  是否將像素值除以 255.0（True = 正規化到 [0, 1]）

        Raises:
            ImportError:      TensorFlow 未安裝
            FileNotFoundError: 模型檔不存在
        """
        # 抑制 TF 啟動時的 INFO/WARNING log（只顯示錯誤）
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "載入 Keras 模型需要 TensorFlow，請執行：\n"
                "  pip install tensorflow-cpu"
            ) from exc

        if not _os.path.exists(model_path):
            raise FileNotFoundError(f"模型檔不存在：{model_path}")

        # 抑制 keras 的 UserWarning（舊版 input_shape 寫法）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = tf.keras.models.load_model(model_path)

        # 從模型讀取預期的輸入尺寸 (H, W)
        self._input_h = self._model.input_shape[1]
        self._input_w = self._model.input_shape[2]
        self._normalize = normalize

        logger.info(
            "Keras 模型已載入: %s | 輸入: (%d, %d, 3) | normalize=%s",
            model_path, self._input_h, self._input_w, normalize,
        )

    def predict(self, jpeg_data: bytes) -> dict:
        """對 JPEG 影像執行 Keras 模型推論。

        流程：
        1. JPEG bytes → OpenCV BGR image
        2. Resize 到模型要求的 (H, W)
        3. BGR → RGB，並正規化到 [0, 1]
        4. 加 batch 維度 → model.predict()
        5. 取出預測值並組裝回傳 dict

        Args:
            jpeg_data: JPEG 影像位元組

        Returns:
            推論結果 dict，包含 "prediction"（單一 float）和 "detections"（空列表，
            保持與 DummyModel / YOLOModel 相同的回傳結構）
        """
        import numpy as np

        # 1. 解碼 JPEG
        img = cv2.imdecode(
            np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            logger.warning("JPEG 解碼失敗，跳過此幀")
            return {"detections": [], "prediction": None}

        # 2. Resize 到模型輸入尺寸 (W, H)——注意 cv2.resize 是 (寬, 高)
        resized = cv2.resize(img, (self._input_w, self._input_h))

        # 3. 保持 BGR（訓練時用 cv2.imread 讀取即為 BGR，未做通道轉換，
        #    因此推論端也必須維持 BGR，否則紅藍通道顛倒會導致預測亂掉）
        # rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # ← 不要轉！

        # 4. 正規化 + 加 batch 維度
        arr = resized.astype(np.float32)
        if self._normalize:
            arr /= 255.0
        batch = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

        # 5. 推論
        output = self._model.predict(batch, verbose=0)   # shape: (1, 1)
        pred_value = float(output[0][0])

        logger.debug("Keras 推論結果: %.4f", pred_value)

        return {
            "detections": [],          # 維持與其他模型相同的 key 結構
            "prediction": pred_value,  # 回歸輸出值
            "image_size": [img.shape[1], img.shape[0]],
        }


class YOLOModel(BaseModel):
    """Ultralytics YOLO 模型包裝器。

    支援 YOLOv5, YOLOv8, YOLO11 等 Ultralytics 支援的所有版本。
    模型檔案可以是 .pt (PyTorch), .onnx (ONNX), .engine (TensorRT) 等格式。

    初始化時會載入模型到指定 device（cuda/cpu），
    之後每次呼叫 predict() 都在同一個 device 上執行。
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """載入 YOLO 模型。

        Args:
            model_path: 模型檔案路徑（如 "yolov8n.pt", "/models/best.onnx"）
            device:     推論裝置 "cpu", "cuda", "cuda:0", "cuda:1" 等

        Raises:
            ImportError: ultralytics 套件未安裝
            FileNotFoundError: 模型檔案不存在
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        logger.info("YOLO 模型已載入: %s (device=%s)", model_path, device)

    def predict(self, jpeg_data: bytes) -> dict:
        """對 JPEG 影像執行 YOLO 物件偵測。

        流程：
        1. 將 JPEG bytes 解碼為 numpy array
        2. 呼叫 YOLO model.predict()
        3. 解析結果中的每個邊界框
        4. 組裝成標準化的回傳格式

        Args:
            jpeg_data: JPEG 影像位元組

        Returns:
            推論結果 dict，包含所有偵測到的物件
        """
        # JPEG bytes → OpenCV BGR image
        img = cv2.imdecode(
            np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
        )

        # 執行推論（verbose=False 避免每幀都印出偵測日誌）
        results = self.model.predict(img, device=self.device, verbose=False)

        # 解析結果
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                })

        return {"detections": detections}


def create_model(model_type: str, model_path: str = "", device: str = "cpu") -> BaseModel:
    """工廠函式：根據 model_type 建立對應的模型實例。

    此函式是設定檔 inference.model_type 的對應入口。

    Args:
        model_type: 模型類型字串
                    - "dummy": DummyModel（測試用，不需要 GPU 或模型檔）
                    - "yolo":  YOLOModel（需要 ultralytics 套件和 .pt 模型檔）
                    - "keras": KerasModel（需要 tensorflow-cpu 和 .h5 模型檔）
        model_path: 模型檔案路徑（dummy 模式可留空）
        device:     推論裝置（"cpu", "cuda" 等；keras 模式目前只用 cpu）

    Returns:
        BaseModel 實例

    Raises:
        ValueError: 不支援的 model_type
    """
    if model_type == "dummy":
        logger.info("使用 DummyModel（測試模式）")
        return DummyModel()
    elif model_type == "yolo":
        return YOLOModel(model_path, device)
    elif model_type == "keras":
        return KerasModel(model_path)
    else:
        raise ValueError(
            f"不支援的 model_type: '{model_type}'。"
            f"可用選項: 'dummy', 'yolo', 'keras'"
        )
