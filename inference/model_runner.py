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


class OnnxModel(BaseModel):
    """ONNX Runtime 模型包裝器（支援 NVIDIA GPU / CUDAExecutionProvider）。

    用途：把訓練好的 Keras `.h5` 轉成 `.onnx` 後，用 onnxruntime-gpu 在
    「原生 Windows + NVIDIA GPU（如 RTX 5080）」上推論——避開
    「TensorFlow 原生 Windows GPU 支援已停在 2.10」的死路。

    前處理 / 後處理與 KerasModel 完全相同（BGR 不轉 RGB、resize、/255），
    回傳結構也一致（prediction + detections），所以兩者可直接互換。

    切回 Keras：把 config 的 `inference.model_type` 改回 "keras"、
    `model_path` 指回 `.h5` 即可（onnxruntime 與 tensorflow 可並存，不衝突）。

    產生 .onnx：見 scripts/convert_to_onnx.py（會順便比對 keras vs onnx 精度）。
    """

    def __init__(self, model_path: str, device: str = "cuda", normalize: bool = True):
        """載入 ONNX 模型。

        Args:
            model_path: .onnx 模型檔路徑
            device:     "cuda"（用 GPU，失敗自動 fallback 到 CPU）或 "cpu"
            normalize:  是否將像素值除以 255.0

        Raises:
            ImportError:       onnxruntime 未安裝
            FileNotFoundError: 模型檔不存在
        """
        import os as _os

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "載入 ONNX 模型需要 onnxruntime，請執行：\n"
                "  pip install onnxruntime-gpu   # 有 NVIDIA GPU（預設）\n"
                "  pip install onnxruntime       # 純 CPU"
            ) from exc

        if not _os.path.exists(model_path):
            raise FileNotFoundError(f"模型檔不存在：{model_path}")

        # 依 device 選 execution provider；CUDA 載入失敗 onnxruntime 會自動退到 CPU
        if device.lower().startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, providers=providers)
        active = self._session.get_providers()

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._output_name = self._session.get_outputs()[0].name
        # tf2onnx 預設保留 NHWC layout：shape = [batch, H, W, C]
        shape = inp.shape
        self._input_h = shape[1] if isinstance(shape[1], int) else None
        self._input_w = shape[2] if isinstance(shape[2], int) else None
        self._normalize = normalize

        if self._input_h is None or self._input_w is None:
            raise ValueError(
                f"ONNX 輸入的空間維度是動態的 {shape}，無法決定 resize 尺寸。"
                f"請用固定尺寸重新轉檔（見 scripts/convert_to_onnx.py）。"
            )

        # 確認是否「真的」吃到 GPU——CUDA 載入失敗會靜默 fallback 成 CPU，
        # 不檢查的話你會以為在跑 GPU、其實在跑 CPU。
        on_gpu = "CUDAExecutionProvider" in active
        logger.info(
            "ONNX 模型已載入: %s | 輸入: (%d, %d, 3) | providers=%s | GPU=%s",
            model_path, self._input_h, self._input_w, active, on_gpu,
        )
        if device.lower().startswith("cuda") and not on_gpu:
            logger.warning(
                "⚠️ 要求用 CUDA 但實際 fallback 到 CPU！請確認："
                "(1) 裝的是 onnxruntime-gpu 不是 onnxruntime；"
                "(2) CUDA / cuDNN 版本支援此 GPU（RTX 5080 需 CUDA 12.8+）。"
            )

    def predict(self, jpeg_data: bytes) -> dict:
        """對 JPEG 影像執行 ONNX 推論（前處理與 KerasModel 完全一致）。"""
        import numpy as np

        img = cv2.imdecode(
            np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            logger.warning("JPEG 解碼失敗，跳過此幀")
            return {"detections": [], "prediction": None}

        # Resize 到模型輸入 (W, H)；保持 BGR（與訓練一致，勿轉 RGB）
        resized = cv2.resize(img, (self._input_w, self._input_h))
        arr = resized.astype(np.float32)
        if self._normalize:
            arr /= 255.0
        batch = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

        output = self._session.run([self._output_name], {self._input_name: batch})
        pred_value = float(np.asarray(output[0]).reshape(-1)[0])

        logger.debug("ONNX 推論結果: %.4f", pred_value)

        return {
            "detections": [],
            "prediction": pred_value,
            "image_size": [img.shape[1], img.shape[0]],
        }


class _SimpleCNN:
    """建立與 Code/IMG/ 訓練腳本完全一致的 CNN 架構。

    必須與訓練時的 SimpleCNN class 結構完全對齊（conv_layers / fc_layers
    屬性名稱、層順序），否則 load_state_dict 會 key mismatch。
    """

    @staticmethod
    def build(num_channels: int, image_height: int, image_width: int,
              num_outputs: int = 2):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                dummy = torch.randn(1, num_channels, image_height, image_width)
                conv_out = self.conv_layers(dummy).data.view(1, -1).size(1)
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(conv_out, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_outputs),
                )

            def forward(self, x):
                return self.fc_layers(self.conv_layers(x))

        return _Net()


class PytorchModel(BaseModel):
    """PyTorch .pth 模型包裝器（PGML / PINN CNN，支援 NVIDIA GPU）。

    用途：載入 Code/IMG/ 訓練的 SimpleCNN .pth 權重，在 RTX 5080 等
    NVIDIA GPU 上以 PyTorch CUDA 執行推論——繞過 onnxruntime-gpu
    在 Blackwell 架構上尚未支援的問題。

    前處理與 KerasModel / OnnxModel 相同（BGR、resize、/255），
    回傳結構也一致（prediction + detections），可直接互換。

    切回 Keras/ONNX：把 config 的 inference.model_type 改回 "keras"
    或 "onnx" 即可，不需要移除 torch 套件。
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 normalize: bool = True,
                 input_height: int = 516, input_width: int = 182):
        import os as _os

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "載入 PyTorch 模型需要 torch，請執行：\n"
                "  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128"
            ) from exc

        if not _os.path.exists(model_path):
            raise FileNotFoundError(f"模型檔不存在：{model_path}")

        self._torch = torch
        self._device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._input_h = input_height
        self._input_w = input_width
        self._normalize = normalize

        self._model = _SimpleCNN.build(3, input_height, input_width, 2)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self._model.to(self._device)
        self._model.eval()

        on_gpu = self._device.type == "cuda"
        logger.info(
            "PyTorch 模型已載入: %s | 輸入: (%d, %d, 3) | device=%s | GPU=%s",
            model_path, self._input_h, self._input_w, self._device, on_gpu,
        )
        if device.lower().startswith("cuda") and not on_gpu:
            logger.warning(
                "⚠️ 要求用 CUDA 但 CUDA 不可用，已 fallback 到 CPU！"
            )

    def predict(self, jpeg_data: bytes) -> dict:
        """對 JPEG 影像執行 PyTorch 模型推論。

        流程與 KerasModel / OnnxModel 一致：
        1. JPEG bytes → OpenCV BGR
        2. Resize 到模型輸入尺寸
        3. /255 正規化，HWC → CHW（PyTorch NCHW 格式）
        4. 推論，取出 volume（第一輸出）作為 prediction
        """
        import numpy as np

        img = cv2.imdecode(
            np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            logger.warning("JPEG 解碼失敗，跳過此幀")
            return {"detections": [], "prediction": None}

        # PGML 模型訓練時用 PIL（RGB），而 cv2.imdecode 輸出 BGR，
        # 必須轉回 RGB 否則預測會完全亂掉。
        # （Keras/ONNX 模型不需要轉，因為它們訓練時也用 cv2 = BGR）
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h))
        arr = resized.astype(np.float32)
        if self._normalize:
            arr /= 255.0

        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self._device)

        with self._torch.no_grad():
            output = self._model(tensor)

        pred_volume = float(output[0, 0].cpu().item())
        pred_height = float(output[0, 1].cpu().item())

        logger.debug(
            "PyTorch 推論結果: volume=%.4f, height=%.6f",
            pred_volume, pred_height,
        )

        return {
            "detections": [],
            "prediction": pred_volume,
            "prediction_height": pred_height,
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
                    - "dummy":   DummyModel（測試用，不需要 GPU 或模型檔）
                    - "pytorch": PytorchModel（PGML CNN；需 torch 和 .pth）
                    - "onnx":    OnnxModel（需 onnxruntime-gpu 和 .onnx）
                    - "keras":   KerasModel（CPU fallback；需 tensorflow-cpu 和 .h5）
                    - "yolo":    YOLOModel（需要 ultralytics 套件和 .pt 模型檔）
        model_path: 模型檔案路徑（dummy 模式可留空）
        device:     推論裝置（"cpu", "cuda" 等；pytorch/onnx/yolo 會用到）

    Returns:
        BaseModel 實例

    Raises:
        ValueError: 不支援的 model_type
    """
    if model_type == "dummy":
        logger.info("使用 DummyModel（測試模式）")
        return DummyModel()
    elif model_type == "pytorch":
        return PytorchModel(model_path, device)
    elif model_type == "onnx":
        return OnnxModel(model_path, device)
    elif model_type == "yolo":
        return YOLOModel(model_path, device)
    elif model_type == "keras":
        return KerasModel(model_path)
    else:
        raise ValueError(
            f"不支援的 model_type: '{model_type}'。"
            f"可用選項: 'dummy', 'pytorch', 'onnx', 'keras', 'yolo'"
        )
