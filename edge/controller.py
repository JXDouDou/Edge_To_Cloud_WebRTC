"""推論結果處理與控制動作觸發模組。

此模組負責接收從推論伺服器回傳的結果 JSON，
並根據偵測到的物件類別觸發對應的控制動作。

典型用途：
  - 偵測到 "person" → 啟動 GPIO 警報
  - 偵測到 "car"    → 觸發柵欄開啟
  - 偵測到 "fire"   → 發送告警通知

使用方式：
    controller = Controller()

    async def on_person(det):
        print(f"偵測到人！信心度: {det['confidence']}")

    controller.register("person", on_person)

    # 當推論結果到達時：
    await controller.handle_result(result_payload)
"""

import logging
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

# 結果處理函式的型別簽名：接收一個 detection dict，回傳 Awaitable
ResultHandler = Callable[[dict], Awaitable[None]]


class Controller:
    """推論結果分發器，將偵測結果路由到對應的 handler。

    設計理念：
    - 解耦推論結果與控制邏輯
    - 每個物件類別可以註冊獨立的 handler
    - 新增控制動作只需 register() 新的 handler，不需修改核心程式碼
    """

    def __init__(self):
        """初始化 Controller，建立空的 handler 對照表。"""
        self._handlers: Dict[str, ResultHandler] = {}

    def register(self, class_name: str, handler: ResultHandler):
        """註冊某個偵測類別的處理函式。

        當推論結果中包含該類別的偵測時，會呼叫對應的 handler。
        每個類別只能有一個 handler，重複註冊會覆蓋舊的。

        Args:
            class_name: 偵測類別名稱（需與模型輸出一致，如 "person", "car"）
            handler:    async 函式，接收一個 detection dict 作為參數，格式為：
                        {"class": "person", "confidence": 0.92, "bbox": [x1,y1,x2,y2]}
        """
        self._handlers[class_name] = handler
        logger.info("已註冊 handler: class=%s", class_name)

    async def handle_result(self, result: dict):
        """處理一筆推論結果。

        從 payload 中取出 detections 列表，
        對每個 detection 查找是否有對應的 handler 並呼叫。

        預期的 result 格式（由 inference server 回傳）::

            {
                "frame_id": "a1b2c3d4",
                "edge_id": "edge-001",
                "seq": 42,
                "result": {
                    "detections": [
                        {"class": "person", "confidence": 0.92, "bbox": [100, 50, 300, 400]},
                        {"class": "car",    "confidence": 0.87, "bbox": [400, 200, 600, 350]},
                    ]
                }
            }

        Args:
            result: 推論結果 payload dict（即 Message.payload）

        Note:
            - handler 的例外不會中斷其他 detection 的處理
            - 沒有對應 handler 的類別會被靜默忽略（只記 log）
        """
        inner = result.get("result", {})
        detections = inner.get("detections", [])
        prediction = inner.get("prediction")  # 回歸模型（Keras）才會有

        # 組裝 log：回歸模型顯示 prediction，偵測模型顯示 detections 數量
        if prediction is not None:
            extra = f"prediction={prediction:.4f}"
        else:
            extra = f"detections={len(detections)}"

        logger.info(
            "收到結果: frame=%s, seq=%s, %s",
            result.get("frame_id", "?"),
            result.get("seq", "?"),
            extra,
        )

        # 逐一處理每個偵測結果
        for det in detections:
            cls_name = det.get("class", "")
            if cls_name in self._handlers:
                try:
                    await self._handlers[cls_name](det)
                except Exception:
                    logger.exception("Handler 執行錯誤: class=%s", cls_name)
