"""泵控制抽象層（通用、不碰硬體）。

設計：
  - 這個檔案進 git，但**只有安全的預設實作 LogPump**（只 log，不碰 GPIO），
    所以在 Windows / 沒有 GPIO 的環境跑也不會壞。
  - 真正的 GPIO 控制放在 gitignored 的 edge/pump_local.py（只存在於 Pi）。
    get_pump() 會優先載入它；找不到就 fallback 成 LogPump。
  - 因此 `git pull` 永遠不會動到你 Pi 上的硬體碼，也不會把硬體碼推上雲端。

介面（edge/main.py 會這樣用）：
    pump = get_pump()
    pump.start()        # 注液開始 → 開泵
    pump.set(0.3)       # 改變流量大小（PWM duty 0.0~1.0）
    pump.stop()         # 關泵

第一次在 Pi 上設定：見 edge/pump_local.py.example 的說明。
"""

import logging

logger = logging.getLogger("edge.pump")


class LogPump:
    """預設實作：只 log，不碰硬體（Windows / 無 GPIO 環境用）。"""

    def start(self):
        logger.info("[LogPump] start（開泵 value=1.0）— 未接硬體，只 log")

    def set(self, value):
        logger.info("[LogPump] set value=%.2f — 未接硬體，只 log", value)

    def stop(self):
        logger.warning("[LogPump] stop（關泵 value=0.0）— 未接硬體，只 log")


def get_pump():
    """回傳泵控制物件。

    優先用 edge/pump_local.py 的 get_pump()（Pi 上才有，實際控 GPIO）；
    找不到或載入失敗就回 LogPump（只 log，安全 fallback）。
    """
    try:
        from edge import pump_local
    except ImportError:
        logger.info("未找到 edge/pump_local.py → 使用 LogPump（只 log，不控硬體）")
        return LogPump()
    try:
        pump = pump_local.get_pump()
        logger.info("已載入 edge/pump_local.py 的泵實作")
        return pump
    except Exception:
        logger.exception("載入 edge/pump_local.py 失敗 → fallback 成 LogPump")
        return LogPump()
