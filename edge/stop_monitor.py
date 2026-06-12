"""Edge 端：依推論結果 (prediction) 觸發停止 / 控泵動作。

使用情境：注液到目標量時讓泵停下來。

判定方式（參考 newnewnew 的「連續確認」，但改成 frame-based）：
  連續 N 張 prediction 低於 target 才觸發 → 去抖，避免單張雜訊誤觸。
  一旦觸發就 latch（只觸發一次），之後不再重複呼叫 callback。

實際的控泵動作不寫在這裡（保持與硬體解耦）：
  由 on_trigger callback 決定要做什麼（log、把 PWM 設 0、改小等），
  edge/main.py 的 pump_stop_action() 就是預設的 callback。
"""

import logging

logger = logging.getLogger("edge.stop")


class StopMonitor:
    """監看 prediction 串流，達停止條件時呼叫 on_trigger（只一次）。"""

    def __init__(self, target, consecutive=5, mode="below", on_trigger=None):
        """初始化。

        Args:
            target:      目標門檻值
            consecutive: 連續幾張滿足條件才觸發（去抖；至少 1）
            mode:        "below" = prediction < target 觸發；"above" = > target
            on_trigger:  觸發時呼叫的 callback，簽名 on_trigger(prediction)
        """
        self.target = target
        self.consecutive = max(1, consecutive)
        self.mode = mode
        self.on_trigger = on_trigger
        self._streak = 0
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    def _meets(self, pred) -> bool:
        return pred < self.target if self.mode == "below" else pred > self.target

    def observe(self, prediction) -> bool:
        """餵入一張 prediction。回傳 True 代表「這一張剛好觸發了停止」。

        已觸發過 (latch) 或 prediction 為 None 時不動作。
        """
        if self._triggered or prediction is None:
            return False

        cmp = "<" if self.mode == "below" else ">"

        if self._meets(prediction):
            self._streak += 1
            # 第一張進入連續區、或快達標時印一下進度，方便觀察
            if self._streak == 1 or self._streak >= self.consecutive - 1:
                logger.info(
                    "接近停止: prediction=%.4f %s %.2f，連續 %d/%d",
                    prediction, cmp, self.target, self._streak, self.consecutive,
                )
            if self._streak >= self.consecutive:
                self._triggered = True
                logger.warning(
                    "✋ 已達停止條件（連續 %d 張 prediction %s %.2f）→ 觸發停止動作",
                    self.consecutive, cmp, self.target,
                )
                if self.on_trigger:
                    try:
                        self.on_trigger(prediction)
                    except Exception:
                        logger.exception("停止 callback 執行失敗")
                return True
        else:
            # 一旦有一張不滿足，連續計數歸零（去抖）
            if self._streak:
                logger.debug(
                    "連續中斷: prediction=%.4f 不 %s %.2f，streak 歸零",
                    prediction, cmp, self.target,
                )
            self._streak = 0
        return False
