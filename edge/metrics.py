"""Edge 端輕量級指標收集器。

記錄每幀的 send → result 延遲、模型 prediction 值、丟包數，
程式結束時 dump CSV / summary / 圖表。

設計目標：
  - 對 Pi 的 runtime overhead 極小（hot path 只有 dict O(1) 操作）
  - matplotlib 只在結束時 import，不影響啟動速度
  - 沒裝 matplotlib 也能用，只少了 PNG 圖表，CSV / summary 照常產生

輸出檔案（在 output_dir 內）：
  - raw_data.csv             每幀的詳細數據，給離線分析用
  - summary.json             機器可讀的摘要
  - summary.txt              人類可讀的摘要
  - latency_over_time.png    延遲隨時間變化曲線
  - latency_histogram.png    延遲分布直方圖
  - prediction_over_time.png 模型預測值隨時間變化曲線

用法（在 edge/main.py 中）：
    metrics = MetricsCollector("metrics/run_xxx")
    metrics.record_sent(frame_id, seq, len(jpeg), sent_ok)
    metrics.record_received(payload)     # payload = Message.payload
    metrics.finalize()                   # 結束時呼叫
"""

import csv
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("edge.metrics")


class MetricsCollector:
    """收集 Edge 端的延遲/預測/丟包指標。"""

    def __init__(self, output_dir: str):
        """初始化。

        Args:
            output_dir: 結束時把所有檔案寫到這個目錄。會自動建立。
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 已送出但還沒收到結果的幀：frame_id → (seq, send_time, size_bytes)
        self._pending = {}
        # 已完成的紀錄（送出 + 收到結果）
        self._completed = []
        # 計數器
        self._sent_count = 0
        self._send_failed = 0
        self._start_time = time.time()

    # ── Hot path methods（每幀都會呼叫，要快）──────────────

    def record_sent(self, frame_id: str, seq: int, size: int, sent_ok: bool):
        """記錄一幀送出事件。

        Args:
            frame_id: 幀的唯一 ID（同 header['frame_id']）
            seq:      序號
            size:     JPEG 大小（bytes）
            sent_ok:  send_frame() 的回傳值；False 代表 data channel 沒接受
        """
        self._sent_count += 1
        if not sent_ok:
            self._send_failed += 1
            return  # 沒成功送出的不進 _pending，不會等回應
        self._pending[frame_id] = (seq, time.time(), size)

    def record_received(self, payload: dict):
        """記錄一筆推論結果回來。

        Args:
            payload: Message.payload dict，至少含 frame_id 和 result.prediction
        """
        recv_time = time.time()
        frame_id = payload.get("frame_id", "")

        info = self._pending.pop(frame_id, None)
        if info is None:
            # 結果回來時對應的送出紀錄已經沒了（極少見：重複收、超時清理）
            return
        seq, send_time, size = info
        latency_ms = (recv_time - send_time) * 1000.0

        inner = payload.get("result", {})
        prediction = inner.get("prediction")  # Keras 回歸模型才會有

        self._completed.append({
            "frame_id": frame_id,
            "seq": seq,
            "send_time": send_time,
            "recv_time": recv_time,
            "latency_ms": latency_ms,
            "size_bytes": size,
            "prediction": prediction,
        })

    # ── Shutdown：寫檔 + 畫圖 ──────────────────────────────

    def finalize(self):
        """程式結束時呼叫：把所有資料寫到 output_dir。"""
        elapsed = time.time() - self._start_time
        n_completed = len(self._completed)
        n_no_result = len(self._pending)  # 送出但沒收到結果（疑似丟包）

        try:
            self._write_csv()
            self._write_summary(elapsed, n_completed, n_no_result)
        except Exception:
            logger.exception("寫入 metrics 檔案失敗")

        try:
            self._try_plot()
        except Exception:
            logger.exception("產生圖表失敗（可能 matplotlib 出問題），CSV 已寫入")

        logger.info("Metrics 已輸出到: %s", self.output_dir.resolve())
        logger.info(
            "送出=%d, 完成=%d, 未收到結果=%d, send 失敗=%d, 執行時間=%.1fs",
            self._sent_count, n_completed, n_no_result,
            self._send_failed, elapsed,
        )

    def _write_csv(self):
        if not self._completed:
            return
        path = self.output_dir / "raw_data.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._completed[0].keys()))
            writer.writeheader()
            writer.writerows(self._completed)

    def _write_summary(self, elapsed, n_completed, n_no_result):
        latencies = [r["latency_ms"] for r in self._completed]
        predictions = [
            r["prediction"] for r in self._completed
            if r["prediction"] is not None
        ]

        def pct(data, p):
            if not data:
                return None
            s = sorted(data)
            return s[min(int(len(s) * p / 100), len(s) - 1)]

        summary = {
            "elapsed_sec": round(elapsed, 1),
            "elapsed_human": (
                f"{elapsed / 3600:.2f} h" if elapsed >= 3600
                else f"{elapsed / 60:.1f} min" if elapsed >= 60
                else f"{elapsed:.1f} s"
            ),
            "frames": {
                "sent_total": self._sent_count,
                "send_failed": self._send_failed,
                "completed": n_completed,
                "no_result": n_no_result,
                "completion_rate_pct": round(
                    100 * n_completed / max(self._sent_count, 1), 2
                ),
                "avg_fps": round(n_completed / max(elapsed, 1e-9), 2),
            },
            "latency_ms": {
                "min":  round(min(latencies), 1) if latencies else None,
                "p50":  round(pct(latencies, 50), 1) if latencies else None,
                "p95":  round(pct(latencies, 95), 1) if latencies else None,
                "p99":  round(pct(latencies, 99), 1) if latencies else None,
                "max":  round(max(latencies), 1) if latencies else None,
                "avg":  round(sum(latencies) / len(latencies), 1) if latencies else None,
            },
            "prediction": ({
                "count": len(predictions),
                "min":   round(min(predictions), 4),
                "max":   round(max(predictions), 4),
                "first": round(predictions[0], 4),
                "last":  round(predictions[-1], 4),
                "avg":   round(sum(predictions) / len(predictions), 4),
            } if predictions else None),
        }

        # JSON
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 人類可讀
        with open(self.output_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("  Edge Streaming Metrics Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"執行時間: {summary['elapsed_human']}  ({summary['elapsed_sec']} s)\n\n")

            fr = summary["frames"]
            f.write("[Frame 統計]\n")
            f.write(f"  送出總數:       {fr['sent_total']}\n")
            f.write(f"  send 失敗:      {fr['send_failed']}\n")
            f.write(f"  完整收到結果:   {fr['completed']}\n")
            f.write(f"  送出但無結果:   {fr['no_result']}\n")
            f.write(f"  完整率:         {fr['completion_rate_pct']} %\n")
            f.write(f"  實際平均 fps:   {fr['avg_fps']}\n\n")

            lt = summary["latency_ms"]
            f.write("[端到端延遲 ms]\n")
            f.write(f"  min  : {lt['min']}\n")
            f.write(f"  p50  : {lt['p50']}\n")
            f.write(f"  avg  : {lt['avg']}\n")
            f.write(f"  p95  : {lt['p95']}\n")
            f.write(f"  p99  : {lt['p99']}\n")
            f.write(f"  max  : {lt['max']}\n\n")

            if summary["prediction"]:
                pr = summary["prediction"]
                f.write("[模型預測值]\n")
                f.write(f"  count: {pr['count']}\n")
                f.write(f"  first: {pr['first']}\n")
                f.write(f"  last : {pr['last']}\n")
                f.write(f"  min  : {pr['min']}\n")
                f.write(f"  max  : {pr['max']}\n")
                f.write(f"  avg  : {pr['avg']}\n")

    def _try_plot(self):
        """畫圖。沒裝 matplotlib 就跳過。"""
        if not self._completed:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")   # 無 GUI 環境（Pi 上沒桌面）
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("未安裝 matplotlib，跳過 PNG 圖表。CSV 仍可在開發機畫圖。")
            return

        records = self._completed
        # 把 send_time 轉成「從開始的分鐘數」
        rel_times_min = [
            (r["send_time"] - self._start_time) / 60.0 for r in records
        ]
        latencies = [r["latency_ms"] for r in records]

        predictions = [
            (rel_times_min[i], records[i]["prediction"])
            for i in range(len(records))
            if records[i]["prediction"] is not None
        ]

        # ── 圖 1: 延遲隨時間 ──
        plt.figure(figsize=(10, 4))
        plt.plot(rel_times_min, latencies, linewidth=0.6, color="tab:blue")
        plt.xlabel("Time since start (min)")
        plt.ylabel("Latency (ms)")
        plt.title(f"End-to-end Latency over Time  (n={len(latencies)})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_over_time.png", dpi=100)
        plt.close()

        # ── 圖 2: 延遲分布 ──
        plt.figure(figsize=(8, 4))
        plt.hist(latencies, bins=50, edgecolor="black", alpha=0.7,
                 color="tab:blue")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frame count")
        plt.title(f"Latency Distribution  (n={len(latencies)})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_histogram.png", dpi=100)
        plt.close()

        # ── 圖 3: 預測值隨時間 ──
        if predictions:
            xs = [p[0] for p in predictions]
            ys = [p[1] for p in predictions]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, ys, linewidth=0.6, color="tab:orange")
            plt.xlabel("Time since start (min)")
            plt.ylabel("Prediction")
            plt.title(f"Model Prediction over Time  (n={len(ys)})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "prediction_over_time.png", dpi=100)
            plt.close()

        logger.info("已產生 PNG 圖表")
