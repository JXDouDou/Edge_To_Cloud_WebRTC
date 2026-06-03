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
from collections import Counter
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

    def record_sent(self, frame_id: str, seq: int, size: int, sent_ok: bool,
                    capture_ms: float = 0.0, preprocess_ms: float = 0.0):
        """記錄一幀送出事件。

        Args:
            frame_id:      幀的唯一 ID（同 header['frame_id']）
            seq:           序號
            size:          JPEG 大小（bytes）
            sent_ok:       send_frame() 的回傳值；False 代表 data channel 沒接受
            capture_ms:    擷取(grab+解碼)耗時 ms，不含 FPS 節流 sleep（可選）
            preprocess_ms: 前處理(ROI+resize+JPEG)耗時 ms（可選）
        """
        self._sent_count += 1
        if not sent_ok:
            self._send_failed += 1
            return  # 沒成功送出的不進 _pending，不會等回應
        self._pending[frame_id] = (seq, time.time(), size, capture_ms, preprocess_ms)

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
        seq, send_time, size, capture_ms, preprocess_ms = info
        latency_ms = (recv_time - send_time) * 1000.0

        inner = payload.get("result", {})
        prediction = inner.get("prediction")  # Keras 回歸模型才會有
        # 哪台 dispatcher 服務了這張結果（由 dispatcher 在回傳前注入）。
        # 用來偵測 failover 切換點：連續兩筆 dispatcher_id 不同 = 發生切換。
        dispatcher_id = payload.get("dispatcher_id", "")

        self._completed.append({
            "frame_id": frame_id,
            "seq": seq,
            "send_time": send_time,
            "recv_time": recv_time,
            "latency_ms": latency_ms,
            "size_bytes": size,
            "prediction": prediction,
            "dispatcher_id": dispatcher_id,
            # 注意：以下兩項是「送出前」的本地處理耗時，不含在 latency_ms 內。
            "capture_ms": round(capture_ms, 2),
            "preprocess_ms": round(preprocess_ms, 2),
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
        # 快速亂序預覽（詳細數字寫在 summary.txt）
        if n_completed > 1:
            seqs = [r["seq"] for r in self._completed]
            ooo = sum(1 for i in range(1, len(seqs)) if seqs[i - 1] > seqs[i])
            logger.info(
                "結果亂序: %d/%d (%.2f%%)",
                ooo, len(seqs) - 1, 100 * ooo / (len(seqs) - 1),
            )
            # 快速 failover 預覽
            switches = 0
            for i in range(1, n_completed):
                p = self._completed[i - 1].get("dispatcher_id", "")
                c = self._completed[i].get("dispatcher_id", "")
                if c != p and p and c:
                    switches += 1
            if switches:
                logger.info("Failover 切換次數: %d（細節見 summary.txt）", switches)

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

        # ── 亂序率統計 ──────────────────────────────────────
        # self._completed 是「按結果回到 edge 的順序」紀錄的。
        # 每筆有 seq（edge 送出時的順序號）。
        # 如果第 i 筆的 seq 比第 i-1 筆小，代表「比較晚送出的反而比較早回來」=亂序。
        #
        # 為什麼會亂序：
        # - data channel 設定 ordered=False（追求低延遲，不保證順序）
        # - dispatcher 用 asyncio.create_task 並行轉發，多 frame 可能交換順序
        # - 網路本身（5G / Tailscale）也可能交換 UDP 封包順序
        #
        # 這個指標讓你看「實際亂序頻率」，決定是否需要：
        #   - 改成 ordered=True（用 latency 換 order）
        #   - 在 edge 端依 seq 排序結果再消費
        seqs_in_recv_order = [r["seq"] for r in self._completed]
        out_of_order_count = 0
        max_seq_diff = 0  # 最大「往回退」距離
        for i in range(1, len(seqs_in_recv_order)):
            diff = seqs_in_recv_order[i - 1] - seqs_in_recv_order[i]
            if diff > 0:
                out_of_order_count += 1
                if diff > max_seq_diff:
                    max_seq_diff = diff
        ooo_pct = (
            round(100 * out_of_order_count / (len(seqs_in_recv_order) - 1), 2)
            if len(seqs_in_recv_order) > 1
            else 0.0
        )

        # ── Failover / dispatcher 統計 ──────────────────────
        # _completed 是按「結果回到 edge 的順序」記的（recv_time 單調遞增）。
        # 每筆有 dispatcher_id（哪台 dispatcher 服務了這張）。
        # 連續兩筆的 dispatcher_id 不同 → 那一刻發生了 failover 切換。
        #
        # 切換空檔 (gap_ms) = 新 dispatcher 第一筆結果的 recv_time
        #                     − 舊 dispatcher 最後一筆結果的 recv_time
        # 這個空檔約等於「edge 偵測失敗 → 切到備援 → 串流恢復」的可見停頓。
        dispatcher_dist = Counter(
            r.get("dispatcher_id", "") for r in self._completed
        )
        switch_events = []
        for i in range(1, len(self._completed)):
            prev_d = self._completed[i - 1].get("dispatcher_id", "")
            cur_d = self._completed[i].get("dispatcher_id", "")
            if cur_d != prev_d and prev_d and cur_d:
                gap_ms = round(
                    (self._completed[i]["recv_time"]
                     - self._completed[i - 1]["recv_time"]) * 1000.0, 1
                )
                switch_events.append({
                    "from": prev_d,
                    "to": cur_d,
                    # 切換發生在跑了多少秒時（以新 dispatcher 第一筆為準）
                    "at_sec": round(
                        self._completed[i]["recv_time"] - self._start_time, 1
                    ),
                    # 可見停頓：舊 dispatcher 最後一筆 → 新 dispatcher 第一筆
                    "gap_ms": gap_ms,
                    # 切換時新 dispatcher 第一筆的 seq（用來對照丟了哪些 frame）
                    "resumed_at_seq": self._completed[i]["seq"],
                })

        # ── 送出前本地處理耗時 + JPEG 大小統計 ─────────────────
        # capture_ms / preprocess_ms 是「送出前」就花掉的時間，
        # 不含在 latency_ms（latency_ms 只算送出→收到結果）。
        # 真實「快門→結果」≈ capture_ms + preprocess_ms + latency_ms。
        cap_list = [r["capture_ms"] for r in self._completed]
        pp_list = [r["preprocess_ms"] for r in self._completed]
        size_list = [r["size_bytes"] for r in self._completed]

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
            "ordering": {
                # 結果回 edge 的順序中，「seq 比上一筆小」出現幾次 / 佔比
                "out_of_order_count": out_of_order_count,
                "out_of_order_pct": ooo_pct,
                # 最大往回退距離（例：上一筆 seq=100、這筆 seq=95 → max_seq_diff 至少 5）
                "max_seq_step_back": max_seq_diff,
            },
            "failover": {
                # 每台 dispatcher 服務了幾張結果
                "dispatcher_distribution": dict(dispatcher_dist),
                # 偵測到的切換次數
                "switch_count": len(switch_events),
                # 每次切換的細節（from/to/at_sec/gap_ms/resumed_at_seq）
                "switches": switch_events,
            },
            # 送出前的本地處理（不含在 latency_ms）
            "local_pipeline": {
                "capture_ms": {
                    "avg": round(sum(cap_list) / len(cap_list), 2) if cap_list else None,
                    "p50": round(pct(cap_list, 50), 2) if cap_list else None,
                    "p95": round(pct(cap_list, 95), 2) if cap_list else None,
                    "max": round(max(cap_list), 2) if cap_list else None,
                },
                "preprocess_ms": {
                    "avg": round(sum(pp_list) / len(pp_list), 2) if pp_list else None,
                    "p50": round(pct(pp_list, 50), 2) if pp_list else None,
                    "p95": round(pct(pp_list, 95), 2) if pp_list else None,
                    "max": round(max(pp_list), 2) if pp_list else None,
                },
                "jpeg_bytes": {
                    "avg": round(sum(size_list) / len(size_list), 1) if size_list else None,
                    "min": min(size_list) if size_list else None,
                    "max": max(size_list) if size_list else None,
                },
                # 估計的真實端到端（快門→結果）平均 = 擷取+前處理+傳輸推論
                "est_glass_to_result_avg_ms": (
                    round(
                        (sum(cap_list) / len(cap_list))
                        + (sum(pp_list) / len(pp_list))
                        + (sum(latencies) / len(latencies)),
                        1,
                    )
                    if cap_list and pp_list and latencies else None
                ),
            },
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

            od = summary["ordering"]
            f.write("\n[結果回流順序]\n")
            f.write(f"  亂序次數:        {od['out_of_order_count']}\n")
            f.write(f"  亂序比例:        {od['out_of_order_pct']} %\n")
            f.write(f"  最大往回退距離:  {od['max_seq_step_back']} (seq)\n")
            if od['out_of_order_pct'] == 0:
                f.write("  → 結果全部按 seq 順序回傳\n")
            elif od['out_of_order_pct'] < 1:
                f.write("  → 幾乎都按順序，偶爾亂序\n")
            elif od['out_of_order_pct'] < 5:
                f.write("  → 輕微亂序，正常範圍\n")
            else:
                f.write("  → 亂序明顯，若 prediction 依賴順序可考慮 ordered=True\n")

            fo = summary["failover"]
            f.write("\n[Dispatcher / Failover]\n")
            if fo["dispatcher_distribution"]:
                f.write("  各 dispatcher 服務筆數:\n")
                for did, cnt in fo["dispatcher_distribution"].items():
                    label = did if did else "(未知/舊版未注入 id)"
                    f.write(f"    {label}: {cnt}\n")
            f.write(f"  切換次數:        {fo['switch_count']}\n")
            if fo["switch_count"] == 0:
                f.write("  → 全程由同一台 dispatcher 服務，未發生 failover\n")
            else:
                for i, sw in enumerate(fo["switches"], 1):
                    f.write(
                        f"  切換 #{i}: {sw['from']} → {sw['to']}  "
                        f"在 {sw['at_sec']}s 時，"
                        f"可見停頓 {sw['gap_ms']} ms，"
                        f"恢復於 seq={sw['resumed_at_seq']}\n"
                    )

            lp = summary["local_pipeline"]
            f.write("\n[送出前本地處理] (不含在上面的端到端延遲內)\n")
            cm, pm, jb = lp["capture_ms"], lp["preprocess_ms"], lp["jpeg_bytes"]
            f.write(f"  擷取 capture_ms     : avg={cm['avg']}  p95={cm['p95']}  max={cm['max']}\n")
            f.write(f"  前處理 preprocess_ms: avg={pm['avg']}  p95={pm['p95']}  max={pm['max']}\n")
            if jb["avg"] is not None:
                f.write(
                    f"  JPEG 大小 bytes     : avg={jb['avg']}  "
                    f"min={jb['min']}  max={jb['max']}\n"
                )
            if lp["est_glass_to_result_avg_ms"] is not None:
                f.write(
                    f"  ≈ 真實端到端(快門→結果) avg = capture + preprocess + 延遲 "
                    f"= {lp['est_glass_to_result_avg_ms']} ms\n"
                )

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

        # 找出 failover 切換點（dispatcher_id 變動的那一筆），標在圖上
        switch_x = []
        for i in range(1, len(records)):
            prev_d = records[i - 1].get("dispatcher_id", "")
            cur_d = records[i].get("dispatcher_id", "")
            if cur_d != prev_d and prev_d and cur_d:
                switch_x.append(rel_times_min[i])

        # ── 圖 1: 延遲隨時間（含 failover 切換標記）──
        plt.figure(figsize=(10, 4))
        plt.plot(rel_times_min, latencies, linewidth=0.6, color="tab:blue")
        for j, x in enumerate(switch_x):
            plt.axvline(
                x, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8,
                label="dispatcher switch" if j == 0 else None,
            )
        plt.xlabel("Time since start (min)")
        plt.ylabel("Latency (ms)")
        plt.title(f"End-to-end Latency over Time  (n={len(latencies)})")
        if switch_x:
            plt.legend(loc="upper right", fontsize=8)
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
