"""推論除錯視覺化工具

完整模擬 Edge → Inference 的預處理鏈，把每一步的圖像存到
quick_test/debug_frames/ 資料夾，讓你肉眼確認 inference 收到什麼。

每張輸出圖包含三個區域（左→右）：
  ① 原始幀（影片原畫）
  ② Inference 收到的 JPEG（edge 壓縮後，再解碼顯示）
  ③ 模型實際輸入（resize 到模型要求的尺寸後）

圖片右上角標注模型預測值。

使用方式（在專案根目錄執行）：
  python quick_test/debug_inference.py --model output_model_v1.h5
  python quick_test/debug_inference.py --model output_model_v1.h5 --video edge/video/40.mp4
  python quick_test/debug_inference.py --model output_model_v1.h5 --frames 20
  python quick_test/debug_inference.py --model output_model_v1.h5 --show   # 即時視窗顯示

結果存於：quick_test/debug_frames/
"""

import argparse
import os
import sys
import warnings

# ── Windows UTF-8 修正 ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "quick_test", "debug_frames")


def log(tag: str, msg: str):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def apply_edge_preprocess(frame, jpeg_quality: int = 80):
    """模擬 edge/preprocess.py 的處理（roi disabled, no resize）。

    和 test.yaml 的設定一致：
      - roi.enabled = false  → 不裁切
      - resize_width = 0     → 不 resize
      - jpeg_quality = 80

    Returns:
        (jpeg_bytes, decoded_frame)
        - jpeg_bytes:    edge 實際傳給 inference 的 bytes
        - decoded_frame: 從 jpeg_bytes 解碼回來的影像（就是 inference 拿到的）
    """
    import cv2
    import numpy as np

    # JPEG 壓縮（模擬 edge 的 Preprocessor.process()）
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 編碼失敗")
    jpeg_bytes = buf.tobytes()

    # 解碼回來 → 這就是 inference 實際 decode 到的圖
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return jpeg_bytes, decoded


def run_keras_model(model, jpeg_bytes: bytes):
    """對 jpeg_bytes 執行模型推論，回傳 (預測值, 模型輸入圖)。"""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    H, W = model._input_h, model._input_w
    resized = cv2.resize(img, (W, H))          # 模型實際輸入（視覺化用）
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype("float32")
    if model._normalize:
        arr /= 255.0
    import numpy as np
    batch = np.expand_dims(arr, axis=0)
    output = model._model.predict(batch, verbose=0)
    pred = float(output[0][0])
    return pred, resized   # resized 是 BGR，方便 cv2 顯示


def make_debug_panel(orig, decoded, model_input, pred_value, seq):
    """把三張圖水平拼接成一張除錯面板，加上標注文字。

    面板佈局（統一高度 = 320px）：
      [原始幀 640×480 → 縮放] | [Inference 收到] | [模型輸入]

    Args:
        orig:        原始幀 (BGR)
        decoded:     edge JPEG 解碼後的影像 (BGR)
        model_input: resize 到模型輸入尺寸的影像 (BGR)
        pred_value:  模型預測值 (float)
        seq:         幀序號

    Returns:
        panel: 三格拼接後的 BGR 影像
    """
    import cv2
    import numpy as np

    TARGET_H = 320   # 統一顯示高度

    def resize_to_height(img, h):
        if img is None:
            return np.zeros((h, h, 3), dtype="uint8")
        ih, iw = img.shape[:2]
        w = int(iw * h / ih)
        return cv2.resize(img, (w, h))

    p1 = resize_to_height(orig, TARGET_H)
    p2 = resize_to_height(decoded, TARGET_H)
    p3 = resize_to_height(model_input, TARGET_H)

    # 在每格上方加標題
    def add_title(img, title, subtitle=""):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(out, title, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        if subtitle:
            cv2.putText(out, subtitle, (6, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        return out

    h1, w1 = p1.shape[:2]
    h2, w2 = p2.shape[:2]
    h3, w3 = p3.shape[:2]

    p1 = add_title(p1, f"1. Original  (seq={seq})",
                   f"{w1}x{h1}px (video raw)")
    p2 = add_title(p2, "2. Inference received",
                   f"JPEG q80 -> decoded  {w2}x{h2}px")
    p3 = add_title(p3, f"3. Model input",
                   f"resized {w3}x{h3}px  /255 normalized")

    panel = np.hstack([p1, p2, p3])

    # 在整張圖右上角印預測值
    pred_text = f"prediction: {pred_value:.4f}" if pred_value is not None else "prediction: N/A"
    tw, th = panel.shape[1], 48
    cv2.rectangle(panel, (panel.shape[1] - 310, 0), (panel.shape[1], th), (20, 20, 60), -1)
    cv2.putText(panel, pred_text, (panel.shape[1] - 305, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (80, 255, 120), 2, cv2.LINE_AA)

    return panel


def main():
    parser = argparse.ArgumentParser(
        description="推論除錯視覺化：看 inference 實際收到什麼",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python quick_test/debug_inference.py --model output_model_v1.h5
  python quick_test/debug_inference.py --model output_model_v1.h5 --video edge/video/40.mp4
  python quick_test/debug_inference.py --model output_model_v1.h5 --frames 30 --show
        """,
    )
    parser.add_argument("--model", required=True,
                        help="Keras .h5 模型路徑（相對於專案根目錄）")
    parser.add_argument("--video", default="edge/video/30.mp4",
                        help="影片路徑（預設：edge/video/30.mp4）")
    parser.add_argument("--frames", type=int, default=10,
                        help="要處理的幀數（預設 10）")
    parser.add_argument("--skip", type=int, default=15,
                        help="每幾幀取一幀（預設每 15 幀，避免圖片太相似）")
    parser.add_argument("--show", action="store_true",
                        help="同時用 cv2.imshow 即時顯示（需要桌面環境）")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG 壓縮品質（預設 80，與 test.yaml 一致）")
    args = parser.parse_args()

    import cv2

    # ── 路徑解析 ──
    model_path = (os.path.join(PROJECT_ROOT, args.model)
                  if not os.path.isabs(args.model) else args.model)
    video_path = (os.path.join(PROJECT_ROOT, args.video)
                  if not os.path.isabs(args.video) else args.video)

    for p, name in [(model_path, "模型"), (video_path, "影片")]:
        if not os.path.exists(p):
            log("錯誤", f"找不到{name}：{p}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 載入模型 ──
    log("模型", f"載入中：{args.model}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from inference.model_runner import KerasModel
        model = KerasModel(model_path)

    log("模型", f"輸入尺寸: ({model._input_h}, {model._input_w}, 3)  normalize={model._normalize}")

    # ── 開啟影片 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("錯誤", f"無法開啟影片：{video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log("影片", f"{args.video}  {orig_w}x{orig_h}  {fps_src:.1f}fps  共 {total_frames} 幀")
    log("輸出", f"儲存至：{OUTPUT_DIR}")
    print()

    saved = 0
    frame_idx = 0
    predictions = []

    while saved < args.frames:
        ret, orig_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, orig_frame = cap.read()
            if not ret:
                break

        frame_idx += 1
        if frame_idx % args.skip != 0:
            continue   # 跳幀，避免圖片太相似

        # ── 模擬 edge 預處理 ──
        jpeg_bytes, decoded = apply_edge_preprocess(orig_frame, args.quality)

        # ── 模型推論 ──
        pred_value, model_input_img = run_keras_model(model, jpeg_bytes)
        predictions.append(pred_value)

        # ── 製作除錯面板 ──
        panel = make_debug_panel(orig_frame, decoded, model_input_img, pred_value, frame_idx)

        # ── 儲存圖片 ──
        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}_pred{pred_value:.3f}.jpg")
        cv2.imwrite(out_path, panel)
        log("幀", f"#{frame_idx:5d}  預測值={pred_value:8.4f}  → {os.path.basename(out_path)}")

        # ── 即時顯示 ──
        if args.show:
            cv2.imshow("Debug: inference input", panel)
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):   # ESC 或 q 結束
                log("系統", "使用者中斷")
                break

        saved += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    # ── 統計摘要 ──
    print()
    print("=" * 55)
    print("  除錯摘要")
    print("=" * 55)
    if predictions:
        import statistics
        valid = [p for p in predictions if p is not None]
        print(f"  處理幀數：{len(valid)}")
        print(f"  預測值範圍：{min(valid):.4f}  ~  {max(valid):.4f}")
        print(f"  平均值：    {statistics.mean(valid):.4f}")
        print(f"  中位數：    {statistics.median(valid):.4f}")
    print(f"  圖片存於：  {OUTPUT_DIR}")
    print("=" * 55)
    print()
    log("完成", f"請開啟 quick_test/debug_frames/ 查看除錯圖片")


if __name__ == "__main__":
    main()
