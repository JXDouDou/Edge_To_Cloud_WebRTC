"""完全照 Code/实时演示.py 的方式跑推論，作為正確答案基準。

目的：確認模型正確用法，並告訴我們每支影片的真實預測應該是多少。

關鍵步驟（來自原始訓練流程）：
  1. 讀取 1920×1080 影格
  2. YOLO 格式 ROI 裁切：[0, 0.463021, 0.744444, 0.092708, 0.511111]
  3. 把 ROI 的寬高各除以 2（不是 resize 到固定尺寸！）
  4. /255.0 正規化
  5. 保持 BGR，不做色彩空間轉換
  6. 餵給模型
"""
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "Code")

import cv2
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

# YOLO 格式 ROI box（來自 2_img_extraction.py / 实时演示.py）
BOX = [0, 0.463021, 0.744444, 0.092708, 0.511111]

# 三個可用模型 + 它們的 input_shape（由 .h5 讀出來）
MODELS = [
    "output_model_v1.h5",           # (88, 275, 3)
    "output_model_v1_0.25.h5",      # (88, 275, 3)
    "output_model_v1_0.25_ori.h5",  # (275, 88, 3)
]

VIDEOS = ["20.mp4", "30.mp4", "40.mp4", "50.mp4"]


def extract_roi(frame, box):
    """依 YOLO 格式 bbox 從原始影格裁出 ROI。"""
    h, w, _ = frame.shape
    cx = int(box[1] * w)
    cy = int(box[2] * h)
    bw = int(box[3] * w)
    bh = int(box[4] * h)
    x1 = max(cx - bw // 2, 0)
    y1 = max(cy - bh // 2, 0)
    x2 = min(cx + bw // 2, w)
    y2 = min(cy + bh // 2, h)
    return frame[y1:y2, x1:x2]


def preprocess(roi):
    """照 Code/实时演示.py 的 preprocess_image()：尺寸減半 + /255。"""
    ow, oh = roi.shape[1], roi.shape[0]
    target = (ow // 2, oh // 2)  # (width, height)
    img = cv2.resize(roi, target)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_video(model, video_path, n_samples=10, skip=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    preds = []
    idx = 0
    while len(preds) < n_samples:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % skip != 0:
            continue
        roi = extract_roi(frame, BOX)
        batch = preprocess(roi)
        # 若尺寸與模型預期不同，跳過本幀
        expected_h = model.input_shape[1]
        expected_w = model.input_shape[2]
        if batch.shape[1] != expected_h or batch.shape[2] != expected_w:
            # 嘗試轉置（處理 (88,275) vs (275,88) 的模型）
            batch_t = np.transpose(batch, (0, 2, 1, 3))
            if batch_t.shape[1] == expected_h and batch_t.shape[2] == expected_w:
                batch = batch_t
            else:
                return {"skipped": f"batch{batch.shape} vs model{(expected_h, expected_w)}"}
        out = model.predict(batch, verbose=0)
        preds.append(float(out[0][0]))
    cap.release()
    return preds


def main():
    print()
    print("=" * 80)
    print("  模型 × 影片 矩陣（照 Code/实时演示.py 原始處理流程）")
    print("=" * 80)

    loaded = {}
    for m in MODELS:
        mp = os.path.join(PROJECT_ROOT, m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded[m] = tf.keras.models.load_model(mp)
        print(f"  {m}  input_shape={loaded[m].input_shape}")

    print("-" * 80)
    print(f"{'Model':<32} {'Video':<10} {'pred avg':>10} {'range':>18}  期望")
    print("-" * 80)

    for m in MODELS:
        for v in VIDEOS:
            vp = os.path.join(CODE_DIR, v)
            res = predict_video(loaded[m], vp)
            expected = v.split(".")[0]
            if res is None:
                print(f"{m:<32} {v:<10}   無法開啟")
            elif isinstance(res, dict):
                print(f"{m:<32} {v:<10}   {res['skipped']}")
            else:
                avg = np.mean(res)
                lo, hi = min(res), max(res)
                print(f"{m:<32} {v:<10} {avg:>10.3f} {lo:>7.2f} ~ {hi:>6.2f}    {expected}")
        print("-" * 80)


if __name__ == "__main__":
    main()
