"""測試不同前處理組合，找出哪種最接近訓練時的設定。

測試的組合：
  A) BGR  + /255   (目前用法)
  B) RGB  + /255   (常見 Keras 訓練方式)
  C) BGR  + 不除   (原始像素 0-255)
  D) RGB  + 不除   (原始像素 0-255)

只用 output_model_v1.h5 + 20/30/40/50 四個資料夾測試。
"""
import os, sys, warnings, statistics
import cv2, numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import tensorflow as tf
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_tf = tf.keras.models.load_model("output_model_v1.h5")

MH, MW = model_tf.input_shape[1], model_tf.input_shape[2]
BASE    = "D:/Lab/Code/Extracted_IMG"
FOLDERS = [("20", 20), ("30", 30), ("40", 40), ("50", 50)]
BATCH   = 64

COMBOS = [
    ("A: BGR  /255 (現在)", False, True),
    ("B: RGB  /255        ", True,  True),
    ("C: BGR  raw         ", False, False),
    ("D: RGB  raw         ", True,  False),
]


def predict_folder(folder_path, to_rgb: bool, normalize: bool):
    paths = sorted([os.path.join(folder_path, f)
                    for f in os.listdir(folder_path) if f.endswith(".jpg")])
    preds = []
    for start in range(0, len(paths), BATCH):
        chunk = []
        for p in paths[start:start + BATCH]:
            img = cv2.imread(p)          # BGR
            if img is None: continue
            r = cv2.resize(img, (MW, MH))
            arr = r.astype("float32")
            if to_rgb:
                arr = arr[:, :, ::-1]    # BGR → RGB
            if normalize:
                arr /= 255.0
            chunk.append(arr)
        if not chunk: continue
        out = model_tf.predict(np.stack(chunk), verbose=0)
        preds.extend(float(v[0]) for v in out)
    return preds


print(f"模型輸入: H={MH} W={MW}")
print()
print(f"{'組合':<22} | {'GT=20':>8} | {'GT=30':>8} | {'GT=40':>8} | {'GT=50':>8} | {'MAE':>6}")
print("-" * 75)

for label, to_rgb, normalize in COMBOS:
    row_means = []
    for folder_name, gt in FOLDERS:
        preds = predict_folder(os.path.join(BASE, folder_name), to_rgb, normalize)
        row_means.append((statistics.mean(preds), gt))
    mae = statistics.mean(abs(m - gt) for m, gt in row_means)
    cols = "  ".join(f"{m:+6.2f}({m-gt:+.1f})" for m, gt in row_means)
    print(f"{label} | {cols} | {mae:6.2f}")

print()
print("格式: mean(err)   err = mean - ground_truth")
