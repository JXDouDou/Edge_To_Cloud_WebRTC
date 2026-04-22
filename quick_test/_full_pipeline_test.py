"""驗證：照分散式系統的完整 pipeline（含 JPEG 壓縮）是否依然準確。

流程：原影格 → ROI 裁切 → resize 到 88×275 → JPEG q80 壓縮 → 解碼 → /255 → 模型
"""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError: pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

BOX = [0, 0.463021, 0.744444, 0.092708, 0.511111]
MODEL = "output_model_v1_0.25_ori.h5"
VIDEOS = ["20.mp4", "30.mp4", "40.mp4", "50.mp4"]


def pipeline(frame, model):
    # 1. ROI 裁切（edge 端）
    h, w, _ = frame.shape
    cx, cy = int(BOX[1]*w), int(BOX[2]*h)
    bw, bh = int(BOX[3]*w), int(BOX[4]*h)
    roi = frame[max(cy-bh//2,0):min(cy+bh//2,h),
                max(cx-bw//2,0):min(cx+bw//2,w)]
    # 2. Resize 到 88×275（edge 端）
    resized = cv2.resize(roi, (88, 275))
    # 3. JPEG 壓縮（edge 端傳輸）
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # 4. JPEG 解碼（inference 端）
    decoded = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    # 5. 再 resize（model_runner.py 會做一次，這裡是 no-op）
    decoded = cv2.resize(decoded, (88, 275))
    # 6. /255 正規化，BGR 不轉
    arr = decoded.astype(np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return float(model.predict(batch, verbose=0)[0][0])


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = tf.keras.models.load_model(os.path.join(PROJECT_ROOT, MODEL))

print(f"\n模型: {MODEL}  input_shape={model.input_shape}")
print(f"完整 pipeline（含 JPEG q80 壓縮/解碼）")
print("-" * 60)
print(f"{'Video':<12} {'期望':>6}  {'預測平均':>10}  {'誤差':>8}")
print("-" * 60)
for v in VIDEOS:
    cap = cv2.VideoCapture(os.path.join(PROJECT_ROOT, "Code", v))
    preds, idx = [], 0
    while len(preds) < 10:
        ok, f = cap.read()
        if not ok: break
        idx += 1
        if idx % 30: continue
        preds.append(pipeline(f, model))
    cap.release()
    avg = np.mean(preds)
    expected = int(v.split(".")[0])
    err = avg - expected
    print(f"{v:<12} {expected:>6}  {avg:>10.3f}  {err:>+8.3f}")
