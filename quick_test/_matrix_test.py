"""矩陣測試：模型 x 影片 x 色彩空間 的所有組合。

目的：找出哪個模型 + 哪個色彩空間（BGR vs RGB）能對每支影片
輸出接近檔名所示的高度（20、30、40、50）。
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
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

MODELS = [
    "output_model_v1.h5",
    "output_model_v1_0.25.h5",
    "output_model_v1_0.25_ori.h5",
]
VIDEOS = ["edge/video/20.mp4", "edge/video/30.mp4",
          "edge/video/40.mp4", "edge/video/50.mp4"]

N_FRAMES = 5      # 每支影片取樣幾幀
SKIP = 10         # 每幾幀取一幀


def sample_frames(video_path, n, skip):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
            if not ok:
                break
        idx += 1
        if idx % skip == 0:
            # 模擬 edge 的 JPEG 壓縮（quality=80）
            ok2, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            decoded = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8),
                                   cv2.IMREAD_COLOR)
            frames.append(decoded)
    cap.release()
    return frames


def predict(model, frames, color_space):
    """color_space: 'BGR' 保持原樣; 'RGB' 做 cv2.cvtColor。"""
    H = model.input_shape[1]
    W = model.input_shape[2]
    preds = []
    for f in frames:
        r = cv2.resize(f, (W, H))
        if color_space == "RGB":
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        arr = r.astype(np.float32) / 255.0
        batch = np.expand_dims(arr, axis=0)
        out = model.predict(batch, verbose=0)
        preds.append(float(out[0][0]))
    return preds


def main():
    print()
    print("=" * 88)
    print(f"{'Model':<32} {'Video':<22} {'BGR (avg)':>12} {'RGB (avg)':>12}  期望")
    print("=" * 88)

    # 預先載入全部模型
    loaded = {}
    for m in MODELS:
        path = os.path.join(PROJECT_ROOT, m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded[m] = tf.keras.models.load_model(path)
        in_shape = loaded[m].input_shape
        print(f"  已載入 {m}  input_shape={in_shape}")
    print("-" * 88)

    for m in MODELS:
        for v in VIDEOS:
            video_abs = os.path.join(PROJECT_ROOT, v)
            frames = sample_frames(video_abs, N_FRAMES, SKIP)
            if not frames:
                continue
            bgr_preds = predict(loaded[m], frames, "BGR")
            rgb_preds = predict(loaded[m], frames, "RGB")
            expected = os.path.basename(v).split(".")[0]
            print(f"{m:<32} {v:<22} "
                  f"{np.mean(bgr_preds):>12.3f} "
                  f"{np.mean(rgb_preds):>12.3f}   {expected}")
        print("-" * 88)

    print()
    print("解讀：欄位值越接近『期望』那個數字，代表那組合越正確。")
    print()


if __name__ == "__main__":
    main()
