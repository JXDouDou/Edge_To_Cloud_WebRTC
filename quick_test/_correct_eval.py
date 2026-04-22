"""根據 dataset.npz 確認的正確前處理，重新評估四個資料夾。

NPZ ground truth:
  images shape : (7493, 275, 88, 3)  → H=275, W=88, RGB, uint8
  labels       : [20.0, 30.0, 40.0, 50.0]

正確前處理（training 時完全一致）:
  1. PIL 開圖 → RGB
  2. resize 到 (W=88, H=275)：cv2.resize(img, (88, 275)) → shape (275, 88, 3)
  3. /255.0 normalize
  4. 只有 output_model_v1_0.25_ori.h5 的 input_shape=(275,88,3) 與 NPZ 完全吻合

同時也用 NPZ 裡的實際訓練樣本直接推論，確認模型本身是否正常。
"""

import os, sys, warnings, statistics
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import tensorflow as tf

BASE    = "D:/Lab/Code/Extracted_IMG"
NPZ     = "D:/Lab/Code/dataset.npz"
MODELS  = [
    "output_model_v1.h5",
    "output_model_v1_0.25.h5",
    "output_model_v1_0.25_ori.h5",
]
FOLDERS = [("20", 20), ("30", 30), ("40", 40), ("50", 50)]
BATCH   = 64

# ── 正確前處理 ──────────────────────────────────────────────
def preprocess_correct(img_bgr, out_w=88, out_h=275):
    """和訓練時完全一致：BGR→RGB → resize(88,275) → /255"""
    rgb   = img_bgr[:, :, ::-1]                 # BGR → RGB
    resized = cv2.resize(rgb, (out_w, out_h))   # → shape (275, 88, 3)
    return resized.astype("float32") / 255.0

def run_folder(model_tf, folder_path, out_w, out_h):
    paths = sorted([os.path.join(folder_path, f)
                    for f in os.listdir(folder_path) if f.endswith(".jpg")])
    preds = []
    for start in range(0, len(paths), BATCH):
        chunk = []
        for p in paths[start:start + BATCH]:
            img = cv2.imread(p)
            if img is None: continue
            chunk.append(preprocess_correct(img, out_w, out_h))
        if not chunk: continue
        out = model_tf.predict(np.stack(chunk), verbose=0)
        preds.extend(float(v[0]) for v in out)
    return preds

# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1：用 NPZ 訓練樣本直接推論（確認模型本身正常）")
print("=" * 65)

npz = np.load(NPZ)
npz_images = npz["images"].astype("float32") / 255.0   # uint8 → float
npz_labels = npz["labels"]
print(f"NPZ  images: {npz_images.shape}  labels: {npz_labels.shape}")
print(f"NPZ  labels 分布: { {int(v): int((npz_labels==v).sum()) for v in sorted(set(npz_labels))} }")
print()

for mname in MODELS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = tf.keras.models.load_model(mname)
    MH, MW = m.input_shape[1], m.input_shape[2]
    print(f"[{mname}]  input=({MH},{MW},3)")

    # 用 NPZ 裡的 images（已是正確 shape）直接預測
    # 但只有 _ori 的 input shape 與 NPZ (275,88,3) 一致
    if MH == 275 and MW == 88:
        preds = m.predict(npz_images, verbose=0, batch_size=128).flatten()
        for gt in [20, 30, 40, 50]:
            mask = npz_labels == gt
            p = preds[mask]
            print(f"  GT={gt:2d}  n={mask.sum():4d}  "
                  f"mean={p.mean():.3f}  median={np.median(p):.3f}  "
                  f"std={p.std():.3f}  err={p.mean()-gt:+.3f}")
    else:
        # 需要把 NPZ 的 (275,88,3) resize 成 (MH,MW,3)
        print(f"  NPZ shape (275,88,3) 與此模型 ({MH},{MW},3) 不符，重新 resize...")
        resized_imgs = np.stack([
            cv2.resize((npz_images[i]*255).astype("uint8"), (MW, MH)).astype("float32")/255.0
            for i in range(len(npz_images))
        ])
        preds = m.predict(resized_imgs, verbose=0, batch_size=128).flatten()
        for gt in [20, 30, 40, 50]:
            mask = npz_labels == gt
            p = preds[mask]
            print(f"  GT={gt:2d}  n={mask.sum():4d}  "
                  f"mean={p.mean():.3f}  median={np.median(p):.3f}  "
                  f"std={p.std():.3f}  err={p.mean()-gt:+.3f}")
    print()

# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 2：用 Extracted_IMG 正確前處理推論")
print(f"  前處理：BGR→RGB  resize(88,275)→shape(275,88,3)  /255")
print("=" * 65)
print()

for mname in MODELS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = tf.keras.models.load_model(mname)
    MH, MW = m.input_shape[1], m.input_shape[2]
    print(f"[{mname}]  input=({MH},{MW},3)")

    for folder_name, gt in FOLDERS:
        preds = run_folder(m, os.path.join(BASE, folder_name), 88, 275)
        mean  = statistics.mean(preds)
        std   = statistics.stdev(preds)
        err   = mean - gt
        print(f"  GT={gt:2d}  n={len(preds):4d}  "
              f"mean={mean:6.3f}  std={std:.3f}  err={err:+.3f}")
    print()

# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 3：視覺比較 NPZ 樣本 vs Extracted_IMG 樣本（統計）")
print("=" * 65)
print()

for folder_name, gt in FOLDERS:
    # NPZ 樣本（對應 label）
    mask      = npz_labels == gt
    npz_sub   = (npz_images[mask] * 255).astype("uint8")   # 還原為 uint8

    # Extracted_IMG 樣本（前5張）
    folder_path = os.path.join(BASE, folder_name)
    paths = sorted([os.path.join(folder_path, f)
                    for f in os.listdir(folder_path) if f.endswith(".jpg")])[:5]
    ext_imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            rgb = img[:,:,::-1]
            ext_imgs.append(cv2.resize(rgb, (88, 275)))

    npz_mean = npz_sub.mean()
    npz_std  = npz_sub.std()
    ext_mean = np.mean([im.mean() for im in ext_imgs]) if ext_imgs else 0
    ext_std  = np.mean([im.std()  for im in ext_imgs]) if ext_imgs else 0

    print(f"  label={gt:2d}  NPZ({len(npz_sub)}張)  pixel_mean={npz_mean:.1f}  std={npz_std:.1f}")
    print(f"         ExtImg({len(ext_imgs)}張)  pixel_mean={ext_mean:.1f}  std={ext_std:.1f}")
    print(f"         差異: mean_diff={abs(npz_mean-ext_mean):.1f}  "
          f"std_diff={abs(npz_std-ext_std):.1f}")
    print()
