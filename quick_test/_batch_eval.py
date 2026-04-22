import os, sys, warnings, time, statistics
import cv2, numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from inference.model_runner import KerasModel

BASE    = "D:/Lab/Code/Extracted_IMG"
FOLDERS = [("20", 20), ("30", 30), ("40", 40), ("50", 50)]
MODELS  = [
    "output_model_v1.h5",
    "output_model_v1_0.25.h5",
    "output_model_v1_0.25_ori.h5",
]
BATCH = 64


def run_folder(model, folder_path):
    MH, MW = model._input_h, model._input_w
    paths = sorted([os.path.join(folder_path, f)
                    for f in os.listdir(folder_path) if f.endswith(".jpg")])
    preds = []
    for start in range(0, len(paths), BATCH):
        chunk = []
        for p in paths[start:start + BATCH]:
            img = cv2.imread(p)
            if img is None:
                continue
            r   = cv2.resize(img, (MW, MH))
            rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            chunk.append(rgb)
        if not chunk:
            continue
        out = model._model.predict(np.stack(chunk), verbose=0)
        preds.extend(float(v[0]) for v in out)
    return preds


# ── 載入三個模型 ──
models = {}
for mname in MODELS:
    print(f"載入 {mname} ...", end=" ", flush=True)
    models[mname] = KerasModel(mname)
    print("OK")
print()

# ── 收集所有結果 ──
# results[folder_name][mname] = preds list
results = {}
for folder_name, gt in FOLDERS:
    folder_path = os.path.join(BASE, folder_name)
    n = len([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    results[folder_name] = {"_gt": gt, "_n": n}
    for mname in MODELS:
        results[folder_name][mname] = run_folder(models[mname], folder_path)

# ── 印主表格 ──
SHORT = {
    "output_model_v1.h5":         "v1",
    "output_model_v1_0.25.h5":    "v1_0.25",
    "output_model_v1_0.25_ori.h5":"v1_0.25_ori",
}

print("=" * 78)
print(f"  {'Folder':>6}  GT    N    | {'v1':^18} | {'v1_0.25':^18} | {'v1_0.25_ori':^18}")
print(f"  {'':->6}  {'':->2}  {'':->4}   {'':->18}   {'':->18}   {'':->18}")

for folder_name, gt in FOLDERS:
    row_parts = []
    for mname in MODELS:
        preds = results[folder_name][mname]
        mean  = statistics.mean(preds)
        std   = statistics.stdev(preds)
        err   = mean - gt
        row_parts.append(f"mean={mean:5.2f} err={err:+5.2f} s={std:.2f}")
    n = results[folder_name]["_n"]
    print(f"  {folder_name:>6}  {gt:>2}  {n:>4}   "
          f"{row_parts[0]:^18}   {row_parts[1]:^18}   {row_parts[2]:^18}")

# ── MAE 摘要 ──
print()
print("  MAE (各資料夾 mean 和 ground truth 的平均絕對誤差):")
for mname in MODELS:
    errs = []
    for folder_name, gt in FOLDERS:
        preds = results[folder_name][mname]
        errs.append(abs(statistics.mean(preds) - gt))
    print(f"    {SHORT[mname]:15s}: MAE = {statistics.mean(errs):.3f}"
          f"   per-folder: " + "  ".join(f"{e:.2f}" for e in errs))

print("=" * 78)
