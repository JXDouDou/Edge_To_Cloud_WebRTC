"""ROI 對照工具：把「原圖 + ROI 框 + 裁切 + 模型實際輸入」一次輸出成圖。

用途：每次移動機器 / 調背景打光 / 改 ROI 後，一鍵產生對照圖，
      確認餵進模型的畫面「像不像訓練資料」（背景、對比、液面清不清楚）。

輸出三張（預設前綴 cmp）：
  <prefix>_overlay.png   原圖 + 綠色 ROI 框（看框有沒有對準）
  <prefix>_crop.png      ROI 裁切（看裁出來的範圍）
  <prefix>_input.png     resize 成模型輸入尺寸（88x275）後的樣子 = 模型真正看到的
若有裝 tensorflow + .h5，會順便印出 prediction。

用法（專案根目錄）：
  # 最簡單：用 config 的攝影機抓一張來比（一個指令搞定）
  python scripts/roi_compare.py

  # 指定來源
  python scripts/roi_compare.py --image roi_frame.png
  python scripts/roi_compare.py --video edge/video/30.mp4 --frame 200
  python scripts/roi_compare.py --camera 0

  # 換 config / 輸出前綴 / 模型
  python scripts/roi_compare.py --config config/staging.yaml --prefix cam --model output_model_v1_0.25_ori.h5

比對重點：把 <prefix>_input.png 跟訓練資料的樣子（例如 cmp_vid30_input.png）並排，
看背景/對比/液面是否相近。差很多 → 先調背景打光，而不是怪 ROI。
"""

import argparse
import os
import sys
import warnings

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from shared.config import load_config


def log(tag, msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def _grab_frame(args, cap_cfg):
    """依參數取得一張 BGR 影像。"""
    import cv2

    if args.image:
        path = (os.path.join(PROJECT_ROOT, args.image)
                if not os.path.isabs(args.image) else args.image)
        if not os.path.exists(path):
            log("錯誤", f"找不到圖片：{path}")
            sys.exit(1)
        img = cv2.imread(path)
        if img is None:
            log("錯誤", f"無法讀取圖片：{path}")
            sys.exit(1)
        return img

    if args.video:
        path = (os.path.join(PROJECT_ROOT, args.video)
                if not os.path.isabs(args.video) else args.video)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log("錯誤", f"無法開啟影片：{path}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
        ok, f = cap.read()
        cap.release()
        if not ok:
            log("錯誤", "影片讀取失敗")
            sys.exit(1)
        return f

    # 攝影機（--camera N，或預設用 config 的 capture 設定）
    from edge.capture import FrameCapture
    if args.camera >= 0:
        cap_cfg.mode = "camera"
        cap_cfg.source = str(args.camera)
    cap_cfg.sample_mode = "fast_replay"
    capture = FrameCapture(cap_cfg)
    capture.open()
    for _ in range(10):           # 暖機
        capture.read()
    ok, f = capture.read()
    capture.release()
    if not ok:
        log("錯誤", "攝影機讀取失敗（被 edge/其他程式佔用？）")
        sys.exit(1)
    return f


def _load_model(model_path):
    """嘗試載入 KerasModel；失敗就回 None（仍可只出圖、不印預測）。"""
    if not model_path or not os.path.exists(model_path):
        log("提示", "找不到模型檔，略過預測（只出對照圖）")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from inference.model_runner import KerasModel
            return KerasModel(model_path)
    except Exception as e:
        log("提示", f"模型載入失敗（{e}），略過預測（只出對照圖）")
        return None


def main():
    parser = argparse.ArgumentParser(description="ROI 對照圖輸出工具")
    parser.add_argument("--config", default="config/staging.yaml", help="config 路徑（取 ROI）")
    parser.add_argument("--model", default="output_model_v1_0.25_ori.h5",
                        help="Keras .h5（印 prediction 用；沒有就只出圖）")
    parser.add_argument("--image", default="", help="來源：單張圖片")
    parser.add_argument("--video", default="", help="來源：影片")
    parser.add_argument("--frame", type=int, default=100, help="影片起始幀")
    parser.add_argument("--camera", type=int, default=-1, help="來源：攝影機索引")
    parser.add_argument("--prefix", default="cmp", help="輸出檔名前綴（預設 cmp）")
    parser.add_argument("--out-dir", default=".", help="輸出資料夾（預設目前目錄）")
    args = parser.parse_args()

    import cv2
    import numpy as np

    cfg = load_config(args.config)
    roi = cfg.edge.preprocess.roi
    log("ROI", f"x={roi.x} y={roi.y} w={roi.width} h={roi.height}（來自 {args.config}）")

    model_path = (os.path.join(PROJECT_ROOT, args.model)
                  if (args.model and not os.path.isabs(args.model)) else args.model)
    model = _load_model(model_path)

    # 模型輸入尺寸：有模型用模型的，沒有就用 config 的 resize
    if model is not None:
        in_w, in_h = model._input_w, model._input_h
    else:
        in_w = cfg.edge.preprocess.resize_width or 88
        in_h = cfg.edge.preprocess.resize_height or 275

    frame = _grab_frame(args, cfg.edge.capture)
    H, W = frame.shape[:2]
    log("影像", f"{W}x{H}")

    # ── 裁切 ROI（夾在邊界內）──
    x1 = max(0, roi.x); y1 = max(0, roi.y)
    x2 = min(W, roi.x + roi.width); y2 = min(H, roi.y + roi.height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        log("錯誤", "ROI 在畫面外，裁出空圖！檢查 config 的 ROI 座標")
        sys.exit(1)

    # ── 模型輸入（resize 到 in_w x in_h，保持 BGR，與正式推論一致）──
    model_input = cv2.resize(crop, (in_w, in_h))

    # ── 預測（與 KerasModel.predict 相同：BGR、/255）──
    pred_text = "prediction: N/A"
    if model is not None:
        try:
            arr = model_input.astype(np.float32)
            if model._normalize:
                arr = arr / 255.0
            out = model._model(np.expand_dims(arr, 0), training=False)
            pred = float(np.asarray(out).reshape(-1)[0])
            pred_text = f"prediction: {pred:.4f}"
            log("預測", pred_text)
        except Exception as e:
            log("提示", f"預測失敗：{e}")

    # ── overlay：原圖畫 ROI 框 + 預測 ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(overlay, f"ROI {roi.width}x{roi.height}  {pred_text}",
                (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(args.out_dir, exist_ok=True)
    p = args.prefix
    paths = {
        f"{p}_overlay.png": overlay,
        f"{p}_crop.png": crop,
        f"{p}_input.png": model_input,
    }
    for name, img in paths.items():
        out = os.path.join(args.out_dir, name)
        cv2.imwrite(out, img)
        log("輸出", os.path.abspath(out))

    print()
    log("完成", f"把 {p}_input.png 跟訓練資料（如 cmp_vid30_input.png）並排比對："
                "背景/對比/液面像不像 → 不像就先調背景打光")


if __name__ == "__main__":
    main()
