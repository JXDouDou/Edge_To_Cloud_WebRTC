"""互動式 ROI 選取工具

在影片幀上用滑鼠框選感興趣區域（管子），
即時預覽裁切後餵入模型的效果，輸出要貼進 YAML 的座標。

使用方式（專案根目錄執行）：
  python quick_test/roi_selector.py --model output_model_v1.h5
  python quick_test/roi_selector.py --model output_model_v1.h5 --video edge/video/40.mp4
  python quick_test/roi_selector.py --model output_model_v1.h5 --frame 500   # 跳到第 500 幀

操作說明：
  滑鼠拖曳   → 框選 ROI
  r          → 重新選取
  ← →        → 切換到前/後一幀（找好的幀）
  Enter      → 確認並輸出座標
  ESC / q    → 離開
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


def log(tag, msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


# ── 全域滑鼠狀態 ────────────────────────────────────────────
_drawing = False
_roi_start = (-1, -1)
_roi_end = (-1, -1)
_roi_confirmed = None   # (x, y, w, h) 確認後存這裡


def _mouse_cb(event, x, y, flags, param):
    """OpenCV 滑鼠回呼：拖曳畫框。"""
    global _drawing, _roi_start, _roi_end, _roi_confirmed
    import cv2

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _roi_start = (x, y)
        _roi_end = (x, y)
        _roi_confirmed = None

    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _roi_end = (x, y)


def _get_roi_rect(scale: float):
    """把滑鼠座標換算回原始影像座標並正規化（確保 x1<x2, y1<y2）。

    Returns:
        (x, y, w, h) in 原始影像像素，或 None（若框太小）
    """
    x1 = int(min(_roi_start[0], _roi_end[0]) / scale)
    y1 = int(min(_roi_start[1], _roi_end[1]) / scale)
    x2 = int(max(_roi_start[0], _roi_end[0]) / scale)
    y2 = int(max(_roi_start[1], _roi_end[1]) / scale)
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None
    return (x1, y1, w, h)


def make_preview(orig_frame, model, roi_rect, scale):
    """製作即時預覽面板（原始幀 + ROI 方框 + 裁切後 + 模型輸入 + 預測值）。

    Args:
        orig_frame: 原始 BGR 幀
        model:      已載入的 KerasModel（None 代表尚未載入）
        roi_rect:   (x, y, w, h) 或 None
        scale:      顯示縮放比例（原始→顯示）

    Returns:
        BGR 面板影像
    """
    import cv2
    import numpy as np

    PANEL_H = 360

    # ── 左欄：原始幀 + ROI 框 ──
    disp_frame = cv2.resize(orig_frame, None, fx=scale, fy=scale)
    if roi_rect is not None:
        x, y, w, h = roi_rect
        sx, sy = int(x * scale), int(y * scale)
        sw, sh = int(w * scale), int(h * scale)
        cv2.rectangle(disp_frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 80), 2)
        cv2.putText(disp_frame, f"ROI {w}x{h}", (sx + 4, sy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1, cv2.LINE_AA)

    left_h = int(orig_frame.shape[0] * scale)
    left_w = int(orig_frame.shape[1] * scale)
    left = cv2.resize(disp_frame, (left_w, left_h))

    # ── 右欄：裁切圖 + 模型輸入 ──
    MH = model._input_h if model else 88
    MW = model._input_w if model else 275

    if roi_rect is not None:
        rx, ry, rw, rh = roi_rect
        # 確保不超出邊界
        ry2, rx2 = min(ry + rh, orig_frame.shape[0]), min(rx + rw, orig_frame.shape[1])
        crop = orig_frame[ry:ry2, rx:rx2]

        # 裁切後顯示（縮放到顯示高度）
        crop_disp_h = min(PANEL_H, rh)
        crop_disp_w = int(rw * crop_disp_h / max(rh, 1))
        crop_disp = cv2.resize(crop, (crop_disp_w, crop_disp_h))

        # 模型實際輸入（resize 到 MH×MW）
        model_in = cv2.resize(crop, (MW, MH))
        # 顯示時放大
        model_disp_h = PANEL_H
        model_disp_w = int(MW * model_disp_h / max(MH, 1))
        model_disp = cv2.resize(model_in, (model_disp_w, model_disp_h))

        # 加文字標題
        def add_label(img, text, sub=""):
            out = img.copy()
            cv2.rectangle(out, (0, 0), (out.shape[1], 38), (30, 30, 30), -1)
            cv2.putText(out, text, (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            if sub:
                cv2.putText(out, sub, (5, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 200, 140), 1, cv2.LINE_AA)
            return out

        # 裁切圖要補到 PANEL_H 高
        if crop_disp.shape[0] < PANEL_H:
            pad = np.zeros((PANEL_H - crop_disp.shape[0], crop_disp.shape[1], 3), dtype="uint8")
            crop_disp = np.vstack([crop_disp, pad])

        crop_disp = add_label(crop_disp, f"ROI crop  ({rw}x{rh}px)", "cropped region")
        model_disp = add_label(model_disp,
                               f"Model input  ({MW}x{MH}px)",
                               f"/255 normalized  → predict")

        # 跑推論
        pred_text = "prediction: ..."
        if model is not None:
            try:
                rgb = cv2.cvtColor(model_in, cv2.COLOR_BGR2RGB)
                arr = rgb.astype("float32")
                if model._normalize:
                    arr /= 255.0
                import numpy as _np
                out_val = model._model.predict(_np.expand_dims(arr, 0), verbose=0)
                pred = float(out_val[0][0])
                pred_text = f"prediction: {pred:.4f}"
            except Exception as e:
                pred_text = f"prediction: ERROR {e}"

        # 把預測值貼在 model_disp 上
        bw = model_disp.shape[1]
        cv2.rectangle(model_disp,
                      (0, model_disp.shape[0] - 40), (bw, model_disp.shape[0]),
                      (20, 20, 60), -1)
        cv2.putText(model_disp, pred_text,
                    (6, model_disp.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (80, 255, 120), 2, cv2.LINE_AA)

        right = np.hstack([crop_disp, model_disp])

    else:
        # 還沒框選時顯示提示
        ph = PANEL_H
        pw = 600
        right = np.zeros((ph, pw, 3), dtype="uint8")
        msg_lines = [
            "Drag mouse on the left to select ROI",
            "",
            f"Model input size: {MW} x {MH} px",
            "  (W x H)",
            "",
            "Left/Right arrow: change frame",
            "r: reset   Enter: confirm",
        ]
        for i, line in enumerate(msg_lines):
            cv2.putText(right, line, (20, 60 + i * 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    # ── 上下對齊後水平拼接 ──
    lh, lw = left.shape[:2]
    rh_cur, rw_cur = right.shape[:2]

    # 統一高度
    target_h = max(lh, rh_cur)
    if lh < target_h:
        pad = np.zeros((target_h - lh, lw, 3), dtype="uint8")
        left = np.vstack([left, pad])
    if rh_cur < target_h:
        pad = np.zeros((target_h - rh_cur, rw_cur, 3), dtype="uint8")
        right = np.vstack([right, pad])

    panel = np.hstack([left, right])

    # 底部提示列
    bar_h = 30
    bar = np.full((bar_h, panel.shape[1], 3), (40, 40, 40), dtype="uint8")
    guide = ("Drag=select ROI  |  r=reset  |  Left/Right=prev/next frame  |  Enter=confirm & output coords  |  q/ESC=quit")
    cv2.putText(bar, guide, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    panel = np.vstack([panel, bar])

    return panel


def main():
    parser = argparse.ArgumentParser(description="互動式 ROI 選取工具")
    parser.add_argument("--model", required=True, help="Keras .h5 模型路徑")
    parser.add_argument("--video", default="edge/video/30.mp4", help="影片路徑")
    parser.add_argument("--frame", type=int, default=100, help="起始幀（預設第 100 幀）")
    args = parser.parse_args()

    import cv2

    model_path = (os.path.join(PROJECT_ROOT, args.model)
                  if not os.path.isabs(args.model) else args.model)
    video_path = (os.path.join(PROJECT_ROOT, args.video)
                  if not os.path.isabs(args.video) else args.video)

    for p, n in [(model_path, "模型"), (video_path, "影片")]:
        if not os.path.exists(p):
            log("錯誤", f"找不到{n}：{p}")
            sys.exit(1)

    # ── 載入模型 ──
    log("模型", f"載入中：{args.model}（請稍候...）")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from inference.model_runner import KerasModel
        model = KerasModel(model_path)
    log("模型", f"輸入尺寸 H×W = {model._input_h}×{model._input_w}")

    # ── 開啟影片 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("錯誤", f"無法開啟影片：{video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log("影片", f"{orig_w}x{orig_h}  共 {total} 幀")

    # 顯示縮放比（讓左欄不超過螢幕高度 720px）
    scale = min(1.0, 720 / orig_h)
    log("顯示", f"縮放比例 = {scale:.2f}（顯示尺寸 {int(orig_w*scale)}x{int(orig_h*scale)}）")
    log("操作", "用滑鼠在原始幀上拖曳框選管子 ROI，Enter 確認輸出座標")
    print()

    frame_idx = max(0, min(args.frame, total - 1))

    # 讀取指定幀
    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    frame = read_frame(frame_idx)
    if frame is None:
        log("錯誤", "讀取幀失敗")
        sys.exit(1)

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Selector", _mouse_cb)

    global _roi_start, _roi_end, _drawing

    while True:
        roi_rect = _get_roi_rect(scale)
        panel = make_preview(frame, model, roi_rect, scale)

        # 幀號標示
        cv2.putText(panel, f"frame {frame_idx}/{total-1}", (8, panel.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 255), 1, cv2.LINE_AA)

        # 滑鼠拖曳中：即時畫框（在 panel 的左欄部分）
        if _drawing or roi_rect is not None:
            x1d = int(min(_roi_start[0], _roi_end[0]))
            y1d = int(min(_roi_start[1], _roi_end[1]))
            x2d = int(max(_roi_start[0], _roi_end[0]))
            y2d = int(max(_roi_start[1], _roi_end[1]))
            cv2.rectangle(panel, (x1d, y1d), (x2d, y2d), (0, 200, 255), 1)

        cv2.imshow("ROI Selector", panel)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):           # ESC / q 離開
            break

        elif key == ord("r"):               # 重新選取
            _roi_start = (-1, -1)
            _roi_end = (-1, -1)
            _drawing = False

        elif key == 81 or key == 2:         # ← 上一幀（Linux: 2, Windows: 81）
            frame_idx = max(0, frame_idx - 30)
            frame = read_frame(frame_idx)

        elif key == 83 or key == 3:         # → 下一幀（Linux: 3, Windows: 83）
            frame_idx = min(total - 1, frame_idx + 30)
            frame = read_frame(frame_idx)

        elif key == 13:                     # Enter 確認
            roi_rect = _get_roi_rect(scale)
            if roi_rect is None:
                log("提示", "請先框選 ROI 再按 Enter")
                continue

            rx, ry, rw, rh = roi_rect
            print()
            print("=" * 55)
            print("  ROI 座標確認")
            print("=" * 55)
            print(f"  原始幀尺寸：{orig_w} x {orig_h}")
            print(f"  ROI 座標  ：x={rx}  y={ry}  w={rw}  h={rh}")
            print(f"  模型輸入  ：{model._input_w} x {model._input_h}")
            print()
            print("  貼進 YAML 的設定（video_test.yaml / h5_test.yaml）：")
            print()
            print("    preprocess:")
            print("      roi:")
            print("        enabled: true")
            print(f"        x: {rx}")
            print(f"        y: {ry}")
            print(f"        width: {rw}")
            print(f"        height: {rh}")
            print("      jpeg_quality: 80")
            print(f"      resize_width: {model._input_w}")
            print(f"      resize_height: {model._input_h}")
            print("=" * 55)
            print()
            log("完成", "座標已輸出，複製上方 YAML 貼入設定檔即可")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
