"""在 Pi（或任何接攝影機的機器）抓「當前畫面」，給 ROI 選取用。

headless 也能跑（不開任何視窗，只存檔），所以 Pi 沒有桌面也沒問題。
用 edge config 的 capture 設定開攝影機，確保抓到的畫面和 edge 實際送出的
解析度 / 格式（MJPG、1920x1080 等）完全一致 —— 這樣選出的 ROI 才會準。

用法（在 Pi 上，專案根目錄）：
  python scripts/grab_frame.py --config config/staging.yaml                # 存一張 roi_frame.png
  python scripts/grab_frame.py --config config/staging.yaml --out shot.png
  python scripts/grab_frame.py --config config/staging.yaml --clip 3       # 改存 3 秒影片 roi_clip.mp4
  python scripts/grab_frame.py --config config/staging.yaml --source 0     # 覆寫攝影機索引

抓完把檔案傳到桌機（例：在桌機跑）：
  scp mawatarilab@100.98.101.5:~/Project_Edge/roi_frame.png .
再用 ROI 選取工具：
  python quick_test/roi_selector.py --model output_model_v1_0.25_ori.h5 --image roi_frame.png
"""

import argparse
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shared.config import load_config
from edge.capture import FrameCapture


def main():
    parser = argparse.ArgumentParser(description="抓攝影機當前畫面（給 ROI 選取用）")
    parser.add_argument("--config", default="config/staging.yaml", help="edge config 路徑")
    parser.add_argument("--out", default="", help="輸出檔名（預設 roi_frame.png / roi_clip.mp4）")
    parser.add_argument("--source", default="", help="覆寫攝影機索引/來源（如 0）")
    parser.add_argument("--mode", default="camera", choices=["camera", "video"],
                        help="來源類型（預設 camera）")
    parser.add_argument("--warmup", type=int, default=15,
                        help="先丟掉幾幀讓自動曝光穩定（預設 15）")
    parser.add_argument("--clip", type=float, default=0.0,
                        help="改成錄影：錄幾秒（0 = 只存一張圖）")
    args = parser.parse_args()

    import cv2

    cfg = load_config(args.config)
    cap_cfg = cfg.edge.capture
    if args.mode:
        cap_cfg.mode = args.mode
    if args.source:
        cap_cfg.source = args.source
    # 抓畫面不需要降取樣/節流，全速拿最即時的
    cap_cfg.sample_mode = "fast_replay"

    capture = FrameCapture(cap_cfg)
    capture.open()

    # 暖機：丟掉前幾幀（攝影機剛開時常常偏暗/對焦未穩）
    for _ in range(max(0, args.warmup)):
        capture.read()

    if args.clip > 0:
        out = args.out or "roi_clip.mp4"
        ok, frame = capture.read()
        if not ok:
            print("讀取失敗，無法錄影")
            capture.release()
            sys.exit(1)
        h, w = frame.shape[:2]
        fps = max(1, cap_cfg.fps)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        t0 = time.time()
        n = 0
        writer.write(frame)
        n += 1
        while time.time() - t0 < args.clip:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            n += 1
        writer.release()
        capture.release()
        print(f"已存 {args.clip}s 影片：{os.path.abspath(out)}（{w}x{h}, {n} 幀）")
    else:
        out = args.out or "roi_frame.png"
        ok, frame = capture.read()
        capture.release()
        if not ok:
            print("讀取失敗，無法截圖")
            sys.exit(1)
        cv2.imwrite(out, frame)
        h, w = frame.shape[:2]
        print(f"已存截圖：{os.path.abspath(out)}（{w}x{h}）")


if __name__ == "__main__":
    main()
