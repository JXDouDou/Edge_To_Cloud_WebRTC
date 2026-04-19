"""產生測試用影片檔。

當沒有真實影片可用時，執行此腳本產生一個簡單的測試影片：
- 640×480 解析度
- 30 fps
- 5 秒長（150 幀）
- 內容：移動的彩色方塊 + 幀序號文字

使用方式：
    cd distributed-inference
    python test_data/generate_test_video.py

產生的檔案：
    test_data/test_video.mp4
"""

import os
import sys
import cv2
import numpy as np

# Windows CP932 / CP936 terminal 無法顯示中文，強制 UTF-8 輸出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def generate_test_video(
    output_path: str = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    duration_sec: int = 5,
):
    """產生測試影片。

    影片內容：
    - 深灰色背景
    - 一個水平來回移動的綠色方塊
    - 左上角顯示幀序號（方便追蹤資料流）

    Args:
        output_path:  輸出檔案路徑（預設 test_data/test_video.mp4）
        width:        影片寬度
        height:       影片高度
        fps:          幀率
        duration_sec: 影片時長（秒）
    """
    if output_path is None:
        # 預設路徑：此腳本所在目錄下的 test_video.mp4
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "test_video.mp4")

    total_frames = fps * duration_sec

    # 使用 mp4v 編碼器（跨平台相容性較好）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"錯誤：無法建立影片檔: {output_path}")
        print("請確認 OpenCV 有安裝 ffmpeg 支援")
        sys.exit(1)

    box_size = 80
    box_y = height // 2 - box_size // 2

    print(f"產生測試影片: {output_path}")
    print(f"  解析度: {width}×{height}")
    print(f"  幀率:   {fps} fps")
    print(f"  時長:   {duration_sec} 秒 ({total_frames} 幀)")

    for i in range(total_frames):
        # 深灰色背景
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # 移動的綠色方塊（水平來回）
        progress = i / total_frames
        # 用 sin 函式讓方塊來回移動
        box_x = int((width - box_size) * (0.5 + 0.5 * np.sin(progress * 4 * np.pi)))
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_size, box_y + box_size),
            (0, 220, 0),  # 綠色 (BGR)
            -1,            # 填滿
        )

        # 顯示幀序號文字
        text = f"Frame: {i:04d}"
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        # 顯示時間戳
        time_text = f"Time: {i / fps:.2f}s"
        cv2.putText(
            frame, time_text, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
        )

        writer.write(frame)

    writer.release()
    print(f"完成！檔案大小: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    generate_test_video()
