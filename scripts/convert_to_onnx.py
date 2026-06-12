"""把 Keras .h5 模型轉成 ONNX，並驗證兩者預測一致。

為什麼要轉 ONNX：
  RTX 5080 (Blackwell) 在「原生 Windows」上跑不了 TensorFlow GPU
  （TF 原生 Windows GPU 支援停在 2.10 / CUDA 11.2，認不得 5080）。
  轉成 .onnx 後用 onnxruntime-gpu 就能在原生 Windows 直接吃 GPU。

安裝（在 inference 那台桌機）：
    pip install tf2onnx onnxruntime          # 轉換 + 驗證用
    # 之後實際推論要 GPU 版：pip install onnxruntime-gpu

用法：
    python scripts/convert_to_onnx.py output_model_v1_0.25_ori.h5
    python scripts/convert_to_onnx.py output_model_v1_0.25_ori.h5 --out model.onnx

轉完會用隨機輸入比對 keras vs onnx 的輸出，差異應 < 1e-4；
若差異很大代表轉壞了，先不要拿去用。
"""

import argparse
import os
import sys

# Windows 主控台預設 cp932/cp950，無法輸出中文，強制 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Keras .h5 → ONNX 轉換 + 精度驗證")
    parser.add_argument("h5_path", help="輸入的 Keras .h5 模型路徑")
    parser.add_argument(
        "--out", default="",
        help="輸出 .onnx 路徑（預設：與 .h5 同名換副檔名）",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本")
    parser.add_argument(
        "--samples", type=int, default=5,
        help="用幾組隨機輸入做 keras vs onnx 精度比對",
    )
    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"❌ 找不到模型檔：{args.h5_path}")
        sys.exit(1)

    out_path = args.out or (os.path.splitext(args.h5_path)[0] + ".onnx")

    # 抑制 TF 啟動雜訊
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import warnings
    warnings.simplefilter("ignore")

    try:
        import tensorflow as tf
        import tf2onnx
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"❌ 缺少套件：{exc}\n   請先：pip install tf2onnx onnxruntime tensorflow-cpu")
        sys.exit(1)

    print(f"📥 載入 Keras 模型：{args.h5_path}")
    model = tf.keras.models.load_model(args.h5_path)
    in_shape = model.input_shape          # (None, H, W, 3)
    h, w, c = in_shape[1], in_shape[2], in_shape[3]
    print(f"   輸入形狀：(None, {h}, {w}, {c})")

     # ── 轉檔 ──
    print(f"🔄 轉換成 ONNX（opset={args.opset}）...")
    spec = (tf.TensorSpec((None, h, w, c), tf.float32, name="input"),)

    @tf.function(input_signature=spec)
    def model_fn(x):
        return model(x, training=False)

    concrete_func = model_fn.get_concrete_function()
    tf2onnx.convert.from_function(
        concrete_func,
        input_signature=spec,
        opset=args.opset,
        output_path=out_path,
    )
    print(f"✅ 已輸出：{out_path}")

    # ── 精度驗證 ──
    print(f"🔍 用 {args.samples} 組隨機輸入比對 keras vs onnx ...")
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    max_diff = 0.0
    for i in range(args.samples):
        x = np.random.rand(1, h, w, c).astype(np.float32)
        keras_out = float(np.asarray(model(x, training=False)).reshape(-1)[0])
        onnx_out = float(np.asarray(sess.run(None, {in_name: x})[0]).reshape(-1)[0])
        diff = abs(keras_out - onnx_out)
        max_diff = max(max_diff, diff)
        print(f"   #{i}: keras={keras_out:.6f}  onnx={onnx_out:.6f}  diff={diff:.2e}")

    print(f"\n最大誤差：{max_diff:.2e}")
    if max_diff < 1e-4:
        print("✅ 通過：onnx 與 keras 輸出一致，可放心使用。")
    else:
        print("⚠️ 誤差偏大！請檢查轉換流程，先別拿去推論。")


if __name__ == "__main__":
    main()
