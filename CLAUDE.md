# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A distributed video inference pipeline: edge devices (Raspberry Pi) capture and preprocess video frames, stream them via WebRTC to dispatcher nodes (EC2), which forward frames to a centralized inference server for ML model processing. Designed for real-time object detection with automatic failover across dispatchers.

## Running the System

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Local end-to-end test (spins up all 5 components):**
```bash
python run_local_test.py
# or explicitly:
python run_local_test.py --config config/test.yaml
```

**Run individual components:**
```bash
python signaling/server.py --config config/test.yaml
python inference/main.py --config config/test.yaml
python dispatcher/main.py --config config/test.yaml --id dispatcher-001
python dispatcher/main.py --config config/test.yaml --id dispatcher-002
python edge/main.py --config config/test.yaml
```

There is no formal test framework, linter, or CI configuration in this repo.

## Architecture

```
Edge (RPi)  ──WebRTC──►  Dispatcher-1 (EC2) ─┐
                    └──►  Dispatcher-2 (EC2) ──┴──WebSocket──►  Inference Server
         ◄──── result ──────────────────────────────────────────◄
                     Signaling Server (WebSocket) coordinates WebRTC setup
```

**Data flow:**
1. Edge captures frames → ROI crop → resize → JPEG encode
2. Edge sends binary frames over WebRTC data channel to a Dispatcher
3. Dispatcher injects `edge_id` into the frame header, forwards to Inference via WebSocket
4. Inference decodes JPEG, runs model, returns JSON detections
5. Dispatcher routes result back to the originating Edge

**Failover:** Edge tracks consecutive failures per dispatcher; after N failures it switches to the next dispatcher in the config list.

## Binary Frame Protocol (`shared/protocol.py`)

All frames use a simple length-prefixed binary format:
```
[2 bytes: header_len (big-endian)] [JSON header bytes] [JPEG bytes]
```
Header JSON includes `frame_id`, `edge_id`, `seq`. Signaling messages are plain JSON with a `type` field (register, offer, answer, request_dispatchers, dispatcher_list, ping, pong, result).

## Configuration (`config/`)

- `config/test.yaml` — localhost, dummy inference model (no GPU required), 2 dispatchers, video file input
- `config/prod.yaml` — real camera, YOLO on CUDA GPU, SSL/TLS signaling, STUN/TURN for NAT traversal

`shared/config.py` loads YAML into typed dataclasses (`EdgeConfig`, `DispatcherConfig`, `InferenceConfig`, etc.).

## Inference Models (`inference/model_runner.py`)

Model backend controlled by `inference.model_type` in config:
- `dummy` — returns fake detections (use for local testing without a GPU)
- `onnx` — **default / GPU path.** ONNX Runtime with CUDAExecutionProvider (requires `onnxruntime-gpu` and a `.onnx` file). Needed for RTX 5080 on native Windows, where TensorFlow GPU support ended at TF 2.10. Honors `inference.device` (`cuda`/`cpu`); falls back to CPU with a warning if CUDA can't load.
- `keras` — CPU fallback. TensorFlow `.h5` model (`tensorflow-cpu` + `h5py`).
- `yolo` — Ultralytics YOLOv8 (requires `ultralytics` package and a `.pt` model file)

`onnx` and `keras` share identical pre/post-processing (BGR, resize, /255, regression output), so they are drop-in interchangeable.

**Switching GPU (onnx) ↔ CPU (keras):** change two lines in the inference config:
```yaml
inference:
  model_type: "onnx"     # GPU (default)  | "keras" for CPU fallback
  model_path: "model.onnx"  # .onnx for onnx | .h5 for keras
  device: "cuda"         # "cuda" for GPU  | "cpu"
```
Generate the `.onnx` from your `.h5` with `python scripts/convert_to_onnx.py <model>.h5` (also verifies keras vs onnx accuracy). Both runtimes coexist, so reverting is just editing the config back.

## Extensibility Points

- **New inference backend:** subclass `BaseModel` in `inference/model_runner.py`
- **Result handling on edge:** register async handlers via `edge/controller.py`
- **Custom preprocessing:** extend `Preprocessor` in `edge/preprocess.py`
