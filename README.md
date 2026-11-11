# Edge-to-Cloud WebRTC Inference Pipeline

A cloud–edge system that closes an image-based **microfluidic control loop over the network**. A Raspberry Pi captures images and controls the pump, while volume estimation runs on a remote GPU server instead of a dedicated per-instrument workstation.

Master's thesis project, Mawatari Lab, Graduate School of Information, Production and Systems, Waseda University (2024–2026): *Design of a Cloud–Edge Collaborative System for Image-Based Microfluidic Control*.

## Architecture

```
Edge (Raspberry Pi) ──WebRTC──► Dispatcher-1 (EC2) ─┐
                    └─────────► Dispatcher-2 (EC2) ─┴──WebSocket──► Inference Server (GPU)
          ◄──────────────────── result ─────────────────────────────────┘
                     Signaling Server (WebSocket) coordinates WebRTC setup
```

| Component | Runs on | Role |
| --- | --- | --- |
| `edge/` | Raspberry Pi | Captures frames, crops the ROI, JPEG-encodes, sends over a WebRTC data channel; makes the final control decision (pump PWM) |
| `dispatcher/` | AWS EC2 | Relays frames to the inference server and routes results back to the right edge device; two instances for failover |
| `signaling/` | AWS EC2 | WebSocket server that registers nodes and exchanges WebRTC offers/answers |
| `inference/` | GPU workstation | Runs the volume-estimation model (ONNX Runtime on GPU, or Keras on CPU) |

**Hybrid design:** inference runs in the cloud, but the control decision stays on the edge, so a network delay never blocks the pump control directly.

## Key features

- **Binary frame protocol** (`shared/protocol.py`): `[2-byte header length][JSON header][JPEG bytes]`
- **Failover:** the edge switches to the standby dispatcher on PONG timeout or repeated failures; stale connection events are ignored and each incident triggers only one switch
- **Liveness detection:** any incoming dispatcher message counts as proof of liveness, which prevents false failovers on lossy 5G links
- **Pluggable model backends** (`inference/model_runner.py`): `onnx` (GPU), `keras` (CPU), `dummy` (for testing)
- **Hardware abstraction** (`edge/pump.py`): the default `LogPump` only logs, so the edge runs on any machine; real GPIO control lives in a Pi-only `edge/pump_local.py` (see `edge/pump_local.py.example`)
- **Keras → ONNX conversion:** `scripts/convert_to_onnx.py` converts the `.h5` model and checks that both give the same output

## Quick start: run everything on one machine

The full system normally runs on four machines, but `run_local_test.py` starts all five processes locally (signaling, inference, two dispatchers, edge) and streams a video through the whole pipeline. No GPU, Raspberry Pi or cloud server is needed.

```bash
pip install -r requirements.txt

# Dummy model (no GPU, fastest); a test video is generated automatically
python run_local_test.py --duration 30

# Real Keras model on a sample microfluidics video
python run_local_test.py --model output_model_v1_0.25_ori.h5 --video edge/video/30.mp4
```

When the pipeline works, the edge log shows `WebRTC 連線建立完成` followed by one `Result: ... latency=…ms` line per frame. Press Ctrl+C to stop all components.

## Real deployment

Each node installs only what it needs:

| Node | Requirements |
| --- | --- |
| Raspberry Pi | `requirements/edge.txt` |
| Dispatcher (EC2) | `requirements/dispatcher.txt` |
| Signaling (EC2) | `requirements/signaling.txt` |
| Inference (GPU PC) | `requirements/inference.txt` |

Nodes connect over a Tailscale (WireGuard) VPN, which also allows the Raspberry Pi to join from a 5G network. TURN server setup: `docs/COTURN_SETUP.md`.

**Deployment note:** if the connection is established but the dispatcher never receives frames, check whether traffic is going over IPv6. Disabling IPv6 on the Raspberry Pi forces the Tailscale IPv4 path.

## Repository structure

```
edge/          capture, preprocessing, WebRTC client, pump control
dispatcher/    WebRTC answerer + relay to inference
signaling/     WebSocket signaling server
inference/     inference server and model backends
shared/        config loader and frame protocol
config/        local test configs
scripts/       ONNX conversion, deployment and metrics scripts
quick_test/    tests with your own videos
```
config/        local test configs
scripts/       ONNX conversion, deployment and metrics scripts
quick_test/    tests with your own videos
