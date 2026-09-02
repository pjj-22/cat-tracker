# cattrack C++ core

Native reimplementation of the pipeline, parity-tested against the Python
originals in `cat_tracker/`. Builds as a static lib plus a pybind11 module
(for the parity suite), and, with ONNX Runtime available, a standalone
`cattrack` binary.

## Status

| Component | C++ | Bound | Parity test |
|-----------|-----|-------|-------------|
| `BBoxKalmanFilter` (`kalman.cpp`) | yes | yes | yes |
| Hungarian assignment (`hungarian.cpp`) | yes | yes | yes |
| IoU / center distance (`geom.cpp`) | yes | yes | yes |
| `Track` lifecycle (`track.cpp`) | yes | yes | yes |
| `MultiTracker` cycle (`multi_tracker.cpp`) | yes | yes | yes |
| YOLO output parse + NMS (`detection.cpp`) | yes | yes | yes |
| `config.yaml` reader (`config.cpp`) | yes | yes | yes |
| Bilinear resize / CHW (`preprocess.cpp`) | yes | yes | yes |
| Frame source, rpicam-vid pipe (`camera.cpp`) | yes | - | - |
| ONNX inference (`inference.cpp`) | yes | - | needs ORT to build |
| Standalone binary (`main.cpp`) | yes | - | needs ORT to build |
| Frame stream + servo (via `stream_bridge.py`) | yes | - | - |
| Identification, record, motion compensation | no | - | - |

Coat identification (HSV histograms) stays in Python; the C++ `Track` only
carries the resulting `name` / `_candidate_*` fields for deduplication. The
standalone binary has no servo/ID/stream yet.

## Layout

```
include/cattrack/   public headers
src/                library sources (no Python dependency)
bindings/           pybind11 module (cattrack_cpp)
bench_*.py          Python vs C++ microbenchmarks
```

## Build

```bash
pip install pybind11                      # build-time only (apt: pybind11-dev python3-pybind11)
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

Produces `cpp/build/libcattrack.a` and `cpp/build/cattrack_cpp*.so`.
`../tests/test_cpp_parity.py` adds `cpp/build` to `sys.path` automatically and
skips if the module is not built.

### Standalone binary

Needs ONNX Runtime C++. Download the matching release (headers + lib) and
point CMake at it:

```bash
ORT_VER=1.20.1
curl -L -o /tmp/ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-aarch64-$ORT_VER.tgz
mkdir -p ~/onnxruntime && tar xzf /tmp/ort.tgz -C ~/onnxruntime --strip-components=1

cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release \
  -DCATTRACK_WITH_ORT=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime
cmake --build cpp/build -j

./cpp/build/cattrack --config config.yaml            # live camera
./cpp/build/cattrack --source frames.yuv420          # a recorded raw stream
```

Emits `frame,id,cx,cy,w,h` CSV on stdout, fps on stderr.

### Streaming (C++ core, Python UI)

`--emit-frames` makes the binary draw track boxes and write, per frame, a
header line (`<frame> <n> <id,cx,cy,w,h> ...`) plus the annotated RGB frame.
`stream_bridge.py` pipes the frames into the existing MJPEG server + web UI and
runs the pan/tilt servo (`ServoController`) against the track positions:

```bash
python3 cpp/stream_bridge.py --port 5000            # servo + stream
python3 cpp/stream_bridge.py --port 5000 --no-servo
```

Servo mode / manual pan-tilt / center work from the web UI. Record, debug
overlay, identification, and camera-motion compensation aren't wired through.

## Benchmarks

```bash
python3 cpp/bench_tracker.py --frames 5000 --cats 4    # ~45x on ARM
python3 cpp/bench_detection.py --iters 300             # ~30x
```

`bench_tracker` times `MultiTracker.update()` in isolation; end to end that
gain is small because YOLO inference dominates. `bench_detection` times the
post-inference parse + NMS, which the pipeline profile showed as the largest
cost after inference itself (~12 ms/frame in Python), so that one does move the
frame budget.
