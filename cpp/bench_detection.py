"""Microbenchmark: Python vs C++ YOLO output parsing + NMS.

This is the post-inference decode step (argmax over every box's class scores,
then NMS). In the cProfile of the live pipeline it was the largest cost after
ONNX inference itself.

    python3 cpp/bench_detection.py [--iters 300]
"""

import argparse
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
if (_HERE / "build").is_dir():
    sys.path.insert(0, str(_HERE / "build"))

import cattrack_cpp  # noqa: E402
from cat_tracker import detection as py_detection  # noqa: E402


def make_output(rng, n_boxes=8400, n_classes=80):
    """YOLO11 imgsz=320 output shape, with ~6 planted cat boxes."""
    out = rng.uniform(0.0, 0.08, size=(1, 4 + n_classes, n_boxes)).astype(np.float32)
    out[0, :4, :] = rng.uniform(20, 300, size=(4, n_boxes)).astype(np.float32)
    for col in rng.choice(n_boxes, size=6, replace=False):
        out[0, 4 + py_detection.CAT_CLASS_ID, col] = rng.uniform(0.4, 0.95)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    outputs = [make_output(rng) for _ in range(args.iters)]

    py_detection.parse_yolo_output(outputs[0])          # warm
    cattrack_cpp.parse_yolo_output(outputs[0])

    t0 = time.perf_counter()
    for o in outputs:
        py_detection.parse_yolo_output(o, 0.15, 0.4)
    py_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for o in outputs:
        cattrack_cpp.parse_yolo_output(o, 0.15, 0.4)
    cpp_s = time.perf_counter() - t0

    print(f"iters={args.iters}  (output 1x84x8400)")
    print(f"  python : {py_s * 1e3:8.1f} ms  ({py_s / args.iters * 1e3:6.2f} ms/call)")
    print(f"  c++    : {cpp_s * 1e3:8.1f} ms  ({cpp_s / args.iters * 1e3:6.2f} ms/call)")
    print(f"  speedup: {py_s / cpp_s:.1f}x")


if __name__ == "__main__":
    main()
