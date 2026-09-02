"""Microbenchmark: Python vs C++ tracking core, in isolation.

Times MultiTracker.update() over a synthetic detection stream (no camera, no
YOLO). This measures only the predict -> match -> update -> dedup cycle, which
is the part ported to C++.

    python3 cpp/bench_tracker.py [--frames 5000] [--cats 4]
"""

import argparse
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # repo root, for cat_tracker
if (_HERE / "build").is_dir():
    sys.path.insert(0, str(_HERE / "build"))   # built extension

import cattrack_cpp  # noqa: E402
from cat_tracker.multi_tracker import MultiTracker as PyMT  # noqa: E402


def make_stream(frames, cats, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([40, 40, 30, 30], [280, 280, 60, 60], size=(cats, 4))
    vel = rng.normal(0, 1.2, size=(cats, 2))
    stream = []
    for _ in range(frames):
        pos[:, :2] += vel + rng.normal(0, 0.3, size=(cats, 2))
        pos[:, :2] = np.clip(pos[:, :2], 20, 300)
        dets = [
            {"box": pos[i].copy() + rng.normal(0, 1.5, 4), "confidence": 0.9}
            for i in range(cats)
        ]
        stream.append(dets)
    return stream


def run(tracker, stream):
    t0 = time.perf_counter()
    for dets in stream:
        tracker.update(dets)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5000)
    ap.add_argument("--cats", type=int, default=4)
    args = ap.parse_args()

    stream = make_stream(args.frames, args.cats)

    # warm the branch predictors / allocator
    run(PyMT(), make_stream(200, args.cats))
    run(cattrack_cpp.MultiTracker(), make_stream(200, args.cats))

    py_s = run(PyMT(), stream)
    cpp_s = run(cattrack_cpp.MultiTracker(), stream)

    print(f"frames={args.frames}  cats={args.cats}")
    print(f"  python : {py_s * 1e3:8.1f} ms  ({py_s / args.frames * 1e6:6.1f} us/update)")
    print(f"  c++    : {cpp_s * 1e3:8.1f} ms  ({cpp_s / args.frames * 1e6:6.1f} us/update)")
    print(f"  speedup: {py_s / cpp_s:.1f}x")


if __name__ == "__main__":
    main()
