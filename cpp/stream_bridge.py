"""Serve the C++ tracker's annotated frames over the existing MJPEG UI, and
drive the pan/tilt servo from Python.

The C++ binary does capture -> inference -> tracking and writes, per frame,
a header line ("<frame> <n> <id,cx,cy,w,h> ...", pixel coords) followed by the
raw RGB frame. This reads both, pushes the frame to StreamServer, and runs the
existing ServoController against the track positions.

Camera-motion compensation (feeding servo deltas back into the tracker) is not
wired yet, so tracks wobble during a pan and recover on the next YOLO frame.

    python3 cpp/stream_bridge.py [--port 5000] [--inference-every 3] [--no-servo]
"""

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from cat_tracker.config import load_config
from cat_tracker.servo import ServoController
from cat_tracker.stream_server import StreamServer


def make_servo(cfg, disabled):
    s = cfg.get("servo", {})
    return ServoController(
        pan_channel=s.get("pan_channel", 0),
        tilt_channel=s.get("tilt_channel", 1),
        enabled=s.get("enabled", True) and not disabled,
        pan_center=s.get("pan_center", 60),
        tilt_center=s.get("tilt_center", 90),
    )


def handle_command(servo, cmd):
    action = cmd.get("cmd")
    if action == "servo_mode":
        servo.toggle_mode()
    elif action == "center":
        servo.center()
    elif action == "pan_left":
        servo.manual_pan_left()
    elif action == "pan_right":
        servo.manual_pan_right()
    elif action == "tilt_up":
        servo.manual_tilt_up()
    elif action == "tilt_down":
        servo.manual_tilt_down()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--config", default=str(_ROOT / "config.yaml"))
    ap.add_argument("--binary", default=str(_ROOT / "cpp" / "build" / "cattrack"))
    ap.add_argument("--inference-every", type=int, default=None)
    ap.add_argument("--no-servo", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    w = cfg["camera"]["width"]
    h = cfg["camera"]["height"]
    frame_bytes = w * h * 3

    cmd = [args.binary, "--config", args.config, "--emit-frames"]
    if args.inference_every is not None:
        cmd += ["--inference-every", str(args.inference_every)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    servo = make_servo(cfg, args.no_servo)
    server = StreamServer(port=args.port)

    n, t0 = 0, time.time()
    fps = 0.0
    try:
        while True:
            header = proc.stdout.readline()
            if not header:
                break
            parts = header.decode().split()
            n_tracks = int(parts[1]) if len(parts) > 1 else 0
            tracks = [tuple(map(int, tok.split(","))) for tok in parts[2:2 + n_tracks]]

            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            server.push(rgb[:, :, ::-1])  # RGB -> BGR for cv2.imencode

            while (c := server.get_command()) is not None:
                handle_command(servo, c)

            if servo.mode == ServoController.MODE_AUTO:
                if tracks:
                    _, cx, cy, bw, bh = max(tracks, key=lambda t: t[3] * t[4])
                    servo.auto_follow(cx, cy, w, h)
                else:
                    servo.patrol()

            n += 1
            if n % 30 == 0:
                fps = 30 / (time.time() - t0)
                t0 = time.time()
                pan, tilt = servo.get_angles()
                server.update_status({
                    "fps": round(fps, 1),
                    "tracked": len(tracks),
                    "servo_mode": servo.get_mode_name(),
                    "pan": round(pan, 1) if pan is not None else None,
                    "tilt": round(tilt, 1) if tilt is not None else None,
                })
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        if servo.enabled:
            servo.center()

    print("[bridge] cattrack exited")


if __name__ == "__main__":
    main()
