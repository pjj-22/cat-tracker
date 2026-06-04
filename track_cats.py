"""
Multi-cat tracking with Kalman filters and Hungarian algorithm.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time
from datetime import datetime
import os
import argparse

from cat_tracker.multi_tracker import MultiTracker
from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier
from cat_tracker.utils import bbox_to_pixel_xyxy
from cat_tracker.detection import (
    load_yolo_model,
    parse_yolo_output,
    preprocess_frame,
    TRACK_COLORS,
)
from cat_tracker.spatial import PositionLogger
from cat_tracker.config import load_config
from cat_tracker.servo import ServoController
from cat_tracker.stream_server import StreamServer, FLASK_AVAILABLE


def draw_track(frame, track, model_w, model_h, name_to_id, debug=False, is_tentative=False, is_target=False):
    orig_h, orig_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_to_pixel_xyxy(track.bbox, model_w, model_h, orig_w, orig_h)

    display_id = name_to_id.get(track.name)

    if is_tentative:
        color = (128, 128, 128)
        label = "?"
    else:
        color = TRACK_COLORS[(display_id - 1) % len(TRACK_COLORS)] if display_id else (128, 128, 128)
        if display_id:
            label = f"{track.name} #{display_id}"
        else:
            label = "?"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if is_tentative:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    if debug:
        label += f" H:{track.hits} M:{track.missed_frames} C:{track.confidence:.2f}"

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    bg = color if not is_tentative else (128, 128, 128)
    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), bg, -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if is_target:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)


def start_recording(fps, frame_size, directory):
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{directory}/{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    print(f"[REC] Started → {path}")
    return writer, path


def stop_recording(writer, path, frames):
    writer.release()
    print(f"[REC] Saved {frames} frames → {path}")


def main(debug=True, record=False, fps=None, log_positions=False, no_servo=False,
         config_path=None, stream=False, stream_port=5000, inference_every=None,
         auto_record=False, auto_record_grace=30):
    cfg = load_config(config_path)
    det_cfg = cfg['detection']
    trk_cfg = cfg['tracking']
    id_cfg  = cfg['identification']
    cam_cfg = cfg['camera']
    srv_cfg = cfg['servo']
    log_cfg = cfg['logging']

    inference_every = inference_every if inference_every is not None else trk_cfg.get('inference_every', 3)
    fps = fps if fps is not None else cam_cfg['fps']
    effective_max_missed = max(1, trk_cfg['max_missed'] // inference_every)

    print("Loading ONNX model...")
    session, input_name, model_h, model_w = load_yolo_model(det_cfg['model_path'])
    px_per_deg_h = model_w / srv_cfg.get('camera_hfov', 66)
    px_per_deg_v = model_h / srv_cfg.get('camera_vfov', 49)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (cam_cfg['width'], cam_cfg['height']), "format": "RGB888"},
        controls={"FrameRate": float(cam_cfg['fps'])},
    ))
    picam2.start()
    time.sleep(2)

    tracker = MultiTracker(
        max_missed=effective_max_missed,
        min_hits=trk_cfg['min_hits'],
        iou_threshold=trk_cfg['iou_threshold'],
        model_w=model_w,
        model_h=model_h,
    )
    extractor = ColorHistogramExtractor(
        bins_h=id_cfg['bins_h'], bins_s=id_cfg['bins_s'], bins_v=id_cfg['bins_v'],
        min_saturation=id_cfg['min_saturation'], min_value=id_cfg['min_value'],
    )
    identifier = ColorHistogramIdentifier(
        profile_path=id_cfg['profile_path'], hsv_weights=id_cfg['hsv_weights'],
    )
    cat_list = list(identifier.profiles.keys())
    name_to_id = {name: i + 1 for i, name in enumerate(cat_list)}
    servo_ctrl = ServoController(
        pan_channel=srv_cfg['pan_channel'], tilt_channel=srv_cfg['tilt_channel'],
        enabled=srv_cfg['enabled'] and not no_servo,
        pan_center=srv_cfg['pan_center'], tilt_center=srv_cfg['tilt_center'],
    )

    pos_logger = None
    log_count = 0
    if log_positions:
        pos_logger = PositionLogger(log_cfg['position_log_path'])
        print(f"[LOG] Position logging enabled → {log_cfg['position_log_path']}")

    out = None
    output_path = None
    written_frames = 0
    recording = record
    target_cat_name = None
    auto_out = None
    auto_output_path = None
    auto_written_frames = 0
    last_cat_seen = None

    if recording:
        out, output_path = start_recording(fps, (cam_cfg['width'], cam_cfg['height']), "demos")

    stream_server = None
    if stream:
        if not FLASK_AVAILABLE:
            print("[STREAM] flask not installed; run: sudo apt install python3-flask python3-flask-sock")
            stream = False
        else:
            stream_server = StreamServer(port=stream_port)

    print("\nHotkeys:")
    if not stream:
        print("  [q] quit  |  [r] record on/off  |  [d] debug on/off")
    else:
        print("  Ctrl-C to quit  (controls via browser)")
    if servo_ctrl.enabled:
        print("  [s] servo mode  |  [c] center servos  |  [arrows] manual control")
        print("  [0-9] target specific cat (AUTO mode)")

    fps_start = time.time()
    fps_count = 0
    current_fps = 0.0
    frame_count = 0
    detections = []

    if not stream:
        cv2.namedWindow("Cat Tracking", cv2.WINDOW_AUTOSIZE)

    def handle_command(cmd_dict):
        nonlocal debug, recording, written_frames, out, output_path, target_cat_name
        action = cmd_dict.get("cmd")
        if action == "toggle_record":
            recording = not recording
            if recording:
                written_frames = 0
                out, output_path = start_recording(fps, (cam_cfg['width'], cam_cfg['height']))
            else:
                stop_recording(out, output_path, written_frames)
                out = None
                output_path = None
        elif action == "toggle_debug":
            debug = not debug
            print(f"[DEBUG] {'ON' if debug else 'OFF'}")
        elif action == "servo_mode":
            servo_ctrl.toggle_mode()
        elif action == "center":
            servo_ctrl.center()
            print("[SERVO] Centered")
        elif action == "target":
            name = cmd_dict.get("name")
            if not name:
                target_cat_name = None
                print("[SERVO] Tracking any cat")
            else:
                target_cat_name = name
                print(f"[SERVO] Targeting {name}")
        elif action in ("pan_left", "pan_right", "tilt_up", "tilt_down"):
            if servo_ctrl.mode != ServoController.MODE_MANUAL:
                servo_ctrl.mode = ServoController.MODE_MANUAL
                print("[SERVO] Mode: MANUAL")
            pan_d, tilt_d = 0.0, 0.0
            if action == "pan_left":
                pan_d = servo_ctrl.manual_pan_left()
            elif action == "pan_right":
                pan_d = servo_ctrl.manual_pan_right()
            elif action == "tilt_up":
                tilt_d = servo_ctrl.manual_tilt_up()
            elif action == "tilt_down":
                tilt_d = servo_ctrl.manual_tilt_down()
            if pan_d or tilt_d:
                tracker.compensate_camera_motion(pan_d * px_per_deg_h, tilt_d * px_per_deg_v)

    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1

            if frame_count % inference_every == 0:
                input_data = preprocess_frame(frame, model_w, model_h)
                outputs = session.run(None, {input_name: input_data})[0]
                detections = parse_yolo_output(
                    outputs,
                    conf_threshold=det_cfg['confidence_threshold'],
                    iou_threshold=det_cfg['iou_threshold'],
                )
                confirmed_tracks = tracker.update(detections)
            else:
                confirmed_tracks = tracker.predict_only()

            orig_h, orig_w = frame.shape[:2]
            to_identify = []
            for track in confirmed_tracks:
                if track.name == "Unknown" or track.frames_since_identified >= 30:
                    x1, y1, x2, y2 = bbox_to_pixel_xyxy(
                        track.bbox, model_w, model_h, orig_w, orig_h
                    )
                    h, s, v = extractor.extract(frame, (x1, y1, x2, y2))
                    if h is not None:
                        to_identify.append((track, h, s, v))

            if to_identify:
                histograms = [(h, s, v) for _, h, s, v in to_identify]
                reidentifying = {id(track) for track, *_ in to_identify}
                already_claimed = {
                    t.name for t in confirmed_tracks
                    if t.name != "Unknown" and id(t) not in reidentifying
                }
                assignments = identifier.identify_exclusive(histograms, exclude=already_claimed)
                for (track, _, _, _), (name, conf) in zip(to_identify, assignments):
                    track.name = name
                    track.name_confidence = conf
                    track.frames_since_identified = 0

            if auto_record:
                if confirmed_tracks:
                    last_cat_seen = time.time()
                    if auto_out is None:
                        auto_out, auto_output_path = start_recording(fps, (cam_cfg['width'], cam_cfg['height']), "recordings")
                        auto_written_frames = 0
                elif auto_out is not None and last_cat_seen is not None:
                    if time.time() - last_cat_seen >= auto_record_grace:
                        stop_recording(auto_out, auto_output_path, auto_written_frames)
                        auto_out = None
                        auto_output_path = None
                        last_cat_seen = None

            target_track = None
            if servo_ctrl.mode == ServoController.MODE_AUTO:
                if confirmed_tracks:
                    if target_cat_name:
                        target_track = next(
                            (t for t in confirmed_tracks if t.name == target_cat_name), None
                        ) or confirmed_tracks[0]
                    else:
                        target_track = confirmed_tracks[0]
                    vx, vy = target_track.velocity
                    look_ahead = trk_cfg.get('look_ahead', 2)
                    x_pred = float(np.clip(target_track.bbox[0] + vx * look_ahead, 0, model_w))
                    y_pred = float(np.clip(target_track.bbox[1] + vy * look_ahead, 0, model_h))
                    x_center = x_pred / model_w * orig_w
                    y_center = y_pred / model_h * orig_h
                    pan_d, tilt_d = servo_ctrl.auto_follow(x_center, y_center, orig_w, orig_h)
                    if pan_d or tilt_d:
                        tracker.compensate_camera_motion(pan_d * px_per_deg_h, tilt_d * px_per_deg_v)
                else:
                    servo_ctrl.patrol()

            if debug:
                for track in tracker.tracks:
                    if not track.is_confirmed():
                        draw_track(frame, track, model_w, model_h, name_to_id, debug, True)

            for track in confirmed_tracks:
                if pos_logger is not None:
                    x_center = track.bbox[0] / model_w * orig_w
                    y_center = track.bbox[1] / model_h * orig_h
                    width    = track.bbox[2] / model_w * orig_w
                    height   = track.bbox[3] / model_h * orig_h
                    pos_logger.log(track.name, x_center, y_center, width, height)
                    log_count += 1
                is_target = (target_track is not None and track.id == target_track.id)
                draw_track(frame, track, model_w, model_h, name_to_id, debug, is_target=is_target)

            if debug:
                for det in detections:
                    x_c, y_c = det["box"][:2]
                    cv2.circle(frame, (int(x_c / model_w * orig_w), int(y_c / model_h * orig_h)),
                               6, (255, 0, 0), -1)
                cx, cy = orig_w // 2, orig_h // 2
                cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

            fps_count += 1
            if fps_count >= 30:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0

            pan, tilt = servo_ctrl.get_angles()

            status_lines = [
                f"FPS: {current_fps:.1f}",
                f"Tracked: {len(confirmed_tracks)}",
                f"REC: {'ON' if recording else 'OFF'}",
            ]
            if servo_ctrl.enabled:
                status_lines.append(f"Servo: {servo_ctrl.get_mode_name()}")
                if debug and pan is not None:
                    status_lines.append(f"Pan: {pan:.0f}° Tilt: {tilt:.0f}°")
                if target_track is not None:
                    status_lines.append(f"Target: {target_track.name}")
            if log_positions:
                status_lines.append(f"LOG: {log_count}")

            if not stream:
                for i, line in enumerate(status_lines):
                    color = (0, 0, 255) if (i == 2 and recording) else (255, 255, 255)
                    cv2.putText(frame, line, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if out is not None:
                out.write(frame)
                written_frames += 1
            if auto_out is not None:
                auto_out.write(frame)
                auto_written_frames += 1

            if stream:
                stream_server.push(frame)
                while True:
                    cmd = stream_server.get_command()
                    if cmd is None:
                        break
                    handle_command(cmd)
                stream_server.update_status({
                    "fps":        round(current_fps, 1),
                    "tracked":    len(confirmed_tracks),
                    "cats":       cat_list,
                    "active":     [t.name for t in confirmed_tracks if t.name != "Unknown"],
                    "recording":  recording,
                    "debug":      debug,
                    "servo_mode": servo_ctrl.get_mode_name(),
                    "pan":        round(pan, 1) if pan is not None else None,
                    "tilt":       round(tilt, 1) if tilt is not None else None,
                    "target":     target_cat_name or "",
                    "log":        log_count if log_positions else None,
                })
            else:
                cv2.imshow("Cat Tracking", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("d"):
                    handle_command({"cmd": "toggle_debug"})
                elif key == ord("r"):
                    handle_command({"cmd": "toggle_record"})
                elif key == ord("s"):
                    handle_command({"cmd": "servo_mode"})
                elif key == ord("c"):
                    handle_command({"cmd": "center"})
                elif key in (81, ord('a')):
                    handle_command({"cmd": "pan_left"})
                elif key == 83:
                    handle_command({"cmd": "pan_right"})
                elif key in (82, ord('w')):
                    handle_command({"cmd": "tilt_up"})
                elif key == 84:
                    handle_command({"cmd": "tilt_down"})
                elif ord('0') <= key <= ord('9') and servo_ctrl.mode == ServoController.MODE_AUTO:
                    if key == ord('0'):
                        handle_command({"cmd": "target", "name": None})
                    else:
                        idx = int(chr(key)) - 1
                        if 0 <= idx < len(cat_list):
                            handle_command({"cmd": "target", "name": cat_list[idx]})
                        else:
                            print(f"[SERVO] No cat #{int(chr(key))} in config")

    finally:
        if out is not None:
            stop_recording(out, output_path, written_frames)
        if auto_out is not None:
            stop_recording(auto_out, auto_output_path, auto_written_frames)
        if pos_logger is not None:
            pos_logger.close()
            print(f"[LOG] Logged {log_count} positions to occupancy_log.csv")
        if servo_ctrl.enabled:
            servo_ctrl.center()
            print("[SERVO] Centered and shutdown")
        picam2.stop()
        if not stream:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live multi-cat tracker")
    parser.add_argument("--debug",        action="store_true", help="Enable debug overlays")
    parser.add_argument("--record",       action="store_true", help="Start with recording enabled")
    parser.add_argument("--fps",          type=float, default=None, help="Override FPS (default: camera.fps from config)")
    parser.add_argument("--log-positions",action="store_true", help="Log pixel positions to occupancy_log.csv")
    parser.add_argument("--no-servo",     action="store_true", help="Disable servo control")
    parser.add_argument("--config",       default=None, help="Path to config.yaml")
    parser.add_argument("--stream",       action="store_true", help="Serve MJPEG stream to browser")
    parser.add_argument("--stream-port",     type=int, default=5000, help="Stream server port (default: 5000)")
    parser.add_argument("--inference-every", type=int, default=None,
                        help="Run YOLO every N frames; Kalman predicts in between (default: tracking.inference_every from config)")
    parser.add_argument("--auto-record",     action="store_true",
                        help="Auto-record to recordings/ when a cat is in frame")
    parser.add_argument("--auto-record-grace", type=int, default=30,
                        help="Seconds after last cat before stopping auto-record (default: 30)")

    args = parser.parse_args()
    main(
        debug=args.debug, record=args.record, fps=args.fps,
        log_positions=args.log_positions, no_servo=args.no_servo,
        config_path=args.config, stream=args.stream, stream_port=args.stream_port,
        inference_every=args.inference_every,
        auto_record=args.auto_record, auto_record_grace=args.auto_record_grace,
    )
