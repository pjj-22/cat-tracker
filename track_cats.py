"""
Multi-cat tracking with Kalman filters and Hungarian algorithm.
"""

from picamera2 import Picamera2
import cv2
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


def draw_track(frame, track, model_w, model_h, debug=False, is_tentative=False):
    orig_h, orig_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_to_pixel_xyxy(
        track.bbox, model_w, model_h, orig_w, orig_h
    )

    color = TRACK_COLORS[(track.id - 1) % len(TRACK_COLORS)]

    if is_tentative:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
        label = f"Track #{track.id} (tent)"
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{track.name} #{track.id}"
            if track.name != "Unknown"
            else f"Cat #{track.id}"
        )

    if debug:
        label += f" H:{track.hits} M:{track.missed_frames} C:{track.confidence:.2f}"

    (w, h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )

    bg = color if not is_tentative else (128, 128, 128)
    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), bg, -1)

    cv2.putText(
        frame, label, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )


def start_recording(fps, frame_size):
    os.makedirs("demos", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"demos/phase2_tracking_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    print(f"[REC] Started recording → {path}")
    return writer, path


def stop_recording(writer, path, frames):
    writer.release()
    print(f"[REC] Saved {frames} frames → {path}")


def main(debug=True, record=False, fps=20.0):
    print("Loading ONNX model...")
    session, input_name, model_h, model_w = load_yolo_model()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    tracker = MultiTracker(max_missed=15, min_hits=3, iou_threshold=0.3)
    extractor = ColorHistogramExtractor()
    identifier = ColorHistogramIdentifier()

    out = None
    output_path = None
    written_frames = 0
    recording = record

    if recording:
        out, output_path = start_recording(fps, (640, 480))

    print("Hotkeys: [q] quit  |  [r] record on/off  |  [d] debug on/off")

    fps_start = time.time()
    fps_count = 0
    current_fps = 0.0

    window_title = "Cat Tracking"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame = picam2.capture_array()

            input_data = preprocess_frame(frame, model_w, model_h)
            outputs = session.run(None, {input_name: input_data})[0]
            detections = parse_yolo_output(outputs)

            confirmed_tracks = tracker.update(detections)

            orig_h, orig_w = frame.shape[:2]
            for track in confirmed_tracks:
                if track.name == "Unknown" or track.frames_since_identified >= 30:
                    x1, y1, x2, y2 = bbox_to_pixel_xyxy(
                        track.bbox, model_w, model_h, orig_w, orig_h
                    )
                    h, s, v = extractor.extract(frame, (x1, y1, x2, y2))
                    if h is not None:
                        track.name, track.name_confidence, _ = identifier.identify(h, s, v)
                        track.frames_since_identified = 0

            if debug:
                for track in tracker.tracks:
                    if not track.is_confirmed():
                        draw_track(frame, track, model_w, model_h, debug, True)

            for track in confirmed_tracks:
                draw_track(frame, track, model_w, model_h, debug)

            if debug:
                for det in detections:
                    x_c, y_c = det["box"][:2]
                    x = int(x_c / model_w * orig_w)
                    y = int(y_c / model_h * orig_h)
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

            fps_count += 1
            if fps_count >= 30:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0

            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(confirmed_tracks)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"REC: {'ON' if recording else 'OFF'}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if recording else (200, 200, 200), 2)

            if out is not None:
                out.write(frame)
                written_frames += 1

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("d"):
                debug = not debug
                print(f"[DEBUG] {'ON' if debug else 'OFF'}")

            elif key == ord("r"):
                recording = not recording
                if recording:
                    written_frames = 0
                    out, output_path = start_recording(fps, (640, 480))
                else:
                    stop_recording(out, output_path, written_frames)
                    out = None
                    output_path = None

    finally:
        if out is not None:
            stop_recording(out, output_path, written_frames)

        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live multi-cat tracker")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlays")
    parser.add_argument("--record", action="store_true", help="Start with recording enabled")
    parser.add_argument("--fps", type=float, default=20.0, help="Recording FPS")

    args = parser.parse_args()
    main(debug=args.debug, record=args.record, fps=args.fps)
