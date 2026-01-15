"""
Multi-cat tracking with Kalman filters and Hungarian algorithm.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time
from datetime import datetime
import os
from cat_tracker.multi_tracker import MultiTracker
from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier
from cat_tracker.utils import bbox_to_pixel_xyxy
from cat_tracker.detection import load_yolo_model, parse_yolo_output, preprocess_frame, TRACK_COLORS


def draw_track(frame, track, model_w, model_h, debug=False, is_tentative=False):
    """Draw bounding box and ID for a track."""
    orig_h, orig_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_to_pixel_xyxy(track.bbox, model_w, model_h, orig_w, orig_h)

    # Get color for this track ID
    color = TRACK_COLORS[(track.id - 1) % len(TRACK_COLORS)]

    # Draw box (dashed if tentative)
    if is_tentative:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)
        label = f"Track #{track.id} (tent)"
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track.name != "Unknown":
            label = f"{track.name} #{track.id}"
        else:
            label = f"Cat #{track.id}"

    # Add debug info if enabled
    if debug:
        label += f" H:{track.hits} M:{track.missed_frames} C:{track.confidence:.2f}"

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Draw label background
    cv2.rectangle(frame,
                  (x1, y1 - label_size[1] - 10),
                  (x1 + label_size[0], y1),
                  color if not is_tentative else (128, 128, 128), -1)

    # Draw label text
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def main(debug=True, record=False):
    """
    Run live cat tracking with identification.

    Args:
        debug: Show debug overlays (tentative tracks, detection dots, extra stats)
        record: Save video to demos/ directory
    """
    print("Loading ONNX model...")
    session, input_name, model_h, model_w = load_yolo_model()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    tracker = MultiTracker(max_missed=15, min_hits=3, iou_threshold=0.3)

    extractor = ColorHistogramExtractor()
    identifier = ColorHistogramIdentifier()

    out = None
    output_filename = None
    if record:
        os.makedirs('demos', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"demos/phase2_tracking_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))
        print(f"Recording to {output_filename}")

    print("Press 'q' to stop")

    fps_start = time.time()
    fps_count = 0
    frame_count = 0
    current_fps = 0.0

    try:
        while True:
            frame = picam2.capture_array()

            # Preprocess and run detection
            input_data = preprocess_frame(frame, model_w, model_h)
            outputs = session.run(None, {input_name: input_data})[0]
            detections = parse_yolo_output(outputs)

            # Update tracker
            confirmed_tracks = tracker.update(detections)

            # Identify cats by color histogram
            orig_h, orig_w = frame.shape[:2]
            for track in confirmed_tracks:
                if track.name == "Unknown" or track.frames_since_identified >= 30:
                    x1, y1, x2, y2 = bbox_to_pixel_xyxy(track.bbox, model_w, model_h, orig_w, orig_h)
                    hist_h, hist_s, hist_v = extractor.extract(frame, (x1, y1, x2, y2))
                    if hist_h is not None:
                        track.name, track.name_confidence, _ = identifier.identify(hist_h, hist_s, hist_v)
                        track.frames_since_identified = 0

            if debug:
                # Draw tentative tracks
                for track in tracker.tracks:
                    if not track.is_confirmed():
                        draw_track(frame, track, model_w, model_h, debug=debug, is_tentative=True)

            # Draw confirmed tracks
            for track in confirmed_tracks:
                draw_track(frame, track, model_w, model_h, debug=debug, is_tentative=False)

            # Show raw detections as blue circles 
            if debug:
                for det in detections:
                    x_center, y_center = det['box'][:2]
                    orig_h, orig_w = frame.shape[:2]
                    x = int(x_center / model_w * orig_w)
                    y = int(y_center / model_h * orig_h)
                    cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

            # Calculate FPS
            fps_count += 1
            if fps_count >= 30:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0
                print(f"FPS: {current_fps:.1f} | Confirmed: {len(confirmed_tracks)} | Total: {len(tracker.tracks)} | Dets: {len(detections)}")

            # Add overlays
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(confirmed_tracks)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if debug:
                cv2.putText(frame, f"Total Tracks: {len(tracker.tracks)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Detections: {len(detections)}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            frame_count += 1

            window_title = "Cat Tracking (DEBUG)" if debug else "Cat Tracking"
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print(f"\nRecorded {frame_count} frames to {output_filename}")
        out.release()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(debug=True)