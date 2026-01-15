"""

Runs cat detection and automatically saves all detected cats to a session directory.

Output structure:
    captures/
        session_20260112_143022/
            track_001/
                00001.jpg
                00002.jpg
                ...
            track_002/
                00001.jpg
                ...
            metadata.json

Usage:
    python3 capture.py [--duration SECONDS]

Controls:
    q - Stop capture and save
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
import json
import argparse
from datetime import datetime
from cat_tracker.multi_tracker import MultiTracker
from cat_tracker.tracker import Track
from cat_tracker.utils import bbox_to_pixel_xyxy
from cat_tracker.detection import load_yolo_model, parse_yolo_output, preprocess_frame, TRACK_COLORS

def main(duration=None):
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = f"captures/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    print("="*70)
    print("AUTO-CAPTURE MODE")
    print("="*70)
    print(f"\nSession: {session_id}")
    print(f"Output: {session_dir}/")
    if duration:
        print(f"Duration: {duration} seconds")
    print("\nControls:")
    print("  [q] - Stop and save")
    print("\nCapturing all detected cats automatically...")
    print("="*70)

    print("\nLoading YOLO model...")
    session_onnx, input_name, model_h, model_w = load_yolo_model()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    Track._next_id = 1
    tracker = MultiTracker(max_missed=30, min_hits=3, iou_threshold=0.3)

    track_frame_counts = {}
    track_last_save = {}
    save_interval = 10  # Save every 10 frames to avoid too many duplicates

    fps_start = time.time()
    fps_count = 0
    current_fps = 0.0
    start_time = time.time()

    print("Camera ready!\n")

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                print(f"\nDuration limit reached ({duration}s)")
                break

            frame = picam2.capture_array()
            orig_h, orig_w = frame.shape[:2]

            # YOLO detection
            input_data = preprocess_frame(frame, model_w, model_h)
            outputs = session_onnx.run(None, {input_name: input_data})[0]
            detections = parse_yolo_output(outputs)

            # Update tracker
            confirmed_tracks = tracker.update(detections)

            # Save frames for each track (use clean frame without overlays)
            for track in confirmed_tracks:
                track_id = track.id
                x1, y1, x2, y2 = bbox_to_pixel_xyxy(track.bbox, model_w, model_h, orig_w, orig_h)

                if track_id not in track_frame_counts:
                    track_frame_counts[track_id] = 0
                    track_last_save[track_id] = -save_interval
                    track_dir = f"{session_dir}/track_{track_id:03d}"
                    os.makedirs(track_dir, exist_ok=True)

                frames_since_save = track_frame_counts[track_id] - track_last_save[track_id]
                if frames_since_save >= save_interval:
                    # Extract crop from clean frame
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        track_dir = f"{session_dir}/track_{track_id:03d}"
                        frame_num = track_frame_counts[track_id]
                        filename = f"{track_dir}/{frame_num:05d}.jpg"
                        cv2.imwrite(filename, crop)
                        track_last_save[track_id] = track_frame_counts[track_id]

                track_frame_counts[track_id] += 1

            display_frame = frame.copy()
            for track in confirmed_tracks:
                track_id = track.id
                x1, y1, x2, y2 = bbox_to_pixel_xyxy(track.bbox, model_w, model_h, orig_w, orig_h)
                color = TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                saved_count = len([f for f in os.listdir(f"{session_dir}/track_{track_id:03d}") if f.endswith('.jpg')])
                cv2.putText(display_frame, f"Track #{track_id} ({saved_count} saved)", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            fps_count += 1
            if fps_count >= 30:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0

            elapsed = int(time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f} | Time: {elapsed}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Tracks: {len(track_frame_counts)} | Capturing...", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Auto-Capture - Press 'q' to stop", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nStopping capture...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

        # Save metadata
        metadata = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': int(time.time() - start_time),
            'total_tracks': len(track_frame_counts),
            'tracks': {}
        }

        for track_id, frame_count in track_frame_counts.items():
            track_dir = f"{session_dir}/track_{track_id:03d}"
            saved_count = len([f for f in os.listdir(track_dir) if f.endswith('.jpg')])
            metadata['tracks'][track_id] = {
                'frames_tracked': frame_count,
                'frames_saved': saved_count
            }

        metadata_path = f"{session_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*70)
        print("CAPTURE COMPLETE")
        print("="*70)
        print(f"\nSession: {session_dir}/")
        print(f"Duration: {int(time.time() - start_time)}s")
        print(f"Tracks captured: {len(track_frame_counts)}")
        print("\nTrack details:")
        for track_id in sorted(track_frame_counts.keys()):
            saved = metadata['tracks'][track_id]['frames_saved']
            print(f"  Track {track_id}: {saved} images")

        print("\n" + "="*70)
        print("NEXT STEP: Label the captures")
        print(f"Run: python3 label.py {session_dir}")
        print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-capture cat detections')
    parser.add_argument('--duration', type=int, help='Auto-stop after N seconds')
    args = parser.parse_args()

    main(duration=args.duration)
