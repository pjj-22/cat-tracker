"""
Train cat color profiles for individual cat recognition.

Usage:
1. Run this script - it will auto-detect and track cats
2. When you see a cat labeled "Cat #1", "Cat #2", etc.
3. Press the number key (1, 2, 3...) to label that track
4. Enter the cat's name when prompted
5. Script automatically collects 15 samples from that track
6. Press 'q' to quit and save profiles
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time
from cat_tracker.multi_tracker import MultiTracker
from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier
from cat_tracker.utils import bbox_to_pixel_xyxy

def parse_yolo_output(output, conf_threshold=0.2, iou_threshold=0.4):
    output = output[0].T
    boxes, scores, class_ids = [], [], []

    for detection in output:
        box = detection[:4]
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if class_id == 15 and confidence > conf_threshold:
            boxes.append(box)
            scores.append(float(confidence))
            class_ids.append(class_id)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores, conf_threshold, iou_threshold)

    if len(indices) > 0:
        indices = indices.flatten()

    detections = []
    for i in indices:
        detections.append({
            'box': boxes[i],
            'confidence': scores[i]
        })

    return detections

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]

print("Loading model...")
session = ort.InferenceSession("yolov8s.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
model_h, model_w = input_shape[2], input_shape[3]

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

tracker = MultiTracker(max_missed=15, min_hits=3, iou_threshold=0.3)
extractor = ColorHistogramExtractor()
identifier = ColorHistogramIdentifier()

track_labels = {}
sample_counts = {}
target_samples = 15

print("\nInstructions:")
print("- Cats will be auto-detected and tracked")
print("- Press number key (1-9) to label that track ID")
print("- Script collects 15 samples automatically")
print("- Press 'q' to quit and save profiles")
print()

fps_start = time.time()
fps_count = 0
current_fps = 0.0

try:
    while True:
        frame = picam2.capture_array()
        orig_h, orig_w = frame.shape[:2]

        resized = cv2.resize(frame, (model_w, model_h))
        input_data = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        outputs = session.run(None, {input_name: input_data})[0]
        detections = parse_yolo_output(outputs)

        confirmed_tracks = tracker.update(detections)

        for track in confirmed_tracks:
            x1, y1, x2, y2 = bbox_to_pixel_xyxy(track.bbox, model_w, model_h, orig_w, orig_h)

            color = COLORS[(track.id - 1) % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if track.id in track_labels:
                cat_name = track_labels[track.id]
                count = sample_counts.get(cat_name, 0)
                label = f"{cat_name} #{track.id} ({count}/{target_samples})"

                if count < target_samples:
                    hist_h, hist_s, hist_v = extractor.extract(frame, (x1, y1, x2, y2))
                    if hist_h is not None:
                        identifier.add_training_sample(cat_name, hist_h, hist_s, hist_v)
                        sample_counts[cat_name] = identifier.profiles[cat_name]['sample_count']
                        print(f"Collected sample {sample_counts[cat_name]}/{target_samples} for {cat_name}")
            else:
                label = f"Cat #{track.id} (press {track.id} to label)"

            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps_count += 1
        if fps_count >= 30:
            current_fps = fps_count / (time.time() - fps_start)
            fps_start = time.time()
            fps_count = 0

        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Training", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('9'):
            track_id = key - ord('0')
            if any(t.id == track_id for t in confirmed_tracks):
                if track_id not in track_labels:
                    cat_name = input(f"\nEnter name for Cat #{track_id}: ").strip()
                    if cat_name:
                        track_labels[track_id] = cat_name
                        sample_counts[cat_name] = 0
                        print(f"Labeled track #{track_id} as '{cat_name}', collecting samples...")
                else:
                    print(f"Track #{track_id} already labeled as '{track_labels[track_id]}'")
            else:
                print(f"No active track #{track_id}")

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

    if identifier.profiles:
        identifier.save_profiles()
        print(f"\nSaved profiles for {len(identifier.profiles)} cats:")
        for cat_name, profile in identifier.profiles.items():
            print(f"  - {cat_name}: {profile['sample_count']} samples")
    else:
        print("\nNo profiles saved")

print("Done!")
