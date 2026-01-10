"""
Enhanced training script with debug image capture.

IMPROVEMENTS OVER ORIGINAL:
- Saves actual image crops when collecting samples
- Stores HSV histograms alongside images
- Creates debug directory structure for inspection
- Allows reviewing what samples contributed to each profile

Directory structure created:
    training_data/
        Chai/
            sample_001.jpg          (RGB crop)
            sample_001_hsv.jpg      (HSV visualization)
            sample_001_hist.json    (histogram values)
            sample_002.jpg
            ...
        OtherCat/
            sample_001.jpg
            ...
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import json
from cat_tracker.multi_tracker import MultiTracker
from cat_tracker.tracker import Track
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

def save_training_sample(frame_rgb, bbox, cat_name, sample_number, hist_h, hist_s, hist_v):
    """
    Save training sample with visualizations for debugging.
    
    Args:
        frame_rgb: Full frame (RGB)
        bbox: (x1, y1, x2, y2) in pixel coordinates
        cat_name: Name of the cat
        sample_number: Sample index
        hist_h, hist_s, hist_v: Computed histograms
    """
    # Create directory structure
    cat_dir = f"training_data/{cat_name}"
    os.makedirs(cat_dir, exist_ok=True)
    
    x1, y1, x2, y2 = bbox
    
    # Extract and save RGB crop
    crop_rgb = frame_rgb[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return
    
    rgb_path = f"{cat_dir}/sample_{sample_number:03d}.jpg"
    cv2.imwrite(rgb_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    
    # Convert crop to HSV and save visualization
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    
    # Create HSV visualization (side-by-side H, S, V channels)
    h, s, v = cv2.split(crop_hsv)
    
    # Normalize for visualization
    h_vis = cv2.applyColorMap((h * 255 // 180).astype(np.uint8), cv2.COLORMAP_HSV)
    s_vis = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v_vis = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    
    # Resize to reasonable size
    max_height = 200
    if h_vis.shape[0] > max_height:
        scale = max_height / h_vis.shape[0]
        new_width = int(h_vis.shape[1] * scale)
        h_vis = cv2.resize(h_vis, (new_width, max_height))
        s_vis = cv2.resize(s_vis, (new_width, max_height))
        v_vis = cv2.resize(v_vis, (new_width, max_height))
    
    # Combine into single image
    hsv_vis = np.hstack([h_vis, s_vis, v_vis])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(hsv_vis, "Hue", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(hsv_vis, "Saturation", (h_vis.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(hsv_vis, "Value", (h_vis.shape[1] * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    hsv_path = f"{cat_dir}/sample_{sample_number:03d}_hsv.jpg"
    cv2.imwrite(hsv_path, hsv_vis)
    
    # Save histogram data as JSON
    hist_data = {
        'sample_number': sample_number,
        'bbox': bbox,
        'timestamp': time.time(),
        'hist_h': hist_h.tolist(),
        'hist_s': hist_s.tolist(),
        'hist_v': hist_v.tolist(),
        'dominant_hue_bin': int(np.argmax(hist_h)),
        'dominant_sat_bin': int(np.argmax(hist_s)),
        'dominant_val_bin': int(np.argmax(hist_v)),
    }
    
    hist_path = f"{cat_dir}/sample_{sample_number:03d}_hist.json"
    with open(hist_path, 'w') as f:
        json.dump(hist_data, f, indent=2)
    
    print(f"  Saved: {rgb_path}, {hsv_path}, {hist_path}")

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

# Reset track counter for clean IDs
Track._next_id = 1

# Increase max_missed during training so tracks don't die
tracker = MultiTracker(max_missed=50, min_hits=3, iou_threshold=0.3)

extractor = ColorHistogramExtractor()
identifier = ColorHistogramIdentifier()

track_labels = {}
sample_counts = {}
target_samples = 20  # Collect more samples for robustness

# Create training_data directory
os.makedirs("training_data", exist_ok=True)

print("\n" + "="*70)
print("ENHANCED TRAINING MODE - Saves debug images")
print("="*70)
print("\nIMPROVEMENTS:")
print("- Saves RGB crops of each sample")
print("- Saves HSV channel visualizations")
print("- Saves histogram JSON for each sample")
print("- Collects 20 samples (was 15)")
print("\nInstructions:")
print("- Move cat around to different positions/angles")
print("- Press 'L' to label a track, then type track ID")
print("- Wait for 20 samples to be collected")
print("- Press 'q' to quit and save profiles")
print("\nSaved data location: training_data/<cat_name>/")
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
                
                # Check current sample count
                current_count = identifier.profiles.get(cat_name, {}).get('sample_count', 0)
                label = f"{cat_name} #{track.id} ({current_count}/{target_samples})"

                # Collect samples
                if current_count < target_samples:
                    hist_h, hist_s, hist_v = extractor.extract(frame, (x1, y1, x2, y2))
                    if hist_h is not None:
                        # Add to profile
                        identifier.add_training_sample(cat_name, hist_h, hist_s, hist_v)
                        new_count = identifier.profiles[cat_name]['sample_count']
                        
                        # Save debug images
                        print(f"\nCollecting sample {new_count}/{target_samples} for {cat_name}:")
                        save_training_sample(frame, (x1, y1, x2, y2), cat_name, 
                                           new_count, hist_h, hist_s, hist_v)
                
                # Check if done
                if current_count >= target_samples:
                    label += " ✓ COMPLETE"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green when done
            else:
                label = f"Cat #{track.id} (press 'L' to label)"

            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps_count += 1
        if fps_count >= 30:
            current_fps = fps_count / (time.time() - fps_start)
            fps_start = time.time()
            fps_count = 0

        # Status overlay
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Active tracks: {len(confirmed_tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'L' to label | 'Q' to quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MOVE CAT TO DIFFERENT POSITIONS!", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Training (with Debug Capture)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('l') or key == ord('L'):
            # Prompt for track ID
            if len(confirmed_tracks) > 0:
                print("\n" + "="*50)
                print("Active tracks:")
                for t in confirmed_tracks:
                    status = f"'{track_labels[t.id]}'" if t.id in track_labels else "unlabeled"
                    print(f"  Track #{t.id} ({status})")
                print("="*50)
                
                try:
                    track_id = int(input("Enter track ID to label: "))
                    
                    # Check if track exists
                    track_exists = any(t.id == track_id for t in confirmed_tracks)
                    if not track_exists:
                        print(f"❌ Track #{track_id} not found")
                        continue
                    
                    # Check if already labeled
                    if track_id in track_labels:
                        cat_name = track_labels[track_id]
                        current_count = identifier.profiles.get(cat_name, {}).get('sample_count', 0)
                        print(f"ℹ️  Track #{track_id} already labeled as '{cat_name}' ({current_count} samples)")
                        print("   New samples will be added to existing profile")
                        continue
                    
                    # Get cat name
                    cat_name = input(f"Enter name for Cat #{track_id}: ").strip()
                    if cat_name:
                        track_labels[track_id] = cat_name
                        print(f"✓ Labeled track #{track_id} as '{cat_name}'")
                        print(f"Collecting {target_samples} samples...")
                        print("IMPORTANT: Move the cat to different positions/angles!")
                    else:
                        print("❌ Empty name, cancelled")
                        
                except ValueError:
                    print("❌ Invalid input")
            else:
                print("❌ No active tracks to label")

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

    if identifier.profiles:
        identifier.save_profiles()
        print("\n" + "="*70)
        print(f"✓ Saved profiles for {len(identifier.profiles)} cats:")
        print("="*70)
        for cat_name, profile in identifier.profiles.items():
            print(f"\n{cat_name}:")
            print(f"  Samples: {profile['sample_count']}")
            print(f"  Debug images: training_data/{cat_name}/")
            
            if profile['sample_count'] < 15:
                print(f"  ⚠️  WARNING: Only {profile['sample_count']} samples (recommended: 20+)")
            elif profile['sample_count'] < 20:
                print(f"  ⚠️  Marginal sample count (recommend 20+)")
            else:
                print(f"  ✓ Good sample count")
    else:
        print("\n❌ No profiles saved")

print("\n" + "="*70)
print("Debug data saved in training_data/ directory")
print("Use inspect_training_data.py to review samples")
print("="*70)
print("\nDone!")