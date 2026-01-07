"""
Phase 2: Multi-cat tracking with Kalman filters and Hungarian algorithm.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time
from datetime import datetime
import os
from cat_tracker.multi_tracker import MultiTracker

DEBUG = True  # Set to False once working


def parse_yolo_output(output, conf_threshold=0.2, iou_threshold=0.4):
    """Parse YOLOv8 ONNX output."""
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
    
    detections = []
    for i in indices:
        detections.append({
            'box': boxes[i],
            'confidence': scores[i]
        })
    
    return detections


# Colors for different track IDs (up to 10 cats)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (128, 255, 0),  # Lime
]


def draw_track(frame, track, model_w, model_h, is_tentative=False):
    """Draw bounding box and ID for a track."""
    orig_h, orig_w = frame.shape[:2]
    
    # Convert bbox to pixel coordinates
    x_center, y_center, w, h = track.bbox
    x_center = x_center / model_w * orig_w
    y_center = y_center / model_h * orig_h
    w = w / model_w * orig_w
    h = h / model_h * orig_h
    
    x1 = int(x_center - w/2)
    y1 = int(y_center - h/2)
    x2 = int(x_center + w/2)
    y2 = int(y_center + h/2)
    
    # Get color for this track ID
    color = COLORS[(track.id - 1) % len(COLORS)]
    
    # Draw box (dashed if tentative)
    if is_tentative:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)
        label = f"Track #{track.id} (tent)"
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Cat #{track.id}"
    
    # Add debug info if enabled
    if DEBUG:
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


# Create demos directory
os.makedirs('demos', exist_ok=True)

# Load ONNX model
print("Loading ONNX model...")
session = ort.InferenceSession("yolov8s.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
model_h, model_w = input_shape[2], input_shape[3]

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Initialize tracker (more lenient settings)
tracker = MultiTracker(max_missed=15, min_hits=3, iou_threshold=0.3)
# Setup video writer
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
        
        # Preprocess for detection
        resized = cv2.resize(frame, (model_w, model_h))
        input_data = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run detection
        outputs = session.run(None, {input_name: input_data})[0]
        detections = parse_yolo_output(outputs)
        
        # Update tracker
        confirmed_tracks = tracker.update(detections)
        
        # Draw ALL tracks if debugging
        if DEBUG:
            # Draw tentative tracks (gray)
            for track in tracker.tracks:
                if not track.is_confirmed():
                    draw_track(frame, track, model_w, model_h, is_tentative=True)
        
        # Draw confirmed tracks
        for track in confirmed_tracks:
            draw_track(frame, track, model_w, model_h, is_tentative=False)
        
        # Show raw detections as blue circles (for debugging)
        if DEBUG:
            for det in detections:
                x_center, y_center = det['box'][:2]
                orig_h, orig_w = frame.shape[:2]
                x = int(x_center / model_w * orig_w)
                y = int(y_center / model_h * orig_h)
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)  # Blue dot for detection
        
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
        
        if DEBUG:
            cv2.putText(frame, f"Total Tracks: {len(tracker.tracks)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Display
        cv2.imshow("Cat Tracking (DEBUG)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    print(f"\nRecorded {frame_count} frames to {output_filename}")
    out.release()
    picam2.stop()
    cv2.destroyAllWindows()