from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time

def parse_yolo_output(output, conf_threshold=0.5, iou_threshold=0.4):
    """Parse YOLOv8 ONNX output and apply NMS"""
    # Output shape: [1, 84, 2100] - (batch, [x,y,w,h + 80 classes], predictions)
    output = output[0].T  # Transpose to [2100, 84]
    
    boxes = []
    scores = []
    class_ids = []
    
    for detection in output:
        # First 4 values are box coords, rest are class scores
        box = detection[:4]
        class_scores = detection[4:]
        
        # Get class with highest score
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        # Only keep cats (class 15) with good confidence
        if class_id == 15 and confidence > conf_threshold:
            boxes.append(box)
            scores.append(float(confidence))
            class_ids.append(class_id)
    
    if len(boxes) == 0:
        return []
    
    # Apply NMS
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

print("Loading ONNX model...")
session = ort.InferenceSession("yolov8s.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
model_h, model_w = input_shape[2], input_shape[3]

# Initialize pi camera, would need changes for USB
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

print(f"Model input: {model_w}x{model_h}")
print("Detecting cats... Press 'q' to quit")

fps_start = time.time()
fps_count = 0

try:
    while True:
        frame = picam2.capture_array()
        orig_h, orig_w = frame.shape[:2]
        
        resized = cv2.resize(frame, (model_w, model_h))
        input_data = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        outputs = session.run(None, {input_name: input_data})[0]
        
        detections = parse_yolo_output(outputs)
        
        for det in detections:
            # Convert from YOLO format (center_x, center_y, w, h) to pixel coords
            x_center, y_center, w, h = det['box']
            
            x_center = x_center / model_w * orig_w
            y_center = y_center / model_h * orig_h
            w = w / model_w * orig_w
            h = h / model_h * orig_h
            
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Cat {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Cats: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        fps_count += 1
        if fps_count >= 30:
            fps = fps_count / (time.time() - fps_start)
            print(f"FPS: {fps:.1f} | Cats detected: {len(detections)}")
            fps_start = time.time()
            fps_count = 0
        
        cv2.imshow("Cat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()