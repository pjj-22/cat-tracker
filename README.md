cat > README.md << 'EOF'
# Cat Guardian Robot

Real-time cat detection on Raspberry Pi 5. Currently getting 15-19 FPS.

## What This Does

Points a camera at cats and draws boxes around them in real-time

## Hardware You Need

- Raspberry Pi 5
- Pi Camera (I'm using Arducam IMX708)

## Setup
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv ffmpeg
pip3 install --break-system-packages "numpy==1.24.2" onnxruntime opencv-python ultralytics onnx

# Get the detection model
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320)"

# Run it
python3 detect_cat.py
```


## What I'm Building

This is phase 1 of a 4-phase project:

1. Detect Cats
2. Track multiple cats and give them IDs
3. Recognize which specific cat is which
4. Log what they're doing (sitting, playing, etc.)

End goal: A robot that can watch and identify my cats.

## Why It's Fast

Started with PyTorch (3 FPS, too slow). Converted to ONNX format which runs way faster on Raspberry Pi's ARM processor (18 FPS). 


## Files

- `detect_cat.py` - Main script
- `camera_test.py` - Test if camera works
- `requirements.txt` - What to install

## What's Next

Phase 2: Track multiple cats across frames using Kalman filters.
EOF