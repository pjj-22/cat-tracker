# Cat Tracker

A real-time cat detection and identification system for Raspberry Pi. It watches your cats, learns who's who based on fur color, and can even follow them with a pan/tilt camera mount.

I built this because I have two cats and wanted to know what they do when I'm not around. Turns out, mostly sleep in the same spots. But the journey to find that out was pretty interesting.

## What It Actually Does

1. **Detects cats** using YOLO11s running on ONNX (~10 FPS inference, ~15-18 FPS effective with frame skipping)
2. **Tracks them** across frames using Kalman filters so they don't lose their identity when they walk behind furniture
3. **Identifies which cat** is which using HSV color histograms of their fur
4. **Follows them** with servo-controlled pan/tilt if you have the hardware
5. **Logs where they go** so you can generate heatmaps and see their favorite spots

## Hardware

**Required:**
- Raspberry Pi 5 (Pi 4 works but slower)
- Pi Camera (I use Arducam IMX708, but any Pi-compatible camera works)

**Optional but fun:**
- [Arducam Pan Tilt Platform](https://www.arducam.com/arducam-pan-tilt-platform-for-raspberry-pi-camera-2-dof-bracket-kit-with-digital-servos-and-ptz-control-broad-b0283.html), 2-DOF bracket kit with digital servos and PTZ control board ([manual](https://blog.arducam.com/downloads/arducam-pan-tilt-kit-manual.pdf))
- PCA9685 servo controller (Adafruit or clone)

## Quick Start

```bash
# System packages
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv ffmpeg

# Python packages
pip3 install --break-system-packages numpy==1.24.2 onnxruntime opencv-python ultralytics onnx filterpy scipy

# Get the YOLO model (imgsz=320 is critical, 640 is 4x slower on Pi)
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', imgsz=320)"

# Run the tracker
python3 track_cats.py --debug
```


## Training Pipeline

```
capture.py  →  label.py  →  build_profiles.py  →  track_cats.py
   (film)       (name)          (learn)              (run)
```

The first three steps are one-time setup per cat. After that you just run `track_cats.py`.

### Step 1: Capture (`capture.py`)

Runs YOLO on the live camera and saves cropped images of every detected cat, organized by track.

```bash
python3 capture.py              # run until you press q
python3 capture.py --no-display # headless over SSH, stop with Ctrl-C
python3 capture.py --duration 300  # auto-stop after 5 minutes
```

- Loads `yolo11s.onnx`, opens the Pi camera at 640×480
- For each confirmed track, saves a crop every 10 frames to avoid near-duplicate images
- Crops come from the clean frame before any overlay is drawn

**Output:**

```
captures/
└── session_YYYYMMDD_HHMMSS/
    ├── metadata.json
    ├── track_001/
    │   ├── 00000.jpg
    │   ├── 00010.jpg
    │   └── ...
    └── track_002/
        └── ...
```

Track numbers reset to 1 each session. **Run multiple sessions** since different times of day and rooms matter. Lighting changes how HSV looks.

### Step 2: Label (`label.py`)

Opens each saved crop as a fullscreen image. You press a number key to assign it to a cat.

```bash
python3 label.py captures/session_YYYYMMDD_HHMMSS/
```

| Key | Action |
|-----|--------|
| `1`–`9` | Assign to cat #N (first use prompts for a name) |
| `s` | Skip |
| `d` | Delete the file from disk |
| `n` | Jump to next track |
| `q` | Save and quit |

Cat names are set the first time you press a number key. You can resume a session; existing labels are reloaded automatically.

#### Labeling remotely

`label.py` needs a display. If you're SSH'd in headlessly, label on your local machine instead:

```bash
./scripts/fetch_captures.sh                          # pull crops from Pi → local captures/
python3 label.py captures/session_YYYYMMDD_HHMMSS/  # label locally
./scripts/push_labels.sh                             # push labels.json back to Pi
```

> **Note: `scripts/`**
>
> Everything in `scripts/` is a utility you run occasionally, not part of the main pipeline. The shell scripts sync data over SSH; the Python scripts are hardware tests and analysis tools.
>
> **SSH sync scripts** (default to `patrickjamesdev@10.0.0.209`, override with `PI_HOST`):
>
> ```bash
> export PI_HOST="pi@192.168.1.42"   # add to ~/.bashrc to make permanent
> ./scripts/fetch_captures.sh
> ```
>
> | Script | Direction | What it syncs |
> |--------|-----------|---------------|
> | `fetch_captures.sh` | Pi → local | `captures/` (images for labeling) |
> | `push_labels.sh` | local → Pi | `captures/**/labels.json` only |
> | `fetch_recordings.sh` | Pi → local | `recordings/` (auto-record clips) |
>
> Requires `rsync` and SSH key-based auth. Run `ssh-copy-id $PI_HOST` once if you haven't already.
>
> **Python utilities** (run on the Pi, or locally for the analysis scripts):
>
> | Script | When to use |
> |--------|-------------|
> | `scripts/camera_test.py` | Verify the camera works, saves a still to `cat_test.jpg` |
> | `scripts/calibrate_camera.py` | One-time floor homography setup for position logging |
> | `scripts/servo_camera_test.py` | Manual servo + live feed test before running the tracker |
> | `scripts/test_profiles.py` | Check identification accuracy against labeled sessions |
> | `scripts/generate_heatmap.py` | Generate a heatmap PNG from `occupancy_log.csv` |
> | `scripts/analyze_zones.py` | Zone occupancy breakdown from `occupancy_log.csv` |
> | `scripts/visualize_profiles.py` | Plot the HSV histogram profiles for each cat |

**Output:** `captures/session_.../labels.json`

```json
{
  "cat_names": { "1": "Chai", "2": "Shadow" },
  "labels": {
    "track_001/00000.jpg": "1",
    "track_002/00000.jpg": "2"
  }
}
```

### Step 3: Build Profiles (`build_profiles.py`)

Reads labeled sessions, extracts HSV histograms from each image, and averages them into a profile per cat.

```bash
python3 build_profiles.py captures/session_*/
```

**What it does per image:**

1. Converts the crop to HSV
2. Masks out pixels with saturation < 20 or brightness < 20 (drops shadows, near-white, near-black)
3. Computes three normalized histograms on the remaining pixels:
   - Hue: 30 bins (0–180°)
   - Saturation: 32 bins (0–255)
   - Value: 32 bins (0–255)
4. Merges into the cat's running average: `(old * n + new) / (n + 1)`

Source paths are tracked so re-running the script on the same sessions is safe; duplicates are skipped.

**Profile separation** is printed at the end (Bhattacharyya distance on hue histograms):

| Distance | Meaning |
|----------|---------|
| < 0.2 | Too similar, will confuse frequently |
| 0.2–0.35 | Moderate, occasional confusion |
| > 0.35 | Well separated, reliable |

**Output:** `cat_profiles.json` in the project root

```json
{
  "Chai": {
    "hist_h": [0.02, 0.15, ...],
    "hist_s": [0.01, 0.03, ...],
    "hist_v": [0.02, 0.04, ...],
    "sample_count": 45
  }
}
```

### Now Run It

```bash
python3 track_cats.py --stream --inference-every 3
```

Stream opens at `http://raspberrypi.local:5000`. You should see cats identified by name instead of "Cat #1".

## Servo Control

If you have a pan/tilt mount, the tracker can automatically follow cats around the room.

### Hardware Setup

Wire your PCA9685 to the Pi's I2C pins:
- VCC → 3.3V or 5V (check your board)
- GND → Ground
- SDA → GPIO 2 (Pin 3)
- SCL → GPIO 3 (Pin 5)

Connect servos to channels 0 (pan) and 1 (tilt).

Enable I2C:
```bash
sudo raspi-config  # Interface Options → I2C → Enable
```

### Mounting Offset

My pan/tilt mount isn't perfectly centered when assembled - the "straight ahead" position is at pan=60°, tilt=90° instead of 90°/90°. Adjust these in `config.yaml`:

```yaml
servo:
  pan_center: 60    # your "straight ahead" pan angle
  tilt_center: 90   # your "straight ahead" tilt angle
```

### Controls

- `s`: Cycle servo modes (AUTO → MANUAL → OFF)
- `c`: Center servos
- Arrow keys / WASD: Manual pan/tilt (in MANUAL mode)
- `0-9`: Target specific cat by track ID (in AUTO mode)

In AUTO mode with no cats visible, the camera will slowly patrol back and forth looking for cats.

## Position Logging & Heatmaps

> **Note:** Position logging assumes a stationary camera. The homography calibration maps pixel coordinates to floor coordinates for one fixed camera angle. If the servo is active and panning/tilting, logged positions will be wrong. Run with `--no-servo` (or set `servo.enabled: false` in `config.yaml`) when collecting data for heatmaps.

### Calibrate the Camera

You need a homography that maps pixel coordinates to real floor positions. Run this once:

```bash
python3 scripts/calibrate_camera.py
```

Click 4+ points on the floor that you can measure, enter their real-world positions in meters. Saves a `calibration.json` that the heatmap and zone scripts pick up automatically.

### Log Positions

```bash
python3 track_cats.py --log-positions --no-servo
```

This writes timestamped positions to `occupancy_log.csv`. Let it run for a day or two to collect meaningful data.

### Generate Heatmaps

```bash
# All cats, last 24 hours
python3 scripts/generate_heatmap.py

# Just one cat
python3 scripts/generate_heatmap.py --cat Honey --hours 8
```

### Zone Analysis

Define zones in `zones.json`:
```json
{
  "Window Sill": {"x1": 2.0, "y1": 2.0, "x2": 3.0, "y2": 2.5, "color": "#FFD700"},
  "Food Bowl": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5, "color": "#FF6347"}
}
```

Then analyze:
```bash
python3 scripts/analyze_zones.py --hours 24
```

## How It Works

### Detection

YOLO11s runs on ONNX runtime, which is way faster than PyTorch on ARM. I started with PyTorch and got 3 FPS. ONNX gets to ~10 FPS raw inference; with `--inference-every 3` the effective tracking rate is ~15-18 FPS since Kalman handles the in-between frames (many of those frames are predicted, not detected). The model detects COCO class 15 (cat) with a confidence threshold of 0.15, lower than typical because I'd rather have false positives that get filtered by tracking than miss a cat.

### Tracking

Each detected cat gets a Kalman filter with an 8-dimensional state vector `[x, y, w, h, vx, vy, vw, vh]` (position, size, and their velocities). Per frame:

1. Predict where each existing track should be (Kalman forward step)
2. Match predictions to new detections via the Hungarian algorithm (minimizes total IoU cost)
3. Update matched tracks, create new ones for unmatched detections
4. Delete tracks missing for `max_missed` frames (default 15)

With `--inference-every 3`, YOLO only runs every 3rd frame. Kalman predicts forward on the other frames, which raises the effective frame rate (~10 → ~15-18 FPS) with minimal accuracy loss.

One thing I've noticed: the Kalman filter assumes constant velocity between updates, which holds up fine for slow movement but starts drifting when a cat changes direction quickly. With 2 unobserved frames between corrections, the predicted position can be far enough off that the Hungarian matching occasionally associates the wrong detection to a track, particularly when both cats are close together. It works well enough in practice but it's not invisible.

### Identification

For each confirmed track, the live crop is compared to `cat_profiles.json` using Bhattacharyya distance:

```
BC_h = sum(sqrt(obs_h * profile_h))
BC_s = sum(sqrt(obs_s * profile_s))
BC_v = sum(sqrt(obs_v * profile_v))

BC       = 0.7 * BC_h + 0.2 * BC_s + 0.1 * BC_v
distance = sqrt(1 - BC)    ← 0 = identical, 1 = completely different
```

Hue gets 70% of the weight because it's most stable under lighting changes. Value gets only 10% because a dim room shifts it dramatically. The cat with the lowest distance wins. Re-identification runs whenever a track is new or hasn't been checked in 30 frames.

### Servo Control

Auto-follow uses proportional control; the error between cat center and frame center drives a pan/tilt adjustment, capped at 5 degrees/frame with a 50px deadzone to prevent jitter. When no cats are visible, patrol mode sweeps at 0.6°/frame looking for them.

## Configuration (`config.yaml`)

All tunable values in one place; edit this rather than touching source files.

```yaml
camera:
  width: 640
  height: 480

detection:
  model_path: "yolo11s.onnx"
  confidence_threshold: 0.15   # low on purpose, tracking filters false positives
  iou_threshold: 0.4

tracking:
  max_missed: 15    # frames before a track is deleted
  min_hits: 3       # detections needed before a track is confirmed

identification:
  profile_path: "cat_profiles.json"
  hsv_weights: [0.7, 0.2, 0.1]   # hue/saturation/value
  min_saturation: 20
  min_value: 20

servo:
  enabled: true
  pan_channel: 0
  tilt_channel: 1
  pan_center: 60    # mount-specific, "straight ahead" for this hardware
  tilt_center: 90
```

## File Map

```
cat-tracker/
│
├── track_cats.py              main loop, live tracking
├── capture.py                 step 1: collect training images
├── label.py                   step 2: assign cats to images
├── build_profiles.py          step 3: build cat_profiles.json
│
├── config.yaml                all tunable config
├── cat_profiles.json          generated (trained HSV profiles)
├── yolo11s.onnx               YOLO model (not in git, must export)
│
├── cat_tracker/
│   ├── detection.py           YOLO loading + output parsing
│   ├── kalman_filter.py       Kalman filter (8D state: x,y,w,h + velocities)
│   ├── tracker.py             single track state machine
│   ├── multi_tracker.py       Hungarian algorithm, manages all tracks
│   ├── prefix_colors.py       HSV histogram extract + identify
│   ├── servo.py               pan/tilt servo control via adafruit_servokit
│   ├── stream_server.py       MJPEG stream + WebSocket browser UI
│   ├── spatial.py             position logger → occupancy_log.csv
│   ├── config.py              config.yaml loader + defaults
│   └── utils.py               bbox math, IoU, coordinate helpers
│
├── scripts/
│   ├── fetch_captures.sh      Pi → local: pull captures/ for labeling
│   ├── push_labels.sh         local → Pi: push labels.json back
│   ├── fetch_recordings.sh    Pi → local: pull recordings/ clips
│   │
│   ├── camera_test.py         sanity check, captures a still from the Pi camera
│   ├── calibrate_camera.py    floor homography calibration (run once)
│   ├── servo_camera_test.py   manual servo + camera test with live feed
│   │
│   ├── generate_heatmap.py    heatmap from occupancy_log.csv
│   ├── analyze_zones.py       zone occupancy stats from occupancy_log.csv
│   ├── visualize_profiles.py  plot HSV histogram profiles per cat
│   └── test_profiles.py       run identification against labeled sessions, report accuracy
│
└── captures/                  generated during capture sessions
    └── session_YYYYMMDD_HHMMSS/
        ├── metadata.json
        ├── labels.json
        └── track_NNN/
            └── NNNNN.jpg
```

## Why These Choices?

**YOLO11s instead of 11n or 11m?**
11n (nano) is faster but misses cats at distance. 11m (medium) is more accurate but the inference rate drops too low to be useful. 11s is the sweet spot, around 10 FPS raw and 15-18 FPS effective with frame skipping.

**Kalman filter instead of just IoU matching?**
Pure IoU matching fails when cats move fast or when there are brief occlusions. Kalman prediction bridges these gaps. Plus it smooths out detection jitter.

**Color histograms instead of a trained classifier?**
Simpler to implement, no GPU needed for training, works well enough for 2-4 cats with different colors. If I had 10 cats that all looked the same, I'd probably train a ResNet.

**HSV instead of RGB?**
Hue is more stable under lighting changes. An orange cat is still "orange" in dim light, even if the RGB values shift dramatically.

**Why servo patrol?**
Cats are sneaky. If the camera is pointed at the couch and they walk in through the door, you won't see them until they cross into view. A slow patrol sweep catches them faster.

## Troubleshooting

**"Cat #1" instead of names**
You haven't trained profiles yet. Run the capture → label → build pipeline.

**Wrong cat identified**
- Check profile separation; `build_profiles.py` prints Bhattacharyya distance between each pair. Under 0.35 means trouble.
- Capture more sessions in different lighting and rooms. Lighting shifts HSV significantly.
- In `label.py`, delete (`d`) blurry images and any crop where both cats overlap.
- If you have a near-white or grey cat, lower `min_saturation` to 10 in `config.yaml`; the default mask of 20 may be discarding too many valid pixels.

**Track keeps switching identity**
The tracker is probably losing the track during occlusions and creating a new one. Raise `max_missed` from 15 to 30–50 in `config.yaml`.

**Low FPS**
- Use `--inference-every 3` to run YOLO every 3rd frame; Kalman predicts in between
- Lower camera resolution in `config.yaml`
- Make sure you're using ONNX, not PyTorch

**Servo jitters**
- Increase `deadzone` in `config.yaml`
- Lower `max_step` for smoother movement
- Check your power supply; servos can draw significant current

## What's Next?

Things I'm considering:
- Activity recognition (sitting, walking, playing, eating)
- Multi-camera support for whole-house coverage
- Web dashboard for remote monitoring
- Push notifications when specific events happen
- Rewriting the core pipeline in C++ to see whether dropping Python overhead entirely would give meaningful gains on the Pi, or whether ONNX already soaks up most of the available performance

But honestly, knowing where my cats nap is already more than I expected to learn.

## License

MIT. Do whatever you want with it. If you build something cool, I'd love to hear about it.
