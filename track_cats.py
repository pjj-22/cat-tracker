"""
Multi-cat tracking with Kalman filters and Hungarian algorithm.
Now with optional servo auto-follow!
"""

from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import os
import argparse
import sys

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

# Try to import servo control
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "example", "Jetson"))
    from ServoKit import ServoKit
    SERVO_AVAILABLE = True
except Exception:
    SERVO_AVAILABLE = False


class ServoController:
    """
    Proportional controller for pan/tilt servos.
    
    Modes:
    - AUTO: Automatically follows detected cats
    - MANUAL: Arrow key control
    - OFF: Servos disabled
    """
    
    MODE_OFF = 0
    MODE_MANUAL = 1
    MODE_AUTO = 2
    
    def __init__(self, pan_channel=0, tilt_channel=1, enabled=True,
                 pan_center=118, tilt_center=90):
        self.enabled = enabled and SERVO_AVAILABLE
        self.mode = self.MODE_AUTO if self.enabled else self.MODE_OFF

        if not self.enabled:
            return

        try:
            self.servo = ServoKit(num_ports=max(pan_channel, tilt_channel) + 1)
            self.pan_ch = pan_channel
            self.tilt_ch = tilt_channel

            self.deadzone = 50       # pixels from center to ignore
            self.max_step = 5        # max degrees per frame
            self.manual_step = 5     # manual control step size

            # Center positions (adjust for mounting offset)
            self.pan_center = pan_center
            self.tilt_center = tilt_center

            # Angle limits (adjust based on your mount)
            self.pan_min = 0
            self.pan_max = 180
            self.tilt_min = 60
            self.tilt_max = 120

            # Patrol settings (when no cat detected)
            self.patrol_step = 0.15
            self.patrol_direction = 1
            self.patrol_pan = float(pan_center)  # track position internally

            # Center servos
            self.center()
            print(f"[SERVO] Initialized (Pan: {self.pan_ch}, Tilt: {self.tilt_ch})")
            print(f"[SERVO] Mode: AUTO-FOLLOW")
            
        except Exception as e:
            print(f"[SERVO] Failed to initialize: {e}")
            self.enabled = False
            self.mode = self.MODE_OFF
    
    def center(self):
        """Reset servos to center position."""
        if not self.enabled:
            return
        try:
            self.servo.setAngle(self.pan_ch, self.pan_center)
            self.servo.setAngle(self.tilt_ch, self.tilt_center)
            self.patrol_pan = float(self.pan_center)
        except Exception as e:
            print(f"[SERVO] Error centering: {e}")
    
    def toggle_mode(self):
        """Cycle through modes: AUTO -> MANUAL -> OFF -> AUTO"""
        if not self.enabled:
            return
        
        self.mode = (self.mode + 1) % 3
        mode_names = ["OFF", "MANUAL", "AUTO"]
        print(f"[SERVO] Mode: {mode_names[self.mode]}")
        
        if self.mode == self.MODE_OFF:
            self.center()
    
    def get_mode_name(self):
        """Get current mode name for display."""
        if not self.enabled:
            return "DISABLED"
        mode_names = ["OFF", "MANUAL", "AUTO"]
        return mode_names[self.mode]
    
    def manual_pan_left(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.pan_ch)
            new_angle = max(self.pan_min, current - self.manual_step)
            self.servo.setAngle(self.pan_ch, new_angle)
        except Exception as e:
            print(f"[SERVO] Error: {e}")
    
    def manual_pan_right(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.pan_ch)
            new_angle = min(self.pan_max, current + self.manual_step)
            self.servo.setAngle(self.pan_ch, new_angle)
        except Exception as e:
            print(f"[SERVO] Error: {e}")
    
    def manual_tilt_up(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.tilt_ch)
            new_angle = min(self.tilt_max, current + self.manual_step)
            self.servo.setAngle(self.tilt_ch, new_angle)
        except Exception as e:
            print(f"[SERVO] Error: {e}")
    
    def manual_tilt_down(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.tilt_ch)
            new_angle = max(self.tilt_min, current - self.manual_step)
            self.servo.setAngle(self.tilt_ch, new_angle)
        except Exception as e:
            print(f"[SERVO] Error: {e}")
    
    def auto_follow(self, bbox_center_x, bbox_center_y, frame_w, frame_h):
        """Auto-follow mode: adjust servos to center camera on bbox."""
        if not self.enabled or self.mode != self.MODE_AUTO:
            return
        
        try:
            center_x = frame_w / 2
            center_y = frame_h / 2
            
            error_x = bbox_center_x - center_x
            error_y = bbox_center_y - center_y
            
            if abs(error_x) < self.deadzone and abs(error_y) < self.deadzone:
                return
            
            pan_adjust = (error_x / frame_w) * self.max_step
            tilt_adjust = -(error_y / frame_h) * self.max_step            
            current_pan = self.servo.getAngle(self.pan_ch)
            current_tilt = self.servo.getAngle(self.tilt_ch)
            
            new_pan = max(self.pan_min, min(self.pan_max, current_pan + pan_adjust))
            new_tilt = max(self.tilt_min, min(self.tilt_max, current_tilt + tilt_adjust))
            
            self.servo.setAngle(self.pan_ch, new_pan)
            self.servo.setAngle(self.tilt_ch, new_tilt)
            self.patrol_pan = new_pan  # sync for smooth patrol resume

        except Exception as e:
            print(f"[SERVO] Auto-follow error: {e}")

    def patrol(self):
        """Gentle patrol sweep when no cat is detected."""
        if not self.enabled or self.mode != self.MODE_AUTO:
            return

        try:
            # Reverse direction at limits
            if self.patrol_pan >= self.pan_max - 5:
                self.patrol_direction = -1
            elif self.patrol_pan <= self.pan_min + 5:
                self.patrol_direction = 1

            self.patrol_pan += self.patrol_step * self.patrol_direction
            self.patrol_pan = max(self.pan_min, min(self.pan_max, self.patrol_pan))
            self.servo.setAngle(self.pan_ch, self.patrol_pan)

        except Exception as e:
            print(f"[SERVO] Patrol error: {e}")

    def get_angles(self):
        """Get current servo angles for display."""
        if not self.enabled:
            return None, None
        try:
            pan = self.servo.getAngle(self.pan_ch)
            tilt = self.servo.getAngle(self.tilt_ch)
            return pan, tilt
        except Exception:
            return None, None


def draw_track(frame, track, model_w, model_h, debug=False, is_tentative=False, is_target=False):
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
    
    if is_target:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.drawMarker(frame, (center_x, center_y), (0, 255, 255),
                      cv2.MARKER_CROSS, 20, 2)


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


def main(debug=True, record=False, fps=20.0, log_positions=False, no_servo=False):
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
    
    servo_ctrl = ServoController(enabled=not no_servo)

    pos_logger = None
    log_count = 0

    if log_positions:
        pos_logger = PositionLogger()
        print("[LOG] Position logging enabled → occupancy_log.csv")

    out = None
    output_path = None
    written_frames = 0
    recording = record
    
    target_track_id = None 

    if recording:
        out, output_path = start_recording(fps, (640, 480))

    print("\nHotkeys:")
    print("  [q] quit  |  [r] record on/off  |  [d] debug on/off")
    if servo_ctrl.enabled:
        print("  [s] servo mode  |  [c] center servos  |  [arrows] manual control")
        print("  [0-9] target specific cat (AUTO mode)")

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

            target_track = None
            if servo_ctrl.mode == ServoController.MODE_AUTO:
                if confirmed_tracks:
                    if target_track_id is not None:
                        target_track = next((t for t in confirmed_tracks if t.id == target_track_id), None)
                        if target_track is None:
                            target_track = confirmed_tracks[0]
                    else:
                        target_track = confirmed_tracks[0]

                    x_center = target_track.bbox[0] / model_w * orig_w
                    y_center = target_track.bbox[1] / model_h * orig_h

                    servo_ctrl.auto_follow(x_center, y_center, orig_w, orig_h)
                else:
                    servo_ctrl.patrol()

            if debug:
                for track in tracker.tracks:
                    if not track.is_confirmed():
                        draw_track(frame, track, model_w, model_h, debug, True)

            for track in confirmed_tracks:
                if pos_logger is not None:
                    x_center = track.bbox[0] / model_w * orig_w
                    y_center = track.bbox[1] / model_h * orig_h
                    width = track.bbox[2] / model_w * orig_w
                    height = track.bbox[3] / model_h * orig_h
                    pos_logger.log(track.name, x_center, y_center, width, height)
                    log_count += 1

                is_target = (target_track is not None and track.id == target_track.id)
                draw_track(frame, track, model_w, model_h, debug, is_target=is_target)

            if debug:
                for det in detections:
                    x_c, y_c = det["box"][:2]
                    x = int(x_c / model_w * orig_w)
                    y = int(y_c / model_h * orig_h)
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)
                
                center_x, center_y = orig_w // 2, orig_h // 2
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)

            fps_count += 1
            if fps_count >= 30:
                current_fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0

            y_offset = 30
            status_lines = [
                f"FPS: {current_fps:.1f}",
                f"Tracked: {len(confirmed_tracks)}",
                f"REC: {'ON' if recording else 'OFF'}",
            ]
            
            if servo_ctrl.enabled:
                status_lines.append(f"Servo: {servo_ctrl.get_mode_name()}")
                pan, tilt = servo_ctrl.get_angles()
                if pan is not None and tilt is not None:
                    status_lines.append(f"Pan: {pan:.0f}° Tilt: {tilt:.0f}°")
                if target_track is not None:
                    status_lines.append(f"Target: Cat #{target_track.id}")
            
            if log_positions:
                status_lines.append(f"LOG: {log_count}")
            
            for i, line in enumerate(status_lines):
                color = (0, 0, 255) if (i == 2 and recording) else (255, 255, 255)
                cv2.putText(frame, line, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

            # Servo controls
            elif key == ord("s"):
                servo_ctrl.toggle_mode()
            
            elif key == ord("c"):
                servo_ctrl.center()
                print("[SERVO] Centered")
            
            elif key == 81 or key == ord('a'):
                servo_ctrl.manual_pan_left()
            elif key == 83 or key == ord('d'):
                servo_ctrl.manual_pan_right()
            elif key == 82 or key == ord('w'):
                servo_ctrl.manual_tilt_up()
            elif key == 84:
                servo_ctrl.manual_tilt_down()
            
            elif ord('0') <= key <= ord('9'):
                if servo_ctrl.mode == ServoController.MODE_AUTO:
                    if key == ord('0'):
                        target_track_id = None
                        print("[SERVO] Tracking any cat")
                    else:
                        target_track_id = int(chr(key))
                        print(f"[SERVO] Targeting cat #{target_track_id}")

    finally:
        if out is not None:
            stop_recording(out, output_path, written_frames)

        if pos_logger is not None:
            pos_logger.close()
            print(f"[LOG] Logged {log_count} positions to occupancy_log.csv")

        if servo_ctrl.enabled:
            servo_ctrl.center()
            print("[SERVO] Centered and shutdown")

        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live multi-cat tracker")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlays")
    parser.add_argument("--record", action="store_true", help="Start with recording enabled")
    parser.add_argument("--fps", type=float, default=20.0, help="Recording FPS")
    parser.add_argument("--log-positions", action="store_true",
                        help="Log pixel positions to occupancy_log.csv")
    parser.add_argument("--no-servo", action="store_true", help="Disable servo control")

    args = parser.parse_args()
    main(debug=args.debug, record=args.record, fps=args.fps, 
         log_positions=args.log_positions, no_servo=args.no_servo)