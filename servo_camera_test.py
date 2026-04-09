"""
Pan/Tilt Servo Camera Test

Control a pan/tilt servo mount while viewing the camera feed.

Controls:
    Arrow Keys: Pan (left/right) and Tilt (up/down)
    WASD: Alternative pan/tilt control
    r: Reset servos to center position
    c: Capture image
    +/-: Adjust step size
    q: Quit

Requirements:
    pip install adafruit-circuitpython-servokit
"""

import sys
import os
import time
from datetime import datetime
import argparse

# Add example directory to path for ServoKit import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "example", "Jetson"))

import cv2
from picamera2 import Picamera2

try:
    from ServoKit import ServoKit
    SERVO_AVAILABLE = True
except ImportError:
    print("[WARN] ServoKit not available - running in camera-only mode")
    SERVO_AVAILABLE = False


class PanTiltController:
    """Controller for pan/tilt servo mount."""

    def __init__(self, pan_channel=0, tilt_channel=1, num_ports=2,
                 pan_center=118, tilt_center=90):
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.step = 5

        # Center positions (adjust for mounting offset)
        self.pan_center = pan_center
        self.tilt_center = tilt_center

        # Angle limits
        self.pan_min = 0
        self.pan_max = 180
        self.tilt_min = 15  # Limit tilt to prevent mechanical issues
        self.tilt_max = 165

        if SERVO_AVAILABLE:
            self.servo = ServoKit(num_ports)
        else:
            self.servo = None
            self._mock_pan = pan_center
            self._mock_tilt = tilt_center

    @property
    def pan_angle(self):
        if self.servo:
            return self.servo.getAngle(self.pan_channel)
        return self._mock_pan

    @property
    def tilt_angle(self):
        if self.servo:
            return self.servo.getAngle(self.tilt_channel)
        return self._mock_tilt

    def pan_left(self):
        new_angle = max(self.pan_min, self.pan_angle - self.step)
        self._set_pan(new_angle)

    def pan_right(self):
        new_angle = min(self.pan_max, self.pan_angle + self.step)
        self._set_pan(new_angle)

    def tilt_up(self):
        new_angle = min(self.tilt_max, self.tilt_angle + self.step)
        self._set_tilt(new_angle)

    def tilt_down(self):
        new_angle = max(self.tilt_min, self.tilt_angle - self.step)
        self._set_tilt(new_angle)

    def reset(self):
        self._set_pan(self.pan_center)
        self._set_tilt(self.tilt_center)

    def _set_pan(self, angle):
        if self.servo:
            self.servo.setAngle(self.pan_channel, angle)
        else:
            self._mock_pan = angle

    def _set_tilt(self, angle):
        if self.servo:
            self.servo.setAngle(self.tilt_channel, angle)
        else:
            self._mock_tilt = angle

    def increase_step(self):
        self.step = min(20, self.step + 1)

    def decrease_step(self):
        self.step = max(1, self.step - 1)


def draw_overlay(frame, controller, capture_count):
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Draw crosshair at center
    center_x, center_y = w // 2, h // 2
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)

    # Draw status info
    y_offset = 30
    info_lines = [
        f"Pan: {controller.pan_angle:.1f}",
        f"Tilt: {controller.tilt_angle:.1f}",
        f"Step: {controller.step}",
        f"Captures: {capture_count}",
    ]

    for i, line in enumerate(info_lines):
        cv2.putText(
            frame, line, (10, y_offset + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # Draw controls hint
    controls = "Arrows/WASD: Move | r: Reset | c: Capture | +/-: Step | q: Quit"
    cv2.putText(
        frame, controls, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
    )

    if not SERVO_AVAILABLE:
        cv2.putText(
            frame, "SERVO NOT CONNECTED", (w // 2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )


def capture_image(frame):
    """Save current frame as image."""
    os.makedirs("captures", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captures/capture_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[CAPTURE] Saved: {filename}")
    return filename


def main(pan_channel=0, tilt_channel=1, resolution=(640, 480)):
    print("Initializing pan/tilt camera test...")

    # Initialize controller
    controller = PanTiltController(
        pan_channel=pan_channel,
        tilt_channel=tilt_channel,
        num_ports=max(pan_channel, tilt_channel) + 1
    )

    # Initialize camera
    print("Starting camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": resolution, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    capture_count = 0
    window_title = "Pan/Tilt Camera Test"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    print("\nControls:")
    print("  Arrow Keys / WASD: Pan and Tilt")
    print("  r: Reset to center")
    print("  c: Capture image")
    print("  +/-: Adjust step size")
    print("  q: Quit\n")

    try:
        while True:
            frame = picam2.capture_array()

            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            draw_overlay(frame_bgr, controller, capture_count)
            cv2.imshow(window_title, frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                break

            # Pan/Tilt controls
            elif key == ord("a") or key == 81:  # Left arrow
                controller.pan_left()
            elif key == ord("d") or key == 83:  # Right arrow
                controller.pan_right()
            elif key == ord("w") or key == 82:  # Up arrow
                controller.tilt_up()
            elif key == ord("s") or key == 84:  # Down arrow
                controller.tilt_down()

            # Reset
            elif key == ord("r"):
                controller.reset()
                print("[RESET] Servos centered")

            # Capture
            elif key == ord("c"):
                capture_image(frame_bgr)
                capture_count += 1

            # Step size
            elif key == ord("+") or key == ord("="):
                controller.increase_step()
                print(f"[STEP] {controller.step}")
            elif key == ord("-"):
                controller.decrease_step()
                print(f"[STEP] {controller.step}")

    finally:
        print("\nShutting down...")
        if controller.servo:
            controller.reset()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pan/Tilt Servo Camera Test")
    parser.add_argument(
        "--pan-channel", type=int, default=0,
        help="Servo channel for pan (default: 0)"
    )
    parser.add_argument(
        "--tilt-channel", type=int, default=1,
        help="Servo channel for tilt (default: 1)"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera height (default: 480)"
    )

    args = parser.parse_args()
    main(
        pan_channel=args.pan_channel,
        tilt_channel=args.tilt_channel,
        resolution=(args.width, args.height)
    )
