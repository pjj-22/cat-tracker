import os
import sys

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "Jetson"))
    from ServoKit import ServoKit
    SERVO_AVAILABLE = True
except Exception:
    SERVO_AVAILABLE = False


class ServoController:
    """Proportional pan/tilt servo controller.

    Modes: AUTO (follows cats), MANUAL (button/key control), OFF.
    """

    MODE_OFF = 0
    MODE_MANUAL = 1
    MODE_AUTO = 2

    def __init__(self, pan_channel=0, tilt_channel=1, enabled=True,
                 pan_center=60, tilt_center=90):
        self.enabled = enabled and SERVO_AVAILABLE
        self.mode = self.MODE_AUTO if self.enabled else self.MODE_OFF

        if not self.enabled:
            return

        try:
            self.servo = ServoKit(num_ports=max(pan_channel, tilt_channel) + 1)
            self.pan_ch = pan_channel
            self.tilt_ch = tilt_channel

            self.deadzone = 50
            self.max_step = 5
            self.manual_step = 5

            self.pan_center = pan_center
            self.tilt_center = tilt_center

            self.pan_min = 0
            self.pan_max = 180
            self.tilt_min = 60
            self.tilt_max = 120

            self.patrol_step = 0.6
            self.patrol_direction = 1
            self.patrol_pan = float(pan_center)

            self.center()
            print(f"[SERVO] Initialized (Pan: {self.pan_ch}, Tilt: {self.tilt_ch})")
            print(f"[SERVO] Mode: AUTO-FOLLOW")

        except Exception as e:
            print(f"[SERVO] Failed to initialize: {e}")
            self.enabled = False
            self.mode = self.MODE_OFF

    def center(self):
        if not self.enabled:
            return
        try:
            self.servo.setAngle(self.pan_ch, self.pan_center)
            self.servo.setAngle(self.tilt_ch, self.tilt_center)
            self.patrol_pan = float(self.pan_center)
        except Exception as e:
            print(f"[SERVO] Error centering: {e}")

    def toggle_mode(self):
        if not self.enabled:
            return
        self.mode = (self.mode + 1) % 3
        print(f"[SERVO] Mode: {self.get_mode_name()}")
        if self.mode == self.MODE_OFF:
            self.center()

    def get_mode_name(self):
        if not self.enabled:
            return "DISABLED"
        return ["OFF", "MANUAL", "AUTO"][self.mode]

    def manual_pan_left(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.pan_ch)
            self.servo.setAngle(self.pan_ch, max(self.pan_min, current - self.manual_step))
        except Exception as e:
            print(f"[SERVO] Error: {e}")

    def manual_pan_right(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.pan_ch)
            self.servo.setAngle(self.pan_ch, min(self.pan_max, current + self.manual_step))
        except Exception as e:
            print(f"[SERVO] Error: {e}")

    def manual_tilt_up(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.tilt_ch)
            self.servo.setAngle(self.tilt_ch, min(self.tilt_max, current + self.manual_step))
        except Exception as e:
            print(f"[SERVO] Error: {e}")

    def manual_tilt_down(self):
        if not self.enabled or self.mode != self.MODE_MANUAL:
            return
        try:
            current = self.servo.getAngle(self.tilt_ch)
            self.servo.setAngle(self.tilt_ch, max(self.tilt_min, current - self.manual_step))
        except Exception as e:
            print(f"[SERVO] Error: {e}")

    def auto_follow(self, bbox_center_x, bbox_center_y, frame_w, frame_h):
        if not self.enabled or self.mode != self.MODE_AUTO:
            return
        try:
            error_x = bbox_center_x - frame_w / 2
            error_y = bbox_center_y - frame_h / 2
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
            self.patrol_pan = new_pan
        except Exception as e:
            print(f"[SERVO] Auto-follow error: {e}")

    def patrol(self):
        if not self.enabled or self.mode != self.MODE_AUTO:
            return
        try:
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
        if not self.enabled:
            return None, None
        try:
            return self.servo.getAngle(self.pan_ch), self.servo.getAngle(self.tilt_ch)
        except Exception:
            return None, None
