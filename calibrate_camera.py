"""
Camera calibration tool for spatial tracking.

Usage:
    python3 calibrate_camera.py

Instructions:
    1. Shows live camera feed
    2. Click reference points anywhere on the floor (minimum 4, recommended 10-20)
    3. Enter real-world position for each point
    4. Saves homography matrix to calibration.json

Tips:
    - Use painter's tape, coins, or small objects as markers
    - More points = better accuracy (10-20 recommended)
    - Spread points across the whole visible area
    - Can use any pattern: L-shape, grid, scattered, etc.
    - Measure positions from a corner you choose as (0, 0)
"""

from picamera2 import Picamera2
import cv2
import time
import numpy as np

from cat_tracker.spatial import Calibration


def calibrate():
    """Run interactive calibration with unlimited reference points."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("=" * 60)
    print("CAMERA CALIBRATION")
    print("=" * 60)
    print("\nPlace markers anywhere in your room and click them.")
    print("Minimum: 4 points | Recommended: 10-20 points")
    print("\nTips:")
    print("  - Use tape, coins, or small objects as markers")
    print("  - Spread them across the whole visible area")
    print("  - Can be any pattern (L-shape, grid, scattered)")
    print("  - Measure from a corner you choose as (0, 0)")
    print("\nControls:")
    print("  LEFT CLICK - Add reference point")
    print("  r          - Remove last point")
    print("  c          - Clear all points")
    print("  ENTER      - Done (minimum 4 points)")
    print("  q          - Quit")
    print("=" * 60)

    points_pixel = []
    points_real = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_pixel.append([x, y])
            print(f"\n  Point {len(points_pixel)}: Pixel ({x}, {y})")
            
            print(f"  Enter floor position for point {len(points_pixel)}:")
            while True:
                try:
                    real_x = float(input("    X (meters): "))
                    real_y = float(input("    Y (meters): "))
                    points_real.append([real_x, real_y])
                    print(f"  ✓ Point {len(points_pixel)}: ({x}, {y}) → ({real_x:.2f}m, {real_y:.2f}m)")
                    break
                except ValueError:
                    print("    Invalid input. Enter a number.")

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for i, pt in enumerate(points_pixel):
                cv2.circle(frame_bgr, tuple(pt), 6, (0, 255, 0), -1)
                cv2.putText(frame_bgr, str(i+1), (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(points_pixel) >= 2:
                for i in range(len(points_pixel) - 1):
                    cv2.line(frame_bgr, tuple(points_pixel[i]), 
                            tuple(points_pixel[i+1]), (0, 255, 0), 1)

            # Status
            status = f"Points: {len(points_pixel)}"
            if len(points_pixel) < 4:
                status += " (need 4 minimum)"
                color = (0, 0, 255)
            else:
                status += " (press ENTER when done)"
                color = (0, 255, 0)
            
            cv2.putText(frame_bgr, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Calibration", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nCalibration cancelled.")
                return

            elif key == ord('r') and points_pixel:
                removed_px = points_pixel.pop()
                removed_real = points_real.pop()
                print(f"\n  Removed point {len(points_pixel)+1}")

            elif key == ord('c'):
                points_pixel.clear()
                points_real.clear()
                print("\n  Cleared all points")

            elif key == 13:  # Enter
                if len(points_pixel) >= 4:
                    break
                else:
                    print(f"\n  Need at least 4 points (have {len(points_pixel)})")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    pts_pixel = np.array(points_pixel, dtype=np.float32)
    pts_real = np.array(points_real, dtype=np.float32)
    
    H, status = cv2.findHomography(pts_pixel, pts_real, method=cv2.RANSAC)
    pts_real_predicted = cv2.perspectiveTransform(
        pts_pixel.reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    
    errors = np.linalg.norm(pts_real - pts_real_predicted, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print("\n" + "=" * 60)
    print("CALIBRATION QUALITY")
    print("=" * 60)
    print(f"Points used: {len(points_pixel)}")
    print(f"Mean error: {mean_error:.3f} meters")
    print(f"Max error: {max_error:.3f} meters")
    
    if max_error > 0.1:
        print("\n⚠️  WARNING: High error detected")
        print("   Some points may be misplaced or measurements are inaccurate")
        print("   Points with largest errors:")
        
        error_indices = np.argsort(errors)[::-1][:3]
        for idx in error_indices:
            print(f"     Point {idx+1}: error = {errors[idx]:.3f}m")
    else:
        print("\n✓ Calibration quality: Good")

    calibration = Calibration()
    calibration.save(points_pixel, points_real)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\nSaved to: calibration.json")
    print("\nTest transformation:")

    center_pixel = (320, 240)
    floor_x, floor_y = calibration.pixel_to_floor(*center_pixel)
    print(f"  Frame center {center_pixel} → Floor ({floor_x:.2f}, {floor_y:.2f}) meters")

    print("\nNext steps:")
    print("  1. Visualize pattern: python3 visualize_calibration.py")
    print("  2. Start tracking: python3 track_cats.py --log-positions")


if __name__ == "__main__":
    calibrate()