#!/usr/bin/env python3
"""
Fix color channels in captured images.

The original capture.py incorrectly converted colors, making cats appear blue.
This script swaps R and B channels back to correct the colors.

Usage:
    python3 fix_colors.py                    # Fix all sessions
    python3 fix_colors.py captures/session_* # Fix specific sessions
"""

import os
import sys
import cv2
from glob import glob


def fix_image(filepath):
    """Swap R and B channels in an image."""
    img = cv2.imread(filepath)
    if img is None:
        return False

    # Swap R and B channels (BGR -> RGB, then save as BGR = effectively swap)
    fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filepath, fixed)
    return True


def fix_session(session_dir):
    """Fix all images in a capture session."""
    if not os.path.exists(session_dir):
        print(f"  Skipping {session_dir} (not found)")
        return 0

    # Find all jpg files in track directories
    images = glob(f"{session_dir}/track_*/*.jpg")

    if not images:
        print(f"  Skipping {session_dir} (no images)")
        return 0

    fixed_count = 0
    for img_path in images:
        if fix_image(img_path):
            fixed_count += 1

    print(f"  Fixed {fixed_count} images in {os.path.basename(session_dir)}")
    return fixed_count


def main():
    print("=" * 60)
    print("FIX CAPTURED IMAGE COLORS")
    print("=" * 60)

    # Determine which sessions to fix
    if len(sys.argv) > 1:
        session_dirs = sys.argv[1:]
    else:
        # Find all sessions
        session_dirs = sorted(glob("captures/session_*"))

    if not session_dirs:
        print("\nNo capture sessions found.")
        print("Run from the cat-tracker directory, or specify paths:")
        print("  python3 fix_colors.py captures/session_*")
        return

    print(f"\nFound {len(session_dirs)} session(s) to fix:\n")

    total_fixed = 0
    for session_dir in session_dirs:
        total_fixed += fix_session(session_dir)

    print("\n" + "=" * 60)
    print(f"DONE - Fixed {total_fixed} images total")
    print("=" * 60)
    print("\nYour labels are still valid - no need to relabel!")
    print("You can now rebuild profiles:")
    print("  python3 build_profiles.py captures/session_*")


if __name__ == "__main__":
    main()
