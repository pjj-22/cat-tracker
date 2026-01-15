"""
Step 3: Build color profiles from labeled images

Reads labeled capture sessions and builds HSV color histograms for each cat.
Can merge multiple sessions for better diversity.

Usage:
    python3 build_profiles.py <session_dir> [session_dir2 ...]

Examples:
    python3 build_profiles.py captures/session_20260112_143022/
    python3 build_profiles.py captures/session_*/  # All sessions

Output:
    cat_profiles.json
"""

import os
import sys
import json
import cv2
import numpy as np
from glob import glob
from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier

def build_profiles_from_sessions(session_dirs, output_path="cat_profiles.json"):
    """
    Build color profiles from one or more labeled capture sessions.
    """
    print("="*70)
    print("BUILDING CAT COLOR PROFILES")
    print("="*70)

    extractor = ColorHistogramExtractor()
    identifier = ColorHistogramIdentifier(profile_path=output_path)

    total_sessions = 0
    total_images = 0
    skipped_images = 0

    for session_dir in session_dirs:
        if not os.path.exists(session_dir):
            print(f"\nWARNING:  Session not found: {session_dir}")
            continue

        labels_path = f"{session_dir}/labels.json"
        if not os.path.exists(labels_path):
            print(f"\nWARNING:  {session_dir}: Not labeled yet")
            print(f"   Run: python3 label.py {session_dir}")
            continue

        with open(labels_path, 'r') as f:
            labels_data = json.load(f)

        cat_names = labels_data.get('cat_names', {})
        labels = labels_data.get('labels', {})

        if not labels:
            print(f"\nWARNING:  {session_dir}: No images labeled")
            continue

        print(f"\n{os.path.basename(session_dir)}:")
        print(f"  Images labeled: {len(labels)}")

        session_images = 0
        session_skipped = 0

        for img_rel_path, cat_num in labels.items():
            cat_name = cat_names.get(cat_num)
            if not cat_name:
                continue

            img_path = f"{session_dir}/{img_rel_path}"
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # Use full image as bbox (already cropped)
            bbox = (0, 0, w, h)

            # Extract histogram
            hist_h, hist_s, hist_v = extractor.extract(img_rgb, bbox)

            if hist_h is not None:
                # Use relative path as source identifier for deduplication
                source_id = f"{os.path.basename(session_dir)}/{img_rel_path}"
                if identifier.add_training_sample(cat_name, hist_h, hist_s, hist_v, source_path=source_id):
                    session_images += 1
                else:
                    session_skipped += 1

        print(f"  Processed: {session_images} new images", end="")
        if session_skipped > 0:
            print(f" ({session_skipped} already in profile)")
        else:
            print()
        total_sessions += 1
        total_images += session_images
        skipped_images += session_skipped

    if not identifier.profiles:
        print("\nERROR: No valid profiles created")
        print("\nMake sure you have:")
        print("  1. Captured images: python3 capture.py")
        print("  2. Labeled them: python3 label.py <session_dir>")
        return

    # Save profiles
    identifier.save_profiles()

    print("\n" + "="*70)
    print("PROFILES SAVED")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"Sessions used: {total_sessions}")
    print(f"Total images: {total_images}")
    print(f"\nCats:")

    for cat_name, profile in identifier.profiles.items():
        sample_count = profile['sample_count']
        print(f"  {cat_name}: {sample_count} samples", end="")

        if sample_count < 15:
            print(" WARNING:  Low (recommend 30+ from diverse sessions)")
        elif sample_count < 30:
            print(" WARNING:  Moderate (recommend 30+)")
        else:
            print(" OK")

    # Profile separation analysis
    if len(identifier.profiles) >= 2:
        print("\n" + "="*70)
        print("PROFILE SEPARATION ANALYSIS")
        print("="*70)

        cat_names_list = list(identifier.profiles.keys())
        for i, cat1 in enumerate(cat_names_list):
            for cat2 in cat_names_list[i+1:]:
                profile1 = identifier.profiles[cat1]
                profile2 = identifier.profiles[cat2]

                h1 = np.array(profile1['hist_h'])
                h2 = np.array(profile2['hist_h'])

                # Bhattacharyya distance (0 = identical, 1 = completely different)
                bc = np.sum(np.sqrt(h1 * h2))
                distance = np.sqrt(1.0 - bc)

                print(f"\n{cat1} vs {cat2}: {distance:.3f}", end="")

                if distance < 0.2:
                    print(" ERROR: Too similar (will confuse)")
                elif distance < 0.35:
                    print(" WARNING:  Moderately similar")
                else:
                    print(" OK Well separated")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test tracking: python3 track_cats.py")
    print("2. Visualize profiles: python3 visualize_profiles.py")
    print("\nTo improve profiles:")
    print("  - Capture more sessions in different lighting/locations")
    print("  - Label and add them: python3 build_profiles.py captures/session_*/")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_profiles.py <session_dir> [session_dir2 ...]")
        print("\nExamples:")
        print("  python3 build_profiles.py captures/session_20260112_143022/")
        print("  python3 build_profiles.py captures/session_*/  # All sessions")
        print("\nAvailable labeled sessions:")

        if os.path.exists("captures"):
            sessions = sorted(glob("captures/session_*"), reverse=True)
            labeled = []
            for session in sessions:
                labels_path = f"{session}/labels.json"
                if os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        data = json.load(f)
                    image_count = len(data.get('labels', {}))
                    cat_count = len(data.get('cat_names', {}))
                    labeled.append(f"  {session} ({cat_count} cats, {image_count} images)")

            if labeled:
                for line in labeled:
                    print(line)
            else:
                print("  (none labeled yet - run label.py first)")
        else:
            print("  (no captures directory - run capture.py first)")

        sys.exit(1)

    session_dirs = sys.argv[1:]
    build_profiles_from_sessions(session_dirs)
