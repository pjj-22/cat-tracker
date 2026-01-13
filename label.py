"""
Step 2: Label captured images

Interactive slideshow to label captured cat images.
Shows images one by one, you assign them to cats using number keys.

Usage:
    python3 label.py captures/session_20260112_143022/

Controls:
    1-9   - Assign image to cat #1-9 (prompts for name on first use)
    s     - Skip this image
    d     - Delete this image
    n     - Next track
    q     - Save and quit

Output:
    Creates labels.json in the session directory mapping images to cat names
"""

import os
import sys
import json
import cv2
from glob import glob

def label_session(session_dir):
    """Interactive labeling of captured images."""

    if not os.path.exists(session_dir):
        print(f"ERROR: Session directory not found: {session_dir}")
        return

    metadata_path = f"{session_dir}/metadata.json"
    if not os.path.exists(metadata_path):
        print(f"ERROR: No metadata found in {session_dir}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    labels_path = f"{session_dir}/labels.json"

    # Load existing labels if any
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        labels = labels_data.get('labels', {})
        cat_names = labels_data.get('cat_names', {})
        print(f"\nLoaded existing labels ({len(labels)} images labeled)")
    else:
        labels = {}  # image_path: cat_number
        cat_names = {}  # cat_number: cat_name

    print("\n" + "="*70)
    print("INTERACTIVE LABELING")
    print("="*70)
    print(f"\nSession: {session_dir}")
    print(f"Tracks: {metadata['total_tracks']}")

    if cat_names:
        print("\nConfigured cats:")
        for num, name in sorted(cat_names.items()):
            print(f"  [{num}] {name}")
    else:
        print("\nNo cats configured yet (will prompt when you press 1-9)")

    print("\nControls:")
    print("  [1-9] - Assign to cat #1-9")
    print("  [s]   - Skip image")
    print("  [d]   - Delete image")
    print("  [n]   - Next track")
    print("  [q]   - Save and quit")
    print("="*70)

    # Get all tracks
    track_dirs = sorted(glob(f"{session_dir}/track_*"))

    for track_dir in track_dirs:
        track_num = os.path.basename(track_dir).split('_')[1]
        image_files = sorted(glob(f"{track_dir}/*.jpg"))

        if not image_files:
            continue

        print(f"\n{'='*70}")
        print(f"Track {track_num}: {len(image_files)} images")
        print('='*70)

        img_idx = 0
        while img_idx < len(image_files):
            img_path = image_files[img_idx]
            img_name = os.path.relpath(img_path, session_dir)

            # Load and display image
            img = cv2.imread(img_path)
            if img is None:
                print(f"WARNING: Could not load {img_name}")
                img_idx += 1
                continue

            # Resize for display if too large
            display_img = img.copy()
            max_height = 600
            h, w = display_img.shape[:2]
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                display_img = cv2.resize(display_img, (new_w, max_height))

            # Add info overlay
            info_text = f"Track {track_num} | Image {img_idx + 1}/{len(image_files)}"
            cv2.putText(display_img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show existing label if any
            if img_name in labels:
                cat_num = labels[img_name]
                cat_name = cat_names.get(cat_num, f"Cat #{cat_num}")
                cv2.putText(display_img, f"Labeled: {cat_name}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Label Images - Press 1-9 to assign cat, s=skip, d=delete, n=next track, q=quit", display_img)
            key = cv2.waitKey(0) & 0xFF

            # Handle input
            if key == ord('q'):
                print("\nSaving and quitting...")
                cv2.destroyAllWindows()
                save_labels(session_dir, labels, cat_names)
                return

            elif key == ord('n'):
                print("Skipping to next track...")
                cv2.destroyAllWindows()
                break

            elif key == ord('s'):
                print(f"Skipped {img_name}")
                img_idx += 1

            elif key == ord('d'):
                print(f"Deleting {img_name}...")
                os.remove(img_path)
                # Remove from labels if it was labeled
                if img_name in labels:
                    del labels[img_name]
                # Remove from list
                image_files.pop(img_idx)
                print(f"Deleted")

            elif chr(key).isdigit() and chr(key) in '123456789':
                cat_num = chr(key)

                # Prompt for name if first time using this number
                if cat_num not in cat_names:
                    cv2.destroyAllWindows()
                    cat_name = input(f"\nEnter name for cat #{cat_num}: ").strip()
                    if not cat_name:
                        print("ERROR: Empty name, cancelled")
                        continue
                    cat_names[cat_num] = cat_name
                    print(f"Cat #{cat_num} = {cat_name}")
                    # Recreate window
                    cv2.imshow("Label Images - Press 1-9 to assign cat, s=skip, d=delete, n=next track, q=quit", display_img)

                # Label the image
                labels[img_name] = cat_num
                cat_name = cat_names[cat_num]
                print(f"{img_name} -> {cat_name}")
                img_idx += 1

            else:
                print(f"Unknown key: {chr(key) if key < 128 else key}")

        cv2.destroyAllWindows()

    # Auto-save at end
    print("\n" + "="*70)
    print("Labeling complete!")
    save_labels(session_dir, labels, cat_names)

def save_labels(session_dir, labels, cat_names):
    """Save labels to JSON file."""
    labels_path = f"{session_dir}/labels.json"

    data = {
        'cat_names': cat_names,
        'labels': labels
    }

    with open(labels_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved labels to {labels_path}")
    print(f"  Cats defined: {len(cat_names)}")
    print(f"  Images labeled: {len(labels)}")

    # Show breakdown
    if cat_names and labels:
        print("\nLabel breakdown:")
        for cat_num, cat_name in sorted(cat_names.items()):
            count = sum(1 for label in labels.values() if label == cat_num)
            print(f"  {cat_name}: {count} images")

    print("\n" + "="*70)
    print("NEXT STEP: Build color profiles")
    print(f"Run: python3 build_profiles.py {session_dir}")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 label.py <session_directory>")
        print("\nExample:")
        print("  python3 label.py captures/session_20260112_143022/")
        print("\nAvailable sessions:")

        if os.path.exists("captures"):
            sessions = sorted(glob("captures/session_*"), reverse=True)
            if sessions:
                for session in sessions[:5]:  # Show last 5
                    labels_file = f"{session}/labels.json"
                    if os.path.exists(labels_file):
                        with open(labels_file, 'r') as f:
                            data = json.load(f)
                        labeled_count = len(data.get('labels', {}))
                        print(f"  {session} ({labeled_count} images labeled)")
                    else:
                        print(f"  {session} (not labeled yet)")
            else:
                print("  (none found)")
        else:
            print("  (no captures directory found)")

        sys.exit(1)

    session_dir = sys.argv[1].rstrip('/')
    label_session(session_dir)
