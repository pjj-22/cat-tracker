"""
Step 2: Label captured tracks

Reviews each track from a capture session and lets you assign names.
Shows sample images from each track so you can see what was captured.

Usage:
    python3 label_tracks.py training_sessions/session_TIMESTAMP/
"""

import os
import sys
import json
import cv2
import numpy as np
from glob import glob

def show_track_montage(track_dir, num_samples=12):
    """
    Create a montage of sample images from a track.
    """
    image_files = sorted(glob(f"{track_dir}/frame_*.jpg"))
    
    if not image_files:
        print(f"  No images found in {track_dir}")
        return None
    
    # Select evenly spaced samples
    if len(image_files) > num_samples:
        indices = np.linspace(0, len(image_files)-1, num_samples, dtype=int)
        sample_files = [image_files[i] for i in indices]
    else:
        sample_files = image_files
    
    # Load images
    images = []
    for img_path in sample_files:
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to standard size for montage
            img = cv2.resize(img, (150, 150))
            images.append(img)
    
    if not images:
        return None
    
    # Create montage (grid layout)
    rows = 3
    cols = 4
    
    # Pad with blank images if needed
    while len(images) < rows * cols:
        images.append(np.zeros_like(images[0]))
    
    # Only use first 12 images
    images = images[:rows * cols]
    
    # Create grid
    montage_rows = []
    for row in range(rows):
        row_images = images[row*cols:(row+1)*cols]
        montage_rows.append(np.hstack(row_images))
    
    montage = np.vstack(montage_rows)
    return montage

def label_session(session_dir):
    """
    Label all tracks in a capture session.
    """
    # Load metadata
    metadata_path = f"{session_dir}/metadata.json"
    if not os.path.exists(metadata_path):
        print(f"❌ No metadata found in {session_dir}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\n" + "="*70)
    print("STEP 2: LABEL TRACKS")
    print("="*70)
    print(f"\nSession: {session_dir}")
    print(f"Captured: {metadata['total_tracks']} tracks")
    print(f"Total frames: {metadata['total_frames']}")
    
    # Check for existing labels
    labels_path = f"{session_dir}/labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"\nFound existing labels: {len(labels)} tracks labeled")
    else:
        labels = {}
    
    print("\n" + "="*70)
    print("Instructions:")
    print("- Review each track's images")
    print("- Enter cat name (or 'skip' to skip this track)")
    print("- Enter 'quit' to save and exit")
    print("="*70)
    
    track_dirs = sorted(glob(f"{session_dir}/track_*"))
    
    for track_dir in track_dirs:
        track_id = int(os.path.basename(track_dir).split('_')[1])
        
        # Skip if already labeled
        if str(track_id) in labels:
            print(f"\n✓ Track {track_id} already labeled as '{labels[str(track_id)]}'")
            continue
        
        # Get track info
        track_info = metadata['tracks'].get(str(track_id), {})
        sample_count = track_info.get('sample_count', 0)
        
        print(f"\n{'='*70}")
        print(f"Track {track_id}: {sample_count} samples")
        print('='*70)
        
        # Show montage
        print("Loading images...")
        montage = show_track_montage(track_dir)
        
        if montage is not None:
            cv2.imshow(f"Track {track_id} - Sample Images", montage)
            cv2.waitKey(100)  # Brief delay to ensure window appears
        
        # Get label
        while True:
            label = input(f"\nEnter name for Track {track_id} (or 'skip'/'quit'): ").strip()
            
            if label.lower() == 'quit':
                print("\nSaving and exiting...")
                cv2.destroyAllWindows()
                # Save labels
                with open(labels_path, 'w') as f:
                    json.dump(labels, f, indent=2)
                return labels
            
            elif label.lower() == 'skip':
                print(f"Skipping Track {track_id}")
                cv2.destroyAllWindows()
                break
            
            elif label == '':
                print("❌ Empty name not allowed")
                continue
            
            else:
                labels[str(track_id)] = label
                print(f"✓ Labeled Track {track_id} as '{label}'")
                cv2.destroyAllWindows()
                break
    
    # Save labels
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print("\n" + "="*70)
    print("LABELING COMPLETE")
    print("="*70)
    print(f"\nLabeled {len(labels)} tracks:")
    for track_id, name in sorted(labels.items(), key=lambda x: int(x[0])):
        print(f"  Track {track_id}: {name}")
    
    print("\n" + "="*70)
    print("NEXT STEP: Build color profiles")
    print(f"Run: python3 build_profiles.py {session_dir}")
    print("="*70)
    
    return labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 label_tracks.py <session_directory>")
        print("\nExample:")
        print("  python3 label_tracks.py training_sessions/session_20260110_143022/")
        print("\nAvailable sessions:")
        
        if os.path.exists("training_sessions"):
            sessions = sorted(glob("training_sessions/session_*"))
            if sessions:
                for session in sessions:
                    print(f"  {session}")
            else:
                print("  (none found)")
        else:
            print("  (no training_sessions directory found)")
        
        sys.exit(1)
    
    session_dir = sys.argv[1].rstrip('/')
    
    if not os.path.exists(session_dir):
        print(f"❌ Session directory not found: {session_dir}")
        sys.exit(1)
    
    label_session(session_dir)