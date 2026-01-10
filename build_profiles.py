"""
Step 3: Build color profiles from labeled tracks

Reads labeled tracks and builds HSV color histograms for each cat.
Saves profiles to cat_profiles.json for use in tracking.

Usage:
    python3 build_profiles.py training_sessions/session_TIMESTAMP/
"""

import os
import sys
import json
import cv2
import numpy as np
from glob import glob
from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier

def build_profiles_from_session(session_dir, output_path="cat_profiles.json"):
    """
    Build color profiles from a labeled training session.
    """
    # Load labels
    labels_path = f"{session_dir}/labels.json"
    if not os.path.exists(labels_path):
        print(f"❌ No labels found in {session_dir}")
        print(f"   Run: python3 label_tracks.py {session_dir}")
        return
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    if not labels:
        print("❌ No tracks labeled")
        return
    
    print("\n" + "="*70)
    print("STEP 3: BUILD COLOR PROFILES")
    print("="*70)
    print(f"\nSession: {session_dir}")
    print(f"Labeled tracks: {len(labels)}")
    
    # Initialize extractor and identifier
    extractor = ColorHistogramExtractor()
    identifier = ColorHistogramIdentifier(profile_path=output_path)
    
    # Process each labeled track
    for track_id, cat_name in sorted(labels.items(), key=lambda x: int(x[0])):
        track_dir = f"{session_dir}/track_{int(track_id):03d}"
        
        if not os.path.exists(track_dir):
            print(f"\n⚠️  Track directory not found: {track_dir}")
            continue
        
        # Get all images for this track
        image_files = sorted(glob(f"{track_dir}/frame_*.jpg"))
        
        if not image_files:
            print(f"\n⚠️  No images found for Track {track_id}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing Track {track_id}: {cat_name}")
        print(f"  Found {len(image_files)} images")
        
        # Extract histograms from each image
        valid_samples = 0
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            # Use full image as bbox (it's already a crop)
            bbox = (0, 0, w, h)
            
            # Extract histogram
            hist_h, hist_s, hist_v = extractor.extract(img_rgb, bbox)
            
            if hist_h is not None:
                identifier.add_training_sample(cat_name, hist_h, hist_s, hist_v)
                valid_samples += 1
        
        print(f"  Extracted {valid_samples} valid histograms")
        
        if cat_name in identifier.profiles:
            profile = identifier.profiles[cat_name]
            print(f"  Total samples for {cat_name}: {profile['sample_count']}")
    
    # Save profiles
    identifier.save_profiles()
    
    print("\n" + "="*70)
    print("PROFILE BUILDING COMPLETE")
    print("="*70)
    print(f"\nSaved to: {output_path}")
    print(f"\nProfiles created for {len(identifier.profiles)} cats:")
    
    for cat_name, profile in identifier.profiles.items():
        print(f"\n{cat_name}:")
        print(f"  Samples: {profile['sample_count']}")
        
        # Show dominant hue
        hist_h = np.array(profile['hist_h'])
        dominant_bin = np.argmax(hist_h)
        dominant_hue = int(dominant_bin * 180 / len(hist_h))
        dominant_pct = hist_h[dominant_bin] * 100
        
        print(f"  Dominant hue: {dominant_hue}° ({dominant_pct:.1f}% of pixels)")
        
        if profile['sample_count'] < 20:
            print(f"  ⚠️  Low sample count (recommended: 50+)")
        elif profile['sample_count'] < 50:
            print(f"  ✓ Moderate sample count")
        else:
            print(f"  ✓ Good sample count")
    
    # Compare profiles if multiple cats
    if len(identifier.profiles) >= 2:
        print("\n" + "="*70)
        print("Profile Separation Analysis")
        print("="*70)
        
        cat_names = list(identifier.profiles.keys())
        for i, cat1 in enumerate(cat_names):
            for cat2 in cat_names[i+1:]:
                profile1 = identifier.profiles[cat1]
                profile2 = identifier.profiles[cat2]
                
                h1 = np.array(profile1['hist_h'])
                h2 = np.array(profile2['hist_h'])
                
                # Bhattacharyya distance
                bc = np.sum(np.sqrt(h1 * h2))
                distance = np.sqrt(1.0 - bc)
                
                print(f"\n{cat1} vs {cat2}:")
                print(f"  Distance: {distance:.3f}")
                
                if distance < 0.2:
                    print("  ❌ PROBLEM: Too similar! May confuse frequently")
                elif distance < 0.35:
                    print("  ⚠️  WARNING: Moderately similar, may confuse occasionally")
                else:
                    print("  ✓ Well separated (good)")
    
    print("\n" + "="*70)
    print("READY TO TEST")
    print("="*70)
    print("\nRun: python3 track_cats.py")
    print("Your cats should now be identified by name!")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_profiles.py <session_directory>")
        print("\nExample:")
        print("  python3 build_profiles.py training_sessions/session_20260110_143022/")
        print("\nAvailable sessions:")
        
        if os.path.exists("training_sessions"):
            sessions = sorted(glob("training_sessions/session_*"))
            if sessions:
                for session in sessions:
                    # Check if labeled
                    labels_path = f"{session}/labels.json"
                    if os.path.exists(labels_path):
                        with open(labels_path, 'r') as f:
                            labels = json.load(f)
                        print(f"  {session} ({len(labels)} tracks labeled)")
                    else:
                        print(f"  {session} (not labeled yet)")
            else:
                print("  (none found)")
        else:
            print("  (no training_sessions directory found)")
        
        sys.exit(1)
    
    session_dir = sys.argv[1].rstrip('/')
    
    if not os.path.exists(session_dir):
        print(f"❌ Session directory not found: {session_dir}")
        sys.exit(1)
    
    build_profiles_from_session(session_dir)