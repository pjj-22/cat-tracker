#!/usr/bin/env python3
"""
Run identification on all labeled captures and report accuracy.
Usage: python3 test_profiles.py [captures/session_*/]
"""

import sys
import json
import os
import glob

import cv2
import numpy as np

from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier
from cat_tracker.config import load_config


def main():
    cfg = load_config()
    id_cfg = cfg.get('identification', {})

    extractor = ColorHistogramExtractor(
        min_saturation=id_cfg.get('min_saturation', 20),
        min_value=id_cfg.get('min_value', 20),
    )
    identifier = ColorHistogramIdentifier(
        profile_path=id_cfg.get('profile_path', 'cat_profiles.json'),
        hsv_weights=id_cfg.get('hsv_weights', [0.7, 0.2, 0.1]),
    )

    if not identifier.profiles:
        print("No profiles loaded. Run build_profiles.py first.")
        sys.exit(1)

    session_dirs = sys.argv[1:] or sorted(glob.glob('captures/session_*/'))
    if not session_dirs:
        print("No session dirs found.")
        sys.exit(1)

    correct = 0
    wrong = 0
    skipped = 0
    per_cat_correct = {}
    per_cat_total = {}
    failures = []

    for session_dir in session_dirs:
        labels_path = os.path.join(session_dir, 'labels.json')
        if not os.path.exists(labels_path):
            continue

        with open(labels_path) as f:
            label_data = json.load(f)

        cat_names = label_data.get('cat_names', {})
        labels = label_data.get('labels', {})

        for rel_path, cat_id in labels.items():
            true_name = cat_names.get(str(cat_id))
            if not true_name:
                skipped += 1
                continue

            img_path = os.path.join(session_dir, rel_path)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            hist_h, hist_s, hist_v = extractor.extract(img_rgb, (0, 0, w, h))

            if hist_h is None:
                skipped += 1
                continue

            pred_name, confidence, distances = identifier.identify(hist_h, hist_s, hist_v)

            per_cat_total[true_name] = per_cat_total.get(true_name, 0) + 1

            if pred_name == true_name:
                correct += 1
                per_cat_correct[true_name] = per_cat_correct.get(true_name, 0) + 1
            else:
                wrong += 1
                failures.append((img_path, true_name, pred_name, confidence, distances))

    total = correct + wrong
    if total == 0:
        print("No labeled images found.")
        return

    print(f"\nOverall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Skipped (bad crop/no pixels): {skipped}")

    print("\nPer-cat accuracy:")
    for cat in sorted(per_cat_total):
        c = per_cat_correct.get(cat, 0)
        t = per_cat_total[cat]
        print(f"  {cat}: {c}/{t} ({100*c/t:.1f}%)")

    if failures:
        print(f"\nFirst 10 misidentifications:")
        for img_path, true_name, pred_name, conf, distances in failures[:10]:
            dist_str = ', '.join(f"{n}={d:.3f}" for n, d in sorted(distances.items()))
            print(f"  {os.path.basename(img_path)}  true={true_name}  pred={pred_name}  conf={conf:.2f}  [{dist_str}]")
            print(f"    {img_path}")


if __name__ == '__main__':
    main()
