"""
Analyze position data and generate statistics.

Usage:
    python3 analyze_positions.py
"""

import csv
import numpy as np
from datetime import datetime
from collections import defaultdict


def analyze():
    """Analyze position log and print statistics."""
    try:
        with open('occupancy_log.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print("ERROR: occupancy_log.csv not found")
        print("Run: python3 track_cats.py --log-positions")
        return

    if len(rows) == 0:
        print("No data in occupancy_log.csv")
        return

    cat_counts = defaultdict(int)
    cat_areas = defaultdict(list)
    timestamps = []

    for row in rows:
        cat_name = row['cat_name']
        cat_counts[cat_name] += 1

        width = float(row['width'])
        height = float(row['height'])
        area = width * height
        cat_areas[cat_name].append(area)

        timestamps.append(datetime.fromisoformat(row['timestamp']))

    total_positions = len(rows)
    start_time = min(timestamps)
    end_time = max(timestamps)
    duration = end_time - start_time

    print("=" * 60)
    print("POSITION LOG ANALYSIS")
    print("=" * 60)
    print(f"\nTotal positions logged: {total_positions}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nEstimated tracking time: ~{total_positions/12:.1f} minutes")
    print(f"(Assuming ~12 FPS)")

    print("\n" + "-" * 60)
    print("PER-CAT BREAKDOWN")
    print("-" * 60)

    for cat_name in sorted(cat_counts.keys()):
        count = cat_counts[cat_name]
        percentage = (count / total_positions) * 100
        areas = np.array(cat_areas[cat_name])

        print(f"\n{cat_name}:")
        print(f"  Positions: {count}")
        print(f"  Percentage: {percentage:.1f}%")
        print(f"  Time visible: ~{count/12:.1f} minutes")
        print(f"\n  Bbox statistics:")
        print(f"    Avg area: {np.mean(areas):.0f} px²")
        print(f"    Min area: {np.min(areas):.0f} px² (far/sitting)")
        print(f"    Max area: {np.max(areas):.0f} px² (close/lying)")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\nGenerate heatmaps:")
    for cat_name in sorted(cat_counts.keys()):
        print(f"\n  {cat_name}:")
        print(f"    python3 generate_heatmap.py --cat '{cat_name}' --hours 24")
        print(f"    python3 generate_heatmap.py --cat '{cat_name}' --hours 24 --weighted")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze()