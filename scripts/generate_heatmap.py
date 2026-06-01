"""
Generate spatial occupancy heatmaps from position data.

Usage:
    python3 generate_heatmap.py [--cat NAME] [--hours N]
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta
import argparse


def load_positions(csv_path, cat_name=None, hours=24):
    """Load positions and bbox dimensions from CSV."""
    positions = []

    cutoff = datetime.now() - timedelta(hours=hours)

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = datetime.fromisoformat(row['timestamp'])
                if timestamp < cutoff:
                    continue

                if cat_name and row['cat_name'] != cat_name:
                    continue

                # Handle old format (no width/height) and new format
                width = float(row.get('width', 50))
                height = float(row.get('height', 50))

                positions.append({
                    'x': float(row['pixel_x']),
                    'y': float(row['pixel_y']),
                    'w': width,
                    'h': height
                })
    except FileNotFoundError:
        print(f"ERROR: {csv_path} not found")
        print("Run: python3 track_cats.py --log-positions")
        return None

    return positions


def generate_heatmap(cat_name=None, hours=24, width=640, height=480):
    """Generate heatmap from position data using full bounding boxes."""
    positions = load_positions('occupancy_log.csv', cat_name, hours)

    if positions is None or len(positions) == 0:
        print(f"No data for {cat_name or 'any cat'} in last {hours} hours")
        return

    # Create accumulator for bbox regions
    heatmap = np.zeros((height, width), dtype=np.float32)

    for pos in positions:
        # Calculate bbox corners from center + dimensions
        x1 = int(max(0, pos['x'] - pos['w'] / 2))
        y1 = int(max(0, pos['y'] - pos['h'] / 2))
        x2 = int(min(width, pos['x'] + pos['w'] / 2))
        y2 = int(min(height, pos['y'] + pos['h'] / 2))

        # Fill in the bbox region
        heatmap[y1:y2, x1:x2] += 1

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Apply gaussian blur for smoother visualization
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap, sigma=8)

    # Plot heatmap
    im = ax.imshow(heatmap_smooth, origin='upper', cmap='hot',
                   aspect='auto', interpolation='bilinear', alpha=0.85)

    # Formatting
    title = f"{cat_name or 'All Cats'} - Last {hours} Hours"
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time Spent (frames)', fontsize=12)

    # Stats
    stats_text = f"Positions logged: {len(positions)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    filename = f"heatmap_{cat_name or 'all'}_{hours}h.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate position heatmaps')
    parser.add_argument('--cat', help='Filter by cat name')
    parser.add_argument('--hours', type=int, default=24, help='Last N hours')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    args = parser.parse_args()

    generate_heatmap(cat_name=args.cat, hours=args.hours,
                     width=args.width, height=args.height)
