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
    """Load positions from CSV."""
    positions_x = []
    positions_y = []

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

                positions_x.append(float(row['pixel_x']))
                positions_y.append(float(row['pixel_y']))
    except FileNotFoundError:
        print(f"ERROR: {csv_path} not found")
        print("Run: python3 track_cats.py --log-positions")
        return None, None

    return np.array(positions_x), np.array(positions_y)


def generate_heatmap(cat_name=None, hours=24):
    """Generate heatmap from position data."""
    x, y = load_positions('occupancy_log.csv', cat_name, hours)

    if x is None or len(x) == 0:
        print(f"No data for {cat_name or 'any cat'} in last {hours} hours")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # 2D histogram
    h, xedges, yedges = np.histogram2d(x, y, bins=50)

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # Note: y-axis inverted for image coordinates
    im = ax.imshow(h.T, origin='upper', extent=extent,
                   cmap='hot', aspect='auto', interpolation='gaussian',
                   alpha=0.8)

    # Formatting
    title = f"{cat_name or 'All Cats'} - Last {hours} Hours"
    ax.set_xlabel('Camera Width (pixels)', fontsize=12)
    ax.set_ylabel('Camera Height (pixels)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Set axis limits to camera resolution
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)  # Inverted for image coordinates

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time Spent (frames)', fontsize=12)

    # Stats box
    stats_text = f"Total positions: {len(x)}\n"
    stats_text += f"Tracking time: ~{len(x)/12:.1f} minutes"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    filename = f"heatmap_{cat_name or 'all'}_{hours}h.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmap to {filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate position heatmaps')
    parser.add_argument('--cat', help='Filter by cat name')
    parser.add_argument('--hours', type=int, default=24, help='Last N hours')
    args = parser.parse_args()

    generate_heatmap(cat_name=args.cat, hours=args.hours)