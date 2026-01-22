"""
Analyze time spent in user-defined room zones.

Usage:
    python3 analyze_zones.py [--hours N] [--cat NAME]

Zone definitions are loaded from zones.json. If it doesn't exist,
a sample file will be created that you can customize.

Output:
    - Console report with percentages per zone
    - Pie chart PNG for each cat

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os
import json

from cat_tracker.spatial import ZoneAnalyzer


def create_sample_zones():
    """Create a sample zones.json file."""
    sample_zones = {
        "Window": {"x1": 2.0, "y1": 2.0, "x2": 3.0, "y2": 2.5, "color": "#FFD700"},
        "Food Bowl": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5, "color": "#FF6347"},
        "Doorway": {"x1": 1.0, "y1": 0.0, "x2": 2.0, "y2": 0.5, "color": "#4169E1"},
        "Center": {"x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0, "color": "#32CD32"}
    }

    with open('zones.json', 'w') as f:
        json.dump(sample_zones, f, indent=2)

    print("Created sample zones.json - please customize for your room!")
    print("\nZone format:")
    print('  "Zone Name": {"x1": min_x, "y1": min_y, "x2": max_x, "y2": max_y, "color": "#hex"}')
    print("\nCoordinates are in meters, matching your calibration.")


def analyze_zones(hours=24, cat_filter=None, log_path='occupancy_log.csv', zones_path='zones.json'):
    """
    Analyze and visualize time spent in each zone.

    Args:
        hours: Number of hours to analyze
        cat_filter: Optional cat name to filter
        log_path: Path to position log CSV
        zones_path: Path to zones JSON
    """
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        print("Run 'python3 track_cats.py --log-positions' first to collect data.")
        return

    if not os.path.exists(zones_path):
        print(f"No {zones_path} found. Creating sample file...")
        create_sample_zones()
        return

    df = pd.read_csv(log_path)

    if len(df) == 0:
        print("No data in log file.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by time
    cutoff = datetime.now() - timedelta(hours=hours)
    df = df[df['timestamp'] >= cutoff]

    if len(df) == 0:
        print(f"No data in the last {hours} hours.")
        return

    analyzer = ZoneAnalyzer(zones_path)

    if not analyzer.zones:
        print(f"No zones defined in {zones_path}")
        return

    if cat_filter:
        cats = [cat_filter]
        if cat_filter not in df['cat_name'].values:
            print(f"Cat '{cat_filter}' not found in data.")
            print(f"Available: {', '.join(df['cat_name'].unique())}")
            return
    else:
        cats = df['cat_name'].unique()

    print("=" * 60)
    print(f"ZONE ANALYSIS - Last {hours} Hours")
    print("=" * 60)

    for cat in cats:
        cat_df = df[df['cat_name'] == cat]
        total = len(cat_df)

        print(f"\n{cat}:")
        print(f"  Total positions logged: {total:,}")

        time_span = cat_df['timestamp'].max() - cat_df['timestamp'].min()
        print(f"  Time span: {time_span}")

        results = analyzer.analyze(df, cat_name=cat)

        print(f"\n  Zone breakdown:")
        for zone_name, percentage in sorted(results.items(), key=lambda x: -x[1]):
            bar = '#' * int(percentage / 2)
            print(f"    {zone_name:15s}: {percentage:5.1f}% {bar}")

        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Get colors from zone definitions
            colors = []
            for zone_name in results.keys():
                if zone_name in analyzer.zones:
                    colors.append(analyzer.zones[zone_name].get('color', '#808080'))
                else:
                    colors.append('#808080')  # Gray for "Other"

            # Filter out very small slices for cleaner chart
            labels = []
            values = []
            chart_colors = []
            for (name, pct), color in zip(results.items(), colors):
                if pct >= 1.0:  # Only show zones with >= 1%
                    labels.append(name)
                    values.append(pct)
                    chart_colors.append(color)

            if values:
                ax.pie(
                    values, labels=labels, autopct='%1.1f%%',
                    colors=chart_colors, startangle=90
                )
                ax.set_title(
                    f'{cat} - Zone Activity (Last {hours}h)',
                    fontsize=14, fontweight='bold'
                )

                filename = f'zones_{cat}_{hours}h.png'
                plt.tight_layout()
                plt.savefig(filename, dpi=150)
                print(f"\n  Saved chart: {filename}")
                plt.close()

    print("\n" + "=" * 60)

    print("\nZone definitions (from zones.json):")
    for name, zone in analyzer.zones.items():
        print(f"  {name}: ({zone['x1']}, {zone['y1']}) to ({zone['x2']}, {zone['y2']}) meters")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze time spent in room zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 analyze_zones.py                # All cats, last 24 hours
    python3 analyze_zones.py --hours 8      # Last 8 hours
    python3 analyze_zones.py --cat Honey    # Just Honey

Zone Configuration:
    Edit zones.json to define your room zones. Each zone has:
    - x1, y1: bottom-left corner (meters)
    - x2, y2: top-right corner (meters)
    - color: hex color for charts
        """
    )
    parser.add_argument('--hours', type=int, default=24, help='Last N hours (default: 24)')
    parser.add_argument('--cat', help='Filter by cat name')
    parser.add_argument('--log', default='occupancy_log.csv', help='Path to log file')
    parser.add_argument('--zones', default='zones.json', help='Path to zones file')

    args = parser.parse_args()

    analyze_zones(
        hours=args.hours,
        cat_filter=args.cat,
        log_path=args.log,
        zones_path=args.zones
    )


if __name__ == "__main__":
    main()
