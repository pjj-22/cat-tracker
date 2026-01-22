"""
Spatial tracking utilities for position logging.
"""

import csv
from datetime import datetime


class PositionLogger:
    """Log cat pixel positions to CSV."""

    def __init__(self, filepath='occupancy_log.csv'):
        self.filepath = filepath
        self.file = open(filepath, 'a', newline='')
        self.writer = csv.writer(self.file)

        # Write header if new file
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.writer.writerow(['timestamp', 'cat_name', 'pixel_x', 'pixel_y'])

    def log(self, cat_name, pixel_x, pixel_y):
        """
        Log a position entry.

        Args:
            cat_name: Name of the cat
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
        """
        timestamp = datetime.now().isoformat()
        self.writer.writerow([timestamp, cat_name, pixel_x, pixel_y])
        self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.close()