"""
Visualize calibration pattern and quality metrics.

Usage:
    python3 visualize_calibration.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def visualize():
    """Show calibration pattern and error metrics."""
    try:
        with open('calibration.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("ERROR: calibration.json not found")
        print("Run: python3 calibrate_camera.py")
        return
    
    pts_real = np.array(data['reference_points_real'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot reference points
    ax.scatter(pts_real[:, 0], pts_real[:, 1], c='blue', s=100, zorder=3, label='Reference Points')
    
    # Label points
    for i, pt in enumerate(pts_real):
        ax.annotate(f'{i+1}', (pt[0], pt[1]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold')
    
    # Connect points to show pattern
    if len(pts_real) >= 2:
        ax.plot(pts_real[:, 0], pts_real[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Calibration Pattern ({len(pts_real)} points)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    # Add quality metrics
    mean_err = data.get('mean_error', 0)
    max_err = data.get('max_error', 0)
    
    quality = "Good" if max_err < 0.1 else "Check for errors"
    color = 'green' if max_err < 0.1 else 'red'
    
    info_text = f"Mean error: {mean_err:.3f}m\n"
    info_text += f"Max error: {max_err:.3f}m\n"
    info_text += f"Quality: {quality}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig('calibration_pattern.png', dpi=150)
    print("✓ Saved visualization to calibration_pattern.png")
    plt.show()


if __name__ == "__main__":
    visualize()