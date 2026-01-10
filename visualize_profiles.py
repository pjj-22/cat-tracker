"""
Visualize cat color histograms with actual color representations.

Shows:
- HSV histogram distributions as bar charts
- Color swatches for dominant hue bins
- RGB approximations of what the histograms represent
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def hsv_to_rgb_array(h, s, v):
    """Convert HSV values to RGB for visualization."""
    # Create a 1x1 pixel in HSV
    hsv_pixel = np.uint8([[[h, s, v]]])
    # Convert to RGB
    rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)
    return rgb_pixel[0, 0]

def visualize_cat_profile(cat_name, profile, save_path=None):
    """
    Visualize a cat's color profile with actual colors.
    """
    hist_h = np.array(profile['hist_h'])
    hist_s = np.array(profile['hist_s'])
    hist_v = np.array(profile['hist_v'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{cat_name}'s Color Profile ({profile['sample_count']} samples)", 
                 fontsize=16, fontweight='bold')
    
    # 1. Hue histogram with color bars
    ax1 = plt.subplot(3, 2, 1)
    bins_h = len(hist_h)
    x = np.arange(bins_h)
    
    # Create color bars for each hue bin
    colors_h = []
    for i in range(bins_h):
        # Map bin index to hue value (0-180 in OpenCV)
        hue = int(i * 180 / bins_h)
        # Use medium saturation and value to show the hue clearly
        rgb = hsv_to_rgb_array(hue, 200, 200)
        colors_h.append(rgb / 255.0)  # Normalize for matplotlib
    
    bars = ax1.bar(x, hist_h, color=colors_h, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Hue (Color)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Hue Distribution (Actual Colors)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add hue degree labels
    hue_labels = [0, 30, 60, 90, 120, 150, 180]
    hue_positions = [int(h * bins_h / 180) for h in hue_labels]
    ax1.set_xticks(hue_positions)
    ax1.set_xticklabels([f"{h}°" for h in hue_labels], fontsize=9)
    
    # 2. Dominant colors palette
    ax2 = plt.subplot(3, 2, 2)
    ax2.axis('off')
    ax2.set_title('Dominant Colors (Top 5 Hues)', fontsize=12, fontweight='bold')
    
    # Find top 5 hue bins
    top_hue_indices = np.argsort(hist_h)[-5:][::-1]
    
    palette_height = 0.8
    palette_y_start = 0.1
    
    for idx, bin_idx in enumerate(top_hue_indices):
        if hist_h[bin_idx] < 0.01:  # Skip if < 1%
            continue
            
        # Calculate hue for this bin
        hue_start = int(bin_idx * 180 / bins_h)
        hue_end = int((bin_idx + 1) * 180 / bins_h)
        hue_mid = (hue_start + hue_end) // 2
        
        # Get dominant saturation and value for this cat
        sat_idx = np.argmax(hist_s)
        val_idx = np.argmax(hist_v)
        sat = int(sat_idx * 255 / len(hist_s))
        val = int(val_idx * 255 / len(hist_v))
        
        # Create color swatch
        rgb = hsv_to_rgb_array(hue_mid, sat, val)
        
        # Draw color rectangle
        y_pos = palette_y_start + idx * (palette_height / 5)
        rect = plt.Rectangle((0.1, y_pos), 0.3, palette_height / 6,
                            facecolor=rgb/255.0, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        
        # Add label
        percentage = hist_h[bin_idx] * 100
        ax2.text(0.45, y_pos + palette_height / 12,
                f"{percentage:.1f}% @ {hue_mid}° HSV({hue_mid}, {sat}, {val})",
                fontsize=10, verticalalignment='center')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. Saturation histogram
    ax3 = plt.subplot(3, 2, 3)
    bins_s = len(hist_s)
    x = np.arange(bins_s)
    
    # Create gradient from gray to saturated
    colors_s = []
    for i in range(bins_s):
        sat = int(i * 255 / bins_s)
        # Use orange hue (typical cat color) to show saturation
        rgb = hsv_to_rgb_array(15, sat, 200)
        colors_s.append(rgb / 255.0)
    
    ax3.bar(x, hist_s, color=colors_s, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Saturation (Color Intensity)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Saturation Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add saturation labels
    sat_labels = [0, 64, 128, 192, 255]
    sat_positions = [int(s * bins_s / 255) for s in sat_labels]
    ax3.set_xticks(sat_positions)
    ax3.set_xticklabels(sat_labels, fontsize=9)
    
    # 4. Value (brightness) histogram
    ax4 = plt.subplot(3, 2, 4)
    bins_v = len(hist_v)
    x = np.arange(bins_v)
    
    # Create gradient from black to white
    colors_v = []
    for i in range(bins_v):
        val = int(i * 255 / bins_v)
        gray = val / 255.0
        colors_v.append([gray, gray, gray])
    
    ax4.bar(x, hist_v, color=colors_v, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Value (Brightness)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Brightness Distribution', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    val_labels = [0, 64, 128, 192, 255]
    val_positions = [int(v * bins_v / 255) for v in val_labels]
    ax4.set_xticks(val_positions)
    ax4.set_xticklabels(val_labels, fontsize=9)
    
    # 5. Color composition pie chart
    ax5 = plt.subplot(3, 2, 5)
    
    # Group hues into color families
    color_families = {
        'Red/Orange': (0, 30),      # 0-30°
        'Orange': (30, 60),          # 30-60°
        'Yellow': (60, 90),          # 60-90°
        'Green': (90, 150),          # 90-150°
        'Blue/Cyan': (150, 180),     # 150-180°
    }
    
    family_percentages = {}
    family_colors = {
        'Red/Orange': '#FF6B35',
        'Orange': '#FF9F1C',
        'Yellow': '#FFD700',
        'Green': '#2EC4B6',
        'Blue/Cyan': '#4A90E2'
    }
    
    for family, (hue_min, hue_max) in color_families.items():
        bin_min = int(hue_min * bins_h / 180)
        bin_max = int(hue_max * bins_h / 180)
        percentage = np.sum(hist_h[bin_min:bin_max]) * 100
        if percentage > 0.5:  # Only show if > 0.5%
            family_percentages[family] = percentage
    
    if family_percentages:
        wedges, texts, autotexts = ax5.pie(
            family_percentages.values(),
            labels=family_percentages.keys(),
            colors=[family_colors[k] for k in family_percentages.keys()],
            autopct='%1.1f%%',
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax5.set_title('Color Family Distribution', fontsize=12, fontweight='bold')
    
    # 6. Profile statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    ax6.set_title('Profile Statistics', fontsize=12, fontweight='bold')
    
    # Calculate stats
    dominant_hue_idx = np.argmax(hist_h)
    dominant_hue = int(dominant_hue_idx * 180 / bins_h)
    dominant_hue_pct = hist_h[dominant_hue_idx] * 100
    
    dominant_sat_idx = np.argmax(hist_s)
    dominant_sat = int(dominant_sat_idx * 255 / bins_s)
    
    dominant_val_idx = np.argmax(hist_v)
    dominant_val = int(dominant_val_idx * 255 / bins_v)
    
    hue_entropy = -np.sum(hist_h * np.log(hist_h + 1e-10))
    
    stats_text = f"""
Sample Count: {profile['sample_count']}

Dominant Hue: {dominant_hue}° ({dominant_hue_pct:.1f}%)
Dominant Saturation: {dominant_sat}/255
Dominant Brightness: {dominant_val}/255

Hue Entropy: {hue_entropy:.2f}
(Lower = more distinctive color)

Profile Quality:
{'✓ Excellent' if profile['sample_count'] >= 15 else '⚠ Needs more samples'}
{'✓ Distinctive' if hue_entropy < 2.0 else '⚠ High color variation'}
    """
    
    ax6.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
            family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    profile_path = "cat_profiles.json"
    
    if not os.path.exists(profile_path):
        print(f"❌ Profile file not found: {profile_path}")
        print("\nRun train_cat_profiles_fixed.py first to create profiles")
        exit(1)
    
    with open(profile_path, "r") as f:
        data = json.load(f)
    
    if not data:
        print("❌ No profiles found in file")
        exit(1)
    
    print("="*60)
    print(f"Visualizing {len(data)} cat profile(s)")
    print("="*60)
    
    for cat_name, profile in data.items():
        print(f"\nGenerating visualization for {cat_name}...")
        save_path = f"{cat_name.lower().replace(' ', '_')}_profile.png"
        visualize_cat_profile(cat_name, profile, save_path)
        print(f"Close the window to continue...\n")
    
    print("="*60)
    print("Visualization complete!")
    print("="*60)