import os

import colorspacious
import numpy as np

# Font size configuration
TICKFONT_SIZE = 24
AXISFONT_SIZE = 24

TITLE_FONT_SIZE = 32
SUBTITLE_FONT_SIZE = 28

LEGENDTITLE_FONT_SIZE = 24
LEGEND_FONT_SIZE = 18

# Trace colors configuration
TRACE_COLORS = {
    'air': '#ff00af',
    'Sapph': '#6f3198',
    'Boro': '#ff7e00',
    'CaF2': '#00b7ef',
    'Empty': '#787878',
    'Aluminum': '#C0C0C0',
    'Acrylic': '#90EE90',
    'AlumAlum': '#C0C0C0',  # Same as Aluminum
    'BlackBlack': '#90EE90',  # Same as Acrylic
    'Alt': '#00b7ef',  # Same as CaF2
    'solar': '#ffc20e',
    'ir_net': '#ff0000',
    'ir_in': '#990030',
    'ir_sky': '#4d6df3',
    'pyrge': '#ed1c24',
    'A_default': '#1f77b4',
    'B_default': '#ff9500',
    'C_default': '#00cc00',
}

# Plot configuration
MIN_GAP_SIZE = 10
MAX_TIME_DIFF_SECONDS = 10
MAX_TIME_DIFF_AIR_TEMP = 1  # minutes
TEMP_Y_RANGE = [0, 65]
RADIATION_Y_RANGE = [0, 950]
TEMP_DIFF_Y_RANGE = [-1.5, 4.0]
if os.environ.get("TWOPANE"):
    TEMP_DIFF_Y_RANGE = [-0.5, 6.5]
TEMP_DIFF_TICK_INTERVAL = 0.5
TEMP_DIFF_TICK_INTERVAL_ALUMINUM = 1.0
PLOT_WIDTH = 1400
ROW_HEIGHTS = [1, 1, 0.6, 0.6, 0.6]
PLOT_HEIGHT = 600*sum(ROW_HEIGHTS)

# Condition colors configuration
CONDITION_COLORS = {
    'direct sun': {'color': '#FFFFE0', 'opacity': 0.3},  # Gold/yellow
    'indirect sun': {'color': '#C6C6C4', 'opacity': 0.2},  # Light yellow
    'night': {'color': '#605c94', 'opacity': 0.2},  # Dark blue (transparent)
    'cloudy': {'color': '#87CEEB', 'opacity': 0.3},  # Light blue
    'adjustment period': {'color': '#FFB6C1', 'opacity': 0.3},  # Light red
}

# Color conversion helpers
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-255)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple (0-255) to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def rgb_to_lab(rgb):
    """Convert RGB (0-255) to Lab color space"""
    # Convert to 0-1 range for colorspacious
    rgb_normalized = np.array(rgb) / 255.0
    # colorspacious expects sRGB in 0-1 range
    lab = colorspacious.cspace_convert(rgb_normalized, "sRGB1", "CIELab")
    return lab

def lab_to_rgb(lab):
    """Convert Lab color space to RGB (0-255)"""
    # colorspacious returns sRGB in 0-1 range
    rgb_normalized = colorspacious.cspace_convert(lab, "CIELab", "sRGB1")
    # Clip to valid range and convert to 0-255
    rgb = np.clip(rgb_normalized * 255, 0, 255)
    return rgb

def average_colors_weighted(colors_with_weights):
    """
    Average colors in Lab space with weights.
    
    Args:
        colors_with_weights: list of tuples (hex_color, opacity, weight)
    
    Returns:
        tuple: (averaged_hex_color, averaged_opacity)
    """
    if not colors_with_weights:
        return '#FFFFFF', 0.0
    
    total_weight = sum(w for _, _, w in colors_with_weights)
    if total_weight == 0:
        return '#FFFFFF', 0.0
    
    # Convert all colors to Lab and accumulate weighted values
    lab_sum = np.zeros(3)
    opacity_sum = 0.0
    
    for hex_color, opacity, weight in colors_with_weights:
        rgb = hex_to_rgb(hex_color)
        lab = rgb_to_lab(rgb)
        normalized_weight = weight / total_weight
        lab_sum += lab * normalized_weight
        opacity_sum += opacity * normalized_weight
    
    # Convert averaged Lab back to RGB and hex
    avg_rgb = lab_to_rgb(lab_sum)
    avg_hex = rgb_to_hex(avg_rgb)
    
    return avg_hex, opacity_sum
