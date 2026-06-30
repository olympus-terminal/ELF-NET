#!/usr/bin/env python3
"""
Earth-from-Space Palette for TARA Oceans Manuscript
=====================================================

Unified color scheme based on natural Earth colors as seen from orbit:
deep ocean blues, coastal turquoise, land greens, earth tones, cloud/ice white.

Usage:
    from palette import (
        get_sequential_cmap, get_diverging_cmap, get_categorical_colors,
        BASIN_COLORS, LINEAGE_COLORS, ENV_CATEGORY_COLORS
    )

Created: 2026-02-05
"""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY COLOR DEFINITIONS - Earth from Space
# ══════════════════════════════════════════════════════════════════════════════

# ── Deep Ocean Blues ──────────────────────────────────────────────────────────
ABYSS = (0.004, 0.098, 0.220)           # #010E38 - Mariana depth
DEEP_OCEAN = (0.031, 0.188, 0.420)      # #08306B - Mid-Atlantic
OCEAN_BLUE = (0.129, 0.400, 0.545)      # #21668B - Open ocean
COASTAL_BLUE = (0.255, 0.573, 0.612)    # #41929C - Continental shelf

# ── Coastal Transition ────────────────────────────────────────────────────────
TURQUOISE = (0.369, 0.678, 0.651)       # #5EADA6 - Tropical shallows
SEAFOAM = (0.498, 0.757, 0.718)         # #7FC1B7 - Reef/lagoon
PALE_AQUA = (0.725, 0.875, 0.839)       # #B9DFD6 - Tidal zone

# ── Land Greens ───────────────────────────────────────────────────────────────
FOREST_GREEN = (0.180, 0.420, 0.239)    # #2E6B3D - Tropical forest
SAVANNA = (0.420, 0.533, 0.290)         # #6B884A - Grassland
PALE_GREEN = (0.682, 0.800, 0.565)      # #AECC90 - Coastal vegetation

# ── Earth Tones ───────────────────────────────────────────────────────────────
SAND = (0.871, 0.776, 0.624)            # #DEC69F - Beach/dune
DESERT_TAN = (0.824, 0.659, 0.447)      # #D2A872 - Sahara
CLAY = (0.678, 0.486, 0.306)            # #AD7C4E - Exposed soil
SIENNA = (0.545, 0.353, 0.200)          # #8B5A33 - Laterite

# ── Atmosphere/Ice ────────────────────────────────────────────────────────────
CLOUD_WHITE = (0.965, 0.973, 0.980)     # #F7F8FA - Cirrus
ICE_WHITE = (0.918, 0.941, 0.961)       # #EAF0F5 - Polar ice
SNOW = (0.980, 0.984, 0.988)            # #FAFBFC - Fresh snow

# ══════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL COLORMAPS (for non-negative data: abundance, counts, R², etc.)
# ══════════════════════════════════════════════════════════════════════════════

# Ocean depth: white → deep blue (for abundance, counts)
_OCEAN_SEQ = [
    SNOW,
    ICE_WHITE,
    PALE_AQUA,
    TURQUOISE,
    COASTAL_BLUE,
    OCEAN_BLUE,
    DEEP_OCEAN,
    ABYSS,
]
OCEAN_CMAP = LinearSegmentedColormap.from_list('earth_ocean', _OCEAN_SEQ, N=256)

# Forest: white → green (for chlorophyll, productivity)
_FOREST_SEQ = [
    SNOW,
    (0.925, 0.953, 0.906),  # Hint of green
    PALE_GREEN,
    SAVANNA,
    FOREST_GREEN,
    (0.102, 0.290, 0.165),  # Darker forest
]
FOREST_CMAP = LinearSegmentedColormap.from_list('earth_forest', _FOREST_SEQ, N=256)

# Thermal: white → tan → sienna (for temperature, positive metrics)
_THERMAL_SEQ = [
    SNOW,
    ICE_WHITE,
    SAND,
    DESERT_TAN,
    CLAY,
    SIENNA,
]
THERMAL_CMAP = LinearSegmentedColormap.from_list('earth_thermal', _THERMAL_SEQ, N=256)

# Coastal: white → turquoise → deep (for coastal/marine metrics)
_COASTAL_SEQ = [
    SNOW,
    PALE_AQUA,
    SEAFOAM,
    TURQUOISE,
    COASTAL_BLUE,
    OCEAN_BLUE,
]
COASTAL_CMAP = LinearSegmentedColormap.from_list('earth_coastal', _COASTAL_SEQ, N=256)

# ══════════════════════════════════════════════════════════════════════════════
# DIVERGING COLORMAPS (for data centered at zero: correlations, residuals)
# ══════════════════════════════════════════════════════════════════════════════

# Ocean-Land diverging: blue (negative) ↔ white ↔ brown (positive)
_OCEAN_LAND_DIV = [
    ABYSS,              # -1.0: Strong negative
    DEEP_OCEAN,         # -0.7
    OCEAN_BLUE,         # -0.4
    COASTAL_BLUE,       # -0.2
    PALE_AQUA,          # -0.1
    CLOUD_WHITE,        #  0.0: Center
    SAND,               # +0.1
    DESERT_TAN,         # +0.2
    CLAY,               # +0.4
    SIENNA,             # +0.7
    (0.400, 0.240, 0.120),  # +1.0: Strong positive (darker brown)
]
DIVERGING_CMAP = LinearSegmentedColormap.from_list('earth_diverging', _OCEAN_LAND_DIV, N=256)

# Cool diverging: deep blue ↔ white ↔ forest green (alternative)
_COOL_DIV = [
    ABYSS,
    DEEP_OCEAN,
    OCEAN_BLUE,
    PALE_AQUA,
    CLOUD_WHITE,
    PALE_GREEN,
    SAVANNA,
    FOREST_GREEN,
    (0.102, 0.290, 0.165),
]
COOL_DIVERGING_CMAP = LinearSegmentedColormap.from_list('earth_cool_div', _COOL_DIV, N=256)

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORICAL COLORS (for discrete groups: basins, lineages, modules)
# ══════════════════════════════════════════════════════════════════════════════

# 10-color categorical palette from Earth tones
EARTH_CATEGORICAL = [
    DEEP_OCEAN,         # 0: Deep blue
    TURQUOISE,          # 1: Turquoise
    FOREST_GREEN,       # 2: Forest green
    SAVANNA,            # 3: Savanna yellow-green
    DESERT_TAN,         # 4: Desert tan
    CLAY,               # 5: Clay brown
    COASTAL_BLUE,       # 6: Coastal blue
    PALE_GREEN,         # 7: Pale green
    SIENNA,             # 8: Sienna
    OCEAN_BLUE,         # 9: Ocean blue
]

# Basin colors (standardized across all figures)
BASIN_COLORS = {
    'Atlantic': DEEP_OCEAN,          # Deep blue
    'Pacific': TURQUOISE,            # Turquoise
    'Indian': FOREST_GREEN,          # Forest green
    'Southern': COASTAL_BLUE,        # Coastal blue
    'Mediterranean': DESERT_TAN,     # Tan
    'Arctic': ICE_WHITE,             # Ice white
    'Red Sea': CLAY,                 # Clay
}

# Lineage colors (for algal taxonomic groups)
LINEAGE_COLORS = {
    'Chlorellaceae': FOREST_GREEN,
    'Haptophyta': TURQUOISE,
    'Bolidophyceae': COASTAL_BLUE,
    'Mamiellophyceae': SAVANNA,
    'Pelagophyceae': OCEAN_BLUE,
    'Chrysophyceae': DESERT_TAN,
    'Prasinophyceae': PALE_GREEN,
    'Pyramimonadales': SEAFOAM,
    'Trebouxiophyceae': DEEP_OCEAN,
    'Other': (0.6, 0.6, 0.6),  # Gray
}

# Environmental category colors (for annotation bars)
ENV_CATEGORY_COLORS = {
    'Temperature': COASTAL_BLUE,
    'Productivity': FOREST_GREEN,
    'Ocean Color': TURQUOISE,
    'Bathymetry': DEEP_OCEAN,
    'Atmospheric': PALE_AQUA,
    'Salinity': OCEAN_BLUE,
    'Nutrient': SAVANNA,
}

# Dataset colors
DATASET_COLORS = {
    'LA4SR': DEEP_OCEAN,
    'AlgaGPT': TURQUOISE,
    'TARA': FOREST_GREEN,
}

# Module colors (for clustering results)
MODULE_COLORS = [
    DEEP_OCEAN,
    TURQUOISE,
    FOREST_GREEN,
    SAVANNA,
    DESERT_TAN,
    CLAY,
    COASTAL_BLUE,
    OCEAN_BLUE,
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_sequential_cmap(data_type='ocean'):
    """
    Get appropriate sequential colormap for data type.

    Parameters
    ----------
    data_type : str
        One of: 'ocean', 'forest', 'thermal', 'coastal'

    Returns
    -------
    LinearSegmentedColormap
    """
    cmaps = {
        'ocean': OCEAN_CMAP,
        'forest': FOREST_CMAP,
        'thermal': THERMAL_CMAP,
        'coastal': COASTAL_CMAP,
        'abundance': OCEAN_CMAP,
        'chlorophyll': FOREST_CMAP,
        'productivity': FOREST_CMAP,
        'temperature': THERMAL_CMAP,
        'r2': OCEAN_CMAP,
        'performance': OCEAN_CMAP,
    }
    return cmaps.get(data_type.lower(), OCEAN_CMAP)

def get_diverging_cmap(style='ocean_land'):
    """
    Get diverging colormap for correlation/residual data.

    Parameters
    ----------
    style : str
        One of: 'ocean_land' (blue-white-brown), 'cool' (blue-white-green)

    Returns
    -------
    LinearSegmentedColormap
    """
    if style == 'cool':
        return COOL_DIVERGING_CMAP
    return DIVERGING_CMAP

def get_categorical_colors(n, palette='earth'):
    """
    Get n categorical colors.

    Parameters
    ----------
    n : int
        Number of colors needed
    palette : str
        One of: 'earth', 'basins', 'lineages', 'modules'

    Returns
    -------
    list of RGB tuples
    """
    if palette == 'basins':
        return list(BASIN_COLORS.values())[:n]
    elif palette == 'lineages':
        return list(LINEAGE_COLORS.values())[:n]
    elif palette == 'modules':
        return MODULE_COLORS[:n]
    else:
        # Cycle through earth categorical if n > 10
        return [EARTH_CATEGORICAL[i % len(EARTH_CATEGORICAL)] for i in range(n)]

def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex string."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

def hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION / USAGE GUIDE
# ══════════════════════════════════════════════════════════════════════════════

COLORMAP_RULES = """
COLORMAP SELECTION RULES
========================

USE SEQUENTIAL (get_sequential_cmap) FOR:
- Abundance / counts (OCEAN_CMAP)
- R² / performance metrics (OCEAN_CMAP)
- Chlorophyll / productivity (FOREST_CMAP)
- Temperature (THERMAL_CMAP)
- Cosine similarity 0-1 (OCEAN_CMAP or THERMAL_CMAP)
- Any non-negative continuous data

USE DIVERGING (get_diverging_cmap) FOR:
- Correlation coefficients (-1 to +1)
- Residuals (centered at 0)
- Log fold changes
- Z-scores
- Any data with meaningful center point

NEVER USE:
- Diverging colormap for non-diverging data (e.g., R², abundance)
- Rainbow colormaps (not perceptually uniform)
- Default matplotlib colors for categorical data
"""

if __name__ == '__main__':
    # Demo/test
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 4))

    # Sequential colormaps
    for ax, (name, cmap) in zip(axes[0], [
        ('Ocean', OCEAN_CMAP),
        ('Forest', FOREST_CMAP),
        ('Thermal', THERMAL_CMAP),
        ('Coastal', COASTAL_CMAP),
    ]):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_title(name, fontsize=8)
        ax.set_yticks([])
        ax.set_xticks([])

    # Diverging colormaps
    gradient = np.linspace(-1, 1, 256).reshape(1, -1)
    axes[1, 0].imshow(gradient, aspect='auto', cmap=DIVERGING_CMAP)
    axes[1, 0].set_title('Diverging (Ocean-Land)', fontsize=8)
    axes[1, 0].set_yticks([])

    axes[1, 1].imshow(gradient, aspect='auto', cmap=COOL_DIVERGING_CMAP)
    axes[1, 1].set_title('Diverging (Cool)', fontsize=8)
    axes[1, 1].set_yticks([])

    # Categorical
    for i, color in enumerate(EARTH_CATEGORICAL):
        axes[1, 2].barh(i, 1, color=color)
    axes[1, 2].set_title('Categorical', fontsize=8)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_yticks([])

    # Basin colors
    for i, (name, color) in enumerate(BASIN_COLORS.items()):
        axes[1, 3].barh(i, 1, color=color, label=name)
    axes[1, 3].set_title('Basins', fontsize=8)
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_yticks(range(len(BASIN_COLORS)))
    axes[1, 3].set_yticklabels(list(BASIN_COLORS.keys()), fontsize=6)

    plt.tight_layout()
    plt.savefig('palette_demo.pdf', bbox_inches='tight')
    print("Saved palette_demo.pdf")
