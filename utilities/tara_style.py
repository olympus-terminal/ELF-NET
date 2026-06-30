"""Shared matplotlib style for TARA-Oceans manuscript figures."""

from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

JOURNAL_SPECS = {
    'cell': {
        'single': 85,   # mm
        'onehalf': 114,  # mm
        'double': 178,   # mm
    }
}


def apply_tara_style():
    """Set rcParams for consistent 6pt Arial, thin lines, no top/right spines."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 6.0,
        'axes.labelsize': 6.0,
        'axes.titlesize': 6.0,
        'xtick.labelsize': 6.0,
        'ytick.labelsize': 6.0,
        'legend.fontsize': 6.0,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.25,
        'xtick.major.width': 0.25,
        'ytick.major.width': 0.25,
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        # Tight tick-label spacing matched to Fig 3D (default 3.5pt wastes space).
        'xtick.major.pad': 1.0,
        'ytick.major.pad': 1.0,
        'xtick.minor.pad': 1.0,
        'ytick.minor.pad': 1.0,
        'lines.markersize': 3.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def check_figure_size(fig, journal: str = 'cell', width: str = 'double',
                      tolerance_mm: float = 5.0) -> dict:
    """Check whether figure width matches journal spec."""
    spec_mm = JOURNAL_SPECS.get(journal, {}).get(width, 178)
    w_in, h_in = fig.get_size_inches()
    w_mm = w_in * 25.4
    ok = abs(w_mm - spec_mm) <= tolerance_mm
    return {'width_mm': w_mm, 'spec_mm': spec_mm, 'ok': ok,
            'width_in': w_in, 'height_in': h_in}


def save_figure(fig, name: str, output_dir: str = 'figures',
                formats: Optional[List[str]] = None) -> List[Path]:
    """Save figure in PDF and SVG (default) formats."""
    if formats is None:
        formats = ['pdf', 'svg']
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f'{name}.{fmt}'
        fig.savefig(p, dpi=300, bbox_inches='tight', transparent=False)
        paths.append(p)
    return paths


def add_panel_labels(axes, labels=None, x=-0.15, y=1.05, fontsize=8,
                     fontweight='bold'):
    """Add A, B, C... labels to a list of axes."""
    if labels is None:
        labels = [chr(65 + i) for i in range(len(axes))]
    for ax, lbl in zip(axes, labels):
        ax.text(x, y, lbl, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, va='top', ha='right')


def detect_text_overlaps(fig, threshold_pts=2.0):
    """Simple heuristic: warn if any two text artists overlap."""
    renderer = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts]
    texts += fig.texts
    overlaps = []
    for i, t1 in enumerate(texts):
        bb1 = t1.get_window_extent(renderer)
        for t2 in texts[i+1:]:
            bb2 = t2.get_window_extent(renderer)
            if bb1.overlaps(bb2):
                overlaps.append((t1.get_text()[:20], t2.get_text()[:20]))
    return overlaps
