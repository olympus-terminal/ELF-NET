#!/usr/bin/env python3
"""
edge_style.py — RALPH42 shared edge styling for data marks
===========================================================

Single source of truth for the 0.1pt black edges that must appear on every
scatter/hexbin/bar/violin/boxplot data mark in the TARA-Oceans main figures.

Usage:
    from edge_style import EDGE_KWARGS, apply_edges, scatter_kw, hexbin_kw

    # scatter:
    ax.scatter(x, y, c=values, cmap='viridis', **scatter_kw())

    # hexbin:
    hb = ax.hexbin(x, y, gridsize=40, cmap='viridis', **hexbin_kw())

    # bar:
    ax.bar(positions, heights, color='C0', **EDGE_KWARGS)

    # post-hoc on an existing collection:
    apply_edges(ax.collections[-1])

Design constraints (from FIGURE_PROTOCOL.md and the /artist skill):
    - Line weight is 0.1 pt (extremely thin but visible at print resolution)
    - Color is pure black (never gray, never a palette color)
    - No outline for lines/paths/fill-between regions — only for discrete marks

Created: 2026-04-11 (RALPH42 figure overhaul)
"""

# Canonical edge width in points (matplotlib uses points for linewidths)
EDGE_LINEWIDTH = 0.1
EDGE_COLOR = "black"

# The dict every figure script should unpack into scatter/bar/violin calls.
# Kept as a module-level constant so grep can locate all usages during audit.
EDGE_KWARGS = {
    "edgecolor": EDGE_COLOR,
    "linewidth": EDGE_LINEWIDTH,
}


def scatter_kw(**overrides):
    """Return edge kwargs merged with user overrides for ax.scatter.

    ax.scatter accepts 'edgecolors' (plural) and 'linewidths' (plural).
    Pass both singular and plural so the call works regardless of
    matplotlib version drift.
    """
    kw = {
        "edgecolors": EDGE_COLOR,
        "linewidths": EDGE_LINEWIDTH,
    }
    kw.update(overrides)
    return kw


def hexbin_kw(**overrides):
    """Return edge kwargs for ax.hexbin.

    hexbin uses 'edgecolors' / 'linewidths' like PatchCollection.
    """
    kw = {
        "edgecolors": EDGE_COLOR,
        "linewidths": EDGE_LINEWIDTH,
    }
    kw.update(overrides)
    return kw


def bar_kw(**overrides):
    """Return edge kwargs for ax.bar / ax.barh."""
    kw = dict(EDGE_KWARGS)
    kw.update(overrides)
    return kw


def violin_kw(**overrides):
    """Return edge kwargs for ax.violinplot bodies.

    violinplot returns a dict of collections; users must call
    apply_edges() on body['bodies'] after construction.
    """
    kw = dict(EDGE_KWARGS)
    kw.update(overrides)
    return kw


def apply_edges(artist, linewidth=EDGE_LINEWIDTH, color=EDGE_COLOR):
    """Apply the 0.1pt black edge to an already-drawn artist.

    Works on PathCollection (scatter), PolyCollection (hexbin, violin),
    Rectangle (bar patches), and plain lists of artists.
    """
    if artist is None:
        return
    if isinstance(artist, (list, tuple)):
        for a in artist:
            apply_edges(a, linewidth=linewidth, color=color)
        return
    # PathCollection / PolyCollection (scatter, hexbin, violin bodies)
    if hasattr(artist, "set_edgecolor") and hasattr(artist, "set_linewidth"):
        artist.set_edgecolor(color)
        artist.set_linewidth(linewidth)
        return
    # BarContainer from ax.bar
    if hasattr(artist, "patches"):
        for p in artist.patches:
            p.set_edgecolor(color)
            p.set_linewidth(linewidth)
        return


def style_existing_axes(ax, linewidth=EDGE_LINEWIDTH, color=EDGE_COLOR):
    """Retrofit 0.1pt black edges onto every data collection on an axes.

    Safe to call after all plotting is done. Does NOT touch spines,
    tick marks, legend frames, or text elements.
    """
    for coll in list(ax.collections):
        if hasattr(coll, "set_edgecolor"):
            coll.set_edgecolor(color)
            coll.set_linewidth(linewidth)
    for patch in list(ax.patches):
        if hasattr(patch, "set_edgecolor"):
            patch.set_edgecolor(color)
            patch.set_linewidth(linewidth)


__all__ = [
    "EDGE_LINEWIDTH",
    "EDGE_COLOR",
    "EDGE_KWARGS",
    "scatter_kw",
    "hexbin_kw",
    "bar_kw",
    "violin_kw",
    "apply_edges",
    "style_existing_axes",
]
