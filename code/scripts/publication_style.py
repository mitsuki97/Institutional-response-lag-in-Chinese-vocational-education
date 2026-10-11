"""Shared publication styling for manuscript figures.

This module changes presentation only. It does not transform data, estimates,
panel structure or figure captions.
"""

from __future__ import annotations

import matplotlib as mpl
import seaborn as sns


# Okabe-Ito colour-blind-safe palette. Semantic assignments are fixed in the
# figure scripts; these tokens are not to be remapped between figures.
NHB_COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "blue": "#0072B2",
    "teal": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "violet": "#CC79A7",
    "grey": "#999999",
    "light_grey": "#D9D9D9",
}

# High-contrast scientific palette used for the revised manuscript figures.
# It follows the clean information-visualization style common in Nie's
# computer-vision papers: deep blue as the anchor, with restrained accents.
NIE_COLORS = {
    "ink": "#1F2933",
    "blue": "#245A8D",       # deep anchor blue
    "sky_blue": "#5E9BC2",   # medium blue
    "orange": "#D28B70",     # restrained terracotta accent
    "teal": "#5B858D",       # greyed blue-teal accent
    "violet": "#7898B2",     # desaturated blue for Q4
    "grey": "#A9C1D1",       # cool blue-grey, not neutral grey
    "light_grey": "#EAF1F5",
}

NHB_MARKERS = ("o", "s", "D", "^", "v", "P", "X")
NHB_HATCHES = ("", "///", "\\\\", "xx", "--", "..", "++")


def apply_nhb_style() -> None:
    """Apply a compact, consistent style for Nature-format manuscript figures."""
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.dpi": 180,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": "#D7DEE5",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
        }
    )
