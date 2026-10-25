"""Shared matplotlib style so every figure in the paper looks consistent."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (colour-blind validated). Each method keeps the same
# colour in every figure, so readers can track it across the paper.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLORS = {
    "Random": "#8a8984",
    "Keyword": PALETTE[3],
    "Jaccard (tags)": PALETTE[1],
    "TF-IDF": PALETTE[2],
    "LSA": PALETTE[6],
    "Hybrid": PALETTE[0],
}
MARKERS = {"Random": "x", "Keyword": "s", "Jaccard (tags)": "^", "TF-IDF": "o", "LSA": "D", "Hybrid": "*"}


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#52514e",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )
