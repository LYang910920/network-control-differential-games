"""Small plotting helpers shared by scripts and README figure generators."""

from __future__ import annotations


def add_box(ax, xy, text, width=1.8, height=0.55, fc="#f7f7f7", ec="#333333", fontsize=8.5):
    """Draw a labeled rectangle and return the Matplotlib patch."""

    import matplotlib.pyplot as plt

    rect = plt.Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return rect


def add_arrow(ax, start, end, color="#333333", linewidth=1.4):
    """Draw a simple arrow between two points."""

    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": linewidth, "color": color})


def clean_axes(ax, alpha: float = 0.25):
    """Apply a consistent lightweight grid used by tutorial figures."""

    ax.grid(alpha=alpha)
    return ax
