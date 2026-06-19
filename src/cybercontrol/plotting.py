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
    """Apply a consistent lightweight grid used by guide figures."""

    ax.grid(alpha=alpha)
    return ax


def apply_clean_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None,
                     title: str | None = None, alpha: float = 0.25) -> None:
    """Apply labels, title, and the shared readable grid style."""

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=alpha, linewidth=0.8)


def plot_time_series(ax, t, y, label: str, *, linewidth: float = 2.0, **kwargs) -> None:
    """Plot one time-indexed curve with the repository default linewidth."""

    kwargs.setdefault("linewidth", linewidth)
    ax.plot(t, y, label=label, **kwargs)
