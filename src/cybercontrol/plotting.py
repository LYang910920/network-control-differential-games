"""Shared plotting helpers for guide figures and publication-style outputs."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterable, Mapping


IEEE_SINGLE_COLUMN_IN = 3.5
IEEE_DOUBLE_COLUMN_IN = 7.16
PUBLICATION_DPI = 600
GUIDE_DPI = 220

PUBLICATION_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
PUBLICATION_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
PUBLICATION_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def figure_size(width: str = "single", ratio: float = 0.72) -> tuple[float, float]:
    """Return an IEEE-friendly figure size in inches.

    Parameters
    ----------
    width:
        ``"single"`` for one-column figures, ``"double"`` for full-width
        figures, or a numeric width in inches.
    ratio:
        Height divided by width.  The default leaves room for axis labels while
        keeping dense plots compact.
    """

    if width == "single":
        w = IEEE_SINGLE_COLUMN_IN
    elif width == "double":
        w = IEEE_DOUBLE_COLUMN_IN
    else:
        w = float(width)
    return w, w * ratio


@contextmanager
def publication_style():
    """Matplotlib context for IEEE Transactions-oriented experiment figures."""

    import matplotlib as mpl

    with mpl.rc_context(
        {
            "figure.dpi": 120,
            "savefig.dpi": PUBLICATION_DPI,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.4,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.prop_cycle": mpl.cycler(color=PUBLICATION_COLORS),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    ):
        yield


@contextmanager
def guide_style():
    """Matplotlib context for larger tutorial diagrams and README figures."""

    import matplotlib as mpl

    with mpl.rc_context(
        {
            "figure.dpi": 120,
            "savefig.dpi": GUIDE_DPI,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    ):
        yield


def style_axis(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = False,
    grid: bool = True,
    alpha: float = 0.25,
) -> None:
    """Apply common labels, optional title, grid, and legend."""

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        panel_label(ax, title)
    if grid:
        ax.grid(True, alpha=alpha)
    if legend:
        ax.legend(loc="best", frameon=False)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    """Place a small panel label such as ``(a)`` in axes coordinates."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize="medium",
        fontweight="bold",
    )


def _normalise_formats(formats: Iterable[str]) -> tuple[str, ...]:
    out = []
    for fmt in formats:
        clean = fmt.lower().lstrip(".")
        if clean and clean not in out:
            out.append(clean)
    return tuple(out)


def _write_manifest(
    stem: Path,
    *,
    fig,
    formats: tuple[str, ...],
    dpi: int,
    style: str,
    metadata: Mapping[str, object] | None,
) -> Path:
    width_in, height_in = fig.get_size_inches()
    payload = {
        "style": style,
        "formats": list(formats),
        "dpi": dpi,
        "width_in": float(width_in),
        "height_in": float(height_in),
        "files": [stem.with_suffix(f".{fmt}").name for fmt in formats],
        "primary_file_bytes": stem.with_suffix(f".{formats[0]}").stat().st_size,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    manifest = stem.with_suffix(".figure.json")
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def save_publication_figure(
    fig,
    stem: str | Path,
    *,
    formats: Iterable[str] = ("pdf", "png"),
    dpi: int = PUBLICATION_DPI,
    metadata: Mapping[str, object] | None = None,
) -> list[Path]:
    """Save a figure as vector PDF plus high-resolution raster output.

    ``stem`` may include an extension.  The returned paths include each figure
    file followed by a small JSON manifest that records style, DPI, and source
    metadata.
    """

    stem_path = Path(stem)
    if stem_path.suffix:
        stem_path = stem_path.with_suffix("")
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    clean_formats = _normalise_formats(formats)
    written = []
    for fmt in clean_formats:
        out = stem_path.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(out, dpi=dpi)
        else:
            fig.savefig(out)
        written.append(out)
    written.append(
        _write_manifest(
            stem_path,
            fig=fig,
            formats=clean_formats,
            dpi=dpi,
            style="publication",
            metadata=metadata,
        )
    )
    return written


def save_guide_figure(
    fig,
    stem: str | Path,
    *,
    formats: Iterable[str] = ("png",),
    dpi: int = GUIDE_DPI,
    metadata: Mapping[str, object] | None = None,
) -> list[Path]:
    """Save a larger tutorial/README figure and a JSON manifest."""

    stem_path = Path(stem)
    if stem_path.suffix:
        stem_path = stem_path.with_suffix("")
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    clean_formats = _normalise_formats(formats)
    written = []
    for fmt in clean_formats:
        out = stem_path.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(out, dpi=dpi)
        else:
            fig.savefig(out)
        written.append(out)
    written.append(
        _write_manifest(
            stem_path,
            fig=fig,
            formats=clean_formats,
            dpi=dpi,
            style="guide",
            metadata=metadata,
        )
    )
    return written


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

    style_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, alpha=alpha)


def plot_time_series(ax, t, y, label: str, *, linewidth: float = 2.0, **kwargs) -> None:
    """Plot one time-indexed curve with the repository default linewidth."""

    kwargs.setdefault("linewidth", linewidth)
    ax.plot(t, y, label=label, **kwargs)
