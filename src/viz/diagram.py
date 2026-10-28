"""A small declarative toolkit for architecture diagrams.

Diagrams are drawn in matplotlib (not Graphviz) because every figure in this
project needs to share one visual language with the charts and slides, and
because tensor-shape annotations want precise manual placement. Output is
vector SVG + PDF, with PNG for embedding.

Coordinate space is a plain 0..100 grid in both axes; helpers take centre
coordinates and sizes in those units.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.viz import theme as T

# Node roles -> (fill, edge, title colour). Roles carry meaning:
# "data" = tensors in flight, "compute" = learned modules, "crypto" = anything
# touching ciphertext, "frozen" = pretrained/non-learned, "out" = outputs.
ROLES: dict[str, tuple[str, str, str]] = {
    "data":    ("#16213C", "#2B3C5C", T.INK),
    "compute": ("#15315C", T.ACCENT, T.INK),
    "crypto":  ("#2A1F3D", "#8B5CF6", "#DDD6FE"),
    "frozen":  ("#1A2436", "#3C4C68", T.INK_SECONDARY),
    "out":     ("#123A2E", T.SERIES[2], "#D1FAE5"),
    "warn":    ("#3A2A12", T.WARNING, "#FDE68A"),
}


@dataclass
class Node:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> tuple[float, float]:
        return (self.x - self.w / 2, self.y)

    @property
    def right(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y)

    @property
    def top(self) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.x, self.y - self.h / 2)


def canvas(width: float = 16, height: float = 9, title: str = "",
           subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    """A blank diagram canvas with the project chrome."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(T.PAGE)
    ax.set_facecolor(T.PAGE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if title:
        ax.text(2, 97, title, fontsize=19, fontweight="bold", color=T.INK,
                va="top", ha="left")
    if subtitle:
        ax.text(2, 92.4, subtitle, fontsize=11.5, color=T.INK_MUTED,
                va="top", ha="left")
    return fig, ax


def node(ax, x: float, y: float, w: float, h: float, label: str,
         sub: str = "", role: str = "compute", fontsize: float = 11,
         subsize: float = 9) -> Node:
    """A rounded box with a title and optional second line."""
    fill, edge, tc = ROLES[role]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=1.1",
        facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2,
    ))
    if sub:
        # Offsets scale with line count: a two-line title and a two-line
        # subtitle collide at fixed offsets.
        n_title = label.count("\n") + 1
        n_sub = sub.count("\n") + 1
        off_t = h * (0.12 + 0.065 * (n_sub - 1))
        off_s = h * (0.20 + 0.105 * (n_title - 1))
        ax.text(x, y + off_t, label, ha="center", va="center", color=tc,
                fontsize=fontsize, fontweight="semibold", zorder=3)
        ax.text(x, y - off_s, sub, ha="center", va="center",
                color=T.INK_MUTED, fontsize=subsize, zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center", color=tc,
                fontsize=fontsize, fontweight="semibold", zorder=3)
    return Node(x, y, w, h)


def arrow(ax, start: tuple[float, float], end: tuple[float, float],
          label: str = "", color: str | None = None, style: str = "-|>",
          dashed: bool = False, rad: float = 0.0, labeloff: float = 1.8,
          fontsize: float = 8.5) -> None:
    """A connector, optionally curved and labelled with a tensor shape."""
    color = color or T.ACCENT
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=1.5, color=color, zorder=1,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=3,
    ))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + labeloff, label, ha="center", va="bottom",
                color=T.INK_MUTED, fontsize=fontsize, zorder=3)


def region(ax, x: float, y: float, w: float, h: float, label: str,
           color: str | None = None, dashed: bool = True,
           align: str = "left") -> None:
    """A labelled enclosure, e.g. the client/server trust boundary."""
    color = color or T.BORDER
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=1.4",
        facecolor="none", edgecolor=color, linewidth=1.5,
        linestyle=(0, (6, 4)) if dashed else "-", zorder=0,
    ))
    lx = x - w / 2 + 1.6 if align == "left" else x
    ax.text(lx, y + h / 2 - 2.2, label, ha=align, va="center",
            color=color, fontsize=10.5, fontweight="bold", zorder=1)


def band(ax, x: float, y: float, w: float, h: float, color: str,
         alpha: float = 0.10) -> None:
    """A soft background wash to group related stages."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color,
                           alpha=alpha, edgecolor="none", zorder=0))


def caption(ax, text: str, y: float = 3.0, color: str | None = None) -> None:
    ax.text(50, y, text, ha="center", va="center",
            color=color or T.INK_MUTED, fontsize=9.5, style="italic")


def tensor(ax, x: float, y: float, text: str, color: str | None = None) -> None:
    """A monospace tensor-shape annotation."""
    ax.text(x, y, text, ha="center", va="center", color=color or T.ACCENT_BRIGHT,
            fontsize=8.5, family="monospace", zorder=3)


def save(fig, path, formats: tuple[str, ...] = ("svg", "png")) -> list:
    """Write the figure in vector + raster form."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt, facecolor=T.PAGE, bbox_inches="tight",
                    dpi=200 if fmt == "png" else None)
        out.append(p)
    plt.close(fig)
    return out
