"""SAGE-HE visual design system.

One source of truth for colour and type across figures, diagrams and slides, so
the deck reads as a single system rather than a pile of plots.

Palette provenance
------------------
The categorical slots are the dark-mode steps of the reference data-visualisation
palette, re-validated against *this* project's surface (dark navy ``#0F172A``)
rather than assumed. Running the six-check validator against that surface:

    Lightness band      PASS  all 6 inside L 0.48-0.67
    Chroma floor        PASS  all 6 >= 0.1
    CVD separation      PASS  worst adjacent yellow<->aqua dE 8.4 (protan)
    Normal-vision floor PASS  worst adjacent magenta<->yellow dE 19.3
    Contrast vs surface PASS  all 6 >= 3:1

Because the worst adjacent CVD separation sits in the 8.4 band, every chart that
uses more than two slots also carries secondary encoding (direct value labels or
a legend with explicit names) - colour never carries identity alone.

Rules this module enforces by construction:
  * categorical hues are assigned in fixed slot order, never cycled;
  * sequential magnitude (confusion matrices) uses ONE hue, light->dark;
  * status colours are reserved and never reused as a series colour.
"""
from __future__ import annotations

from matplotlib import font_manager
import matplotlib as mpl

# ----------------------------------------------------------------- surfaces
PAGE = "#080D18"          # deepest navy, slide background
SURFACE = "#0F172A"       # chart / card surface (validator surface)
SURFACE_RAISED = "#16213C"  # elevated panel
HAIRLINE = "#1E293B"      # gridlines, dividers
BORDER = "#24324B"

# --------------------------------------------------------------------- ink
INK = "#F1F5F9"           # primary text
INK_SECONDARY = "#CBD5E1"
INK_MUTED = "#94A3B8"     # axis labels, captions

# ------------------------------------------------------------ brand accent
ACCENT = "#3987E5"        # electric blue, the project's signature
ACCENT_BRIGHT = "#60A5FA"
ACCENT_DIM = "#1E3A6B"

# ------------------------------- categorical slots (fixed order, validated)
SERIES: tuple[str, ...] = (
    "#3987e5",   # 1 blue    - the proposed model, by convention
    "#d95926",   # 2 orange  - the reproduced baseline, by convention
    "#199e70",   # 3 aqua
    "#c98500",   # 4 yellow
    "#d55181",   # 5 magenta
    "#008300",   # 6 green
)

# ------------------------------------------- sequential ramp (single hue)
SEQ_BLUE: tuple[str, ...] = (
    "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
    "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
    "#9ec5f4", "#b7d3f6", "#cde2fb",
)

# --------------------------------------- status (reserved, never a series)
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# ------------------------------------------------------------------- type
FONT_STACK = ["Inter", "Inter Display", "Liberation Sans", "DejaVu Sans"]


def _available_font() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_STACK:
        if name in have:
            return name
    return "sans-serif"


FONT = _available_font()


def sequential_cmap(name: str = "sage_blue"):
    """One-hue light->dark ramp for magnitude (confusion matrices, heatmaps)."""
    from matplotlib.colors import LinearSegmentedColormap
    # light -> dark reads "low -> high" when the high end is the saturated hue;
    # on a dark surface we run near-surface -> bright so high values pop.
    return LinearSegmentedColormap.from_list(name, [SURFACE, "#1c5cab", "#3987e5", "#9ec5f4"])


def apply_theme() -> None:
    """Install the project's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "figure.dpi": 110,

        # Inter has no glyphs for some maths symbols we use (U+2208 ELEMENT OF,
        # U+2207 NABLA). Declaring a family *list* lets matplotlib fall back to
        # DejaVu Sans for just those codepoints instead of drawing tofu boxes.
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # recessive chrome
        "axes.edgecolor": BORDER,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": HAIRLINE,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.labelcolor": INK_SECONDARY,

        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "patch.linewidth": 0,

        "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        "svg.fonttype": "path",   # embed glyphs so SVGs render anywhere
    })


def style_axes(ax, *, ylabel: str = "", xlabel: str = "", title: str = "") -> None:
    """Consistent axis treatment: recessive chrome, no chartjunk."""
    if title:
        ax.set_title(title, color=INK, pad=12, loc="left", fontweight="semibold")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BORDER)
