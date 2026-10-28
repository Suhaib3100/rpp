"""Slide-building toolkit for the SAGE-HE presentation.

python-pptx has no concept of a design system, so this module supplies one: a
fixed grid, a type scale, a colour vocabulary shared with the figures
(src/viz/theme.py), and a set of layout primitives. Slides are composed from
those primitives rather than positioned ad hoc, which is what keeps 30+ slides
visually consistent.

Grid
----
16:9 at 13.333 x 7.5 in. A 12-column grid with a 0.62 in margin and 0.18 in
gutter; ``col_x`` and ``col_w`` convert column spans to inches so that every
element lands on the same rhythm.

Type scale
----------
    display 40 / title 28 / subtitle 17 / heading 15 / body 13 / small 11 / micro 9
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from src.viz import theme as T

# ------------------------------------------------------------------ geometry
SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN = 0.62
GUTTER = 0.18
NCOLS = 12
COL_W = (SLIDE_W - 2 * MARGIN - GUTTER * (NCOLS - 1)) / NCOLS

TITLE_Y = 0.54
SUBTITLE_Y = 1.16
BODY_Y = 1.78
FOOTER_Y = 6.98


def col_x(i: int) -> float:
    """Left edge of column i (0-indexed)."""
    return MARGIN + i * (COL_W + GUTTER)


def col_w(n: int) -> float:
    """Width spanning n columns."""
    return n * COL_W + (n - 1) * GUTTER


# -------------------------------------------------------------------- colour
def rgb(hex_str: str) -> RGBColor:
    return RGBColor.from_string(hex_str.lstrip("#").upper())


PAGE = rgb(T.PAGE)
SURFACE = rgb(T.SURFACE)
SURFACE_RAISED = rgb(T.SURFACE_RAISED)
INK = rgb(T.INK)
INK_SECONDARY = rgb(T.INK_SECONDARY)
INK_MUTED = rgb(T.INK_MUTED)
ACCENT = rgb(T.ACCENT)
ACCENT_BRIGHT = rgb(T.ACCENT_BRIGHT)
BORDER = rgb(T.BORDER)
HAIRLINE = rgb(T.HAIRLINE)
GOOD = rgb(T.GOOD)
WARNING = rgb(T.WARNING)
CRITICAL = rgb(T.CRITICAL)
SERIES = [rgb(c) for c in T.SERIES]

FONT = "Inter"
MONO = "Source Code Pro"

AUTHOR = "SUHAIB S Z"
PROJECT = "SAGE-HE"


@dataclass
class Deck:
    """Thin wrapper over a python-pptx Presentation with our layouts."""

    prs: Presentation
    slide_number: int = 0
    expected_total: int = 0
    content_top: float = 0.0

    @classmethod
    def create(cls) -> "Deck":
        prs = Presentation()
        prs.slide_width = Inches(SLIDE_W)
        prs.slide_height = Inches(SLIDE_H)
        return cls(prs=prs)

    # ------------------------------------------------------------ primitives
    def _blank(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _bg(self, slide, color=PAGE) -> None:
        """A subtle vertical gradient rather than a flat fill.

        Flat #080D18 across 13 inches of projector reads as a dead rectangle.
        A barely-there lift toward the navy surface colour gives the slide
        depth without becoming a "decorative gradient" — the stops are only a
        few RGB steps apart.
        """
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                    Inches(SLIDE_W), Inches(SLIDE_H))
        fill = bg.fill
        fill.gradient()
        fill.gradient_angle = 90.0           # top -> bottom
        stops = fill.gradient_stops
        stops[0].color.rgb = rgb("#0C1322")
        stops[0].position = 0.0
        stops[1].color.rgb = color
        stops[1].position = 1.0
        bg.line.fill.background()
        bg.shadow.inherit = False
        slide.shapes._spTree.remove(bg._element)
        slide.shapes._spTree.insert(2, bg._element)

    def text(self, slide, x, y, w, h, text, size=13, color=INK, bold=False,
             align=PP_ALIGN.LEFT, font=FONT, italic=False, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.18, space_after=4):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        lines = text.split("\n") if isinstance(text, str) else list(text)
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = line_spacing
            p.space_after = Pt(space_after)
            r = p.add_run()
            r.text = line
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.name = font
            r.font.color.rgb = color
        return box

    def bullets(self, slide, x, y, w, h, items, size=13.5, color=INK_SECONDARY,
                bullet_color=ACCENT, gap=9, line_spacing=1.22):
        """Bulleted list with a coloured marker glyph and hanging indent."""
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.line_spacing = line_spacing
            p.space_after = Pt(gap)
            marker = p.add_run()
            marker.text = "— "
            marker.font.size = Pt(size)
            marker.font.name = FONT
            marker.font.bold = True
            marker.font.color.rgb = bullet_color
            # an item may be (lead, rest) to emphasise the opening phrase
            if isinstance(item, (tuple, list)):
                lead, rest = item
                r1 = p.add_run()
                r1.text = lead
                r1.font.size = Pt(size)
                r1.font.name = FONT
                r1.font.bold = True
                r1.font.color.rgb = INK
                r2 = p.add_run()
                r2.text = rest
                r2.font.size = Pt(size)
                r2.font.name = FONT
                r2.font.color.rgb = color
            else:
                r = p.add_run()
                r.text = item
                r.font.size = Pt(size)
                r.font.name = FONT
                r.font.color.rgb = color
        return box

    def card(self, slide, x, y, w, h, fill=SURFACE, line=BORDER, radius=True):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
        shape.shadow.inherit = False
        if radius:
            try:
                shape.adjustments[0] = 0.06
            except (IndexError, KeyError):
                pass
        return shape

    def rule(self, slide, x, y, w, color=ACCENT, h=0.035):
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                     Inches(w), Inches(h))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        bar.shadow.inherit = False
        return bar

    # Slides are rendered at most ~1920 px wide, so embedding a 2700 px figure
    # just inflates the file. Resampling to the placed size at this density
    # keeps the deck visually identical while cutting it from ~12 MB to a few —
    # which also matters because some importers (Keynote) reject a partially
    # transferred file with a misleading "file format is invalid".
    EMBED_DPI = 150

    def _optimised(self, path, width_in: float) -> str:
        """Return a path to the image resampled for its placed width."""
        from PIL import Image as _Image
        path = Path(path)
        target_w = max(1, int(round(width_in * self.EMBED_DPI)))
        with _Image.open(str(path)) as im:
            if im.width <= target_w:
                return str(path)
            ratio = target_w / im.width
            out = im.convert("RGB").resize(
                (target_w, max(1, int(round(im.height * ratio)))),
                _Image.LANCZOS)
            cache = Path(".cache_slide_images")
            cache.mkdir(exist_ok=True)
            dest = cache / f"{path.stem}_{target_w}.png"
            if not dest.exists():
                out.save(dest, format="PNG", optimize=True)
            return str(dest)

    def image(self, slide, path, x, y, w=None, h=None):
        kw = {}
        if w:
            kw["width"] = Inches(w)
        if h:
            kw["height"] = Inches(h)
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)

    def image_fit(self, slide, path, x, y, max_w, max_h, centre=True):
        """Place an image scaled to fit inside (max_w x max_h), preserving aspect.

        Sizing by width alone is the easy way to run a wide diagram off the
        bottom of the slide; this picks whichever dimension binds and optionally
        centres the result in the box.
        """
        from PIL import Image as _Image
        with _Image.open(str(path)) as im:
            iw, ih = im.size
        scale = min(max_w / iw, max_h / ih)
        w, h = iw * scale, ih * scale
        px = x + (max_w - w) / 2 if centre else x
        py = y + (max_h - h) / 2 if centre else y
        src = self._optimised(path, w)
        return slide.shapes.add_picture(src, Inches(px), Inches(py),
                                        width=Inches(w), height=Inches(h))

    def notes(self, slide, text: str) -> None:
        slide.notes_slide.notes_text_frame.text = text

    # --------------------------------------------------------------- layouts
    def title_slide(self, title, subtitle, author=AUTHOR, role="Research, Architecture & Implementation"):
        s = self._blank()
        self._bg(s)
        # accent field on the left edge
        self.rule(s, 0, 0, 0.14, ACCENT, h=SLIDE_H)
        self.text(s, MARGIN + 0.25, 1.62, col_w(10), 1.1, PROJECT,
                  size=62, color=INK, bold=True)
        self.rule(s, MARGIN + 0.25, 2.92, 1.5, ACCENT)
        self.text(s, MARGIN + 0.25, 3.18, col_w(9), 1.2, title,
                  size=19, color=INK_SECONDARY, line_spacing=1.3)
        self.text(s, MARGIN + 0.25, 4.82, col_w(8), 0.3, role.upper(),
                  size=10.5, color=INK_MUTED, bold=True)
        self.text(s, MARGIN + 0.25, 5.16, col_w(8), 0.62, author,
                  size=31, color=INK, bold=True)
        self.text(s, MARGIN + 0.25, FOOTER_Y - 0.06, col_w(10), 0.3, subtitle,
                  size=10.5, color=INK_MUTED)
        self.slide_number += 1
        return s

    def section_slide(self, number, title, subtitle=""):
        s = self._blank()
        self._bg(s)
        self.text(s, MARGIN, 2.78, col_w(2), 1.0, f"{number:02d}",
                  size=58, color=ACCENT_BRIGHT, bold=True)
        self.rule(s, MARGIN, 3.92, 1.2, ACCENT)
        self.text(s, MARGIN, 4.18, col_w(9), 0.7, title, size=30, color=INK, bold=True)
        if subtitle:
            self.text(s, MARGIN, 5.02, col_w(8), 0.6, subtitle,
                      size=13.5, color=INK_MUTED)
        self.slide_number += 1
        return s

    def content_slide(self, title, subtitle="", footer_note=""):
        """Standard content slide: title, optional deck, accent rule, footer."""
        s = self._blank()
        self._bg(s)
        self.text(s, MARGIN, TITLE_Y, col_w(11), 0.62, title,
                  size=27, color=INK, bold=True)
        y = TITLE_Y + 0.66
        rule_y = 1.62
        if subtitle:
            # A subtitle long enough to wrap would run into the accent rule and
            # read as a strikethrough. Measure it and push the rule down a line
            # instead of silently overlapping.
            wrapped = len(subtitle) > 95
            self.text(s, MARGIN, y, col_w(11), 0.42, subtitle,
                      size=12.5 if wrapped else 13, color=INK_MUTED)
            if wrapped:
                rule_y += 0.26
        self.rule(s, MARGIN, rule_y, 0.9, ACCENT, h=0.03)
        self.content_top = rule_y + 0.16
        self.slide_number += 1
        self._footer(s, footer_note)
        return s

    def _footer(self, slide, note="", total: int | None = None):
        # Hairline above the footer, then a thin progress rail: the audience
        # can see how far through the deck they are without a page-count stare.
        self.rule(slide, MARGIN, FOOTER_Y - 0.17, col_w(12), HAIRLINE, h=0.008)
        total = total or self.expected_total
        if total:
            frac = min(1.0, self.slide_number / total)
            self.rule(slide, MARGIN, FOOTER_Y - 0.17, col_w(12) * frac, ACCENT, h=0.008)
        self.text(slide, MARGIN, FOOTER_Y, col_w(8), 0.26,
                  note or f"{PROJECT}  ·  {AUTHOR}",
                  size=9, color=INK_MUTED)
        self.text(slide, SLIDE_W - MARGIN - 0.6, FOOTER_Y, 0.6, 0.26,
                  str(self.slide_number), size=9, color=INK_MUTED,
                  align=PP_ALIGN.RIGHT)

    # ----------------------------------------------------------- composites
    def stat_tiles(self, slide, y, tiles, height=1.32, cols=None):
        """A row of headline figures. tiles = [(value, label, color?), ...]"""
        n = len(tiles)
        cols = cols or n
        span = NCOLS // cols
        for i, tile in enumerate(tiles):
            value, label = tile[0], tile[1]
            color = tile[2] if len(tile) > 2 else ACCENT_BRIGHT
            x = col_x((i % cols) * span)
            w = col_w(span)
            yy = y + (i // cols) * (height + 0.16)
            self.card(slide, x, yy, w, height, SURFACE, BORDER)
            self.text(slide, x + 0.22, yy + 0.17, w - 0.44, 0.6, str(value),
                      size=28, color=color, bold=True)
            self.text(slide, x + 0.22, yy + 0.82, w - 0.44, 0.42, label,
                      size=10.5, color=INK_MUTED, line_spacing=1.08)

    def table(self, slide, x, y, w, headers, rows, col_widths=None,
              size=11, header_size=10.5, row_h=0.34, highlight_row=None,
              highlight_color=None):
        """A light, rule-based table - no heavy borders, no zebra fill."""
        n = len(headers)
        col_widths = col_widths or [w / n] * n
        # header
        cx = x
        for hdr, cw in zip(headers, col_widths):
            self.text(slide, cx, y, cw, 0.3, hdr, size=header_size,
                      color=INK_MUTED, bold=True)
            cx += cw
        self.rule(slide, x, y + 0.32, w, BORDER, h=0.012)
        # rows
        for ri, row in enumerate(rows):
            ry = y + 0.42 + ri * row_h
            is_hl = highlight_row is not None and ri == highlight_row
            if is_hl:
                self.card(slide, x - 0.1, ry - 0.055, w + 0.2, row_h - 0.02,
                          SURFACE_RAISED, SURFACE_RAISED)
            cx = x
            for ci, (cell, cw) in enumerate(zip(row, col_widths)):
                col = INK if (is_hl or ci == 0) else INK_SECONDARY
                if is_hl and highlight_color:
                    col = highlight_color if ci > 0 else INK
                self.text(slide, cx, ry, cw, row_h, str(cell), size=size,
                          color=col, bold=is_hl,
                          font=MONO if ci > 0 else FONT)
                cx += cw
        return y + 0.42 + len(rows) * row_h

    def code_block(self, slide, x, y, w, h, lines, size=10.5):
        self.card(slide, x, y, w, h, SURFACE, BORDER)
        self.text(slide, x + 0.2, y + 0.16, w - 0.4, h - 0.32, lines,
                  size=size, color=ACCENT_BRIGHT, font=MONO, line_spacing=1.3)

    def callout(self, slide, x, y, w, h, title, body, color=ACCENT):
        self.card(slide, x, y, w, h, SURFACE, BORDER)
        self.rule(slide, x, y, 0.045, color, h=h)
        self.text(slide, x + 0.24, y + 0.15, w - 0.46, 0.3, title,
                  size=11.5, color=color, bold=True)
        self.text(slide, x + 0.24, y + 0.52, w - 0.46, h - 0.68, body,
                  size=11.5, color=INK_SECONDARY, line_spacing=1.24)

    # ------------------------------------------------------------- motion
    def animate(self, slide, build_shapes=None, transition: bool = True,
                dur_ms: int = 350) -> None:
        """Give a slide a fade transition and an optional staggered build."""
        from src.viz.motion import add_fade_in_build, add_fade_transition
        if transition:
            add_fade_transition(slide)
        if build_shapes:
            add_fade_in_build(slide, build_shapes, dur_ms=dur_ms)

    def animate_all(self, transition_ms: int = 400, build: bool = True,
                    max_build_shapes: int = 6) -> None:
        """Fade transition on every slide, plus a staggered build of its content.

        Chrome — the background, title, subtitle, accent rule and footer — is
        deliberately excluded and appears immediately; animating it would make
        the deck feel slow rather than considered. Only the content below the
        rule builds in, capped at ``max_build_shapes`` so a dense slide does
        not turn into a sequence the presenter has to wait through.
        """
        from src.viz.motion import add_fade_in_build, add_fade_transition
        from pptx.util import Inches as _In
        chrome_below = _In(1.95)          # anything above this is chrome
        for sl in self.prs.slides:
            add_fade_transition(sl, duration_ms=transition_ms)
            if not build:
                continue
            content = [sh for sh in sl.shapes
                       if sh.top is not None and sh.top >= chrome_below
                       and sh.width is not None
                       and not (sh.width >= _In(SLIDE_W) and sh.height >= _In(SLIDE_H))]
            # keep document order, cap the count, and skip the footer strip
            content = [sh for sh in content if sh.top < _In(FOOTER_Y - 0.25)]
            if content:
                add_fade_in_build(sl, content[:max_build_shapes])

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(path))
        return path
