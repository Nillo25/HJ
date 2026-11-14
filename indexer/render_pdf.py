#!/usr/bin/env python3
"""Vector PDF page -> standalone SVG (lines, rectangles, curves, text).

    python render_pdf.py <file.pdf> <out.svg> [--page 1]

Coordinates are kept in PDF points with Y flipped, so a stored bbox [x0,y0,x1,y1]
(PDF space, Y up) maps to the SVG as (x0, H - y1, x1 - x0, y1 - y0). Embedded
raster images are not drawn. The page size is written to the SVG viewBox.
"""
import argparse
import sys
from xml.sax.saxutils import escape

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTLine, LTRect, LTTextLine, LTChar


def _col(c, default="#000"):
    """pdfminer colour (gray / rgb / cmyk tuple or number) -> css colour."""
    if c is None:
        return default
    if isinstance(c, (int, float)):
        c = (c,)
    try:
        c = tuple(float(x) for x in c)
    except (TypeError, ValueError):
        return default
    if len(c) == 1:
        v = int(max(0, min(1, c[0])) * 255)
        return f"#{v:02x}{v:02x}{v:02x}"
    if len(c) == 3:
        r, g, b = (int(max(0, min(1, x)) * 255) for x in c)
        return f"#{r:02x}{g:02x}{b:02x}"
    if len(c) == 4:
        C, M, Y, K = c
        r, g, b = ((1 - min(1, x + K)) * 255 for x in (C, M, Y))
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return default


def render(pdf, page_no=1):
    page = None
    for i, pg in enumerate(extract_pages(pdf), 1):
        if i == page_no:
            page = pg
            break
    if page is None:
        raise ValueError(f"page {page_no} not found")
    _, _, W, H = page.bbox

    parts = []            # (style key, d) in drawing order; consecutive equal styles are merged
    texts = []

    def flat(o):
        for c in o:
            if isinstance(c, (LTLine, LTRect, LTCurve)):
                yield c
            elif isinstance(c, LTTextLine):
                yield c
            elif hasattr(c, "__iter__"):
                yield from flat(c)

    for o in flat(page):
        if isinstance(o, LTTextLine):
            t = o.get_text().strip()
            if not t:
                continue
            ch = next((c for c in o if isinstance(c, LTChar)), None)
            size = ch.size if ch else max(o.height, 4)
            texts.append((o.x0, H - o.y0, o.x1 - o.x0, size, t))
            continue
        pts = list(o.pts) if getattr(o, "pts", None) else []
        if len(pts) < 2:
            continue
        d = "M" + "L".join(f"{x:.1f} {H - y:.1f}" for x, y in pts)
        if isinstance(o, (LTRect, LTCurve)) and not isinstance(o, LTLine) and (o.fill or len(pts) > 2):
            d += "Z"
        fill = _col(o.non_stroking_color, "none") if getattr(o, "fill", False) else "none"
        stroke = _col(o.stroking_color) if (getattr(o, "stroke", True) or isinstance(o, LTLine)) else "none"
        lw = max(getattr(o, "linewidth", 0) or 0, 0.3)
        key = (fill, stroke, round(lw, 1))
        if parts and parts[-1][0] == key:
            parts[-1][1].append(d)
        else:
            parts.append((key, [d]))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}" '
           f'width="{W:.0f}" height="{H:.0f}">',
           f'<rect width="{W:.1f}" height="{H:.1f}" fill="#fff"/>']
    for (fill, stroke, lw), ds in parts:
        out.append(f'<path d="{" ".join(ds)}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
    out.append('<g font-family="Arial, Helvetica, sans-serif" fill="#111">')
    for x, y, w, size, t in texts:
        out.append(f'<text x="{x:.1f}" y="{y - size * 0.2:.1f}" font-size="{size:.1f}" '
                   f'textLength="{max(w, 1):.1f}" lengthAdjust="spacingAndGlyphs">{escape(t)}</text>')
    out.append("</g></svg>")
    return "\n".join(out), (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--page", type=int, default=1)
    a = ap.parse_args()
    svg, (w, h) = render(a.pdf, a.page)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"{w:.0f}x{h:.0f} pt, {len(svg) // 1024} KB")


if __name__ == "__main__":
    sys.exit(main())
