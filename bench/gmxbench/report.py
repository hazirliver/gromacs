"""Self-contained HTML report (no external resources) from one or more results directories.

Design: every chart is paired with its numbers. Comparisons are tables whose last column is an
inline SVG mark on an axis shared by the whole table (forest plots for speedups with 95% CI,
paired bars for A/B values, strip plots for repeat distributions), so the table *is* the
table-view of the chart. Divergence curves are real line charts with a crosshair tooltip.
Colours: baseline A = categorical slot 1 (blue), candidate B = slot 2 (orange); verdicts use the
status palette and always carry an icon + label. Light/dark via CSS custom properties.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from .results import load_session

ESC = html.escape

# ----------------------------------------------------------------------------------------------
# style & script
# ----------------------------------------------------------------------------------------------
CSS = r"""
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --band: #f0efec;
  --a: #2a78d6; --b: #eb6834; --emph: #2a78d6; --rest: #c3c2b7;
  --good: #0ca30c; --warn: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --good-ink: #006300; --critical-ink: #b42f2f; --warn-ink: #8a5a00; --serious-ink: #a4461d;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --band: #262624;
    --a: #3987e5; --b: #d95926; --emph: #3987e5; --rest: #6b6a64;
    --good-ink: #0ca30c; --critical-ink: #e66767; --warn-ink: #fab219; --serious-ink: #ec835a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --band: #262624;
  --a: #3987e5; --b: #d95926; --emph: #3987e5; --rest: #6b6a64;
  --good-ink: #0ca30c; --critical-ink: #e66767; --warn-ink: #fab219; --serious-ink: #ec835a;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 24px 16px 64px; }
header.top { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 36px 0 8px; padding-top: 8px; border-top: 1px solid var(--grid); }
h3 { font-size: 15px; margin: 22px 0 6px; }
p, li { color: var(--ink-2); }
.sub { color: var(--ink-2); margin: 0; }
.small { font-size: 12px; color: var(--muted); }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; margin: 10px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin: 12px 0; }
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 10px; margin: 12px 0; }
@media (max-width: 640px) { .charts { grid-template-columns: 1fr; } }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12px; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; background: var(--surface); }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--grid); vertical-align: middle; white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; position: sticky; top: 0; background: var(--surface); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.wrap { white-space: normal; min-width: 220px; }
tr[data-tip]:hover td, tr[data-tip]:focus-within td { background: var(--band); }
.badge { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 600; }
.badge .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.st-good { color: var(--good-ink); } .st-good .dot { background: var(--good); }
.st-warn { color: var(--warn-ink); } .st-warn .dot { background: var(--warn); }
.st-serious { color: var(--serious-ink); } .st-serious .dot { background: var(--serious); }
.st-critical { color: var(--critical-ink); } .st-critical .dot { background: var(--critical); }
.st-muted { color: var(--muted); } .st-muted .dot { background: var(--axis); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--ink-2); font-size: 12px; margin: 6px 0; align-items: center; }
.legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -2px; }
.legend .ln { display: inline-block; width: 14px; height: 2px; margin-right: 5px; vertical-align: 3px; }
details { margin: 6px 0; }
details > summary { cursor: pointer; color: var(--ink); font-weight: 600; padding: 4px 0; }
svg text { fill: var(--muted); font-size: 11px; font-family: system-ui, sans-serif; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .ref { stroke: var(--ink-2); stroke-width: 1; }
svg .band { fill: var(--band); }
svg .ci { stroke: var(--ink-2); stroke-width: 2; stroke-linecap: round; }
svg .ring { stroke: var(--surface); stroke-width: 2; }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 7px 9px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
  display: none; max-width: 420px; }
#tip .v { font-weight: 600; } #tip .k { color: var(--ink-2); }
#tip .row { display: flex; gap: 8px; align-items: center; white-space: nowrap; }
#tip .key { display: inline-block; width: 12px; height: 2px; }
.theme { font: inherit; font-size: 12px; background: var(--surface); color: var(--ink); border: 1px solid var(--ring); border-radius: 6px; padding: 3px 8px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; font-size: 12px; }
.kv div:nth-child(odd) { color: var(--muted); }
@media (max-width: 640px) { .tile .value { font-size: 20px; } }
"""

JS = r"""
(function(){
  const tip = document.getElementById('tip');
  function show(ev, rows) {
    tip.replaceChildren();
    for (const r of rows) {
      const d = document.createElement('div'); d.className = 'row';
      if (r.c) { const k = document.createElement('span'); k.className = 'key'; k.style.background = r.c; d.appendChild(k); }
      const v = document.createElement('span'); v.className = 'v'; v.textContent = r.v; d.appendChild(v);
      if (r.k) { const k = document.createElement('span'); k.className = 'k'; k.textContent = r.k; d.appendChild(k); }
      tip.appendChild(d);
    }
    tip.style.display = 'block';
    const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hide(){ tip.style.display = 'none'; }
  document.querySelectorAll('[data-tip]').forEach(el => {
    const rows = JSON.parse(el.getAttribute('data-tip'));
    el.addEventListener('pointermove', ev => show(ev, rows));
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', ev => { const r = el.getBoundingClientRect(); show({clientX: r.left, clientY: r.bottom}, rows); });
    el.addEventListener('blur', hide);
  });
  // line charts: crosshair snapping to nearest x
  document.querySelectorAll('svg[data-line]').forEach(svg => {
    const spec = JSON.parse(svg.getAttribute('data-line'));
    const cross = svg.querySelector('.cross');
    svg.addEventListener('pointermove', ev => {
      const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      let best = null, bd = 1e9;
      for (const px of spec.px) { const d = Math.abs(px - p.x); if (d < bd) { bd = d; best = px; } }
      if (best === null) return;
      const i = spec.px.indexOf(best);
      cross.setAttribute('x1', best); cross.setAttribute('x2', best); cross.style.display = 'block';
      const rows = [{v: spec.xl + ' ' + spec.xs[i]}];
      for (const s of spec.series) if (s.ys[i] !== null) rows.push({v: s.ys[i], k: s.name, c: s.color});
      show(ev, rows);
    });
    svg.addEventListener('pointerleave', () => { cross.style.display = 'none'; hide(); });
  });
  const sel = document.getElementById('theme');
  if (sel) sel.addEventListener('change', () => {
    if (sel.value === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', sel.value);
  });
})();
"""

# ----------------------------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------------------------
STATUS_CLASS = {
    "IDENTICAL": ("st-good", "✓"), "EQUIVALENT": ("st-warn", "≈"), "DIFFERENT": ("st-serious", "≠"),
    "FAIL": ("st-critical", "✕"), "SKIPPED": ("st-muted", "–"),
    "faster": ("st-good", "▲"), "slower": ("st-critical", "▼"), "negligible": ("st-muted", "≈"),
    "no-change": ("st-muted", "–"), "insufficient-data": ("st-muted", "?"),
    "passed": ("st-good", "✓"), "failed": ("st-critical", "✕"), "skipped": ("st-muted", "–"), "notrun": ("st-muted", "–"),
    "ok": ("st-good", "✓"),
}
STATUS_COLOR = {"st-good": "var(--good)", "st-critical": "var(--critical)", "st-muted": "var(--axis)",
                "st-warn": "var(--warn)", "st-serious": "var(--serious)"}


def badge(status: str) -> str:
    cls, icon = STATUS_CLASS.get(status, ("st-muted", "•"))
    return f'<span class="badge {cls}"><span class="dot"></span>{icon} {ESC(status)}</span>'


def fnum(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:.{digits}g}" if ax < 100 else f"{x:.1f}"
    if ax >= 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.2e}"


def fsp(x) -> str:
    return "–" if x is None else f"{x:.4f}×"


def fdelta(x) -> str:
    return "–" if x is None else f"{(x - 1) * 100:+.2f}%"


def fp(p) -> str:
    if p is None:
        return "–"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def tip_attr(rows: list) -> str:
    return f" tabindex=\"0\" data-tip='{ESC(json.dumps(rows), quote=True)}'"


# ----------------------------------------------------------------------------------------------
# inline marks
# ----------------------------------------------------------------------------------------------
W_MARK = 240


def _log_domain(vals, min_half_width=0.03, clamp=(0.25, 4.0)):
    lo = min([v for v in vals if v and v > 0] + [1 - min_half_width])
    hi = max([v for v in vals if v and v > 0] + [1 + min_half_width])
    lo, hi = max(lo, clamp[0]), min(hi, clamp[1])
    m = max(abs(math.log(lo)), abs(math.log(hi))) * 1.08
    return math.exp(-m), math.exp(m)


def forest_axis(dom, band) -> str:
    lo, hi = dom
    w = W_MARK
    X = lambda v: 18 + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (w - 36)
    ticks = _nice_ratio_ticks(lo, hi)
    parts = [f'<svg width="{w}" height="22" role="img" aria-label="speedup axis">']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="14" y2="22"/>')
        parts.append(f'<text x="{X(t):.1f}" y="11" text-anchor="middle">{t:g}×</text>')
    parts.append("</svg>")
    return "".join(parts)


def _nice_ratio_ticks(lo, hi):
    for step in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        n = (hi - lo) / step
        if n <= 6:
            break
    t0 = math.ceil(lo / step) * step
    out, t = [], t0
    while t <= hi + 1e-12:
        out.append(round(t, 4))
        t += step
    if 1.0 not in out and lo < 1 < hi:
        out.append(1.0)
    return sorted(set(out))


def forest_mark(sp, ci, verdict, dom, band) -> str:
    lo, hi = dom
    w = W_MARK
    X = lambda v: 18 + (math.log(min(max(v, lo), hi)) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (w - 36)
    s = [f'<svg width="{w}" height="20" aria-hidden="true">',
         f'<rect class="band" x="{X(1 - band):.1f}" y="2" width="{X(1 + band) - X(1 - band):.1f}" height="16"/>',
         f'<line class="ref" x1="{X(1):.1f}" x2="{X(1):.1f}" y1="0" y2="20"/>']
    if sp is not None:
        if ci:
            s.append(f'<line class="ci" x1="{X(ci[0]):.1f}" x2="{X(ci[1]):.1f}" y1="10" y2="10"/>')
        col = STATUS_COLOR.get(STATUS_CLASS.get(verdict, ("st-muted", ""))[0], "var(--axis)")
        s.append(f'<circle class="ring" cx="{X(sp):.1f}" cy="10" r="5" fill="{col}"/>')
    s.append("</svg>")
    return "".join(s)


def pair_bars(a, b, vmax, w=200) -> str:
    """Two thin bars from a common baseline: A (slot 1) above B (slot 2)."""
    if not vmax:
        return ""
    L = lambda v: max(1.0, (v or 0) / vmax * (w - 4))
    return (f'<svg width="{w}" height="20" aria-hidden="true">'
            f'<path d="{_hbar(0, 2, L(a), 7)}" fill="var(--a)"/>'
            f'<path d="{_hbar(0, 11, L(b), 7)}" fill="var(--b)"/></svg>')


def one_bar(v, vmax, color, w=220) -> str:
    if not vmax:
        return ""
    L = max(1.0, (v or 0) / vmax * (w - 4))
    return f'<svg width="{w}" height="16" aria-hidden="true"><path d="{_hbar(0, 3, L, 10)}" fill="{color}"/></svg>'


def _hbar(x, y, length, h, r=3.0) -> str:
    """Horizontal bar, square at the baseline, rounded data end."""
    r = min(r, h / 2, length / 2)
    x2 = x + length
    return (f"M{x:.1f},{y:.1f} H{x2 - r:.1f} Q{x2:.1f},{y:.1f} {x2:.1f},{y + r:.1f} V{y + h - r:.1f} "
            f"Q{x2:.1f},{y + h:.1f} {x2 - r:.1f},{y + h:.1f} H{x:.1f} Z")


def strip(a_vals, b_vals, w=200) -> str:
    vals = [v for v in (a_vals or []) + (b_vals or []) if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.01 or 1
    lo, hi = lo - pad, hi + pad
    X = lambda v: 6 + (v - lo) / (hi - lo) * (w - 12)
    s = [f'<svg width="{w}" height="26" aria-hidden="true">',
         f'<line class="grid" x1="0" x2="{w}" y1="7" y2="7"/><line class="grid" x1="0" x2="{w}" y1="19" y2="19"/>']
    for v in a_vals or []:
        s.append(f'<circle class="ring" cx="{X(v):.1f}" cy="7" r="4.5" fill="var(--a)"/>')
    for v in b_vals or []:
        s.append(f'<circle class="ring" cx="{X(v):.1f}" cy="19" r="4.5" fill="var(--b)"/>')
    s.append("</svg>")
    return "".join(s)


_line_id = [0]


def line_chart(series: list[dict], xlabel: str, ylabel: str, logy: bool = False, w: int = 560, h: int = 220,
               vlines: list | None = None, ymin0: bool = True) -> str:
    """series: [{name, color, points: [[x, y], ...]}]; one shared x grid (union of xs).
    vlines: [(x, label)] vertical reference markers (e.g. start of the timed window)."""
    pts = [(x, y) for s in series for x, y in s["points"] if y is not None and (not logy or y > 0)]
    if not pts:
        return '<p class="small">no data</p>'
    xs = sorted({x for s in series for x, _ in s["points"]})
    x0, x1 = min(xs), max(xs) or 1
    ys = [y for _, y in pts]
    if logy:
        y0, y1 = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))
        if y0 == y1:
            y1 += 1
        Y = lambda v: (h - 28) - (math.log10(v) - y0) / (y1 - y0) * (h - 40)
        yt = [10 ** e for e in range(y0, y1 + 1)]
    else:
        yt = _nice_ticks(0.0, max(ys) or 1.0, 5)
        y0, y1 = yt[0], yt[-1]
        Y = lambda v: (h - 28) - (v - y0) / (y1 - y0) * (h - 40)
    L, R = 56, w - 12
    X = lambda v: L + (v - x0) / ((x1 - x0) or 1) * (R - L)
    out = []
    for t in yt:
        out.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        out.append(f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{fnum(t, 2) if logy else _tick(t)}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{h - 28}" y2="{h - 28}"/>')
    for i in range(5):
        xv = x0 + i * (x1 - x0) / 4
        out.append(f'<text x="{X(xv):.1f}" y="{h - 12}" text-anchor="middle">{fnum(xv, 3)}</text>')
    out.append(f'<text x="{R}" y="{h - 1}" text-anchor="end">{ESC(xlabel)}</text>')
    out.append(f'<text x="{L}" y="10">{ESC(ylabel)}</text>')
    spec = {"px": [round(X(x), 1) for x in xs], "xs": [fnum(x, 4) for x in xs], "xl": xlabel, "series": []}
    for s in series:
        d = [(X(x), Y(y)) for x, y in s["points"] if y is not None and (not logy or y > 0)]
        if d:
            path = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in d)
            out.append(f'<path d="{path}" fill="none" stroke="{s["color"]}" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
        m = {x: y for x, y in s["points"]}
        spec["series"].append({"name": s["name"], "color": s["color"],
                               "ys": [fnum(m.get(x), 3) if m.get(x) is not None else None for x in xs]})
    for vx, vlab in vlines or []:
        if vx is not None and x0 <= vx <= x1:
            out.append(f'<line class="axis" x1="{X(vx):.1f}" x2="{X(vx):.1f}" y1="12" y2="{h - 28}"/>'
                       f'<text x="{X(vx) + 3:.1f}" y="22">{ESC(vlab)}</text>')
    out.append(f'<line class="cross ref" x1="0" x2="0" y1="12" y2="{h - 28}" style="display:none"/>')
    _line_id[0] += 1
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" '
            f"aria-label=\"{ESC(ylabel)} vs {ESC(xlabel)}\" data-line='{ESC(json.dumps(spec), quote=True)}'>"
            + "".join(out) + "</svg>")


def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / (n - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    t = math.floor(lo / step) * step
    out = [round(t, 10)]
    while out[-1] < hi - 1e-12 * max(1.0, abs(hi)):
        t += step
        out.append(round(t, 10))
    return out


def _tick(v) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:g}"


def scatter(points: list[dict], xlabel: str, ylabel: str, w: int = 420, h: int = 230) -> str:
    """points: [{x, y, color, tip}] -> SVG scatter with hover targets larger than the marks."""
    pts = [p for p in points if p.get("x") is not None and p.get("y") is not None]
    if len(pts) < 2:
        return '<p class="small">not enough data</p>'
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    xt, yt = _nice_ticks(min(xs), max(xs)), _nice_ticks(min(ys), max(ys))
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]
    L, R, T, B = 58, w - 14, 14, h - 34
    X = lambda v: L + (v - x0) / ((x1 - x0) or 1) * (R - L)
    Y = lambda v: B - (v - y0) / ((y1 - y0) or 1) * (B - T)
    o = []
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{_tick(t)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}"/>')
    for i, t in enumerate(xt):
        anchor = "end" if i == len(xt) - 1 else "middle"
        o.append(f'<text x="{X(t):.1f}" y="{B + 15}" text-anchor="{anchor}">{_tick(t)}</text>')
    o.append(f'<text x="{R}" y="{h - 2}" text-anchor="end">{ESC(xlabel)}</text>')
    o.append(f'<text x="{L}" y="10">{ESC(ylabel)}</text>')
    # emphasised points last, so they are drawn on top
    for p in sorted(pts, key=lambda q: q.get("color") == "var(--emph)"):
        o.append(f'<g{tip_attr(p.get("tip") or [])}><circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="12" '
                 f'fill="transparent"/><circle class="ring" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="4.5" '
                 f'fill="{p.get("color", "var(--a)")}"/></g>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" '
            f'aria-label="{ESC(ylabel)} vs {ESC(xlabel)}">' + "".join(o) + "</svg>")


def correlations(xs, ys) -> dict:
    """Pearson r and Spearman rho of paired samples (None if undefined)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    import numpy as np
    from scipy import stats as st
    a, b = np.asarray([p[0] for p in pairs], float), np.asarray([p[1] for p in pairs], float)
    if a.std() == 0 or b.std() == 0:
        return {"n": len(pairs)}
    return {"n": len(pairs), "pearson": float(st.pearsonr(a, b)[0]), "spearman": float(st.spearmanr(a, b)[0])}


def fcorr(c: dict) -> str:
    if c.get("spearman") is None:
        return "–"
    return f"ρ {c['spearman']:+.2f} · r {c['pearson']:+.2f}"


TEL_LABELS = {"energy_kj_per_ns": "energy per simulated ns", "ns_per_day_per_kw": "ns/day per kW of GPU power",
              "power_mean": "GPU power (mean)", "power_max": "GPU power (max)", "util_gpu_mean": "GPU utilisation",
              "util_mem_mean": "GPU memory-controller utilisation", "vram_proc_max": "VRAM used by mdrun",
              "vram_used_max": "VRAM used on the device", "sm_clock_mean": "SM clock", "temp_max": "GPU temperature (max)",
              "pcie_tx_mean": "PCIe GPU→host", "pcie_rx_mean": "PCIe host→GPU", "cpu_cores_busy": "CPU cores busy (mdrun)"}


def _tel_median(c: dict, metric: str, side: str):
    for t in c.get("telemetry", []):
        if t["metric"] == metric:
            return (t.get(side.lower()) or {}).get("median")
    return None


# ----------------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------------
def _build_line(meta, side):
    b = meta.get("builds", {}).get(side)
    if not b:
        return ""
    v = b.get("version", {})
    env = b.get("runtime_env") or {}
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    return (f"<div>{ESC(side)}</div><div><code>{ESC(str(b.get('spec')))}</code> · {ESC(str(b.get('id')))} · "
            f"profile {ESC(str(b.get('profile')))} · SIMD {ESC(v.get('SIMD instructions', '?'))} · "
            f"GPU {ESC(v.get('GPU support', '?'))}{(' · env <code>' + ESC(envs) + '</code>') if envs else ''}</div>")


def section_header(meta) -> str:
    h = meta.get("host", {})
    cpu = h.get("cpu", {})
    gpus = ", ".join(f"{g.get('name')} (driver {g.get('driver_version')})" for g in h.get("gpus", [])) or "none"
    rows = [f"<div>session</div><div><code>{ESC(meta['id'])}</code> · {ESC(meta.get('command', ''))} · tier "
            f"{ESC(str(meta.get('tier')))} · {ESC(str(meta.get('started')))} → {ESC(str(meta.get('finished')))}</div>",
            f"<div>host</div><div>{ESC(str(cpu.get('model')))} · {cpu.get('logical_cpus')} logical CPUs · "
            f"{h.get('memory_gb')} GB · GPU: {ESC(gpus)} · CUDA {ESC(str(h.get('cuda_toolkit')))} · "
            f"{ESC(str(h.get('virtualization')))}</div>"]
    for side in ("A", "B", "S", "U"):
        rows.append(_build_line(meta, side))
    return f'<div class="card kv">{"".join(rows)}</div>'


def tiles(meta) -> str:
    s = meta.get("summary", {})
    out = []
    q = s.get("quality")
    if q:
        c = q.get("counts", {})
        tot = sum(c.values())
        bad = c.get("DIFFERENT", 0) + c.get("FAIL", 0)
        out.append(_tile("Quality: bitwise identical", f"{c.get('IDENTICAL', 0)} / {tot}",
                         f"{c.get('EQUIVALENT', 0)} equivalent · {bad} different/failed · {c.get('SKIPPED', 0)} skipped"))
    p = s.get("perf")
    if p:
        rows = [x for x in p.get("cases", []) if x.get("e2e", {}).get("speedup")]
        if rows:
            geo = math.exp(sum(math.log(x["e2e"]["speedup"]) for x in rows) / len(rows))
            nf = sum(1 for x in rows if x["e2e"].get("verdict") == "faster")
            ns = sum(1 for x in rows if x["e2e"].get("verdict") == "slower")
            out.append(_tile("Performance: geomean speedup", fsp(geo),
                             f"{len(rows)} case×config · {nf} faster · {ns} slower"))
    m = s.get("micro")
    if m and m.get("nbnxm"):
        g = [x["geomean_speedup"] for x in m["nbnxm"] if x.get("geomean_speedup")]
        if g:
            geo = math.exp(sum(math.log(v) for v in g) / len(g))
            out.append(_tile("Nonbonded kernels: geomean", fsp(geo), f"{len(g)} benchmark sets"))
    sw = s.get("sweep")
    if sw and sw.get("systems"):
        gains = [x["gain_single_vs_default"] for x in sw["systems"] if x.get("gain_single_vs_default")]
        if gains:
            out.append(_tile("Sweep: best args vs default", f"up to {max(gains):.2f}×",
                             f"{len(sw['systems'])} systems"))
    u = s.get("upstream")
    if u:
        c = u.get("counts", {})
        out.append(_tile("Upstream ctest", f"{c.get('passed', 0)} passed", f"{len(u.get('failed', []))} failed"))
    return f'<div class="tiles">{"".join(out)}</div>' if out else ""


def _tile(label, value, note) -> str:
    return (f'<div class="tile"><div class="label">{ESC(label)}</div><div class="value">{ESC(value)}</div>'
            f'<div class="note">{ESC(note)}</div></div>')


def section_quality(q) -> str:
    res = q.get("results", [])
    if not res:
        return ""
    tol = q.get("tolerance", {})
    out = ["<h2>1 · Quality: are the outputs unchanged?</h2>",
           f"<p>Each case runs the <em>same</em> .tpr with the baseline (twice) and the candidate. "
           f"<b>IDENTICAL</b>: every TRR frame (x, v, f, box at full precision), EDR frame and xvg output is bit-for-bit "
           f"equal. Otherwise step-0 forces/energies are compared: <b>EQUIVALENT</b> when within tolerance "
           f"(force rel-RMS ≤ {tol.get('force_rel_rms')}, every energy term's change ≤ {tol.get('energy_rel')} of the summed "
           f"component energy magnitudes, or {tol.get('noise_factor')}× "
           f"the baseline's own run-to-run noise for non-deterministic configurations), <b>DIFFERENT</b> beyond it.</p>"]
    gc = q.get("grompp_counts", {})
    out.append('<p class="small">grompp check (candidate .tpr content vs baseline): '
               + " · ".join(f"{badge(k)} {v}" for k, v in sorted(gc.items())) + "</p>")
    # matrix: case x config
    configs = []
    for r in res:
        if r["config"] not in configs:
            configs.append(r["config"])
    cases = []
    for r in res:
        if r["case"] not in cases:
            cases.append(r["case"])
    by = {(r["case"], r["config"]): r for r in res}
    order = {"FAIL": 0, "DIFFERENT": 1, "EQUIVALENT": 2, "IDENTICAL": 3, "SKIPPED": 4}
    suite_cases = [c for c in cases if not c.startswith("rt/")]
    rt_cases = [c for c in cases if c.startswith("rt/")]

    def matrix(case_list):
        head = "".join(f"<th>{ESC(c)}</th>" for c in configs)
        rows = []
        for c in case_list:
            first = next((by[(c, k)] for k in configs if (c, k) in by), {})
            cells = []
            for k in configs:
                r = by.get((c, k))
                if r is None:
                    cells.append("<td></td>")
                    continue
                det = r.get("deterministic")
                tip = [{"v": r["status"], "k": f"{c} · {k}"},
                       {"v": "run-to-run reproducible" if det else ("not reproducible run-to-run" if det is False
                                                                     else "reproducibility not established (GPU)")}]
                for key, lab in (("f_rel_rms", "step-0 force rel-RMS"), ("noise_f_rel_rms", "baseline noise (force)"),
                                 ("e_max_rel", "step-0 energy rel"), ("noise_e_max_rel", "baseline noise (energy)"),
                                 ("first_diff_step", "first differing step")):
                    if r.get(key) is not None:
                        tip.append({"v": fnum(r[key]), "k": lab})
                if r.get("diff_arrays"):
                    tip.append({"v": ", ".join(r["diff_arrays"]), "k": "differs"})
                if r.get("message"):
                    tip.append({"v": r["message"][:160]})
                mark = "" if det is not False else ' <span class="small" title="not reproducible run-to-run">nd</span>'
                cells.append(f"<td{tip_attr(tip)}>{badge(r['status'])}{mark}</td>")
            rows.append(f"<tr><td class='mono'>{ESC(c)}</td><td class='num'>{fnum(first.get('natoms'))}</td>"
                        f"{''.join(cells)}</tr>")
        return (f'<div class="scroll"><table><thead><tr><th>case</th><th class="num">atoms</th>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>')

    out.append("<h3>Suite cases</h3>")
    out.append('<p class="small"><span class="mono">nd</span> = configuration is not bitwise reproducible '
               'run-to-run on the baseline itself (e.g. GPU reductions with atomics); judged by tolerance.</p>')
    out.append(matrix(suite_cases))
    if rt_cases:
        nbad = sum(1 for c in rt_cases for k in configs if by.get((c, k), {}).get("status") in ("FAIL", "DIFFERENT"))
        out.append(f"<details{' open' if nbad else ''}><summary>Upstream regressiontests inputs ({len(rt_cases)} cases, "
                   f"{nbad} different/failed)</summary>{matrix(rt_cases)}</details>")
    # numeric detail for non-identical rows
    nonid = [r for r in res if r["status"] in ("EQUIVALENT", "DIFFERENT")]
    if nonid:
        out.append("<h3>Size of the differences (non-identical results)</h3>")
        out.append('<div class="legend"><span><span class="sw" style="background:var(--b)"></span>candidate vs baseline</span>'
                   '<span><span class="sw" style="background:var(--a)"></span>baseline vs itself (noise floor)</span>'
                   '<span><span class="ln" style="background:var(--ink-2)"></span>tolerance</span></div>')
        vals = [r.get(k) for r in nonid for k in ("f_rel_rms", "noise_f_rel_rms", "tol_f") if r.get(k)]
        lo = 10 ** math.floor(math.log10(min(vals))) if vals else 1e-9
        hi = 10 ** math.ceil(math.log10(max(vals))) if vals else 1e-3
        if hi <= lo:  # all values in one decade (or exactly-zero differences): show at least two decades
            lo, hi = lo / 10, hi * 10
        rows = []
        for r in sorted(nonid, key=lambda r: (order.get(r["status"], 9), r["case"])):
            rows.append(
                f"<tr{tip_attr([{'v': fnum(r.get('f_rel_rms')), 'k': 'candidate force rel-RMS', 'c': 'var(--b)'}, {'v': fnum(r.get('noise_f_rel_rms')), 'k': 'baseline noise', 'c': 'var(--a)'}, {'v': fnum(r.get('tol_f')), 'k': 'tolerance'}])}>"
                f"<td class='mono'>{ESC(r['case'])}</td><td>{ESC(r['config'])}</td><td>{badge(r['status'])}</td>"
                f"<td class='num'>{fnum(r.get('f_rel_rms'))}</td><td class='num'>{fnum(r.get('noise_f_rel_rms'))}</td>"
                f"<td class='num'>{fnum(r.get('e_max_rel'))}</td><td class='mono small'>{ESC(str(r.get('e_max_rel_term') or ''))}</td>"
                f"<td class='num'>{fnum(r.get('first_diff_step'))}</td>"
                f"<td>{_log_dots(r.get('f_rel_rms'), r.get('noise_f_rel_rms'), r.get('tol_f'), lo, hi)}</td></tr>")
        out.append('<div class="scroll"><table><thead><tr><th>case</th><th>config</th><th>status</th>'
                   '<th class="num">force rel-RMS</th><th class="num">noise</th><th class="num">energy rel</th>'
                   f'<th>worst term</th><th class="num">first diff step</th><th>{_log_axis(lo, hi)}</th></tr></thead><tbody>'
                   + "".join(rows) + "</tbody></table></div>")
        curves = [r for r in nonid if (r.get("curves") or {}).get("x_rmsd")]
        if curves:
            out.append("<details><summary>Trajectory divergence (RMS position difference vs step)</summary>"
                       '<p class="small">Chaotic dynamics amplify any rounding difference exponentially; the slope is '
                       'physics, the starting level is the numerical difference.</p><div class="charts">')
            for r in curves[:24]:
                pts = [[s, v if v > 0 else None] for s, v in r["curves"]["x_rmsd"]]
                out.append(f'<div class="tile"><div class="label mono">{ESC(r["case"])} · {ESC(r["config"])}</div>'
                           + line_chart([{"name": "RMS Δx (nm)", "color": "var(--b)", "points": pts}], "step",
                                        "RMS Δx (nm)", logy=True, w=420, h=200) + "</div>")
            out.append("</div></details>")
    bad = [r for r in res if r["status"] == "FAIL"]
    if bad:
        out.append("<h3>Failures</h3><div class='scroll'><table><thead><tr><th>case</th><th>config</th><th>message</th>"
                   "</tr></thead><tbody>" + "".join(
                       f"<tr><td class='mono'>{ESC(r['case'])}</td><td>{ESC(r['config'])}</td>"
                       f"<td class='wrap'>{ESC(r.get('message') or '')}</td></tr>" for r in bad) + "</tbody></table></div>")
    return "".join(out)


def _log_axis(lo, hi, w=W_MARK) -> str:
    X = lambda v: 18 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (w - 36)
    s = [f'<svg width="{w}" height="22" aria-label="log scale">']
    e = math.log10(lo)
    while e <= math.log10(hi) + 1e-9:
        s.append(f'<line class="grid" x1="{X(10 ** e):.1f}" x2="{X(10 ** e):.1f}" y1="14" y2="22"/>'
                 f'<text x="{X(10 ** e):.1f}" y="11" text-anchor="middle">1e{int(round(e))}</text>')
        e += 1
    return "".join(s) + "</svg>"


def _log_dots(cand, noise, tol, lo, hi, w=W_MARK) -> str:
    X = lambda v: 18 + (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (w - 36)
    s = [f'<svg width="{w}" height="20" aria-hidden="true">']
    if tol:
        s.append(f'<line class="ref" x1="{X(tol):.1f}" x2="{X(tol):.1f}" y1="1" y2="19"/>')
    if noise:
        s.append(f'<circle class="ring" cx="{X(noise):.1f}" cy="10" r="5" fill="var(--a)"/>')
    if cand:
        s.append(f'<circle class="ring" cx="{X(cand):.1f}" cy="10" r="5" fill="var(--b)"/>')
    return "".join(s) + "</svg>"


def _forest_table(rows, cols, band, title_axis="speedup (B vs A)") -> str:
    """rows: dicts with 'cells' (list of html), 'sp', 'ci', 'verdict', 'tip'."""
    vals = [v for r in rows for v in ([r.get("sp")] + list(r.get("ci") or []))]
    dom = _log_domain(vals, max(band * 2, 0.03))
    head = "".join(f"<th{' class=num' if c.startswith('#') else ''}>{ESC(c.lstrip('#'))}</th>" for c in cols)
    body = []
    for r in rows:
        body.append(f"<tr{tip_attr(r['tip'])}>{''.join(r['cells'])}<td>{forest_mark(r.get('sp'), r.get('ci'), r.get('verdict'), dom, band)}</td></tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}<th>{forest_axis(dom, band)}</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def section_perf(p, records) -> str:
    cases = p.get("cases", [])
    if not cases:
        return ""
    st = p.get("settings", {})
    band = float(st.get("min_effect", 0.01))
    out = ["<h2>2 · Performance: end-to-end A/B</h2>",
           f"<p>{st.get('repeats')} interleaved repeats per side in seeded random order after a warm-up; timed window "
           f"≈{st.get('target_seconds')} s after <code>-resetstep</code>; PME tuning "
           f"{'on' if st.get('tunepme') else 'off'}. Speedup = ratio of geometric means of ns/day (B/A) with a 95% Welch "
           f"CI on log values; the grey band marks ±{band * 100:g}% (smaller significant effects are "
           f"<em>negligible</em>).</p>",
           '<div class="legend"><span><span class="sw" style="background:var(--a)"></span>A baseline</span>'
           '<span><span class="sw" style="background:var(--b)"></span>B candidate</span>'
           f'<span>{badge("faster")}</span><span>{badge("slower")}</span><span>{badge("no-change")}</span></div>']
    reps = {}
    for r in records:
        if r["suite"] == "perf" and r["kind"] == "e2e" and r["metric"] == "ns_per_day":
            reps.setdefault((r["case"], r["config"], r["side"]), []).append(r["value"])
    rows = []
    for c in cases:
        e = c.get("e2e") or {}
        if c.get("status") != "ok":
            rows.append({"cells": [f"<td class='mono'>{ESC(c['case'])}</td>", f"<td>{ESC(c['config'])}</td>",
                                   f"<td class='num'>{fnum(c.get('natoms'))}</td>", "<td></td><td></td>",
                                   f"<td class='wrap small' colspan='4'>{ESC(str(c.get('status')))}</td>"],
                         "sp": None, "ci": None, "verdict": None, "tip": [{"v": str(c.get("status"))}]})
            continue
        a, b = e.get("a", {}), e.get("b", {})
        tip = [{"v": fsp(e.get("speedup")), "k": "speedup"},
               {"v": f"[{fsp((e.get('ci') or [None])[0])}, {fsp((e.get('ci') or [None, None])[1])}]", "k": "95% CI"},
               {"v": fnum(a.get("median")), "k": "A median ns/day", "c": "var(--a)"},
               {"v": fnum(b.get("median")), "k": "B median ns/day", "c": "var(--b)"},
               {"v": f"{(a.get('cv') or 0) * 100:.2f}% / {(b.get('cv') or 0) * 100:.2f}%", "k": "CV A / B"},
               {"v": fp(e.get("p")), "k": "p (Welch, log)"}, {"v": fp(e.get("p_mwu")), "k": "p (Mann-Whitney)"},
               {"v": f"{(e.get('mde') or 0) * 100:.2f}%", "k": "min. detectable effect"}]
        rows.append({"cells": [
            f"<td class='mono'>{ESC(c['case'])}</td>", f"<td>{ESC(c['config'])}</td>",
            f"<td class='num'>{fnum(c.get('natoms'))}</td>",
            f"<td class='num'>{fnum(a.get('median'))}</td>", f"<td class='num'>{fnum(b.get('median'))}</td>",
            f"<td>{strip(reps.get((c['case'], c['config'], 'A')), reps.get((c['case'], c['config'], 'B')), 140)}</td>",
            f"<td class='num'>{fdelta(e.get('speedup'))}</td>", f"<td class='num'>{fp(e.get('p'))}</td>",
            f"<td>{badge(e.get('verdict', '?'))}</td>"],
            "sp": e.get("speedup"), "ci": e.get("ci"), "verdict": e.get("verdict"), "tip": tip})
    out.append(_forest_table(rows, ["case", "config", "#atoms", "#A ns/day", "#B ns/day", "repeats (A top, B bottom)",
                                    "#Δ", "#p", "verdict"], band))
    tel_cases = [c for c in cases if c.get("telemetry")]
    if tel_cases:
        out.append("<h3>GPU load and energy efficiency (timed window)</h3>"
                   "<p>Sampled with NVML every 0.1 s during each timed repeat and restricted to the window after mdrun's "
                   "counter reset. Energy comes from the GPU's hardware energy counter (GPU board only — CPU and the rest of "
                   "the node are not included); <em>energy per simulated ns</em> is "
                   "the GPU energy needed to advance the simulation by 1 ns (lower is better) — the quantity that sets the "
                   "cost of a long campaign. The forest plot shows its improvement factor A/B.</p>")
        rows = []
        for c in tel_cases:
            en = next((t for t in c["telemetry"] if t["metric"] == "energy_kj_per_ns"), None)
            if en is None:
                continue
            v = en.get("verdict_fdr", en.get("verdict"))
            g = lambda m, sd: fnum(_tel_median(c, m, sd))
            tip = [{"v": g("util_gpu_mean", "A") + " / " + g("util_gpu_mean", "B") + " %", "k": "GPU util A / B"},
                   {"v": g("power_mean", "A") + " / " + g("power_mean", "B") + " W", "k": "power A / B"},
                   {"v": g("energy_kj_per_ns", "A") + " / " + g("energy_kj_per_ns", "B") + " kJ/ns", "k": "energy A / B"},
                   {"v": fp(en.get("q")), "k": "q"}]
            rows.append({"cells": [f"<td class='mono'>{ESC(c['case'])}</td>", f"<td>{ESC(c['config'])}</td>",
                                   f"<td class='num'>{g('util_gpu_mean', 'A')} / {g('util_gpu_mean', 'B')}</td>",
                                   f"<td class='num'>{g('power_mean', 'A')} / {g('power_mean', 'B')}</td>",
                                   f"<td class='num'>{g('vram_proc_max', 'A')} / {g('vram_proc_max', 'B')}</td>",
                                   f"<td class='num'>{g('energy_kj_per_ns', 'A')}</td>",
                                   f"<td class='num'>{g('energy_kj_per_ns', 'B')}</td>",
                                   f"<td class='num'>{g('ns_per_day_per_kw', 'A')} / {g('ns_per_day_per_kw', 'B')}</td>",
                                   f"<td class='num'>{fdelta(en.get('speedup'))}</td>", f"<td>{badge(v)}</td>"],
                         "sp": en.get("speedup"), "ci": en.get("ci"), "verdict": v, "tip": tip})
        out.append(_forest_table(rows, ["case", "config", "#GPU util % A / B", "#power W A / B", "#VRAM MiB A / B",
                                        "#A kJ/ns", "#B kJ/ns", "#ns/day per kW A / B", "#Δ efficiency", "verdict"], band))
    # per case details: stages and kernels
    out.append("<h3>Where the time goes: stages and GPU kernels</h3>"
               "<p>Per-stage times come from mdrun's cycle counters (with sub-counters) over the timed window; GPU "
               "kernel times from Nsight Systems runs. Each family is FDR-controlled (Benjamini-Hochberg q-values); "
               "rows below "
               "1% of the time are omitted.</p>")
    for c in cases:
        if c.get("status") != "ok":
            continue
        e = c.get("e2e", {})
        setup = (c.get("setup") or {}).get("A", {})
        sig = [s for s in (c.get("stages", []) + c.get("kernels", [])) if s.get("verdict_fdr") in ("faster", "slower")]
        summ = (f"{ESC(c['case'])} · {ESC(c['config'])} — {fdelta(e.get('speedup'))} e2e"
                f"{f' · {len(sig)} significant stage/kernel changes' if sig else ''}")
        out.append(f"<details{' open' if sig else ''}><summary>{summ}</summary>")
        out.append(f'<p class="small">nsteps {c.get("nsteps")} (counters reset at {c.get("resetstep")}) · mdrun chose '
                   f'nstlist {setup.get("nstlist", "?")}, {setup.get("ranks", "?")} rank(s) × '
                   f'{setup.get("omp_threads", "?")} threads'
                   f'{", update on GPU" if setup.get("update") == "gpu" else ""}'
                   f'{", CUDA graphs active" if setup.get("cuda_graphs") else ""}'
                   f'{", GPU mapping " + ESC(setup["gpu_mapping"]) if setup.get("gpu_mapping") else ""}</p>')
        for fam_name, fam, unit in (("Stages (ms/step)", c.get("stages", []), "ms/step"),
                                    ("GPU kernels & copies (µs/step)", c.get("kernels", []), "µs/step")):
            if not fam:
                continue
            fam = sorted(fam, key=lambda x: -(x.get("pct") or 0))
            vmax = max(max(x["a"].get("median") or 0, x["b"].get("median") or 0) for x in fam)
            rows = []
            for s in fam:
                v = s.get("verdict_fdr", s.get("verdict"))
                tip = [{"v": fnum(s["a"].get("median")), "k": f"A {unit}", "c": "var(--a)"},
                       {"v": fnum(s["b"].get("median")), "k": f"B {unit}", "c": "var(--b)"},
                       {"v": fsp(s.get("speedup")), "k": "speedup"}, {"v": fp(s.get("q")), "k": "q (FDR)"}]
                rows.append({"cells": [f"<td class='mono'>{ESC(s['metric'])}</td>",
                                       f"<td class='num'>{(s.get('pct') or 0):.1f}%</td>",
                                       f"<td class='num'>{fnum(s['a'].get('median'))}</td>",
                                       f"<td class='num'>{fnum(s['b'].get('median'))}</td>",
                                       f"<td>{pair_bars(s['a'].get('median'), s['b'].get('median'), vmax, 160)}</td>",
                                       f"<td class='num'>{fdelta(s.get('speedup'))}</td>",
                                       f"<td class='num'>{fp(s.get('q'))}</td>", f"<td>{badge(v)}</td>"],
                             "sp": s.get("speedup"), "ci": s.get("ci"), "verdict": v, "tip": tip})
            out.append(f"<h3>{ESC(fam_name)}</h3>")
            out.append(_forest_table(rows, ["name", "#share", "#A", "#B", "A / B", "#Δ", "#q", "verdict"], band))
        if c.get("telemetry"):
            out.append(_telemetry_detail(c))
        out.append("</details>")
    return "".join(out)


def _telemetry_detail(c: dict) -> str:
    out = ["<h3>GPU telemetry (timed window, median over repeats)</h3>"]
    rows = []
    for t in c["telemetry"]:
        a, b = (t.get("a") or {}).get("median"), (t.get("b") or {}).get("median")
        rel = (b / a - 1) * 100 if a and b else None
        v = "" if t.get("informational") else badge(t.get("verdict_fdr", t.get("verdict")))
        rows.append(f"<tr><td>{ESC(TEL_LABELS.get(t['metric'], t['metric']))}</td><td>{ESC(t.get('unit', ''))}</td>"
                    f"<td class='num'>{fnum(a)}</td><td class='num'>{fnum(b)}</td>"
                    f"<td class='num'>{'–' if rel is None else f'{rel:+.2f}%'}</td>"
                    f"<td>{pair_bars(a, b, max(a or 0, b or 0), 140)}</td><td>{v}</td></tr>")
    out.append('<div class="scroll"><table><thead><tr><th>metric</th><th>unit</th><th class="num">A</th>'
               '<th class="num">B</th><th class="num">B vs A</th><th>A / B</th><th>verdict</th></tr></thead><tbody>'
               + "".join(rows) + "</tbody></table></div>")
    ser = c.get("telemetry_series") or {}
    if ser.get("A", {}).get("series") and ser.get("B", {}).get("series"):
        charts = []
        for key, lab in (("util_gpu", "GPU utilisation (%)"), ("power", "GPU power (W)"), ("vram_used", "VRAM used (MiB)")):
            series = [{"name": f"{side} ({ser[side]['ns_per_day']:.4g} ns/day)", "color": f"var(--{side.lower()})",
                       "points": list(zip(ser[side]["series"]["t"], ser[side]["series"][key]))} for side in ("A", "B")]
            charts.append('<div class="tile">' + line_chart(series, "time since launch (s)", lab, w=420, h=200,
                                                             vlines=[(ser["A"].get("reset_at_s"), "timed window →")])
                          + "</div>")
        thr = sorted(set((ser["A"].get("throttle") or []) + (ser["B"].get("throttle") or [])))
        out.append('<div class="legend"><span><span class="ln" style="background:var(--a)"></span>A (median repeat)</span>'
                   '<span><span class="ln" style="background:var(--b)"></span>B (median repeat)</span>'
                   + (f"<span>{badge('slower')} clock throttling seen: {ESC(', '.join(thr))}</span>" if thr else "")
                   + '</div><div class="charts">' + "".join(charts) + "</div>")
    return "".join(out)


def section_correlation(meta: dict) -> str:
    """How throughput relates to GPU load, across sweep configurations and across perf repeats."""
    summ = meta.get("summary", {})
    out = []
    sw = summ.get("sweep") or {}
    systems = [s for s in sw.get("systems", []) if any((p.get("telemetry") or {}).get("util_gpu_mean") is not None
                                                        for p in s["points"])]
    if systems:
        out.append("<h3>Across run configurations (sweep)</h3>"
                   "<p>Every measured configuration of the sweep is one point (median of its repeats). The coefficients "
                   "are Spearman ρ (rank) and Pearson r between ns/day and each quantity.</p>")
        head = "".join(f"<th>{ESC(TEL_LABELS[k])}</th>" for k in ("util_gpu_mean", "power_mean", "sm_clock_mean",
                                                                   "pcie_rx_mean", "vram_proc_max", "cpu_cores_busy"))
        rows = []
        charts = []
        for sy in systems:
            pts = [p for p in sy["points"] if p["status"] == "ok" and p.get("telemetry")]
            nsd = [p["median"] for p in pts]
            cells = []
            for k in ("util_gpu_mean", "power_mean", "sm_clock_mean", "pcie_rx_mean", "vram_proc_max", "cpu_cores_busy"):
                cells.append(f"<td class='num'>{fcorr(correlations([p['telemetry'].get(k) for p in pts], nsd))}</td>")
            rows.append(f"<tr><td class='mono'>{ESC(sy['name'])}</td><td class='num'>{len(pts)}</td>{''.join(cells)}</tr>")
            best = sy["best_single"]["label"]
            for k, xl in (("util_gpu_mean", "GPU utilisation (%)"), ("power_mean", "GPU power (W)")):
                sp = [{"x": p["telemetry"].get(k), "y": p["median"],
                       "color": "var(--emph)" if p["label"] == best else "var(--muted)",
                       "tip": [{"v": f"{fnum(p['median'])} ns/day", "k": p["label"]},
                               {"v": fnum(p["telemetry"].get(k)), "k": xl}]} for p in pts]
                c = correlations([q["x"] for q in sp], [q["y"] for q in sp])
                charts.append(f'<div class="tile"><div class="label mono">{ESC(sy["name"])} · {fcorr(c)}</div>'
                              + scatter(sp, xl, "ns/day") + "</div>")
        out.append('<div class="scroll"><table><thead><tr><th>system</th><th class="num">configs</th>' + head
                   + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
        out.append('<div class="legend"><span><span class="sw" style="background:var(--emph)"></span>best configuration'
                   '</span><span><span class="sw" style="background:var(--muted)"></span>other configurations</span></div>'
                   '<div class="charts">' + "".join(charts) + "</div>")
    # perf repeats: does run-to-run variation follow GPU state?
    recs = [r for r in meta.get("records", []) if r["suite"] == "perf" and r["kind"] == "telemetry"]
    if recs:
        groups: dict = {}
        for r in recs:
            groups.setdefault((r["case"], r["config"]), {}).setdefault((r["side"], r["repeat"]), {})[r["metric"]] = r["value"]
            groups[(r["case"], r["config"])][(r["side"], r["repeat"])]["ns_per_day"] = (r.get("tags") or {}).get("ns_per_day")
        keys = ("util_gpu_mean", "power_mean", "sm_clock_mean", "temp_max")
        rows = []
        for (case, cfg), runs in groups.items():
            vals = list(runs.values())
            nsd = [v.get("ns_per_day") for v in vals]
            cells = "".join(f"<td class='num'>{fcorr(correlations([v.get(k) for v in vals], nsd))}</td>" for k in keys)
            rows.append(f"<tr><td class='mono'>{ESC(case)}</td><td>{ESC(cfg)}</td><td class='num'>{len(vals)}</td>{cells}</tr>")
        out.append("<h3>Across repeats of the same configuration (perf)</h3>"
                   "<p>Within one configuration the work is identical, so a strong correlation of ns/day with clock or "
                   "temperature points to throttling or interference rather than to the code.</p>"
                   '<div class="scroll"><table><thead><tr><th>case</th><th>config</th><th class="num">runs</th>'
                   + "".join(f"<th>{ESC(TEL_LABELS[k])}</th>" for k in keys) + "</tr></thead><tbody>" + "".join(rows)
                   + "</tbody></table></div>")
    if not out:
        return ""
    return "<h2>GPU load vs performance</h2>" + "".join(out)


def section_micro(m) -> str:
    sets = m.get("nbnxm", [])
    if not sets:
        return ""
    band = float(m.get("settings", {}).get("min_effect", 0.01))
    out = ["<h2>3 · Kernel micro-benchmarks (CPU non-bonded)</h2>",
           "<p><code>gmx nonbonded-benchmark</code>: every kernel flavour (Coulomb type / LJ on all or half of the "
           "atoms / combination rule / SIMD layout / interaction modifier / forces F or forces+energies VF) timed in "
           "isolation; lower Mcycles per iteration is better. Verdicts are FDR-controlled across the flavours.</p>"]
    rows = []
    for s in sets:
        fam = s.get("kernels", [])
        nf = sum(1 for k in fam if k.get("verdict_fdr") == "faster")
        ns = sum(1 for k in fam if k.get("verdict_fdr") == "slower")
        rows.append({"cells": [f"<td class='mono'>{ESC(s['name'])}</td>", f"<td class='num'>{fnum(s['size_atoms'])}</td>",
                               f"<td class='num'>{s['threads']}</td>", f"<td class='num'>{len(fam)}</td>",
                               f"<td class='num'>{nf}</td>", f"<td class='num'>{ns}</td>",
                               f"<td class='num'>{fdelta(s.get('geomean_speedup'))}</td>"],
                     "sp": s.get("geomean_speedup"), "ci": None,
                     "verdict": "faster" if nf and not ns else ("slower" if ns and not nf else "no-change"),
                     "tip": [{"v": fsp(s.get("geomean_speedup")), "k": "geomean speedup"}]})
    out.append(_forest_table(rows, ["benchmark", "#atoms", "#threads", "#flavours", "#faster", "#slower", "#geomean Δ"], band))
    for s in sets:
        fam = s.get("kernels", [])
        if not fam:
            continue
        rows = []
        for k in fam:
            v = k.get("verdict_fdr", k.get("verdict"))
            rows.append({"cells": [f"<td class='mono'>{ESC(k['metric'])}</td>",
                                   f"<td class='num'>{fnum(k['a'].get('median'))}</td>",
                                   f"<td class='num'>{fnum(k['b'].get('median'))}</td>",
                                   f"<td class='num'>{fdelta(k.get('speedup'))}</td>",
                                   f"<td class='num'>{fp(k.get('q'))}</td>", f"<td>{badge(v)}</td>"],
                         "sp": k.get("speedup"), "ci": k.get("ci"), "verdict": v,
                         "tip": [{"v": fsp(k.get("speedup")), "k": k["metric"]}, {"v": fp(k.get("q")), "k": "q"}]})
        out.append(f"<details><summary>{ESC(s['name'])}: all {len(fam)} flavours</summary>"
                   + _forest_table(rows, ["flavour", "#A Mcyc/it", "#B Mcyc/it", "#Δ", "#q", "verdict"], band)
                   + "</details>")
    return "".join(out)


def section_sweep(sw) -> str:
    systems = sw.get("systems", [])
    if not systems:
        return ""
    out = ["<h2>4 · Run-argument sweep: which mdrun arguments to use</h2>",
           "<p>Staged search per system: default → offload mode × threads → nstlist → CUDA graphs → PME tuning → "
           "several simulations sharing the node. ns/day is the median over repeats (aggregate over simulations for "
           "throughput points).</p>"]
    rows = []
    for s in systems:
        d, b, t = s["default"], s["best_single"], s["best_throughput"]
        env = " ".join(f"{k}={v}" for k, v in (b.get("env") or {}).items())
        eff = [p for p in s["points"] if p["status"] == "ok" and (p.get("telemetry") or {}).get("energy_kj_per_ns")]
        e = min(eff, key=lambda p: p["telemetry"]["energy_kj_per_ns"]) if eff else None
        btel = b.get("telemetry") or {}
        rows.append(f"<tr><td class='mono'>{ESC(s['name'])}</td><td class='num'>{fnum(s['natoms'])}</td>"
                    f"<td class='num'>{fnum(d['median'])}</td><td class='num'>{fnum(b['median'])}</td>"
                    f"<td class='num'>{fsp(s.get('gain_single_vs_default'))}</td>"
                    f"<td class='wrap mono'>{ESC((env + ' ') if env else '')}gmx mdrun {ESC(b['args'])}"
                    f"{' (PME tuning on)' if b.get('tunepme') else ' -notunepme'}</td>"
                    f"<td class='num'>{fnum(btel.get('util_gpu_mean'))}</td><td class='num'>{fnum(btel.get('power_mean'))}</td>"
                    f"<td class='num'>{fnum(btel.get('energy_kj_per_ns'))}</td>"
                    f"<td class='num'>{fnum(t['median'])}</td><td>{ESC(t['label'])}</td>"
                    f"<td>{ESC(e['label']) + ' · ' + fnum(e['telemetry']['energy_kj_per_ns']) + ' kJ/ns · ' + fnum(e['median']) + ' ns/day' if e else '–'}</td></tr>")
    out.append('<h3>Recommendations</h3><div class="scroll"><table><thead><tr><th>system</th><th class="num">atoms</th>'
               '<th class="num">default ns/day</th><th class="num">best ns/day</th><th class="num">gain</th>'
               '<th>best single-simulation arguments</th><th class="num">GPU util %</th><th class="num">power W</th>'
               '<th class="num">kJ/ns</th><th class="num">best aggregate ns/day</th><th>throughput setup</th>'
               '<th>most energy-efficient configuration</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")
    for s in systems:
        pts = [p for p in s["points"] if p["status"] == "ok" and p["median"]]
        vmax = max([p["median"] for p in pts] or [1])
        best_label, best_tp, default_label = s["best_single"]["label"], s["best_throughput"]["label"], s["default"]["label"]
        rows = []
        for p in sorted(s["points"], key=lambda p: -(p["median"] or 0)):
            role = ("var(--emph)" if p["label"] in (best_label, best_tp) else
                    "var(--b)" if p["label"] == default_label else "var(--rest)")
            tip = [{"v": fnum(p["median"]), "k": "ns/day"}, {"v": p["args"] or "(defaults)", "k": "args"}]
            if p.get("cv") is not None:
                tip.append({"v": f"{p['cv'] * 100:.1f}%", "k": "CV"})
            if p["status"] != "ok":
                tip.append({"v": p["status"] + ": " + (p.get("message") or "")[:120]})
            pt = p.get("telemetry") or {}
            rows.append(f"<tr{tip_attr(tip)}><td>{ESC(p['label'])}</td><td>{ESC(p['stage'])}</td>"
                        f"<td class='num'>{fnum(p['median']) if p['status'] == 'ok' else ESC(p['status'])}</td>"
                        f"<td class='num'>{fnum(p.get('per_sim_median')) if p.get('concurrency', 1) > 1 else ''}</td>"
                        f"<td>{one_bar(p['median'], vmax, role) if p['status'] == 'ok' else ''}</td>"
                        f"<td class='num'>{fnum(pt.get('util_gpu_mean'))}</td><td class='num'>{fnum(pt.get('power_mean'))}</td>"
                        f"<td class='num'>{fnum(pt.get('vram_proc_max'))}</td><td class='num'>{fnum(pt.get('energy_kj_per_ns'))}</td>"
                        f"<td class='num'>{fnum(pt.get('cpu_cores_busy'))}</td></tr>")
        out.append(f"<details><summary>{ESC(s['name'])}: all {len(s['points'])} measured configurations</summary>"
                   '<div class="legend"><span><span class="sw" style="background:var(--emph)"></span>best</span>'
                   '<span><span class="sw" style="background:var(--b)"></span>default</span>'
                   '<span><span class="sw" style="background:var(--rest)"></span>other</span></div>'
                   '<div class="scroll"><table><thead><tr><th>configuration</th><th>stage</th>'
                   '<th class="num">ns/day</th><th class="num">per sim</th><th></th><th class="num">GPU util %</th>'
                   '<th class="num">power W</th><th class="num">VRAM MiB</th><th class="num">kJ/ns</th>'
                   '<th class="num">CPU cores busy</th></tr></thead><tbody>'
                   + "".join(rows) + "</tbody></table></div></details>")
    return "".join(out)


def section_upstream(u) -> str:
    if not u:
        return ""
    c = u.get("counts", {})
    out = ["<h2>5 · GROMACS ctest (unit + upstream regression tests)</h2>",
           "<p>" + " · ".join(f"{badge(k)} {v}" for k, v in sorted(c.items())) + f" · wall {fnum(u.get('wall'))} s</p>"]
    if u.get("failed"):
        out.append("<div class='scroll'><table><thead><tr><th>failed test</th></tr></thead><tbody>"
                   + "".join(f"<tr><td class='mono'>{ESC(n)}</td></tr>" for n in u["failed"]) + "</tbody></table></div>")
    return "".join(out)


def section_history(sessions) -> str:
    """When several sessions are given: e2e speedup per case/config across sessions."""
    perf = [(s, s["summary"].get("perf")) for s in sessions if s["summary"].get("perf")]
    if len(perf) < 2:
        return ""
    keys = []
    for _, p in perf:
        for c in p.get("cases", []):
            k = (c["case"], c["config"])
            if k not in keys:
                keys.append(k)
    head = "".join(f"<th class='num'>{ESC(s['id'])}</th>" for s, _ in perf)
    rows = []
    for k in keys:
        cells = []
        for _, p in perf:
            c = next((x for x in p.get("cases", []) if (x["case"], x["config"]) == k), None)
            e = (c or {}).get("e2e") or {}
            cells.append(f"<td class='num'>{fdelta(e.get('speedup'))} {badge(e['verdict']) if e.get('verdict') else ''}</td>")
        rows.append(f"<tr><td class='mono'>{ESC(k[0])}</td><td>{ESC(k[1])}</td>{''.join(cells)}</tr>")
    return ("<h2>History across sessions</h2><div class='scroll'><table><thead><tr><th>case</th><th>config</th>"
            f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# ----------------------------------------------------------------------------------------------
def write_csv(sessions, path: Path) -> None:
    fields = ["session", "suite", "kind", "case", "config", "side", "build", "repeat", "metric", "value", "unit",
              "better", "tags"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sessions:
            for r in s["records"]:
                row = {k: r.get(k) for k in fields}
                row["tags"] = json.dumps(r.get("tags") or {}, sort_keys=True)
                w.writerow(row)


def write_report(dirs, out: Path | None = None) -> Path:
    sessions = [load_session(Path(d)) for d in dirs]
    out = Path(out) if out else Path(dirs[0]) / "report.html"
    body = []
    for s in sessions:
        summ = s.get("summary", {})
        body.append(f"<h2 style='border:0;margin-top:20px'>Session {ESC(s['id'])}</h2>" if len(sessions) > 1 else "")
        body.append(section_header(s))
        body.append(tiles(s))
        if summ.get("quality"):
            body.append(section_quality(summ["quality"]))
        if summ.get("perf"):
            body.append(section_perf(summ["perf"], s["records"]))
        if summ.get("micro"):
            body.append(section_micro(summ["micro"]))
        if summ.get("sweep"):
            body.append(section_sweep(summ["sweep"]))
        body.append(section_correlation(s))
        if summ.get("upstream"):
            body.append(section_upstream(summ["upstream"]))
    body.append(section_history(sessions))
    title = "GROMACS benchmark report"
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body><main>
<header class="top"><div><h1>{title}</h1><p class="sub">gmxbench · quality (bitwise A/B) · performance (statistical A/B) ·
micro-benchmarks · run-argument sweep</p></div>
<label class="small">theme <select id="theme" class="theme"><option value="auto">auto</option><option value="light">light</option>
<option value="dark">dark</option></select></label></header>
{''.join(body)}
<p class="small" style="margin-top:40px">Raw data: <code>records.jsonl</code> (one JSON record per measurement) and
<code>session.json</code> in each results directory; <code>{ESC(out.with_suffix('.csv').name)}</code> next to this file.</p>
</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>"""
    out.write_text(doc)
    write_csv(sessions, out.with_suffix(".csv"))
    return out
