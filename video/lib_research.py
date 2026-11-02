"""Shared helpers for the research-backed videos (optimizers, train-your-own, residuals-layernorm, arch-lab, scaling).

All data comes from public/research/*.json, produced by train/experiments/*.py (real CPU training runs).
"""
import json
import math
import os

from vscene import *  # noqa: F401,F403

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "public", "research")


def load(name):
    return json.load(open(os.path.join(RES, name + ".json")))


def fmt_lr(lr):
    s = f"{lr:.0e}" if lr < 0.01 else f"{lr:g}"
    return s.replace("e-0", "e-").replace("e+0", "e")


def fmt_n(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n / 1e3:.1f}k"
    return str(int(n))


class Plot(VGroup):
    """Axes with readable mono tick labels, optional log10 x / y. Map data with .p(x, y)."""

    def __init__(self, x_range, y_range, width=9.0, height=4.6, xlog=False, ylog=False, xticks=(), yticks=(),
                 xlabel="", ylabel="", xfmt=None, yfmt=None, tick_size=22, label_size=24, grid=True):
        super().__init__()
        self.xlog, self.ylog = xlog, ylog
        tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
        ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
        self.tx, self.ty = tx, ty
        x0, x1 = tx(x_range[0]), tx(x_range[1])
        y0, y1 = ty(y_range[0]), ty(y_range[1])
        self.ax = Axes(x_range=[x0, x1, (x1 - x0)], y_range=[y0, y1, (y1 - y0)], x_length=width, y_length=height,
                       tips=False, axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False})
        self.add(self.ax)
        self.grid = VGroup()
        xfmt = xfmt or (lambda v: f"{v:g}")
        yfmt = yfmt or (lambda v: f"{v:g}")
        self.xl = VGroup()
        for v in xticks:
            p = self.ax.c2p(tx(v), y0)
            if grid:
                self.grid.add(Line(p, self.ax.c2p(tx(v), y1), color=LINE, stroke_width=1))
            self.xl.add(Text(xfmt(v), font=MONO, font_size=tick_size, color=MUTED).next_to(p, DOWN, buff=0.14))
        self.yl = VGroup()
        for v in yticks:
            p = self.ax.c2p(x0, ty(v))
            if grid:
                self.grid.add(Line(p, self.ax.c2p(x1, ty(v)), color=LINE, stroke_width=1))
            self.yl.add(Text(yfmt(v), font=MONO, font_size=tick_size, color=MUTED).next_to(p, LEFT, buff=0.14))
        self.add_to_back(self.grid)
        self.add(self.xl, self.yl)
        self.xt = Text(xlabel or " ", font=SANS, font_size=label_size, color=MUTED)
        self.yt = Text(ylabel or " ", font=SANS, font_size=label_size, color=MUTED)
        if xlabel:
            self.xt.next_to(self.xl if len(self.xl) else self.ax, DOWN, buff=0.18)
            self.add(self.xt)
        if ylabel:
            self.yt.next_to(self.ax.c2p(x0, y1), UP, buff=0.22).align_to(self.ax, LEFT)
            self.add(self.yt)
        self.yr = (y_range[0], y_range[1])

    def p(self, x, y):
        return self.ax.c2p(self.tx(x), self.ty(y))

    def clampy(self, y):
        return min(max(y, self.yr[0]), self.yr[1])

    def line(self, xs, ys, color, width=4, dashed=False, clip=True, opacity=1.0):
        pts = []
        for x, y in zip(xs, ys):
            if y is None or (isinstance(y, float) and not math.isfinite(y)):
                break
            if (self.xlog and x <= 0):
                continue
            pts.append(self.p(x, self.clampy(y) if clip else y))
        m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
        if len(pts) < 2:
            pts = pts * 2 if pts else [self.p(xs[0], self.yr[1])] * 2
        m.set_points_as_corners(pts)
        if dashed:
            m = DashedVMobject(m, num_dashes=max(20, len(pts) * 2))
        return m


def legend(items, size=22, buff=0.45, direction=RIGHT):
    """items: [(label, color, dashed)] → row of swatches."""
    g = VGroup()
    for lab, col, dashed in items:
        sw = (DashedLine(ORIGIN, RIGHT * 0.5, color=col, stroke_width=3, dash_length=0.08) if dashed
              else Line(ORIGIN, RIGHT * 0.5, color=col, stroke_width=5))
        g.add(VGroup(sw, Text(lab, font=SANS, font_size=size, color=col)).arrange(RIGHT, buff=0.15))
    return g.arrange(direction, buff=buff, **({"aligned_edge": LEFT} if direction is DOWN else {}))


def illus(text="toy example · illustration", size=22):
    return Text(text, font=SANS, font_size=size, color=MUTED, slant=ITALIC)


def pill(text, color, size=26, pad=0.16):
    t = Text(text, font=SANS, font_size=size, color=FG)
    box = RoundedRectangle(corner_radius=0.12, width=t.width + 2 * pad + 0.1, height=t.height + 2 * pad,
                           stroke_color=color, stroke_width=2.2, fill_color=color, fill_opacity=0.14)
    t.move_to(box)
    return VGroup(box, t)


def panel(w, h, color=LINE):
    return RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color, stroke_width=2,
                            fill_color=PANEL, fill_opacity=0.9)
