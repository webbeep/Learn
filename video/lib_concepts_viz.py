"""Manim helpers for the 'concepts' videos: CScene adds a layout self-check at the end of every narration block
(prints [layout] warnings for text that leaves the frame or overlaps other text)."""
from contextlib import contextmanager

from vscene import *  # noqa: F401,F403
from vscene import VScene


def _visible(m):
    try:
        return m.get_fill_opacity() > 0.05 or (m.get_stroke_opacity() > 0.05 and m.get_stroke_width() > 0)
    except Exception:  # noqa: BLE001
        return True


class CScene(VScene):
    @contextmanager
    def say(self, text, **kw):
        if kw.get("chapter") and self.renderer.time > 1.0:
            self.wait(0.5)   # a breath at every chapter change
        with super().say(text, **kw) as line:
            yield line
            self._layout_check(text)

    def _layout_check(self, text):
        fw, fh = config.frame_width / 2, config.frame_height / 2
        texts = []
        for top in self.mobjects:
            for m in top.get_family():
                if isinstance(m, (Text, MarkupText)) and any(_visible(s) for s in m.get_family() if s.has_points()):
                    texts.append(m)
        for t in texts:
            ul, dr = t.get_corner(UL), t.get_corner(DR)
            if ul[0] < -fw + 0.05 or dr[0] > fw - 0.05 or ul[1] > fh - 0.05 or dr[1] < -fh + 0.05:
                print(f"[layout] off-frame text {getattr(t, 'text', '')[:40]!r} in line {text[:40]!r}")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = texts[i], texts[j]
                a0, a1 = a.get_corner(DL), a.get_corner(UR)
                b0, b1 = b.get_corner(DL), b.get_corner(UR)
                ox = min(a1[0], b1[0]) - max(a0[0], b0[0])
                oy = min(a1[1], b1[1]) - max(a0[1], b0[1])
                if ox > 0.04 and oy > 0.04:
                    print(f"[layout] overlap {getattr(a, 'text', '')[:30]!r} / {getattr(b, 'text', '')[:30]!r} in line {text[:40]!r}")

    def fitw(self, m, w):
        if m.width > w:
            m.scale_to_fit_width(w)
        return m


def pipeline(scene, n_layers="× 3 layers", w=1.62, h=0.95, size=26, loop=True):
    """TinyGPT's forward loop as a row of boxes. Returns (group, parts dict)."""
    spec = [("tokens", BLUE), ("embed", TEAL), ("attention", BLUE), ("MLP", TEAL), ("probabilities", YELLOW),
            ("pick one", PINK)]
    boxes = VGroup()
    for name, col in spec:
        t = scene.T(name, size, col, BOLD)
        bw = max(w, t.width + 0.35)
        b = RoundedRectangle(corner_radius=0.14, width=bw, height=h, stroke_color=col, stroke_width=2.5,
                             fill_color=col, fill_opacity=0.1)
        boxes.add(VGroup(b, t.move_to(b)))
    boxes.arrange(RIGHT, buff=0.42)
    arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.05, color=MUTED, stroke_width=3,
                            max_tip_length_to_length_ratio=0.35) for i in range(len(boxes) - 1)])
    br = SurroundingRectangle(VGroup(boxes[2], boxes[3]), buff=0.16, corner_radius=0.18, color=FG, stroke_width=2)
    br.set_stroke(opacity=0.6)
    lay = scene.T(n_layers, 26, FG).next_to(br, UP, buff=0.12)
    parts = dict(boxes=boxes, arrows=arrows, bracket=br, layers=lay)
    g = VGroup(boxes, arrows, br, lay)
    if loop:
        lp = CurvedArrow(boxes[-1].get_bottom() + DOWN * 0.08, boxes[0].get_bottom() + DOWN * 0.08, angle=-PI / 6,
                         color=MUTED, stroke_width=3)
        lt = scene.T("append it & repeat", 24, MUTED).next_to(lp, DOWN, buff=0.12)
        parts.update(loop=lp, loop_label=lt)
        g.add(lp, lt)
    return g, parts


def log_axis(scene, lo=5, hi=12, x0=-3.0, x1=6.4, y_top=2.3, y_bot=-2.0, labels=None):
    """Vertical gridlines for a horizontal log-scale bar chart. Returns (group, x_of(value))."""
    per = (x1 - x0) / (hi - lo)
    labels = labels or {5: "100K", 6: "1M", 7: "10M", 8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T"}
    g = VGroup()
    for e in range(lo, hi + 1):
        x = x0 + (e - lo) * per
        g.add(DashedLine([x, y_top, 0], [x, y_bot, 0], color=LINE, stroke_width=1.5, dash_length=0.08))
        g.add(scene.M(labels.get(e, f"1e{e}"), 22, MUTED).move_to([x, y_bot - 0.25, 0]))

    def x_of(v):
        import math
        return x0 + (math.log10(v) - lo) * per
    return g, x_of
