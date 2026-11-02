import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from lib_research import load, Plot, legend, pill, panel, illus, fmt_lr

VIDEO = dict(id="optimizers", cls="Optimizers", title="Optimizers: learning rate, momentum & Adam",
             subtitle="Why the learning rate is the knob that matters most, and why AdamW is the default (real sweeps)",
             relates="train", track="training", order=40, poster=("Too small, too big", 1, 0.95))

LR = load("lr_sweep")
GR = load("gradients")
STEPS = LR["steps"]
ADAM = [r for r in LR["runs"] if r["optimizer"] == "adamw"]
SGD = [r for r in LR["runs"] if r["optimizer"] == "sgd"]
MOM = [r for r in LR["runs"] if r["optimizer"] == "sgd-momentum"]


def best(runs):
    ok = [r for r in runs if not r["diverged"]]
    return min(ok, key=lambda r: r["final_val"])


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def spk(lr):
    """Learning rate as words, so TTS doesn't read digits one by one: 1e-4 → 'one e minus four'."""
    if lr >= 0.01:
        return f"{lr:g}"
    e = int(math.floor(math.log10(lr) + 1e-9)); m = round(lr / 10 ** e)
    return f"{WORDS[m]} e minus {WORDS[-e]}"


def short_name(n):
    """'blocks.0.ln1.weight' → 'block1.ln1.w' (1-indexed like the other videos)."""
    if n.startswith("blocks."):
        _, i, rest = n.split(".", 2)
        n = f"block{int(i) + 1}.{rest}"
    return n.replace(".weight", ".w").replace(".bias", ".b")


def fv(r):
    return "NaN" if (r["final_val"] is None or (r["diverged"] and len(r["curve"]["step"]) < 3)) else f"{r['final_val']:.2f}"


A_BEST, S_BEST, M_BEST = best(ADAM), best(SGD), best(MOM)
A_LO = ADAM[0]
A_HI = ADAM[-1]
LR_COLS = ["#5fd0ff", BLUE, "#b48cff", TEAL, YELLOW, ORANGE, PINK]


class Optimizers(VScene):
    VIDEO_ID = "optimizers"

    def lr_plot(self, ymax=6.0, w=10.0, h=4.6):
        return Plot([0, STEPS], [0, ymax], width=w, height=h, xticks=[0, 250, 500, 750, 1000],
                    yticks=[0, 1, 2, 3, 4, 5, 6], xlabel="training step", ylabel="validation loss")

    def construct(self):
        # ------------------------------------------------------------ hook: real fan of curves
        pl = self.lr_plot(w=8.6).move_to(DOWN * 0.35 + LEFT * 1.9)
        curves = VGroup(*[pl.line(r["curve"]["step"], r["curve"]["val"], c, 3.5) for r, c in zip(ADAM, LR_COLS)])
        labs = VGroup()
        for r, c in zip(ADAM, LR_COLS):
            labs.add(VGroup(Line(ORIGIN, RIGHT * 0.4, color=c, stroke_width=5),
                            self.M(f"lr {fmt_lr(r['lr']):>5}", 22, c), self.M(fv(r), 22, FG)).arrange(RIGHT, buff=0.18))
        labs.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(pl.ax, RIGHT, buff=0.5)
        lhead = self.T("AdamW, final val loss", 22, MUTED).next_to(labs, UP, buff=0.25).align_to(labs, LEFT)
        cav0 = self.T(f"sweep setup: constant learning rate · no warmup · no clipping · 1 seed · {STEPS:,} steps",
                      22, MUTED).to_edge(DOWN, buff=0.3)
        with self.say(f"Same model, same data, same {STEPS:,} steps. Only one number changed: the learning rate. And it "
                      f"decides everything.",
                      scene=f"Real validation-loss curves of TinyGPT trained {STEPS} steps with AdamW at 7 learning rates "
                            f"(1e-4 … 0.1): " + ", ".join(f"lr {fmt_lr(r['lr'])} → {fv(r)}" for r in ADAM) + ".",
                      chapter="One knob"):
            self.play(FadeIn(pl), FadeIn(lhead), FadeIn(cav0), run_time=0.6)
            self.mark("sweep-setup", cav0, f"sweep setup: constant LR, no warmup, no gradient clipping, seed 0, {STEPS} steps")
            self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.12), LaggedStart(
                *[FadeIn(l, shift=LEFT * 0.2) for l in labs], lag_ratio=0.12), run_time=2.6)
            for r, c, l in zip(ADAM, curves, labs):
                self.mark(f"adam-{fmt_lr(r['lr'])}", VGroup(c, l) if False else l,
                          f"AdamW lr {fmt_lr(r['lr'])}: final val loss {fv(r)}" + (" (diverged)" if r["diverged"] else ""))
            self.mark("fan", curves, f"7 real AdamW runs, {STEPS} steps each, differing only in learning rate")
            self.sync("decides", 0.1)
            self.flash("the learning rate", YELLOW, 60, hold=0.6, pos=UP * 2.6)

        # ------------------------------------------------------------ downhill (toy)
        hdr = self.header("Downhill")
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 5, 1], x_length=10, y_length=4.2, tips=False,
                  axis_config={"color": LINE, "stroke_width": 2}).move_to(DOWN * 0.2)
        f = lambda x: 0.35 * x ** 2 + 0.5 * math.sin(2.0 * x) + 1.1 + 0.1 * x
        df = lambda x: 0.7 * x + 1.0 * math.cos(2.0 * x) + 0.1
        land = ax.plot(f, x_range=[-3, 3, 0.02], color=BLUE, stroke_width=4)
        tag = illus("toy loss landscape · illustration").next_to(ax, DOWN, buff=0.2)
        x0 = -2.5
        ball = Dot(ax.c2p(x0, f(x0)) + UP * 0.17, radius=0.17, color=YELLOW)
        tan_len = 1.2
        tang = Line(ax.c2p(x0 - tan_len, f(x0) - tan_len * df(x0)), ax.c2p(x0 + tan_len, f(x0) + tan_len * df(x0)),
                    color=ORANGE, stroke_width=4)
        rule = VGroup(self.M("w", 40, FG), self.M("←", 40, MUTED), self.M("w", 40, FG), self.M("−", 40, MUTED),
                      self.M("lr", 40, YELLOW), self.M("×", 40, MUTED), self.M("slope", 40, ORANGE)).arrange(RIGHT, buff=0.22)
        rule.move_to(UP * 2.5)
        xs = [x0]
        for _ in range(8):
            xs.append(xs[-1] - 0.35 * df(xs[-1]))
        with self.say("Picture the loss as a landscape and the weights as a ball. The gradient is the slope under the ball. "
                      "Step downhill by the learning rate times the slope. Repeat.",
                      scene="Toy 1-D loss landscape (illustration) with a yellow ball, an orange tangent showing the slope, "
                            "and the update rule w ← w − lr × slope; the ball takes 8 steps downhill.", chapter="Downhill"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pl, curves, labs, lhead, cav0)), FadeIn(hdr), run_time=0.4)
            self.play(Create(ax), Create(land), FadeIn(tag), FadeIn(ball, scale=2), run_time=0.8)
            self.mark("landscape", land, "toy loss landscape (illustration, one weight)")
            self.sync("The gradient", 0.1)
            self.play(Create(tang), run_time=0.5)
            self.mark("slope", tang, "the gradient = slope of the loss at the current weight")
            self.sync("Step downhill", 0.1)
            self.play(FadeIn(rule, shift=DOWN * 0.2), run_time=0.5)
            self.mark("rule", rule, "gradient descent update: w ← w − lr × slope")
            self.play(FadeOut(tang), run_time=0.2)
            for a, b in zip(xs, xs[1:]):
                self.play(ball.animate.move_to(ax.c2p(b, f(b)) + UP * 0.17), run_time=0.22)

        # ------------------------------------------------------------ too small / right / too big (toy)
        hdr2 = self.header("Too small, too big")
        panels = VGroup()
        cfgs = [("too small", 0.05, "#5fd0ff"), ("just right", 0.7, TEAL), ("too big", 2.1, PINK)]
        balls, paths = [], []
        for name, klr, col in cfgs:
            a = Axes(x_range=[-2.7, 2.7, 1], y_range=[0, 3.7, 1], x_length=3.7, y_length=2.6, tips=False,
                     axis_config={"color": LINE, "stroke_width": 2})
            bowl = a.plot(lambda x: 0.5 * x * x, x_range=[-2.7, 2.7, 0.02], color=BLUE, stroke_width=3)
            ttl = self.T(name, 30, col, BOLD).next_to(a, UP, buff=0.2)
            sub = self.M(f"lr × curvature = {klr:g}" + (" > 2" if klr > 2 else ""), 22, PINK if klr > 2 else MUTED)
            sub.next_to(a, DOWN, buff=0.3)
            if klr > 2:
                sub = VGroup(sub, self.T("→ diverges", 22, PINK).next_to(sub, DOWN, buff=0.1))
            xs_ = [-1.6]
            for _ in range(7):
                xs_.append(xs_[-1] * (1 - klr))
            xs_ = [max(-2.2, min(2.2, x)) for x in xs_]
            pts = [a.c2p(x, 0.5 * x * x) for x in xs_]
            path = VMobject(stroke_color=col, stroke_width=3).set_points_as_corners(pts)
            b = Dot(pts[0], radius=0.12, color=col)
            panels.add(VGroup(a, bowl, ttl, sub))
            balls.append((b, pts)); paths.append(path)
        panels.arrange(RIGHT, buff=0.8).move_to(DOWN * 0.1)
        for (b, pts), path, pnl in zip(balls, paths, panels):
            pass
        # re-derive points after arrange
        balls, paths = [], []
        for (name, klr, col), pnl in zip(cfgs, panels):
            a = pnl[0]
            xs_ = [-1.6]
            for _ in range(5):
                xs_.append(xs_[-1] * (1 - klr))
            pts = [a.c2p(x, 0.5 * x * x) for x in xs_]
            paths.append(VMobject(stroke_color=col, stroke_width=3).set_points_as_corners(pts))
            balls.append((Dot(pts[0], radius=0.12, color=col), pts))
        tag2 = illus("toy bowl · illustration").to_edge(DOWN, buff=0.5)
        with self.say("Too small: you crawl. Just right: you settle in a few steps. Too big: every step overshoots, "
                      "bouncing higher and higher.",
                      scene="Three toy bowls (illustration): too small (creeps), just right (settles), too big "
                            "(overshoots, bouncing out of the valley).", chapter="Too small, too big"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax, land, tag, ball, rule)), ReplacementTransform(hdr, hdr2), run_time=0.4)
            self.play(FadeIn(panels), FadeIn(tag2), *[FadeIn(b) for b, _ in balls], run_time=0.6)
            for k, ph in enumerate(["Too small", "Just right", "Too big"]):
                self.sync(ph, 0.1)
                b, pts = balls[k]
                self.play(Create(paths[k]), MoveAlongPath(b, paths[k].copy()), run_time=1.1, rate_func=linear)
                self.mark(f"toy-{k}", panels[k], f"toy illustration: learning rate {cfgs[k][0]}")

        pl2 = self.lr_plot(w=8.6).move_to(DOWN * 0.35 + LEFT * 2.0)
        trio = [(A_LO, "#5fd0ff", "too small"), (A_BEST, TEAL, "just right"), (A_HI, PINK, "too big")]
        cs = VGroup(*[pl2.line(r["curve"]["step"], r["curve"]["val"], c, 4) for r, c, _ in trio])
        tl = VGroup()
        for (r, c, name) in trio:
            tl.add(VGroup(self.T(name, 26, c, BOLD), self.M(f"lr {fmt_lr(r['lr'])} → {fv(r)}", 24, c))
                   .arrange(DOWN, aligned_edge=LEFT, buff=0.08))
        tl.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(pl2.ax, RIGHT, buff=0.45)
        hi_note = ("its loss blew up" if A_HI["diverged"] else f"it thrashes around and ends at {fv(A_HI)}")
        with self.say(f"The real runs agree. Learning rate {spk(A_LO['lr'])} is still crawling at {fv(A_LO)}. "
                      f"{spk(A_BEST['lr'])} reaches {fv(A_BEST)}. At {spk(A_HI['lr'])}, {hi_note}.",
                      scene=f"Real AdamW validation curves: lr {fmt_lr(A_LO['lr'])} (too small) ends at {fv(A_LO)}, "
                            f"lr {fmt_lr(A_BEST['lr'])} (best) ends at {fv(A_BEST)}, lr {fmt_lr(A_HI['lr'])} (too big) "
                            f"ends at {fv(A_HI)}{' (diverged)' if A_HI['diverged'] else ''}."):
            self.play(FadeOut(VGroup(panels, tag2, *paths, *[b for b, _ in balls])), run_time=0.4)
            self.play(FadeIn(pl2), run_time=0.4)
            for k, ph in enumerate(["Learning rate", f"{spk(A_BEST['lr'])} reaches", f"At {spk(A_HI['lr'])}"]):
                self.sync(ph, 0.1)
                self.play(Create(cs[k]), FadeIn(tl[k]), run_time=0.9, rate_func=linear)
                r = trio[k][0]
                self.mark(f"real-{k}", VGroup(cs[k], tl[k]),
                          f"real AdamW run, lr {fmt_lr(r['lr'])}: final val {fv(r)}, best {r['best_val']:.3f}")

        # ------------------------------------------------------------ momentum (toy 2-D valley)
        hdr3 = self.header("Momentum")
        A_, B_, SY = 0.2, 10.0, 1.6
        def run(lr, beta, n=30):
            p = np.array([-4.5, 0.5]); v = np.zeros(2); P = [p.copy()]
            for _ in range(n):
                g = np.array([A_ * p[0], B_ * p[1]]); v = beta * v + g; p = p - lr * v; P.append(p.copy())
            return P
        def c2(p, y0):
            return np.array([p[0] + 0.9, p[1] * SY + y0, 0])
        def ell(level, y0, col):
            a = math.sqrt(2 * level / A_); b = math.sqrt(2 * level / B_) * SY
            return Ellipse(width=2 * a, height=2 * b, color=col, stroke_width=1.5, stroke_opacity=0.7).move_to([0.9, y0, 0])
        rows = VGroup()
        trails = []
        for name, lr, beta, col, y0 in [("plain gradient descent (lr 0.19)", 0.19, 0.0, ORANGE, 1.15),
                                        ("same lr + momentum (β = 0.5)", 0.19, 0.5, TEAL, -2.0)]:
            cont = VGroup(*[ell(l, y0, BLUE if l < 0.3 else LINE) for l in (0.08, 0.6, 1.6, 3.4)])
            P = run(lr, beta)
            pts = [c2(p, y0) for p in P]
            tr = VMobject(stroke_color=col, stroke_width=3.5).set_points_as_corners(pts)
            dots = VGroup(*[Dot(p, radius=0.05, color=col) for p in pts])
            star = Star(5, outer_radius=0.14, color=YELLOW, fill_opacity=1).move_to([0.9, y0, 0])
            lab = self.T(name, 26, col, BOLD).move_to([0, y0 + 1.25, 0]).align_to([-6.6, 0, 0], LEFT)
            res = self.M(f"after 30 steps: {np.linalg.norm(P[-1]):.2f} from the ★", 22, col)
            res.next_to(lab, RIGHT, buff=0.5)
            rows.add(VGroup(cont, star, lab, res)); trails.append((tr, dots))
        tag3 = illus("toy valley · illustration").to_corner(DR, buff=0.4)
        with self.say("Loss landscapes often have long, narrow valleys. Plain gradient descent zigzags across the walls and "
                      "creeps along the floor. Momentum adds up past gradients: the back-and-forth parts cancel, and the "
                      "steady downhill part builds speed.",
                      scene="Toy 2-D elongated valley (illustration, contour ellipses), same learning rate 0.19 for both. "
                            "Top: plain gradient descent zigzags for all 30 steps and ends 1.41 from the minimum; bottom: "
                            "with momentum (β=0.5, v = 0.5·v + gradient) the zigzag dies out within ~8 steps and it ends "
                            "0.32 from the minimum.",
                      chapter="Momentum"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pl2, cs, tl)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(rows[0][0]), FadeIn(rows[0][1]), FadeIn(rows[1][0]), FadeIn(rows[1][1]), FadeIn(tag3),
                      run_time=0.6)
            self.sync("Plain", 0.1)
            self.play(FadeIn(rows[0][2]), Create(trails[0][0]), FadeIn(trails[0][1], lag_ratio=0.1), run_time=1.6)
            self.play(FadeIn(rows[0][3]), run_time=0.3)
            self.mark("gd-path", VGroup(trails[0][0], rows[0][2]), "toy: plain gradient descent zigzags across the valley")
            self.sync("Momentum adds", 0.1)
            self.play(FadeIn(rows[1][2]), Create(trails[1][0]), FadeIn(trails[1][1], lag_ratio=0.1), run_time=1.6)
            self.play(FadeIn(rows[1][3]), run_time=0.3)
            self.mark("mom-path", VGroup(trails[1][0], rows[1][2]), "toy: momentum sums past gradients; zigzags cancel, speed builds along the floor")
        pl3 = self.lr_plot(w=10.5).move_to(DOWN * 0.35)
        pair = [(S_BEST, ORANGE, "SGD"), (M_BEST, TEAL, "SGD + momentum 0.9")]
        cs3 = VGroup(*[pl3.line(r["curve"]["step"], r["curve"]["val"], c, 4) for r, c, _ in pair])
        lg3 = VGroup(*[VGroup(self.T(n, 24, c, BOLD), self.M(f"best lr {fmt_lr(r['lr'])} → {fv(r)}", 22, c))
                       .arrange(RIGHT, buff=0.2) for r, c, n in pair]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        lg3.move_to(pl3.p(STEPS, 5.5), aligned_edge=RIGHT)
        with self.say(f"On the real model, the best plain SGD ends at {fv(S_BEST)}. With momentum, {fv(M_BEST)}, at a "
                      f"lower learning rate, since momentum sums gradients.",
                      scene=f"Real runs, best learning rate of each: SGD (lr {fmt_lr(S_BEST['lr'])}) → {fv(S_BEST)}, "
                            f"SGD + momentum 0.9 (lr {fmt_lr(M_BEST['lr'])}) → {fv(M_BEST)}."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows, tag3, *[t for p in trails for t in p])), FadeIn(pl3), run_time=0.5)
            self.play(Create(cs3[0]), FadeIn(lg3[0]), run_time=0.9, rate_func=linear)
            self.mark("sgd", VGroup(cs3[0], lg3[0]), f"real: best plain SGD (lr {fmt_lr(S_BEST['lr'])}) final val {fv(S_BEST)}")
            self.sync("With momentum", 0.1)
            self.play(Create(cs3[1]), FadeIn(lg3[1]), run_time=0.9, rate_func=linear)
            self.mark("mom", VGroup(cs3[1], lg3[1]), f"real: best SGD+momentum (lr {fmt_lr(M_BEST['lr'])}) final val {fv(M_BEST)}")

        # ------------------------------------------------------------ Adam: per-parameter step sizes
        hdr4 = self.header("Adam")
        pick = ["blocks.0.ln1.weight", "blocks.0.attn_qkv.weight", "pos.weight", "blocks.1.mlp_in.weight",
                "ln_f.weight", "tok.weight", "blocks.1.attn_out.bias", "blocks.0.attn_out.bias"]
        pt = {t["name"]: t for t in GR["per_tensor"]}
        vals = [pt[n]["grad_rms_step0"] for n in pick]
        lo_, hi_ = min(vals), max(vals)
        spread = hi_ / lo_
        bp = Plot([1e-5, 1e-1], [0, len(pick)], width=6.6, height=4.4, xlog=True, xticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                  yticks=[], xlabel="typical gradient size per weight (log scale)", ylabel="",
                  xfmt=lambda v: f"1e{int(round(math.log10(v)))}", grid=True).move_to(RIGHT * 2.3 + DOWN * 0.5)
        bars = VGroup(); names = VGroup()
        for k, (n, v) in enumerate(zip(pick, vals)):
            y = len(pick) - k - 0.5
            r = Rectangle(width=max(0.05, bp.p(v, y)[0] - bp.p(1e-5, y)[0]), height=0.36, stroke_width=0,
                          fill_color=BLUE, fill_opacity=0.85)
            r.move_to(bp.p(1e-5, y), aligned_edge=LEFT)
            bars.add(r)
            names.add(self.M(short_name(n), 22, FG)
                      .next_to(bp.p(1e-5, y), LEFT, buff=0.2))
        sp = self.T(f"{spread:.0f}× spread", 34, YELLOW, BOLD).next_to(bp.ax, UP, buff=0.5).align_to(bp.ax, RIGHT)
        src = self.T("real gradients at initialisation (step 0) · one batch", 22, MUTED).next_to(sp, LEFT, buff=0.6).align_to(sp, DOWN)
        tr_vals = [t["grad_rms"] for t in GR["per_tensor"]]
        all0 = [t["grad_rms_step0"] for t in GR["per_tensor"]]
        spread_tr = max(tr_vals) / min(tr_vals)
        trn = self.T(f"after training the spread is still {spread_tr:.0f}×", 22, MUTED).next_to(bp.xt, DOWN, buff=0.2)
        with self.say(f"Here's the catch. At step zero, typical gradient sizes differ {spread:.0f} times between weight "
                      f"tensors. One learning rate can't suit them all.",
                      scene=f"Real per-weight gradient size (RMS of the gradient) for 8 of TinyGPT's tensors at "
                            f"initialisation, log scale: from {lo_:.1e} ({pick[0]}) to {hi_:.1e} ({pick[-1]}), a "
                            f"{spread:.0f}× spread.", chapter="Adam"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pl3, cs3, lg3)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(FadeIn(bp), FadeIn(names), FadeIn(src), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1),
                      run_time=1.4)
            for n, b, nm, v in zip(pick, bars, names, vals):
                self.mark("g-" + n, VGroup(b, nm), f"{n}: gradient RMS {v:.2e} per weight at step 0")
            self.sync("differ", 0.1)
            self.play(FadeIn(sp, scale=1.2), FadeIn(trn), run_time=0.4)
            self.mark("spread", sp, f"largest / smallest gradient size at step 0 = {spread:.0f}× "
                                    f"(over all {len(all0)} tensors: {max(all0) / min(all0):.0f}×)")
            self.mark("spread-trained", trn, f"on the trained model the spread is {spread_tr:.0f}×")

        den_t = self.M("avg(gradient²)", 32, TEAL)
        root = self.M("√", 40, TEAL).next_to(den_t, LEFT, buff=0.02).align_to(den_t, DOWN)
        vinc = Line(root.get_corner(UR) + LEFT * 0.03, [den_t.get_right()[0], root.get_top()[1], 0], color=TEAL, stroke_width=2.5)
        den = VGroup(root, vinc, den_t)
        num = self.M("avg(gradient)", 32, ORANGE)
        form = VGroup(self.M("step", 36, FG), self.M("=", 36, MUTED), self.M("lr", 36, YELLOW), self.M("×", 36, MUTED),
                      VGroup(num, Line(LEFT * 1.85, RIGHT * 1.85, color=FG, stroke_width=2.5), den).arrange(DOWN, buff=0.3))
        form.arrange(RIGHT, buff=0.25)
        form.move_to(UP * 0.8)
        f1 = self.T("momentum: the direction, smoothed", 26, ORANGE).next_to(form, UP, buff=0.5).shift(RIGHT * 2.0)
        f2 = self.T("each weight's own typical gradient size", 26, TEAL).next_to(form, DOWN, buff=0.5).shift(RIGHT * 2.0)
        res = self.T("→ each weight moves at most ~lr per step, whatever its gradient scale", 28, FG).move_to(DOWN * 2.1)
        defaults = self.T("PyTorch defaults: β1 = 0.9 (gradient avg), β2 = 0.999 (squared avg), + tiny ε", 22, MUTED)
        defaults.next_to(res, DOWN, buff=0.3)
        with self.say("Adam keeps two running averages for every weight: the gradient, which is momentum, and the "
                      "gradient squared. Dividing one by the square root of the other gives each weight its own step size.",
                      scene="Adam's update: step = lr × avg(gradient) / √avg(gradient²). Top term = momentum (smoothed "
                            "direction); bottom = each weight's own typical gradient size, so each weight moves at most about "
                            "lr per step. PyTorch defaults β1 0.9, β2 0.999, + ε."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(bp, bars, names, sp, src, trn)), run_time=0.4)
            self.play(FadeIn(form, shift=UP * 0.2), run_time=0.6)
            self.mark("adam-rule", form, "Adam: step = lr × avg(g) / sqrt(avg(g²)) (+ small ε), per weight")
            self.sync("the gradient,", 0.1)
            self.play(FadeIn(f1), Indicate(form[4][0], color=ORANGE), run_time=0.6)
            self.sync("gradient squared", 0.1)
            self.play(FadeIn(f2), Indicate(form[4][2], color=TEAL), run_time=0.6)
            self.sync("Dividing", 0.1)
            self.play(FadeIn(res, shift=UP * 0.2), FadeIn(defaults), run_time=0.5)
            self.mark("adam-result", res, "Adam normalises step sizes per weight: at most about lr when the gradient "
                                          "sign is consistent, smaller when it is noisy")
            self.mark("adam-defaults", defaults, "Adam hyper-parameters: beta1 0.9, beta2 0.999, epsilon 1e-8, with bias "
                                                 "correction for the first steps")

        # ------------------------------------------------------------ why AdamW: robustness map
        hdr5 = self.header("Why AdamW")
        groups = [("AdamW", ADAM, BLUE), ("SGD + momentum", MOM, TEAL), ("SGD", SGD, ORANGE)]
        good = A_BEST["final_val"] * 1.25
        grid = VGroup()
        for name, runs, col in groups:
            row = VGroup(self.T(name, 28, col, BOLD).set_width(min(3.2, self.T(name, 28).width)))
            cells = VGroup()
            for r in runs:
                ok = (not r["diverged"]) and r["final_val"] <= good
                c = RoundedRectangle(corner_radius=0.08, width=1.18, height=0.95, stroke_color=TEAL if ok else LINE,
                                     stroke_width=2.5, fill_color=TEAL if ok else PANEL, fill_opacity=0.25 if ok else 0.8)
                t = VGroup(self.M(fmt_lr(r["lr"]), 22, MUTED), self.M(fv(r), 24, FG if ok else (PINK if r["diverged"] else MUTED)))
                t.arrange(DOWN, buff=0.08).move_to(c)
                cells.add(VGroup(c, t))
            cells.arrange(RIGHT, buff=0.15)
            row.add(cells)
            grid.add(row)
        for row in grid:
            row[0].move_to(LEFT * 5.2, aligned_edge=LEFT) if False else None
        rows_ = VGroup()
        for row in grid:
            row[1].next_to(LEFT * 3.0, RIGHT, buff=0)
            row[0].next_to(row[1], LEFT, buff=0.35)
            rows_.add(row)
        rows_.arrange(DOWN, buff=0.35, aligned_edge=RIGHT).move_to(DOWN * 0.1)
        for row in rows_:
            row[1].move_to([-3.25, row[1].get_y(), 0], aligned_edge=LEFT)
            row[0].next_to(row[1], LEFT, buff=0.35)
        n_ok = {name: sum(1 for r in runs if not r["diverged"] and r["final_val"] <= good) for name, runs, _ in groups}
        cap = VGroup(self.T(f"each box: learning rate → final val loss · teal = within 25% of the best ({good:.2f})", 22, MUTED),
                     self.T(f"one sweep: 1 seed, {STEPS:,} steps, constant learning rate", 22, MUTED)).arrange(DOWN, buff=0.1)
        cap.to_edge(DOWN, buff=0.35)
        ok_lrs = [r["lr"] for r in ADAM if not r["diverged"] and r["final_val"] <= good]
        span = max(ok_lrs) / min(ok_lrs)
        ok_m = [r["lr"] for r in MOM if not r["diverged"] and r["final_val"] <= good]
        span_m = max(ok_m) / min(ok_m) if ok_m else 0
        wd = VGroup(self.T("weight decay: shrink every weight a little each step", 22, BLUE),
                    self.T("AdamW applies it to the weights directly, not through Adam's rescaled gradient", 22, BLUE)) \
            .arrange(DOWN, buff=0.08).next_to(rows_, UP, buff=0.4)
        with self.say(f"AdamW adds weight decay, a small shrink of every weight each step, applied directly to the "
                      f"weights. In this one sweep its good range spans {span:.0f} times; momentum SGD's, {span_m:.0f}; plain "
                      f"SGD never got close. That forgiveness is a big reason it's the usual default.",
                      scene=f"Grid of every real sweep run (learning rate → final val loss). Teal boxes are within 25% of "
                            f"the best ({good:.2f}): AdamW {n_ok['AdamW']}/{len(ADAM)}, SGD+momentum "
                            f"{n_ok['SGD + momentum']}/{len(MOM)}, SGD {n_ok['SGD']}/{len(SGD)}.", chapter="Why AdamW"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(form, f1, f2, res, defaults)), ReplacementTransform(hdr4, hdr5), run_time=0.4)
            for k, row in enumerate(rows_):
                self.play(FadeIn(row[0]), LaggedStart(*[FadeIn(c, scale=0.9) for c in row[1]], lag_ratio=0.08),
                          run_time=0.7)
                name, runs, _ = groups[k]
                for r, c in zip(runs, row[1]):
                    self.mark(f"cell-{name}-{fmt_lr(r['lr'])}", c, f"{name} lr {fmt_lr(r['lr'])}: final val {fv(r)}"
                              + (" (diverged)" if r["diverged"] else ""))
            self.play(FadeIn(cap), FadeIn(wd), run_time=0.3)
            self.mark("wd", wd, "weight decay shrinks each weight by lr × 0.05 × weight per step (AdamW, decoupled)")

        # ------------------------------------------------------------ schedule (the recipe used everywhere else)
        hdr6 = self.header("Schedule")
        S_STEPS, S_LR, WARM = 3000, 3e-3, 100     # the shipped model's recipe (train/train.py)
        sched = lambda st: S_LR * min(1, (st + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * st / S_STEPS))
        ps_ = Plot([0, S_STEPS], [0, 3.2e-3], width=9.0, height=3.8, xticks=[0, 1000, 2000, 3000],
                   yticks=[0, 1e-3, 2e-3, 3e-3], xlabel="training step (shipped TinyGPT recipe)", ylabel="learning rate",
                   yfmt=lambda v: "0" if v == 0 else f"{v * 1e3:g}e-3").move_to(DOWN * 0.4 + LEFT * 0.6)
        xs_s = list(range(0, S_STEPS + 1, 10))
        lr_line = ps_.line(xs_s, [sched(x) for x in xs_s], YELLOW, 4)
        wl = self.T("warm-up: first 100 steps", 24, TEAL).move_to(ps_.p(260, 0.9e-3), aligned_edge=LEFT)
        warm = VGroup(Arrow(wl.get_left() + LEFT * 0.05, ps_.p(60, 1.9e-3), buff=0.05, color=TEAL, stroke_width=3), wl)
        dec = self.T("cosine decay to 0", 24, ORANGE).move_to(ps_.p(2350, 1.4e-3))
        clip = self.T("+ gradient clipping: rescale any step whose gradient norm exceeds 1.0", 22, MUTED).to_edge(DOWN, buff=0.35)
        with self.say("Our real recipe also uses a schedule: warm up for 100 steps while Adam's averages are still noisy, "
                      "then decay along a cosine so the model settles. The sweep kept the rate constant on purpose.",
                      scene=f"Real learning-rate schedule of the shipped TinyGPT recipe (train/train.py): linear warm-up over "
                            f"{WARM} steps to {S_LR:g}, then cosine decay to 0 at step {S_STEPS}; plus gradient clipping at "
                            f"norm 1.0. The sweep in this video used a constant rate instead.", chapter="Schedule"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows_, cap, wd)), ReplacementTransform(hdr5, hdr6), run_time=0.4)
            self.play(FadeIn(ps_), run_time=0.4)
            self.play(Create(lr_line), run_time=1.6, rate_func=linear)
            self.mark("schedule", lr_line, f"learning rate: linear warm-up over {WARM} steps to {S_LR:g}, cosine decay to 0 "
                                           f"at step {S_STEPS}")
            self.sync("warm up", 0.1)
            self.play(FadeIn(warm), run_time=0.4)
            self.mark("warmup", warm, "warm-up: the rate ramps from ~0 to 3e-3 over the first 100 steps")
            self.sync("then decay", 0.1)
            self.play(FadeIn(dec), FadeIn(clip), run_time=0.4)
            self.mark("decay", dec, "cosine decay: lr = 3e-3 × ½(1 + cos(π·step/3000))")
            self.mark("clip", clip, "gradient clipping at total norm 1.0")

        # ------------------------------------------------------------ recap
        tip_txt = [("start with AdamW", BLUE), ("sweep the learning rate ×3 at a time", YELLOW),
                   ("warm up, then decay", TEAL), ("loss spikes? lower it", PINK)]
        tips = VGroup(*[pill(t, c, 30) for t, c in tip_txt]).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        tips.move_to(DOWN * 0.1)
        with self.say("So: start with AdamW, sweep the learning rate in steps of three, warm up then decay, and if the "
                      "loss spikes, turn it down.",
                      scene="Recap pills: start with AdamW; sweep the learning rate ×3 at a time; warm up, then decay; "
                            "loss spikes → lower it.", chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ps_, lr_line, warm, dec, clip)), ReplacementTransform(hdr6, self.header("Recap")),
                      run_time=0.4)
            for k, ph in enumerate(["start with", "sweep", "warm up", "if the"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(tips[k], shift=RIGHT * 0.2), run_time=0.3)
                self.mark(f"tip-{k}", tips[k], "recap: " + tip_txt[k][0])
        self.wait(0.6)
