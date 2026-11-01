import os, sys, textwrap, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from model import VOCAB, TRAIN, META, tok

PROMPT = TRAIN["prompts"][0]                      # "mia has a red ball. tom"
SNAP = {s["step"]: s["samples"][0] for s in TRAIN["snapshots"]}
STEPS, LOSS, VAL = TRAIN["steps"], TRAIN["train"], TRAIN["val"]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Training(VScene):
    VIDEO_ID = "training"

    def sample_block(self, step, width=46, size=28, maxlen=None, hl=None):
        """Prompt (muted) + model continuation (bright), manually wrapped. hl: (substring, color) to highlight."""
        full = SNAP[step].replace("\n", " ↵ ")
        if maxlen:
            full = full[:maxlen].rstrip() + "…"
        lines = textwrap.wrap(full, width)
        out, pos = VGroup(), 0
        for ln in lines:
            i = full.index(ln, pos)
            j = i + len(ln)
            pos = j
            a = max(0, min(len(ln), len(PROMPT) - i))
            body = esc(ln[a:])
            if hl and hl[0] in body:
                body = body.replace(hl[0], f'<span foreground="{hl[1]}" weight="bold">{hl[0]}</span>')
            mk = f'<span foreground="{MUTED}">{esc(ln[:a])}</span><span foreground="{FG}">{body}</span>'
            out.add(MarkupText(mk, font=MONO, font_size=size))
        out.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        return out

    def construct(self):
        loss_at = {s: l for s, l in zip(STEPS, LOSS)}
        params = TRAIN["params"]

        # ------------------------------------------------------------ hook
        b0 = self.sample_block(0, maxlen=96)
        b3 = self.sample_block(3000, maxlen=96)
        l0 = self.T("step 0", 30, PINK, BOLD)
        l3 = self.T("step 3000", 30, TEAL, BOLD)
        g0 = VGroup(l0, b0).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        g3 = VGroup(l3, b3).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        VGroup(g0, g3).arrange(DOWN, aligned_edge=LEFT, buff=0.8).move_to(ORIGIN)
        with self.say("At step zero, this model wrote this. 80 seconds of training later, it wrote this. "
                      "What happened in between?",
                      scene="Two real samples from the prompt 'mia has a red ball. tom': at step 0 the continuation is "
                            "gibberish ('sh hill w timog say litam…'); at step 3000 it is a coherent story ending "
                            "'now tom has the red ball.'", chapter="Gibberish to stories"):
            self.play(FadeIn(l0), AddTextLetterByLetter(b0[0]) if False else FadeIn(b0, shift=UP * 0.2), run_time=0.7)
            self.mark("sample-0", g0, f"real sample at step 0 (loss {loss_at[0]:.2f}): random gibberish")
            self.sync("80 seconds", 0.1)
            self.play(FadeIn(l3), FadeIn(b3, shift=UP * 0.2), run_time=0.7)
            self.mark("sample-3000", g3, f"real sample at step 3000 (loss {loss_at[3000]:.2f}): coherent text")

        # ------------------------------------------------------------ random start
        hdr = self.header("Random start")
        n = 256
        bars = VGroup(*[Rectangle(width=12.0 / n * 0.75, height=1.2 * (1 + 0.08 * math.sin(i * 1.7) * math.cos(i * 0.31)),
                                  fill_color=BLUE, fill_opacity=0.8, stroke_width=0) for i in range(n)])
        bars.arrange(RIGHT, buff=12.0 / n * 0.25, aligned_edge=DOWN).move_to(UP * 0.6)
        cap = self.T("≈ 1/256 = 0.39% for every token", 30, FG).next_to(bars, UP, buff=0.4)
        fact = VGroup(self.T("loss at step 0:", 30, MUTED), self.M(f"{loss_at[0]:.2f}", 36, PINK),
                      self.T("pure guessing would be  ln 256 =", 30, MUTED), self.M(f"{math.log(256):.2f}", 36, TEAL))
        fact.arrange(RIGHT, buff=0.25).move_to(DOWN * 1.7)
        with self.say("Every weight starts as tiny random noise. So at first, it spreads its bets almost evenly over "
                      "all 256 tokens. Its loss is 5.56, almost exactly what pure guessing scores.",
                      scene="256 near-equal bars (each ≈ 0.39%): the untrained model's flat guess. Below: real loss at "
                            "step 0 = 5.56 vs ln 256 = 5.55 for uniform guessing.", chapter="Random start"):
            self.unmark("sample-0", "sample-3000")
            self.play(FadeOut(VGroup(g0, g3)), FadeIn(hdr), run_time=0.5)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.004), FadeIn(cap), run_time=1.4)
            self.mark("flat", bars, "untrained model: ~uniform probability over 256 tokens (illustration)")
            self.sync("Its loss", 0.1)
            self.play(FadeIn(fact, shift=UP * 0.2), run_time=0.6)
            self.mark("loss0", fact, f"real loss at step 0 = {loss_at[0]:.4f} vs ln(256) = {math.log(256):.4f}")

        # ------------------------------------------------------------ the game
        hdr2 = self.header("The loss")
        s = "mia has a red ball."
        ids = tok(s)
        tk = [VOCAB[i] for i in ids]
        inp = VGroup(*[self.chip(t, CYCLE[k % len(CYCLE)], 32) for k, t in enumerate(tk[:-1])]).arrange(RIGHT, buff=0.5)
        tgt = VGroup(*[self.chip(t, CYCLE[(k + 1) % len(CYCLE)], 32) for k, t in enumerate(tk[1:])])
        for a, b in zip(inp, tgt):
            b.next_to(a, DOWN, buff=1.0)
        VGroup(inp, tgt).move_to(UP * 1.0)
        arrows = VGroup(*[Arrow(a.get_bottom(), b.get_top(), buff=0.1, color=MUTED, stroke_width=3) for a, b in zip(inp, tgt)])
        li = self.T("sees", 26, MUTED).next_to(inp, LEFT, buff=0.4)
        lt = self.T("must predict", 26, MUTED).next_to(tgt, LEFT, buff=0.4)
        form = VGroup(self.T("cross-entropy loss =", 34, FG), self.M("−log( p[correct next token] )", 34, YELLOW))
        form.arrange(RIGHT, buff=0.3).move_to(DOWN * 2.1)
        with self.say("Training is one game. At every position, predict the next token. The score is cross-entropy, "
                      "minus the log of the probability you gave the right answer.",
                      scene="Tokens of 'mia has a red ball.' on top, each with an arrow down to the token that follows it "
                            "(the target). Formula: cross-entropy loss = −log(p[correct next token]).", chapter="The loss"):
            self.unmark("flat", "loss0")
            self.play(FadeOut(VGroup(bars, cap, fact)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(li), LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in inp], lag_ratio=0.1), run_time=0.8)
            self.play(FadeIn(lt), LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(b, shift=DOWN * 0.2))
                                                for a, b in zip(arrows, tgt)], lag_ratio=0.15), run_time=1.2)
            for k, (a, b) in enumerate(zip(tk[:-1], tk[1:])):
                self.mark(f"pair-{k}", VGroup(inp[k], tgt[k]), f'after "{a}" the correct next token is "{b}"')
            self.sync("The score", 0.1)
            self.play(FadeIn(form, shift=UP * 0.2), run_time=0.6)
            self.mark("formula", form, "cross-entropy loss = −log p(correct token)")

        # ------------------------------------------------------------ -log p curve
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 7, 1], x_length=7.0, y_length=4.6, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_numbers": False}).move_to(RIGHT * 0.6 + DOWN * 0.45)
        xl = VGroup(*[self.M(f"{v:g}", 22, MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in (0, 0.5, 1)])
        yl = VGroup(*[self.M(str(v), 22, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0, 2, 4, 6)])
        xt = self.T("probability given to the right token", 24, MUTED).next_to(ax, DOWN, buff=0.55)
        yt = self.T("loss", 24, MUTED).next_to(ax.c2p(0, 7), LEFT, buff=0.2)
        curve = ax.plot(lambda p: -math.log(p), x_range=[0.0009, 1, 0.002], color=YELLOW, stroke_width=4)
        pts = [(0.87, PINK, '87% on " red"'), (1 / 256, BLUE, "1 in 256"), (0.001, ORANGE, "confidently wrong (0.1%)")]
        dots = VGroup()
        labs = VGroup()
        for p, c, t in pts:
            d = Dot(ax.c2p(p, -math.log(p)), radius=0.1, color=c)
            l = VGroup(self.T(t, 24, c), self.M(f"loss {-math.log(p):.2f}", 24, c)).arrange(DOWN, aligned_edge=LEFT,
                                                                                         buff=0.06)
            dots.add(d)
            labs.add(l)
        labs[0].next_to(dots[0], UP, buff=0.25).shift(LEFT * 0.6)
        labs[1].next_to(dots[1], RIGHT, buff=0.3).shift(DOWN * 0.35)
        labs[2].next_to(dots[2], RIGHT, buff=0.3).shift(UP * 0.25)
        with self.say("Give the right token 87 percent, and the loss is 0.14. Give it one in 256, and it's 5.55. "
                      "Confidently wrong is the most expensive of all.",
                      scene="Plot of loss = −log p: p = 0.87 → 0.14 (pink), p = 1/256 → 5.55 (blue), p = 0.001 → 6.91 "
                            "(orange, 'confidently wrong')."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(inp, tgt, arrows, li, lt)), form.animate.scale(0.8).move_to(UP * 3.25 + RIGHT * 0.8), run_time=0.6)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), run_time=0.6)
            self.play(Create(curve), run_time=0.8)
            self.mark("curve", curve, "loss = −log(p)")
            for k, ph in enumerate(["87 percent", "one in 256", "Confidently"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(dots[k], scale=2), FadeIn(labs[k]), run_time=0.4)
                p = pts[k][0]
                self.mark(f"pt-{k}", VGroup(dots[k], labs[k]), f"p = {p:.4g} → loss {-math.log(p):.2f}")

        # ------------------------------------------------------------ gradient descent
        hdr3 = self.header("Gradient descent")
        ax2 = Axes(x_range=[-3, 3, 1], y_range=[0, 5, 1], x_length=10, y_length=4.4, tips=False,
                   axis_config={"color": LINE, "stroke_width": 2}).move_to(DOWN * 0.4)
        f = lambda x: 0.35 * x ** 2 + 0.6 * math.sin(2.2 * x) + 1.2 + 0.12 * x
        df = lambda x: 0.7 * x + 1.32 * math.cos(2.2 * x) + 0.12
        land = ax2.plot(f, x_range=[-3, 3, 0.02], color=BLUE, stroke_width=4)
        note = self.T("a 1-D slice for intuition · the real landscape has " + f"{params:,} dimensions", 24, MUTED)
        note.next_to(ax2, DOWN, buff=0.3)
        xs = [-2.6]
        for _ in range(9):
            xs.append(xs[-1] - 0.45 * df(xs[-1]))
        ball = Dot(ax2.c2p(xs[0], f(xs[0])) + UP * 0.17, radius=0.17, color=YELLOW)
        lw = self.T("loss", 24, MUTED).next_to(ax2.c2p(-3, 5), RIGHT, buff=0.1)
        ww = self.T("a weight →", 24, MUTED).next_to(ax2.c2p(3, 0), UP, buff=0.15).shift(LEFT * 0.6)
        with self.say(f"Gradient descent works out which way to nudge each of the {params:,} weights to shrink the loss, "
                      f"takes a small step, and repeats. Like a ball rolling downhill.",
                      scene="A bumpy 1-D loss curve (illustration). A yellow ball takes gradient steps downhill, each step "
                            "shown with a slope arrow, settling near a low point.", chapter="Gradient descent"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax, xl, yl, xt, yt, curve, dots, labs, form)), ReplacementTransform(hdr2, hdr3),
                      run_time=0.5)
            self.play(Create(ax2), Create(land), FadeIn(note), FadeIn(lw), FadeIn(ww), run_time=0.9)
            self.play(FadeIn(ball, scale=2), run_time=0.3)
            self.mark("landscape", land, "loss as a function of one weight (illustrative)")
            self.mark("ball", ball, "current weights; each step moves against the gradient")
            for a, b in zip(xs, xs[1:]):
                slope = Arrow(ax2.c2p(a, f(a)), ax2.c2p(b, f(b)), buff=0, color=ORANGE, stroke_width=4,
                              max_tip_length_to_length_ratio=0.3)
                self.play(GrowArrow(slope), run_time=0.15)
                self.play(ball.animate.move_to(ax2.c2p(b, f(b)) + UP * 0.17), FadeOut(slope), run_time=0.35)
            self.sync("Like a ball", 0.1)
            self.play(Indicate(ball, color=YELLOW, scale_factor=1.4), run_time=0.6)

        # ------------------------------------------------------------ batches
        B = TRAIN["batch"]
        ctx = META["config"]["n_ctx"]
        sq = VGroup(*[Square(0.07, stroke_width=0, fill_color=TEAL, fill_opacity=0.85) for _ in range(B * ctx)])
        sq.arrange_in_grid(rows=B, cols=ctx, buff=0.012).move_to(LEFT * 3.5 + DOWN * 0.3)
        r1 = self.T(f"{B} snippets × {ctx} tokens", 34, FG)
        r2 = self.T(f"= {B * ctx:,} predictions per step", 34, TEAL, BOLD)
        r3 = self.T(f"× 3,000 steps", 34, FG)
        r4 = self.T(f"≈ {B * ctx * 3000 / 1e6:.1f} million graded guesses", 34, YELLOW, BOLD)
        rr = VGroup(r1, r2, r3, r4).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.0 + DOWN * 0.3)
        with self.say(f"Each step grades {B} random snippets of {ctx} tokens at once. That's 4,096 predictions per step, "
                      f"for 3000 steps.",
                      scene=f"A {B}×{ctx} grid of tiny squares = {B * ctx:,} next-token predictions per training step; "
                            f"× 3,000 steps ≈ {B * ctx * 3000 / 1e6:.1f} million graded guesses."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax2, land, note, ball, lw, ww)), run_time=0.4)
            self.play(FadeIn(sq, lag_ratio=0.0005), FadeIn(r1), run_time=1.0)
            self.mark("batch", sq, f"one batch: {B} snippets × {ctx} tokens = {B * ctx:,} predictions")
            self.sync("That's", 0.1)
            self.play(FadeIn(r2, shift=UP * 0.2), run_time=0.4)
            self.sync("for 3000", 0.1)
            self.play(FadeIn(r3), FadeIn(r4, shift=UP * 0.2), run_time=0.5)
            self.mark("total", VGroup(r3, r4), f"3,000 steps × {B * ctx:,} ≈ {B * ctx * 3000:,} predictions")

        # ------------------------------------------------------------ loss curve
        hdr4 = self.header("The loss curve")
        ax3 = Axes(x_range=[0, 3000, 500], y_range=[0, 6, 1], x_length=10.5, y_length=5.0, tips=False,
                   axis_config={"color": MUTED, "stroke_width": 2}).move_to(DOWN * 0.3 + RIGHT * 0.4)
        xl3 = VGroup(*[self.M(f"{v}", 22, MUTED).next_to(ax3.c2p(v, 0), DOWN, buff=0.15) for v in (0, 1000, 2000, 3000)])
        yl3 = VGroup(*[self.M(f"{v}", 22, MUTED).next_to(ax3.c2p(0, v), LEFT, buff=0.15) for v in (0, 1, 2, 3, 4, 5, 6)])
        xt3 = self.T("training step", 24, MUTED).next_to(xl3, DOWN, buff=0.15)
        yt3 = self.T("loss", 24, MUTED).next_to(ax3, UP, buff=0.15).align_to(ax3, LEFT)
        tr_curve = VMobject(color=YELLOW, stroke_width=4).set_points_as_corners([ax3.c2p(s, l) for s, l in zip(STEPS, LOSS)])
        va_curve = DashedVMobject(VMobject(color=TEAL, stroke_width=2.5).set_points_as_corners(
            [ax3.c2p(s, l) for s, l in zip(STEPS, VAL)]), num_dashes=120)
        chance = DashedLine(ax3.c2p(0, math.log(256)), ax3.c2p(3000, math.log(256)), color=PINK, stroke_width=2,
                            dash_length=0.12)
        chance_l = self.T("pure guessing (5.55)", 22, PINK).next_to(ax3.c2p(3000, math.log(256)), DOWN, buff=0.12)
        chance_l.align_to(ax3.c2p(3000, 0), RIGHT)
        legend = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.5, color=YELLOW, stroke_width=4), self.T("train", 22, YELLOW)).arrange(RIGHT, buff=0.15),
                        VGroup(DashedLine(ORIGIN, RIGHT * 0.5, color=TEAL, stroke_width=2.5, dash_length=0.08),
                               self.T("validation", 22, TEAL)).arrange(RIGHT, buff=0.15)).arrange(RIGHT, buff=0.5)
        legend.next_to(ax3, UP, buff=0.15).align_to(ax3, RIGHT)
        i150 = STEPS.index(150)
        a150 = VGroup(Dot(ax3.c2p(150, LOSS[i150]), color=ORANGE, radius=0.08),
                      self.T(f"step 150: {LOSS[i150]:.2f}", 24, ORANGE))
        a150[1].next_to(a150[0], RIGHT, buff=0.2).shift(UP * 0.25)
        aend = VGroup(Dot(ax3.c2p(3000, LOSS[-1]), color=YELLOW, radius=0.08),
                      self.T(f"step 3000: {LOSS[-1]:.2f}", 24, YELLOW))
        aend[1].next_to(aend[0], UP, buff=0.25).shift(LEFT * 0.9)
        with self.say(f"Here's the real loss curve. It starts at 5.56, right at chance, crashes below 1 by step 150, "
                      f"then slowly grinds down to about {LOSS[-1]:.2f}.",
                      scene=f"The real training curve from training.json (121 points every 25 steps): train loss (yellow) "
                            f"and validation loss (dashed teal) start at 5.56 next to the 'pure guessing' line (5.55), drop "
                            f"to {LOSS[i150]:.2f} by step 150 and flatten out near {LOSS[-1]:.2f} at step 3000.",
                      chapter="The loss curve"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(sq, rr)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(Create(ax3), FadeIn(xl3), FadeIn(yl3), FadeIn(xt3), FadeIn(yt3), FadeIn(legend), run_time=0.7)
            self.play(Create(chance), FadeIn(chance_l), run_time=0.4)
            self.mark("chance", VGroup(chance, chance_l), "uniform-guessing loss ln(256) = 5.55")
            self.play(Create(tr_curve), Create(va_curve), run_time=3.0, rate_func=linear)
            self.mark("loss-curve", tr_curve, f"training loss: {LOSS[0]:.2f} at step 0 → {LOSS[-1]:.2f} at step 3000")
            self.mark("val-curve", va_curve, f"validation loss on held-out text: {VAL[0]:.2f} → {VAL[-1]:.2f}")
            self.mark("legend", legend, "yellow = training loss, dashed teal = validation loss")
            self.play(FadeIn(a150), run_time=0.3)
            self.mark("pt-150", a150, f"step 150: train loss {LOSS[i150]:.3f}")
            self.sync("grinds", 0.1)
            self.play(FadeIn(aend), run_time=0.4)
            self.mark("pt-3000", aend, f"step 3000: train loss {LOSS[-1]:.3f}, val {VAL[-1]:.3f}")

        # ------------------------------------------------------------ samples morph
        hdr5 = self.header("What it wrote")
        tag = lambda st: VGroup(self.T(f"step {st}", 36, YELLOW, BOLD),
                                self.M(f"loss {loss_at[st]:.2f}", 28, MUTED)).arrange(RIGHT, buff=0.4)
        blocks = {st: self.sample_block(st, width=44, size=30) for st in (0, 100, 300, 1000, 3000)}
        for st, b in blocks.items():
            b.move_to(DOWN * 0.2)
        tags = {st: tag(st).move_to(UP * 2.5) for st in blocks}
        pr = self.T('prompt: "mia has a red ball. tom" · gray = prompt, white = model', 24, MUTED).to_edge(DOWN, buff=0.6)
        with self.say("And here's what it wrote along the way, from the same prompt. Step zero, alphabet soup. "
                      "Step 100, real words, broken grammar. Step 300, actual sentences.",
                      scene="One panel morphing through real samples: step 0 (loss 5.56) gibberish → step 100 (loss 1.82) "
                            "real words, broken grammar → step 300 (loss 0.69) well-formed sentences.",
                      chapter="What it wrote"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax3, xl3, yl3, xt3, yt3, tr_curve, va_curve, chance, chance_l, legend, a150, aend)),
                      ReplacementTransform(hdr4, hdr5), run_time=0.5)
            self.play(FadeIn(tags[0]), FadeIn(blocks[0], shift=UP * 0.2), FadeIn(pr), run_time=0.6)
            cur_t, cur_b = tags[0], blocks[0]
            self.mark("sample", cur_b, f"real sample at step 0: {SNAP[0][:60]}…")
            self.mark("step", cur_t, f"step 0, training loss {loss_at[0]:.2f}")
            for st, ph in [(100, "Step 100"), (300, "Step 300")]:
                self.sync(ph, 0.1)
                self.play(ReplacementTransform(cur_t, tags[st]), FadeTransform(cur_b, blocks[st]), run_time=0.8)
                cur_t, cur_b = tags[st], blocks[st]
                self.mark("sample", cur_b, f"real sample at step {st}: {SNAP[st][:70]}…")
                self.mark("step", cur_t, f"step {st}, training loss {loss_at[st]:.2f}")

        b1000 = self.sample_block(1000, width=44, size=30, hl=("yellow ball", PINK)).move_to(DOWN * 0.2)
        b3000 = self.sample_block(3000, width=44, size=30, hl=("red ball", TEAL)).move_to(DOWN * 0.2)
        with self.say("At step 1000, it still handed Tom the yellow ball. Wrong. By step 3000, it's the red ball. "
                      "Correct, and nobody told it why.",
                      scene="Step 1000 sample ends 'now tom has the yellow ball' (highlighted pink — wrong, mia gave him the "
                            "red ball); step 3000 sample ends 'now tom has the red ball' (highlighted teal — correct)."):
            self.play(ReplacementTransform(cur_t, tags[1000]), FadeTransform(cur_b, b1000), run_time=0.8)
            cur_t, cur_b = tags[1000], b1000
            x1 = self.T("✗ wrong", 34, PINK, BOLD).move_to([4.3, 2.5, 0])
            self.play(FadeIn(x1, scale=1.3), run_time=0.3)
            self.mark("sample", cur_b, f"real sample at step 1000: …now tom has the yellow ball (wrong)")
            self.mark("step", cur_t, f"step 1000, training loss {loss_at[1000]:.2f}")
            self.sync("By step 3000", 0.1)
            x2 = self.T("✓ correct", 34, TEAL, BOLD).move_to(x1)
            self.play(ReplacementTransform(cur_t, tags[3000]), FadeTransform(cur_b, b3000), ReplacementTransform(x1, x2),
                      run_time=0.8)
            self.mark("sample", b3000, f"real sample at step 3000: …now tom has the red ball (correct)")
            self.mark("step", tags[3000], f"step 3000, training loss {loss_at[3000]:.2f}")
            self.mark("verdict", x2, "the step-3000 model tracks who has which ball")

        # ------------------------------------------------------------ outro
        steps = [("predict", BLUE), ("measure the miss", PINK), ("nudge the weights", TEAL), ("repeat", YELLOW)]
        pills = VGroup(*[self.chip(t, c, 28, font=SANS, pad=0.16, fill=0.12) for t, c in steps]).arrange(RIGHT, buff=0.5)
        pills.move_to(UP * 0.5)
        arr = VGroup(*[Arrow(pills[i].get_right(), pills[i + 1].get_left(), buff=0.08, color=MUTED, stroke_width=3)
                       for i in range(3)])
        back = CurvedArrow(pills[-1].get_bottom() + DOWN * 0.1, pills[0].get_bottom() + DOWN * 0.1, angle=-PI / 4,
                           color=MUTED, stroke_width=3)
        cta = self.T("next: train your own →", 50, YELLOW, BOLD).move_to(DOWN * 2.5)
        with self.say("Nobody wrote a single rule. Predict, measure the miss, nudge, repeat. Next up, train your own.",
                      scene="The training loop as pills: predict → measure the miss → nudge the weights → repeat, with an "
                            "arrow looping back. Final card: 'next: train your own'.", chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(*[m for m in self.mobjects])), run_time=0.5)
            self.sync("Predict", 0.1)
            for k in range(4):
                self.play(FadeIn(pills[k], shift=UP * 0.2), *([GrowArrow(arr[k - 1])] if k else []), run_time=0.35)
            self.play(Create(back), run_time=0.4)
            for (t, _), p in zip(steps, pills):
                self.mark("loop-" + t.split()[0], p, f"training loop step: {t}")
            self.sync("Next up", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "next: train your own model in the Training lab")
        self.wait(0.6)
