import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from model import VOCAB, tok, forward

PROMPT = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the"
ZOE = "zoe has a yellow shoe. q: what color is the shoe? a: the shoe is"
COL = {" red": PINK, " white": "#e8e8e8", " purple": "#b48cff", " black": "#7d8597", " ava": TEAL, ",": ORANGE,
       " yellow": YELLOW, " pink": PINK, " green": TEAL, " blue": BLUE}
X_LAB, X_BAR, BAR_MAX = -3.7, -3.35, 7.2


def col(t):
    return COL.get(t, MUTED)


def softmax(x, T=1.0):
    z = x / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class NextToken(VScene):
    VIDEO_ID = "next-token"

    def construct(self):
        logits = forward(tok(PROMPT))[-1].numpy().astype(np.float64)
        order = np.argsort(-logits)
        top = list(order[:6])
        top_t = [VOCAB[i] for i in top]
        ys = [2.0 - n * 0.68 for n in range(7)]

        # ------------------------------------------------------------ hook: the wall of 256 logits
        wall = VGroup()
        bw = 12.6 / 256
        for i, v in enumerate(logits):
            h = abs(v) * 0.22
            r = Rectangle(width=bw * 0.8, height=max(h, 0.01), stroke_width=0,
                          fill_color=PINK if VOCAB[i] == " red" else (ORANGE if v > 0 else BLUE), fill_opacity=0.9)
            r.move_to([-6.3 + (i + 0.5) * bw, (h / 2 if v > 0 else -h / 2) - 0.6, 0])
            wall.add(r)
        zero = Line([-6.4, -0.6, 0], [6.4, -0.6, 0], color=MUTED, stroke_width=1.5)
        cap = self.T("256 numbers: one score per token in the vocab", 30, FG).to_edge(UP, buff=0.9)
        redlab = self.M("␣red", 28, PINK, t2c={"␣": MUTED}).next_to(wall[VOCAB.index(" red")], UP, buff=0.12)
        with self.say("The model never outputs a word. It outputs 256 numbers. Here's how those turn into one word.",
                      scene="A wall of 256 thin bars — the model's real raw scores for every token in the vocab after the "
                            "prompt; the tallest (pink) is ' red'.", chapter="256 numbers"):
            self.play(Create(zero), run_time=0.3)
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN if logits[i] > 0 else UP) for i, r in enumerate(wall)],
                                  lag_ratio=0.006), FadeIn(cap), run_time=1.8)
            self.play(FadeIn(redlab, shift=DOWN * 0.2), run_time=0.4)
            self.mark("wall", wall, f"all 256 logits for the prompt (min {logits.min():.2f}, max {logits.max():.2f})")
            self.mark("wall-red", VGroup(wall[VOCAB.index(" red")], redlab), f'logit for " red" = {logits[VOCAB.index(" red")]:.2f} (the highest)')

        # ------------------------------------------------------------ logits chart
        hdr = self.header("Logits")
        rows = VGroup()
        for n, (i, t) in enumerate(zip(top, top_t)):
            lab = self.M(self.show_tok(t), 30, col(t), t2c={"␣": MUTED}).move_to([X_LAB, ys[n], 0], aligned_edge=RIGHT)
            bar = Rectangle(width=BAR_MAX * logits[i] / logits[top[0]], height=0.46, fill_color=col(t), fill_opacity=0.85,
                            stroke_width=0).move_to([X_BAR, ys[n], 0], aligned_edge=LEFT)
            val = self.M(f"{logits[i]:.2f}", 28, col(t)).next_to(bar, RIGHT, buff=0.22)
            rows.add(VGroup(lab, bar, val))
        rest_lab = self.M("250 others", 28, MUTED).move_to([X_LAB, ys[6], 0], aligned_edge=RIGHT)
        rest_val = self.M(f"{logits[order[6]]:.2f} … {logits.min():.2f}", 28, MUTED).move_to([X_BAR, ys[6], 0],
                                                                                         aligned_edge=LEFT)
        prompt_t = self.M("…now tom has the  →  ?", 28, MUTED).next_to(rows, UP, buff=0.45).align_to(rows, LEFT)
        with self.say(f"These raw scores are called logits. For our prompt, red scores 10.4, white 7.7, purple 7.5. "
                      f"Two thirds of the vocab sits below zero.",
                      scene="Bar chart of the six highest logits: ' red' 10.37, ' white' 7.68, ' purple' 7.53, ' black' "
                            "4.62, ' ava' 3.98, ',' 3.72; the other 250 range from 3.61 down to −5.88.",
                      chapter="Logits"):
            self.unmark("wall", "wall-red")
            self.play(FadeOut(VGroup(wall, zero, cap, redlab)), FadeIn(hdr), run_time=0.5)
            self.play(FadeIn(prompt_t), LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                                      for r in rows], lag_ratio=0.15), run_time=1.6)
            self.sync("Two thirds", 0.1)
            self.play(FadeIn(rest_lab), FadeIn(rest_val), run_time=0.4)
            for (i, t), r in zip(zip(top, top_t), rows):
                self.mark(f"logit-{t.strip() or 'x'}", r, f'logit for "{t}" = {logits[i]:.2f}')
            self.mark("logit-rest", VGroup(rest_lab, rest_val), "the remaining 250 logits")

        # ------------------------------------------------------------ softmax (live, driven by T and top-k)
        Tt = ValueTracker(1.0)
        Kk = ValueTracker(0.0)  # 0 = no top-k, 1 = top-3 applied

        def probs():
            p = softmax(logits, Tt.get_value())
            k = Kk.get_value()
            if k > 0:
                pk = np.zeros_like(p)
                pk[top[:3]] = p[top[:3]] / p[top[:3]].sum()
                p = (1 - k) * p + k * pk
            return p

        def bar_for(n):
            def f():
                p = probs()
                v = p[top[n]] if n < 6 else 1 - p[top].sum()
                c = col(top_t[n]) if n < 6 else MUTED
                return Rectangle(width=max(0.02, BAR_MAX * v), height=0.46, fill_color=c, fill_opacity=0.85,
                                 stroke_width=0).move_to([X_BAR, ys[n], 0], aligned_edge=LEFT)
            return f

        def val_for(n, bar):
            def f():
                p = probs()
                v = p[top[n]] if n < 6 else 1 - p[top].sum()
                c = col(top_t[n]) if n < 6 else MUTED
                s = f"{v * 100:.2f}%" if (v < 0.1 or v > 0.999) else f"{v * 100:.1f}%"
                return self.T(s, 28, c).next_to(bar, RIGHT, buff=0.22)
            return f

        pbars = [always_redraw(bar_for(n)) for n in range(7)]
        pvals = [always_redraw(val_for(n, pbars[n])) for n in range(7)]
        hdr2 = self.header("Softmax")
        formula = VGroup(self.M("p = exp(logit / T) / sum of exp(all 256)", 28, FG)).to_edge(DOWN, buff=0.75)
        with self.say("Softmax turns them into probabilities. Raise e to each logit, then divide by the total. "
                      "Exponentials blow small gaps up into big ones, so red ends up with 87 percent.",
                      scene="The logit bars morph into probability bars: ' red' 87.1%, ' white' 5.9%, ' purple' 5.1%, "
                            "' black' 0.28%, ' ava' 0.15%, ',' 0.11%, all 250 others together ~1.4%. Formula: "
                            "p = exp(logit) / sum of exp over all 256.", chapter="Softmax"):
            self.unmark(*list(self._marks))
            self.play(ReplacementTransform(hdr, hdr2), FadeIn(formula), run_time=0.5)
            self.sync("Raise e", 0.1)
            self.play(*[ReplacementTransform(rows[n][1], pbars[n]) for n in range(6)],
                      *[ReplacementTransform(rows[n][2], pvals[n]) for n in range(6)],
                      ReplacementTransform(rest_val, pvals[6]), FadeIn(pbars[6]),
                      run_time=1.4)
            self.sync("87 percent", 0.2)
            self.play(Indicate(rows[0][0], color=PINK, scale_factor=1.25), run_time=0.6)
            self.add(*pbars, *pvals)
            self._mark_bars(rows, rest_lab, pbars, pvals, probs, top, top_t, "T = 1")

        # ------------------------------------------------------------ temperature
        hdr3 = self.header("Temperature")
        dlab = self.T("T =", 44, YELLOW, BOLD).move_to([4.0, 2.9, 0])
        dnum = self.live(lambda: f"{Tt.get_value():.1f}", lambda t: t.next_to(dlab, RIGHT, buff=0.25), size=52,
                         color=YELLOW, font=SANS, weight=BOLD)
        dial = VGroup(dlab, dnum)
        with self.say("Temperature divides every logit before the softmax. At temperature one, nothing changes.",
                      scene="A big 'T = 1.0' dial appears top right; the bars stay at their T = 1 values.",
                      chapter="Temperature"):
            self.play(ReplacementTransform(hdr2, hdr3), FadeIn(dial, scale=1.2), run_time=0.6)
            self.mark("dial", dial, "temperature T = 1.0 (logits used as-is)")
            self._mark_bars(rows, rest_lab, pbars, pvals, probs, top, top_t, "T = 1")

        p03 = softmax(logits, 0.3)
        with self.say(f"Turn it down to 0.3, and the gaps get stretched. Red jumps to {p03[top[0]] * 100:.2f} percent. "
                      f"Basically deterministic.",
                      scene=f"T slides to 0.3: ' red' grows to {p03[top[0]] * 100:.2f}%, every other bar collapses to ~0."):
            self.play(Tt.animate.set_value(0.3), run_time=1.8)
            self.mark("dial", dial, "temperature T = 0.3 (sharper)")
            self._mark_bars(rows, rest_lab, pbars, pvals, probs, top, top_t, "T = 0.3")

        p2 = softmax(logits, 2.0)
        tail2 = 1 - p2[top].sum()
        with self.say(f"Crank it to 2, and everything flattens. Red drops to {p2[top[0]] * 100:.0f} percent, and "
                      f"almost half the probability leaks into the long tail of junk.",
                      scene=f"T slides to 2.0: ' red' falls to {p2[top[0]] * 100:.1f}%, ' white' {p2[top[1]] * 100:.1f}%, "
                            f"' purple' {p2[top[2]] * 100:.1f}%, and the 250 others together reach {tail2 * 100:.1f}%."):
            self.play(Tt.animate.set_value(2.0), run_time=2.0)
            self.mark("dial", dial, "temperature T = 2.0 (flatter)")
            self._mark_bars(rows, rest_lab, pbars, pvals, probs, top, top_t, "T = 2")
            self.sync("long tail", 0.1)
            self.play(Indicate(rest_lab, color=YELLOW, scale_factor=1.2), run_time=0.7)

        # ------------------------------------------------------------ top-k
        hdr4 = self.header("Top-k")
        kl = self.T("top-k = 3", 36, TEAL, BOLD).next_to(dial, DOWN, buff=0.35).align_to(dial, RIGHT)
        pk = p2[top[:3]] / p2[top[:3]].sum()
        with self.say(f"Top k is the safety net. Keep only the k best tokens, zero out the rest, and renormalize. "
                      f"With k equals 3, the junk is gone, and red is back to {pk[0] * 100:.0f} percent.",
                      scene=f"At T = 2 with top-k = 3: only ' red', ' white', ' purple' survive and renormalize to "
                            f"{pk[0] * 100:.1f}%, {pk[1] * 100:.1f}%, {pk[2] * 100:.1f}%; everything else drops to 0.",
                      chapter="Top-k"):
            self.play(ReplacementTransform(hdr3, hdr4), FadeIn(kl, shift=LEFT * 0.2), run_time=0.5)
            self.sync("zero out", 0.1)
            self.play(Kk.animate.set_value(1.0), *[rows[n][0].animate.set_opacity(0.3) for n in range(3, 6)],
                      rest_lab.animate.set_opacity(0.3), run_time=1.6)
            self.mark("topk", kl, "top-k = 3: only the 3 most likely tokens can be sampled")
            self._mark_bars(rows, rest_lab, pbars, pvals, probs, top, top_t, "T = 2, top-k = 3")

        # ------------------------------------------------------------ dice roll
        hdr5 = self.header("The dice roll")
        p1 = softmax(logits, 1.0)
        cum = np.cumsum(p1[order])
        strip = VGroup()
        sx0, sw = -6.0, 12.0
        x = sx0
        for j in range(len(order)):
            w_ = p1[order[j]] * sw
            if j < 3:
                r = Rectangle(width=w_, height=0.75, fill_color=col(VOCAB[order[j]]), fill_opacity=0.85,
                              stroke_color=BG, stroke_width=2)
                r.move_to([x, 1.2, 0], aligned_edge=LEFT)
                strip.add(r)
            x += w_
        tail = Rectangle(width=sx0 + sw - strip[-1].get_right()[0], height=0.75, fill_color=MUTED, fill_opacity=0.6,
                         stroke_color=BG, stroke_width=2).next_to(strip[-1], RIGHT, buff=0)
        strip.add(tail)
        slabs = VGroup(self.M("␣red 87%", 28, BG, t2c={"␣": "#5a2a3a"}).move_to(strip[0]),
                       self.M("white", 22, FG).next_to(strip[1], UP, buff=0.12),
                       self.M("purple", 22, "#b48cff").next_to(strip[2], DOWN, buff=0.12))
        for seed in range(200):  # first seed whose 20 real draws show the typical mix (mostly red + white + purple)
            us = np.random.default_rng(seed).random(20)
            picks = [int(order[np.searchsorted(cum, u * cum[-1])]) for u in us]
            got = {VOCAB[i] for i in picks}
            if {" red", " white", " purple"} <= got and 16 <= sum(VOCAB[i] == " red" for i in picks) <= 18:
                break
        others_txt = " and ".join(sorted({VOCAB[i].strip() for i in picks} - {"red"}))
        rolls = VGroup(*[self.chip(VOCAB[i], col(VOCAB[i]), 26, fill=0.2) for i in picks])
        rolls.arrange_in_grid(rows=2, buff=(0.14, 0.3)).move_to(DOWN * 1.6)
        n_red = sum(VOCAB[i] == " red" for i in picks)
        with self.say(f"Then we roll a weighted die. Lay the probabilities end to end, throw a dart, and take whatever "
                      f"it lands on. Here are 20 rolls at temperature one. {n_red} reds, plus the odd {others_txt}.",
                      scene=f"The T = 1 probabilities laid end to end as a strip (' red' 87% takes most of it, then ' white', "
                            f"' purple', and a sliver for everything else). 20 random darts land on it; the picked tokens "
                            f"appear below: {n_red}× ' red', plus " + ", ".join(f"{sum(VOCAB[i] == t for i in picks)}× '{t}'" for t in sorted({VOCAB[i] for i in picks} - {" red"})) + ".", chapter="The dice roll"):
            self.unmark(*list(self._marks))
            Tt.set_value(1.0); Kk.set_value(0.0)
            self.play(FadeOut(VGroup(rows, rest_lab, *pbars, *pvals, formula, dial, kl, prompt_t)),
                      ReplacementTransform(hdr4, hdr5), run_time=0.5)
            for m in pbars + pvals:
                m.clear_updaters()
            dnum.clear_updaters()
            self.play(LaggedStart(*[GrowFromEdge(r, LEFT) for r in strip], lag_ratio=0.3), FadeIn(slabs), run_time=1.0)
            self.mark("strip", strip, "probabilities laid end to end (T = 1): a dart's landing spot picks the token")
            self.sync("Here are 20", 0.1)
            for k, (u, i) in enumerate(zip(us, picks)):
                xx = sx0 + u * sw
                dart = Triangle(color=YELLOW, fill_opacity=1, stroke_width=0).scale(0.12).rotate(PI)
                dart.move_to([xx, 1.75, 0])
                self.add(dart)
                self.play(FadeIn(rolls[k], shift=UP * 0.2), dart.animate.shift(DOWN * 0.08), run_time=0.14)
                self.remove(dart)
                self.add(Line([xx, 0.83, 0], [xx, 1.57, 0], color=YELLOW, stroke_width=2, stroke_opacity=0.7))
            for k, i in enumerate(picks):
                self.mark(f"roll-{k}", rolls[k], f'roll {k + 1}: "{VOCAB[i]}" (p = {p1[i] * 100:.1f}%)')

        # ------------------------------------------------------------ when it's unsure
        hdr6 = self.header("When it's unsure")
        zl = forward(tok(ZOE))[-1].numpy().astype(np.float64)
        zp = softmax(zl)
        zo = list(np.argsort(-zp)[:4])
        ztxt = VGroup(self.M("zoe has a yellow shoe.", 30, FG, t2c={"yellow": YELLOW}),
                      self.M("q: what color is the shoe?", 30, FG),
                      self.M("a: the shoe is  →  ?", 30, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        ztxt.move_to([-3.2, 1.7, 0])
        zrows = VGroup()
        for n, i in enumerate(zo):
            t = VOCAB[i]
            lab = self.M(self.show_tok(t), 30, col(t), t2c={"␣": MUTED}).move_to([-2.0, -0.2 - n * 0.68, 0],
                                                                                  aligned_edge=RIGHT)
            bar = Rectangle(width=max(0.03, 7.0 * zp[i]), height=0.46, fill_color=col(t), fill_opacity=0.85,
                            stroke_width=0).move_to([-1.65, lab.get_y(), 0], aligned_edge=LEFT)
            val = self.T(f"{zp[i] * 100:.1f}%", 28, col(t)).next_to(bar, RIGHT, buff=0.22)
            zrows.add(VGroup(lab, bar, val))
        wrong = 1 - zp[VOCAB.index(" yellow")]
        with self.say(f"Sampling matters most when the model is torn. Zoe has a yellow shoe. What color is the shoe? "
                      f"Yellow, {zp[zo[0]] * 100:.0f} percent. Pink, {zp[zo[1]] * 100:.0f}. A coin flip. "
                      f"Sample at temperature one, and it names the wrong color more often than not.",
                      scene=f"A real mistake: for 'zoe has a yellow shoe. q: what color is the shoe? a: the shoe is' the "
                            f"model gives ' yellow' {zp[zo[0]] * 100:.1f}%, ' pink' {zp[zo[1]] * 100:.1f}%, "
                            f"' black' {zp[zo[2]] * 100:.1f}%, ' purple' {zp[zo[3]] * 100:.1f}%. Flash: 'coin flip'.",
                      chapter="When it's unsure"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(*[m for m in self.mobjects if m is not hdr5])), ReplacementTransform(hdr5, hdr6),
                      run_time=0.5)
            self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in ztxt], lag_ratio=0.3), run_time=1.0)
            self.mark("zoe-prompt", ztxt, "prompt: zoe has a yellow shoe. q: what color is the shoe? a: the shoe is")
            self.sync("Yellow,", 0.1)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2])) for r in zrows],
                                  lag_ratio=0.15), run_time=1.0)
            for i, r in zip(zo, zrows):
                self.mark(f"zoe-{VOCAB[i].strip()}", r, f'P("{VOCAB[i]}") = {zp[i] * 100:.1f}%')
            self.sync("A coin flip", 0.1)
            self.flash("coin flip", PINK, 64, hold=0.7, pos=[3.6, 1.7, 0])
            self.sync("more often", 0.4)
            wr = self.T(f"wrong {wrong * 100:.0f}% of the time", 32, PINK, BOLD).move_to([3.6, 1.7, 0])
            self.play(FadeIn(wr, scale=1.2), run_time=0.4)
            self.mark("wrong", wr, f"at T = 1 the sampled color is wrong with probability {wrong * 100:.1f}%")

        # ------------------------------------------------------------ outro
        lo = VGroup(self.T("low T", 40, BLUE, BOLD), self.T("safe · repetitive", 30, FG)).arrange(DOWN, buff=0.2)
        hi = VGroup(self.T("high T", 40, ORANGE, BOLD), self.T("creative · chaotic", 30, FG)).arrange(DOWN, buff=0.2)
        lo.move_to(LEFT * 3.6 + UP * 0.6)
        hi.move_to(RIGHT * 3.6 + UP * 0.6)
        track = Line(LEFT * 3.6, RIGHT * 3.6, color=MUTED, stroke_width=4).move_to(DOWN * 0.7)
        knob = Dot(track.get_center(), radius=0.16, color=YELLOW)
        cta = self.T("go turn the dial yourself in the X-ray →", 40, YELLOW, BOLD).move_to(DOWN * 2.4)
        with self.say("Low temperature, safe and repetitive. High temperature, creative and chaotic. "
                      "Go turn the dial yourself in the X-ray.",
                      scene="A temperature slider: 'low T — safe · repetitive' on the left, 'high T — creative · chaotic' "
                            "on the right; the knob slides both ways. Final card: 'go turn the dial yourself in the X-ray'.",
                      chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(*[m for m in self.mobjects])), run_time=0.5)
            self.play(Create(track), FadeIn(knob), FadeIn(lo), run_time=0.5)
            self.play(knob.animate.move_to(track.get_left()), run_time=0.6)
            self.sync("High temperature", 0.1)
            self.play(FadeIn(hi), knob.animate.move_to(track.get_right()), run_time=0.8)
            self.mark("low", lo, "low temperature: sharper distribution, more predictable text")
            self.mark("high", hi, "high temperature: flatter distribution, more surprising text")
            self.sync("Go turn", 0.1)
            self.play(knob.animate.move_to(track.get_center()), FadeIn(cta, scale=1.2), run_time=0.5)
            self.mark("cta", cta, "open the X-ray to try temperature and top-k live")
        self.wait(0.5)

    def _mark_bars(self, rows, rest_lab, pbars, pvals, probs, top, top_t, cond):
        p = probs()
        for n in range(6):
            self.mark(f"bar-{top_t[n].strip() or 'x'}", VGroup(rows[n][0], pbars[n], pvals[n]),
                      f'P("{top_t[n]}") = {p[top[n]] * 100:.2f}% at {cond}')
        self.mark("bar-others", VGroup(rest_lab, pbars[6], pvals[6]),
                  f"all 250 other tokens together = {(1 - p[top].sum()) * 100:.2f}% at {cond}")
