import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vscene import *  # noqa
from lib_research import load, Plot, legend, pill, panel, illus

VIDEO = dict(id="residuals-layernorm", cls="ResidualsLayerNorm", title="Residuals & LayerNorm: why deep nets train",
             subtitle="Vanishing signals, the residual highway, and keeping scales sane, with real ablations",
             relates="lab", track="design", order=10, poster=("The residual highway", 0, 0.95))

AB = {r["id"]: r for r in load("ablations")["runs"]}
AB_STEPS = load("ablations")["steps"]
DP = load("depth")
INIT = {r["id"]: r for r in DP["init"]}
DRUN = {r["id"]: r for r in DP["runs"]}
BASE, NORES, NOLN = AB["baseline"], AB["no-residual"], AB["no-layernorm"]


def sci(v):
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    return f"{v / 10 ** e:.1f}e{e}"


class ResidualsLayerNorm(VScene):
    VIDEO_ID = "residuals-layernorm"

    def code_line(self, size=36):
        """'x = x + attn(LN(x))' as one Text; returns (text, the 'x +' glyphs)."""
        t = Text("x = x + attn(LN(x))", font=MONO, font_size=size, color=FG,
                 t2c={"x + ": YELLOW, "attn": BLUE, "LN": TEAL})
        # glyphs (spaces are not glyphs): x = x + a t t n ( L N ( x ) )
        return t, VGroup(t[2], t[3])

    def construct(self):
        # ------------------------------------------------------------ hook
        c1, xplus = self.code_line()
        c1.move_to(UP * 2.6)
        pl = Plot([0, AB_STEPS], [0, 6], width=9.5, height=3.9, xticks=[0, AB_STEPS // 3, 2 * AB_STEPS // 3, AB_STEPS],
                  yticks=[0, 2, 4, 6],
                  xlabel="training step", ylabel="validation loss").move_to(DOWN * 0.75 + LEFT * 0.4)
        cb = pl.line(BASE["curve"]["step"], BASE["curve"]["val"], TEAL, 4)
        cn = pl.line(NORES["curve"]["step"], NORES["curve"]["val"], PINK, 4)
        lb = self.M(f"with  x +  → {BASE['final_val']:.2f}", 24, TEAL).next_to(pl.p(AB_STEPS, BASE["final_val"]), UP, buff=0.18) \
            .align_to(pl.p(AB_STEPS, 0), RIGHT)
        ln_ = self.M(f"without → {NORES['final_val']:.2f}", 24, PINK).next_to(pl.p(AB_STEPS, NORES["final_val"]), UP, buff=0.18) \
            .align_to(pl.p(AB_STEPS, 0), RIGHT)
        strike = Line(xplus.get_left() + LEFT * 0.05, xplus.get_right() + RIGHT * 0.05, color=PINK, stroke_width=5)
        with self.say(f"Delete two characters from TinyGPT: the x plus in each block. Same data, same {AB_STEPS:,} steps. "
                      f"The normal model reaches {BASE['final_val']:.2f}. The edited one gets stuck at "
                      f"{NORES['final_val']:.2f}.",
                      scene=f"Code 'x = x + attn(LN(x))' with 'x +' struck out. Real validation curves, 3-layer TinyGPT "
                            f"trained {AB_STEPS} steps: with residuals (teal) → {BASE['final_val']:.3f}; without (pink) stuck "
                            f"at {NORES['final_val']:.3f}.", chapter="Two characters"):
            self.play(FadeIn(c1, shift=DOWN * 0.2), run_time=0.5)
            self.mark("code", c1, "a transformer block line: x = x + attn(LN(x)) (residual + LayerNorm)")
            self.play(Create(strike), xplus.animate.set_opacity(0.35), run_time=0.5)
            self.play(FadeIn(pl), run_time=0.5)
            self.sync("The normal", 0.1)
            self.play(Create(cb), run_time=1.0, rate_func=linear)
            self.play(FadeIn(lb), run_time=0.3)
            self.mark("with-res", VGroup(cb, lb), f"real: with residuals, final val loss {BASE['final_val']:.3f}")
            self.sync("The edited", 0.1)
            self.play(Create(cn), run_time=1.0, rate_func=linear)
            self.play(FadeIn(ln_), run_time=0.3)
            self.mark("no-res", VGroup(cn, ln_), f"real: no residuals, final val loss {NORES['final_val']:.3f} (stuck)")

        samp = VGroup()
        for s in NORES["samples"]:
            p = s["prompt"] if len(s["prompt"]) < 30 else "…" + s["prompt"][-22:]
            t = s["text"].replace("\n", " ↵ ")[:40]
            samp.add(VGroup(self.M(p, 24, MUTED), self.M(t, 24, PINK)).arrange(RIGHT, buff=0.3))
        samp.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(DOWN * 0.3)
        same = all(s["text"] == NORES["samples"][0]["text"] for s in NORES["samples"])
        st = self.T("three different prompts → " + ("the exact same output" if same else "near-identical output"), 30, FG,
                    BOLD).next_to(samp, UP, buff=0.6)
        with self.say("Worse, it ignores you. Three different prompts, " + ("the exact same words back." if same else
                                                                            "almost the same words back."),
                      scene="Real samples from the no-residual model for three different prompts: "
                            + " | ".join(repr(s["text"][:40]) for s in NORES["samples"]) + "."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pl, cb, cn, lb, ln_)), run_time=0.4)
            self.play(FadeIn(st), LaggedStart(*[FadeIn(s, shift=RIGHT * 0.2) for s in samp], lag_ratio=0.25), run_time=1.0)
            for k, (s, g) in enumerate(zip(NORES["samples"], samp)):
                self.mark(f"samp-{k}", g, f"no-residual model, prompt {s['prompt'][:30]!r} → {s['text'][:40]!r}")

        # ------------------------------------------------------------ vanishing (toy then real)
        hdr = self.header("Deep = multiply many times")
        chain = VGroup()
        for k in range(8):
            chain.add(pill(f"×0.6", BLUE, 26))
        chain.arrange(RIGHT, buff=0.18).move_to(UP * 1.6)
        res_t = VGroup(self.M("0.6 ^ 24 =", 34, FG), self.M(f"{0.6 ** 24:.1e}", 34, PINK)).arrange(RIGHT, buff=0.25) \
            .next_to(chain, DOWN, buff=0.6)
        res_t2 = VGroup(self.M("1.5 ^ 24 =", 34, FG), self.M(f"{1.5 ** 24:,.0f}", 34, ORANGE)).arrange(RIGHT, buff=0.25) \
            .next_to(res_t, DOWN, buff=0.3)
        tag = illus("toy arithmetic · illustration").next_to(res_t2, DOWN, buff=0.35)
        with self.say("Why? Without the plus, a signal passes through every layer in a chain, and each one rescales it. "
                      "Shrink a little twenty-four times and it vanishes. Grow a little and it explodes.",
                      scene=f"Toy arithmetic (illustration): a chain of ×0.6 blocks; 0.6^24 = {0.6 ** 24:.1e} (vanishes), "
                            f"1.5^24 = {1.5 ** 24:,.0f} (explodes).", chapter="Vanishing signals"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(samp, st, c1, strike)), FadeIn(hdr), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in chain], lag_ratio=0.1), run_time=1.0)
            self.sync("Shrink", 0.1)
            self.play(FadeIn(res_t, shift=UP * 0.1), FadeIn(tag), run_time=0.4)
            self.mark("vanish", res_t, f"toy: 0.6^24 = {0.6 ** 24:.2e}")
            self.sync("Grow", 0.1)
            self.play(FadeIn(res_t2, shift=UP * 0.1), run_time=0.4)
            self.mark("explode", res_t2, f"toy: 1.5^24 = {1.5 ** 24:,.0f}")

        L = DP["init_depth"]
        gb, gn = INIT["baseline"]["grad_rms"], INIT["no-residual"]["grad_rms"]
        allv = [v for v in gb + gn if v > 0]
        ylo = 10 ** math.floor(math.log10(min(allv))); yhi = 10 ** math.ceil(math.log10(max(allv)))
        yt = [10 ** e for e in range(int(math.log10(ylo)), int(math.log10(yhi)) + 1, 4)]
        pg = Plot([0, L], [ylo, yhi], width=9.5, height=4.3, ylog=True, xticks=list(range(0, L + 1, 4)), yticks=yt,
                  xlabel="layer (0 = input embeddings)", ylabel="gradient size reaching each layer (log scale)",
                  yfmt=lambda v: f"1e{int(round(math.log10(v)))}").move_to(DOWN * 0.45 + LEFT * 0.6)
        lgb = pg.line(list(range(L + 1)), gb, TEAL, 4)
        lgn = pg.line(list(range(L + 1)), [max(v, ylo) for v in gn], PINK, 4)
        ratio = gn[-1] / max(gn[0], 1e-300)
        lab_b = self.T("with residuals: nearly flat", 26, TEAL, BOLD).next_to(pg.p(1, gb[1]), DOWN, buff=0.18, aligned_edge=LEFT)
        lab_n = self.T("without residuals", 26, PINK, BOLD).move_to(pg.p(L * 0.55, ylo * 1e3), aligned_edge=LEFT)
        drop = self.M(f"{sci(gn[-1])} → {sci(gn[0])}", 26, PINK).next_to(lab_n, DOWN, buff=0.15, aligned_edge=LEFT)
        with self.say(f"Here's the real thing: a freshly built {L}-layer TinyGPT, measuring how much gradient reaches each "
                      f"layer. Without residuals, by the first layer it has shrunk by a factor of 10 to the "
                      f"{round(math.log10(ratio))}. Those layers can't learn. With residuals, it's nearly flat.",
                      scene=f"Real measurement, {L}-layer TinyGPT at initialisation: RMS of the gradient reaching each layer "
                            f"(log scale). Without residuals (pink) it falls from {gn[-1]:.1e} at the top to {gn[0]:.1e} at "
                            f"the input (×10^-{round(math.log10(ratio))}); with residuals (teal) it stays ≈ {gb[0]:.1e}–{gb[-1]:.1e}."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(chain, res_t, res_t2, tag)), run_time=0.4)
            self.play(FadeIn(pg), run_time=0.5)
            self.sync("Without residuals", 0.1)
            self.play(Create(lgn), FadeIn(lab_n), run_time=1.2, rate_func=linear)
            self.play(FadeIn(drop), run_time=0.3)
            self.mark("grad-nores", VGroup(lgn, lab_n, drop), f"real: gradient RMS without residuals, {gn[-1]:.2e} at the "
                      f"top layer → {gn[0]:.2e} at the embeddings")
            self.sync("With residuals", 0.1)
            self.play(Create(lgb), FadeIn(lab_b), run_time=1.0, rate_func=linear)
            self.mark("grad-res", VGroup(lgb, lab_b), f"real: gradient RMS with residuals, {gb[-1]:.2e} → {gb[0]:.2e}: "
                      "nearly flat")

        # ------------------------------------------------------------ the highway
        hdr2 = self.header("The residual highway")
        y = -0.2
        road = Arrow(LEFT * 6.3 + UP * y, RIGHT * 6.3 + UP * y, buff=0, color=YELLOW, stroke_width=10,
                     max_tip_length_to_length_ratio=0.03)
        rl = self.T("residual stream", 26, YELLOW).next_to(road, DOWN, buff=0.25).align_to(road, LEFT)
        blocks = VGroup(); plus = VGroup(); taps = VGroup()
        for k, (name, col) in enumerate([("attn", BLUE), ("MLP", ORANGE), ("attn", BLUE), ("MLP", ORANGE)]):
            x = -3.9 + k * 2.6
            b = pill(name, col, 28).move_to([x + 0.5, y + 1.7, 0])
            pl_ = Circle(0.24, color=YELLOW, stroke_width=4, fill_color=BG, fill_opacity=1).move_to([x + 1.3, y, 0])
            pt = self.T("+", 32, YELLOW, BOLD).move_to(pl_)
            up = Line([x - 0.3, y, 0], [x - 0.3, y + 1.7, 0], color=MUTED, stroke_width=3)
            a1 = Arrow([x - 0.3, y + 1.7, 0], b.get_left(), buff=0.05, color=MUTED, stroke_width=3)
            a2 = Arrow(b.get_right(), [x + 1.3, y + 1.7, 0], buff=0.05, color=MUTED, stroke_width=3)
            dn = Arrow([x + 1.3, y + 1.7, 0], pl_.get_top(), buff=0.0, color=MUTED, stroke_width=3)
            blocks.add(b); plus.add(VGroup(pl_, pt)); taps.add(VGroup(up, a1, a2, dn))
        grad = Arrow(RIGHT * 6.3 + UP * (y - 1.3), LEFT * 6.3 + UP * (y - 1.3), buff=0, color=PINK, stroke_width=6,
                     max_tip_length_to_length_ratio=0.03)
        gl = self.T("gradient flows straight back through every +", 26, PINK).next_to(grad, DOWN, buff=0.2)
        with self.say("The plus sign builds a highway. The signal travels straight along it, and each block just reads "
                      "from it and adds a small edit back. On the way back, the gradient rides the same highway, "
                      "untouched by the plus signs.",
                      scene="Diagram: a yellow 'residual stream' arrow left to right; attention and MLP blocks branch off "
                            "it and add their output back through '+' nodes; a pink arrow below shows the gradient flowing "
                            "straight back through every +.", chapter="The residual highway"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pg, lgb, lgn, lab_b, lab_n, drop)), ReplacementTransform(hdr, hdr2), run_time=0.4)
            self.play(GrowArrow(road), FadeIn(rl), run_time=0.7)
            self.mark("stream", VGroup(road, rl), "the residual stream: x passes straight through, blocks add to it")
            self.sync("each block", 0.1)
            for k in range(4):
                self.play(Create(taps[k]), FadeIn(blocks[k]), FadeIn(plus[k], scale=1.4), run_time=0.4)
            for k in range(4):
                self.mark(f"blk-{k}", blocks[k], f"block reads the stream and adds its output back (x + f(x))")
            self.sync("On the way back", 0.1)
            self.play(GrowArrow(grad), FadeIn(gl), run_time=0.9)
            self.mark("grad-path", VGroup(grad, gl), "d(x + f(x))/dx = 1 + f'(x): the '1' carries gradient straight back")

        # ------------------------------------------------------------ layernorm
        hdr3 = self.header("LayerNorm")
        import random
        rng = random.Random(4)
        vals = [rng.gauss(0.6, 2.2) for _ in range(12)]
        mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        nv = [(v - mu) / sd for v in vals]
        def bars(vs, x0, col, scale=0.42):
            g = VGroup()
            for k, v in enumerate(vs):
                h = max(0.02, abs(v) * scale)
                r = Rectangle(width=0.22, height=h, stroke_width=0, fill_color=col, fill_opacity=0.85)
                r.move_to([x0 + k * 0.3, 0.3 + (h / 2 if v > 0 else -h / 2), 0])
                g.add(r)
            return g
        before = bars(vals, -6.0, ORANGE)
        after = bars(nv, 2.3, TEAL)
        base1 = Line([-6.2, 0.3, 0], [-6.2 + 12 * 0.3 + 0.1, 0.3, 0], color=MUTED, stroke_width=2)
        base2 = Line([2.1, 0.3, 0], [2.1 + 12 * 0.3 + 0.1, 0.3, 0], color=MUTED, stroke_width=2)
        arr = Arrow([-1.8, 0.3, 0], [1.6, 0.3, 0], color=MUTED, stroke_width=4)
        op = self.M("(x − mean) / std", 26, TEAL).next_to(arr, UP, buff=0.15)
        op2 = self.M("× gain + bias", 24, MUTED).next_to(arr, DOWN, buff=0.15)
        lb1 = self.T(f"a vector: mean {mu:.1f}, std {sd:.1f}", 24, ORANGE).move_to([-4.3, -2.6, 0])
        lb2 = self.T("mean 0, std 1", 24, TEAL).move_to([3.95, -2.6, 0])
        tag2 = illus("illustrative numbers").to_corner(DR, buff=0.4)
        with self.say("Residuals keep the signal alive, but adding edits layer after layer lets its size drift. "
                      "LayerNorm resets it: subtract the mean, divide by the spread, then let a learned gain and bias "
                      "pick the scale.",
                      scene=f"Illustration: a 12-number vector (orange, mean {mu:.1f}, std {sd:.1f}) passes through "
                            "LayerNorm, (x − mean)/std × gain + bias, and comes out with mean 0 and std 1 (teal).",
                      chapter="LayerNorm"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(road, rl, blocks, plus, taps, grad, gl)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(base1), LaggedStart(*[GrowFromEdge(b, DOWN) for b in before], lag_ratio=0.05), FadeIn(lb1),
                      FadeIn(tag2), run_time=0.9)
            self.mark("before", VGroup(before, lb1), f"illustrative vector before LayerNorm: mean {mu:.2f}, std {sd:.2f}")
            self.sync("LayerNorm resets", 0.1)
            self.play(GrowArrow(arr), FadeIn(op), FadeIn(op2), run_time=0.5)
            self.play(FadeIn(base2), TransformFromCopy(before, after), FadeIn(lb2), run_time=1.0)
            self.mark("ln-op", VGroup(op, op2), "LayerNorm: (x − mean)/std, then × learned gain + bias")
            self.mark("after", VGroup(after, lb2), "after LayerNorm: mean 0, std 1")

        Ld = DP["train_depth"]
        ab, an = DRUN["baseline"]["act_rms_trained"], DRUN["no-layernorm"]["act_rms_trained"]
        top = max(ab + an)
        pa = Plot([0, Ld], [0, top * 1.15], width=5.6, height=3.6, xticks=list(range(0, Ld + 1, 4)),
                  yticks=[0, round(top / 2, 1), round(top, 1)], xlabel="layer", ylabel="signal size (RMS)",
                  yfmt=lambda v: f"{v:g}").move_to(LEFT * 3.4 + DOWN * 0.5)
        la = pa.line(list(range(Ld + 1)), ab, TEAL, 4)
        lna = pa.line(list(range(Ld + 1)), an, ORANGE, 4)
        steps_d = DP["steps"]
        pc = Plot([0, steps_d], [0, 5], width=5.6, height=3.6, xticks=[0, steps_d // 2, steps_d], yticks=[0, 1, 2, 3, 4, 5],
                  xlabel="training step", ylabel="validation loss").move_to(RIGHT * 3.4 + DOWN * 0.5)
        runs = [("baseline", TEAL, "residual + LN"), ("no-layernorm", ORANGE, "no LN"), ("no-residual", PINK, "no residual")]
        lc = VGroup(*[pc.line(DRUN[i]["curve"]["step"], DRUN[i]["curve"]["val"], c, 3.5) for i, c, _ in runs])
        lgc = VGroup(*[self.M(f"{n}: {DRUN[i]['final_val']:.2f}" if DRUN[i]["final_val"] is not None else f"{n}: diverged",
                              22, c) for i, c, n in runs]).arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
        lgc.move_to(pc.p(steps_d, 3.3), aligned_edge=RIGHT)
        ttl = self.T(f"real {Ld}-layer runs, {steps_d:,} steps", 26, FG, BOLD).move_to(UP * 2.75)
        lga = legend([("with LN", TEAL, False), ("no LN", ORANGE, False)], size=22, direction=DOWN, buff=0.12)
        lga.move_to(pa.p(0.4, top * 1.05), aligned_edge=UL)
        with self.say(f"In real {Ld}-layer models after training, the signal without LayerNorm ends up "
                      f"{an[-1] / ab[-1]:.0f} times bigger at the top layer, and validation loss ends at "
                      f"{DRUN['no-layernorm']['final_val']:.2f} instead of "
                      f"{DRUN['baseline']['final_val']:.2f}. Remove the residuals instead, and it's "
                      f"{DRUN['no-residual']['final_val']:.2f}.",
                      scene=f"Left: real RMS of the hidden state at each layer of trained {Ld}-layer models: with LayerNorm "
                            f"{ab[0]:.2f} → {ab[-1]:.2f}; without {an[0]:.2f} → {an[-1]:.2f}. Right: real validation curves "
                            f"({steps_d} steps): residual+LN {DRUN['baseline']['final_val']:.3f}, no LN "
                            f"{DRUN['no-layernorm']['final_val']:.3f}, no residual {DRUN['no-residual']['final_val']:.3f}."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(before, after, base1, base2, arr, op, op2, lb1, lb2, tag2)), run_time=0.4)
            self.play(FadeIn(ttl), FadeIn(pa), FadeIn(lga), Create(la), run_time=0.8)
            self.play(Create(lna), run_time=0.9)
            self.mark("act-ln", la, f"real: signal RMS per layer with LayerNorm, {ab[0]:.2f} → {ab[-1]:.2f}")
            self.mark("act-noln", lna, f"real: signal RMS per layer without LayerNorm, {an[0]:.2f} → {an[-1]:.2f}")
            self.sync("validation loss", 0.1)
            self.play(FadeIn(pc), LaggedStart(*[Create(c) for c in lc], lag_ratio=0.2), FadeIn(lgc), run_time=1.6)
            for (i, c, n), cv in zip(runs, lc):
                self.mark(f"run-{i}", cv, f"real {Ld}-layer run, {n}: final val loss {DRUN[i]['final_val']}")

        # ------------------------------------------------------------ recap
        hdr4 = self.header("Recap")
        l1 = VGroup(self.M("x = x + attn(LN(x))", 40, FG))
        l2 = VGroup(self.M("x = x + mlp(LN(x))", 40, FG))
        code = VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.6)
        for l in (l1, l2):
            l[0][2:4].set_color(YELLOW)
            l[0][-6:-4].set_color(TEAL) if False else None
        n1 = self.T("+ : the highway keeps gradients alive", 30, YELLOW).move_to(DOWN * 1.3)
        n2 = self.T("LN : keeps every layer's input at a sane scale", 30, TEAL).next_to(n1, DOWN, buff=0.3)
        with self.say("So every block is two lines: read a normalized copy, add the result back. Virtually every modern "
                      "transformer keeps both ideas: a residual stream and a normalization layer.",
                      scene="Recap: the pre-LN block is x = x + attn(LN(x)); x = x + mlp(LN(x)). '+' = the highway that keeps "
                            "gradients alive; LN = keeps every layer's input at a sane scale.", chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ttl, pa, la, lna, lga, pc, lc, lgc)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(FadeIn(code, shift=UP * 0.2), run_time=0.6)
            self.mark("block-code", code, "a pre-LN transformer block: x = x + attn(LN(x)); x = x + mlp(LN(x))")
            self.sync("add the result", 0.1)
            self.play(FadeIn(n1, shift=UP * 0.1), FadeIn(n2, shift=UP * 0.1), run_time=0.5)
            self.mark("n-res", n1, "residual connection: gradient highway")
            self.mark("n-ln", n2, "LayerNorm: keeps activation scales sane")
        self.wait(0.8)
