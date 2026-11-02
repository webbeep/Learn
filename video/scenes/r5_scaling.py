import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vscene import *  # noqa
from lib_research import load, Plot, legend, pill, panel, illus, fmt_n

VIDEO = dict(id="scaling", cls="Scaling", title="Scaling laws: bigger, predictably",
             subtitle="Seven real TinyGPTs from ~10k to ~1.8M parameters, a fitted power law, and what Kaplan & Chinchilla found",
             relates="scaling", track="design", order=30, poster=("Loss vs size", 0, 0.95))

SC = load("scaling")
RUNS = SC["runs"]
FIT = SC["fit"]
TOK = RUNS[0]["tokens_seen"]
COLS = ["#5fd0ff", BLUE, "#b48cff", TEAL, YELLOW, ORANGE, PINK]


def law(n):
    return FIT["c"] + FIT["a"] * n ** (-FIT["alpha"])


def spk_n(n):
    return f"{n / 1e6:.1f} million" if n >= 1e6 else f"{round(n / 1e3)} thousand"


def pow10(v):
    return f"1e{int(round(math.log10(v)))}"


class Scaling(VScene):
    VIDEO_ID = "scaling"

    def construct(self):
        small, big = RUNS[0], RUNS[-1]
        vmax = max(r["curve"]["val"][2] for r in RUNS)
        ymax = math.ceil(min(5.6, vmax + 0.3))
        # ------------------------------------------------------------ hook
        pl = Plot([0, TOK / 1e6], [0, ymax], width=9.4, height=4.6, xticks=[0, 1, 2, 3], yticks=list(range(0, ymax + 1)),
                  xlabel="tokens seen (millions)", ylabel="validation loss").move_to(DOWN * 0.35 + LEFT * 1.4)
        cv = VGroup(*[pl.line([t / 1e6 for t in r["curve"]["tokens"]], r["curve"]["val"], c, 3.5) for r, c in zip(RUNS, COLS)])
        labs = VGroup()
        for r, c in zip(RUNS, COLS):
            labs.add(VGroup(Line(ORIGIN, RIGHT * 0.4, color=c, stroke_width=5), self.M(f"{fmt_n(r['params']):>5}", 22, c),
                            self.M(f"{r['final_val']:.2f}", 22, FG)).arrange(RIGHT, buff=0.2))
        labs.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(pl.ax, RIGHT, buff=0.5)
        lh = self.T("params · final loss", 22, MUTED).next_to(labs, UP, buff=0.25).align_to(labs, LEFT)
        with self.say(f"Seven TinyGPTs, from {spk_n(small['params'])} to {spk_n(big['params'])} parameters. Same data, same "
                      f"{TOK / 1e6:.1f} million tokens each. Bigger models land lower, with smaller gains at the top.",
                      scene=f"Real validation curves of 7 model sizes trained on the same {TOK:,} tokens: "
                            + ", ".join(f"{fmt_n(r['params'])} params → {r['final_val']:.3f}" for r in RUNS) + ".",
                      chapter="Bigger is better"):
            self.play(FadeIn(pl), FadeIn(lh), run_time=0.5)
            self.play(LaggedStart(*[Create(c) for c in cv], lag_ratio=0.1),
                      LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in labs], lag_ratio=0.1), run_time=2.6)
            for r, c, l in zip(RUNS, cv, labs):
                self.mark("size-" + r["id"], l, f"{r['params']:,} params (d_model {r['d_model']}, {r['n_layer']} layers): "
                                                  f"final val loss {r['final_val']:.3f}")
            self.mark("curves", cv, f"7 real training curves, each {TOK:,} tokens")

        # ------------------------------------------------------------ loss vs size (log x)
        hdr = self.header("Loss vs size")
        N = [r["non_embedding_params"] for r in RUNS]
        L = [r["final_val"] for r in RUNS]
        xlo = 10 ** math.floor(math.log10(min(N))); xhi = 10 ** math.ceil(math.log10(max(N)))
        ylo, yhi = 0.3, math.ceil(max(L) * 5) / 5 + 0.1
        yt = [round(0.4 + k * 0.2, 1) for k in range(int((yhi - 0.4) / 0.2) + 1)]
        ps = Plot([xlo, xhi], [ylo, yhi], width=8.0, height=4.6, xlog=True,
                  xticks=[10 ** e for e in range(int(math.log10(xlo)), int(math.log10(xhi)) + 1)], yticks=yt,
                  xlabel="non-embedding parameters (log scale)", ylabel="final validation loss",
                  xfmt=lambda v: fmt_n(v)).move_to(DOWN * 0.4 + LEFT * 2.3)
        dots = VGroup(*[Dot(ps.p(n, l), radius=0.1, color=c) for n, l, c in zip(N, L, COLS)])
        x_fit0 = max(math.log10(xlo), math.log10(((yhi - FIT["c"]) / FIT["a"]) ** (-1 / FIT["alpha"])) + 0.02)
        fit = ps.ax.plot(lambda lx: law(10 ** lx), x_range=[x_fit0, math.log10(xhi), 0.02], color=FG,
                         stroke_width=3)
        floor = DashedLine(ps.p(xlo, FIT["c"]), ps.p(xhi, FIT["c"]), color=MUTED, stroke_width=2, dash_length=0.1)
        fl = self.T(f"floor c = {FIT['c']:.2f}", 22, MUTED).next_to(ps.p(xlo, FIT["c"]), DOWN, buff=0.1).shift(RIGHT * 1.2)
        form = VGroup(self.M("L(N) = c + a · N", 30, FG), self.M("-α", 22, FG)).arrange(RIGHT, buff=0.04, aligned_edge=UP)
        nums = VGroup(self.M(f"a = {FIT['a']:.2f}", 26, MUTED), self.M(f"α = {FIT['alpha']:.2f}", 26, YELLOW),
                      self.M(f"c = {FIT['c']:.2f}", 26, MUTED), self.M(f"R² = {FIT['r2']:.3f}", 22, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = VGroup(form, nums).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(ps.ax, RIGHT, buff=0.45).shift(UP * 0.9)
        with self.say(f"Plot final loss against size on a log axis, and the dots fall on a smooth curve: a power law. "
                      f"Every time you multiply the size, the loss above a floor shrinks by a predictable factor. Here the "
                      f"floor is {FIT['c']:.2f}.",
                      scene=f"Real final losses vs non-embedding parameters (log x) with the fitted power law L(N) = c + a·N^-α: "
                            f"a = {FIT['a']:.3f}, α = {FIT['alpha']:.3f}, c = {FIT['c']:.3f} (R² = {FIT['r2']:.3f}).",
                      chapter="Loss vs size"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pl, cv, labs, lh)), FadeIn(hdr), run_time=0.4)
            self.play(FadeIn(ps), LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.15), run_time=1.2)
            for r, d in zip(RUNS, dots):
                self.mark("pt-" + r["id"], d, f"{r['non_embedding_params']:,} non-embedding params → val {r['final_val']:.3f} "
                                              f"(fit {law(r['non_embedding_params']):.3f})")
            self.sync("a power law", 0.1)
            self.play(Create(fit), FadeIn(box, shift=LEFT * 0.2), run_time=1.0)
            self.mark("fit", fit, f"fitted power law: L = {FIT['c']:.3f} + {FIT['a']:.3f}·N^-{FIT['alpha']:.3f}")
            self.mark("formula", box, f"fit parameters a = {FIT['a']:.3f}, alpha = {FIT['alpha']:.3f}, c = {FIT['c']:.3f}")
            self.sync("Here the", 0.1)
            self.play(Create(floor), FadeIn(fl), run_time=0.6)
            self.mark("floor", VGroup(floor, fl), f"irreducible loss of this data (fit): c = {FIT['c']:.3f}")

        # ------------------------------------------------------------ compute
        hdr2 = self.header("Loss vs compute")
        F_all = [f for r in RUNS for f in r["curve"]["flops"] if f > 0]
        fxlo = 10 ** math.floor(math.log10(min(F_all))); fxhi = 10 ** math.ceil(math.log10(max(F_all)))
        pc = Plot([fxlo, fxhi], [0, ymax], width=8.4, height=4.6, xlog=True,
                  xticks=[10 ** e for e in range(int(math.log10(fxlo)), int(math.log10(fxhi)) + 1, 2)],
                  yticks=list(range(0, ymax + 1)), xlabel="training compute, FLOPs ≈ 6 · params · tokens (log scale)",
                  ylabel="validation loss", xfmt=pow10).move_to(DOWN * 0.35 + LEFT * 2.0)
        cc = VGroup(*[pc.line(r["curve"]["flops"][1:], r["curve"]["val"][1:], c, 3) for r, c in zip(RUNS, COLS)])
        # frontier: for a grid of budgets, best loss any run reached within that budget
        import bisect
        grid = [10 ** (math.log10(fxlo) + k * (math.log10(fxhi) - math.log10(fxlo)) / 120) for k in range(121)]
        front, who = [], []
        for B in grid:
            best, bi = None, None
            for i, r in enumerate(RUNS):
                vals = [v for f, v in zip(r["curve"]["flops"], r["curve"]["val"]) if 0 < f <= B]
                if vals and (best is None or min(vals) < best):
                    best, bi = min(vals), i
            front.append(best); who.append(bi)
        fpts = [(g, f) for g, f in zip(grid, front) if f is not None]
        fr = pc.line([g for g, _ in fpts], [f for _, f in fpts], FG, 9, opacity=0.4)
        winners = []
        for g, w in zip(grid, who):
            if w is not None and (not winners or winners[-1][1] != w):
                winners.append((g, w))
        win_txt = " → ".join(fmt_n(RUNS[w]["params"]) for _, w in winners)
        lab_f = self.T("best loss for each budget", 24, FG).next_to(pc.p(fxlo * 30, 1.0), DOWN, buff=0.1) if False else \
            self.T("best per budget", 22, FG)
        lab_f.next_to(pc.ax, RIGHT, buff=0.35).shift(DOWN * 0.8)
        labs2 = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.4, color=c, stroke_width=5), self.M(fmt_n(r["params"]), 22, c))
                         .arrange(RIGHT, buff=0.2) for r, c in zip(RUNS, COLS)]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        labs2.next_to(pc.ax, RIGHT, buff=0.35).shift(UP * 0.9)
        lf_sw = VGroup(Line(ORIGIN, RIGHT * 0.4, color=FG, stroke_width=9, stroke_opacity=0.4), lab_f).arrange(RIGHT, buff=0.2)
        lf_sw.next_to(labs2, DOWN, buff=0.35).align_to(labs2, LEFT)
        with self.say(f"Now plot the same runs against compute. For a tiny budget, small models win. As the budget grows, "
                      f"the best choice keeps shifting to bigger models. Following the lower edge, the winner goes from "
                      f"{spk_n(RUNS[winners[0][1]]['params'])} to {spk_n(RUNS[winners[-1][1]]['params'])} parameters.",
                      scene=f"Real validation loss vs training FLOPs (log x) for all 7 sizes; the faint white envelope is the "
                            f"best loss reachable at each budget. Winners along it, small to large budget: {win_txt}.",
                      chapter="Loss vs compute"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ps, dots, fit, floor, fl, box)), ReplacementTransform(hdr, hdr2), run_time=0.4)
            self.play(FadeIn(pc), FadeIn(labs2), run_time=0.5)
            self.play(LaggedStart(*[Create(c) for c in cc], lag_ratio=0.12), run_time=2.0)
            for r, c in zip(RUNS, cc):
                self.mark("fl-" + r["id"], c, f"{fmt_n(r['params'])} model: val loss vs compute, up to {r['flops']:.2e} FLOPs")
            self.sync("Following", 0.1)
            self.play(Create(fr), FadeIn(lf_sw), run_time=1.2)
            self.mark("frontier", VGroup(fr, lf_sw), f"compute frontier (real): best run for each budget, {win_txt}")

        # ------------------------------------------------------------ the papers
        hdr3 = self.header("What the papers found")
        k = VGroup(self.T("Kaplan et al., 2020", 32, BLUE, BOLD),
                   self.T("loss falls as a smooth power law in parameters, data and compute,", 26, FG),
                   self.T("across many orders of magnitude", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        c = VGroup(self.T("Hoffmann et al., 2022 (Chinchilla)", 32, TEAL, BOLD),
                   self.T("for a fixed compute budget, grow parameters and training tokens", 26, FG),
                   self.T("in roughly equal proportion", 26, FG),
                   self.T("70B params on 1.4T tokens ≈ 20 tokens / param", 26, YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        W_ = max(k.width, c.width) + 0.9
        pk = VGroup(panel(W_, k.height + 0.6, BLUE), k)
        pc_ = VGroup(panel(W_, c.height + 0.6, TEAL), c)
        k.move_to(pk[0]).align_to(pk[0], LEFT).shift(RIGHT * 0.45); c.move_to(pc_[0]).align_to(pc_[0], LEFT).shift(RIGHT * 0.45)
        VGroup(pk, pc_).arrange(DOWN, buff=0.45).move_to(DOWN * 0.2)
        with self.say("Big labs found the same shape at vastly larger scale. Kaplan and colleagues, 2020: loss follows "
                      "smooth power laws in parameters, data and compute. Then Chinchilla, 2022: for a fixed budget, grow "
                      "the model and the training data together. Their 70 billion parameter model saw 1.4 trillion tokens, "
                      "about 20 per parameter.",
                      scene="Two cards. Kaplan et al. 2020: loss falls as a smooth power law in parameters, data and compute "
                            "over many orders of magnitude. Hoffmann et al. 2022 (Chinchilla): at fixed compute, scale "
                            "parameters and tokens in roughly equal proportion; Chinchilla = 70B params on 1.4T tokens "
                            "(≈ 20 tokens per parameter).", chapter="What the papers found"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pc, cc, fr, labs2, lf_sw)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(pk, shift=UP * 0.2), run_time=0.5)
            self.mark("kaplan", pk, "Kaplan et al. 2020, 'Scaling Laws for Neural Language Models'")
            self.sync("Then Chinchilla", 0.1)
            self.play(FadeIn(pc_, shift=UP * 0.2), run_time=0.5)
            self.mark("chinchilla", pc_, "Hoffmann et al. 2022, 'Training Compute-Optimal Large Language Models' (Chinchilla)")

        # ------------------------------------------------------------ our tokens per param
        hdr4 = self.header("Our runs, in that light")
        tpp = [TOK / r["params"] for r in RUNS]
        pb = Plot([0.5, 3000], [0, len(RUNS)], width=8.0, height=4.4, xlog=True, xticks=[1, 10, 100, 1000], yticks=[],
                  xlabel="training tokens per parameter (log scale)", ylabel="").move_to(DOWN * 0.4 + RIGHT * 0.9)
        bars = VGroup(); names = VGroup()
        for i, (r, t, col) in enumerate(zip(RUNS, tpp, COLS)):
            y = len(RUNS) - i - 0.5
            w = pb.p(t, y)[0] - pb.p(0.5, y)[0]
            b = Rectangle(width=max(0.05, w), height=0.4, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to(pb.p(0.5, y), aligned_edge=LEFT)
            v = self.M(f"{t:,.0f}" if t >= 10 else f"{t:.1f}", 22, FG).next_to(b, RIGHT, buff=0.12)
            n = self.M(f"{fmt_n(r['params'])}", 22, col).next_to(pb.p(0.5, y), LEFT, buff=0.2)
            bars.add(VGroup(b, v)); names.add(n)
        ch = DashedLine(pb.p(20, 0), pb.p(20, len(RUNS)), color=YELLOW, stroke_width=3, dash_length=0.1)
        chl = self.T("≈ 20 (Chinchilla)", 24, YELLOW).next_to(pb.p(20, len(RUNS)), UP, buff=0.12)
        under = [r for r, t in zip(RUNS, tpp) if t < 20]
        with self.say(f"By that yardstick, our {TOK / 1e6:.1f} million tokens are plenty for the small models, but our "
                      f"{len(under)} biggest are undertrained. The largest saw just {tpp[-1]:.1f} tokens per parameter: "
                      f"by Chinchilla's yardstick, it's starved for data.",
                      scene=f"Real tokens-per-parameter of each run ({TOK:,} tokens / params): "
                            + ", ".join(f"{fmt_n(r['params'])}: {t:.1f}" for r, t in zip(RUNS, tpp))
                            + ". A yellow dashed line marks Chinchilla's ≈ 20 tokens per parameter.",
                      chapter="Our runs, in that light"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pk, pc_)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(FadeIn(pb), FadeIn(names), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1),
                      run_time=1.3)
            for r, b, t in zip(RUNS, bars, tpp):
                self.mark("tpp-" + r["id"], b, f"{r['params']:,} params trained on {TOK:,} tokens = {t:.1f} tokens/param")
            self.sync("undertrained", 0.1)
            self.play(Create(ch), FadeIn(chl), run_time=0.6)
            self.mark("chin-line", VGroup(ch, chl), "Chinchilla rule of thumb: about 20 training tokens per parameter")

        # ------------------------------------------------------------ takeaways
        hdr5 = self.header("For your own models")
        tips = VGroup(*[pill(t, c, 30) for t, c in [
            ("train a ladder of small models first", BLUE),
            ("fit the curve, predict the big one", YELLOW),
            ("balance size and data for your budget", TEAL),
            ("the floor is set by the data", PINK)]]).arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to(UP * 0.2)
        cav = self.T(f"caveat: our fit is {len(RUNS)} tiny models, one seed, one learning rate: the shape, not a law", 22,
                     MUTED).to_edge(DOWN, buff=0.5)
        with self.say("For your own models: train a ladder of small ones first, fit the curve, and predict before you "
                      "spend. Balance size and data for your budget. And remember the floor comes from the data, not "
                      "the model.",
                      scene="Takeaways: train a ladder of small models first; fit the curve, predict the big one; balance "
                            "size and data for your budget; the loss floor is set by the data. Caveat: our fit uses 7 tiny "
                            "models, one seed and one learning rate.", chapter="For your own models"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pb, bars, names, ch, chl)), ReplacementTransform(hdr4, hdr5), run_time=0.4)
            for k_, ph in enumerate(["train a ladder", "fit the curve", "Balance", "the floor"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(tips[k_], shift=RIGHT * 0.2), run_time=0.3)
                self.mark(f"tip-{k_}", tips[k_], tips[k_][1].text)
            self.play(FadeIn(cav), run_time=0.3)
            self.mark("caveat", cav, "caveat: small-scale fit, single seed and learning rate")
        self.wait(0.8)
