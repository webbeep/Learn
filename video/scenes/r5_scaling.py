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
ABL = load("ablations"); OVF = load("overfit")
BASE_RUN = next(r for r in RUNS if r["id"] == "base")
ABL_BASE = next(r for r in ABL["runs"] if r["id"] == "baseline")
ABL_TOK = ABL["steps"] * ABL["batch"] * 64
OV256 = next(r for r in OVF["runs"] if r["data_tokens"] == 262144)
OVALL = OVF["runs"][-1]


def nfmt(v):
    """1k, 10k, 100k, 1M, 10M"""
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"
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
        ps = Plot([xlo, xhi], [ylo, yhi], width=7.6, height=4.6, xlog=True,
                  xticks=[10 ** e for e in range(int(math.log10(xlo)), int(math.log10(xhi)) + 1)], yticks=yt,
                  xlabel="non-embedding parameters (log scale)", ylabel="final validation loss",
                  xfmt=nfmt).move_to(DOWN * 0.4 + LEFT * 2.3)
        dots = VGroup(*[Dot(ps.p(n, l), radius=0.1, color=c) for n, l, c in zip(N, L, COLS)])
        x_fit0 = max(math.log10(xlo), math.log10(((yhi - FIT["c"]) / FIT["a"]) ** (-1 / FIT["alpha"])) + 0.02)
        x_fit1 = math.log10(max(N))
        fit = VGroup(ps.ax.plot(lambda lx: law(10 ** lx), x_range=[x_fit0, x_fit1, 0.02], color=FG, stroke_width=3),
                     DashedVMobject(ps.ax.plot(lambda lx: law(10 ** lx), x_range=[x_fit1, math.log10(xhi), 0.02],
                                               color=FG, stroke_width=2.5), num_dashes=12))
        floor = DashedLine(ps.p(xlo, FIT["c"]), ps.p(xhi, FIT["c"]), color=MUTED, stroke_width=2, dash_length=0.1)
        fl = self.T("c", 22, MUTED).next_to(ps.p(xhi, FIT["c"]), RIGHT, buff=0.1)
        form = VGroup(self.T("L = c + a · N", 32, FG), self.T("−alpha", 24, FG)).arrange(RIGHT, buff=0.04, aligned_edge=UP)
        nums = VGroup(self.T(f"a = {FIT['a']:.2f}", 26, MUTED), self.T(f"alpha = {FIT['alpha']:.2f}", 26, YELLOW),
                      self.T(f"c = {FIT['c']:.2f}", 26, MUTED), self.T(f"R² = {FIT['r2']:.3f}", 22, MUTED),
                      self.T("(Kaplan, web text: alpha ≈ 0.076)", 20, MUTED),
                      self.T(f"c = where this curve flattens", 20, MUTED), self.T(f"at {TOK / 1e6:.1f}M tokens, not a true floor", 20, MUTED)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        box = VGroup(form, nums).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(ps.ax, RIGHT, buff=0.45).shift(UP * 0.9)
        with self.say(f"Plot final loss against size on a log axis, and the dots fall on a smooth curve: a power law. "
                      f"Every time you multiply the size, the loss above a floor shrinks by a predictable factor. Here it "
                      f"flattens near {FIT['c']:.2f}, but only for this token budget: the same 171 thousand parameter "
                      f"model trained on four times more tokens reaches {ABL_BASE['final_val']:.2f} instead of "
                      f"{BASE_RUN['final_val']:.2f}.",
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
            self.sync("Here it", 0.1)
            self.play(Create(floor), FadeIn(fl), run_time=0.6)
            self.mark("floor", VGroup(floor, fl), f"fitted floor c = {FIT['c']:.3f} at a fixed {TOK:,}-token budget; it moves "
                                                  f"with data (171k model on {ABL_TOK:,} tokens: {ABL_BASE['final_val']:.3f})")
            self.sync("four times", 0.1)
            dot_m = Dot(ps.p(BASE_RUN["non_embedding_params"], ABL_BASE["final_val"]), radius=0.1, color=YELLOW)
            lab_m = self.T(f"same 171k model, {ABL_TOK / TOK:.0f}× tokens: {ABL_BASE['final_val']:.2f}", 22, YELLOW) \
                .move_to(ps.p(3e5, 0.85))
            more = VGroup(dot_m, Arrow(lab_m.get_bottom(), dot_m.get_center(), buff=0.15, color=YELLOW, stroke_width=3), lab_m)
            self.play(FadeIn(more, scale=1.3), run_time=0.4)
            self.mark("more-data", more, f"the 171k model trained on {ABL_TOK:,} tokens (ablation baseline) reaches "
                                         f"{ABL_BASE['final_val']:.3f} vs {BASE_RUN['final_val']:.3f} at {TOK:,} tokens: the flattening point moves with data")

        # ------------------------------------------------------------ compute
        hdr2 = self.header("Loss vs compute")
        F_all = [f for r in RUNS for f in r["curve"]["flops"] if f > 0]
        fxlo = 10 ** math.floor(math.log10(min(F_all))); fxhi = 10 ** math.ceil(math.log10(max(F_all)))
        pc = Plot([fxlo, fxhi], [0, ymax], width=8.4, height=4.6, xlog=True,
                  xticks=[10 ** e for e in range(int(math.log10(fxlo)), int(math.log10(fxhi)) + 1, 2)],
                  yticks=list(range(0, ymax + 1)), xlabel="training compute, FLOPs ≈ 6 · params · tokens (log scale)",
                  ylabel="validation loss", xfmt=pow10).move_to(DOWN * 0.1 + LEFT * 2.0)
        cc = VGroup(*[pc.line(r["curve"]["flops"][1:], r["curve"]["val"][1:], c, 3) for r, c in zip(RUNS, COLS)])
        # frontier: interpolate every curve in log-FLOPs and take the pointwise minimum (the lower envelope)
        grid = [10 ** (math.log10(fxlo) + k * (math.log10(fxhi) - math.log10(fxlo)) / 240) for k in range(241)]
        def interp(r, B):
            fs, vs = r["curve"]["flops"][1:], r["curve"]["val"][1:]
            if B < fs[0] or B > fs[-1]:
                return None
            lb = math.log10(B)
            for (f0, v0), (f1, v1) in zip(zip(fs, vs), zip(fs[1:], vs[1:])):
                if f0 <= B <= f1:
                    t = (lb - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
                    return v0 + t * (v1 - v0)
            return vs[-1]
        front, who = [], []
        for B in grid:
            cand = [(interp(r, B), i) for i, r in enumerate(RUNS)]
            cand = [c_ for c_ in cand if c_[0] is not None]
            if cand:
                v, i = min(cand); front.append(v); who.append(i)
            else:
                front.append(None); who.append(None)
        fpts = [(g, f) for g, f in zip(grid, front) if f is not None]
        fr = pc.line([g for g, _ in fpts], [f for _, f in fpts], FG, 8, opacity=0.45)
        frc = self.T("caveat: mid-run checkpoints; a run planned for each budget does a bit better",
                     22, MUTED).to_edge(DOWN, buff=0.4)
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
                      scene=f"Real validation loss vs training FLOPs (log x; FLOPs = 6 × all params × tokens) for all 7 sizes; "
                            f"the thick gray line is the lower envelope (best loss at each budget, curves interpolated in "
                            f"log FLOPs). Winners along it, small to large budget: {win_txt}. Caveat: intermediate "
                            f"checkpoints of fixed-length cosine runs.",
                      chapter="Loss vs compute"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ps, dots, fit, floor, fl, box, more)), ReplacementTransform(hdr, hdr2), run_time=0.4)
            self.play(FadeIn(pc), FadeIn(labs2), run_time=0.5)
            self.play(LaggedStart(*[Create(c) for c in cc], lag_ratio=0.12), run_time=2.0)
            for r, c in zip(RUNS, cc):
                self.mark("fl-" + r["id"], c, f"{fmt_n(r['params'])} model: val loss vs compute, up to {r['flops']:.2e} FLOPs")
            self.sync("Following", 0.1)
            self.play(Create(fr), FadeIn(lf_sw), FadeIn(frc), run_time=1.2)
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
            self.play(FadeOut(VGroup(pc, cc, fr, labs2, lf_sw, frc)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(pk, shift=UP * 0.2), run_time=0.5)
            self.mark("kaplan", pk, "Kaplan et al. 2020, 'Scaling Laws for Neural Language Models'")
            self.sync("Then Chinchilla", 0.1)
            self.play(FadeIn(pc_, shift=UP * 0.2), run_time=0.5)
            self.mark("chinchilla", pc_, "Hoffmann et al. 2022, 'Training Compute-Optimal Large Language Models' (Chinchilla)")

        # ------------------------------------------------------------ our tokens per param
        hdr4 = self.header("Our runs, in that light")
        tpp = [TOK / r["params"] for r in RUNS]
        pb = Plot([0.5, 3000], [0, len(RUNS)], width=8.0, height=3.9, xlog=True, xticks=[1, 10, 100, 1000], yticks=[],
                  xlabel="training tokens per parameter (log scale)", ylabel="").move_to(DOWN * 0.25 + RIGHT * 0.9)
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
        chl = self.T("≈ 20: Chinchilla's compute-optimal rule of thumb (large web-text runs)", 22, YELLOW) \
            .next_to(pb.p(20, len(RUNS)), UP, buff=0.12)
        rep = VGroup(self.T(f"but our data repeats: for the 171k model, {OV256['data_tokens'] // 1024}k tokens already matched", 22, MUTED),
                     self.T(f"all {OVALL['data_tokens'] / 1e6:.1f}M (val {OV256['final_val']:.3f} vs {OVALL['final_val']:.3f})", 22, MUTED)) \
            .arrange(DOWN, buff=0.06).to_edge(DOWN, buff=0.4)
        under = [r for r, t in zip(RUNS, tpp) if t < 20]
        with self.say(f"By Chinchilla's rule of thumb, derived for huge web-text runs, our {len(under)} biggest models "
                      f"would want more tokens: the largest saw just {tpp[-1]:.1f} per parameter. But our data repeats "
                      f"itself, so treat that as a rough guide. And real labs often train far past 20 tokens per "
                      f"parameter, to make models cheaper to run.",
                      scene=f"Real tokens-per-parameter of each run ({TOK:,} tokens / params): "
                            + ", ".join(f"{fmt_n(r['params'])}: {t:.1f}" for r, t in zip(RUNS, tpp))
                            + ". A yellow dashed line marks Chinchilla's ≈ 20 tokens per parameter (a compute-optimal "
                              "allocation heuristic for large web-text runs, not a 'trained enough' threshold). Our corpus is "
                              f"highly redundant: the 171k model reached val {OV256['final_val']:.3f} on 256k tokens vs "
                              f"{OVALL['final_val']:.3f} on all of them.",
                      chapter="Our runs, in that light"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pk, pc_)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(FadeIn(pb), FadeIn(names), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1),
                      run_time=1.3)
            for r, b, t in zip(RUNS, bars, tpp):
                self.mark("tpp-" + r["id"], b, f"{r['params']:,} params trained on {TOK:,} tokens = {t:.1f} tokens/param")
            self.play(Create(ch), FadeIn(chl), run_time=0.6)
            self.mark("chin-line", VGroup(ch, chl), "Chinchilla rule of thumb: about 20 training tokens per parameter for "
                                                    "compute-optimal large web-text runs")
            self.sync("But our data", 0.1)
            self.play(FadeIn(rep), run_time=0.4)
            self.mark("repeats", rep, f"redundant corpus: 256k tokens ≈ all 1.6M for the 171k model")

        # ------------------------------------------------------------ takeaways
        hdr5 = self.header("For your own models")
        tip_txt = [("train a ladder of small models first", BLUE), ("fit the curve, predict the big one", YELLOW),
                   ("balance size and data for your budget", TEAL), ("where it flattens depends on the data", PINK)]
        tips = VGroup(*[pill(t, c, 30) for t, c in tip_txt]).arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to(UP * 0.4)
        cav = self.T(f"caveat: {len(RUNS)} tiny models · one seed · one learning rate for every size", 26,
                     MUTED).to_edge(DOWN, buff=0.5)
        with self.say("For your own models: train a ladder of small ones first, fit the curve, and predict before you "
                      "spend. Balance size and data for your budget, and remember where the curve flattens depends on "
                      "the data. Our fit is seven tiny models, one seed, one learning rate: the shape, not a law.",
                      scene="Takeaways: train a ladder of small models first; fit the curve, predict the big one; balance "
                            "size and data for your budget; where the curve flattens depends on the data. Caveat: our fit "
                            "uses 7 tiny models, one seed and one learning rate for every size.",
                      chapter="For your own models"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pb, bars, names, ch, chl, rep)), ReplacementTransform(hdr4, hdr5), run_time=0.4)
            for k_, ph in enumerate(["train a ladder", "fit the curve", "Balance", "where the curve"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(tips[k_], shift=RIGHT * 0.2), run_time=0.3)
                self.mark(f"tip-{k_}", tips[k_], "takeaway: " + tip_txt[k_][0])
            self.sync("Our fit", 0.1)
            self.play(FadeIn(cav), run_time=0.3)
            self.mark("caveat", cav, "caveat: small-scale fit, single seed and learning rate")
        self.wait(0.8)
