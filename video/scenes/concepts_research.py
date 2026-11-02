import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from lib_concepts_viz import CScene

VIDEO = dict(id="research-loop", cls="ResearchLoop", title="The research loop: test your own idea",
             subtitle="Hypothesis → smallest experiment → baseline → one change → compare → scale. A real TinyGPT experiment",
             relates="invent", track="research", order=10, poster=("Compare", 0, 0.95))

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "data", "research_nopos.json")))   # written by video/exp_research.py
BASE, NOPOS, REL, LEAK = R["baseline"], R["nopos"], R["released"], R["leak"]
STEPS_ = ["hypothesis", "smallest experiment", "baseline", "change ONE thing", "compare across seeds", "scale up what wins"]
SCOL = [YELLOW, TEAL, BLUE, PINK, ORANGE, "#b48cff"]


class ResearchLoop(CScene):
    VIDEO_ID = "research-loop"

    def pill(self, text, col, size=26, w=None):
        t = self.T(text, size, col, BOLD)
        b = RoundedRectangle(corner_radius=0.2, width=w or t.width + 0.5, height=t.height + 0.42, stroke_color=col,
                             stroke_width=2.5, fill_color=col, fill_opacity=0.1)
        return VGroup(b, t.move_to(b))

    def construct(self):
        gap = NOPOS["val"][-1] - BASE["val"][-1]
        se = max(BASE["val_se"][-1], NOPOS["val_se"][-1])
        n = R["steps"]

        # ============================================================ the loop
        ang = [90, 30, -30, -90, -150, 150]
        pts = [np.array([4.1 * np.cos(np.radians(a)), 2.35 * np.sin(np.radians(a)), 0]) for a in ang]
        pills = VGroup(*[self.pill(t, c).move_to(p) for t, c, p in zip(STEPS_, SCOL, pts)])
        def ell(deg):
            return np.array([4.1 * np.cos(np.radians(deg)), 2.35 * np.sin(np.radians(deg)), 0])

        def inside(p, m, pad=0.12):
            ul, dr = m.get_corner(UL), m.get_corner(DR)
            return ul[0] - pad <= p[0] <= dr[0] + pad and dr[1] - pad <= p[1] <= ul[1] + pad
        arrows = VGroup()
        for i in range(6):
            a0, a1 = ang[i], ang[i] - 60
            ts = np.linspace(a0, a1, 200)
            ok = [t for t in ts if not inside(ell(t), pills[i]) and not inside(ell(t), pills[(i + 1) % 6])]
            arrows.add(CurvedArrow(ell(ok[0]), ell(ok[-1]), angle=-np.radians(ok[0] - ok[-1]) * 0.55, color=MUTED,
                                   stroke_width=3, tip_length=0.2))
        center = self.T("the research loop", 34, FG, BOLD)
        with self.say("Inventing a new model isn't one big idea. It's a loop: a hypothesis, the smallest experiment that "
                      "could test it, a baseline, one change, an honest comparison, and only then, scale up what wins. "
                      "Let's run it, for real, on TinyGPT.",
                      scene="The research loop as six pills around a circle: hypothesis → smallest experiment → baseline → "
                            "change ONE thing → compare across seeds → scale up what wins → back to hypothesis.",
                      chapter="The loop"):
            self.play(FadeIn(center), run_time=0.4)
            phrases = ["a hypothesis", "the smallest", "a baseline", "one change", "an honest", "scale up"]
            for i in range(6):
                self.sync(phrases[i], 0.15)
                self.play(FadeIn(pills[i], scale=1.15), *([Create(arrows[i - 1])] if i else []), run_time=0.4)
                self.mark(f"loop-{i}", pills[i], f"research loop step {i + 1}: {STEPS_[i]}")
            self.play(Create(arrows[5]), run_time=0.4)

        # ============================================================ hypothesis
        hdr = self.header("1 · Hypothesis", YELLOW)
        e1 = self.pill("token embedding", BLUE, 28)
        plus = self.M("+", 40)
        e2 = self.pill("position embedding", TEAL, 28)
        eq = self.M("= input to layer 1", 28, MUTED)
        row = VGroup(e1, plus, e2, eq).arrange(RIGHT, buff=0.3).move_to(UP * 1.6)
        cross = VGroup(Line(e2.get_corner(UL), e2.get_corner(DR), color=PINK, stroke_width=6),
                       Line(e2.get_corner(DL), e2.get_corner(UR), color=PINK, stroke_width=6))
        q = self.T("what if TinyGPT had no position embeddings?", 36, YELLOW, BOLD).move_to(UP * 0.3)
        # causal mask hint
        N = 6
        mask = VGroup()
        for r in range(N):
            for c in range(N):
                mask.add(Square(0.32, stroke_width=1, stroke_color=LINE, fill_color=TEAL if c <= r else BG,
                                fill_opacity=0.55 if c <= r else 1).move_to([c * 0.34, -r * 0.34, 0]))
        mask.move_to([-3.2, -1.95, 0])
        mask_l = VGroup(self.T("causal mask: token 1 sees 1 token,", 26, FG),
                        self.T("token 6 sees 6, so order leaks in", 26, FG),
                        self.T("prediction: maybe it can infer positions anyway?", 26, YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        mask_l.next_to(mask, RIGHT, buff=0.6)
        with self.say("Hypothesis: TinyGPT doesn't need its position embeddings. Attention by itself ignores word order, "
                      "but the causal mask leaks some: each token sees a different number of tokens before it. "
                      "So maybe it can work out positions on its own.",
                      scene="TinyGPT's input = token embedding + position embedding, with the position embedding crossed "
                            "out: 'what if TinyGPT had no position embeddings?'. A 6×6 causal mask: token k sees k tokens, "
                            "so some order information leaks in.", chapter="Hypothesis"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pills, arrows, center)), FadeIn(hdr), run_time=0.5)
            self.play(FadeIn(row, shift=UP * 0.2), run_time=0.6)
            self.mark("tok-emb", e1, "token embedding: one learned vector per token (256 × 64)")
            self.mark("pos-emb", e2, "position embedding: one learned vector per position (64 × 64 = 4,096 weights)")
            self.play(Create(cross), FadeIn(q), run_time=0.6)
            self.mark("question", q, "hypothesis: TinyGPT does not need learned position embeddings")
            self.sync("but the causal", 0.1)
            self.play(LaggedStart(*[FadeIn(s) for s in mask], lag_ratio=0.01), FadeIn(mask_l[:2]), run_time=0.9)
            self.mark("mask", mask, "causal mask: each position attends only to itself and earlier positions")
            self.sync("So maybe", 0.1)
            self.play(FadeIn(mask_l[2]), run_time=0.4)
            self.mark("mask-note", mask_l, "the causal mask gives the model an implicit signal about position")

        # ============================================================ smallest experiment
        hdr2 = self.header("2–4 · Smallest experiment, baseline, one change", TEAL)
        shared = VGroup(*[self.T(t, 26, FG) for t in (
            f"{n} steps each (TinyGPT's real run: 3,000)",
            f"batch {R['batch']} × 64 tokens · AdamW, lr {R['lr']:g}, cosine",
            f"same seed ({R['seed']}) · the exact same batches · same code",
            f"CPU, {R['threads']} threads: {BASE['seconds'] / 60:.0f} and {NOPOS['seconds'] / 60:.0f} minutes")]
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sh_t = self.T("identical in both runs", 26, MUTED, BOLD)
        shg = VGroup(sh_t, shared).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        self.fitw(shg, 8.2).move_to([-6.6, 0.55, 0], aligned_edge=LEFT)
        c1 = VGroup(self.T("baseline", 30, BLUE, BOLD), self.M("x = tok(idx) + pos(t)", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        c2 = VGroup(self.T("no positions", 30, PINK, BOLD), self.M("x = tok(idx)", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cards = VGroup()
        for c, col in ((c1, BLUE), (c2, PINK)):
            b = RoundedRectangle(corner_radius=0.15, width=4.2, height=c.height + 0.45, stroke_color=col, stroke_width=2.5,
                                 fill_color=col, fill_opacity=0.08)
            cards.add(VGroup(b, c.move_to(b).align_to(b, LEFT).shift(RIGHT * 0.25)))
        cards.arrange(DOWN, buff=0.35).move_to([6.7, 0.55, 0], aligned_edge=RIGHT)
        one = self.T("the only difference: one line", 30, YELLOW, BOLD).move_to(DOWN * 2.4)
        with self.say(f"The smallest test: two short runs of {n} steps, about four minutes each on my CPU. Same code, same "
                      f"data, same batches, same seed. The baseline is unchanged TinyGPT; the only difference is one line.",
                      scene=f"Experiment setup (real): both runs use {n} steps, batch {R['batch']}×64, AdamW lr {R['lr']}, "
                            f"cosine schedule, seed {R['seed']}, identical batches, CPU with 2 threads ({BASE['seconds']:.0f} s and "
                            f"{NOPOS['seconds']:.0f} s). Baseline: x = tok(idx) + pos(t). Variant: x = tok(idx).",
                      chapter="Experiment"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(row, cross, q, mask, mask_l)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(sh_t), LaggedStart(*[FadeIn(s, shift=RIGHT * 0.2) for s in shared], lag_ratio=0.25), run_time=1.4)
            self.mark("shared", shg, f"held fixed: {n} steps, batch 64×64, AdamW lr 3e-3 cosine, seed 0, same batches, same code")
            self.sync("The baseline", 0.1)
            self.play(FadeIn(cards[0], shift=LEFT * 0.2), run_time=0.5)
            self.mark("arm-base", cards[0], f"baseline: unchanged TinyGPT ({BASE['params']:,} params)")
            self.sync("the only difference", 0.1)
            self.play(FadeIn(cards[1], shift=LEFT * 0.2), FadeIn(one), run_time=0.5)
            self.mark("arm-nopos", cards[1], f"variant: position embedding removed ({NOPOS['params']:,} params used)")

        # ============================================================ compare
        hdr3 = self.header("Compare", ORANGE)
        S0, Y0, Y1 = 200, 0.62, 0.92
        i0 = BASE["steps"].index(S0)
        ax = Axes(x_range=[S0, n, 100], y_range=[Y0, Y1, 0.05], x_length=8.2, y_length=4.6, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).move_to([-1.9, -0.25, 0])
        xl = VGroup(*[self.M(str(v), 22, MUTED).next_to(ax.c2p(v, Y0), DOWN, buff=0.15) for v in range(S0, n + 1, 100)])
        yl = VGroup(*[self.M(f"{v:.2f}", 22, MUTED).next_to(ax.c2p(S0, v), LEFT, buff=0.15) for v in (0.65, 0.75, 0.85)])
        xt = self.T("training step", 22, MUTED).next_to(xl, DOWN, buff=0.12)
        yt = self.T("validation loss (held-out text)", 22, MUTED).next_to(ax, UP, buff=0.15).align_to(ax, LEFT)

        def curve(run, col):
            return VMobject(color=col, stroke_width=4).set_points_as_corners(
                [ax.c2p(s, v) for s, v in zip(run["steps"][i0:], run["val"][i0:])])
        cb, cn = curve(BASE, BLUE), curve(NOPOS, PINK)
        fin = VGroup(
            VGroup(self.T("baseline", 28, BLUE, BOLD), self.M(f"{BASE['val'][-1]:.3f}", 34, BLUE)).arrange(RIGHT, buff=0.3),
            VGroup(self.T("no positions", 28, PINK, BOLD), self.M(f"{NOPOS['val'][-1]:.3f}", 34, PINK)).arrange(RIGHT, buff=0.3),
            VGroup(self.T("gap", 28, YELLOW, BOLD), self.M(f"+{gap:.3f}", 34, YELLOW)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.35).move_to([4.6, 0.2, 0])
        fin_t = self.T(f"at step {n}, lower is better", 22, MUTED).next_to(fin, UP, buff=0.35)
        ev = self.T(f"measured on the same {R['n_eval_batches']} held-out batches ({R['eval_tokens']:,} tokens)", 22, MUTED)
        ev.to_edge(DOWN, buff=0.3)
        with self.say(f"Then compare, on held-out text, measured on the exact same batches. Baseline: "
                      f"{BASE['val'][-1]:.2f}. No positions: {NOPOS['val'][-1]:.2f}. Removing them hurt, by about "
                      f"{gap:.2f}.",
                      scene=f"Real validation-loss curves from step {S0} to {n} (y axis zoomed to {Y0}–{Y1}): baseline (blue) ends at "
                            f"{BASE['val'][-1]:.4f}, no-positions (pink) ends at {NOPOS['val'][-1]:.4f}; gap +{gap:.4f}. "
                            f"Evaluated on the same {R['n_eval_batches']} held-out batches ({R['eval_tokens']:,} tokens).",
                      chapter="Compare"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(shg, cards, one)), ReplacementTransform(hdr2, hdr3), run_time=0.5)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), FadeIn(ev), run_time=0.6)
            self.play(Create(cb), Create(cn), run_time=1.8, rate_func=linear)
            self.mark("curve-base", cb, f"baseline val loss: {BASE['val'][i0]:.3f} at step {S0} → {BASE['val'][-1]:.3f} at step {n}")
            self.mark("curve-nopos", cn, f"no-positions val loss: {NOPOS['val'][i0]:.3f} at step {S0} → {NOPOS['val'][-1]:.3f} at step {n}")
            self.sync("Baseline:", 0.1)
            self.play(FadeIn(fin_t), FadeIn(fin[0], shift=LEFT * 0.2), run_time=0.4)
            self.sync("No positions", 0.1)
            self.play(FadeIn(fin[1], shift=LEFT * 0.2), run_time=0.4)
            self.sync("Removing", 0.1)
            self.play(FadeIn(fin[2], scale=1.2), run_time=0.4)
            self.mark("final", fin, f"final val loss: baseline {BASE['val'][-1]:.4f}, no positions {NOPOS['val'][-1]:.4f}, "
                                    f"gap {gap:+.4f}")

        # ============================================================ seeds
        hdr4 = self.header("Seeds", ORANGE)
        nl = NumberLine(x_range=[0.62, 0.72, 0.02], length=10, color=MUTED, include_numbers=False).move_to(DOWN * 0.6)
        nlab = VGroup(*[self.M(f"{v:.2f}", 22, MUTED).next_to(nl.n2p(v), DOWN, buff=0.2) for v in np.arange(0.62, 0.7201, 0.02)])
        nt = self.T("validation loss at step 600", 22, MUTED).next_to(nlab, DOWN, buff=0.15)

        def dot(run, col, y):
            v, e = run["val"][-1], run["val_se"][-1]
            d = Dot(nl.n2p(v) + UP * y, radius=0.12, color=col)
            bar = Line(nl.n2p(v - 2 * e) + UP * y, nl.n2p(v + 2 * e) + UP * y, color=col, stroke_width=4)
            return VGroup(bar, d)
        db, dn = dot(BASE, BLUE, 0.9), dot(NOPOS, PINK, 0.9)
        lb = self.T("baseline · seed 0", 26, BLUE).next_to(db, UP, buff=0.3).align_to(db, RIGHT)
        ln = self.T("no positions · seed 0", 26, PINK).next_to(dn, UP, buff=0.3).align_to(dn, LEFT)
        gl = VGroup(self.T("a verdict needs 3+ seeds per arm,", 32, YELLOW, BOLD),
                    self.T("and a gap clearly bigger than the seed-to-seed spread", 32, YELLOW, BOLD)
                    ).arrange(DOWN, buff=0.15).move_to(UP * 2.55)
        self.fitw(gl, 12.6)
        note = VGroup(self.T(f"bars: ±2 × measurement noise ({se:.4f})", 22, MUTED),
                      self.T("seed-to-seed spread is not measured: it is usually larger", 22, MUTED)).arrange(DOWN, buff=0.1)
        note.to_edge(DOWN, buff=0.4)
        with self.say("But that's one seed per arm. Rerun with a different seed and both numbers move. A real verdict needs "
                      "three or more seeds each, and a gap that clearly beats the spread between them. So for now, "
                      "it's a promising hint, not a result.",
                      scene=f"Number line of val loss at step {n}: baseline seed 0 = {BASE['val'][-1]:.4f}, no positions "
                            f"seed 0 = {NOPOS['val'][-1]:.4f}, each with a ±2×SE measurement-noise bar (SE ≈ {se:.4f}). "
                            f"Only one seed per arm was run; text: a verdict needs 3+ seeds per arm and a gap clearly "
                            f"bigger than the seed-to-seed spread.",
                      chapter="Seeds"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax, xl, yl, xt, yt, cb, cn, fin, fin_t, ev)), ReplacementTransform(hdr3, hdr4), run_time=0.5)
            self.play(Create(nl), FadeIn(nlab), FadeIn(nt), run_time=0.5)
            self.play(FadeIn(db), FadeIn(dn), FadeIn(lb), FadeIn(ln), FadeIn(note), run_time=0.6)
            self.mark("dot-base", VGroup(db, lb), f"baseline, seed 0: {BASE['val'][-1]:.4f} ± {BASE['val_se'][-1]:.4f} (measurement)")
            self.mark("dot-nopos", VGroup(dn, ln), f"no positions, seed 0: {NOPOS['val'][-1]:.4f} ± {NOPOS['val_se'][-1]:.4f}")
            self.sync("A real verdict", 0.1)
            self.play(FadeIn(gl, shift=DOWN * 0.2), run_time=0.6)
            self.mark("need-seeds", gl, "rule of thumb: several seeds per arm; compare the gap to the seed spread")

        # ============================================================ traps
        hdr5 = self.header("Traps", PINK)
        fake = NOPOS["val"][-1] - REL["val"]
        cards = VGroup()
        specs = [
            ("different compute", ["vs the released model", f"(3,000 steps, {REL['val']:.3f}):", f"the gap looks {fake:.2f},",
                                   f"{fake / gap:.0f}× the real {gap:.3f}"], ORANGE),
            ("one seed", ["one lucky run", "can fake a win:", "look at the spread"], YELLOW),
            ("data leakage", [f"{LEAK['frac'] * 100:.0f}% of held-out stories", f"({LEAK['in_train']:,} of {LEAK['val_docs']:,}) also",
                              "appear in training:", "val loss partly", "measures memory"], PINK)]
        for title, lines, col in specs:
            t = VGroup(self.T(title, 30, col, BOLD), *[self.T(l, 26, FG) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            b = RoundedRectangle(corner_radius=0.16, width=4.25, height=3.6, stroke_color=col, stroke_width=2.5,
                                 fill_color=col, fill_opacity=0.07)
            self.fitw(t, 3.85)
            t.move_to(b).align_to(b, UP).shift(DOWN * 0.3)
            cards.add(VGroup(b, t))
        cards.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.1)
        with self.say(f"Three classic traps. Different compute: compare against the released model, trained five times "
                      f"longer, and the gap looks {fake / gap:.0f} times bigger than it is. One seed: luck can fake a win. "
                      f"And leakage: in TinyGPT's tiny world, {LEAK['frac'] * 100:.0f} percent of held-out stories also "
                      f"appear in training, so its validation loss partly measures memory.",
                      scene=f"Three trap cards. Different compute: comparing the 600-step no-positions run ({NOPOS['val'][-1]:.3f}) "
                            f"to the released 3,000-step model ({REL['val']:.3f}, same eval batches) suggests a {fake:.3f} gap, "
                            f"{fake / gap:.1f}× the real {gap:.3f}. One seed: luck can fake a win. Data leakage: "
                            f"{LEAK['in_train']:,} of {LEAK['val_docs']:,} held-out stories ({LEAK['frac'] * 100:.1f}%) appear "
                            f"verbatim in the training split.", chapter="Traps"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(nl, nlab, nt, db, dn, lb, ln, gl, note)), ReplacementTransform(hdr4, hdr5),
                      run_time=0.5)
            for k, ph in enumerate(["Different compute", "One seed", "And leakage"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(cards[k], shift=UP * 0.2), run_time=0.5)
            self.mark("trap-compute", cards[0], f"unequal compute: released model val {REL['val']:.4f} (3,000 steps) vs "
                                                f"no-pos {NOPOS['val'][-1]:.4f} (600 steps) → misleading gap {fake:.3f}")
            self.mark("trap-seed", cards[1], "single-seed comparisons can be noise")
            self.mark("trap-leak", cards[2], f"{LEAK['in_train']:,}/{LEAK['val_docs']:,} validation documents also occur in "
                                             f"the training split (templated synthetic corpus)")

        # ============================================================ scale up
        hdr6 = self.header("Scale up only what wins", "#b48cff")
        ladder = VGroup(*[self.pill(t, c, 28) for t, c in (
            ("more seeds", YELLOW), ("longer runs", TEAL), ("bigger model", BLUE), ("does the gap hold?", PINK))]
                        ).arrange(RIGHT, buff=0.5)
        self.fitw(ladder, 12.6).move_to(UP * 0.6)
        la = VGroup(*[Arrow(ladder[i].get_right(), ladder[i + 1].get_left(), buff=0.08, color=MUTED, stroke_width=3)
                      for i in range(3)])
        verdict = self.T(f"today's hint: at this size, the position embeddings earn their {BASE['params'] - NOPOS['params']:,} weights", 28, FG)
        verdict.move_to(DOWN * 1.0)
        cta = self.T("your turn: pick a hypothesis in the Invent lab →", 40, YELLOW, BOLD).move_to(DOWN * 2.5)
        self.fitw(cta, 12.8)
        with self.say("So positions win, for now, at this size. Next: more seeds, then longer runs, then a bigger model, "
                      "checking the gap holds at every step. Your turn: pick a hypothesis.",
                      scene="Scale-up ladder: more seeds → longer runs → bigger model → does the gap hold? Verdict: position "
                            "embeddings (4,096 weights) help at this size (one seed, 600 steps). CTA: Invent lab.",
                      chapter="Scale up"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(cards), ReplacementTransform(hdr5, hdr6), run_time=0.5)
            self.play(FadeIn(verdict), run_time=0.4)
            self.mark("verdict", verdict, f"result: removing position embeddings raised val loss by {gap:.3f} (1 seed, {n} steps)")
            self.sync("Next:", 0.1)
            for k in range(4):
                self.play(FadeIn(ladder[k], shift=RIGHT * 0.2), *([GrowArrow(la[k - 1])] if k else []), run_time=0.35)
            for k, p in enumerate(ladder):
                self.mark(f"ladder-{k}", p, ["rerun both arms with more seeds", "train longer", "try a bigger model",
                                              "keep checking the gap survives"][k])
            self.sync("Your turn", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "next: design your own experiment in the Invent lab")
        self.wait(0.6)
