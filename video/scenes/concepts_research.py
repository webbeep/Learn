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
SHORT, TRADE = LEAK["by_kind"]["short"], LEAK["by_kind"]["trade"]
SEEDS = ["0"] + sorted(R["seed_runs"]["baseline"])                      # extra seeds from exp_research.py --seeds
VB = [BASE["val"][-1]] + [R["seed_runs"]["baseline"][k]["val"] for k in SEEDS[1:]]
VN = [NOPOS["val"][-1]] + [R["seed_runs"]["nopos"][k]["val"] for k in SEEDS[1:]]
GAPS = [n_ - b_ for n_, b_ in zip(VN, VB)]                              # paired: same seed, same batches
SECS = [BASE["seconds"], NOPOS["seconds"]] + [R["seed_runs"][a][k]["seconds"] for a in ("baseline", "nopos") for k in SEEDS[1:]]
ABL = {r["id"]: r for r in json.load(open(os.path.join(HERE, "..", "..", "public", "research", "ablations.json")))["runs"]}
A_BASE, A_NOPOS = ABL["baseline"]["final_val"], ABL["no-positions"]["final_val"]
assert all(g > 0 for g in GAPS)          # the narration says every seed agrees
assert SHORT["in_train"] == SHORT["n"]   # the narration says "all" of the short templated lines
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
            f"CPU, {R['threads']} threads: {min(SECS) / 60:.1f}–{max(SECS) / 60:.1f} min per run (machine load varies)")]
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sh_t = self.T("identical in both runs", 26, MUTED, BOLD)
        shg = VGroup(sh_t, shared).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        self.fitw(shg, 7.6).move_to([-6.6, 0.55, 0], aligned_edge=LEFT)
        c1 = VGroup(self.T("baseline", 30, BLUE, BOLD), self.M("x = tok(idx) + pos(t)", 24, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        c2 = VGroup(self.T("no positions", 30, PINK, BOLD), self.M("x = tok(idx)", 24, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cards = VGroup()
        for c, col in ((c1, BLUE), (c2, PINK)):
            b = RoundedRectangle(corner_radius=0.15, width=max(4.2, c1.width + 0.6, c2.width + 0.6), height=c.height + 0.45, stroke_color=col, stroke_width=2.5,
                                 fill_color=col, fill_opacity=0.08)
            cards.add(VGroup(b, c.move_to(b).align_to(b, LEFT).shift(RIGHT * 0.25)))
        cards.arrange(DOWN, buff=0.35).move_to([6.7, 0.55, 0], aligned_edge=RIGHT)
        one = self.T("the only difference: one line", 30, YELLOW, BOLD).move_to(DOWN * 2.4)
        with self.say('The smallest test: short runs of 600 steps, a few minutes at most on two CPU threads. Same code, same data, same batches, same seed. The baseline is unchanged TinyGPT; the only difference is one line.',
                      scene=f"Experiment setup (real): both runs use {n} steps, batch {R['batch']}×64, AdamW lr {R['lr']}, "
                            f"cosine schedule, seed {R['seed']}, identical batches, CPU with 2 threads ({min(SECS):.0f}–{max(SECS):.0f} s "
                            f"per run). Baseline: x = tok(idx) + pos(t). Variant: x = tok(idx).",
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
        yt = self.T("validation loss (held-out text) · zoomed: axis starts at 0.62, not 0", 22, MUTED)
        yt.next_to(ax, UP, buff=0.15).align_to(ax, LEFT)
        brk = self.M("≈", 30, MUTED).move_to(ax.c2p(S0, Y0) + LEFT * 0.0 + UP * 0.12)
        legend = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.5, color=BLUE, stroke_width=4), self.T("baseline", 22, BLUE)).arrange(RIGHT, buff=0.12),
                        VGroup(Line(ORIGIN, RIGHT * 0.5, color=PINK, stroke_width=4), self.T("no positions", 22, PINK)).arrange(RIGHT, buff=0.12)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(ax.c2p(n, Y1), DL, buff=0.1)

        def curve(run, col):
            return VMobject(color=col, stroke_width=4).set_points_as_corners(
                [ax.c2p(s, v) for s, v in zip(run["steps"][i0:], run["val"][i0:])])
        cb, cn = curve(BASE, BLUE), curve(NOPOS, PINK)
        fin = VGroup(
            VGroup(self.T("baseline", 28, BLUE, BOLD), self.M(f"{BASE['val'][-1]:.3f}", 34, BLUE)).arrange(RIGHT, buff=0.3),
            VGroup(self.T("no positions", 28, PINK, BOLD), self.M(f"{NOPOS['val'][-1]:.3f}", 34, PINK)).arrange(RIGHT, buff=0.3),
            VGroup(self.T("paired gap", 28, YELLOW, BOLD), self.M(f"+{gap:.3f}", 34, YELLOW)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.35).move_to([4.6, 0.2, 0])
        fin_t = self.T(f"seed 0, step {n}, lower is better", 22, MUTED).next_to(fin, UP, buff=0.35)
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
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), FadeIn(ev), FadeIn(brk), FadeIn(legend), run_time=0.6)
            self.mark("legend", legend, "blue = baseline (with positions), pink = no position embeddings")
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

        dots_b, dots_n = VGroup(), VGroup()
        for k, (vb, vn) in enumerate(zip(VB, VN)):
            dots_b.add(Dot(nl.n2p(vb) + UP * (0.55 + 0.32 * k), radius=0.11, color=BLUE))
            dots_n.add(Dot(nl.n2p(vn) + UP * (0.55 + 0.32 * k), radius=0.11, color=PINK))
        sl = VGroup(*[self.M(f"seed {sd}", 20, MUTED).next_to(nl.n2p(0.62) + UP * (0.55 + 0.32 * k), LEFT, buff=0.2)
                      for k, sd in enumerate(SEEDS)])
        lb = self.T("baseline", 26, BLUE, BOLD).next_to(dots_b, UP, buff=0.25)
        ln = self.T("no positions", 26, PINK, BOLD).next_to(dots_n, UP, buff=0.25)
        mg, sdg = float(np.mean(GAPS)), float(np.std(GAPS, ddof=1))
        gl = VGroup(self.T("paired gap per seed (same seed, same batches): " + ", ".join(f"+{g:.3f}" for g in GAPS), 28, YELLOW),
                    self.T(f"mean +{mg:.3f} · seed-to-seed sd {sdg:.3f} · every seed agrees", 28, YELLOW, BOLD)
                    ).arrange(DOWN, buff=0.12).to_edge(UP, buff=1.0)
        self.fitw(gl, 12.6)
        note = self.T(f"measurement noise per number ≈ ±{se:.3f} (SE over the {R['n_eval_batches']} eval batches)", 22, MUTED)
        note.to_edge(DOWN, buff=0.4)
        with self.say('But one seed could be luck. So we ran two more seeds per arm, each with its own initial weights and batch order. Both numbers move, yet every seed agrees: no positions is worse, by 0.03 to 0.04. That gap beats the seed-to-seed spread.',
                      scene=f"Number line of val loss at step {n} for seeds {', '.join(SEEDS)}: baseline "
                            f"{', '.join(f'{v:.4f}' for v in VB)}; no positions {', '.join(f'{v:.4f}' for v in VN)}. Paired gaps "
                            f"{', '.join(f'{g:+.4f}' for g in GAPS)}, mean {mg:+.4f}, sd {sdg:.4f}. Each seed changes the "
                            f"initial weights and the batch order; eval batches are fixed.",
                      chapter="Seeds"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax, xl, yl, xt, yt, cb, cn, fin, fin_t, ev, brk, legend)), ReplacementTransform(hdr3, hdr4), run_time=0.5)
            self.play(Create(nl), FadeIn(nlab), FadeIn(nt), FadeIn(note), run_time=0.5)
            self.play(FadeIn(dots_b[0]), FadeIn(dots_n[0]), FadeIn(sl[0]), FadeIn(lb), FadeIn(ln), run_time=0.5)
            self.sync("So we ran", 0.1)
            for k in range(1, len(SEEDS)):
                self.play(FadeIn(dots_b[k], scale=1.5), FadeIn(dots_n[k], scale=1.5), FadeIn(sl[k]), run_time=0.5)
            self.mark("dots-base", VGroup(dots_b, lb), "baseline val loss per seed: " + ", ".join(f"{v:.4f}" for v in VB))
            self.mark("dots-nopos", VGroup(dots_n, ln), "no-positions val loss per seed: " + ", ".join(f"{v:.4f}" for v in VN))
            self.sync("yet every seed", 0.1)
            self.play(FadeIn(gl, shift=DOWN * 0.2), run_time=0.6)
            self.mark("gaps", gl, f"paired gaps {', '.join(f'{g:+.4f}' for g in GAPS)}; mean {mg:+.4f}, sd {sdg:.4f}")

        # ============================================================ traps
        hdr5 = self.header("Traps", PINK)
        fake = NOPOS["val"][-1] - REL["val"]
        cards = VGroup()
        specs = [
            ("different compute", ["vs the released model", f"(3,000 steps, {REL['val']:.3f}):", f"the gap looks {fake:.2f},",
                                   f"{fake / gap:.0f}× the real {gap:.3f}"], ORANGE),
            ("one seed", ["one lucky run", "can fake a win:", "look at the spread"], YELLOW),
            ("data leakage", [f"{LEAK['frac'] * 100:.0f}% of held-out lines", f"({LEAK['in_train']:,} of {LEAK['val_docs']:,}) appear in",
                              f"training: all {SHORT['in_train']:,} short", "templated ones, but only",
                              f"{100 * TRADE['in_train'] / TRADE['n']:.1f}% of trade stories"], PINK)]
        for title, lines, col in specs:
            t = VGroup(self.T(title, 30, col, BOLD), *[self.T(l, 26, FG) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            b = RoundedRectangle(corner_radius=0.16, width=4.25, height=3.8, stroke_color=col, stroke_width=2.5,
                                 fill_color=col, fill_opacity=0.07)
            self.fitw(t, 3.85)
            t.move_to(b).align_to(b, UP).align_to(b, LEFT).shift(DOWN * 0.3 + RIGHT * 0.22)
            cards.add(VGroup(b, t))
        cards.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.1)
        with self.say(f"Three classic traps. Different compute: compare against the released model, trained five times "
                      f"longer, and the gap looks {fake / gap:.0f} times bigger than it is. One seed: luck can fake a win. "
                      f"And leakage: {LEAK['frac'] * 100:.0f} percent of TinyGPT's held-out lines also appear in training, "
                      f"mostly short templated ones. So part of its validation loss measures memory.",
                      scene=f"Three trap cards. Different compute: comparing the 600-step no-positions run ({NOPOS['val'][-1]:.3f}) "
                            f"to the released 3,000-step model ({REL['val']:.3f}, same eval batches) suggests a {fake:.3f} gap, "
                            f"{fake / gap:.1f}× the real {gap:.3f}. One seed: luck can fake a win. Data leakage: "
                            f"{LEAK['in_train']:,} of {LEAK['val_docs']:,} held-out lines ({LEAK['frac'] * 100:.1f}%) appear "
                            f"verbatim in the training split: all {SHORT['n']:,} short templated lines (Q&A, counting, pets, "
                            f"days, facts) but only {TRADE['in_train']} of {TRADE['n']} trade stories.", chapter="Traps"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(nl, nlab, nt, db, dn, lb, ln, gl, note)), ReplacementTransform(hdr4, hdr5),
                      run_time=0.5)
            for k, ph in enumerate(["Different compute", "One seed", "And leakage"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(cards[k], shift=UP * 0.2), run_time=0.5)
            self.mark("trap-compute", cards[0], f"unequal compute: released model val {REL['val']:.4f} (3,000 steps) vs "
                                                f"no-pos {NOPOS['val'][-1]:.4f} (600 steps) → misleading gap {fake:.3f}")
            self.mark("trap-seed", cards[1], "single-seed comparisons can be noise")
            self.mark("trap-leak", cards[2], f"{LEAK['in_train']:,}/{LEAK['val_docs']:,} validation lines also occur in the "
                                             f"training split; trade stories: {TRADE['in_train']}/{TRADE['n']}")

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
