import os, sys, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vscene import *  # noqa
from lib_research import load, Plot, legend, pill, panel, fmt_n

VIDEO = dict(id="train-your-own", cls="TrainYourOwn", title="Train your own: data, batches, epochs & overfitting",
             subtitle="The whole training recipe, and what to watch, with real overfitting runs",
             relates="train", track="training", order=50, poster=("Overfitting", 0, 0.95))

OV = load("overfit")
RUNS = OV["runs"]
TINY, FULL = RUNS[0], RUNS[-1]
TR_TOK, VA_TOK, CHARS = OV["train_tokens"], OV["val_tokens"], OV["corpus_chars"]
B, T, STEPS = OV["batch"], 64, OV["steps"]
PER_STEP = B * T
EPOCH_STEPS = TR_TOK / PER_STEP


class TrainYourOwn(VScene):
    VIDEO_ID = "train-your-own"

    def construct(self):
        # ------------------------------------------------------------ recipe
        items = [("data", BLUE), ("batches", TEAL), ("steps", ORANGE), ("watch the loss", YELLOW)]
        pills = VGroup(*[pill(t, c, 34) for t, c in items]).arrange(RIGHT, buff=0.7).move_to(UP * 0.3)
        arrows = VGroup(*[Arrow(pills[i].get_right(), pills[i + 1].get_left(), buff=0.1, color=MUTED, stroke_width=3)
                          for i in range(3)])
        title = self.T("training your own model", 52, FG, BOLD).move_to(UP * 2.4)
        with self.say("Training your own model sounds hard. The whole recipe fits on one screen: data, batches, steps, "
                      "and two numbers to watch.",
                      scene="Title 'training your own model' and four pills joined by arrows: data → batches → steps → "
                            "watch the loss.", chapter="The recipe"):
            self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)
            self.sync("data,", 0.1)
            for k in range(4):
                self.play(FadeIn(pills[k], shift=UP * 0.2), *([GrowArrow(arrows[k - 1])] if k else []), run_time=0.3)
            for (t, _), p in zip(items, pills):
                self.mark("recipe-" + t.split()[0], p, f"recipe step: {t}")

        # ------------------------------------------------------------ data
        hdr = self.header("Data")
        W = 12.0
        total = TR_TOK + VA_TOK
        wtr = W * TR_TOK / total
        bar_tr = Rectangle(width=wtr, height=0.7, fill_color=BLUE, fill_opacity=0.55, stroke_width=0)
        bar_va = Rectangle(width=W - wtr, height=0.7, fill_color=TEAL, fill_opacity=0.9, stroke_width=0)
        bar = VGroup(bar_tr, bar_va).arrange(RIGHT, buff=0).move_to(UP * 0.4)
        chars = VGroup(self.M(f"{CHARS:,}", 40, FG), self.T("characters of tiny-world stories", 30, MUTED)) \
            .arrange(RIGHT, buff=0.3).move_to(UP * 2.3)
        toks = VGroup(self.M(f"{total:,}", 40, YELLOW), self.T("tokens", 30, MUTED)).arrange(RIGHT, buff=0.3) \
            .next_to(bar, UP, buff=0.35)
        ltr = VGroup(self.T("train", 30, BLUE, BOLD), self.M(f"{TR_TOK:,} tokens (95%)", 26, BLUE)) \
            .arrange(DOWN, buff=0.1).next_to(bar_tr, DOWN, buff=0.35)
        lva = VGroup(self.T("validation", 30, TEAL, BOLD), self.M(f"{VA_TOK:,} (5%)", 26, TEAL)) \
            .arrange(DOWN, buff=0.1, aligned_edge=RIGHT).next_to(bar_va, DOWN, buff=0.35).align_to(bar, RIGHT)
        never = self.T("never trained on: the honest exam", 28, TEAL).next_to(lva, DOWN, buff=0.35).align_to(bar, RIGHT)
        with self.say(f"Start with text. Ours is {CHARS / 1e6:.1f} million characters, which the tokenizer turns into "
                      f"{total / 1e6:.2f} million tokens. Lock away the last five percent. The model never trains on it. "
                      f"That's your validation set.",
                      scene=f"A long bar of {total:,} tokens split into train ({TR_TOK:,}, blue, 95%) and validation "
                            f"({VA_TOK:,}, teal, 5%) from {CHARS:,} characters of corpus.", chapter="Data"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pills, arrows, title)), FadeIn(hdr), run_time=0.4)
            self.play(FadeIn(chars, shift=DOWN * 0.2), run_time=0.5)
            self.mark("chars", chars, f"corpus: {CHARS:,} characters (synthetic tiny-world stories)")
            self.sync("tokenizer", 0.1)
            self.play(GrowFromEdge(bar, LEFT), FadeIn(toks), run_time=0.8)
            self.mark("tokens", toks, f"{total:,} BPE tokens after tokenizing")
            self.sync("Lock away", 0.1)
            self.play(bar_va.animate.shift(RIGHT * 0.25), FadeIn(ltr), FadeIn(lva), run_time=0.6)
            self.mark("train-split", VGroup(bar_tr, ltr), f"training split: {TR_TOK:,} tokens")
            self.mark("val-split", VGroup(bar_va, lva), f"validation split: {VA_TOK:,} tokens, never trained on")
            self.sync("validation set", 0.1)
            self.play(FadeIn(never, shift=UP * 0.1), run_time=0.4)

        # ------------------------------------------------------------ batches
        hdr2 = self.header("Batches & steps")
        rng = random.Random(3)
        wins = VGroup()
        for k in range(B):
            x = rng.uniform(0, wtr - 0.12)
            wins.add(Rectangle(width=0.1, height=0.7, stroke_width=0, fill_color=YELLOW, fill_opacity=0.9)
                     .move_to(bar_tr.get_left() + RIGHT * (x + 0.05)))
        grid = VGroup(*[Square(0.075, stroke_width=0, fill_color=YELLOW, fill_opacity=0.85) for _ in range(B * T)])
        grid.arrange_in_grid(rows=B, cols=T, buff=0.012).move_to(LEFT * 2.7 + DOWN * 1.55)
        gl = self.T(f"{B} windows × {T} tokens = {PER_STEP:,} predictions", 26, YELLOW).next_to(grid, UP, buff=0.2)
        loop = VGroup(*[pill(t, c, 26) for t, c in [("average loss", PINK), ("gradient", ORANGE),
                                                     ("nudge every weight", TEAL)]]).arrange(DOWN, buff=0.45)
        loop.move_to(RIGHT * 3.9 + DOWN * 1.55)
        la = VGroup(*[Arrow(loop[i].get_bottom(), loop[i + 1].get_top(), buff=0.08, color=MUTED, stroke_width=3)
                      for i in range(2)])
        into = Arrow(grid.get_right(), loop[0].get_left(), buff=0.25, color=MUTED, stroke_width=3)
        one = self.T("= one step", 30, FG, BOLD).next_to(loop, DOWN, buff=0.35)
        with self.say(f"Each step grabs {B} random windows of {T} tokens, predicts the next token at every position, "
                      f"averages the loss, and nudges every weight once.",
                      scene=f"{B} yellow windows light up at random spots of the training bar and drop into a {B}×{T} grid "
                            f"({PER_STEP:,} predictions); then average loss → gradient → nudge every weight = one step.",
                      chapter="Batches & steps"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(chars, toks, ltr, lva, never)), ReplacementTransform(hdr, hdr2),
                      VGroup(bar).animate.move_to(UP * 2.0), run_time=0.5)
            wins.shift(UP * 1.6)
            self.play(LaggedStart(*[FadeIn(w) for w in wins], lag_ratio=0.03), run_time=0.8)
            self.mark("windows", wins, f"{B} random {T}-token windows from the training split")
            self.play(TransformFromCopy(wins, grid), FadeIn(gl), run_time=0.9)
            self.mark("batch", grid, f"one batch: {B} × {T} = {PER_STEP:,} next-token predictions")
            self.sync("averages", 0.1)
            self.play(GrowArrow(into), FadeIn(loop[0]), run_time=0.35)
            self.play(GrowArrow(la[0]), FadeIn(loop[1]), run_time=0.35)
            self.sync("nudges", 0.1)
            self.play(GrowArrow(la[1]), FadeIn(loop[2]), FadeIn(one), run_time=0.4)
            self.mark("step", VGroup(loop, one), "one training step: average loss → gradient → update every weight")

        # ------------------------------------------------------------ epoch
        eq = VGroup(self.M(f"{TR_TOK:,}", 40, BLUE), self.M("÷", 40, MUTED), self.M(f"{PER_STEP:,}", 40, YELLOW),
                    self.M("≈", 40, MUTED), self.M(f"{EPOCH_STEPS:.0f} steps", 40, FG)).arrange(RIGHT, buff=0.3)
        eq.move_to(DOWN * 0.2)
        eql = self.T("one epoch = one full pass over the training tokens", 32, FG).next_to(eq, UP, buff=0.6)
        ours = self.T(f"our runs: {STEPS:,} steps ≈ {STEPS / EPOCH_STEPS:.1f} epochs", 32, TEAL).next_to(eq, DOWN, buff=0.6)
        with self.say(f"An epoch is one full pass over the training data: about {EPOCH_STEPS:.0f} steps here. "
                      f"Our {STEPS:,}-step runs see each token about twice.",
                      scene=f"Equation: {TR_TOK:,} training tokens ÷ {PER_STEP:,} tokens per step ≈ {EPOCH_STEPS:.0f} "
                            f"steps per epoch; {STEPS:,} steps ≈ {STEPS / EPOCH_STEPS:.1f} epochs."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(bar, wins, grid, gl, loop, la, into, one)), run_time=0.4)
            self.play(FadeIn(eql), LaggedStart(*[FadeIn(e, shift=UP * 0.15) for e in eq], lag_ratio=0.15), run_time=0.9)
            self.mark("epoch", VGroup(eql, eq), f"epoch = {TR_TOK:,} / {PER_STEP:,} ≈ {EPOCH_STEPS:.1f} steps")
            self.sync("Our", 0.1)
            self.play(FadeIn(ours, shift=UP * 0.1), run_time=0.4)
            self.mark("ours", ours, f"{STEPS} steps ≈ {STEPS / EPOCH_STEPS:.2f} epochs")

        # ------------------------------------------------------------ train vs val (full data)
        hdr3 = self.header("Train vs validation")
        ymax = 3.0
        pl = Plot([0, STEPS], [0, ymax], width=10.5, height=4.6, xticks=[0, 500, 1000, 1500], yticks=[0, 1, 2, 3],
                  xlabel="training step", ylabel="loss").move_to(DOWN * 0.35)
        c = FULL["curve"]
        tr = pl.line(c["step"], c["train"], YELLOW, 4)
        va = pl.line(c["step"], c["val"], TEAL, 3, dashed=True)
        lg = legend([("train loss", YELLOW, False), ("validation loss", TEAL, True)]).next_to(pl.ax, UP, buff=0.15) \
            .align_to(pl.ax, RIGHT)
        gap = FULL["final_val"] - FULL["final_train"]
        endl = VGroup(self.M(f"train {FULL['final_train']:.3f}", 26, YELLOW), self.M(f"val   {FULL['final_val']:.3f}", 26, TEAL)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(pl.p(STEPS, 1.6), LEFT, buff=0.1)
        with self.say("Now watch two numbers. Training loss: how well it does on text it studies. Validation loss: "
                      "text it has never seen. With all the data, they move together. That's real learning.",
                      scene=f"Real curves, all {TR_TOK:,} training tokens, {STEPS} steps: train loss (yellow) and validation "
                            f"loss (dashed teal) fall together from {c['val'][0]:.2f} to {FULL['final_train']:.3f} / "
                            f"{FULL['final_val']:.3f} (gap {gap:+.3f}).", chapter="Train vs validation"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(eq, eql, ours)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(pl), FadeIn(lg), run_time=0.6)
            self.play(Create(tr), run_time=1.3, rate_func=linear)
            self.mark("full-train", tr, f"training loss with all data: {c['train'][0]:.2f} → {FULL['final_train']:.3f}")
            self.sync("Validation loss", 0.1)
            self.play(Create(va), run_time=1.3, rate_func=linear)
            self.mark("full-val", va, f"validation loss with all data: {c['val'][0]:.2f} → {FULL['final_val']:.3f}")
            self.play(FadeIn(endl), run_time=0.3)
            self.mark("full-end", endl, f"final: train {FULL['final_train']:.3f}, val {FULL['final_val']:.3f}, gap {gap:+.3f}")

        # ------------------------------------------------------------ overfitting
        hdr4 = self.header("Overfitting")
        ct = TINY["curve"]
        tr2 = pl.line(ct["step"], ct["train"], YELLOW, 4)
        va2 = pl.line(ct["step"], ct["val"], PINK, 4)
        lg2 = legend([("train loss", YELLOW, False), ("validation loss", PINK, False)]).move_to(lg).align_to(lg, RIGHT)
        bs, bv = TINY["best_val_step"], TINY["best_val"]
        best = VGroup(Dot(pl.p(bs, bv), color=TEAL, radius=0.09),
                      self.T(f"best val {bv:.2f} at step {bs}", 24, TEAL))
        best[1].move_to(pl.p(290, 0.78), aligned_edge=DL)
        tag = VGroup(self.T(f"only {TINY['data_tokens']:,} training tokens", 30, PINK, BOLD),
                     self.T(f"= {TINY['epochs']:.0f} epochs in {STEPS:,} steps", 26, MUTED)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(pl.p(820, 2.55)).align_to(pl.p(560, 0), LEFT)
        end2 = VGroup(self.M(f"train {TINY['final_train']:.2f}", 26, YELLOW), self.M(f"val   {TINY['final_val']:.2f}", 26, PINK)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(pl.p(STEPS, 1.0), LEFT, buff=0.35)
        with self.say(f"Now the same model and the same {STEPS:,} steps, but only {TINY['data_tokens']:,} tokens of training "
                      f"data. That's {TINY['epochs']:.0f} epochs. Training loss dives to {TINY['final_train']:.2f}. "
                      f"Validation loss bottoms out around step {bs}, then climbs.",
                      scene=f"Real overfitting run: {TINY['data_tokens']:,} training tokens ({TINY['epochs']:.0f} epochs). "
                            f"Train loss (yellow) → {TINY['final_train']:.3f}; validation loss (pink) is best "
                            f"({bv:.3f}) at step {bs} and ends at {TINY['final_val']:.3f}.", chapter="Overfitting"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(tr, va, endl)), ReplacementTransform(lg, lg2), ReplacementTransform(hdr3, hdr4),
                      run_time=0.4)
            self.play(FadeIn(tag, shift=DOWN * 0.1), run_time=0.4)
            self.mark("tiny-tag", tag, f"{TINY['data_tokens']:,} training tokens, {TINY['epochs']} epochs")
            self.sync("Training loss dives", 0.1)
            self.play(Create(tr2), run_time=1.2, rate_func=linear)
            self.mark("tiny-train", tr2, f"train loss on the tiny set: → {TINY['final_train']:.3f}")
            self.sync("Validation loss", 0.1)
            self.play(Create(va2), run_time=1.4, rate_func=linear)
            self.play(FadeIn(best, scale=1.3), FadeIn(end2), run_time=0.4)
            self.mark("tiny-val", va2, f"validation loss: best {bv:.3f} at step {bs}, final {TINY['final_val']:.3f}")
            self.mark("best", best, f"best validation loss {bv:.3f} at step {bs}: the place to stop")
            self.mark("tiny-end", end2, f"final train {TINY['final_train']:.3f} vs val {TINY['final_val']:.3f}")

        gapl = Arrow(pl.p(STEPS - 40, TINY["final_train"]), pl.p(STEPS - 40, TINY["final_val"]), buff=0.05,
                     color=PINK, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        with self.say("It memorized those few thousand tokens instead of learning the language. That widening gap is "
                      "overfitting.",
                      scene=f"A pink arrow spans the gap between train {TINY['final_train']:.2f} and validation "
                            f"{TINY['final_val']:.2f} at step {STEPS}: overfitting (memorizing, not generalizing)."):
            self.play(GrowArrow(gapl), run_time=0.5)
            self.mark("gap", gapl, f"generalization gap: {TINY['final_val'] - TINY['final_train']:.2f} nats")
            self.sync("overfitting", 0.1)
            self.flash("memorized ≠ learned", PINK, 56, hold=0.8, pos=UP * 0.9 + RIGHT * 1.5)

        # ------------------------------------------------------------ small multiples
        hdr5 = self.header("More data closes the gap")
        mini = VGroup()
        for r in RUNS:
            p = Plot([0, STEPS], [0, ymax], width=2.35, height=2.2, xticks=[0, STEPS], yticks=[0, 3], tick_size=20,
                     xlabel="", ylabel="", xfmt=lambda v: f"{v:g}")
            cc = r["curve"]
            l1 = p.line(cc["step"], cc["train"], YELLOW, 3)
            l2 = p.line(cc["step"], cc["val"], PINK, 3)
            ttl = self.T(r["label"], 26, FG, BOLD).next_to(p.ax, UP, buff=0.2)
            g = r["final_val"] - r["final_train"]
            gl_ = self.M(f"gap {g:.2f}", 26, PINK if g > 0.1 else TEAL).next_to(p.xl, DOWN, buff=0.2)
            mini.add(VGroup(p, l1, l2, ttl, gl_))
        mini.arrange(RIGHT, buff=0.85).move_to(DOWN * 0.2)
        lg3 = legend([("train", YELLOW, False), ("validation", PINK, False)]).to_edge(DOWN, buff=0.45)
        gaps = ", ".join(f"{r['final_val'] - r['final_train']:.2f}" for r in RUNS)
        with self.say(f"Give it more data and the gap closes. From {RUNS[0]['final_val'] - RUNS[0]['final_train']:.1f} "
                      f"with four thousand tokens, to almost nothing with all of it.",
                      scene=f"Four real runs side by side ({', '.join(r['label'] for r in RUNS)}), each with train (yellow) "
                            f"and validation (pink) curves; final gaps (val − train): {gaps}.",
                      chapter="More data closes the gap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pl, tr2, va2, best, end2, gapl, tag, lg2)), ReplacementTransform(hdr4, hdr5),
                      run_time=0.4)
            self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.2) for m in mini], lag_ratio=0.25), FadeIn(lg3), run_time=1.6)
            for r, m in zip(RUNS, mini):
                self.mark("mini-" + str(r["data_tokens"]), m,
                          f"{r['label']} ({r['epochs']} epochs): final train {r['final_train']:.3f}, "
                          f"val {r['final_val']:.3f}, gap {r['final_val'] - r['final_train']:.3f}")

        # ------------------------------------------------------------ checklist
        hdr6 = self.header("What to watch")
        chk = [
            (f"step 0 loss ≈ ln(vocab) = {math.log(256):.2f}", f"ours started at {FULL['curve']['val'][0]:.2f}. If not: bug.", BLUE),
            ("fast drop, then a slow grind", "flat from the start: learning rate too low, or a bug", TEAL),
            ("validation follows training", "a growing gap: more data, smaller model, or stop at best val", PINK),
            ("no spikes, no NaN", "spikes or NaN: learning rate too high (see Optimizers)", ORANGE),
        ]
        rows = VGroup()
        for a, b, col in chk:
            box = Square(0.42, stroke_color=col, stroke_width=3)
            tick = Text("✓", font=SANS, font_size=34, color=col).move_to(box)
            t1 = self.T(a, 32, FG, BOLD)
            t2 = self.T(b, 24, MUTED)
            txt = VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            rows.add(VGroup(VGroup(box, tick), txt).arrange(RIGHT, buff=0.4, aligned_edge=UP))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(DOWN * 0.1)
        with self.say("While yours trains, check four things. First loss near log of the vocab size. A fast drop, then "
                      "a slow grind. Validation following training. And no spikes.",
                      scene=f"Checklist: 1) step-0 loss ≈ ln(256) = {math.log(256):.2f} (ours {FULL['curve']['val'][0]:.2f}); "
                            "2) fast drop then slow grind; 3) validation follows training (gap → more data / smaller model / "
                            "early stop); 4) no spikes or NaN (else lower the learning rate).", chapter="What to watch"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(mini, lg3)), ReplacementTransform(hdr5, hdr6), run_time=0.4)
            for k, ph in enumerate(["First loss", "A fast drop", "Validation following", "no spikes"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(rows[k][0][0]), FadeIn(rows[k][1], shift=RIGHT * 0.2), run_time=0.35)
                self.play(Write(rows[k][0][1]), run_time=0.2)
                self.mark(f"check-{k}", rows[k], f"{chk[k][0]}: {chk[k][1]}")

        # ------------------------------------------------------------ cta
        cta = self.T("now train one in your browser →", 54, YELLOW, BOLD).move_to(UP * 0.3)
        sub = self.T("same recipe · real loss curves · your settings", 30, MUTED).next_to(cta, DOWN, buff=0.4)
        with self.say("That's the whole recipe. Now train one in your browser.",
                      scene="Final card: 'now train one in your browser →'.", chapter="Your turn"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows, hdr6)), run_time=0.4)
            self.play(FadeIn(cta, scale=1.2), FadeIn(sub), run_time=0.5)
            self.mark("cta", cta, "open the Train lab to train a model in your browser")
        self.wait(0.8)
