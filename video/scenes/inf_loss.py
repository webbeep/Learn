import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from vscene import *  # noqa
from lib_inference import (VOCAB, TRAIN, GROUPS, tid, tok, token_losses, corpus_chunks, tokhue)

VIDEO = dict(id="loss", cls="Loss", title="Loss: the one number training cares about",
             subtitle="Next-token prediction, cross-entropy with real probabilities, perplexity, and why averaging works",
             relates="training", track="training", order=20, poster=("Every guess, scored", 0, 0.95))

STORY = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the red ball."
STEPS, LOSS, VAL = TRAIN["steps"], TRAIN["train"], TRAIN["val"]


def seq_rgb(v, vmax):
    a = np.clip(v / vmax, 0, 1)[..., None] ** 0.7
    bg = np.array(color_to_rgb(BG)); hi = np.array(color_to_rgb(PINK))
    return (255 * (bg + (hi - bg) * a)).astype(np.uint8)


class Loss(VScene):
    VIDEO_ID = "loss"

    def construct(self):
        TL = token_losses(STORY)                       # (cur, next, p, loss) for every position
        n_q = len(TL)
        mean_story = float(np.mean([l for *_, l in TL]))
        ids = tok(STORY)
        toks = [VOCAB[i] for i in ids]
        # the corpus writes "an orange" (vowel rule), so after " a" only the consonant-initial colors can follow
        after_a = [c for c in GROUPS["colors"] if c.strip()[0] not in "aeiou"]
        n_colors = len(after_a)
        vowel_cols = [c.strip() for c in GROUPS["colors"] if c.strip()[0] in "aeiou"]

        # ------------------------------------------------------------ the training signal
        s5 = toks[:6]
        inp = VGroup(*[self.chip(t, tokhue(t), 32) for k, t in enumerate(s5[:-1])]).arrange(RIGHT, buff=0.55)
        tgt = VGroup(*[self.chip(t, tokhue(t), 32) for k, t in enumerate(s5[1:])])
        for a, b in zip(inp, tgt):
            b.next_to(a, DOWN, buff=1.1)
        VGroup(inp, tgt).move_to(UP * 0.3)
        hide = VGroup(*[RoundedRectangle(corner_radius=0.08, width=b.width + 0.1, height=b.height + 0.1, fill_color=PANEL,
                                         fill_opacity=1, stroke_color=MUTED, stroke_width=2).move_to(b) for b in tgt])
        qm = VGroup(*[self.T("?", 30, MUTED, BOLD).move_to(h) for h in hide])
        arrows = VGroup(*[Arrow(a.get_bottom(), b.get_top(), buff=0.1, color=MUTED, stroke_width=3) for a, b in zip(inp, tgt)])
        li = self.T("sees", 26, MUTED).next_to(inp, LEFT, buff=0.45)
        lt = self.T("guesses", 26, MUTED).next_to(tgt, LEFT, buff=0.45)
        note = self.T(f"our {len(ids)}-token story = {n_q} free quiz questions", 30, YELLOW).move_to(DOWN * 2.6)
        with self.say("How do you learn from raw text with no labels? Hide the next token and make the model guess. "
                      "Every position becomes a free quiz question.",
                      scene=f"Tokens of 'mia has a red ball.' on top; under each, the next token is hidden behind a '?' card, then "
                            f"revealed. Caption: our {len(ids)}-token story = {n_q} free quiz questions.",
                      chapter="The training signal"):
            self.play(FadeIn(li), LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in inp], lag_ratio=0.1), run_time=0.8)
            self.sync("Hide", 0.1)
            self.play(FadeIn(lt), LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(h), FadeIn(q)) for a, h, q in zip(arrows, hide, qm)],
                                              lag_ratio=0.12), run_time=1.0)
            self.sync("make the model guess", 0.1)
            self.add(tgt)
            self.bring_to_front(hide, qm)
            self.play(LaggedStart(*[AnimationGroup(FadeOut(h), FadeOut(q)) for h, q in zip(hide, qm)], lag_ratio=0.12), run_time=0.8)
            for k, (a, b) in enumerate(zip(s5[:-1], s5[1:])):
                self.mark(f"pair-{k}", VGroup(inp[k], tgt[k]), f'after "{a}" the correct next token is "{b}"')
            self.sync("Every position", 0.1)
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.5)
            self.mark("note", note, f"a {len(ids)}-token text gives {n_q} next-token predictions to grade")

        # ------------------------------------------------------------ cross-entropy
        hdr = self.header("1 · Cross-entropy")
        form = VGroup(self.T("loss =", 40, FG), self.M("−ln( p[right token] )", 40, YELLOW)).arrange(RIGHT, buff=0.3).move_to(UP * 2.6)
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 5, 1], x_length=6.4, y_length=3.9, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).move_to(LEFT * 2.6 + DOWN * 0.75)
        xl = VGroup(*[self.M(f"{v:g}", 24, MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in (0, 0.5, 1)])
        yl = VGroup(*[self.M(str(v), 24, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0, 2, 4)])
        xt = self.T("probability it gave the right token", 24, MUTED).next_to(xl, DOWN, buff=0.12)
        curve = ax.plot(lambda p: -math.log(p), x_range=[math.exp(-5), 1, 0.002], color=YELLOW, stroke_width=4)
        # real examples from the story
        pick = {}
        for k, (a, b, p, l) in enumerate(TL):
            pick.setdefault((a, b), (k, p, l))
        ex = [(" now", " tom"), (" the", " red"), (" a", " red"), (".", " tom")]
        exs = []
        for a, b in ex:
            cands = [(k, p, l) for k, (aa, bb, p, l) in enumerate(TL) if aa == a and bb == b]
            k, p, l = cands[-1] if a == " the" else cands[0]
            exs.append((a, b, p, l, k))
        cols = [TEAL, PINK, ORANGE, BLUE]
        dots, labs, badges = VGroup(), VGroup(), VGroup()

        def badge(n, c):
            b = VGroup(Circle(radius=0.17, color=c, stroke_width=2.5).set_fill(BG, 1), self.T(str(n), 22, c, BOLD))
            b[1].move_to(b[0])
            return b
        for n_, ((a, b, p, l, k), c) in enumerate(zip(exs, cols), start=1):
            d = Dot(ax.c2p(p, l), radius=0.1, color=c)
            dots.add(d)
            bd = badge(n_, c).next_to(d, UR if n_ != 1 else UP, buff=0.06)
            badges.add(bd)
            labs.add(VGroup(badge(n_, c), VGroup(self.M(f'"{a.strip() or a}" → "{b.strip()}"', 26, c),
                                               self.M(f"p {p * 100:.1f}%  loss {l:.2f}", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
                            ).arrange(RIGHT, buff=0.25, aligned_edge=UP))
        labs.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.9 + DOWN * 0.6)
        with self.say("Each guess is scored with cross-entropy: minus the natural log of the probability it gave the right token. "
                      "Sure and right is nearly free. Unsure costs more. Badly wrong costs a lot.",
                      scene="Formula: loss = −ln(p[right token]) (natural log); its curve; four real guesses from the story placed on it: " +
                            "; ".join(f"'{a}' → '{b}' p = {p * 100:.2f}%, loss {l:.3f}" for a, b, p, l, _ in exs) + ".",
                      chapter="Cross-entropy"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(inp, tgt, arrows, li, lt, note)), FadeIn(hdr), run_time=0.5)
            self.play(FadeIn(form, shift=DOWN * 0.2), run_time=0.5)
            self.mark("formula", form, "cross-entropy loss for one prediction = −ln p(correct token)")
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), Create(curve), run_time=0.9)
            self.mark("curve", VGroup(ax, curve), "loss = −ln p: 0 at p = 1, grows without limit as p → 0")
            for k, ph in enumerate(["Sure and right", "", "Unsure", "Badly wrong"]):
                if ph:
                    self.sync(ph, 0.1)
                self.play(FadeIn(dots[k], scale=2), FadeIn(badges[k]), FadeIn(labs[k], shift=LEFT * 0.2), run_time=0.35)
                a, b, p, l, kk = exs[k]
                self.mark(f"ex-{k}", labs[k], f'real: after "{a}" the model gave "{b}" p = {p * 100:.2f}% → loss {l:.3f}')
                self.mark(f"exdot-{k}", VGroup(dots[k], badges[k]), f'point {k + 1} on the curve: p = {p * 100:.2f}%, loss {l:.3f}')

        # ------------------------------------------------------------ every guess on the story
        hdr2 = self.header("2 · Every guess, scored")
        n = len(TL)
        bw = 12.6 / n
        base = -1.4
        lmax = max(l for *_, l in TL)
        sc = 2.3 / max(3.2, lmax)
        bars, tl = VGroup(), VGroup()
        for k, (a, b, p, l) in enumerate(TL):
            x = -6.3 + bw * (k + 0.5)
            hgt = max(0.03, l * sc)
            col = PINK if l > 1 else TEAL
            bars.add(Rectangle(width=bw * 0.7, height=hgt, fill_color=col, fill_opacity=0.85, stroke_width=0).move_to([x, base + hgt / 2, 0]))
            t = self.M(self.show_tok(b), 24, FG if l > 1 else MUTED, t2c={"␣": MUTED}).rotate(PI / 2)
            t.move_to([x, base - 0.15 - t.height / 2, 0])
            tl.add(t)
        axl = Line([-6.35, base, 0], [6.35, base, 0], color=LINE, stroke_width=2)
        ttl = self.T("loss on each next token of the story (real)", 26, MUTED).move_to(UP * 3.0)
        hard = [(k, a, b, p, l) for k, (a, b, p, l) in enumerate(TL) if l > 1]
        easy = sum(1 for *_, l in TL if l < 0.05)
        with self.say(f"Here's every guess on our story. {easy} of the {n} are nearly free. The spikes are the real coin flips: "
                      f"which color Mia has, which toy, who comes next.",
                      scene=f"Bar chart of the real loss for each of the {n} next-token predictions in the story (labels = the "
                            f"token being predicted). {easy} cost under 0.05; the spikes (pink) are " +
                            ", ".join(f"'{b}' ({l:.2f})" for _, a, b, p, l in hard) + ".",
                      chapter="Every guess, scored"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(form, ax, xl, yl, xt, curve, dots, labs, badges)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(ttl), Create(axl), FadeIn(tl, lag_ratio=0.02), run_time=0.8)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.04), run_time=1.2)
            for k, (a, b, p, l) in enumerate(TL):
                self.mark(f"bar-{k}", VGroup(bars[k], tl[k]), f'predicting "{b}" after "{a}": p = {p * 100:.2f}%, loss {l:.3f}')
            self.sync("The spikes", 0.1)
            self.play(*[Indicate(bars[k], color=PINK, scale_factor=1.15) for k, *_ in hard], run_time=0.8)

        # the coin flip vs the predictable one
        k_col = next(k for k, a, b, p, l in hard if b == " red")
        _, _, p_col, l_col = TL[k_col]
        k_end = max(k for k, (a, b, p, l) in enumerate(TL) if a == " the" and b == " red")
        _, _, p_end, l_end = TL[k_end]
        best = 1 / n_colors
        c1 = VGroup(self.M('"mia has a" → "red"', 28, PINK), self.M(f"p {p_col * 100:.1f}%  loss {l_col:.2f}", 28, FG),
                    self.T(f"{n_colors} colors can follow \"a\" ({', '.join(vowel_cols)} takes \"an\")", 26, MUTED),
                    self.T(f"best possible: 1/{n_colors} = {best * 100:.1f}% · the model: {p_col * 100:.1f}%", 26, MUTED)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        c2 = VGroup(self.M('"now tom has the" → "red"', 28, TEAL), self.M(f"p {p_end * 100:.1f}%  loss {l_end:.2f}", 28, FG),
                    self.T("predictable from the story", 26, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        cc = VGroup(c1, c2).arrange(RIGHT, buff=1.0, aligned_edge=UP).move_to(UP * 2.1)
        avg = VGroup(self.T(f"average of all {n}:", 30, FG), self.M(f"{mean_story:.2f}", 36, YELLOW, weight=BOLD)).arrange(RIGHT, buff=0.3)
        with self.say(f"The first red is a pure guess. Only {n_colors} colors can follow 'a', since {', '.join(vowel_cols)} takes 'an', "
                      f"so the best possible is {best * 100:.1f} percent. The model gives {p_col * 100:.1f}: essentially perfect. "
                      f"The last red follows from the story: {p_end * 100:.0f} percent. Averaged, the story's loss is {mean_story:.2f}.",
                      scene=f"Two callouts: 'mia has a' → 'red' p = {p_col * 100:.2f}% (loss {l_col:.3f}); {n_colors} colors can follow "
                            f"'a' ({', '.join(vowel_cols)} takes 'an'), so the best possible is 1/{n_colors} = {best * 100:.1f}% "
                            f"(loss {-math.log(best):.3f}): the model is essentially optimal. 'now tom has the' → "
                            f"'red' p = {p_end * 100:.1f}% (loss {l_end:.2f}). Average over all {n} predictions = {mean_story:.3f}."):
            self.play(FadeOut(ttl), run_time=0.3)
            hl1 = SurroundingRectangle(VGroup(bars[k_col], tl[k_col]), color=PINK, buff=0.06)
            self.play(Create(hl1), FadeIn(c1, shift=DOWN * 0.2), run_time=0.6)
            self.mark("coinflip", c1, f'"mia has a" → "red": p = {p_col * 100:.2f}% (loss {l_col:.3f}); {n_colors} colors can follow "a" ({", ".join(vowel_cols)} takes "an"), '
                                       f'so the best possible is {best * 100:.1f}% (loss {-math.log(best):.3f}): essentially optimal')
            self.sync("The last red", 0.1)
            hl2 = SurroundingRectangle(VGroup(bars[k_end], tl[k_end]), color=TEAL, buff=0.06)
            self.play(Create(hl2), FadeIn(c2, shift=DOWN * 0.2), run_time=0.6)
            self.mark("predictable", c2, f'"now tom has the" → "red": p = {p_end * 100:.2f}%, loss {l_end:.3f}')
            self.sync("Averaged", 0.1)
            avg.move_to(DOWN * 3.25 + RIGHT * 3.5)
            self.play(FadeIn(avg, scale=1.2), run_time=0.5)
            self.mark("avg", avg, f"mean cross-entropy over the story's {n} predictions = {mean_story:.4f}")

        # ------------------------------------------------------------ pure guessing
        hdr3 = self.header("3 · Pure guessing")
        ln256 = math.log(256)
        flat = VGroup(*[Rectangle(width=12.0 / 256 * 0.75, height=1.2, fill_color=BLUE, fill_opacity=0.75, stroke_width=0)
                        for _ in range(256)]).arrange(RIGHT, buff=12.0 / 256 * 0.25, aligned_edge=DOWN).move_to(UP * 1.0)
        flat_l = self.T("untrained: 1/256 = 0.39% on every token (illustration)", 26, MUTED).next_to(flat, UP, buff=0.3)
        g1 = VGroup(self.T("every guess costs", 30, FG), self.M(f"−ln(1/256) = ln 256 = {ln256:.2f}", 32, TEAL)).arrange(RIGHT, buff=0.3)
        g2 = VGroup(self.T("real loss at step 0:", 30, FG), self.M(f"{LOSS[0]:.2f}", 36, PINK, weight=BOLD)).arrange(RIGHT, buff=0.3)
        gg = VGroup(g1, g2).arrange(DOWN, buff=0.4).move_to(DOWN * 1.6)
        with self.say(f"Now rewind. With random weights, the model spreads its bets evenly over 256 tokens: a loss of "
                      f"{ln256:.2f} per guess. The real step-zero loss: {LOSS[0]:.2f}.",
                      scene=f"256 equal bars (illustration of a uniform guess, 0.39% each). Every guess costs ln 256 = {ln256:.3f}; "
                            f"the real step-0 training loss from training.json is {LOSS[0]:.4f}.",
                      chapter="Pure guessing"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(bars, tl, axl, c1, c2, cc, avg, hl1, hl2)), ReplacementTransform(hdr2, hdr3), run_time=0.5)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in flat], lag_ratio=0.003), FadeIn(flat_l), run_time=1.1)
            self.mark("flat", flat, "uniform guess over 256 tokens: 1/256 = 0.39% each (illustration)")
            self.sync("a loss of", 0.1)
            self.play(FadeIn(g1, shift=UP * 0.2), run_time=0.5)
            self.mark("ln256", g1, f"loss of a uniform guess = ln 256 = {ln256:.4f}")
            self.sync("The real step", 0.1)
            self.play(FadeIn(g2, shift=UP * 0.2), run_time=0.5)
            self.mark("step0", g2, f"real training loss at step 0 = {LOSS[0]:.4f}")

        # ------------------------------------------------------------ perplexity
        hdr4 = self.header("4 · Perplexity")
        ppl0, pplf = math.exp(LOSS[0]), math.exp(LOSS[-1])
        pf = VGroup(self.T("perplexity =", 40, FG), self.M("e^loss", 40, YELLOW)).arrange(RIGHT, buff=0.3).move_to(UP * 2.95)
        pf_s = self.T("≈ how many tokens it is effectively choosing between", 28, MUTED).next_to(pf, DOWN, buff=0.2)
        sq0 = VGroup(*[Square(0.2, stroke_width=0, fill_color=PINK, fill_opacity=0.8) for _ in range(round(ppl0))])
        sq0.arrange_in_grid(cols=20, buff=0.05).move_to(LEFT * 3.3 + DOWN * 1.2)
        l0 = VGroup(self.T("step 0", 30, PINK, BOLD), self.M(f"e^{LOSS[0]:.2f} ≈ {ppl0:.0f} choices", 28, PINK)).arrange(DOWN, buff=0.12).next_to(sq0, UP, buff=0.3)
        ghost = Rectangle(width=sq0.width, height=sq0.height, stroke_color=LINE, stroke_width=2).move_to(RIGHT * 3.3 + DOWN * 1.2)
        sqf = VGroup(Square(0.2, stroke_width=0, fill_color=TEAL, fill_opacity=0.95),
                     Rectangle(width=0.2 * (pplf - 1), height=0.2, stroke_width=0, fill_color=TEAL, fill_opacity=0.5)
                     ).arrange(RIGHT, buff=0.05, aligned_edge=LEFT)
        sqf.move_to(ghost.get_corner(UL) + sqf.get_center() - sqf.get_corner(UL))
        ghost_l = self.T("same scale: 1 square = 1 choice (an average)", 24, MUTED).next_to(ghost, DOWN, buff=0.15)
        lf = VGroup(self.T("step 3000", 30, TEAL, BOLD), self.M(f"e^{LOSS[-1]:.2f} ≈ {pplf:.1f} choices", 28, TEAL)).arrange(DOWN, buff=0.12)
        lf.move_to([ghost.get_x(), l0.get_y(), 0])
        with self.say(f"Take e to the loss, and you get perplexity: roughly how many tokens it's choosing between. At step zero, "
                      f"about {ppl0:.0f}. After training, {pplf:.1f} on average: near one for most tokens, more for a fresh color or name.",
                      scene=f"Perplexity = e^loss ≈ the effective number of choices. Step 0: e^{LOSS[0]:.2f} ≈ {ppl0:.0f} squares "
                            f"(≈ the 256-token vocabulary). Step 3000: e^{LOSS[-1]:.2f} ≈ {pplf:.2f}: one full square plus a partial one; an average: ~1 for most tokens, several for a fresh color or name.",
                      chapter="Perplexity"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(flat, flat_l, gg)), ReplacementTransform(hdr3, hdr4), run_time=0.5)
            self.play(FadeIn(pf), FadeIn(pf_s), run_time=0.5)
            self.mark("ppl", VGroup(pf, pf_s), "perplexity = e^(cross-entropy loss) ≈ effective number of equally likely choices")
            self.sync("At step zero", 0.1)
            self.play(FadeIn(l0), LaggedStart(*[FadeIn(s) for s in sq0], lag_ratio=0.003), run_time=1.0)
            self.mark("ppl0", VGroup(l0, sq0), f"step 0: e^{LOSS[0]:.4f} = {ppl0:.1f} effective choices")
            self.sync("After training", 0.1)
            self.play(FadeIn(lf), Create(ghost), FadeIn(ghost_l), FadeIn(sqf, scale=0.5), run_time=0.6)
            self.mark("pplf", VGroup(lf, sqf, ghost), f"step 3000: e^{LOSS[-1]:.4f} = {pplf:.2f} effective choices")

        # ------------------------------------------------------------ the real curve
        hdr5 = self.header("5 · The real curve")
        ax3 = Axes(x_range=[0, 3000, 500], y_range=[0, 6, 1], x_length=10.2, y_length=4.9, tips=False,
                   axis_config={"color": MUTED, "stroke_width": 2}).move_to(DOWN * 0.35 + RIGHT * 0.2)
        xl3 = VGroup(*[self.M(f"{v}", 24, MUTED).next_to(ax3.c2p(v, 0), DOWN, buff=0.15) for v in (0, 1000, 2000, 3000)])
        yl3 = VGroup(*[self.M(f"{v}", 24, MUTED).next_to(ax3.c2p(0, v), LEFT, buff=0.15) for v in (0, 2, 4, 6)])
        xt3 = self.T("training step", 24, MUTED).next_to(xl3, DOWN, buff=0.1)
        yt3 = self.T("loss", 24, MUTED).next_to(ax3, UP, buff=0.15).align_to(ax3, LEFT)
        trc = VMobject(color=YELLOW, stroke_width=4).set_points_as_corners([ax3.c2p(s, l) for s, l in zip(STEPS, LOSS)])
        vac = DashedVMobject(VMobject(color=TEAL, stroke_width=2.5).set_points_as_corners(
            [ax3.c2p(s_, l_) for s_, l_ in zip(STEPS, VAL)]), num_dashes=110)
        leg = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.5, color=YELLOW, stroke_width=4), self.T("train", 24, YELLOW)).arrange(RIGHT, buff=0.15),
                     VGroup(DashedLine(ORIGIN, RIGHT * 0.5, color=TEAL, stroke_width=2.5, dash_length=0.08),
                            self.T("validation (unseen text)", 24, TEAL)).arrange(RIGHT, buff=0.15)).arrange(RIGHT, buff=0.5)
        chance = DashedLine(ax3.c2p(0, ln256), ax3.c2p(3000, ln256), color=PINK, stroke_width=2, dash_length=0.12)
        chance_l = self.T(f"pure guessing {ln256:.2f}", 24, PINK).next_to(ax3.c2p(3000, ln256), DOWN, buff=0.12).align_to(ax3.c2p(3000, 0), RIGHT)
        i_below1 = next(i for i, l in enumerate(LOSS) if l < 1)
        a1 = VGroup(Dot(ax3.c2p(STEPS[i_below1], LOSS[i_below1]), color=ORANGE, radius=0.08),
                    self.T(f"below 1 by step {STEPS[i_below1]} (perplexity {math.exp(LOSS[i_below1]):.1f})", 24, ORANGE))
        a1[1].next_to(a1[0], RIGHT, buff=0.25).shift(UP * 1.35)
        a2 = VGroup(Dot(ax3.c2p(3000, LOSS[-1]), color=YELLOW, radius=0.08),
                    self.T(f"step 3000: {LOSS[-1]:.2f} · validation {VAL[-1]:.2f}", 24, YELLOW))
        a2[1].next_to(a2[0], UP, buff=0.45).shift(LEFT * 1.6)
        a0 = VGroup(Dot(ax3.c2p(0, LOSS[0]), color=PINK, radius=0.08), self.T(f"{LOSS[0]:.2f}", 24, PINK))
        a0[1].next_to(a0[0], RIGHT, buff=0.15).shift(UP * 0.2)
        leg.next_to(ax3, UP, buff=0.15).align_to(ax3, RIGHT)
        with self.say(f"Here's the real run: {LOSS[0]:.2f} at the start, below one by step {STEPS[i_below1]}, then {LOSS[-1]:.2f}. "
                      f"On unseen text, also {VAL[-1]:.2f}: it learned patterns, not pages. Some of what's left is coin flips "
                      f"no model can beat.",
                      scene=f"The real training-loss curve from training.json ({len(STEPS)} points): {LOSS[0]:.2f} at step 0 on the "
                            f"pure-guessing line (ln 256 = {ln256:.2f}), {LOSS[i_below1]:.2f} at step {STEPS[i_below1]}, "
                            f"{LOSS[-1]:.2f} at step 3000; validation loss on held-out text (dashed teal) ends at {VAL[-1]:.2f}. "
                            "Part of the remaining loss is irreducible (random colors, names).",
                      chapter="The real curve"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pf, pf_s, sq0, l0, sqf, lf, ghost, ghost_l)), ReplacementTransform(hdr4, hdr5), run_time=0.5)
            self.play(Create(ax3), FadeIn(xl3), FadeIn(yl3), FadeIn(xt3), FadeIn(yt3), Create(chance), FadeIn(chance_l), run_time=0.7)
            self.play(FadeIn(a0), run_time=0.3)
            self.play(Create(trc), Create(vac), FadeIn(leg), run_time=2.2, rate_func=linear)
            self.mark("val", vac, f"validation loss on held-out text: {VAL[0]:.3f} → {VAL[-1]:.3f}")
            self.mark("curve", trc, f"real training loss: {LOSS[0]:.3f} at step 0 → {LOSS[-1]:.3f} at step 3000 (logged every 25 steps)")
            self.mark("chance", VGroup(chance, chance_l), f"uniform-guess loss ln 256 = {ln256:.3f}")
            self.mark("start", a0, f"step 0 loss {LOSS[0]:.4f}")
            self.sync("below one", 0.1)
            self.play(FadeIn(a1), run_time=0.4)
            self.mark("below1", a1, f"first logged loss below 1: step {STEPS[i_below1]}, loss {LOSS[i_below1]:.4f}")
            self.sync("then", 0.1)
            self.play(FadeIn(a2), run_time=0.4)
            self.mark("end", a2, f"step 3000: train loss {LOSS[-1]:.4f} (perplexity {pplf:.2f}), validation {VAL[-1]:.4f}")

        # ------------------------------------------------------------ why averaging works
        hdr6 = self.header("6 · Why averaging works")
        X, RC = corpus_chunks(3000, 123, T=65)
        L = F.cross_entropy(RC["logits"].reshape(-1, 256), X[:, 1:].reshape(-1), reduction="none").view(X.shape[0], -1)
        g = torch.Generator().manual_seed(0)
        batches = [torch.randint(X.shape[0], (TRAIN["batch"],), generator=g) for _ in range(6)]
        bmeans = [float(L[b].mean()) for b in batches]
        B0 = L[batches[0]].numpy()                       # (64, 64)
        Lf = L.flatten()
        f_easy = float((Lf < 0.1).float().mean()); f_hard = float((Lf > 1).float().mean())
        share = float(Lf[Lf > 1].sum() / Lf.sum())
        img = ImageMobject(seq_rgb(B0, 3.0))
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.stretch_to_fit_width(4.0).stretch_to_fit_height(4.0).move_to(LEFT * 4.5 + DOWN * 0.45)
        imf = SurroundingRectangle(img, buff=0.02, stroke_color=LINE, stroke_width=2)
        iml = self.T(f"one real batch: {B0.shape[0]} × {B0.shape[1]} tokens", 24, MUTED).next_to(img, UP, buff=0.15)
        iml2 = self.T("brighter = costlier guess", 24, MUTED).next_to(img, DOWN, buff=0.15)
        f1 = VGroup(self.M(f"{f_easy * 100:.0f}%", 34, TEAL, weight=BOLD), self.T("of tokens cost almost nothing (< 0.1)", 26, FG)).arrange(RIGHT, buff=0.25)
        f2 = VGroup(self.M(f"{f_hard * 100:.0f}%", 34, PINK, weight=BOLD), self.T(f"are hard (> 1), and carry {share * 100:.0f}% of the loss", 26, FG)).arrange(RIGHT, buff=0.25)
        f3 = VGroup(self.T(f"yet the average loss of {len(bmeans)} random batches:", 26, FG),
                    self.M("  ".join(f"{m:.2f}" for m in bmeans), 30, YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        ff = VGroup(f1, f2, f3).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        ff.move_to(DOWN * 0.45).align_to(np.array([-2.2, 0, 0]), LEFT)
        N_pred = B0.size
        with self.say(f"Each step averages {N_pred:,} guesses: {B0.shape[0]} snippets of {B0.shape[1]} tokens. Single tokens are wildly "
                      f"uneven: {f_easy * 100:.0f} percent are nearly free, while the hard {f_hard * 100:.0f} percent carry "
                      f"{share * 100:.0f} percent of the loss. Yet the batch average barely moves.",
                      scene=f"Heatmap of real per-token losses for one {B0.shape[0]}×{B0.shape[1]} batch of fresh tiny-world text "
                            f"(brighter = higher loss). Stats over {Lf.numel():,} fresh predictions: {f_easy * 100:.1f}% cost < 0.1; "
                            f"{f_hard * 100:.1f}% cost > 1 and carry {share * 100:.1f}% of the total. Six random batch averages: "
                            + ", ".join(f"{m:.3f}" for m in bmeans) + ".",
                      chapter="Why averaging works"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax3, xl3, yl3, xt3, yt3, trc, vac, leg, chance, chance_l, a0, a1, a2)),
                      ReplacementTransform(hdr5, hdr6), run_time=0.5)
            self.play(FadeIn(img), Create(imf), FadeIn(iml), FadeIn(iml2), run_time=0.8)
            self.mark("batch", Group(img, imf), f"per-token loss for one real batch of {N_pred:,} predictions; mean {bmeans[0]:.3f}")
            self.sync("Single tokens", 0.1)
            self.play(FadeIn(f1, shift=UP * 0.2), run_time=0.4)
            self.sync("while", 0.1)
            self.play(FadeIn(f2, shift=UP * 0.2), run_time=0.4)
            self.mark("easy", f1, f"{f_easy * 100:.1f}% of {Lf.numel():,} fresh predictions have loss < 0.1")
            self.mark("hard", f2, f"{f_hard * 100:.1f}% have loss > 1 and account for {share * 100:.1f}% of the total loss")
            self.sync("Yet the batch", 0.1)
            self.play(FadeIn(f3, shift=UP * 0.2), run_time=0.5)
            self.mark("bmeans", f3, "mean loss of six random 64×64 batches of fresh text: " + ", ".join(f"{m:.3f}" for m in bmeans))

        eq = self.T("gradient of the average = average of the gradients", 30, YELLOW, BOLD)
        eq.to_edge(DOWN, buff=0.85)
        # toy picture: many small per-token nudges, one averaged nudge (illustration)
        rng = np.random.default_rng(3)
        mean_dir = np.array([1.0, 0.35, 0])
        org = RIGHT * 1.2 + UP * 0.2
        smalls = VGroup(*[Arrow(org, org + 1.1 * (mean_dir + rng.normal(0, 0.75, 3) * np.array([1, 1, 0])), buff=0,
                                color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.2) for _ in range(12)])
        big = Arrow(org, org + 1.5 * mean_dir, buff=0, color=YELLOW, stroke_width=7, max_tip_length_to_length_ratio=0.2)
        toy_l = self.T("toy picture", 24, MUTED).next_to(smalls, UP, buff=0.15)
        vote = self.T("every token casts a small vote on each nudge", 26, FG).next_to(eq, DOWN, buff=0.15)
        with self.say("Here's why that works. The gradient of an average is just the average of the gradients.",
                      scene="Caption: gradient of the average = average of the gradients."):
            self.play(Group(img, imf, iml, iml2, ff).animate.shift(UP * 0.45 + LEFT * 0.0), run_time=0.4)
            self.play(FadeIn(eq, shift=UP * 0.2), run_time=0.6)
            self.mark("eq", eq, "linearity: ∇(mean loss) = mean of per-token ∇loss")
        with self.say("So every token casts a small vote on how to nudge each weight. The votes disagree, but their "
                      "average points somewhere steady.",
                      scene="Toy picture (illustration): twelve thin gray arrows (per-token gradient directions) scattered "
                            "around one thick yellow arrow (their average). Caption: every token casts a small vote on each nudge."):
            self.play(FadeOut(ff), run_time=0.3)
            self.play(LaggedStart(*[GrowArrow(a_) for a_ in smalls], lag_ratio=0.08), FadeIn(toy_l), FadeIn(vote), run_time=1.2)
            self.mark("toy-votes", VGroup(smalls, toy_l), "toy illustration: per-token gradient directions (not real values)")
            self.sync("their", 0.1)
            self.play(GrowArrow(big), smalls.animate.set_opacity(0.35), run_time=0.7)
            self.mark("toy-avg", big, "toy illustration: the averaged gradient direction")

        # ------------------------------------------------------------ recap
        steps = [("predict the next token", BLUE), ("score: −ln p", YELLOW), ("average everything", TEAL), ("nudge", PINK)]
        pills = VGroup(*[self.chip(t, c, 26, font=SANS, pad=0.16, fill=0.12) for t, c in steps]).arrange(RIGHT, buff=0.38)
        pills.move_to(UP * 0.5)
        arr = VGroup(*[Arrow(pills[i].get_right(), pills[i + 1].get_left(), buff=0.08, color=MUTED, stroke_width=3)
                       for i in range(3)])
        back = CurvedArrow(pills[-1].get_bottom() + DOWN * 0.1, pills[0].get_bottom() + DOWN * 0.1, angle=-PI / 5,
                           color=MUTED, stroke_width=3)
        cta = self.T("now watch the loss fall in the Training lab →", 42, YELLOW, BOLD).move_to(DOWN * 2.5)
        with self.say("Predict, score, average, nudge. That's the whole signal. Now watch the loss fall in the training lab.",
                      scene="Recap loop: predict the next token → score: −ln p → average everything → nudge, looping back. "
                            "CTA: watch the loss fall in the Training lab.", chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(img, imf, iml, iml2, eq, vote, smalls, big, toy_l, hdr6)), run_time=0.5)
            for k in range(4):
                self.play(FadeIn(pills[k], shift=UP * 0.2), *([GrowArrow(arr[k - 1])] if k else []), run_time=0.35)
            self.play(Create(back), run_time=0.4)
            for (t, _), p in zip(steps, pills):
                self.mark("loop-" + t.split()[0], p, f"training signal step: {t}")
            self.sync("Now watch", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "open the Training lab to watch the loss fall")
        self.wait(0.6)
