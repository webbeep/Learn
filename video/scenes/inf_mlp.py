import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from vscene import *  # noqa
from lib_inference import (VOCAB, W, PROBE, GROUPS, tid, tok, run, neuron_push, corpus_chunks)

VIDEO = dict(id="mlp", cls="MLP", title="The MLP: where tokens think alone",
             subtitle="Expand, GELU, project back: real neurons in TinyGPT that detect patterns and push predictions",
             relates="xray", track="inference", order=50, poster=("Neurons as detectors", 1, 0.9))

ROWS = [(0, 6), (6, 12), (12, 19), (19, 23)]
DET = (0, 1)          # layer index, neuron index: the "a color comes next" detector found by inspection
N_DOCS, SEED = 400, 7


def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class MLP(VScene):
    VIDEO_ID = "mlp"

    def construct(self):
        ids = tok(PROBE)
        toks = [VOCAB[i] for i in ids]
        R = run(ids)
        acts = [a[0] for a in R["acts"]]              # per layer (T, 256)
        RED = tid(" red")
        colors = set(tid(w) for w in GROUPS["colors"])

        # parameter counts per block
        d, m = 64, 256
        n_attn = sum(W[f"blocks.0.{k}"].numel() for k in ("attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias"))
        n_mlp = sum(W[f"blocks.0.{k}"].numel() for k in ("mlp_in.weight", "mlp_in.bias", "mlp_out.weight", "mlp_out.bias"))

        # ------------------------------------------------------------ hook
        def block_panel(hi_mlp=False):
            att = RoundedRectangle(corner_radius=0.14, width=3.2, height=1.0, fill_color=BLUE, fill_opacity=0.15,
                                   stroke_color=BLUE, stroke_width=2.5)
            mlp = RoundedRectangle(corner_radius=0.14, width=3.2, height=1.0, fill_color=TEAL, fill_opacity=0.3,
                                   stroke_color=TEAL, stroke_width=4)
            VGroup(att, mlp).arrange(DOWN, buff=0.35)
            at = self.T("attention", 32, BLUE).move_to(att)
            mt = self.T("MLP", 34, TEAL, BOLD).move_to(mlp)
            return VGroup(att, at), VGroup(mlp, mt)

        attb, mlpb = block_panel()
        panel = VGroup(attb, mlpb).move_to(LEFT * 3.2 + UP * 0.1)
        lab_a = self.T("tokens share information", 26, BLUE).next_to(attb, RIGHT, buff=0.4)
        lab_m = self.T("each token thinks alone", 26, TEAL).next_to(mlpb, RIGHT, buff=0.4)
        bars = VGroup()
        tot = n_attn + n_mlp
        for n, c, name in [(n_attn, BLUE, "attention"), (n_mlp, TEAL, "MLP")]:
            b = Rectangle(width=6.5 * n / tot, height=0.5, fill_color=c, fill_opacity=0.8, stroke_width=0)
            bars.add(b)
        bars.arrange(RIGHT, buff=0.04).move_to(DOWN * 2.4)
        bl = VGroup(self.M(f"{n_attn:,}", 26, FG).move_to(bars[0]), self.M(f"{n_mlp:,}", 26, BG).move_to(bars[1]))
        bt = self.T(f"weights in each layer: the MLP holds {n_mlp / tot * 100:.0f}%", 28, FG).next_to(bars, UP, buff=0.25)
        with self.say(f"Attention lets tokens share information. Then every token goes off alone, through a small network "
                      f"called the MLP. It holds two thirds of each layer's weights.",
                      scene=f"One transformer layer: an attention box ('tokens share information') above an MLP box ('each "
                            f"token thinks alone'). A bar splits each layer's weights: attention {n_attn:,} vs MLP {n_mlp:,} "
                            f"({n_mlp / tot * 100:.0f}%).", chapter="After attention"):
            self.play(FadeIn(attb, shift=DOWN * 0.2), FadeIn(lab_a), run_time=0.6)
            self.sync("Then every", 0.1)
            self.play(FadeIn(mlpb, shift=UP * 0.2), FadeIn(lab_m), run_time=0.6)
            self.mark("attn-box", VGroup(attb, lab_a), "attention: mixes information between token positions")
            self.mark("mlp-box", VGroup(mlpb, lab_m), "MLP: processes each token's vector on its own")
            self.sync("two thirds", 0.1)
            self.play(FadeIn(bt), GrowFromEdge(bars[0], LEFT), GrowFromEdge(bars[1], LEFT), FadeIn(bl), run_time=0.8)
            self.mark("split", VGroup(bars, bl, bt), f"per layer: attention {n_attn:,} weights, MLP {n_mlp:,} weights ({n_mlp / tot * 100:.1f}%)")

        # ------------------------------------------------------------ anatomy
        hdr = self.header("1 · Anatomy")
        xs = [-5.4, -2.3, 0.8]
        hgt = 5.4

        def column(n, x, col, r):
            return VGroup(*[Dot([x, hgt / 2 - hgt * (i + 0.5) / n - 0.3, 0], radius=r, color=col) for i in range(n)])
        c_in = column(64, xs[0], BLUE, 0.035)
        c_hid = column(128, xs[1], YELLOW, 0.026)      # 256 neurons drawn as 128 dots (2 per dot), labelled below
        c_out = column(64, xs[2], TEAL, 0.035)
        rng = np.random.default_rng(0)
        w1 = VGroup(*[Line(c_in[i].get_center(), c_hid[j].get_center(), stroke_width=0.6, stroke_opacity=0.25, color=BLUE)
                      for i, j in zip(rng.integers(0, 64, 260), rng.integers(0, 128, 260))])
        w2 = VGroup(*[Line(c_hid[j].get_center(), c_out[i].get_center(), stroke_width=0.6, stroke_opacity=0.25, color=TEAL)
                      for i, j in zip(rng.integers(0, 64, 260), rng.integers(0, 128, 260))])
        l_in = self.M("64", 30, BLUE).next_to(c_in, DOWN, buff=0.2)
        l_hid = self.M("256 neurons", 28, YELLOW).next_to(c_hid, DOWN, buff=0.2)
        l_out = self.M("64", 30, TEAL).next_to(c_out, DOWN, buff=0.2)
        t1 = self.T("expand", 28, BLUE).move_to([(xs[0] + xs[1]) / 2, 3.0, 0])
        t2 = self.T("project back", 28, TEAL).move_to([(xs[1] + xs[2]) / 2, 3.0, 0])
        ax = Axes(x_range=[-3, 3, 1], y_range=[-0.5, 3, 1], x_length=3.4, y_length=2.4, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).move_to(RIGHT * 4.6 + UP * 1.0)
        gcurve = ax.plot(gelu, x_range=[-3, 3, 0.02], color=YELLOW, stroke_width=4)
        gl = self.T("GELU", 30, YELLOW, BOLD).next_to(ax, UP, buff=0.15)
        gn = VGroup(self.T("negative in → ~0 out", 24, MUTED), self.T("positive in → passes", 24, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        gn.next_to(ax, DOWN, buff=0.25)
        plus = VGroup(self.T("+ add the result back", 28, FG), self.T("to the token's vector", 28, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        plus.next_to(gn, DOWN, buff=0.6).align_to(gn, LEFT)
        with self.say("Inside, it's three steps. Expand the token's 64 numbers into 256 neurons. Pass each one through GELU, "
                      "which lets positive signals through and squashes negative ones to nearly zero. "
                      "Then project back down to 64, and add the result to the token's vector.",
                      scene="Network diagram: 64 inputs (blue) fan out to 256 hidden neurons (yellow; drawn 2 per dot) then back "
                            "to 64 outputs (teal). Inset: the real GELU curve (negative in → ~0, positive in → passes). "
                            "The output is added back to the token's vector.", chapter="Anatomy"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(panel, lab_a, lab_m, bars, bl, bt)), FadeIn(hdr), run_time=0.5)
            self.play(FadeIn(c_in), FadeIn(l_in), run_time=0.4)
            self.sync("Expand", 0.1)
            self.play(Create(w1, lag_ratio=0.002), FadeIn(c_hid), FadeIn(l_hid), FadeIn(t1), run_time=1.0)
            self.mark("w-in", VGroup(w1, t1), f"W_in: 64 × 256 weights (+256 biases) expand the vector")
            self.mark("hidden", VGroup(c_hid, l_hid), "256 hidden neurons")
            self.sync("Pass each", 0.1)
            self.play(Create(ax), Create(gcurve), FadeIn(gl), run_time=0.8)
            self.play(FadeIn(gn, shift=UP * 0.1), run_time=0.4)
            self.mark("gelu", VGroup(ax, gcurve, gl, gn), f"GELU(x) ≈ x for large positive x, ≈ 0 for negative x; GELU(-1) = {gelu(-1):.3f}, GELU(1) = {gelu(1):.3f}")
            self.sync("Then project", 0.1)
            self.play(Create(w2, lag_ratio=0.002), FadeIn(c_out), FadeIn(l_out), FadeIn(t2), run_time=1.0)
            self.bring_to_front(c_hid)
            self.mark("w-out", VGroup(w2, t2), "W_out: 256 × 64 weights (+64 biases) project back to 64 numbers")
            self.play(FadeIn(plus, shift=LEFT * 0.2), run_time=0.4)

        # real activations, layer 3, last token
        a3 = acts[2][-1].numpy()
        frac = float((a3 > 0.1).mean())
        bw = 11.5 / 256
        base_y = -2.2
        abars = VGroup(*[Rectangle(width=bw * 0.8, height=max(0.01, abs(v) * 1.5), stroke_width=0,
                                   fill_color=YELLOW if v > 0.1 else MUTED, fill_opacity=0.9 if v > 0.1 else 0.5)
                         .move_to([-5.75 + bw * (i + 0.5), base_y + (abs(v) * 1.5 / 2) * (1 if v >= 0 else -1), 0])
                         for i, v in enumerate(a3)])
        axis = Line([-5.9, base_y, 0], [5.9, base_y, 0], color=LINE, stroke_width=2)
        atitle = self.T("layer 3's 256 neurons, on the final \"the\" of our story (real values)", 26, MUTED).move_to(UP * 2.75)
        story = self.M('"…now tom has the"', 26, FG).next_to(atitle, DOWN, buff=0.2)
        imax = int(a3.argmax())
        tag_max = self.M(f"#{imax}: {a3[imax]:.2f}", 26, YELLOW).next_to(abars[imax], UP, buff=0.12)
        stat = self.T(f"only {frac * 100:.0f}% fire above 0.1", 34, YELLOW, BOLD).move_to(DOWN * 3.05)
        with self.say(f"Here are the real 256 neuron values in layer 3, for the last word of our story. "
                      f"Most stay quiet. Only {frac * 100:.0f} percent fire above 0.1.",
                      scene=f"Bar chart of the 256 real post-GELU activations in layer 3 at the final ' the' of 'mia has a red "
                            f"ball … now tom has the'. Most are ≈0 (gray); {frac * 100:.0f}% exceed 0.1 (yellow); the largest is "
                            f"neuron #{imax} at {a3[imax]:.2f}."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(c_in, c_hid, c_out, w1, w2, l_in, l_hid, l_out, t1, t2, ax, gcurve, gl, gn, plus)),
                      run_time=0.5)
            self.play(FadeIn(atitle), FadeIn(story), Create(axis), run_time=0.5)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in abars], lag_ratio=0.004), run_time=1.3)
            self.play(FadeIn(tag_max), run_time=0.3)
            self.mark("acts", abars, f"256 real layer-3 activations at the last token; {frac * 100:.1f}% above 0.1, max {a3.max():.2f} (#{imax})")
            self.sync("Only", 0.1)
            self.play(FadeIn(stat, scale=1.2), run_time=0.4)
            self.mark("stat", stat, f"{(a3 > 0.1).sum()} of 256 layer-3 neurons above 0.1")

        # ------------------------------------------------------------ detector neuron
        hdr2 = self.header("2 · Neurons as detectors")
        L, N = DET
        a_det = acts[L][:, N].numpy()
        amax = float(a_det.max())
        chips = [self.chip(t, MUTED, 28) for t in toks]
        rows = VGroup(*[VGroup(*chips[a:b]).arrange(RIGHT, buff=0.14) for a, b in ROWS]).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        rows.move_to(DOWN * 0.45)
        vals = VGroup()
        for c, v in zip(chips, a_det):
            if v > 0.3:
                vals.add(self.M(f"{v:.1f}", 24, YELLOW).next_to(c, DOWN, buff=0.08))
        ntitle = VGroup(self.T(f"layer {L + 1} · neuron #{N}", 34, YELLOW, BOLD),
                        self.T("brightness = how strongly it fires on each token", 24, MUTED)).arrange(DOWN, buff=0.15).move_to(UP * 2.45)
        hits = [k for k, v in enumerate(a_det) if v > 0.8]
        with self.say(f"Each neuron's input weights are a pattern it's looking for. Take neuron {N} in layer {L + 1}. "
                      f"On our story, it fires on 'a', both times. Right before a color.",
                      scene=f"The 23 story tokens as chips; each chip glows yellow by how strongly layer {L + 1} neuron #{N} "
                            f"fires on it. It fires on both ' a' tokens ({', '.join(f'{a_det[k]:.2f}' for k in hits)}), "
                            f"right before ' red' and ' blue'; elsewhere it is near zero.",
                      chapter="Neurons as detectors"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(abars, axis, atitle, story, tag_max, stat)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(ntitle), LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in chips], lag_ratio=0.02), run_time=0.9)
            self.mark("neuron", ntitle, f"layer {L + 1}, neuron #{N} (of 256, counting from 0)")
            self.sync("fires on", 0.1)
            anims = []
            for c, v in zip(chips, a_det):
                t = float(np.clip(v / amax, 0, 1))
                if t > 0.05:
                    anims.append(c[0].animate.set_fill(YELLOW, opacity=0.08 + 0.6 * t).set_stroke(
                        interpolate_color(ManimColor(MUTED), ManimColor(YELLOW), t), width=2.2 + 2 * t))
            self.play(*anims, FadeIn(vals), run_time=0.8)
            for k, (c, t, v) in enumerate(zip(chips, toks, a_det)):
                self.mark(f"tok-{k}", c, f'"{t}" (position {k}): layer {L + 1} neuron #{N} activation {v:.3f}')
            self.sync("Right before", 0.1)
            nxt = VGroup(*[SurroundingRectangle(chips[k + 1], color=PINK, buff=0.06, corner_radius=0.08) for k in hits])
            self.play(Create(nxt), run_time=0.5)

        # corpus evidence + push
        X, RC = corpus_chunks(N_DOCS, SEED)
        Tn = X.shape[1]
        ca = RC["acts"][L][:, :-1, N].reshape(-1)          # exclude last position (its next token is unknown)
        top = torch.topk(ca, 200).indices.tolist()
        nxt_ids = [int(X[i // (Tn - 1), i % (Tn - 1) + 1]) for i in top]
        n_col = sum(j in colors for j in nxt_ids)
        exs, seen = [], set()
        for i in top:
            r_, p_ = i // (Tn - 1), i % (Tn - 1)
            back = []
            for j in range(p_ - 1, max(-1, p_ - 6), -1):
                if VOCAB[int(X[r_, j])] in (".", "\n", "?", "!", ","):
                    break
                back.insert(0, VOCAB[int(X[r_, j])])
            ctx = "".join(back).lstrip()
            cur = VOCAB[int(X[r_, p_])]
            nx = VOCAB[int(X[r_, p_ + 1])]
            if (ctx + cur) in seen:
                continue
            seen.add(ctx + cur)
            exs.append((ctx, cur, nx, float(ca[i])))
            if len(exs) == 4:
                break
        ex_rows = VGroup()
        for ctx, cur, nx, v in exs:
            mk = (f'<span foreground="{MUTED}">{ctx}</span><span foreground="{YELLOW}" weight="bold">{cur}</span>'
                  f'<span foreground="{MUTED}">  →</span><span foreground="{PINK}">{nx}</span>')
            ex_rows.add(MarkupText(mk, font=MONO, font_size=26))
        ex_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        ex_t = self.T(f"its strongest firings in {N_DOCS} fresh stories", 26, MUTED)
        ex_s = self.T(f"{n_col} / 200 come right before a color", 32, PINK, BOLD)
        ex_g = VGroup(ex_t, ex_rows, ex_s).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 2.6 + DOWN * 0.2)
        push, _ = neuron_push(L, N, 5)
        p_chips = VGroup(*[self.chip(t, PINK if tid(t) in colors else MUTED, 28) for t, _ in push]).arrange(DOWN, buff=0.18)
        p_t = VGroup(self.T("its output weights push", 26, MUTED), self.T("(roughly) toward:", 26, MUTED)).arrange(DOWN, buff=0.08)
        p_g = VGroup(p_t, p_chips).arrange(DOWN, buff=0.3).move_to(RIGHT * 4.3 + DOWN * 0.2)
        p_note = self.T("output weights × unembedding; ignores the final LayerNorm's scaling", 24, MUTED).to_edge(DOWN, buff=0.35)
        with self.say(f"In {N_DOCS} fresh stories, {n_col} of its 200 strongest firings came right before a color word. "
                      f"And its output weights push, roughly, toward " + ", ".join(t.strip() for t, _ in push[:4])
                      + f". A 'color comes next' detector, that nobody designed.",
                      scene=f"Left: real top contexts for layer {L + 1} neuron #{N} in {N_DOCS} freshly generated stories, e.g. "
                            + "; ".join(f"'…{c}[{u}] → {n_}'" for c, u, n_, _ in exs[:2])
                            + f"; {n_col}/200 of its strongest firings are followed by a color. Right: tokens its output weights "
                              f"push toward (projected through the unembedding, roughly): "
                            + ", ".join(f"{t.strip()}" for t, _ in push) + "."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows, vals, nxt, ntitle[1])), ntitle[0].animate.scale(0.85).to_corner(UR, buff=0.45), run_time=0.5)
            self.play(FadeIn(ex_t), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in ex_rows], lag_ratio=0.2), run_time=1.0)
            self.mark("examples", ex_rows, f"top-activating contexts of L{L + 1} #{N} in fresh text: " +
                      "; ".join(f"…{c}[{u}] → {n_} (act {v:.2f})" for c, u, n_, v in exs))
            self.play(FadeIn(ex_s, scale=1.1), run_time=0.4)
            self.mark("evidence", ex_s, f"{n_col} of the 200 strongest activations are followed by a color token")
            self.sync("And its output", 0.1)
            self.play(FadeIn(p_t), LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in p_chips], lag_ratio=0.15),
                      FadeIn(p_note), run_time=1.0)
            self.mark("push", VGroup(p_t, p_chips), f"L{L + 1} #{N} output direction · unembedding (roughly): " +
                      ", ".join(f'"{t}" {s:+.2f}' for t, s in push))
            self.mark("note", p_note, "approximation: W_out column → final-LN gain → unembedding; ignores LayerNorm's 1/std")

        # ------------------------------------------------------------ top neurons at the final 'the'
        hdr3 = self.header("3 · Layer 3 at the final \"the\"")
        topn = torch.topk(acts[2][-1], 5)
        n_rows = VGroup()
        tops = []
        for v, n in zip(topn.values.tolist(), topn.indices.tolist()):
            ps, _ = neuron_push(2, n, 4)
            tops.append((n, v, ps))
            name = self.M(f"#{n}", 30, YELLOW)
            bar = Rectangle(width=v * 1.3, height=0.42, fill_color=YELLOW, fill_opacity=0.8, stroke_width=0)
            val = self.M(f"{v:.2f}", 26, YELLOW)
            cs = VGroup(*[self.chip(t, PINK if tid(t) in colors else MUTED, 26) for t, _ in ps]).arrange(RIGHT, buff=0.14)
            n_rows.add(VGroup(name, bar, val, cs))
        for k, r in enumerate(n_rows):
            y = 1.6 - k * 0.92
            r[0].move_to([-5.6, y, 0], aligned_edge=LEFT)
            r[1].move_to([-4.5, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.2)
            r[3].move_to([0.6, y, 0], aligned_edge=LEFT)
        hd = VGroup(self.T("neuron", 24, MUTED).move_to([-5.6, 2.45, 0], aligned_edge=LEFT),
                    self.T("activation", 24, MUTED).move_to([-3.4, 2.45, 0]),
                    self.T("pushes (roughly) toward", 24, MUTED).move_to([0.6, 2.45, 0], aligned_edge=LEFT))
        n0, v0, p0 = tops[0]
        n1, v1, p1 = tops[1]
        with self.say(f"Now the five neurons that fire hardest in layer 3, on the final 'the'. "
                      f"Number {n0} pushes toward {p0[0][0].strip()}, {p0[1][0].strip()} and {p0[3][0].strip()}. "
                      f"Number {n1}, toward {p1[0][0].strip()} and word fragments. Most of them mix several jobs. "
                      f"Researchers call that polysemantic.",
                      scene="The five most active real layer-3 neurons at the final ' the', each with its activation bar and the "
                            "4 tokens its output weights push toward (roughly): "
                            + "; ".join(f"#{n} ({v:.2f}) → " + ", ".join(t.strip() or repr(t) for t, _ in ps) for n, v, ps in tops)
                            + ". Colors are highlighted pink; most neurons mix unrelated tokens (polysemantic).",
                      chapter="The final \"the\""):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ntitle[0], ex_g, p_g, p_note)), ReplacementTransform(hdr2, hdr3), run_time=0.5)
            self.play(FadeIn(hd), LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                                for r in n_rows], lag_ratio=0.12), run_time=1.0)
            for k, (n, v, ps) in enumerate(tops):
                self.mark(f"n{n}", n_rows[k][:3], f"layer 3 neuron #{n}: activation {v:.3f} at the final \" the\"")
            self.sync(f"Number {n0}", 0.1)
            self.play(FadeIn(n_rows[0][3], shift=LEFT * 0.2), run_time=0.4)
            self.sync(f"Number {n1}", 0.1)
            self.play(LaggedStart(*[FadeIn(r[3], shift=LEFT * 0.2) for r in n_rows[1:]], lag_ratio=0.2), run_time=0.9)
            for k, (n, v, ps) in enumerate(tops):
                self.mark(f"push-{n}", n_rows[k][3], f"#{n} pushes roughly toward " + ", ".join(f'"{t}" {s:+.2f}' for t, s in ps))
            self.sync("polysemantic", 0.3)
            self.flash("polysemantic", YELLOW, 52, hold=0.7, pos=DOWN * 2.9)

        # before / after the layer-3 MLP
        st = dict(R["stages"])
        from lib_inference import lens
        pb = lens(st["attn3"][0])[-1].softmax(-1)
        pa = lens(st["mlp3"][0])[-1].softmax(-1)
        jb = int(pb.argmax())
        rb, ra = float(pb[RED]), float(pa[RED])
        bx = VGroup()
        for name, p, col in [("just before layer 3's MLP", rb, MUTED), ("right after it", ra, PINK)]:
            lab = self.T(name, 28, FG)
            bar = Rectangle(width=max(0.04, 7.0 * p), height=0.6, fill_color=PINK, fill_opacity=0.85, stroke_width=0)
            v = self.M(f"{p * 100:.1f}%", 32, PINK)
            bx.add(VGroup(lab, bar, v))
        for k, r in enumerate(bx):
            y = 0.4 - k * 1.3
            r[0].move_to([-1.9, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-1.5, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.25)
        btitle = self.T('probability of " red" (read out with the logit lens)', 28, MUTED).move_to(UP * 1.9)
        bnote = self.T(f'before the MLP, the top guess was "{VOCAB[jb].strip() or VOCAB[jb]}" at {float(pb[jb]) * 100:.0f}%', 24, MUTED).move_to(DOWN * 2.4)
        with self.say(f"Add up all 256 pushes, and the answer appears. Just before this MLP, red had {rb * 100:.1f} percent. "
                      f"Right after it, {ra * 100:.0f}.",
                      scene=f"Two bars for P(' red') read out with the logit lens: {rb * 100:.1f}% just before layer 3's MLP vs "
                            f"{ra * 100:.1f}% right after it. Note: before the MLP the top guess was '{VOCAB[jb]}' at "
                            f"{float(pb[jb]) * 100:.1f}%."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(hd, n_rows)), run_time=0.4)
            self.play(FadeIn(btitle), FadeIn(bx[0][0]), GrowFromEdge(bx[0][1], LEFT), FadeIn(bx[0][2]), FadeIn(bnote), run_time=0.7)
            self.mark("before", bx[0], f'P(" red") just before layer-3 MLP = {rb * 100:.2f}%')
            self.mark("before-top", bnote, f'top guess before the layer-3 MLP: "{VOCAB[jb]}" {float(pb[jb]) * 100:.1f}%')
            self.sync("Right after", 0.1)
            self.play(FadeIn(bx[1][0]), GrowFromEdge(bx[1][1], LEFT), FadeIn(bx[1][2]), run_time=0.8)
            self.mark("after", bx[1], f'P(" red") right after layer-3 MLP = {ra * 100:.2f}% (final answer)')

        # ------------------------------------------------------------ memories
        hdr4 = self.header("4 · Where facts live")
        keyb = RoundedRectangle(corner_radius=0.18, width=4.4, height=2.2, fill_color=BLUE, fill_opacity=0.12, stroke_color=BLUE, stroke_width=2.5)
        valb = RoundedRectangle(corner_radius=0.18, width=4.4, height=2.2, fill_color=TEAL, fill_opacity=0.12, stroke_color=TEAL, stroke_width=2.5)
        VGroup(keyb, valb).arrange(RIGHT, buff=1.6).move_to(UP * 0.2)
        kt = VGroup(self.T("key", 34, BLUE, BOLD), self.T("input weights", 24, MUTED),
                    self.T("recognize a situation", 26, FG)).arrange(DOWN, buff=0.15).move_to(keyb)
        vt = VGroup(self.T("value", 34, TEAL, BOLD), self.T("output weights", 24, MUTED),
                    self.T("write an answer", 26, FG)).arrange(DOWN, buff=0.15).move_to(valb)
        kv_ar = Arrow(keyb.get_right(), valb.get_left(), buff=0.15, color=MUTED, stroke_width=4)
        ex = VGroup(self.M('"… has a"', 26, BLUE), self.T("→", 28, MUTED), self.T("push toward colors", 26, TEAL),
                    self.T(f"(layer {L + 1}, #{N}, as seen above)", 24, MUTED)).arrange(RIGHT, buff=0.25).next_to(VGroup(keyb, valb), DOWN, buff=0.5)
        cite = self.T("one leading view · Geva et al. 2021 · Meng et al. 2022", 24, MUTED).to_edge(DOWN, buff=0.35)
        with self.say("So where do facts live? One leading view: an MLP works like a key-value memory. Input weights are "
                      "keys that recognize a situation. Output weights are values that write an answer. Researchers can "
                      "edit a few MLP weights and change a fact a model states. Strong evidence, but not the whole story.",
                      scene=f"Diagram: key (input weights: recognize a situation) → value (output weights: write an answer). "
                            f"Example from this model: '… has a' → push toward colors (layer {L + 1} neuron #{N}). Footnote: one "
                            "leading view (Geva et al. 2021; Meng et al. 2022), not the whole story.",
                      chapter="Where facts live"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(btitle, bx, bnote)), ReplacementTransform(hdr3, hdr4), run_time=0.5)
            self.sync("key-value", 0.2)
            self.play(FadeIn(cite), run_time=0.3)
            self.sync("Input weights", 0.1)
            self.play(FadeIn(keyb), FadeIn(kt), run_time=0.5)
            self.sync("Output weights", 0.1)
            self.play(GrowArrow(kv_ar), FadeIn(valb), FadeIn(vt), run_time=0.6)
            self.mark("key", VGroup(keyb, kt), "key: an MLP neuron's input weights detect a pattern in the token's vector")
            self.mark("value", VGroup(valb, vt), "value: the neuron's output weights add a direction (e.g. toward some tokens)")
            self.play(FadeIn(ex, shift=UP * 0.2), run_time=0.5)
            self.mark("kv-example", ex, f"this model's layer {L + 1} neuron #{N}: fires on '… has a', pushes toward color tokens")
            self.mark("cite", cite, "Geva et al. 2021, 'Transformer Feed-Forward Layers Are Key-Value Memories'; Meng et al. 2022, 'Locating and Editing Factual Associations in GPT' (ROME)")

        # ------------------------------------------------------------ recap
        def mini(t, c):
            return self.chip(t, c, 30)
        a_row = VGroup(mini("mia", TEAL), mini(" red", PINK), mini(" ball", YELLOW), mini(" the", "#5fd0ff")).arrange(RIGHT, buff=0.3)
        arcs = VGroup(CurvedArrow(a_row[1].get_top() + UP * 0.05, a_row[3].get_top() + UP * 0.05, angle=-PI / 2.5, color=BLUE, stroke_width=4))
        card_a = VGroup(self.T("attention", 36, BLUE, BOLD), self.T("moves information between tokens", 26, FG), VGroup(a_row, arcs)).arrange(DOWN, buff=0.35)
        m_in = mini(" the", "#5fd0ff")
        m_box = RoundedRectangle(corner_radius=0.12, width=1.3, height=0.7, fill_color=TEAL, fill_opacity=0.2, stroke_color=TEAL)
        m_lab = self.T("MLP", 24, TEAL).move_to(m_box)
        m_out = mini(" red", PINK)
        m_row = VGroup(m_in, Arrow(ORIGIN, RIGHT * 0.7, color=MUTED, buff=0), VGroup(m_box, m_lab),
                       Arrow(ORIGIN, RIGHT * 0.7, color=MUTED, buff=0), m_out).arrange(RIGHT, buff=0.15)
        card_m = VGroup(self.T("MLP", 36, TEAL, BOLD), self.T("transforms it, inside each token", 26, FG), m_row).arrange(DOWN, buff=0.35)
        cards = VGroup(card_a, card_m).arrange(RIGHT, buff=1.4).move_to(UP * 0.5)
        cta = self.T("now poke the neurons in the X-ray →", 44, YELLOW, BOLD).move_to(DOWN * 2.5)
        with self.say("So: attention moves information between tokens. The MLP transforms it, inside each token. "
                      "Stack both three times, and 'the' turns into red. Go poke the neurons yourself in the X-ray.",
                      scene="Recap cards: attention (blue) moves information between tokens (an arc from ' red' to ' the'); the "
                            "MLP (teal) transforms each token on its own (' the' → MLP → leaning ' red'). CTA: poke the neurons in the X-ray.",
                      chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(keyb, valb, kt, vt, kv_ar, ex, cite, hdr4)), run_time=0.5)
            self.play(FadeIn(card_a[:2]), FadeIn(a_row), Create(arcs), run_time=0.8)
            self.mark("card-attn", card_a, "attention moves information between token positions")
            self.sync("The MLP", 0.1)
            self.play(FadeIn(card_m, shift=UP * 0.2), run_time=0.7)
            self.mark("card-mlp", card_m, "the MLP transforms each token's vector independently")
            self.sync("Go poke", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "open the X-ray to inspect MLP neurons live")
        self.wait(0.6)
