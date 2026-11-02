import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from vscene import *  # noqa
from lib_inference import (VOCAB, W, E, PROBE, tid, tok, run, lens, _ln)

VIDEO = dict(id="residual-stream", cls="ResidualStream", title="The residual stream: a shared whiteboard",
             subtitle="Every block reads it and adds a note: real vector sizes, the logit lens, LayerNorm, and knockouts",
             relates="xray", track="inference", order=60, poster=("Logit lens", 1, 0.95))

NAMES = ["embed", "attn1", "mlp1", "attn2", "mlp2", "attn3", "mlp3"]
NICE = {"embed": "token + position", "attn1": "+ attention 1", "mlp1": "+ MLP 1", "attn2": "+ attention 2",
        "mlp2": "+ MLP 2", "attn3": "+ attention 3", "mlp3": "+ MLP 3"}


def div_color(v, vmax):
    a = float(np.clip(v / vmax, -1, 1))
    return interpolate_color(ManimColor(BG), ManimColor(ORANGE if a > 0 else BLUE), abs(a) ** 0.8)


def tshow(t):
    return t.replace(" ", "␣")


class ResidualStream(VScene):
    VIDEO_ID = "residual-stream"

    def strip(self, vec, vmax, cw=0.1, ch=0.38):
        return VGroup(*[Rectangle(width=cw, height=ch, stroke_width=0.5, stroke_color=BG, fill_color=div_color(v, vmax),
                                  fill_opacity=1) for v in vec]).arrange(RIGHT, buff=0)

    def construct(self):
        ids = tok(PROBE)
        R = run(ids)
        st = {k: x[0] for k, x in R["stages"]}
        last = {k: st[k][-1] for k in NAMES}
        norms = {k: float(last[k].norm()) for k in NAMES}
        writes = {f"{kind}{l + 1}": float(w[0, -1].norm()) for kind, l, w in R["writes"]}
        mean_norm = {k: float(st[k].norm(dim=-1).mean()) for k in NAMES}
        RED = tid(" red")

        # ------------------------------------------------------------ the whiteboard
        y0 = -1.3
        stream = Arrow([-6.6, y0, 0], [6.6, y0, 0], buff=0, color=FG, stroke_width=7, max_tip_length_to_length_ratio=0.03)
        s_lab = self.T("residual stream", 28, FG, BOLD).next_to(stream, DOWN, buff=0.25).align_to(stream, LEFT).shift(RIGHT * 0.3)
        bx = np.linspace(-4.6, 4.6, 6)
        blocks, taps = VGroup(), VGroup()
        for k, (nm, x) in enumerate(zip(NAMES[1:], bx)):
            col = BLUE if nm.startswith("attn") else TEAL
            b = RoundedRectangle(corner_radius=0.12, width=1.35, height=0.8, fill_color=col, fill_opacity=0.15,
                                 stroke_color=col, stroke_width=2.5).move_to([x, 1.0, 0])
            t = self.T(("attn " if nm.startswith("attn") else "MLP ") + nm[-1], 24, col).move_to(b)
            rd = Arrow([x - 0.3, y0 + 0.05, 0], [x - 0.3, 0.6, 0], buff=0, color=MUTED, stroke_width=3,
                       max_tip_length_to_length_ratio=0.12)
            wr = Arrow([x + 0.3, 0.6, 0], [x + 0.3, y0 + 0.05, 0], buff=0, color=col, stroke_width=3,
                       max_tip_length_to_length_ratio=0.12)
            pl = self.T("+", 30, col, BOLD).next_to(wr.get_end(), RIGHT, buff=0.06).shift(UP * 0.25)
            blocks.add(VGroup(b, t))
            taps.add(VGroup(rd, wr, pl))
        rl = self.T("read", 24, MUTED).next_to(taps[0][0], LEFT, buff=0.1).shift(UP * 0.55)
        wl = self.T("add", 24, BLUE).next_to(taps[0][1], RIGHT, buff=0.35).shift(DOWN * 0.1)
        chip = self.chip(" the", "#5fd0ff", 30).next_to(stream.get_start(), UP, buff=0.25).shift(RIGHT * 0.3)
        out = self.T("readout", 24, MUTED).next_to(stream.get_end(), UP, buff=0.25).shift(LEFT * 0.5)
        with self.say("Every token carries one vector through the whole model, called the residual stream. "
                      "Think of it as a whiteboard. No layer ever erases it. Each one reads it, then adds a note.",
                      scene="A long horizontal arrow (the residual stream) runs from the token ' the' to the readout. Six blocks "
                            "sit above it (attention 1, MLP 1, … MLP 3); each has a gray 'read' arrow up from the stream and a "
                            "colored 'add' arrow back down with a + sign.", chapter="The whiteboard"):
            self.play(GrowArrow(stream), FadeIn(s_lab), FadeIn(chip), FadeIn(out), run_time=0.8)
            self.mark("stream", VGroup(stream, s_lab), "the residual stream: one 64-number vector per token, shared by all blocks")
            self.sync("whiteboard", 0.1)
            self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.2) for b in blocks], lag_ratio=0.12), run_time=0.9)
            self.sync("reads it", 0.1)
            self.play(LaggedStart(*[GrowArrow(t[0]) for t in taps], lag_ratio=0.08), FadeIn(rl), run_time=0.6)
            self.sync("adds a note", 0.1)
            self.play(LaggedStart(*[AnimationGroup(GrowArrow(t[1]), FadeIn(t[2])) for t in taps], lag_ratio=0.08),
                      FadeIn(wl), run_time=0.7)
            for k, nm in enumerate(NAMES[1:]):
                self.mark(f"block-{nm}", VGroup(blocks[k], taps[k]), f"{NICE[nm][2:]}: reads the stream (through a LayerNorm) and adds its output")

        code = VGroup(self.M("for each layer:", 28, MUTED),
                      self.M("x = x + attention(LayerNorm(x))", 28, BLUE),
                      self.M("x = x + MLP(LayerNorm(x))", 28, TEAL)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        code[1:].shift(RIGHT * 0.6)
        code.move_to(UP * 2.75)
        with self.say("In code it's one line per block. x equals x, plus something. Nothing is overwritten, only added.",
                      scene="The two lines of code that define the stream: x = x + attention(LayerNorm(x)); x = x + MLP(LayerNorm(x)), "
                            "repeated for each of the 3 layers. A dot travels along the stream picking up each block's note."):
            self.play(FadeIn(code, shift=DOWN * 0.2), run_time=0.6)
            self.mark("code", code, "pre-LN transformer block: x = x + attn(LN(x)); x = x + mlp(LN(x))")
            dot = Dot(stream.get_start() + RIGHT * 0.2, radius=0.12, color=YELLOW)
            self.play(FadeIn(dot), run_time=0.2)
            for k in range(6):
                self.play(dot.animate.move_to([bx[k] + 0.3, y0, 0]), Indicate(taps[k][1], color=YELLOW), run_time=0.45)
            self.play(dot.animate.move_to(stream.get_end() + LEFT * 0.3), run_time=0.4)
            self.play(FadeOut(dot), run_time=0.2)

        # ------------------------------------------------------------ real vectors
        hdr = self.header("1 · Adding up")
        vmax = float(torch.stack([last[k] for k in NAMES]).abs().max()) * 0.6
        rows = VGroup()
        for k, nm in enumerate(NAMES):
            col = FG if nm == "embed" else (BLUE if nm.startswith("attn") else TEAL)
            lab = self.T(NICE[nm], 26, col)
            s = self.strip(last[nm].numpy(), vmax)
            n = self.M(f"{norms[nm]:.2f}", 26, YELLOW)
            rows.add(VGroup(lab, s, n))
        for k, r in enumerate(rows):
            y = 2.35 - k * 0.66
            r[1].move_to([0.2, y, 0])
            r[0].next_to(r[1], LEFT, buff=0.35)
            r[2].next_to(r[1], RIGHT, buff=0.35)
        nh = self.T("length", 24, YELLOW).next_to(rows[0][2], UP, buff=0.2)
        sub = self.T('the real stream at the final "the" of "…now tom has the"', 26, MUTED).to_edge(DOWN, buff=0.4)
        with self.say(f"Here's the real stream for the last word of our story. It starts as token plus position. "
                      f"Block by block, notes pile up, and its length grows from {norms['embed']:.2f} to {norms['mlp3']:.2f}.",
                      scene="Seven real 64-cell strips for the final ' the' (orange positive, blue negative): token + position, then "
                            "after each block. Their lengths: " + ", ".join(f"{NICE[k]} {norms[k]:.2f}" for k in NAMES) + ".",
                      chapter="Adding up"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(stream, s_lab, blocks, taps, rl, wl, chip, out, code)), FadeIn(hdr), run_time=0.5)
            self.play(FadeIn(sub), FadeIn(rows[0]), FadeIn(nh), run_time=0.6)
            self.sync("Block by block", 0.1)
            for k in range(1, 7):
                self.play(TransformFromCopy(rows[k - 1][1], rows[k][1]), FadeIn(rows[k][0]), FadeIn(rows[k][2]), run_time=0.42)
            for k, nm in enumerate(NAMES):
                self.mark(f"vec-{nm}", rows[k], f"stream at the last token {('after ' + NICE[nm][2:]) if k else '(token + position embedding)'}: "
                                                f"length {norms[nm]:.3f}")

        wl_ = {k: writes[k] for k in NAMES[1:]}
        wtags = VGroup()
        for k, nm in enumerate(NAMES[1:], start=1):
            col = PINK if nm == "mlp3" else MUTED
            wtags.add(self.M(f"+{wl_[nm]:.2f}", 26, col).next_to(rows[k][2], RIGHT, buff=0.5))
        wh = self.T("note size", 24, MUTED).next_to(wtags[0], UP, buff=0.2).set_y(nh.get_y())
        with self.say(f"Most notes are modest, around length one. But the last MLP writes a big one, {wl_['mlp3']:.2f}, "
                      f"larger than the whole board before it.",
                      scene="Next to each strip, the size (length) of the note that block added: " +
                            ", ".join(f"{NICE[k][2:]} {wl_[k]:.2f}" for k in NAMES[1:]) +
                            f". MLP 3's note ({wl_['mlp3']:.2f}) is larger than the stream before it ({norms['attn3']:.2f})."):
            self.play(FadeIn(wh), LaggedStart(*[FadeIn(t, shift=LEFT * 0.1) for t in wtags], lag_ratio=0.1), run_time=0.8)
            for k, nm in enumerate(NAMES[1:]):
                self.mark(f"write-{nm}", wtags[k], f"length of the vector {NICE[nm][2:]} added at the last token: {wl_[nm]:.3f}")
            self.sync("But the last", 0.1)
            box = SurroundingRectangle(VGroup(rows[6], wtags[5]), color=PINK, buff=0.1, corner_radius=0.1)
            self.play(Create(box), Indicate(wtags[5], color=PINK, scale_factor=1.3), run_time=0.8)

        # ------------------------------------------------------------ logit lens
        hdr2 = self.header("2 · Logit lens")
        checkpoints = [("embed", "token + position"), ("mlp1", "after layer 1"), ("mlp2", "after layer 2"), ("mlp3", "after layer 3")]
        lens_res = {}
        for nm, _ in checkpoints:
            p = lens(st[nm])[-1].softmax(-1)
            t = torch.topk(p, 3)
            lens_res[nm] = [(VOCAB[int(i)], float(v)) for v, i in zip(t.values, t.indices)]
        cols = VGroup()
        for nm, title in checkpoints:
            hd = self.T(title, 28, FG, BOLD)
            items = VGroup()
            for j, (t, p) in enumerate(lens_res[nm]):
                c = PINK if t == " red" else ("#e8e8e8" if j == 0 else MUTED)
                lab = self.M(tshow(t), 26, c, t2c={"␣": MUTED})
                bar = Rectangle(width=max(0.03, 2.2 * p), height=0.34, fill_color=c, fill_opacity=0.85, stroke_width=0)
                val = self.M(f"{p * 100:.0f}%", 24, c)
                row = VGroup(lab, VGroup(bar, val).arrange(RIGHT, buff=0.12)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
                items.add(row)
            items.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            cols.add(VGroup(hd, items).arrange(DOWN, aligned_edge=LEFT, buff=0.35))
        cols.arrange(RIGHT, buff=0.55, aligned_edge=UP).move_to(DOWN * 0.35)
        lt = VGroup(self.T("read the board at any depth with the model's own final readout", 26, MUTED),
                    self.M("final LayerNorm → unembedding → softmax", 24, MUTED)).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.95)
        (t0, p0), (t1, p1), (t2, p2), (t3, p3) = [lens_res[nm][0] for nm, _ in checkpoints]
        with self.say("Because every block shares one board, you can read it at any depth with the model's own final "
                      "readout. That's the logit lens.",
                      scene="Four checkpoints on the stream (token + position, after layer 1, after layer 2, after layer 3) will "
                            "each be decoded with the model's final LayerNorm + unembedding + softmax: the logit lens.",
                      chapter="Logit lens"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows, nh, sub, wtags, wh, box)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(lt), LaggedStart(*[FadeIn(c[0], shift=DOWN * 0.2) for c in cols], lag_ratio=0.15), run_time=0.9)
            self.mark("lens", lt, "logit lens: apply the final LayerNorm and unembedding to an intermediate stream vector")

        with self.say(f"Straight after embedding, it just says '{t0.strip()}', {p0 * 100:.0f} percent: the readout reuses the "
                      f"embedding table, so it sees the token it's sitting on. After layer 1, still '{t1.strip()}', {p1 * 100:.0f}. "
                      f"After layer 2, '{t2.strip()}', {p2 * 100:.0f}. After layer 3, red, {p3 * 100:.0f}.",
                      scene="Logit-lens top-3 at each checkpoint for the final ' the': " +
                            "; ".join(f"{title}: " + ", ".join(f"'{t}' {p * 100:.1f}%" for t, p in lens_res[nm])
                                      for nm, title in checkpoints) + "."):
            for k, (nm, title) in enumerate(checkpoints):
                if k:
                    self.sync(["", "After layer 1", "After layer 2", "After layer 3"][k], 0.1)
                self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in cols[k][1]], lag_ratio=0.15), run_time=0.6)
                self.mark(f"lens-{nm}", cols[k], f"logit lens {title}: " + ", ".join(f'"{t}" {p * 100:.1f}%' for t, p in lens_res[nm]))
            self.play(Indicate(cols[3][1][0], color=PINK, scale_factor=1.15), run_time=0.6)

        # ------------------------------------------------------------ layernorm
        hdr3 = self.header("3 · LayerNorm")
        ax = Axes(x_range=[0, 6, 1], y_range=[0, 8, 2], x_length=5.2, y_length=3.8, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).move_to(LEFT * 3.6 + DOWN * 0.1)
        pts = [ax.c2p(k, mean_norm[nm]) for k, nm in enumerate(NAMES)]
        curve = VMobject(color=YELLOW, stroke_width=4).set_points_as_corners(pts)
        dots = VGroup(*[Dot(p, radius=0.07, color=YELLOW) for p in pts])
        xl = VGroup(*[self.M(s, 24, MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.15)
                      for k, s in enumerate(["emb", "a1", "m1", "a2", "m2", "a3", "m3"])])
        yl = VGroup(*[self.M(str(v), 24, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0, 4, 8)])
        at = self.T("average stream length (all 23 tokens)", 26, MUTED).next_to(ax, UP, buff=0.25)
        v_s = self.M(f"{mean_norm['embed']:.1f}", 26, YELLOW).next_to(dots[0], UR, buff=0.1)
        v_e = self.M(f"{mean_norm['mlp3']:.1f}", 26, YELLOW).next_to(dots[-1], UP, buff=0.15)
        xr = last["mlp3"]
        xn = (xr - xr.mean()) / xr.std(unbiased=False)
        vm = float(xn.abs().max()) * 0.6
        s_raw = self.strip(xr.numpy(), vm, cw=0.07, ch=0.42)
        s_ln = self.strip(xn.numpy(), vm, cw=0.07, ch=0.42)
        lng = VGroup(VGroup(self.T(f"raw stream (spread {float(xr.std(unbiased=False)):.2f})", 24, MUTED), s_raw).arrange(DOWN, aligned_edge=LEFT, buff=0.12),
                     VGroup(self.T("LayerNorm:", 28, YELLOW, BOLD), self.T("subtract the mean,", 26, FG),
                            self.T("divide by the spread", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
                     VGroup(self.T("standard size (spread 1)", 24, MUTED), s_ln).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.6 + DOWN * 0.1)
        with self.say(f"One catch: the board's scale drifts. Averaged over the story, its length grows from "
                      f"{mean_norm['embed']:.1f} to {mean_norm['mlp3']:.1f}. So before any block reads it, LayerNorm "
                      f"re-centres the vector and rescales it to a standard size.",
                      scene=f"Line chart of the real average stream length over the 23 story tokens at each stage: "
                            + ", ".join(f"{k} {mean_norm[k]:.2f}" for k in NAMES) + ". Right: the final stream vector before "
                            "and after LayerNorm (subtract the mean, divide by the spread; then a learned scale and shift).",
                      chapter="LayerNorm"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(cols, lt)), ReplacementTransform(hdr2, hdr3), run_time=0.5)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(at), run_time=0.6)
            self.sync("Averaged", 0.1)
            self.play(Create(curve), LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.15), FadeIn(v_s), run_time=1.2)
            self.play(FadeIn(v_e), run_time=0.3)
            self.mark("growth", VGroup(ax, curve, dots, v_s, v_e), "average residual-stream length over the 23 tokens: " +
                      " → ".join(f"{mean_norm[k]:.2f}" for k in NAMES))
            self.sync("So before", 0.1)
            self.play(FadeIn(lng[0]), run_time=0.4)
            self.play(FadeIn(lng[1]), FadeIn(lng[2], shift=DOWN * 0.2), run_time=0.7)
            self.mark("ln", lng, f"LayerNorm on the final stream vector: mean {float(xr.mean()):+.3f}, spread {float(xr.std(unbiased=False)):.3f} "
                                 "→ mean 0, spread 1 (then a learned per-dimension scale and shift)")

        p_raw = float((xr @ E.T).softmax(-1)[RED])
        p_ln = float(lens(st["mlp3"])[-1].softmax(-1)[RED])
        cmp = VGroup()
        for name, p in [("with the final LayerNorm", p_ln), ("skip it", p_raw)]:
            lab = self.T(name, 26, FG)
            bar = Rectangle(width=max(0.04, 5.0 * p), height=0.5, fill_color=PINK, fill_opacity=0.85, stroke_width=0)
            v = self.M(f"{p * 100:.1f}%", 28, PINK)
            cmp.add(VGroup(lab, bar, v))
        cmp[0][0].move_to([-2.6, -2.35, 0], aligned_edge=RIGHT); cmp[0][1].move_to([-2.3, -2.35, 0], aligned_edge=LEFT)
        cmp[1][0].move_to([-2.6, -3.1, 0], aligned_edge=RIGHT); cmp[1][1].move_to([-2.3, -3.1, 0], aligned_edge=LEFT)
        for r in cmp:
            r[2].next_to(r[1], RIGHT, buff=0.2)
        cmp_t = self.T('P(" red")', 26, PINK).next_to(cmp, LEFT, buff=0.3)
        with self.say(f"The final readout needs one too. Skip it, and red drops from {p_ln * 100:.0f} percent to "
                      f"{p_raw * 100:.0f}.",
                      scene=f"P(' red') at the end: {p_ln * 100:.1f}% with the final LayerNorm vs {p_raw * 100:.1f}% if the raw "
                            "stream is fed straight into the unembedding."):
            self.play(VGroup(ax, curve, dots, xl, yl, at, v_s, v_e).animate.scale(0.85).shift(UP * 0.55),
                      lng.animate.scale(0.85).shift(UP * 0.55), run_time=0.5)
            self.play(FadeIn(cmp[0][0]), GrowFromEdge(cmp[0][1], LEFT), FadeIn(cmp[0][2]), run_time=0.5)
            self.sync("Skip it", 0.1)
            self.play(FadeIn(cmp[1][0]), GrowFromEdge(cmp[1][1], LEFT), FadeIn(cmp[1][2]), run_time=0.5)
            self.mark("with-ln", cmp[0], f'P(" red") with the final LayerNorm: {p_ln * 100:.2f}%')
            self.mark("no-ln", cmp[1], f'P(" red") reading the raw stream without the final LayerNorm: {p_raw * 100:.2f}%')

        # ------------------------------------------------------------ why it matters: knockouts
        hdr4 = self.header("4 · Why it matters")
        full = float(R["logits"][0, -1].softmax(-1)[RED])
        ko = []
        order = [("attn", 0), ("mlp", 0), ("attn", 1), ("mlp", 1), ("attn", 2), ("mlp", 2)]
        for kind, l in order:
            p = float(run(ids, skip={(kind, l)})["logits"][0, -1].softmax(-1)[RED])
            ko.append((kind, l, p))
        krows = VGroup()
        allr = [("nothing removed", None, full)] + [(("attention " if k == "attn" else "MLP ") + str(l + 1), k, p) for k, l, p in ko]
        for name, kind, p in allr:
            c = FG if kind is None else (BLUE if kind == "attn" else TEAL)
            lab = self.T(name, 26, c)
            bar = Rectangle(width=max(0.04, 6.0 * p), height=0.42, fill_color=PINK, fill_opacity=0.85 if kind else 0.5, stroke_width=0)
            v = self.M(f"{p * 100:.1f}%", 26, PINK)
            krows.add(VGroup(lab, bar, v))
        for k, r in enumerate(krows):
            y = 2.0 - k * 0.66 - (0.2 if k else 0)
            r[0].move_to([-2.2, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-1.9, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.2)
        kt = self.T('P(" red") when one block\'s note is erased (at every position)', 26, MUTED).move_to(UP * 2.85)
        p_m3 = ko[5][2]; p_a2 = ko[2][2]
        with self.say(f"For interpretability, this is a gift. Every note is just a vector you can measure, read, or erase. "
                      f"Erase layer 3's MLP note, and red falls to {p_m3 * 100:.0f} percent. Erase layer 2's attention, home of "
                      f"the color-copying head, and it falls to {p_a2 * 100:.0f}.",
                      scene="Real knockouts: P(' red') with nothing removed vs with one block's output erased (zeroed at every "
                            "position): " + ", ".join(f"{n} {p * 100:.1f}%" for n, _, p in allr) + ".",
                      chapter="Why it matters"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax, curve, dots, xl, yl, at, v_s, v_e, lng, cmp, cmp_t)), ReplacementTransform(hdr3, hdr4),
                      run_time=0.5)
            self.play(FadeIn(kt), FadeIn(krows[0][0]), GrowFromEdge(krows[0][1], LEFT), FadeIn(krows[0][2]), run_time=0.6)
            self.sync("Erase layer 3", 0.1)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2])) for r in krows[1:]],
                                  lag_ratio=0.12), run_time=1.0)
            self.play(Indicate(krows[6], color=PINK, scale_factor=1.08), run_time=0.6)
            self.sync("Erase layer 2", 0.1)
            self.play(Indicate(krows[3], color=PINK, scale_factor=1.08), run_time=0.6)
            for (name, kind, p), r in zip(allr, krows):
                self.mark("ko-" + name.replace(" ", "-"), r, f'P(" red") with {name if kind is None else name + " erased"}: {p * 100:.2f}%')

        # design
        g1 = VGroup(self.T("add, don't overwrite", 34, YELLOW, BOLD),
                    self.T("gradients flow straight back through the + signs,", 26, FG),
                    self.T("a big part of why very deep networks can train", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        g2 = VGroup(self.T("keep the board, swap the writers", 34, TEAL, BOLD),
                    self.T("e.g. mixture-of-experts MLPs, new kinds of token mixing", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        g = VGroup(g1, g2).arrange(DOWN, aligned_edge=LEFT, buff=0.6).move_to(DOWN * 1.55)
        # diagram: the stream with three "+" junctions; a gradient arrow runs back along it
        ys = 1.55
        dstream = Arrow([-5.2, ys, 0], [5.2, ys, 0], buff=0, color=FG, stroke_width=5, max_tip_length_to_length_ratio=0.03)
        dblocks, djoin = VGroup(), VGroup()
        for k, x in enumerate((-2.6, 0.0, 2.6)):
            bb = RoundedRectangle(corner_radius=0.1, width=1.2, height=0.6, fill_color=BLUE if k % 2 == 0 else TEAL,
                                  fill_opacity=0.15, stroke_color=BLUE if k % 2 == 0 else TEAL, stroke_width=2).move_to([x, ys + 1.15, 0])
            bt = self.T("block", 24, BLUE if k % 2 == 0 else TEAL).move_to(bb)
            j = VGroup(Circle(radius=0.2, color=YELLOW, stroke_width=3).set_fill(BG, 1).move_to([x + 0.45, ys, 0]),
                       self.T("+", 28, YELLOW, BOLD).move_to([x + 0.45, ys, 0]))
            rd = Line([x - 0.35, ys, 0], [x - 0.35, ys + 0.85, 0], color=MUTED, stroke_width=2.5)
            wr = Arrow([x + 0.45, ys + 0.85, 0], [x + 0.45, ys + 0.2, 0], buff=0, color=MUTED, stroke_width=2.5,
                       max_tip_length_to_length_ratio=0.25)
            dblocks.add(VGroup(bb, bt, rd, wr)); djoin.add(j)
        grad = DashedVMobject(Arrow([5.0, ys - 0.45, 0], [-5.0, ys - 0.45, 0], buff=0, color=ORANGE, stroke_width=4,
                                    max_tip_length_to_length_ratio=0.03), num_dashes=40)
        grad_l = self.T("gradient: straight back through every +", 24, ORANGE).next_to(grad, DOWN, buff=0.12)
        with self.say("And for design. Because blocks add instead of overwrite, gradients flow straight back through the "
                      "plus signs. That's a big part of why very deep networks can train at all. Many newer architectures keep this whiteboard, "
                      "and change what writes on it.",
                      scene="Two design takeaways: 'add, don't overwrite' (gradients flow straight back through the additions, "
                            "which lets very deep networks train) and 'keep the board, swap the writers' (e.g. mixture-of-experts "
                            "MLPs, new kinds of token mixing)."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(kt, krows)), run_time=0.4)
            self.play(GrowArrow(dstream), FadeIn(dblocks), FadeIn(djoin), run_time=0.7)
            self.play(Create(grad), FadeIn(grad_l), FadeIn(g1, shift=UP * 0.2), run_time=0.9)
            self.mark("highway", VGroup(dstream, djoin, grad, grad_l), "the + junctions pass gradients straight back to early layers")
            self.mark("design-1", g1, "residual connections: the identity path lets gradients reach early layers (ResNets, 2015)")
            self.sync("Many newer", 0.1)
            self.play(FadeIn(g2, shift=UP * 0.2), run_time=0.6)
            self.mark("design-2", g2, "newer designs often keep the residual stream and replace the blocks that write to it")

        # ------------------------------------------------------------ recap
        stream2 = Arrow([-6.0, -0.6, 0], [6.0, -0.6, 0], buff=0, color=FG, stroke_width=7, max_tip_length_to_length_ratio=0.03)
        lab_chips = VGroup()
        for k, (nm, title) in enumerate(checkpoints):
            t, p = lens_res[nm][0]
            c = self.chip(t, PINK if t == " red" else MUTED, 28)
            pp = self.M(f"{p * 100:.0f}%", 24, PINK if t == " red" else MUTED)
            lab_chips.add(VGroup(c, pp).arrange(DOWN, buff=0.1).move_to([-4.8 + 3.2 * k, 0.4, 0]))
        cta = self.T("now watch the board fill up in the X-ray →", 42, YELLOW, BOLD).move_to(DOWN * 2.4)
        with self.say("One shared board. Six notes. One readout. Go watch it fill up in the X-ray.",
                      scene="Recap: the stream arrow with the logit-lens guesses along it (' the' → ' the' → ' white' → ' red'). "
                            "CTA: watch the board fill up in the X-ray.", chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(g, hdr4, dstream, dblocks, djoin, grad, grad_l)), run_time=0.4)
            self.play(GrowArrow(stream2), LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in lab_chips], lag_ratio=0.25), run_time=1.2)
            self.mark("recap", VGroup(stream2, lab_chips), "logit-lens guesses along the stream: " +
                      " → ".join(f'"{lens_res[nm][0][0]}" {lens_res[nm][0][1] * 100:.0f}%' for nm, _ in checkpoints))
            self.sync("Go watch", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "open the X-ray to see the residual stream live")
        self.wait(0.6)
