import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
import torch
import model as M
from model import VOCAB, W, tok, forward, lens

PROMPT = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the"
TOKCOL = {" red": PINK, " blue": BLUE, "mia": TEAL, " mia": TEAL, " tom": ORANGE, " ball": YELLOW,
          " kite": "#b48cff", " the": "#5fd0ff"}
QC, KC, VC, OC = BLUE, TEAL, ORANGE, YELLOW   # query / key / value / output colors

# toy 2-D example (clearly labelled as a toy on screen)
TOY_Q = np.array([2.0, 1.0])
TOY = [(" red", np.array([1.8, 1.2]), np.array([0.6, 1.8])),
       (" mia", np.array([-0.6, 1.4]), np.array([1.5, -0.5])),
       (" ball", np.array([0.4, -1.6]), np.array([-1.2, -0.6]))]


def tcol(t):
    return TOKCOL.get(t, MUTED)


def hit(mob, min_u=0.32):
    """Pad a tiny mobject's click box to >= ~43 px with an invisible backing rect (moves/fades with it)."""
    if mob.width < min_u or mob.height < min_u:
        r = Rectangle(width=max(mob.width, min_u), height=max(mob.height, min_u), stroke_width=0,
                      fill_color=BG, fill_opacity=0.04).move_to(mob)
        mob.add_to_back(r)
    return mob


def handle(parent, point, size=0.34):
    """Small invisible click target glued to `parent` (e.g. an attention line), so its box stays compact."""
    r = Square(size, stroke_width=0, fill_color=BG, fill_opacity=0.04).move_to(point)
    parent.add(r)
    return r


@torch.no_grad()
def p_tok_without(ids, layer, head, tok_id):
    """Forward pass with one head's output zeroed (ablation); returns P(tok_id) at the last position."""
    import torch.nn.functional as F
    T = len(ids); D = M.D
    x = W["tok.weight"][ids] + W["pos.weight"][:T]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for l in range(3):
        p = f"blocks.{l}."
        q, k, v = M._lin(M._ln(x, p + "ln1"), p + "attn_qkv").split(D, -1)
        q, k, v = (t.view(T, M.H, M.HD).transpose(0, 1) for t in (q, k, v))
        a = (q @ k.transpose(-1, -2) / M.HD ** 0.5).masked_fill(mask, float("-inf")).softmax(-1)
        o = a @ v
        if l == layer:
            o[head] = 0
        x = x + M._lin(o.transpose(0, 1).reshape(T, D), p + "attn_out")
        x = x + M._lin(F.gelu(M._lin(M._ln(x, p + "ln2"), p + "mlp_in"), approximate="tanh"), p + "mlp_out")
    return float(lens(x)[-1].softmax(-1)[tok_id])


class Attention(VScene):
    VIDEO_ID = "attention"

    def construct(self):
        ids = tok(PROMPT)
        toks = [VOCAB[i] for i in ids]
        _, tr = forward(ids, trace=True)

        # ------------------------------------------------------------ hook
        ctx = VGroup(*[self.chip(t, tcol(t), 34) for t in toks[:5]]).arrange(RIGHT, buff=0.14)
        dots = self.T("…", 40, MUTED)
        end = VGroup(*[self.chip(t, tcol(t), 34) for t in toks[-3:]]).arrange(RIGHT, buff=0.14)
        line = VGroup(ctx, dots, end).arrange(RIGHT, buff=0.35).move_to(DOWN * 0.4)
        if line.width > 12.0:
            line.scale_to_fit_width(12.0)
        arc = CurvedArrow(end[-1].get_top() + UP * 0.1, ctx[3].get_top() + UP * 0.1, angle=PI / 3.2, color=YELLOW,
                          stroke_width=5)
        qm = self.T("?", 60, YELLOW, BOLD).next_to(arc, UP, buff=0.1)
        with self.say("How does the word 'the' know to look back at 'red'? Attention. Let's build it from three arrows.",
                      scene="The prompt start 'mia has a red ball' … 'tom has the' with a yellow curved arrow from the last "
                            "' the' back to ' red' and a question mark.", chapter="The question"):
            self.play(LaggedStart(FadeIn(ctx, shift=RIGHT * 0.2), FadeIn(dots), FadeIn(end, shift=LEFT * 0.2),
                                  lag_ratio=0.3), run_time=0.9)
            self.play(Create(arc), FadeIn(qm, scale=1.5), run_time=0.7)
            self.mark("tok-the", end[-1], 'the last token " the" (position 22)')
            self.mark("tok-red", ctx[3], 'token " red" (position 3)')
            self.sync("Attention.", 0.1)
            self.flash("attention", YELLOW, 72, hold=0.6, pos=UP * 2.3, bg=False)

        # ------------------------------------------------------------ q k v
        hdr = self.header("Query · Key · Value")
        src = self.chip(" the", tcol(" the"), 40).move_to(LEFT * 5.3 + DOWN * 0.1)
        rows = VGroup()
        for name, col, desc, mat in [("query", QC, "what am I looking for?", "W_Q"),
                                     ("key", KC, "what do I contain?", "W_K"),
                                     ("value", VC, "what do I hand over if picked?", "W_V")]:
            a = Arrow(ORIGIN, RIGHT * 1.3 + UP * 0.35, buff=0, color=col, stroke_width=6)
            n = self.T(name, 34, col, BOLD)
            d = self.T(desc, 28, FG)
            r = VGroup(a, n, d).arrange(RIGHT, buff=0.35)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.75)
        rows.move_to([-3.2 + rows.width / 2, -0.1, 0])
        mats = VGroup(self.T("each = the token's vector × its own learned matrix", 26, MUTED),
                      self.M("W_Q · W_K · W_V", 24, MUTED)).arrange(DOWN, buff=0.12)
        mats.next_to(rows, DOWN, buff=0.55)
        links = VGroup(*[Line(src.get_right() + RIGHT * 0.08, r[0].get_left() + LEFT * 0.08, color=LINE,
                              stroke_width=2.5) for r in rows])
        with self.say("Every token turns its vector into three new ones. A query that asks, what am I looking for. "
                      "A key that says, here's what I've got. And a value, the info it hands over if it gets picked.",
                      scene="The token ' the' spawns three arrows: query (blue) 'what am I looking for?', key (teal) "
                            "'what do I contain?', value (orange) 'what do I hand over if picked?'. Note: each = the token's "
                            "vector × its own learned matrix (W_Q, W_K, W_V).", chapter="Query, key, value"):
            self.unmark("tok-the", "tok-red")
            self.play(FadeOut(VGroup(ctx, dots, end[:-1], arc, qm)), ReplacementTransform(end[-1], src), FadeIn(hdr),
                      run_time=0.6)
            self.mark("src", src, 'token " the" — its vector x feeds three learned matrices')
            for k, ph in enumerate(["A query", "A key", "And a value"]):
                self.sync(ph, 0.1)
                self.play(Create(links[k]), GrowArrow(rows[k][0]), FadeIn(rows[k][1:], shift=RIGHT * 0.2), run_time=0.6)
                self.mark(["q", "k", "v"][k], rows[k], ["query vector: what this token is looking for",
                                                        "key vector: what this token offers to be found by",
                                                        "value vector: the information it passes on"][k])
            self.play(FadeIn(mats), run_time=0.4)
            self.mark("mats", mats, "q = x·W_Q, k = x·W_K, v = x·W_V: three learned 64×16 matrices per head")

        # ------------------------------------------------------------ dot product
        hdr2 = self.header("Scores")
        U = 0.95
        org = np.array([-3.6, -0.5, 0])
        plane = NumberPlane(x_range=[-3, 3, 1], y_range=[-2.5, 2.5, 1], x_length=6 * U, y_length=5 * U,
                            background_line_style={"stroke_color": LINE, "stroke_width": 1.2},
                            axis_config={"stroke_color": MUTED, "stroke_width": 1.5}).move_to(org)
        toy = self.T("toy 2-D vectors", 24, MUTED).next_to(plane, UP, buff=0.15).align_to(plane, LEFT)

        def vec(v, col, w=6):
            return Arrow(plane.c2p(0, 0), plane.c2p(*v), buff=0, color=col, stroke_width=w,
                         max_tip_length_to_length_ratio=0.18)

        qv = vec(TOY_Q, QC)
        ql = self.T("q", 32, QC, BOLD).next_to(qv.get_end(), RIGHT, buff=0.12)
        ang = ValueTracker(np.arctan2(1.2, 1.8))
        kv = always_redraw(lambda: vec(2.0 * np.array([np.cos(ang.get_value()), np.sin(ang.get_value())]), KC))
        rlab = self.M("q · k =", 34, FG).move_to(RIGHT * 2.6 + UP * 2.4)
        rnum = self.live(lambda: f"{float(TOY_Q @ (2.0 * np.array([np.cos(ang.get_value()), np.sin(ang.get_value())]))):+.2f}",
                         lambda t: t.next_to(rlab, RIGHT, buff=0.3), size=40, color=YELLOW)
        readout = VGroup(rlab, rnum)
        with self.say("To score a match, take the dot product of a query with each earlier key. Pointing the same way "
                      "scores high. At right angles, zero. Opposite, negative.",
                      scene="Toy 2-D plane: a blue query arrow and a teal key arrow sweeping around it; a live readout of "
                            "q · k goes high when aligned, 0 at right angles, negative when opposite.", chapter="Scores"):
            self.unmark("src", "q", "k", "v", "mats")
            self.play(FadeOut(VGroup(src, rows, links, mats)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(plane), FadeIn(toy), GrowArrow(qv), FadeIn(ql), run_time=0.8)
            self.add(kv)
            self.play(FadeIn(readout), run_time=0.3)
            self.mark("plane", plane, "toy 2-D space for query/key vectors")
            self.mark("qvec", qv, "query q = (2.0, 1.0)")
            self.mark("readout", readout, "live dot product q · k")
            q_ang = np.arctan2(1.0, 2.0)
            self.sync("same way", 0.1)
            self.play(ang.animate.set_value(q_ang), run_time=0.7)
            self.sync("right angles", 0.1)
            self.play(ang.animate.set_value(q_ang + PI / 2), run_time=0.8)
            self.sync("Opposite", 0.1)
            self.play(ang.animate.set_value(q_ang + PI), run_time=0.8)

        # three real-looking keys (toy) with scores
        dk = 2
        scores = np.array([TOY_Q @ k for _, k, _ in TOY]) / np.sqrt(dk)
        w = np.exp(scores) / np.exp(scores).sum()
        kvecs = VGroup(*[vec(k, KC, 5) for _, k, _ in TOY])
        klabs = VGroup(*[self.M(self.show_tok(t), 26, tcol(t), t2c={"␣": MUTED}).next_to(
            plane.c2p(*k), normalize(np.append(k, 0)), buff=0.15) for (t, k, _) in TOY])
        table = VGroup()
        for (t, k, _), s in zip(TOY, scores):
            table.add(VGroup(self.M(self.show_tok(t), 30, tcol(t), t2c={"␣": MUTED}),
                             self.M(f"{TOY_Q @ k:+.1f}", 30, FG)).arrange(RIGHT, buff=0.6))
        table.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.3 + UP * 0.3)
        th = self.T("q · k for each earlier token", 26, MUTED).next_to(table, UP, buff=0.35)
        with self.say("Do it for every earlier token, and you get one score each. Red's key lines up with the query, "
                      "so it scores highest.",
                      scene="Three toy keys for ' red', ' mia', ' ball' drawn in the plane; a table of dot products: "
                            "' red' +4.8, ' mia' +0.2, ' ball' −0.8."):
            self.unmark("readout")
            self.remove(kv)
            rnum.clear_updaters()
            self.play(FadeOut(readout), LaggedStart(*[GrowArrow(k) for k in kvecs], lag_ratio=0.2),
                      LaggedStart(*[FadeIn(l) for l in klabs], lag_ratio=0.2), run_time=1.0)
            self.play(FadeIn(th), LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in table], lag_ratio=0.2), run_time=0.9)
            for (t, k, _), r, kvec in zip(TOY, table, kvecs):
                self.mark(f"score-{t.strip()}", r, f'toy score q·k for "{t}" = {TOY_Q @ k:+.1f}')
                self.mark(f"key-{t.strip()}", kvec, f'toy key vector for "{t}" = ({k[0]}, {k[1]})')
            self.sync("scores highest", 0.3)
            self.play(Indicate(table[0], color=YELLOW, scale_factor=1.15), run_time=0.6)

        # ------------------------------------------------------------ softmax
        hdr3 = self.header("Softmax")
        bars = VGroup()
        for (t, _, _), s, ww in zip(TOY, scores, w):
            lab = self.M(self.show_tok(t), 30, tcol(t), t2c={"␣": MUTED})
            sb = Rectangle(width=max(0.05, abs(s) * 1.1), height=0.5, fill_color=MUTED, fill_opacity=0.7, stroke_width=0)
            wb = Rectangle(width=max(0.05, ww * 6.0), height=0.5, fill_color=tcol(t), fill_opacity=0.9, stroke_width=0)
            sv = self.M(f"{s:+.2f}", 28, MUTED)
            wv = self.T(f"{ww * 100:.1f}%", 30, tcol(t), BOLD)
            bars.add(VGroup(lab, sb, wb, sv, wv))
        X0 = -2.3  # zero line: positive scores grow right, negative scores grow left
        for n, (b, sc) in enumerate(zip(bars, scores)):
            y = 1.0 - n * 1.0
            b[0].move_to([-3.75, y, 0], aligned_edge=RIGHT)
            b[1].move_to([X0, y, 0], aligned_edge=LEFT if sc >= 0 else RIGHT)
            b[2].move_to([X0, y, 0], aligned_edge=LEFT)
            if sc >= 0:
                b[3].next_to(b[1], RIGHT, buff=0.25)
            else:
                b[3].next_to(np.array([X0, y, 0]), RIGHT, buff=0.25)
            b[4].next_to(b[2], RIGHT, buff=0.25)
        zero = VGroup(Line([X0, 1.45, 0], [X0, -1.45, 0], color=MUTED, stroke_width=2),
                      self.M("0", 22, MUTED).move_to([X0, -1.7, 0]))
        f1 = VGroup(self.M("score = q · k / √d", 30, FG),
                    self.T("d = numbers per key (toy: 2 · TinyGPT's real heads: 16)", 24, MUTED)).arrange(DOWN, buff=0.15)
        f1.move_to(UP * 2.45 + LEFT * 0.6)
        f2 = self.M("weights = softmax(scores)   → positive, sum to 1", 28, FG).move_to(DOWN * 2.3)
        with self.say("Divide by the square root of the vector size. Real heads here use 16 numbers, so divide by 4. "
                      "That stops scores from growing with size and saturating the softmax. Then softmax turns every "
                      "score into a positive weight, and the weights add up to one.",
                      scene="Score bars drawn from a zero line (q·k/√2: ' red' +3.39 and ' mia' +0.14 to the right, ' ball' "
                            "−0.57 to the left) morph into softmax weight bars: ' red' 94.5%, ' mia' 3.7%, ' ball' 1.8%. "
                            "Note: d is the key size, 2 in the toy, 16 in TinyGPT's heads.", chapter="Softmax"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(plane, toy, qv, ql, kvecs, klabs, th)), ReplacementTransform(hdr2, hdr3),
                      *[ReplacementTransform(table[i][0], bars[i][0]) for i in range(3)],
                      *[ReplacementTransform(table[i][1], bars[i][3]) for i in range(3)], run_time=0.8)
            self.play(FadeIn(f1), FadeIn(zero), *[GrowFromEdge(b[1], LEFT if sc >= 0 else RIGHT)
                                                  for b, sc in zip(bars, scores)], run_time=0.6)
            for (t, _, _), b, sc in zip(TOY, bars, scores):
                self.mark(f"s-{t.strip()}", VGroup(b[0], b[1], b[3]), f'toy scaled score for "{t}" = {sc:+.2f}')
            self.sync("Then softmax", 0.1)
            self.unmark("s-red", "s-mia", "s-ball")
            self.play(*[ReplacementTransform(b[1], b[2]) for b in bars], *[ReplacementTransform(b[3], b[4]) for b in bars],
                      FadeIn(f2), run_time=1.0)
            for (t, _, _), b, ww in zip(TOY, bars, w):
                self.mark(f"w-{t.strip()}", VGroup(b[0], b[2], b[4]), f'toy attention weight on "{t}" = {ww * 100:.1f}%')
            self.mark("formula", f1, "scaled dot-product score q·k/√d; d = 16 per head in TinyGPT, so scores are divided by 4")

        # ------------------------------------------------------------ weighted sum of values
        org2 = np.array([2.9, -0.4, 0])
        S = 1.2
        plane2 = NumberPlane(x_range=[-2, 3, 1], y_range=[-1.5, 2.5, 1], x_length=5 * S, y_length=4 * S,
                             background_line_style={"stroke_color": LINE, "stroke_width": 1.2},
                             axis_config={"stroke_color": MUTED, "stroke_width": 1.5}).move_to(org2)
        vvecs = VGroup(*[Arrow(plane2.c2p(0, 0), plane2.c2p(*v), buff=0, color=VC, stroke_width=5,
                               max_tip_length_to_length_ratio=0.15) for _, _, v in TOY])
        vlabs = VGroup(*[self.M("v" + self.show_tok(t), 24, VC, t2c={"␣": MUTED}).next_to(
            plane2.c2p(*v), normalize(np.append(v, 0)), buff=0.12) for t, _, v in TOY])
        out = sum(ww * v for ww, (_, _, v) in zip(w, TOY))
        # tip-to-tail scaled values
        chain, p0 = VGroup(), np.zeros(2)
        for ww, (_, _, v) in zip(w, TOY):
            p1 = p0 + ww * v
            chain.add(Arrow(plane2.c2p(*p0), plane2.c2p(*p1), buff=0, color=VC, stroke_width=5,
                            max_tip_length_to_length_ratio=0.25))
            p0 = p1
        outv = Arrow(plane2.c2p(0, 0), plane2.c2p(*out), buff=0, color=OC, stroke_width=8,
                     max_tip_length_to_length_ratio=0.15)
        outl = VGroup(self.T("output", 30, OC, BOLD), self.M("≈ v␣red", 26, VC, t2c={"␣": MUTED})).arrange(
            DOWN, aligned_edge=RIGHT, buff=0.08).next_to(plane2.c2p(*out), LEFT, buff=0.5)
        wo = VGroup(self.T("then one more matrix (W_O) mixes the heads,", 24, MUTED),
                    self.T("and the result is added onto the token's vector", 24, MUTED)).arrange(DOWN, buff=0.08)
        wo.move_to([0, -3.35, 0])
        small = VGroup(*[VGroup(b[0], b[2], b[4]) for b in bars])
        with self.say("The output is the weighted sum of the values, so it's mostly the value of whatever matched best. "
                      "One more matrix mixes the heads' outputs, and the result gets added onto the token's vector.",
                      scene="Left: the three weights shrink to the side. Right: toy value vectors (orange) for ' red', "
                            "' mia', ' ball'; scaled by their weights and added tip to tail, they give the yellow output "
                            "arrow, almost exactly red's value (red's value arrow stays visible on top). Caption: one more "
                            "matrix (W_O) mixes the heads and the result is added onto the token's vector."):
            self.play(FadeOut(f1), FadeOut(f2), FadeOut(zero), small.animate.scale(0.62).to_edge(LEFT, buff=0.5).shift(UP * 0.2),
                      run_time=0.6)
            self.play(FadeIn(plane2), LaggedStart(*[GrowArrow(v) for v in vvecs], lag_ratio=0.2), FadeIn(vlabs),
                      run_time=0.9)
            self.mark("values", VGroup(vvecs, vlabs), "toy value vectors v for ' red', ' mia', ' ball'")
            self.sync("weighted sum", 0.1)
            self.play(vvecs.animate.set_opacity(0.25), vlabs[0].animate.set_opacity(0), LaggedStart(*[GrowArrow(c) for c in chain], lag_ratio=0.5),
                      run_time=1.0)
            self.play(GrowArrow(outv), FadeIn(outl), run_time=0.5)
            vred = vvecs[0].copy().set_opacity(1).set_stroke(width=3)
            self.add(vred)
            self.play(FadeIn(vred), run_time=0.2)
            self.sync("One more matrix", 0.1)
            self.play(FadeIn(wo, shift=UP * 0.15), run_time=0.5)
            self.mark("w-o", wo, "after attention: concat heads × W_O (64×64), then add to the residual stream")
            self.mark("output", VGroup(outv, outl), f"output = Σ weight·value = ({out[0]:.2f}, {out[1]:.2f}) ≈ v_red")
            for (t, _, _), b, ww in zip(TOY, bars, w):
                self.mark(f"w-{t.strip()}", VGroup(b[0], b[2], b[4]), f'toy attention weight on "{t}" = {ww * 100:.1f}%')

        # ------------------------------------------------------------ causal mask (real L2H4 matrix)
        hdr4 = self.header("Causal mask")
        A = tr["attn"][1][3].numpy()
        n = A.shape[0]
        cs = 0.235
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                if j <= i:
                    v = A[i, j] ** 0.6
                    c = Square(cs, stroke_width=0.6, stroke_color=BG,
                               fill_color=interpolate_color(ManimColor(PANEL), ManimColor(YELLOW), v), fill_opacity=1)
                else:
                    c = Square(cs, stroke_width=0.6, stroke_color=BG, fill_color="#3a1f2c", fill_opacity=1)
                c.move_to([j * cs, -i * cs, 0])
                cells.add(c)
        cells.move_to(RIGHT * 3.4 + DOWN * 0.25)
        gx0, gy0 = cells[0].get_corner(UL)[0], cells[0].get_corner(UL)[1]
        G = lambda c, r: np.array([gx0 + c * cs, gy0 - r * cs, 0])  # grid corner (column c, row r)
        stair = [G(1, 0), G(n, 0), G(n, n - 1)]
        for r in range(n - 1, 0, -1):
            stair += [G(r, r), G(r, r - 1)]
        tri = Polygon(*stair, stroke_color=PINK, stroke_width=3, fill_opacity=0)
        future = self.T("the future: blocked", 24, PINK, BOLD).move_to(G(15.5, 3.4))
        ktitle = self.T("key position (being looked at) →", 24, MUTED).next_to(cells, UP, buff=0.42)
        qtitle = self.T("query position", 24, MUTED).rotate(PI / 2).next_to(cells, LEFT, buff=0.5)
        ticks = VGroup(self.M("0", 20, MUTED).next_to(G(0.5, 0), UP, buff=0.08),
                       self.M("22", 20, MUTED).next_to(G(22.5, 0), UP, buff=0.08),
                       self.M("0", 20, MUTED).next_to(G(0, 0.5), LEFT, buff=0.08),
                       self.M("22", 20, MUTED).next_to(G(0, 22.5), LEFT, buff=0.08))
        redcol = self.M("␣red", 22, PINK, t2c={"␣": MUTED}).next_to(G(3.5, 0), UP, buff=0.08)
        rowlab = self.T("each row = one token looking back", 24, FG)
        collab = self.T("each column = a token being looked at", 24, FG)
        ex = self.T("real weights · layer 2, head 4", 24, MUTED)
        every = self.T("every token is a query at once", 24, BLUE)
        side = VGroup(rowlab, collab, every, ex).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(LEFT, buff=0.55)
        side.shift(DOWN * 0.2)
        last_row = SurroundingRectangle(VGroup(cells[(n - 1) * n], cells[n * n - 1]), color=BLUE, buff=0.02)
        red_cell = SurroundingRectangle(cells[(n - 1) * n + 3], color=YELLOW, buff=0.03, stroke_width=4)
        with self.say("Every token asks its own query at the same time, one row each. One rule. No peeking at the "
                      "future. A causal mask blocks each token from looking at anything after itself, which keeps "
                      "next-token training honest.",
                      scene="The real 23×23 attention matrix of layer 2 head 4 on the prompt (rows = query positions 0–22, "
                            "columns = key positions 0–22): lower triangle colored by weight, the upper triangle (future "
                            "positions) blocked in dark red and outlined along the cell edges.",
                      chapter="Causal mask"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(plane2, vvecs, vlabs, chain, outv, outl, small, vred, wo)), ReplacementTransform(hdr3, hdr4),
                      run_time=0.5)
            self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.002), FadeIn(side[0]), FadeIn(side[1]),
                      FadeIn(ktitle), FadeIn(qtitle), FadeIn(ticks), FadeIn(redcol), FadeIn(side[2]), run_time=1.4)
            self.sync("A causal mask", 0.1)
            self.play(Create(tri), FadeIn(future, scale=1.2), run_time=0.7)
            self.play(FadeIn(ex), Create(last_row), Create(red_cell), run_time=0.6)
            self.mark("matrix", cells, "real attention matrix, layer 2 head 4 (23 × 23)")
            self.mark("mask", future, "causal mask: positions after the query get weight 0 (score −∞)")
            self.mark("axes", ktitle, "columns = key positions 0–22; rows = query positions 0–22 (every token is a query)")
            self.mark("last-row", last_row, 'last row: what the final " the" attends to')
            self.mark("cell-red", red_cell, f'row " the" (22), column " red" (3): weight {A[-1, 3] * 100:.1f}%')

        # ------------------------------------------------------------ the real head
        hdr5 = self.header("A real head")
        a_head = A[-1]
        s_head = tr["scores"][1][3, -1].numpy()
        keys = VGroup()
        for k, t in enumerate(toks):
            txt = self.M(self.show_tok(t), 24, tcol(t), t2c={"␣": MUTED})
            txt.move_to([-0.9, 3.3 - k * 0.31, 0], aligned_edge=RIGHT)
            keys.add(txt)
        posn = VGroup(*[self.M(str(k), 20, MUTED).move_to([-2.9, keys[k].get_y(), 0], aligned_edge=RIGHT)
                        for k in range(23)])
        q_chip = self.chip(" the", tcol(" the"), 36).move_to([3.4, 0.1, 0])
        hl = self.T("layer 2 · head 4", 30, YELLOW, BOLD).move_to([3.4, 3.3, 0])
        q_lab = self.T("query: the last token", 24, MUTED).next_to(q_chip, DOWN, buff=0.25)
        kright = [k_.get_right() for k_ in keys]
        for k_ in keys:
            hit(k_)

        def wire(k, ww, col):
            return Line(q_chip.get_left() + LEFT * 0.08, kright[k] + RIGHT * 0.15, color=col,
                        stroke_width=1.2 + 10 * ww, stroke_opacity=min(1, 0.3 + 1.5 * ww))

        wires = {k: wire(k, float(a_head[k]), YELLOW if k == 3 else BLUE) for k in range(23) if a_head[k] > 0.008}
        info = VGroup(self.M(f"score {s_head[3]:.2f}", 26, FG), self.T(f"{a_head[3] * 100:.0f}%", 40, YELLOW, BOLD))
        info.arrange(DOWN, buff=0.12)
        w3 = wires[3]
        d_ = w3.get_end() - w3.get_start(); nrm = np.array([-d_[1], d_[0], 0]); nrm /= np.linalg.norm(nrm)
        nrm = nrm if nrm[1] > 0 else -nrm
        info.move_to(w3.point_from_proportion(0.42) + nrm * 0.95)
        with self.say("Now the real thing. Layer two, head four, on our prompt. From the final 'the', it puts 93 percent "
                      "of its attention on 'red', way back at position three.",
                      scene="All 23 prompt tokens in a column; lines fan from the final ' the' (query) to the tokens so far. "
                            "The line to ' red' (position 3) is thick and yellow, labeled score 4.28 → 93%; the other lines "
                            "are thin and unlabeled (each under 3%).", chapter="A real head"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(cells, tri, future, side, last_row, red_cell, ktitle, qtitle, ticks, redcol)),
                      ReplacementTransform(hdr4, hdr5),
                      run_time=0.5)
            self.play(LaggedStart(*[FadeIn(k_, shift=RIGHT * 0.2) for k_ in keys], lag_ratio=0.03), FadeIn(posn),
                      FadeIn(q_chip), FadeIn(q_lab), FadeIn(hl), run_time=1.0)
            self.play(LaggedStart(*[Create(x) for x in wires.values()], lag_ratio=0.1), run_time=1.0)
            self.sync("93", 0.2)
            self.play(FadeIn(info, scale=1.3), keys[3].animate.scale(1.35, about_edge=RIGHT), run_time=0.6)
            self.mark("query", q_chip, 'query: the final " the" (position 22)')
            self.mark("arc-red", info,
                      f'L2H4: " the" → " red" (pos 3): score {s_head[3]:.2f}, weight {a_head[3] * 100:.1f}%')
            for k, x in wires.items():
                if k != 3:
                    hd = handle(x, x.point_from_proportion(0.8))
                    self.mark(f"arc-{k}", hd, f'L2H4: " the" → "{toks[k]}" (pos {k}): weight {a_head[k] * 100:.1f}%')
            for k, t in enumerate(toks):
                self.mark(f"tok-{k}", keys[k], f'token "{t}" (id {ids[k]}), position {k}: weight {a_head[k] * 100:.1f}%')

        pulse = Dot(radius=0.11, color=VC).move_to(w3.get_end())
        RED = VOCAB.index(" red")
        p_full = float(forward(ids)[-1].softmax(-1)[RED])
        p_abl = p_tok_without(ids, 1, 3, RED)
        abl = VGroup(self.T("switch this head off:", 26, MUTED),
                     self.T(f"P(red) {p_full * 100:.0f}% → {p_abl * 100:.0f}%", 32, PINK, BOLD)).arrange(DOWN, buff=0.12)
        abl.next_to(q_lab, DOWN, buff=0.45)
        with self.say(f"So red's value gets carried forward, right to the spot where the model has to name a color. "
                      f"Switch this one head off, and red drops from {p_full * 100:.0f} percent to {p_abl * 100:.0f}. "
                      f"Nobody programmed that. It fell out of training.",
                      scene=f"An orange 'value' dot travels from ' red' along the yellow line into the final ' the'. "
                            f"Real ablation (this head's output zeroed): P(' red') {p_full * 100:.1f}% → {p_abl * 100:.1f}%. "
                            f"Flash: 'nobody programmed this'."):
            self.play(FadeIn(pulse, scale=2), run_time=0.3)
            self.play(MoveAlongPath(pulse, Line(w3.get_end(), w3.get_start())), run_time=1.1)
            self.play(Flash(q_chip, color=VC, line_length=0.3), FadeOut(pulse), q_chip[0].animate.set_stroke(VC),
                      run_time=0.6)
            self.sync("Switch this", 0.1)
            self.play(FadeIn(abl, shift=UP * 0.15), run_time=0.5)
            self.mark("ablation", abl, f"ablation: zeroing layer 2 head 4 drops P(' red') from {p_full * 100:.1f}% to {p_abl * 100:.1f}%")
            self.sync("Nobody")
            self.flash("nobody programmed this", YELLOW, 36, hold=0.9, pos=[3.2, -2.75, 0])

        # ------------------------------------------------------------ many heads
        hdr6 = self.header("Many heads")
        panels = VGroup()
        info_heads = []
        for l in range(3):
            for h in range(4):
                a = tr["attn"][l][h, -1].numpy()
                j = int(a.argmax())
                box = RoundedRectangle(corner_radius=0.15, width=3.05, height=1.85, fill_color=PANEL, fill_opacity=1,
                                       stroke_color=YELLOW if (l, h) == (1, 3) else LINE, stroke_width=2)
                title = self.T(f"L{l + 1} H{h + 1}", 22, MUTED, BOLD)
                spark = VGroup(*[Rectangle(width=0.1, height=max(0.012, 0.75 * a[k]), stroke_width=0,
                                           fill_color=tcol(toks[k]) if a[k] > 0.08 else MUTED, fill_opacity=0.95)
                                 for k in range(23)]).arrange(RIGHT, buff=0.017, aligned_edge=DOWN)
                pct = self.T(f"{a[j] * 100:.0f}%", 24, tcol(toks[j]), BOLD)
                top = self.M(f"→{self.show_tok(toks[j])} (pos {j})", 22, tcol(toks[j]), t2c={"␣": MUTED})
                title.move_to(box.get_corner(UL) + np.array([0.14 + title.width / 2, -0.24, 0]))
                pct.move_to(box.get_corner(UR) + np.array([-0.14 - pct.width / 2, -0.24, 0]))
                top.move_to(box.get_corner(UL) + np.array([0.14 + top.width / 2, -0.6, 0]))
                spark.move_to(box.get_bottom() + UP * 0.18, aligned_edge=DOWN)
                panels.add(VGroup(box, title, spark, top, pct))
                info_heads.append((l, h, toks[j], j, a[j]))
        panels.arrange_in_grid(rows=3, cols=4, buff=(0.25, 0.25)).move_to(DOWN * 0.2)
        with self.say("Each layer has four heads, and each one learns its own habit. Layer one, head four stares at "
                      "'now'. Layer two, head one looks at 'ball'. Twelve heads, twelve habits.",
                      scene="A 3×4 grid of all 12 heads. Each panel shows that head's real attention from the final ' the' "
                            "over the 23 positions as a mini bar chart, plus its top target (e.g. L1H4 → ' now' 87%, "
                            "L2H1 → ' ball' 60%, L2H4 → ' red' 93%).", chapter="Many heads"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(keys, posn, q_chip, q_lab, hl, info, abl, *wires.values())),
                      ReplacementTransform(hdr5, hdr6), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(p, scale=0.9) for p in panels], lag_ratio=0.07), run_time=1.6)
            for p, (l, h, t, j, aj) in zip(panels, info_heads):
                self.mark(f"head-L{l + 1}H{h + 1}", p,
                          f'layer {l + 1} head {h + 1}: from the final " the", top target "{t}" (pos {j}) = {aj * 100:.1f}%')
            self.sync("Layer one, head four", 0.1)
            self.play(Indicate(panels[3], color=ORANGE, scale_factor=1.08), run_time=0.7)
            self.sync("Layer two, head one", 0.1)
            self.play(Indicate(panels[4], color=YELLOW, scale_factor=1.08), run_time=0.7)

        with self.say("Many heads, many habits, all running in parallel. Go find your favorite head in the X-ray.",
                      scene="The 12-head grid stays up; final card: 'go find your favorite head in the X-ray'."):
            cta = self.T("go find your favorite head in the X-ray →", 40, YELLOW, BOLD)
            plate = RoundedRectangle(corner_radius=0.18, width=cta.width + 0.8, height=cta.height + 0.55,
                                     fill_color=BG, fill_opacity=0.93, stroke_color=YELLOW, stroke_width=3)
            VGroup(plate, cta).move_to(ORIGIN)
            self.play(panels.animate.set_opacity(0.35), run_time=0.5)
            self.sync("Go find", 0.1)
            self.play(FadeIn(VGroup(plate, cta), scale=1.2), run_time=0.4)
            self.mark("cta", VGroup(plate, cta), "open the X-ray to explore every head")
        self.wait(0.5)
