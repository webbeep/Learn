import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from model import VOCAB, tok, forward

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
        src = self.chip(" the", tcol(" the"), 40).move_to(LEFT * 5.0)
        rows = VGroup()
        for name, col, desc, mat in [("query", QC, "what am I looking for?", "W_Q"),
                                     ("key", KC, "what do I contain?", "W_K"),
                                     ("value", VC, "what do I hand over if picked?", "W_V")]:
            a = Arrow(ORIGIN, RIGHT * 1.3 + UP * 0.35, buff=0, color=col, stroke_width=6)
            n = self.T(name, 34, col, BOLD)
            d = self.T(desc, 28, FG)
            m = self.M(f"= x · {mat}", 24, MUTED)
            r = VGroup(a, n, d, m).arrange(RIGHT, buff=0.35)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.75).move_to(RIGHT * 0.9 + DOWN * 0.1)
        links = VGroup(*[Line(src.get_right(), r[0].get_left(), color=LINE, stroke_width=2.5) for r in rows])
        with self.say("Every token turns its vector into three new ones. A query that asks, what am I looking for. "
                      "A key that says, here's what I've got. And a value, the info it hands over if it gets picked.",
                      scene="The token ' the' spawns three arrows: query (blue) 'what am I looking for?', key (teal) "
                            "'what do I contain?', value (orange) 'what do I hand over if picked?' — each = x · a learned "
                            "matrix.", chapter="Query, key, value"):
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
            self.unmark("src", "q", "k", "v")
            self.play(FadeOut(VGroup(src, rows, links)), ReplacementTransform(hdr, hdr2), run_time=0.5)
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
        for n, b in enumerate(bars):
            y = 1.0 - n * 1.0
            b[0].move_to([-3.6, y, 0], aligned_edge=RIGHT)
            b[1].move_to([-3.2, y, 0], aligned_edge=LEFT)
            b[2].move_to([-3.2, y, 0], aligned_edge=LEFT)
            b[3].next_to(b[1], RIGHT, buff=0.25)
            b[4].next_to(b[2], RIGHT, buff=0.25)
        f1 = self.M("score = q · k / √d", 30, FG).move_to(UP * 2.4 + LEFT * 1.0)
        f2 = self.M("weights = softmax(scores)   → positive, sum to 1", 28, FG).move_to(DOWN * 2.3)
        with self.say("Divide by the square root of the vector size, then softmax. Every score becomes a positive "
                      "weight, and the weights add up to one.",
                      scene="Score bars (q·k/√2: ' red' +3.39, ' mia' +0.14, ' ball' −0.57) morph into softmax weight "
                            "bars: ' red' 94.5%, ' mia' 3.7%, ' ball' 1.8%.", chapter="Softmax"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(plane, toy, qv, ql, kvecs, klabs, th)), ReplacementTransform(hdr2, hdr3),
                      *[ReplacementTransform(table[i][0], bars[i][0]) for i in range(3)],
                      *[ReplacementTransform(table[i][1], bars[i][3]) for i in range(3)], run_time=0.8)
            self.play(FadeIn(f1), *[GrowFromEdge(b[1], LEFT) for b in bars], run_time=0.6)
            self.sync("softmax", 0.1)
            self.play(*[ReplacementTransform(b[1], b[2]) for b in bars], *[ReplacementTransform(b[3], b[4]) for b in bars],
                      FadeIn(f2), run_time=1.0)
            for (t, _, _), b, ww in zip(TOY, bars, w):
                self.mark(f"w-{t.strip()}", VGroup(b[0], b[2], b[4]), f'toy attention weight on "{t}" = {ww * 100:.1f}%')
            self.mark("formula", f1, "scaled dot-product score: q·k / √d")

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
        outl = self.T("output", 30, OC, BOLD).next_to(plane2.c2p(*out), LEFT, buff=0.25)
        small = VGroup(*[VGroup(b[0], b[2], b[4]) for b in bars])
        with self.say("The output is the weighted sum of the values. So it's mostly the value of whatever matched best.",
                      scene="Left: the three weights shrink to the side. Right: toy value vectors (orange) for ' red', "
                            "' mia', ' ball'; scaled by their weights and added tip to tail, they give the yellow output "
                            "arrow — almost exactly red's value."):
            self.play(FadeOut(f1), FadeOut(f2), small.animate.scale(0.62).to_edge(LEFT, buff=0.5).shift(UP * 0.2),
                      run_time=0.6)
            self.play(FadeIn(plane2), LaggedStart(*[GrowArrow(v) for v in vvecs], lag_ratio=0.2), FadeIn(vlabs),
                      run_time=0.9)
            self.mark("values", VGroup(vvecs, vlabs), "toy value vectors v for ' red', ' mia', ' ball'")
            self.sync("weighted sum", 0.1)
            self.play(vvecs.animate.set_opacity(0.25), vlabs[0].animate.set_opacity(0), LaggedStart(*[GrowArrow(c) for c in chain], lag_ratio=0.5),
                      run_time=1.0)
            self.play(GrowArrow(outv), FadeIn(outl), run_time=0.5)
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
        tri = Polygon(cells[1].get_corner(UL), cells[n - 1].get_corner(UR), cells[(n - 2) * n + n - 1].get_corner(DR),
                      stroke_color=PINK, stroke_width=3, fill_opacity=0)
        future = self.T("the future: blocked", 28, PINK, BOLD).move_to(cells.get_center() + UR * 1.25)
        rowlab = self.T("each row = one token looking back", 24, FG)
        collab = self.T("each column = a token being looked at", 24, FG)
        ex = self.T("real weights · layer 2, head 4", 24, MUTED)
        side = VGroup(rowlab, collab, ex).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(LEFT, buff=0.55)
        side.shift(DOWN * 0.2)
        last_row = SurroundingRectangle(VGroup(cells[(n - 1) * n], cells[n * n - 1]), color=BLUE, buff=0.02)
        red_cell = SurroundingRectangle(cells[(n - 1) * n + 3], color=YELLOW, buff=0.03, stroke_width=4)
        with self.say("One rule. No peeking at the future. A causal mask blocks every token from looking at anything "
                      "after it, which keeps next-token training honest.",
                      scene="The real 23×23 attention matrix of layer 2 head 4 on the prompt: lower triangle colored by "
                            "weight, the upper triangle (future positions) blocked in dark red and outlined.",
                      chapter="Causal mask"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(plane2, vvecs, vlabs, chain, outv, outl, small)), ReplacementTransform(hdr3, hdr4),
                      run_time=0.5)
            self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.002), FadeIn(side[0]), FadeIn(side[1]),
                      run_time=1.4)
            self.sync("A causal mask", 0.1)
            self.play(Create(tri), FadeIn(future, scale=1.2), run_time=0.7)
            self.play(FadeIn(ex), Create(last_row), Create(red_cell), run_time=0.6)
            self.mark("matrix", cells, "real attention matrix, layer 2 head 4 (23 × 23)")
            self.mark("mask", VGroup(tri, future), "causal mask: positions after the query get weight 0 (score −∞)")
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

        def wire(k, ww, col):
            return Line(q_chip.get_left() + LEFT * 0.08, keys[k].get_right() + RIGHT * 0.15, color=col,
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
                      scene="All 23 prompt tokens in a column; lines fan from the final ' the' (query) to earlier tokens. "
                            "The line to ' red' (position 3) is thick and yellow: score 4.28 → 93%. Others: '.' 2.4%, "
                            "' ball' 1.9%.", chapter="A real head"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(cells, tri, future, side, last_row, red_cell)), ReplacementTransform(hdr4, hdr5),
                      run_time=0.5)
            self.play(LaggedStart(*[FadeIn(k_, shift=RIGHT * 0.2) for k_ in keys], lag_ratio=0.03), FadeIn(posn),
                      FadeIn(q_chip), FadeIn(q_lab), FadeIn(hl), run_time=1.0)
            self.play(LaggedStart(*[Create(x) for x in wires.values()], lag_ratio=0.1), run_time=1.0)
            self.sync("93", 0.2)
            self.play(FadeIn(info, scale=1.3), keys[3].animate.scale(1.35, about_edge=RIGHT), run_time=0.6)
            self.mark("query", q_chip, 'query: the final " the" (position 22)')
            self.mark("arc-red", VGroup(wires[3], info),
                      f'L2H4: " the" → " red" (pos 3): score {s_head[3]:.2f}, weight {a_head[3] * 100:.1f}%')
            for k, x in wires.items():
                if k != 3:
                    self.mark(f"arc-{k}", x, f'L2H4: " the" → "{toks[k]}" (pos {k}): weight {a_head[k] * 100:.1f}%')
            for k, t in enumerate(toks):
                self.mark(f"tok-{k}", keys[k], f'token "{t}" (id {ids[k]}), position {k}: weight {a_head[k] * 100:.1f}%')

        pulse = Dot(radius=0.11, color=VC).move_to(w3.get_end())
        with self.say("So red's value gets copied forward, right to the spot where the model has to name a color. "
                      "Nobody programmed that. It fell out of training.",
                      scene="An orange 'value' dot travels from ' red' along the yellow line into the final ' the'. "
                            "Flash: 'nobody programmed this'."):
            self.play(FadeIn(pulse, scale=2), run_time=0.3)
            self.play(MoveAlongPath(pulse, Line(w3.get_end(), w3.get_start())), run_time=1.1)
            self.play(Flash(q_chip, color=VC, line_length=0.3), FadeOut(pulse), q_chip[0].animate.set_stroke(VC),
                      run_time=0.6)
            self.sync("Nobody")
            self.flash("nobody programmed this", YELLOW, 46, hold=0.9, pos=[3.3, -2.7, 0])

        # ------------------------------------------------------------ many heads
        hdr6 = self.header("Many heads")
        panels = VGroup()
        info_heads = []
        for l in range(3):
            for h in range(4):
                a = tr["attn"][l][h, -1].numpy()
                j = int(a.argmax())
                box = RoundedRectangle(corner_radius=0.15, width=3.05, height=1.62, fill_color=PANEL, fill_opacity=1,
                                       stroke_color=YELLOW if (l, h) == (1, 3) else LINE, stroke_width=2)
                title = self.T(f"L{l + 1} H{h + 1}", 22, MUTED, BOLD)
                spark = VGroup(*[Rectangle(width=0.1, height=max(0.012, 0.75 * a[k]), stroke_width=0,
                                           fill_color=tcol(toks[k]) if a[k] > 0.08 else MUTED, fill_opacity=0.95)
                                 for k in range(23)]).arrange(RIGHT, buff=0.017, aligned_edge=DOWN)
                top = self.M(f"→{self.show_tok(toks[j])} {a[j] * 100:.0f}%", 22, tcol(toks[j]), t2c={"␣": MUTED})
                title.move_to(box.get_corner(UL) + np.array([0.12 + title.width / 2, -0.22, 0]))
                top.move_to(box.get_corner(UR) + np.array([-0.12 - top.width / 2, -0.22, 0]))
                spark.move_to(box.get_bottom() + UP * 0.5, aligned_edge=DOWN)
                panels.add(VGroup(box, title, spark, top))
                info_heads.append((l, h, toks[j], j, a[j]))
        panels.arrange_in_grid(rows=3, cols=4, buff=(0.25, 0.3)).move_to(DOWN * 0.15)
        with self.say("Each layer has four heads, and each one learns its own habit. Layer one, head four stares at "
                      "'now'. Layer two, head one looks at 'ball'. Twelve heads, twelve habits.",
                      scene="A 3×4 grid of all 12 heads. Each panel shows that head's real attention from the final ' the' "
                            "over the 23 positions as a mini bar chart, plus its top target (e.g. L1H4 → ' now' 87%, "
                            "L2H1 → ' ball' 60%, L2H4 → ' red' 93%).", chapter="Many heads"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(keys, posn, q_chip, q_lab, hl, info, *wires.values())),
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
