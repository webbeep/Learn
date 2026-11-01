import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from vscene import *  # noqa
from model import VOCAB, META, W, tok, forward, lens

PROMPT = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the"
ROWS = [(0, 6), (6, 12), (12, 19), (19, 23)]
TOKCOL = {" red": PINK, " blue": BLUE, "mia": TEAL, " mia": TEAL, " tom": ORANGE, " ball": YELLOW,
          " kite": "#b48cff", " the": "#5fd0ff"}


def tcol(t):
    return TOKCOL.get(t, MUTED)


def div_color(v, vmax):
    a = float(np.clip(v / vmax, -1, 1))
    return interpolate_color(ManimColor(BG), ManimColor(ORANGE if a > 0 else BLUE), abs(a) ** 0.8)


class GPTIn100(VScene):
    VIDEO_ID = "gpt-in-100s"

    def construct(self):
        ids = tok(PROMPT)
        toks = [VOCAB[i] for i in ids]
        logits, tr = forward(ids, trace=True)
        probs = logits[-1].softmax(-1)
        p_red = float(probs[VOCAB.index(" red")])

        # ---------------------------------------------------------------- hook
        lines_txt = ["mia has a red ball.", "tom has a blue kite.", "mia gives the ball to tom.", "now tom has the"]
        lines = VGroup(*[self.T(s, 44) for s in lines_txt]).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        lines.move_to(UP * 0.4)
        blank = Line(ORIGIN, RIGHT * 1.3, color=YELLOW, stroke_width=5).next_to(lines[3], RIGHT, buff=0.25)
        blank.align_to(lines[3], DOWN).shift(DOWN * 0.06)
        q = self.T("?", 44, YELLOW, BOLD).next_to(blank, UP, buff=0.08)
        with self.say("Now Tom has the... what? This tiny GPT says red. 87 percent sure. Nobody told it the rules.",
                      scene="The prompt 'mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the ___' "
                            "appears line by line; the blank fills with 'red' and an 87% badge.",
                      chapter="The hook"):
            self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.3) for l in lines], lag_ratio=0.25), run_time=1.0)
            self.play(Create(blank), FadeIn(q, scale=1.4), run_time=0.4)
            self.sync("says red")
            ans = self.T("red", 44, PINK, BOLD).next_to(blank, UP, buff=0.08)
            self.play(ReplacementTransform(q, ans), run_time=0.4)
            badge = VGroup(RoundedRectangle(corner_radius=0.12, width=1.9, height=0.66, fill_color=PINK,
                                            fill_opacity=0.18, stroke_color=PINK, stroke_width=2.5),
                           self.T(f"{p_red * 100:.0f}% sure", 28, PINK, BOLD))
            badge[1].move_to(badge[0])
            badge.next_to(ans, RIGHT, buff=0.35)
            self.sync("87")
            self.play(FadeIn(badge, scale=1.3), run_time=0.35)
            self.mark("answer", VGroup(ans, badge), f'model\'s answer " red" with p = {p_red:.3f}')
            self.mark("prompt", lines, "the prompt: 23 tokens of a tiny-world story")

        # ---------------------------------------------------------------- stats
        prm = sum(int(np.prod(t["shape"])) for t in META["tensors"])
        cards = VGroup()
        for big, small, col in [("3", "layers", BLUE), (f"{prm:,}", "parameters", TEAL), ("~80 s", "of training on a CPU", ORANGE)]:
            b = self.T(big, 60, col, BOLD)
            s = self.T(small, 26, MUTED)
            g = VGroup(b, s).arrange(DOWN, buff=0.15)
            box = RoundedRectangle(corner_radius=0.2, width=3.8, height=2.0, fill_color=PANEL, fill_opacity=1,
                                   stroke_color=col, stroke_width=2)
            g.move_to(box)
            cards.add(VGroup(box, g))
        cards.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.6)
        with self.say("Three layers. 170 thousand parameters. Trained for 80 seconds on a CPU. "
                      "That's about a million times smaller than GPT-3. Let's trace exactly how it thinks.",
                      scene="Three stat cards: 3 layers, 170,560 parameters, ~80 s of training on a CPU. "
                            "Punchline flash: about 1,000,000x smaller than GPT-3."):
            top = VGroup(lines, blank, ans, badge)
            self.play(top.animate.scale(0.6).to_edge(UP, buff=0.5), run_time=0.6)
            for i, ph in enumerate(["Three layers", "170 thousand", "Trained for"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(cards[i], shift=UP * 0.3), run_time=0.35)
            self.mark("card-layers", cards[0], "3 transformer layers (4 heads each, width 64)")
            self.mark("card-params", cards[1], f"{prm:,} trainable parameters")
            self.mark("card-train", cards[2], "trained 3000 steps, ~80 seconds on a CPU")
            self.sync("a million")
            self.flash("~1,000,000x smaller than GPT-3", YELLOW, 52, hold=1.1, pos=DOWN * 0.6)

        # ---------------------------------------------------------------- tokens
        def make_rows():
            cs = [self.chip(t, tcol(t), 28, id_=i) for t, i in zip(toks, ids)]
            rs = VGroup(*[VGroup(*cs[a:b]).arrange(RIGHT, buff=0.14, aligned_edge=UP) for a, b in ROWS])
            rs.arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to(DOWN * 0.25)
            return cs, rs

        chips, rows = make_rows()
        hdr = self.header("1 · Tokens")
        with self.say("Step one, chop the text into tokens. Spaces stick to the front of words, and every chunk "
                      "gets an ID number. 23 tokens.",
                      scene="The four prompt lines turn into 23 colored token chips (one row per sentence), each with its "
                            "ID underneath; the leading space shows as ␣.",
                      chapter="Tokens"):
            self.unmark("answer", "prompt", "card-layers", "card-params", "card-train")
            self.play(FadeOut(cards), FadeOut(blank), FadeOut(ans), FadeOut(badge), run_time=0.4)
            self.play(FadeIn(hdr), *[FadeTransform(lines[k], rows[k]) for k in range(4)], run_time=1.2)
            self.sync("Spaces")
            self.play(Indicate(chips[3], color=PINK, scale_factor=1.25), Indicate(chips[1], color=WHITE, scale_factor=1.25),
                      run_time=0.9)
            self.sync("23 tokens")
            cnt = self.T("23 tokens", 30, YELLOW, BOLD).next_to(rows, DOWN, buff=0.5)
            self.play(FadeIn(cnt, shift=UP * 0.2), run_time=0.3)
            for k, (c, t, i) in enumerate(zip(chips, toks, ids)):
                self.mark(f"tok-{k}", c, f'token "{t}" (id {i}), position {k}')

        # ---------------------------------------------------------------- embeddings
        red_k = 3
        e_tok = W["tok.weight"][ids[red_k]].numpy()
        e_pos = W["pos.weight"][red_k].numpy()
        e_sum = e_tok + e_pos
        vmax = float(np.abs(np.stack([e_tok, e_pos, e_sum])).max()) * 0.7

        def strip(vec):
            cells = VGroup(*[Rectangle(width=0.145, height=0.55, stroke_width=0.6, stroke_color=BG,
                                       fill_color=div_color(v, vmax), fill_opacity=1) for v in vec])
            cells.arrange(RIGHT, buff=0)
            return cells

        s_tok, s_pos, s_sum = strip(e_tok), strip(e_pos), strip(e_sum)
        labs = [self.M('"␣red" #157', 24, PINK), self.M("position 3", 24, MUTED), self.M("= input", 24, FG)]
        grid = VGroup()
        for lab, s in zip(labs, [s_tok, s_pos, s_sum]):
            grid.add(VGroup(lab, s))
        for row in grid:
            row[0].next_to(row[1], LEFT, buff=0.35)
        grid.arrange(DOWN, buff=0.45, aligned_edge=RIGHT).move_to(DOWN * 1.1 + RIGHT * 0.9)
        plus = self.T("+", 36, MUTED).next_to(s_pos, LEFT, buff=0.12).shift(LEFT * 2.6)
        dims = self.T("64 numbers", 24, MUTED).next_to(s_tok, UP, buff=0.15).align_to(s_tok, RIGHT)
        hdr2 = self.header("2 · Embeddings")
        with self.say("Each ID grabs a row from a big lookup table: 64 numbers, called an embedding. Add a second "
                      "vector for the position, and every token becomes a point in 64-dimensional space.",
                      scene="The ' red' chip (id 157) is pulled out; below it a strip of 64 colored cells (its real "
                            "embedding: orange positive, blue negative), plus the position-3 vector, equals the input vector.",
                      chapter="Embeddings"):
            self.unmark(*[f"tok-{k}" for k in range(23)])
            red_chip = chips[red_k]
            others = VGroup(*[c for k, c in enumerate(chips) if k != red_k])
            self.play(FadeOut(cnt), FadeOut(others), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(red_chip.animate.scale(1.3).move_to([s_tok.get_x(), 2.3, 0]), run_time=0.6)
            arrow = Arrow(red_chip.get_bottom(), s_tok.get_top() + UP * 0.1, color=PINK, buff=0.12, stroke_width=4)
            self.play(GrowArrow(arrow), FadeIn(labs[0]), FadeIn(dims),
                      LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in s_tok], lag_ratio=0.02), run_time=1.3)
            self.mark("chip-red", red_chip, 'token " red" (id 157) at position 3')
            self.mark("emb-tok", s_tok, f'embedding of " red": 64 learned numbers (range {e_tok.min():.2f} to {e_tok.max():.2f})')
            self.sync("Add a second")
            self.play(FadeIn(plus), FadeIn(labs[1]), LaggedStart(*[FadeIn(c) for c in s_pos], lag_ratio=0.01), run_time=0.9)
            self.mark("emb-pos", s_pos, "position embedding for slot 3 (64 numbers)")
            self.play(FadeIn(labs[2]), TransformFromCopy(VGroup(s_tok, s_pos), s_sum), run_time=0.9)
            self.mark("emb-sum", s_sum, 'input vector for " red" = token embedding + position embedding')

        # ---------------------------------------------------------------- PCA map
        pca = np.array(META["pca"])
        plane_w, plane_h = 7.6, 5.6
        ctr = DOWN * 0.35 + RIGHT * 1.6

        def P(i):
            x, y = pca[i]
            return ctr + RIGHT * (x - 0.03) / 1.7 * plane_w + UP * (y + 0.11) / 1.8 * plane_h

        frame = RoundedRectangle(corner_radius=0.2, width=plane_w + 0.6, height=plane_h + 0.5, stroke_color=LINE,
                                 stroke_width=2, fill_color=PANEL, fill_opacity=0.6).move_to(ctr)
        names = [" tom", " mia", " zoe", " sam", " leo", " ava", " finn", " ruby", " max", " nora", " lily"]
        colors = [" red", " blue", " green", " yellow", " white", " purple", " pink", " black"]
        things = [" ball", " kite", " boat", " cup", " shoe", " key", " drum", " book", " hat"]
        group_of = {}
        for g, lst in (("n", names), ("c", colors), ("t", things)):
            for w in lst:
                group_of[VOCAB.index(w)] = g
        gcol = {"n": TEAL, "c": PINK, "t": YELLOW}
        dots = VGroup()
        hi = {"n": VGroup(), "c": VGroup(), "t": VGroup()}
        for i in range(len(VOCAB)):
            g = group_of.get(i)
            d = Dot(P(i), radius=0.075 if g else 0.035, color=gcol[g] if g else MUTED,
                    fill_opacity=1 if g else 0.45)
            (hi[g] if g else dots).add(d)
        key = VGroup(*[VGroup(Dot(radius=0.09, color=c), self.T(t, 26, c)).arrange(RIGHT, buff=0.18)
                       for t, c in (("names", TEAL), ("colors", PINK), ("things", YELLOW), ("other tokens", MUTED))])
        key.arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.7).shift(DOWN * 0.3)
        title = self.T("all 256 token embeddings, squashed to 2-D (PCA)", 26, MUTED).next_to(frame, UP, buff=0.18)
        lab_red = self.M("␣red", 24, PINK).next_to(P(VOCAB.index(" red")), LEFT, buff=0.15)
        lab_tom = self.M("␣tom", 24, TEAL).next_to(P(VOCAB.index(" tom")), DOWN, buff=0.12)
        with self.say("Squash those 64 numbers down to two, and you can see what it learned. Names land in one "
                      "neighborhood. Colors land in another. Nobody told it that.",
                      scene="A 2-D PCA map of all 256 real token embeddings: names (teal) cluster at the bottom, "
                            "colors (pink) form a tight cluster near the things (yellow); other tokens are faint gray dots."):
            self.unmark("chip-red", "emb-tok", "emb-pos", "emb-sum")
            self.play(*[FadeOut(m) for m in [red_chip, arrow, grid, plus, dims]], run_time=0.5)
            self.play(FadeIn(frame), FadeIn(title), LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.004),
                      run_time=1.2)
            self.mark("pca", frame, "2-D PCA projection of the 256 learned token embeddings")
            self.sync("Names")
            self.play(FadeIn(key[0]), LaggedStart(*[GrowFromCenter(d) for d in hi["n"]], lag_ratio=0.05),
                      FadeIn(lab_tom), run_time=0.8)
            self.mark("cluster-names", hi["n"], "cluster of name tokens (tom, mia, zoe, sam, leo, ...)")
            self.sync("Colors land")
            self.play(FadeIn(key[1]), FadeIn(key[2]), FadeIn(key[3]),
                      LaggedStart(*[GrowFromCenter(d) for d in hi["c"]], lag_ratio=0.05),
                      LaggedStart(*[GrowFromCenter(d) for d in hi["t"]], lag_ratio=0.05), FadeIn(lab_red), run_time=0.9)
            self.mark("cluster-colors", hi["c"], "cluster of color tokens (red, blue, green, yellow, white, ...)")
            self.mark("cluster-things", hi["t"], "object tokens (ball, kite, boat, cup, ...) next to the colors")

        # ---------------------------------------------------------------- attention
        hdr3 = self.header("3 · Attention")
        L, Hh = 1, 3  # layer 2, head 4 (0-indexed)
        a_head = tr["attn"][L][Hh, -1].numpy()
        a_avg = tr["attn"][L][:, -1].mean(0).numpy()
        keys = VGroup()
        for k, t in enumerate(toks):
            txt = self.M(self.show_tok(t), 24, tcol(t), t2c={"␣": MUTED})
            txt.move_to([-0.6, 3.3 - k * 0.31, 0], aligned_edge=RIGHT)
            keys.add(txt)
        posn = VGroup(*[self.M(str(k), 20, MUTED).move_to([-2.6, keys[k].get_y(), 0], aligned_edge=RIGHT)
                        for k in range(23)])
        q_chip = self.chip(" the", tcol(" the"), 34).move_to([3.6, -0.2, 0])
        q_lab = self.T("query: the last token", 24, MUTED).next_to(q_chip, DOWN, buff=0.25)
        col_lab = self.T("earlier tokens", 24, MUTED).next_to(posn, LEFT, buff=0.3).shift(UP * 0.2)

        def wire(k, w, col):
            s_ = q_chip.get_left() + LEFT * 0.08
            e_ = keys[k].get_right() + RIGHT * 0.15
            return Line(s_, e_, stroke_width=1.2 + 10 * w, color=col, stroke_opacity=min(1, 0.3 + 1.5 * w))

        with self.say("Now the tokens talk. In every attention layer, each token looks back at earlier tokens and "
                      "pulls in whatever looks useful.",
                      scene="The 23 tokens slide into a column; on the right, the last token ' the' is the query. Lines fan "
                            "out from it to earlier tokens (layer 2, averaged over its 4 heads; thickness = attention weight).",
                      chapter="Attention"):
            self.unmark("pca", "cluster-names", "cluster-colors", "cluster-things")
            self.play(*[FadeOut(m) for m in [frame, title, dots, hi["n"], hi["c"], hi["t"], key, lab_red, lab_tom]],
                      run_time=0.5)
            self.play(ReplacementTransform(hdr2, hdr3),
                      LaggedStart(*[FadeIn(k_, shift=RIGHT * 0.2) for k_ in keys], lag_ratio=0.03),
                      FadeIn(posn), run_time=1.0)
            self.play(TransformFromCopy(keys[22], q_chip), FadeIn(q_lab), FadeIn(col_lab), run_time=0.7)
            wires_avg = VGroup(*[wire(k, float(a_avg[k]), BLUE) for k in range(23) if a_avg[k] > 0.01])
            self.play(LaggedStart(*[Create(w) for w in wires_avg], lag_ratio=0.06), run_time=1.3)
            self.mark("query", q_chip, 'query token: the last " the" (position 22)')
            for k, (c, t, i) in enumerate(zip(keys, toks, ids)):
                self.mark(f"tok-{k}", c, f'token "{t}" (id {i}), position {k}; layer-2 average attention from last " the": {a_avg[k] * 100:.1f}%')

        with self.say("One head in layer two has a very clear job. From the last 'the', it puts 93 percent of its "
                      "attention on 'red'. It's a color copier.",
                      scene="Only layer 2 head 4 remains: a thick yellow line from the final ' the' to ' red' (position 3) "
                            "labeled 93%, thin lines to '.' (2.4%) and ' ball' (1.9%). Flash: 'a color-copying head'."):
            wires_h = {k: wire(k, float(a_head[k]), YELLOW if k == 3 else BLUE) for k in range(23) if a_head[k] > 0.01}
            hl = self.T("layer 2 · head 4", 30, YELLOW, BOLD).move_to([3.6, 3.3, 0])
            self.play(FadeOut(wires_avg), FadeIn(hl), run_time=0.4)
            self.play(*[Create(w) for w in wires_h.values()], run_time=0.8)
            pct = self.T(f"{a_head[3] * 100:.0f}%", 34, YELLOW, BOLD)
            w3 = wires_h[3]
            d_ = w3.get_end() - w3.get_start(); n_ = np.array([-d_[1], d_[0], 0]); n_ /= np.linalg.norm(n_)
            if n_[1] < 0:
                n_ = -n_
            pct.move_to(w3.point_from_proportion(0.45) + n_ * 0.72)
            self.sync("93")
            self.play(FadeIn(pct, scale=1.4), keys[3].animate.scale(1.35, about_edge=RIGHT), run_time=0.6)
            self.mark("arc-red", VGroup(wires_h[3], pct),
                      f'attention L2H4: " the" (pos 22) → " red" (pos 3) = {a_head[3] * 100:.1f}%')
            for k, w in wires_h.items():
                if k != 3:
                    self.mark(f"arc-{k}", w, f'attention L2H4: " the" → "{toks[k]}" (pos {k}) = {a_head[k] * 100:.1f}%')
            for k, (c, t, i) in enumerate(zip(keys, toks, ids)):
                self.mark(f"tok-{k}", c, f'token "{t}" (id {i}), position {k}; L2H4 attention from last " the": {a_head[k] * 100:.1f}%')
            self.sync("color copier")
            self.flash("a color-copying head", YELLOW, 44, hold=0.8, pos=[3.1, -2.6, 0])

        # ---------------------------------------------------------------- layers + logit lens
        hdr4 = self.header("4 · Layers")
        layer_boxes = VGroup()
        for l in range(3):
            att = RoundedRectangle(corner_radius=0.12, width=2.0, height=0.75, fill_color=BLUE, fill_opacity=0.15,
                                   stroke_color=BLUE, stroke_width=2)
            mlp = RoundedRectangle(corner_radius=0.12, width=2.0, height=0.75, fill_color=TEAL, fill_opacity=0.15,
                                   stroke_color=TEAL, stroke_width=2)
            VGroup(att, mlp).arrange(DOWN, buff=0.22)
            at = self.T("attention", 24, BLUE).move_to(att)
            mt = self.T("MLP", 24, TEAL).move_to(mlp)
            panel = SurroundingRectangle(VGroup(att, mlp), buff=0.22, corner_radius=0.18, stroke_color=LINE,
                                         stroke_width=2, fill_color=PANEL, fill_opacity=1)
            name = self.T(f"layer {l + 1}", 24, MUTED).next_to(panel, UP, buff=0.12)
            layer_boxes.add(VGroup(panel, att, mlp, at, mt, name))
        layer_boxes.arrange(RIGHT, buff=0.8).move_to(UP * 0.55)
        inp = self.M("vectors", 24, MUTED).next_to(layer_boxes, LEFT, buff=0.55)
        out = self.M("scores", 24, MUTED).next_to(layer_boxes, RIGHT, buff=0.55)
        stream = VGroup(Arrow(inp.get_right(), layer_boxes[0][0].get_left(), buff=0.1, color=MUTED, stroke_width=3))
        for l in range(2):
            stream.add(Arrow(layer_boxes[l][0].get_right(), layer_boxes[l + 1][0].get_left(), buff=0.1, color=MUTED,
                             stroke_width=3))
        stream.add(Arrow(layer_boxes[2][0].get_right(), out.get_left(), buff=0.1, color=MUTED, stroke_width=3))
        with self.say("Attention moves information between tokens. Then a small MLP chews on each token by itself. "
                      "Attention plus MLP is one layer. This model stacks three.",
                      scene="Diagram: vectors → three layer panels, each containing an attention box and an MLP box → scores.",
                      chapter="Layers"):
            self.unmark(*[k for k in list(self._marks)])
            self.play(FadeOut(VGroup(keys, posn, q_chip, q_lab, col_lab, *wires_h.values(), pct, hl)), ReplacementTransform(hdr3, hdr4),
                      run_time=0.5)
            self.play(FadeIn(inp), GrowArrow(stream[0]), FadeIn(layer_boxes[0], shift=RIGHT * 0.3), run_time=0.7)
            self.sync("Then a small")
            self.play(Indicate(layer_boxes[0][2], color=TEAL), run_time=0.6)
            self.sync("stacks three")
            self.play(LaggedStart(GrowArrow(stream[1]), FadeIn(layer_boxes[1], shift=RIGHT * 0.3), GrowArrow(stream[2]),
                                  FadeIn(layer_boxes[2], shift=RIGHT * 0.3), GrowArrow(stream[3]), FadeIn(out),
                                  lag_ratio=0.3), run_time=1.2)
            for l in range(3):
                self.mark(f"layer-{l + 1}", layer_boxes[l], f"layer {l + 1}: 4-head attention + MLP (64 → 256 → 64)")

        lens_res = []
        for kind, l, x in tr["resid"]:
            if kind == "mlp":
                p = lens(x)[-1].softmax(-1)
                j = int(p.argmax())
                lens_res.append((VOCAB[j], float(p[j])))
        lens_chips = VGroup()
        for l, (t, p) in enumerate(lens_res):
            col = PINK if t == " red" else MUTED if t == " the" else "#e8e8e8"
            c = self.chip(t, col, 28)
            pt = self.T(f"{p * 100:.0f}%", 28, col, BOLD)
            g = VGroup(c, pt).arrange(DOWN, buff=0.15).next_to(layer_boxes[l], DOWN, buff=0.55)
            lens_chips.add(g)
        lens_lab = self.T("top guess if we stopped after this layer", 24, MUTED).next_to(lens_chips, DOWN, buff=0.35)
        with self.say("Peek at its best guess after each layer. After layer one, it's 'the'. After layer two, 'white'. "
                      "Only the final MLP locks in red, at 87 percent.",
                      scene="Under each layer panel, the model's top guess for the next token if we stopped there "
                            "(logit lens): layer 1 ' the' 33%, layer 2 ' white' 12%, layer 3 ' red' 87%."):
            self.play(FadeIn(lens_lab), run_time=0.3)
            for l, ph in enumerate(["layer one", "layer two", "final MLP"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(lens_chips[l], shift=DOWN * 0.2), run_time=0.4)
                t, p = lens_res[l]
                self.mark(f"lens-{l + 1}", lens_chips[l], f'logit lens after layer {l + 1}: top guess "{t}" at {p * 100:.1f}%')
            self.play(Indicate(lens_chips[2], color=PINK, scale_factor=1.2), run_time=0.6)

        # ---------------------------------------------------------------- probabilities
        hdr5 = self.header("5 · Probabilities")
        top = torch.topk(probs, 5)
        top_t = [VOCAB[int(i)] for i in top.indices]
        top_p = [float(v) for v in top.values]
        rest = 1 - sum(top_p)
        bar_rows = VGroup()
        maxw = 7.5
        for t, p in list(zip(top_t, top_p)) + [("251 others", rest)]:
            col = PINK if t == " red" else (MUTED if t == "251 others" else BLUE)
            lab = self.M(self.show_tok(t) if t != "251 others" else t, 28, col, t2c={"␣": MUTED})
            bar = Rectangle(width=max(0.03, maxw * p), height=0.5, fill_color=col, fill_opacity=0.85, stroke_width=0)
            val = self.T(f"{p * 100:.1f}%" if p >= 0.001 else f"{p * 100:.2f}%", 26, col)
            bar_rows.add(VGroup(lab, bar, val))
        for n, r in enumerate(bar_rows):
            y = 1.3 - n * 0.78
            r[0].move_to([-3.3, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-3.0, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.25)
        sub = self.T("softmax over all 256 tokens", 26, MUTED).next_to(bar_rows, UP, buff=0.4)
        with self.say("The final vector gets scored against all 256 tokens, and softmax turns those scores into "
                      "probabilities. Red gets 87 percent, white 6, purple 5, and everything else gets crumbs.",
                      scene="Horizontal probability bars for the next token: ' red' 87.1%, ' white' 5.9%, ' purple' 5.1%, "
                            "' black' 0.3%, ' ava' 0.2%, all 251 others combined ~1.4%.",
                      chapter="Probabilities"):
            self.unmark(*[k for k in list(self._marks)])
            self.play(FadeOut(VGroup(layer_boxes, stream, inp, out, lens_chips, lens_lab)),
                      ReplacementTransform(hdr4, hdr5), run_time=0.5)
            self.play(FadeIn(sub), LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                                 for r in bar_rows], lag_ratio=0.15), run_time=1.6)
            for t, p, r in zip(top_t + ["others"], top_p + [rest], bar_rows):
                bid = "bar-" + (t.strip() or "nl")
                self.mark(bid, r, f'P("{t}") = {p * 100:.2f}%' if t != "others" else f"all other 251 tokens together = {p * 100:.2f}%")

        # ---------------------------------------------------------------- sample + loop
        hdr6 = self.header("6 · Sample & repeat")
        gen_ids = list(ids)
        gen = []
        for _ in range(8):
            pr = forward(gen_ids)[-1].softmax(-1)
            j = int(pr.argmax())
            gen.append((VOCAB[j], float(pr[j]), j))
            gen_ids.append(j)
        ctx = [self.chip(t, tcol(t), 28) for t in toks[-4:]]
        ell = self.T("…", 34, MUTED)
        row = VGroup(ell, *ctx).arrange(RIGHT, buff=0.14)
        row.move_to(LEFT * 3.5 + UP * 0.6)
        with self.say("Pick a winner, append it, and run the whole thing again. Red. Ball. Period. Newline. "
                      "Once upon a time. One token at a time.",
                      scene="The context '… now tom has the' gets new chips appended one by one (the model's top pick "
                            "each step: ' red', ' ball', '.', newline, 'once', ' upon', ' a', ' time'); the row slides "
                            "left as it grows, with a loop arrow feeding the output back in.",
                      chapter="Sample & repeat"):
            self.unmark(*[k for k in list(self._marks)])
            self.play(FadeOut(VGroup(bar_rows, sub)), ReplacementTransform(hdr5, hdr6), FadeIn(row), run_time=0.6)
            loop = CurvedArrow(RIGHT * 5.6 + DOWN * 0.1, LEFT * 5.2 + DOWN * 0.1, angle=-PI / 3.2, color=MUTED,
                               stroke_width=3)
            loop_lab = self.T("feed it back in", 24, MUTED).next_to(loop, DOWN, buff=0.1)
            self.play(Create(loop), FadeIn(loop_lab), run_time=0.6)
            self.mark("loop", VGroup(loop, loop_lab), "autoregressive loop: the output token is appended and fed back as input")
            phrases = ["Red.", "Ball.", "Period.", "Newline.", "Once upon", "upon a", "a time", "time."]
            for n, (t, p, j) in enumerate(gen):
                c = self.chip(t, tcol(t) if t in TOKCOL else "#e8e8e8", 28)
                pl = self.T(f"{p * 100:.0f}%", 24, MUTED)
                c.next_to(row, RIGHT, buff=0.14)
                pl.next_to(c, UP, buff=0.12)
                if pl.width > c.width + 0.1:
                    pl.set_opacity(0)
                over = c.get_right()[0] - 6.3
                try:
                    self.sync(phrases[n], 0.05)
                except ValueError:
                    pass
                if over > 0:
                    sh = LEFT * over
                    c.shift(sh); pl.shift(sh)
                    movers = list(row) + self._pls
                    offs = [m for m in movers if m.get_left()[0] - over < -6.8]
                    self.play(*[(m.animate.shift(sh).set_opacity(0) if m in offs else m.animate.shift(sh))
                                for m in movers], run_time=0.25)
                    for m in offs:
                        self.remove(m)
                        if m in row:
                            row.remove(m)
                        if m in self._pls:
                            self._pls.remove(m)
                self.play(FadeIn(c, shift=LEFT * 0.3), FadeIn(pl), run_time=0.3)
                row.add(c)
                self._pls.append(pl)
                self.mark(f"gen-{n}", c, f'generated token {n + 1}: "{t}" (id {j}), p = {p * 100:.1f}%')

        # ---------------------------------------------------------------- outro
        steps = ["text", "tokens", "vectors", "attention ×3", "probs", "sample"]
        cols = [FG, ORANGE, TEAL, YELLOW, PINK, BLUE]
        pills = VGroup(*[self.chip(s, c, 30, fill=0.12, font=SANS, pad=0.16) for s, c in zip(steps, cols)]).arrange(RIGHT, buff=0.42)
        pills.move_to(UP * 0.4)
        arrs = VGroup(*[Arrow(pills[i].get_right(), pills[i + 1].get_left(), buff=0.08, color=MUTED, stroke_width=3,
                              max_tip_length_to_length_ratio=0.35) for i in range(len(pills) - 1)])
        back = CurvedArrow(pills[-1].get_bottom() + DOWN * 0.1, pills[0].get_bottom() + DOWN * 0.1, angle=-PI / 5,
                           color=MUTED, stroke_width=3)
        rep = self.T("repeat", 24, MUTED).next_to(back, DOWN, buff=0.1)
        with self.say("That's GPT. Text in, probabilities out, one token at a time. Now go poke it in the X-ray.",
                      scene="Summary pipeline: text → tokens → vectors → attention ×3 → probabilities → sample, with a "
                            "'repeat' arrow looping back. Final card: 'now go poke it in the X-ray'.",
                      chapter="Recap"):
            self.unmark(*[k for k in list(self._marks)])
            self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.5)
            self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in pills], lag_ratio=0.15),
                      LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.15), run_time=1.3)
            self.play(Create(back), FadeIn(rep), run_time=0.6)
            for s, p in zip(steps, pills):
                self.mark("step-" + s.split()[0], p, f"pipeline step: {s}")
            self.sync("Now go poke")
            cta = self.T("now go poke it in the X-ray →", 46, YELLOW, BOLD).move_to(DOWN * 2.4)
            self.play(FadeIn(cta, scale=1.25), run_time=0.4)
            self.mark("cta", cta, "open the X-ray view to inspect the model live")
        self.wait(0.6)

    def setup(self):
        super().setup()
        self._pls = []
