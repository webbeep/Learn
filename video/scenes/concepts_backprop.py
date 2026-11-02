import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from vscene import *  # noqa
from lib_concepts_viz import CScene
from model import W
from lib_concepts import toy_graph, backprop_data, TOY

VIDEO = dict(id="backprop", cls="Backprop", title="Backprop: blame flows backwards",
             subtitle="The chain rule on a toy graph, then TinyGPT's real gradients for one prediction",
             relates="backprop", track="training", order=30, poster=("The chain rule", 2, 0.95))

FWD, BWD = TEAL, ORANGE
COL = {" red": PINK, " white": "#e8e8e8", " purple": "#b48cff", " black": "#7d8597", " ava": TEAL, ",": ORANGE}


def pct(p):
    return f"{100 * p:.1f}%"


class Backprop(CScene):
    VIDEO_ID = "backprop"

    # ---------------------------------------------------------------- small builders
    def node(self, sym, r=0.42, col=FG):
        c = Circle(radius=r, stroke_color=col, stroke_width=3, fill_color=PANEL, fill_opacity=1)
        t = self.M(sym, 34, col).move_to(c)
        return VGroup(c, t)

    def box(self, s, col=FG, size=30, pad=0.2):
        t = self.M(s, size, col)
        b = RoundedRectangle(corner_radius=0.12, width=t.width + 2 * pad, height=t.height + 2 * pad,
                             stroke_color=col, stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
        t.move_to(b)
        return VGroup(b, t)

    def construct(self):
        tg = toy_graph()
        bd = backprop_data()
        wt = bd["weight"]

        # ============================================================ hook
        ctx = VGroup(*[self.chip(t, BLUE if t != " red" else PINK, 30) for t in bd["toks"][-4:]]).arrange(RIGHT, buff=0.12)
        dots = self.T("…", 36, MUTED)
        guess = self.chip(" red", PINK, 34)
        p_t = self.T(pct(bd["p_red"]), 40, PINK, BOLD)
        row = VGroup(dots, ctx, Arrow(ORIGIN, RIGHT * 0.9, color=MUTED, stroke_width=4), guess, p_t).arrange(RIGHT, buff=0.3)
        row.move_to(UP * 1.6)
        loss_t = VGroup(self.T("loss", 30, MUTED), self.M(f"{bd['loss']:.3f}", 36, YELLOW)).arrange(RIGHT, buff=0.3)
        loss_t.next_to(row, DOWN, buff=0.6)
        # a wall of dots standing in for the weights
        wall = VGroup(*[Dot(radius=0.035, color=BLUE, fill_opacity=0.55) for _ in range(40 * 9)])
        wall.arrange_in_grid(rows=6, cols=60, buff=0.12).move_to(DOWN * 1.55)
        wl = VGroup(self.T(f"{bd['n_params']:,} weights: which ones are to blame?", 30, FG),
                    self.T(f"(each dot stands in for ~{bd['n_params'] // len(wall)} weights)", 22, MUTED)
                    ).arrange(DOWN, buff=0.1).next_to(wall, DOWN, buff=0.25)
        with self.say(f"TinyGPT guessed red at 87 percent. "
                      f"Which of its {bd['n_params']:,} weights deserve the blame, and how much? "
                      f"Backprop answers for all of them, in one sweep backwards.",
                      scene=f"The end of the probe prompt '… tom has the' → ' red' at {pct(bd['p_red'])}, loss "
                            f"{bd['loss']:.3f}; below, a grid of dots standing for the model's {bd['n_params']:,} weights "
                            f"with the question 'which ones are to blame?'.", chapter="Who's to blame?"):
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.8)
            self.mark("guess", VGroup(guess, p_t), f'TinyGPT predicts " red" with p = {bd["p_red"]:.4f}')
            self.play(FadeIn(loss_t, shift=UP * 0.2), run_time=0.5)
            self.mark("loss", loss_t, f"cross-entropy loss = −ln({bd['p_red']:.4f}) = {bd['loss']:.4f}")
            self.sync("Which of", 0.2)
            self.play(LaggedStart(*[FadeIn(d, scale=2) for d in wall], lag_ratio=0.002), FadeIn(wl), run_time=1.2)
            self.mark("weights", VGroup(wall, wl), f"TinyGPT has {bd['n_params']:,} trainable weights (dots are a stand-in)")
            self.sync("one sweep", 0.1)
            self.play(LaggedStart(*[d.animate.set_color(ORANGE) for d in reversed(wall)], lag_ratio=0.002), run_time=1.0)

        # ============================================================ toy graph: forward
        hdr = self.header("The chain rule · toy example")
        CY = 0.9
        P = dict(x=np.array([-5.6, 2.0, 0]), w=np.array([-5.6, -0.2, 0]), mul=np.array([-3.6, CY, 0]),
                 b=np.array([-1.9, -1.75, 0]), add=np.array([-0.6, CY, 0]), y=np.array([3.0, -0.9, 0]),
                 sq=np.array([3.0, CY, 0]), L=np.array([5.5, CY, 0]))
        nx = self.box(f"x = {TOY['x']:g}").move_to(P["x"])
        nw = self.box(f"w = {TOY['w']:g}").move_to(P["w"])
        nb = self.box(f"b = {TOY['b']:g}").move_to(P["b"])
        ny = self.box(f"target t = {TOY['y']:g}", MUTED, 26).move_to(P["y"])
        nmul = self.node("×").move_to(P["mul"])
        nadd = self.node("+").move_to(P["add"])
        nsq = self.box("(z − t)²", FG, 28).move_to(P["sq"])
        nL = self.box(f"L = {tg['L']:g}", YELLOW).move_to(P["L"])

        def edge(start, node):
            end = node.get_center() + 0.47 * normalize(start - node.get_center())
            return Arrow(start, end, buff=0, color=LINE, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        e_x, e_w = edge(nx.get_right() + RIGHT * 0.05, nmul), edge(nw.get_right() + RIGHT * 0.05, nmul)
        e_a = Arrow(nmul.get_right(), nadd.get_left(), buff=0.08, color=LINE, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        e_b = edge(nb.get_top() + UP * 0.05 + RIGHT * 0.25, nadd)
        e_z = Arrow(nadd.get_right(), nsq.get_left(), buff=0.08, color=LINE, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        e_y = Arrow(ny.get_top(), nsq.get_bottom(), buff=0.06, color=LINE, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        e_L = Arrow(nsq.get_right(), nL.get_left(), buff=0.08, color=LINE, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        va = self.M(f"a = {tg['a']:g}", 28, FWD).next_to(e_a.get_center(), UP, buff=0.2)
        vz = self.M(f"z = {tg['z']:g}", 28, FWD).next_to(e_z.get_center(), UP, buff=0.2)
        toy_note = self.T("toy numbers, not TinyGPT", 24, MUTED).to_corner(UR, buff=0.45)
        graph = VGroup(nx, nw, nb, ny, nmul, nadd, nsq, nL, e_x, e_w, e_a, e_b, e_z, e_y, e_L)
        with self.say(f"A toy version first. Forward: two times three is six, plus one is seven. "
                      f"That misses the target five by two. Squared, the loss is four.",
                      scene="Toy computational graph (labelled toy): x = 2 and w = 3 feed a × node giving a = 6; a and b = 1 "
                            "feed a + node giving z = 7; (z − t)² with target t = 5 gives loss L = 4. Forward values in teal.",
                      chapter="The chain rule"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(row, loss_t, wall, wl)), FadeIn(hdr), FadeIn(toy_note), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(m) for m in (nx, nw, nb, nmul, nadd, ny, nsq)], lag_ratio=0.08),
                      LaggedStart(*[GrowArrow(e) for e in (e_x, e_w, e_b, e_a, e_y, e_z, e_L)], lag_ratio=0.08), run_time=1.2)
            for k, m in (("x", nx), ("w", nw), ("b", nb)):
                self.mark(f"in-{k}", m, f"toy input {k} = {TOY[k]:g}")
            self.mark("node-mul", nmul, "multiply node: a = w × x")
            self.mark("node-add", nadd, "add node: z = a + b")
            self.mark("node-sq", nsq, "squared error node: L = (z − t)², target t = 5")
            self.sync("two times", 0.1)
            self.play(e_x.animate.set_color(FWD), e_w.animate.set_color(FWD), Indicate(nmul, color=FWD), run_time=0.5)
            self.play(e_a.animate.set_color(FWD), FadeIn(va, shift=RIGHT * 0.2), run_time=0.4)
            self.mark("val-a", va, "forward value a = w·x = 6")
            self.sync("Plus one", 0.1)
            self.play(e_b.animate.set_color(FWD), Indicate(nadd, color=FWD), run_time=0.4)
            self.play(e_z.animate.set_color(FWD), FadeIn(vz, shift=RIGHT * 0.2), run_time=0.4)
            self.mark("val-z", vz, "forward value z = a + b = 7")
            self.sync("That misses", 0.1)
            self.play(e_y.animate.set_color(FWD), Indicate(nsq, color=FWD), run_time=0.5)
            self.sync("Squared", 0.1)
            self.play(e_L.animate.set_color(FWD), FadeIn(nL, shift=RIGHT * 0.2), run_time=0.5)
            self.mark("loss-L", nL, "toy loss L = (7 − 5)² = 4")

        # ============================================================ toy graph: backward
        def grad_lab(s, near, direction=DOWN, buff=0.2):
            return self.M(s, 26, BWD).next_to(near, direction, buff=buff)
        gz = grad_lab(f"∂L/∂z = {tg['dz']:g}", e_z.get_center(), DOWN, 0.22)
        ga = grad_lab(f"∂L/∂a = {tg['da']:g}", e_a.get_center(), DOWN, 0.22)
        gb = grad_lab(f"∂L/∂b = {tg['db']:g}", nb, DOWN, 0.2)
        gw = grad_lab(f"∂L/∂w = {tg['da']:g} × {TOY['x']:g} = {tg['dw']:g}", nw, DOWN, 0.2).align_to(nw, LEFT).shift(LEFT * 0.4)
        gx = grad_lab(f"∂L/∂x = {tg['da']:g} × {TOY['w']:g} = {tg['dx']:g}", nx, UP, 0.2).align_to(nx, LEFT).shift(LEFT * 0.4)
        rule1 = self.M(f"= 2 × ({tg['z']:g} − {TOY['y']:g})", 24, BWD).next_to(gz, DOWN, buff=0.1).align_to(gz, LEFT)

        def rev(e):   # the same edge, pointing backwards, in the blame color
            return Arrow(e.get_end(), e.get_start(), buff=0, color=BWD, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        r_x, r_w, r_a, r_b, r_z = rev(e_x), rev(e_w), rev(e_a), rev(e_b), rev(e_z)
        with self.say('Now backwards. The slope at z is twice the miss: four. A plus node passes blame straight through: a and b get four. A times node scales it by the other input: w gets eight, x gets twelve.',
                      scene="Backward pass on the toy graph, arrows reversed, gradients in orange flowing right-to-left: ∂L/∂z = 2(z − 5) = 4; "
                            "the + node copies it: ∂L/∂a = 4, ∂L/∂b = 4; the × node multiplies by the other input: "
                            "∂L/∂w = 4 × 2 = 8, ∂L/∂x = 4 × 3 = 12."):
            self.play(*[e.animate.set_color(LINE) for e in (e_x, e_w, e_a, e_b, e_z, e_y, e_L)], run_time=0.3)
            self.sync("slope", 0.2)
            self.play(ReplacementTransform(e_z, r_z), FadeIn(gz, shift=LEFT * 0.2), FadeIn(rule1), run_time=0.5)
            self.mark("g-z", VGroup(gz, rule1), "gradient ∂L/∂z = 2(z − target) = 4")
            self.sync("plus node", 0.1)
            self.play(Indicate(nadd, color=BWD), run_time=0.4)
            self.play(ReplacementTransform(e_a, r_a), ReplacementTransform(e_b, r_b), FadeIn(ga, shift=LEFT * 0.2),
                      FadeIn(gb, shift=LEFT * 0.2), run_time=0.5)
            self.mark("g-a", ga, "∂L/∂a = 4 (a + node copies the incoming gradient)")
            self.mark("g-b", gb, "∂L/∂b = 4")
            self.sync("times node", 0.1)
            self.play(Indicate(nmul, color=BWD), run_time=0.4)
            self.sync("w gets", 0.1)
            self.play(ReplacementTransform(e_w, r_w), FadeIn(gw, shift=LEFT * 0.2), run_time=0.5)
            self.mark("g-w", gw, "∂L/∂w = ∂L/∂a × x = 4 × 2 = 8")
            self.sync("x gets", 0.1)
            self.play(ReplacementTransform(e_x, r_x), FadeIn(gx, shift=LEFT * 0.2), run_time=0.5)
            self.mark("g-x", gx, "∂L/∂x = ∂L/∂a × w = 4 × 3 = 12")

        # ============================================================ finite-difference check
        eps = tg["eps"]
        chk = VGroup(self.T("check by brute force:", 26, MUTED),
                     self.M(f"(L(w+ε) − L(w−ε)) / 2ε = {tg['fd']['w']:.4f}", 28, YELLOW)).arrange(RIGHT, buff=0.3)
        chk.to_edge(DOWN, buff=0.5)
        rule = self.T("blame in  =  blame from above  ×  local slope", 34, BWD, BOLD).move_to(chk)
        with self.say(f"Brute-force check: nudge w by a hair, and the loss moves eight times as much. "
                      f"Each node just multiplies the blame from above by its own local slope.",
                      scene=f"Finite-difference check of the toy gradient: (L(w+ε) − L(w−ε))/2ε with ε = {eps:g} gives "
                            f"{tg['fd']['w']:.4f}, matching ∂L/∂w = 8. Rule shown on top: blame in = blame from above × "
                            f"local slope."):
            self.play(FadeIn(chk, shift=UP * 0.2), Indicate(gw, color=YELLOW), run_time=0.8)
            self.mark("fd", chk, f"finite difference with ε = {eps:g}: {tg['fd']['w']:.6f} ≈ 8 (x: {tg['fd']['x']:.4f}, "
                                 f"b: {tg['fd']['b']:.4f})")
            self.sync("Each node", 0.1)
            self.play(FadeOut(chk), FadeIn(rule, shift=UP * 0.2), run_time=0.5)
            self.unmark("fd")
            self.mark("rule", rule, "chain rule: gradient = upstream gradient × local derivative")

        # ============================================================ real: dLoss/dlogit
        hdr2 = self.header("Real gradients · TinyGPT")
        top = bd["top"][:5]
        ys = [1.45 - 0.7 * k for k in range(len(top))]
        X0, SCALE = 1.0, 16.0
        axis = Line([X0, 1.95, 0], [X0, ys[-1] - 0.4, 0], color=MUTED, stroke_width=2)
        bars = VGroup()
        for k, (t, p, g) in enumerate(top):
            c = COL.get(t, MUTED)
            lab = self.M(self.show_tok(t), 30, c, t2c={"␣": MUTED}).move_to([-5.3, ys[k], 0], aligned_edge=RIGHT)
            pp = self.M(f"p = {pct(p)}", 24, MUTED).move_to([-3.05, ys[k], 0], aligned_edge=RIGHT)
            w_ = max(0.03, abs(g) * SCALE)
            bar = Rectangle(width=w_, height=0.44, stroke_width=0, fill_color=TEAL if g < 0 else BWD, fill_opacity=0.9)
            bar.move_to([X0, ys[k], 0], aligned_edge=RIGHT if g < 0 else LEFT)
            val = self.M(f"{g:+.3f}", 26, TEAL if g < 0 else BWD)
            if g < 0:
                val.next_to(bar, LEFT, buff=0.15)
            else:
                val.next_to(bar, RIGHT, buff=0.15)
            bars.add(VGroup(lab, pp, bar, val))
        form = self.M("∂loss/∂logit  =  p − (1 for the right token, else 0)", 28, FG).move_to(UP * 2.9)
        rest = self.T(f"the other {256 - len(top)} tokens: each between 0 and +{bd['g_rest_max']:.4f}", 24, MUTED)
        rest.move_to([0, ys[-1] - 0.95, 0])
        up_l = self.T("← push up", 24, TEAL).next_to(axis.get_top(), UP, buff=0.12).shift(LEFT * 1.25)
        dn_l = self.T("push down →", 24, BWD).next_to(axis.get_top(), UP, buff=0.12).shift(RIGHT * 1.4)
        with self.say(f"Now the real thing. Each output logit's blame is its probability, minus one for the right "
                      f"answer. Red: minus {abs(top[0][2]):.2f}, push up. White and purple: plus, push down.",
                      scene=f"Real gradient of the loss w.r.t. TinyGPT's output logits on the probe prompt (target ' red'): "
                            f"∂loss/∂logit = p − onehot. ' red' {top[0][2]:+.3f} (teal, push up), ' white' {top[1][2]:+.3f}, "
                            f"' purple' {top[2][2]:+.3f}, ' black' {top[3][2]:+.4f}, ' ava' {top[4][2]:+.4f} (orange, push "
                            f"down); the other 251 are each between 0 and +{bd['g_rest_max']:.4f}.",
                      chapter="Real gradients"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(*[m for m in self.mobjects if m is not hdr])), ReplacementTransform(hdr, hdr2),
                      run_time=0.5)
            self.play(FadeIn(form), Create(axis), run_time=0.6)
            self.mark("formula", form, "gradient of cross-entropy w.r.t. logits = softmax probability − one-hot target")
            self.play(LaggedStart(*[AnimationGroup(FadeIn(b[0]), FadeIn(b[1]), GrowFromEdge(b[2], RIGHT if top[k][2] < 0 else LEFT),
                                                   FadeIn(b[3])) for k, b in enumerate(bars)], lag_ratio=0.15),
                      FadeIn(up_l), FadeIn(dn_l), run_time=1.4)
            for (t, p, g), b in zip(top, bars):
                self.mark(f"glogit-{t.strip() or 'comma'}", b, f'"{t}": p = {p:.4f}, ∂loss/∂logit = {g:+.4f}')
            self.play(FadeIn(rest), run_time=0.4)
            self.mark("rest", rest, f"the remaining {256 - len(top)} logits each get a small positive gradient "
                                    f"(max {bd['g_rest_max']:.4f}); all 256 sum to {bd['g_sum']:.1e} ≈ 0")
            self.sync("Red:", 0.1)
            self.play(Indicate(bars[0], color=TEAL, scale_factor=1.06), run_time=0.6)
            self.sync("White and", 0.1)
            self.play(Indicate(bars[1], color=BWD, scale_factor=1.06), Indicate(bars[2], color=BWD, scale_factor=1.06),
                      run_time=0.6)

        # ============================================================ real: gradient norms by layer
        order = ["final LayerNorm", "layer 3 MLP", "layer 3 attention", "layer 2 MLP", "layer 2 attention", "layer 1 MLP",
                 "layer 1 attention", "position embedding", "token embedding (tied)"]
        nm = bd["norms"]
        gmax = max(nm.values())
        ys2 = [2.35 - 0.56 * k for k in range(len(order))]
        rows = VGroup()
        for k, key in enumerate(order):
            c = BLUE if "attention" in key else (TEAL if "MLP" in key else MUTED)
            lab = self.T(key.replace("token embedding (tied)", "token emb. (tied: in + out)"), 26, c).move_to([-1.0, ys2[k], 0], aligned_edge=RIGHT)
            bar = Rectangle(width=max(0.04, 5.0 * nm[key] / gmax), height=0.36, stroke_width=0, fill_color=BWD,
                            fill_opacity=0.85).move_to([-0.6, ys2[k], 0], aligned_edge=LEFT)
            val = self.M(f"{nm[key]:.2f}", 24, BWD).next_to(bar, RIGHT, buff=0.15)
            rows.add(VGroup(lab, bar, val))
        flow = Arrow([-6.2, ys2[0] + 0.2, 0], [-6.2, ys2[-1] - 0.2, 0], buff=0, color=BWD, stroke_width=6,
                     max_tip_length_to_length_ratio=0.05)
        flow_l = self.T("blame flows down", 24, BWD).rotate(PI / 2).next_to(flow, RIGHT, buff=0.12)
        cap2 = self.T("gradient size (L2 norm) per part · one prediction", 24, MUTED).to_corner(UR, buff=0.45)
        merge = self.T("where paths merge (the residual stream skips past every block), blames add", 22, MUTED)
        merge.next_to(cap2, DOWN, buff=0.12).align_to(cap2, RIGHT)
        nz = VGroup(self.T(f"{bd['n_nonzero']:,} of {bd['n_params']:,} weights get some blame", 30, YELLOW, BOLD),
                    self.T(f"exactly zero: {bd['n_unused_pos']:,} unused position weights, {bd['n_keybias']} attention key "
                           f"biases", 22, MUTED),
                    self.T("(a key bias adds the same number to every score, and softmax ignores that)", 22, MUTED)
                    ).arrange(DOWN, buff=0.08)
        self.fitw(nz, 12.8).to_edge(DOWN, buff=0.25)
        with self.say('That blame flows down through all three layers. Where paths merge, like the residual stream, blames add up. From this one prediction, 167,744 weights get some blame.',
                      scene="Real gradient norms per part of TinyGPT for this one prediction, top (output) to bottom (input): "
                            + ", ".join(f"{k} {nm[k]:.2f}" for k in order)
                            + f". {bd['n_nonzero']:,} of {bd['n_params']:,} weights have a nonzero gradient; the {bd['n_keybias']} attention key "
                              f"biases and {bd['n_unused_pos']:,} unused position weights get exactly zero."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(bars, axis, form, rest, up_l, dn_l)), run_time=0.4)
            self.play(GrowArrow(flow), FadeIn(flow_l), FadeIn(cap2), run_time=0.5)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2])) for r in rows],
                                  lag_ratio=0.18), run_time=2.4)
            for key, r in zip(order, rows):
                self.mark("norm-" + key.replace(" ", "-").replace("(", "").replace(")", ""), r,
                          f"gradient norm of {key}: {nm[key]:.3f}")
            self.sync("Where paths", 0.1)
            self.play(FadeIn(merge), run_time=0.4)
            self.mark("merge", merge, "gradients from different paths (e.g. a block and the residual skip around it) are summed")
            self.sync(f"{bd['n_nonzero']:,}"[:3], 0.3)
            self.play(FadeIn(nz, scale=1.2), run_time=0.5)
            self.mark("nonzero", nz, f"{bd['n_nonzero']:,} of {bd['n_params']:,} weights have nonzero gradient here "
                                     f"(positions 23–63 are unused; the {bd['n_keybias']} key biases get exactly 0 because softmax is shift-invariant)")

        # ============================================================ one weight
        hdr3 = self.header("One step · toy: plain gradient descent on ONE prediction")
        # the gradient matrix of layer 3's MLP output weights (64 × 256), rendered as an image
        from lib_concepts import load_tiny, PROBE
        import torch.nn.functional as F
        from model import tok, VOCAB
        m = load_tiny(torch.float64)
        lg = m(torch.tensor([tok(PROBE)]))[0, -1]
        F.cross_entropy(lg[None], torch.tensor([VOCAB.index(" red")])).backward()
        G = dict(m.named_parameters())[wt["name"]].grad.numpy()
        gm = np.abs(G).max()
        rgb = np.zeros(G.shape + (3,), dtype=np.uint8)
        bg = np.array([21, 26, 40]); oc = np.array([255, 150, 64]); tc = np.array([72, 214, 176])
        a = np.clip(np.abs(G) / gm, 0, 1) ** 0.45
        for s_, cc in ((G > 0, oc), (G <= 0, tc)):
            rgb[s_] = (bg + (cc - bg) * a[s_][:, None]).astype(np.uint8)
        img = ImageMobject(rgb).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.width = 6.3
        img.stretch_to_fit_height(2.6)
        img.move_to([-3.55, -0.1, 0])
        frame = SurroundingRectangle(img, buff=0.02, color=LINE, stroke_width=2)
        cell_w = img.width / G.shape[1]; cell_h = img.height / G.shape[0]
        cpos = img.get_corner(UL) + RIGHT * (wt["j"] + 0.5) * cell_w + DOWN * (wt["i"] + 0.5) * cell_h
        ring = Circle(radius=0.16, color=YELLOW, stroke_width=4).move_to(cpos)
        cap3 = self.T("real ∂loss/∂W, layer 3 MLP (64 × 256)", 24, MUTED).next_to(frame, UP, buff=0.18)
        cap3.align_to(frame, LEFT)
        leg = VGroup(self.T("teal: gradient < 0, the step raises the weight", 22, TEAL),
                     self.T("orange: gradient > 0, the step lowers it", 22, ORANGE)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        leg.next_to(frame, DOWN, buff=0.18).align_to(frame, LEFT)
        info = VGroup(
            self.T(f"row {wt['i']}, column {wt['j']}", 26, MUTED),
            VGroup(self.T("weight", 26, FG), self.M(f"{wt['w']:.4f}", 32, BLUE)).arrange(RIGHT, buff=0.25),
            VGroup(self.T("gradient", 26, FG), self.M(f"{wt['g']:+.4f}", 32, TEAL)).arrange(RIGHT, buff=0.25),
            self.T(f"finite-difference check: {wt['fd']:+.4f}", 24, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        upd = VGroup(self.M(f"step = −{wt['lr']:g} × ({wt['g']:+.4f})", 26, FG),
                     self.M(f"     = {wt['step']:+.5f}", 26, YELLOW),
                     VGroup(self.T("new weight", 26, FG), self.M(f"{wt['w_new']:.4f}", 32, YELLOW)).arrange(RIGHT, buff=0.25)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(info, DOWN, buff=0.55).align_to(info, LEFT)
        VGroup(info, upd).move_to([0.75, -0.1, 0], aligned_edge=LEFT)
        with self.say(f"One real weight: gradient minus {abs(wt['g']):.2f}. Negative means: increase me, and the loss "
                      f"drops. A plain step at learning rate 0.003 adds {wt['step']:.4f}.",
                      scene=f"Heat map of the real gradient of layer 3's MLP output matrix (64×256). Highlighted: row "
                            f"{wt['i']}, column {wt['j']}, weight {wt['w']:.4f}, gradient {wt['g']:+.4f} (finite-difference "
                            f"check {wt['fd']:+.4f}). Gradient step at lr {wt['lr']:g}: −lr × grad = {wt['step']:+.5f}, "
                            f"new weight {wt['w_new']:.4f}. Labelled as a toy: plain gradient descent on one prediction.", chapter="One step"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows, flow, flow_l, cap2, nz, merge)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(img), Create(frame), FadeIn(cap3), FadeIn(leg), run_time=0.8)
            self.mark("gradmap", frame, f"real gradient of blocks.2.mlp_out.weight (64×256), max |g| = {gm:.4f}")
            self.play(Create(ring), run_time=0.4)
            self.play(FadeIn(info, shift=LEFT * 0.2), run_time=0.6)
            self.mark("theweight", VGroup(ring), f"weight [{wt['i']}, {wt['j']}] of layer 3 MLP output: "
                                                 f"{wt['w']:.5f}, gradient {wt['g']:+.5f}")
            self.mark("info", info, f"w = {wt['w']:.5f}, ∂loss/∂w = {wt['g']:+.5f}, finite difference {wt['fd']:+.5f}")
            self.sync("A plain step", 0.1)
            self.play(FadeIn(upd, shift=UP * 0.2), run_time=0.7)
            self.mark("update", upd, f"w ← w − {wt['lr']:g} × grad = {wt['w']:.5f} + {wt['step']:.5f} = {wt['w_new']:.5f}")

        # ============================================================ step size
        gl = bd["total"]
        c1 = VGroup(self.T("raw step on all weights", 28, FG),
                    VGroup(self.chip(" red", PINK, 28), self.M(f"{pct(bd['p_red'])} → {pct(bd['p_after'])}", 34, PINK)
                           ).arrange(RIGHT, buff=0.3),
                    self.T("overshoots", 26, PINK, BOLD)).arrange(DOWN, buff=0.3)
        c2 = VGroup(self.T(f"clipped to length 1 (train.py clips if > 1)", 28, FG),
                    VGroup(self.chip(" red", PINK, 28), self.M(f"{pct(bd['p_red'])} → {pct(bd['p_after_clip'])}", 34, TEAL)
                           ).arrange(RIGHT, buff=0.3),
                    self.T("gentle nudge the right way", 26, TEAL, BOLD)).arrange(DOWN, buff=0.3)
        cmp_ = VGroup(c1, c2).arrange(RIGHT, buff=1.6).move_to(DOWN * 0.1)
        divd = Line(UP * 1.3, DOWN * 1.3, color=LINE, stroke_width=2).move_to(cmp_.get_center())
        note = self.T(f"toy step on ONE prediction · this gradient's length: {gl:.2f} · learning rate 0.003", 24, MUTED)
        note.next_to(cmp_, DOWN, buff=0.7)
        with self.say(f"Take that step on every weight at once, and it overshoots: red falls to "
                      f"{100 * bd['p_after']:.0f} percent. TinyGPT's training clips gradients longer than one, and this one is "
                      f"{gl:.2f}. Clipped, red rises to {100 * bd['p_after_clip']:.1f}.",
                      scene=f"Two real outcomes of one gradient step at lr 0.003 on all {bd['n_params']:,} weights: raw "
                            f"gradient (length {gl:.2f}) → p(red) {pct(bd['p_red'])} → {pct(bd['p_after'])} (overshoot); "
                            f"gradient clipped to length 1 (train.py clips whenever the length exceeds 1) → {pct(bd['p_red'])} → "
                            f"{pct(bd['p_after_clip'])}."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(img, frame, ring, cap3, leg, info, upd)), run_time=0.4)
            self.play(FadeIn(c1, shift=UP * 0.2), FadeIn(note), run_time=0.6)
            self.mark("raw", c1, f"raw gradient step (lr 0.003): p(red) {bd['p_red']:.4f} → {bd['p_after']:.4f}")
            self.sync("Clipped,", 0.1)
            self.play(Create(divd), FadeIn(c2, shift=UP * 0.2), run_time=0.6)
            self.mark("clipped", c2, f"gradient clipped to norm 1 (from {gl:.3f}), lr 0.003: p(red) {bd['p_red']:.4f} → "
                                     f"{bd['p_after_clip']:.4f}")

        # ============================================================ real training
        from model import TRAIN
        B, ctx = TRAIN["batch"], 64
        items = [(f"averages blame over {B} × {ctx} = {B * ctx:,} predictions per batch", BLUE),
                 ("clips the gradient only if its length is over 1 (a safety net)", TEAL),
                 (f"learning rate {TRAIN['lr']:g} is the peak of a warmup + cosine schedule", YELLOW),
                 ("AdamW picks each weight's own step size (see the Optimizers video)", PINK)]
        real = VGroup()
        for t, c in items:
            dot_ = Dot(radius=0.09, color=c)
            real.add(VGroup(dot_, self.T(t, 30, FG)).arrange(RIGHT, buff=0.3))
        real.arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        rt = self.T("what train.py does instead of the toy step", 32, FG, BOLD)
        hdr_rt = self.header("Real training")
        realg = VGroup(rt, real).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        self.fitw(realg, 12.6).move_to(DOWN * 0.1)
        with self.say("And real training averages blame over 4,096 predictions per batch, uses 0.003 only as a cosine schedule's peak, and lets AdamW set each weight's step.",
                      scene=f"Real training differs from the toy step: averages over {B * ctx:,} predictions per batch, clips "
                            f"the gradient only when its length exceeds 1, uses lr {TRAIN['lr']:g} as the peak of a warmup + "
                            f"cosine schedule, and AdamW adapts each weight's step size."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(c1, c2, divd, note)), FadeIn(rt), ReplacementTransform(hdr3, hdr_rt), run_time=0.4)
            for k, ph in enumerate(["averages", "uses 0.003", "uses 0.003", "lets AdamW"]):
                if k != 2:
                    self.sync(ph, 0.1)
                self.play(FadeIn(real[k], shift=RIGHT * 0.2), run_time=0.4)
            for k, (t, _) in enumerate(items):
                self.mark(f"real-{k}", real[k], t)

        # ============================================================ cost
        hdr4 = self.header("Why backward costs ~2×")

        def mat(w_, h_, lab, col):
            r = Rectangle(width=w_, height=h_, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.15)
            t = self.M(lab, 28, col).move_to(r)
            return VGroup(r, t)

        def eq(parts):
            return VGroup(*parts).arrange(RIGHT, buff=0.22)
        S = 0.62
        f_row = eq([mat(0.8, 1.0 * S, "X", BLUE), self.M("·", 32), mat(1.0, 0.8 * S, "W", TEAL),
                    self.M("=", 32), mat(1.0, 1.0 * S, "Y", BLUE)])
        b_row1 = eq([mat(1.0, 1.0 * S, "dY", BWD), self.M("·", 32), mat(0.8, 1.0 * S, "Wᵀ", TEAL), self.M("=", 32),
                     mat(0.8, 1.0 * S, "dX", BWD)])
        b_row2 = eq([mat(1.0, 0.8 * S, "Xᵀ", BLUE), self.M("·", 32), mat(1.0, 1.0 * S, "dY", BWD), self.M("=", 32),
                     mat(1.0, 0.8 * S, "dW", BWD)])
        labs = [self.T("forward: one multiply", 28, FG), self.T("backward: blame for the inputs", 28, BWD),
                self.T("backward: blame for the weights", 28, BWD)]
        blocks = VGroup()
        for k, (lab, rw) in enumerate(zip(labs, (f_row, b_row1, b_row2))):
            y = 2.0 - 1.15 * k
            lab.move_to([-0.4, y, 0], aligned_edge=RIGHT)
            rw.move_to([0.2, y, 0], aligned_edge=LEFT)
            blocks.add(VGroup(lab, rw))
        tally = VGroup(self.T("forward ≈ 1", 30, FG), self.T("backward ≈ 2", 30, BWD, BOLD),
                       self.T("training step ≈ 3 forward passes (roughly)", 30, YELLOW, BOLD)).arrange(RIGHT, buff=0.6)
        self.fitw(tally, 12.6).move_to(DOWN * 1.75)
        mem = self.T("and forward activations are kept for the backward pass (unless you use activation checkpointing)", 24, MUTED)
        self.fitw(mem, 12.8)
        mem.move_to(DOWN * 2.75)
        with self.say('Why does backward cost about twice forward? Each forward matrix multiply needs two going back: blame for its inputs, and for its weights. So a training step is roughly three forward passes.',
                      scene="Matrix diagram: forward Y = X·W (one multiply); backward dX = dY·Wᵀ (blame for inputs) and "
                            "dW = Xᵀ·dY (blame for weights): two multiplies. Tally: forward ≈ 1, backward ≈ 2, training step "
                            "≈ 3 units (approximate). Note: activations must be stored for the backward pass.",
                      chapter="The cost"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(realg), ReplacementTransform(hdr_rt, hdr4), run_time=0.4)
            self.play(FadeIn(blocks[0], shift=UP * 0.2), run_time=0.6)
            self.mark("fwd-mm", blocks[0], "forward: Y = X·W, one matrix multiply per layer weight")
            self.sync("blame for its inputs", 0.1)
            self.play(FadeIn(blocks[1], shift=UP * 0.2), run_time=0.6)
            self.mark("bwd-mm1", blocks[1], "backward: dX = dY·Wᵀ passes the gradient to the layer's input")
            self.sync("and for its weights", 0.1)
            self.play(FadeIn(blocks[2], shift=UP * 0.2), run_time=0.6)
            self.mark("bwd-mm2", blocks[2], "backward: dW = Xᵀ·dY is the gradient for the weights")
            self.sync("So a training", 0.1)
            self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in tally], lag_ratio=0.3), FadeIn(mem), run_time=0.9)
            self.mark("tally", tally, "rule of thumb: backward ≈ 2× forward FLOPs, so a training step ≈ 3× a forward pass "
                                      "(approximate; depends on implementation)")
            self.mark("memory", mem, "backprop needs the forward activations, so training uses much more memory than inference")

        # ============================================================ recap
        steps = [("forward", "get the loss", BLUE), ("backward", "share the blame", ORANGE), ("nudge", "every weight", TEAL)]
        pills = VGroup()
        for t1, t2, c in steps:
            tt = VGroup(self.T(t1, 32, c, BOLD), self.T(t2, 26, FG)).arrange(DOWN, buff=0.12)
            bx = RoundedRectangle(corner_radius=0.15, width=3.2, height=tt.height + 0.5, stroke_color=c, stroke_width=2.5,
                                  fill_color=c, fill_opacity=0.1)
            pills.add(VGroup(bx, tt.move_to(bx)))
        pills.arrange(RIGHT, buff=0.9).move_to(UP * 0.8)
        arr = VGroup(*[Arrow(pills[i].get_right(), pills[i + 1].get_left(), buff=0.1, color=MUTED, stroke_width=3)
                       for i in range(2)])
        back = CurvedArrow(pills[-1].get_bottom() + DOWN * 0.1, pills[0].get_bottom() + DOWN * 0.1, angle=-PI / 5,
                           color=MUTED, stroke_width=3)
        times = self.T("× 3,000 steps", 30, MUTED).next_to(back, DOWN, buff=0.15)
        code = self.M("loss.backward()   # PyTorch runs the backward sweep", 26, ORANGE).next_to(pills, UP, buff=0.6)
        cta = self.T("next: watch the gradients in the Backprop lab →", 40, YELLOW, BOLD).move_to(DOWN * 2.7)
        self.fitw(cta, 12.8)
        with self.say('Forward, backward, nudge. In PyTorch, loss dot backward does the middle step for you. Go watch the gradients flow.',
                      scene="Recap loop: forward (get the loss) → backward (share the blame) → nudge every weight, looping "
                            "back, × 3,000 steps. Final card: 'watch the gradients in the Backprop lab'.", chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(*[m for m in self.mobjects])), run_time=0.4)
            for k in range(3):
                self.play(FadeIn(pills[k], shift=UP * 0.2), *([GrowArrow(arr[k - 1])] if k else []), run_time=0.35)
            for (t1, t2, _), p in zip(steps, pills):
                self.mark("loop-" + t1, p, f"training loop: {t1}, {t2}")
            self.play(Create(back), FadeIn(times), run_time=0.5)
            self.sync("In PyTorch", 0.1)
            self.play(FadeIn(code, shift=DOWN * 0.2), run_time=0.4)
            self.mark("code", code, "in PyTorch, loss.backward() computes every gradient by backprop")
            self.sync("Go watch", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "next: the Backprop lab")
        self.wait(0.6)
