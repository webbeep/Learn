import os, sys, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from lib_concepts_viz import CScene
from model import VOCAB, META, tok, forward
from lib_concepts import PROBE

VIDEO = dict(id="frontier", cls="Frontier", title="The frontier: a map for model inventors",
             subtitle="RoPE, efficient attention, state-space models, GQA, MoE, distillation, retrieval, diffusion LMs, "
                      "test-time compute, better training",
             relates="invent", track="research", order=20, poster=("The map", 0, 0.97))

PURPLE = "#b48cff"
HUB = [("tokens & positions", BLUE), ("attention", BLUE), ("MLP", TEAL), ("next-token loss", YELLOW),
       ("generation loop", PINK)]
CLUSTERS = {"long": ("long sequences & fast inference", TEAL), "cheap": ("more model per FLOP", ORANGE),
            "know": ("knowledge outside the weights", BLUE), "gen": ("new ways to generate", PINK),
            "train": ("how it learns", PURPLE)}


class Frontier(CScene):
    VIDEO_ID = "frontier"

    # ---------------------------------------------------------------- builders
    def hub(self):
        boxes = VGroup()
        for name, col in HUB:
            t = self.T(name, 24, col, BOLD)
            b = RoundedRectangle(corner_radius=0.12, width=3.25, height=0.6, stroke_color=col, stroke_width=2.2,
                                 fill_color=PANEL, fill_opacity=1)
            boxes.add(VGroup(b, t.move_to(b)))
        boxes.arrange(DOWN, buff=0.25)
        title = self.T("TinyGPT", 28, FG, BOLD).next_to(boxes, UP, buff=0.25)
        return VGroup(title, boxes)

    def node(self, text, col):
        t = VGroup(*[self.T(x, 24, col, BOLD) for x in text.split("\n")]).arrange(DOWN, buff=0.06)
        b = RoundedRectangle(corner_radius=0.16, width=t.width + 0.4, height=t.height + 0.32, stroke_color=col,
                             stroke_width=2.2, fill_color=col, fill_opacity=0.12)
        return VGroup(b, t.move_to(b))

    def chip_s(self, s, col, size=24):
        return self.chip(s, col, size, font=SANS if not s.startswith(" ") else MONO, pad=0.1)

    def card(self, tp):
        col = CLUSTERS[tp["cluster"]][1]
        title = self.T(tp["title"], 40, col, BOLD)
        swaps = self.T(f"in TinyGPT terms: {tp['swaps']}", 24, MUTED)
        head = VGroup(title, swaps).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rows = VGroup()
        for lab, key, c in (("idea", "idea", FG), ("open", "open", YELLOW)):
            lines = textwrap.wrap(tp[key], 60)
            body = VGroup(*[self.T(l, 26, c) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            tag = self.T(lab, 24, col, BOLD)
            tag.move_to([0, 0, 0], aligned_edge=RIGHT)
            r = VGroup(tag, body)
            body.next_to(tag, RIGHT, buff=0.3).align_to(tag, UP).shift(UP * 0.02)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        # right-align tags in a column
        tx = max(r[0].get_right()[0] for r in rows)
        for r in rows:
            r[0].align_to([tx, 0, 0], RIGHT)
            r[1].next_to(r[0], RIGHT, buff=0.3).align_to(r[0], UP).shift(UP * 0.02)
        g = VGroup(head, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.fitw(g, 10.0)
        g.move_to([-3.35, 2.95, 0], aligned_edge=UL)
        ref = self.T(tp["refs"], 22, MUTED)
        self.fitw(ref, 10.0)
        ref.to_edge(DOWN, buff=0.3).align_to(g, LEFT)
        return g, ref

    # ---------------------------------------------------------------- mini diagrams (all centred at DIAG)
    def d_moe(self, cfg):
        d, m = cfg["d_model"], cfg["d_mlp"]
        mlp = 2 * d * m + m + d
        tokc = self.chip(" red", PINK, 26)
        router = self.node("router", YELLOW)
        experts = VGroup(*[RoundedRectangle(corner_radius=0.08, width=0.62, height=0.5, stroke_color=TEAL, stroke_width=2,
                                            fill_color=TEAL, fill_opacity=0.5 if k in (2, 5) else 0.06) for k in range(8)])
        experts.arrange(RIGHT, buff=0.12)
        el = self.T("8 expert MLPs", 22, TEAL).next_to(experts, DOWN, buff=0.12)
        row = VGroup(tokc, router, VGroup(experts, el)).arrange(RIGHT, buff=0.8)
        a1 = Arrow(tokc.get_right(), router.get_left(), buff=0.1, color=MUTED, stroke_width=3)
        a2 = VGroup(*[CurvedArrow(router.get_top() + UP * 0.05, experts[k].get_top() + UP * 0.05, angle=-PI / 2.6,
                                  color=ORANGE, stroke_width=3, tip_length=0.18) for k in (2, 5)])
        cap = VGroup(self.T(f"hypothetical TinyGPT-MoE, top-2 of 8 experts:", 24, MUTED),
                     self.T(f"MLP weights per layer {mlp:,} → {8 * mlp:,} (+ router), but each token runs only 2", 24, MUTED)
                     ).arrange(DOWN, buff=0.06)
        g = VGroup(row, a1, a2)
        cap.next_to(g, DOWN, buff=0.3)
        return VGroup(g, cap), f"hypothetical TinyGPT with 8 expert MLPs (top-2 routing): {mlp:,} → {8 * mlp:,} MLP weights per layer"

    def d_ssm(self, toks):
        chips = VGroup(*[self.chip(t, BLUE, 22) for t in toks]).arrange(RIGHT, buff=0.42)
        states = VGroup()
        for k, c in enumerate(chips):
            s = Square(0.42, stroke_color=TEAL, stroke_width=2, fill_color=TEAL, fill_opacity=0.25)
            s.next_to(c, DOWN, buff=0.35)
            states.add(s)
        arr = VGroup(*[Arrow(states[k].get_right(), states[k + 1].get_left(), buff=0.03, color=TEAL, stroke_width=3,
                             max_tip_length_to_length_ratio=0.3) for k in range(len(states) - 1)])
        ups = VGroup(*[Line(c.get_bottom(), s.get_top(), color=LINE, stroke_width=2) for c, s in zip(chips, states)])
        cap = self.T("state h: same size at token 8 or token 8,000 · attention would compare every pair", 22, MUTED)
        g = VGroup(chips, ups, states, arr)
        cap.next_to(g, DOWN, buff=0.25)
        return VGroup(g, cap), "a recurrence passes one fixed-size state from token to token"

    def d_attn(self, cfg):
        n, w = 14, 4
        cs = 0.135
        def grid(band):
            g = VGroup()
            for r in range(n):
                for c in range(n):
                    on = c <= r and (not band or r - c < w)
                    g.add(Square(cs - 0.015, stroke_width=0, fill_color=BLUE if on else LINE,
                                 fill_opacity=0.9 if on else 0.35).move_to([c * cs, -r * cs, 0]))
            return g
        g1, g2 = grid(False), grid(True)
        l1 = VGroup(self.T("full causal", 24, FG), self.T("~n²/2 pairs", 22, MUTED)).arrange(DOWN, buff=0.05).next_to(g1, UP, buff=0.12)
        l2 = VGroup(self.T(f"sliding window", 24, FG), self.T(f"~n·w pairs (w = {w})", 22, MUTED)).arrange(DOWN, buff=0.05).next_to(g2, UP, buff=0.12)
        c = cfg["n_ctx"]
        big = 2048
        txt = VGroup(self.T(f"TinyGPT: up to {c} × {c} = {c * c:,}", 24, FG),
                     self.T(f"scores per head per layer;", 24, FG),
                     self.T(f"the causal mask keeps {c * (c + 1) // 2:,}", 24, FG),
                     self.T(f"at {big:,} tokens: {big * big:,}", 24, ORANGE),
                     self.T(f"({(big * big) // (c * c):,}× more)", 24, ORANGE)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        g = VGroup(VGroup(l1, g1), VGroup(l2, g2), txt).arrange(RIGHT, buff=0.6)
        return g, (f"TinyGPT: up to {c}×{c} = {c * c:,} attention scores per head per layer ({c * (c + 1) // 2:,} kept by the "
                   f"causal mask); {big:,} tokens: {big * big:,}")

    def d_rope(self):
        c = Circle(radius=1.0, color=LINE, stroke_width=2)
        th = ValueTracker(0.35)
        gap = 0.9
        def mk():
            o = c.get_center()
            q = Arrow(o, o + 1.0 * np.array([np.cos(th.get_value() + gap), np.sin(th.get_value() + gap), 0]), buff=0,
                      color=BLUE, stroke_width=5, max_tip_length_to_length_ratio=0.2)
            k = Arrow(o, o + 1.0 * np.array([np.cos(th.get_value()), np.sin(th.get_value()), 0]), buff=0, color=TEAL,
                      stroke_width=5, max_tip_length_to_length_ratio=0.2)
            arc = Arc(radius=0.45, start_angle=th.get_value(), angle=gap, arc_center=o, color=YELLOW, stroke_width=3)
            return VGroup(q, k, arc)
        rot = always_redraw(mk)
        lab = VGroup(self.T("query at position m", 24, BLUE), self.T("key at position n", 24, TEAL),
                     self.T("angle depends only on m − n", 24, YELLOW),
                     self.T("each pair of dimensions spins at its own frequency", 24, MUTED),
                     self.T("TinyGPT instead learned 64 slots (positions 0–63)", 24, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        g = VGroup(VGroup(c, rot), lab).arrange(RIGHT, buff=0.8)
        return g, th, rot

    def d_rag(self):
        q = self.chip("question", BLUE, 24, font=SANS)
        docs = VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.9, height=1.1, stroke_color=MUTED, stroke_width=2,
                                         fill_color=PANEL, fill_opacity=1).shift(RIGHT * 0.15 * k + DOWN * 0.12 * k)
                        for k in range(3)])
        docs[2].set_stroke(YELLOW, 3)
        dl = self.T("document store", 22, MUTED).next_to(docs, DOWN, buff=0.12)
        ctx = VGroup(self.chip("retrieved passage", YELLOW, 24, font=SANS), self.chip("question", BLUE, 24, font=SANS)
                     ).arrange(RIGHT, buff=0.1)
        cl = self.T("what the model reads", 22, MUTED).next_to(ctx, DOWN, buff=0.12)
        g = VGroup(q, VGroup(docs, dl), VGroup(ctx, cl)).arrange(RIGHT, buff=1.0)
        a1 = Arrow(q.get_right(), docs.get_left(), buff=0.12, color=MUTED, stroke_width=3)
        a2 = Arrow(docs.get_right(), ctx.get_left(), buff=0.12, color=YELLOW, stroke_width=3)
        return VGroup(g, a1, a2)

    def d_diff(self):
        steps = [["■", "■", "■", "■", "■"], [" tom", "■", " the", "■", " ball"], [" tom", " has", " the", " red", " ball"]]
        rows = VGroup()
        for k, st in enumerate(steps):
            r = VGroup(*[self.chip(t, MUTED if t == "■" else PINK, 24) for t in st]).arrange(RIGHT, buff=0.12)
            lab = self.T(["start: all masked", "refine", "refine"][k], 22, MUTED)
            rows.add(VGroup(lab, r))
        for r in rows:
            r[0].move_to(ORIGIN, aligned_edge=RIGHT)
            r[1].next_to(r[0], RIGHT, buff=0.4)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        note = self.T("illustration", 22, MUTED).next_to(rows, RIGHT, buff=0.6)
        return VGroup(rows, note)

    def d_ttc(self):
        q = self.chip("17 × 3 = ?", BLUE, 26, font=SANS)
        ans = ["51", "51", "41", "51", "54"]
        samp = VGroup(*[self.chip(a, TEAL if a == "51" else MUTED, 26, font=MONO) for a in ans]).arrange(RIGHT, buff=0.12)
        sl = self.T("5 sampled answers", 24, MUTED).next_to(samp, UP, buff=0.15)
        vote = self.chip("vote → 51", YELLOW, 26, font=SANS)
        g = VGroup(q, VGroup(sl, samp), vote).arrange(RIGHT, buff=0.9)
        a1 = Arrow(q.get_right(), samp.get_left(), buff=0.15, color=MUTED, stroke_width=3)
        a2 = Arrow(samp.get_right(), vote.get_left(), buff=0.15, color=MUTED, stroke_width=3)
        note = self.T("illustration (self-consistency style voting)", 24, MUTED).next_to(g, DOWN, buff=0.35)
        return VGroup(g, a1, a2, note)

    def d_gqa(self, cfg):
        H, D = cfg["n_head"], cfg["d_model"]

        def panel(groups, title):
            qs = VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.7, height=0.45, stroke_color=BLUE, stroke_width=2,
                                           fill_color=BLUE, fill_opacity=0.35) for _ in range(H)]).arrange(RIGHT, buff=0.18)
            kvs = VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.7 * (H // groups) + 0.18 * (H // groups - 1),
                                            height=0.45, stroke_color=ORANGE, stroke_width=2, fill_color=ORANGE,
                                            fill_opacity=0.35) for _ in range(groups)]).arrange(RIGHT, buff=0.18)
            kvs.next_to(qs, DOWN, buff=0.55)
            lines = VGroup(*[Line(qs[h].get_bottom(), kvs[h // (H // groups)].get_top(), color=LINE, stroke_width=2)
                             for h in range(H)])
            t = self.T(title, 26, FG).next_to(qs, UP, buff=0.15)
            per_tok = 2 * groups * (D // H)
            c = self.T(f"cache: {per_tok} numbers per token per layer", 24, ORANGE).next_to(kvs, DOWN, buff=0.15)
            return VGroup(t, qs, lines, kvs, c), per_tok
        p1, n1 = panel(H, f"TinyGPT: {H} query, {H} K/V heads")
        p2, n2 = panel(H // 2, f"grouped: {H} query, {H // 2} K/V heads")
        g = VGroup(p1, p2).arrange(RIGHT, buff=1.2)
        return g, f"TinyGPT caches {n1} numbers per token per layer (keys + values); with 2 shared K/V heads it would be {n2}"

    def d_train(self):
        items = [("optimizers", "AdamW and successors", BLUE), ("data & curricula", "what, in what order", TEAL),
                 ("new objectives", "e.g. multi-token prediction", YELLOW), ("RL post-training", "reward good answers", PINK),
                 ("lower precision", "mixed, 8-bit formats", ORANGE)]
        boxes = VGroup()
        for t1, t2, c in items:
            tt = VGroup(self.T(t1, 26, c, BOLD), self.T(t2, 22, FG)).arrange(DOWN, buff=0.06)
            b = RoundedRectangle(corner_radius=0.12, width=max(3.0, tt.width + 0.3), height=tt.height + 0.3, stroke_color=c,
                                 stroke_width=2, fill_color=c, fill_opacity=0.08)
            boxes.add(VGroup(b, tt.move_to(b)))
        top = VGroup(*boxes[:3]).arrange(RIGHT, buff=0.25)
        bot = VGroup(*boxes[3:]).arrange(RIGHT, buff=0.25)
        return VGroup(top, bot).arrange(DOWN, buff=0.25)

    def d_distill(self, probs):
        def chart(title, items, col, pcol):
            rows = VGroup()
            for k, (t, p) in enumerate(items):
                c = pcol.get(t, MUTED)
                lab = self.M(self.show_tok(t), 26, c, t2c={"␣": MUTED}).move_to([0, -0.45 * k, 0], aligned_edge=RIGHT)
                bar = Rectangle(width=max(0.03, 2.0 * p), height=0.3, stroke_width=0, fill_color=c, fill_opacity=0.85)
                bar.move_to([0.18, -0.45 * k, 0], aligned_edge=LEFT)
                val = self.M(f"{100 * p:.1f}%", 24, c).next_to(bar, RIGHT, buff=0.12)
                rows.add(VGroup(lab, bar, val))
            tt = self.T(title, 24, col).next_to(rows, UP, buff=0.2).align_to(rows, LEFT)
            return VGroup(tt, rows)
        pc = {" red": PINK, " white": "#e8e8e8", " purple": PURPLE}
        hard = chart("one-hot target", [(" red", 1.0), (" white", 0.0), (" purple", 0.0)], MUTED, pc)
        soft = chart("teacher's probabilities (real TinyGPT)", [(t, probs[t]) for t in (" red", " white", " purple")], FG, pc)
        g = VGroup(hard, soft).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        return g

    # ---------------------------------------------------------------- scene
    def construct(self):
        cfg = META["config"]
        lg = forward(tok(PROBE))[-1].softmax(-1)
        probs = {t: float(lg[VOCAB.index(t)]) for t in (" red", " white", " purple")}

        topics = [
            dict(key="rope", label="RoPE &\nlong context", cluster="long", hub=0, pos=(-4.75, 2.25),
                 title="Positions & long context (RoPE)", swaps="replaces the position table",
                 idea="rotate each query and key by angles set by its position, so position enters the score as relative distance",
                 open="stretching past the training length only partly works; using very long contexts well is still hard",
                 refs="RoFormer (Su et al. 2021) · Chen et al. 2023 · Lost in the Middle (Liu et al. 2023)"),
            dict(key="eff", label="efficient\nattention", cluster="long", hub=1, pos=(-4.75, 1.05),
                 title="Linear / efficient attention", swaps="changes attention",
                 idea="approximate or restrict attention (kernel tricks, sparse or sliding windows) so cost grows ~linearly",
                 open="approximations often cost quality; FlashAttention instead stays exact and just runs faster",
                 refs="Katharopoulos et al. 2020 · Longformer (2020) · FlashAttention (Dao et al. 2022)"),
            dict(key="ssm", label="state-space\n/ Mamba", cluster="long", hub=1, pos=(-4.75, -0.15),
                 title="State-space models (Mamba)", swaps="replaces attention (Mamba: the whole block)",
                 idea="a recurrence carries a fixed-size state along the sequence, yet still trains in parallel; Mamba makes "
                      "its updates depend on the input",
                 open="how much exact detail a fixed-size state can recall; hybrids that keep some attention are popular",
                 refs="S4 (Gu et al. 2021) · Mamba (Gu & Dao 2023)"),
            dict(key="gqa", label="grouped-query\nattention", cluster="long", hub=1, pos=(-4.75, -1.35),
                 title="Grouped-query attention", swaps="changes attention and the KV cache",
                 idea="several query heads share one key/value head, so far less has to be cached per token",
                 open="how much sharing before quality drops; one shared K/V head (multi-query) is the extreme",
                 refs="multi-query attention (Shazeer 2019) · GQA (Ainslie et al. 2023)"),
            dict(key="moe", label="mixture of\nexperts", cluster="cheap", hub=2, pos=(4.7, 0.6),
                 title="Mixture of experts", swaps="replaces the MLP",
                 idea="many expert MLPs; a learned router sends each token to a few: more parameters, not proportionally "
                      "more compute",
                 open="balancing load across experts; the memory to hold them all",
                 refs="Shazeer et al. 2017 · Switch Transformer (Fedus et al. 2021) · Mixtral (2024)"),
            dict(key="distill", label="distillation &\nsmall models", cluster="cheap", hub=3, pos=(4.7, -0.55),
                 title="Distillation & small models", swaps="changes the training target",
                 idea="train a small student on a big teacher's outputs: its full probabilities, or text it generated",
                 open="how much capability survives the shrink",
                 refs="Hinton, Vinyals & Dean 2015 · DistilBERT (Sanh et al. 2019)"),
            dict(key="rag", label="retrieval\n(RAG)", cluster="know", hub=0, pos=(4.7, 2.25),
                 title="Retrieval-augmented models", swaps="adds to the input (what it reads)",
                 idea="look up relevant documents in an external store and put them into the context",
                 open="finding the right text, and making the model actually use it",
                 refs="RAG (Lewis et al. 2020) · RETRO (Borgeaud et al. 2021)"),
            dict(key="diff", label="diffusion\nLMs", cluster="gen", hub=4, pos=(-2.65, -3.15),
                 title="Diffusion language models", swaps="replaces the left-to-right generation loop",
                 idea="start from a fully masked sequence and refine every position over a number of steps",
                 open="whether they match autoregressive quality and efficiency at scale is still being studied",
                 refs="D3PM (Austin et al. 2021) · masked diffusion LMs (e.g. Sahoo et al. 2024)"),
            dict(key="ttc", label="test-time\ncompute", cluster="gen", hub=4, pos=(2.65, -3.15),
                 title="Test-time compute & reasoning", swaps="changes how the generation loop is used",
                 idea="spend more compute per question: write out steps, or sample several answers and vote",
                 open="when extra thinking pays off, what it costs, and whether written steps reflect the real computation",
                 refs="chain of thought (Wei et al. 2022) · self-consistency (Wang et al. 2022) · DeepSeek-R1 (2025)"),
            dict(key="train", label="better\ntraining", cluster="train", hub=3, pos=(4.85, -2.45),
                 title="Better training", swaps="changes how it learns, not what it is",
                 idea="the optimizer, the data and its order, the objective, RL after pretraining, the number format",
                 open="many reported gains shrink at larger scale or against a well-tuned baseline",
                 refs="Adam (Kingma & Ba 2015) · Micikevicius et al. 2018 · multi-token: Gloeckle et al. 2024"),
        ]
        SAY = ["Want to invent a new kind of model? Here's a map of where people are pushing. Each idea changes one part of the engine TinyGPT runs on.", 'RoPE rotates queries and keys by their position, so attention sees relative distance. It helps stretch context, but using very long contexts well is still hard.', 'Efficient attention approximates or restricts attention, like sliding windows, to dodge the quadratic cost, often losing some quality. FlashAttention instead stays exact and just runs faster.', 'State-space models like Mamba replace attention with a recurrence that carries a fixed-size state, so cost grows linearly with length. Open question: how much exact detail that state can hold.', 'Grouped-query attention lets several query heads share one key and value head. That shrinks the KV cache and speeds up generation, usually with little quality loss.', 'Mixture of experts swaps the MLP for many expert MLPs, with a router picking a few per token: more parameters, without proportionally more compute. Balancing the experts is the hard part.', "Distillation trains a small student on a big teacher's outputs: its full probabilities, or text it generated. How much skill survives the shrink is the open question.", 'Retrieval-augmented models look things up, putting relevant documents into the context. Knowledge can stay fresh without retraining, if retrieval finds the right text and the model uses it.', 'Diffusion language models drop left-to-right: start fully masked, and refine every position over a number of steps. Whether they match autoregressive quality at scale is still open.', 'Test-time compute spends more thinking per question: write out reasoning steps, or sample several answers and vote. Some recent models are trained with reinforcement learning to reason longer.', "And if you'd rather invent a training framework: optimizers, data quality and curricula, new objectives like predicting several tokens at once, RL after pretraining, and lower-precision math are all active areas.", 'Every one of these changes one box. Pick a box, form a hypothesis, and run the loop.']
        for k, tp in enumerate(topics):
            tp["say"] = SAY[k + 1]

        # ============================================================ the map
        hub = self.hub().move_to([0, 0.05, 0])
        hboxes = hub[1]
        nodes, edges = {}, {}
        for tp in topics:
            col = CLUSTERS[tp["cluster"]][1]
            nd = self.node(tp["label"], col).move_to([*tp["pos"], 0])
            tgt = hboxes[tp["hub"]]
            if tp["pos"][1] < -2.5:
                a, b = nd.get_top(), tgt.get_bottom() + RIGHT * (0.6 if tp["pos"][0] > 0 else -0.6)
            elif tp["pos"][0] < 0:
                a, b = nd.get_right(), tgt.get_left()
            else:
                a, b = nd.get_left(), tgt.get_right()
            edges[tp["key"]] = DashedLine(a, b, color=col, stroke_width=2.5, dash_length=0.1, stroke_opacity=0.8)
            nodes[tp["key"]] = nd
        halos, clabs = VGroup(), VGroup()
        for ck, (cname, col) in CLUSTERS.items():
            members = VGroup(*[nodes[tp["key"]] for tp in topics if tp["cluster"] == ck])
            h = SurroundingRectangle(members, buff=0.18, corner_radius=0.25, stroke_color=col, stroke_width=1.5,
                                     stroke_opacity=0.5, fill_color=col, fill_opacity=0.04)
            lab = self.T(cname, 20 if ck == "gen" else 22, col, BOLD)
            if ck == "gen":
                lab.move_to([0, h.get_center()[1], 0])
            elif ck in ("cheap", "train"):
                lab.next_to(h, DOWN, buff=0.08)
            else:
                lab.next_to(h, UP, buff=0.08)
            halos.add(h); clabs.add(lab)
        VGroup(hub, halos, clabs, *nodes.values(), *edges.values()).scale(0.93, about_point=ORIGIN)
        order = ["long", "cheap", "know", "gen", "train"]
        with self.say(SAY[0],
                      scene="A map: TinyGPT's parts in the centre (tokens & positions, attention, MLP, next-token loss, "
                            "generation loop) with ten research directions around it, grouped into clusters: long sequences & fast "
                            "inference (RoPE, efficient attention, state-space/Mamba, grouped-query attention), more model per "
                            "FLOP (mixture of experts, distillation), knowledge outside the weights (retrieval), new ways to "
                            "generate (diffusion LMs, test-time compute), how it learns (better training). Dashed lines show "
                            "which part each idea changes.",
                      chapter="The map"):
            self.play(FadeIn(hub, shift=UP * 0.2), run_time=0.7)
            for k, (nm, _) in enumerate(HUB):
                self.mark(f"hub-{k}", hboxes[k], f"TinyGPT component: {nm}")
            self.sync("Here's a map", 0.1)
            for ck in order:
                ks = [tp["key"] for tp in topics if tp["cluster"] == ck]
                i = list(CLUSTERS).index(ck)
                self.play(FadeIn(halos[i]), FadeIn(clabs[i]), *[FadeIn(nodes[k], scale=1.15) for k in ks],
                          *[Create(edges[k]) for k in ks], run_time=0.55)
            for tp in topics:
                self.mark("node-" + tp["key"], nodes[tp["key"]], f"{tp['title']}: {tp['idea']} ({tp['swaps']})")
            self.sync("Each idea", 0.1)
            self.play(*[Indicate(nodes[tp["key"]], scale_factor=1.06, color=CLUSTERS[tp["cluster"]][1]) for tp in topics],
                      run_time=0.8)

        # ============================================================ topic by topic
        everything = VGroup(halos, clabs, *nodes.values(), *edges.values())
        self.unmark(*list(self._marks))
        hub_center0 = hub.get_center()
        hub_target = hub.copy().move_to([-5.3, -0.15, 0])
        diag_center = np.array([1.75, -1.75, 0])
        cur = None
        for idx, tp in enumerate(topics):
            col = CLUSTERS[tp["cluster"]][1]
            card, ref = self.card(tp)
            extra = None
            th = rot = None
            if tp["key"] == "moe":
                diag, extra = self.d_moe(cfg)
            elif tp["key"] == "ssm":
                diag, extra = self.d_ssm([VOCAB[i] for i in tok(" tom has the red ball now mia")][:8])
            elif tp["key"] == "eff":
                diag, extra = self.d_attn(cfg)
            elif tp["key"] == "rope":
                diag, th, rot = self.d_rope()
            elif tp["key"] == "rag":
                diag = self.d_rag()
            elif tp["key"] == "diff":
                diag = self.d_diff()
            elif tp["key"] == "ttc":
                diag = self.d_ttc()
            elif tp["key"] == "gqa":
                diag, extra = self.d_gqa(cfg)
            elif tp["key"] == "train":
                diag = self.d_train()
            else:
                diag = self.d_distill(probs)
            self.fitw(diag, 9.6)
            if diag.height > 2.3:
                diag.scale_to_fit_height(2.3)
            diag.move_to(diag_center)
            # keep the diagram clear of the card text
            gap = card.get_bottom()[1] - diag.get_top()[1]
            if gap < 0.3:
                diag.shift(DOWN * (0.3 - gap))
            chapter = {"moe": "More model per FLOP", "rope": "Long sequences & fast inference",
                       "rag": "Knowledge outside the weights", "diff": "New ways to generate",
                       "train": "Better training"}.get(tp["key"])
            with self.say(tp["say"], scene=f"{tp['title']} ({tp['swaps']} in TinyGPT). Idea: {tp['idea']}. "
                                           f"Open question: {tp['open']}. Refs: {tp['refs']}.",
                          chapter=chapter):
                self.unmark(*list(self._marks))
                anims = []
                if idx == 0:
                    anims += [FadeOut(everything), Transform(hub, hub_target)]
                else:
                    anims += [FadeOut(cur)]
                hl = SurroundingRectangle(hboxes[tp["hub"]] if idx else hub_target[1][tp["hub"]], buff=0.08,
                                          corner_radius=0.14, color=col, stroke_width=5)
                self.play(*anims, run_time=0.6)
                self.play(FadeIn(hl), FadeIn(card, shift=LEFT * 0.2), FadeIn(ref), run_time=0.6)
                self.play(FadeIn(diag, shift=UP * 0.2), run_time=0.6)
                self.mark("hub-part", hboxes[tp["hub"]], f"the TinyGPT part this idea touches: {tp['swaps']}")
                self.mark("card-" + tp["key"], card, f"{tp['title']} — idea: {tp['idea']}; open question: {tp['open']}")
                self.mark("diag-" + tp["key"], diag, extra or f"{tp['title']}: schematic diagram")
                if th is not None:
                    self.play(th.animate.set_value(2.2), run_time=1.5)
                    self.play(th.animate.set_value(0.6), run_time=1.0)
                cur = VGroup(card, ref, diag, hl)
            self.wait(0.6)   # let the card breathe before the next one
            if rot is not None:
                rot.clear_updaters()

        # ============================================================ outro
        hub_home = hub.copy().move_to(hub_center0)
        cta = self.T("pick a box · form a hypothesis · run the loop →", 36, YELLOW, BOLD).to_edge(DOWN, buff=0.3)
        with self.say(SAY[-1],
                      scene="Back to the full map: TinyGPT's five parts with the ten directions around them. CTA: pick a "
                            "box, form a hypothesis, run the loop (Invent lab).", chapter="Your move"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(cur), Transform(hub, hub_home), run_time=0.6)
            self.play(FadeIn(everything), run_time=0.7)
            self.play(VGroup(hub, everything).animate.scale(0.9).shift(UP * 0.4), run_time=0.6)
            for tp in topics:
                self.mark("node-" + tp["key"], nodes[tp["key"]], f"{tp['title']} ({tp['swaps']})")
            for k, (nm, _) in enumerate(HUB):
                self.mark(f"hub-{k}", hboxes[k], f"TinyGPT component: {nm}")
            self.sync("Pick a box", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "next: the Invent lab")
        self.wait(0.6)
