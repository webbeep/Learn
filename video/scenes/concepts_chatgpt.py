import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from lib_concepts_viz import CScene, pipeline, log_axis
from model import VOCAB, META, TRAIN, tok, forward
from lib_concepts import PROBE, kv_cache_check

VIDEO = dict(id="tiny-to-chatgpt", cls="TinyToChatGPT", title="From TinyGPT to ChatGPT",
             subtitle="Same engine, very different scale: pretraining, instruction tuning, RLHF, chat tokens, KV cache",
             relates="xray", track="design", order=40, poster=("Scale", 0, 0.9))

# Published figures only (Radford et al. 2019; Brown et al. 2020; Ouyang et al. 2022).
GPT2_SMALL = dict(params=124e6, layers=12, d=768, ctx=1024)
GPT2_XL = dict(params=1.5e9, layers=48, d=1600, ctx=1024)
GPT3 = dict(params=175e9, layers=96, d=12288, ctx=2048, tokens=300e9)
COL = {" red": PINK, " white": "#e8e8e8", " purple": "#b48cff", " black": "#7d8597"}


def human(n):
    for v, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= v:
            x = n / v
            return f"{x:.0f}{s}" if x >= 10 or x == int(x) else f"{x:.1f}{s}"
    return str(int(n))


class TinyToChatGPT(CScene):
    VIDEO_ID = "tiny-to-chatgpt"

    def special(self, s, color, size=24, pad=0.1):
        """A special-token chip; ligatures disabled so '<|' and '|>' stay literal in JetBrains Mono."""
        esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lab = MarkupText(f'<span font_features="calt=0,liga=0">{esc}</span>', font=MONO, font_size=size, color=FG)
        box = RoundedRectangle(corner_radius=0.08, width=lab.width + 2 * pad + 0.04, height=lab.height + 2 * pad + 0.06,
                               stroke_color=color, stroke_width=2.2, fill_color=color, fill_opacity=0.18)
        return VGroup(box, lab.move_to(box))

    def construct(self):
        cfg = META["config"]
        n_tiny = TRAIN["params"]
        tiny_tokens = TRAIN["tokens"]
        kv = kv_cache_check(PROBE, 8)
        logits = forward(tok(PROBE))[-1].double().numpy()

        # ============================================================ same engine
        pipe, parts = pipeline(self)
        pipe.move_to(UP * 0.3)
        name = self.T("TinyGPT", 34, FG, BOLD).next_to(pipe, UP, buff=0.55)
        with self.say("ChatGPT runs the same loop as TinyGPT: read the tokens, score every possible next one, "
                      "pick, append, repeat. Same building blocks too: embeddings, attention, MLPs.",
                      scene="TinyGPT's loop as boxes: tokens → embed → [attention → MLP] × 3 layers → probabilities → pick "
                            "one, with an arrow looping back ('append it & repeat'). ChatGPT-style models use the same loop.",
                      chapter="Same engine"):
            self.play(FadeIn(name), LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in parts["boxes"]], lag_ratio=0.12),
                      LaggedStart(*[GrowArrow(a) for a in parts["arrows"]], lag_ratio=0.12), run_time=1.6)
            self.play(Create(parts["bracket"]), FadeIn(parts["layers"]), run_time=0.5)
            self.play(Create(parts["loop"]), FadeIn(parts["loop_label"]), run_time=0.6)
            for nm, b in zip(["tokens", "embed", "attention", "mlp", "probs", "pick"], parts["boxes"]):
                self.mark("pipe-" + nm, b, f"pipeline stage: {nm}")
            self.mark("pipe-layers", VGroup(parts["bracket"], parts["layers"]), f"TinyGPT stacks {cfg['n_layer']} blocks")
            self.mark("pipe-loop", VGroup(parts["loop"], parts["loop_label"]), "the generation loop: append the picked token, run again")
            self.sync("Same building", 0.1)
            self.play(Indicate(VGroup(parts["boxes"][1:4]), color=YELLOW, scale_factor=1.05), run_time=0.8)

        # ============================================================ scale
        hdr = self.header("Scale")
        ax, x_of = log_axis(self, 5, 12, x0=-2.6, x1=6.3, y_top=2.2, y_bot=-2.15)
        rows = [("TinyGPT", n_tiny, f"{cfg['n_layer']} layers · width {cfg['d_model']}", BLUE),
                ("GPT-2 small", GPT2_SMALL["params"], f"{GPT2_SMALL['layers']} layers · width {GPT2_SMALL['d']}", TEAL),
                ("GPT-2 XL", GPT2_XL["params"], f"{GPT2_XL['layers']} layers · width {GPT2_XL['d']:,}", TEAL),
                ("GPT-3", GPT3["params"], f"{GPT3['layers']} layers · width {GPT3['d']:,}", ORANGE)]
        bars = VGroup()
        for k, (nm, n, sub, c) in enumerate(rows):
            y = 1.55 - 1.08 * k
            lab = VGroup(self.T(nm, 30, c, BOLD), self.T(sub, 22, MUTED)).arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
            lab.move_to([-2.85, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=x_of(n) - x_of(1e5), height=0.5, stroke_width=0, fill_color=c, fill_opacity=0.85)
            bar.move_to([x_of(1e5), y, 0], aligned_edge=LEFT)
            val = self.M(f"{n:,.0f}" if n < 1e6 else human(n), 26, c).next_to(bar, RIGHT, buff=0.15)
            bars.add(VGroup(lab, bar, val))
        axl = self.T("parameters (log scale: each line = 10×)", 22, MUTED).to_corner(UR, buff=0.45)
        ratio = GPT3["params"] / n_tiny
        rt = self.T(f"GPT-3 ≈ {ratio / 1e6:.2f} million × TinyGPT", 32, YELLOW, BOLD).to_edge(DOWN, buff=0.35)
        unpub = self.T("published sizes only: many newer models don't disclose theirs", 22, MUTED).move_to(UP * 2.75)
        with self.say(f"The big difference is scale. GPT-2, in 2019, ranged from 124 million to 1.5 billion "
                      f"parameters. GPT-3, in 2020: 175 billion, 96 layers, vectors 12,288 wide. "
                      f"That's about a million TinyGPTs.",
                      scene=f"Log-scale bar chart of published parameter counts: TinyGPT {n_tiny:,} (3 layers, width 64), "
                            f"GPT-2 small 124M (12 layers, width 768), GPT-2 XL 1.5B (48 layers, width 1,600), GPT-3 175B "
                            f"(96 layers, width 12,288). GPT-3 ≈ {ratio:,.0f} × TinyGPT.", chapter="Scale"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(pipe, name)), FadeIn(hdr), run_time=0.5)
            self.play(FadeIn(ax), FadeIn(axl), run_time=0.5)
            self.play(FadeIn(bars[0][0]), GrowFromEdge(bars[0][1], LEFT), FadeIn(bars[0][2]), run_time=0.6)
            self.mark("bar-tiny", bars[0], f"TinyGPT: {n_tiny:,} parameters, {cfg['n_layer']} layers, d_model {cfg['d_model']}")
            self.sync("GPT-2", 0.1)
            for k in (1, 2):
                self.play(FadeIn(bars[k][0]), GrowFromEdge(bars[k][1], LEFT), FadeIn(bars[k][2]), run_time=0.6)
            self.mark("bar-gpt2s", bars[1], "GPT-2 small (2019): 124M parameters, 12 layers, d_model 768 (published)")
            self.mark("bar-gpt2xl", bars[2], "GPT-2 XL (2019): 1.5B parameters, 48 layers, d_model 1600 (published)")
            self.sync("GPT-3, in", 0.1)
            self.play(FadeIn(bars[3][0]), GrowFromEdge(bars[3][1], LEFT), FadeIn(bars[3][2]), run_time=0.9)
            self.mark("bar-gpt3", bars[3], "GPT-3 (2020): 175B parameters, 96 layers, d_model 12,288 (published)")
            self.sync("That's about", 0.1)
            self.play(FadeIn(rt, scale=1.2), run_time=0.4)
            self.mark("ratio", rt, f"175,000,000,000 / {n_tiny:,} ≈ {ratio:,.0f}")
            self.play(FadeIn(unpub), run_time=0.3)


        # ============================================================ pretraining
        hdr2 = self.header("Stage 1 · Pretraining")
        ax2, x2 = log_axis(self, 6, 12, x0=-2.6, x1=5.4, y_top=1.9, y_bot=-0.3,
                           labels={6: "1M", 7: "10M", 8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T"})
        trows = [("TinyGPT's corpus", tiny_tokens, BLUE), ("GPT-3 training", GPT3["tokens"], ORANGE)]
        tb = VGroup()
        for k, (nm, n, c) in enumerate(trows):
            y = 1.35 - 1.0 * k
            lab = self.T(nm, 28, c, BOLD).move_to([-2.85, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=x2(n) - x2(1e6), height=0.5, stroke_width=0, fill_color=c, fill_opacity=0.85)
            bar.move_to([x2(1e6), y, 0], aligned_edge=LEFT)
            val = self.M(f"{n / 1e6:.2f}M" if n < 1e9 else f"≈ {human(n)}", 26, c).next_to(bar, RIGHT, buff=0.15)
            tb.add(VGroup(lab, bar, val))
        axl2 = self.T("tokens of training text (log scale)", 22, MUTED).to_corner(UR, buff=0.45)
        same = self.T("same game: predict the next token", 30, FG).move_to(DOWN * 1.35)
        doc = VGroup(self.M("Q: What is the capital of France?", 26, FG),
                     self.M("Q: What is the capital of Germany?", 26, MUTED),
                     self.M("Q: What is the capital of Italy? …", 26, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        doc_l = self.T("a base model may just continue the document (illustration)", 22, MUTED)
        docg = VGroup(doc, doc_l).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(DOWN * 2.55)
        with self.say(f"Stage one, pretraining: TinyGPT's exact game, on a huge pile of web text. "
                      f"GPT-3 saw about 300 billion tokens; TinyGPT's world is {tiny_tokens / 1e6:.1f} million. "
                      f"The result is a base model: an autocomplete, not an assistant. Ask it something, "
                      f"and it may just continue the document.",
                      scene=f"Log-scale bars of training tokens: TinyGPT's corpus {tiny_tokens:,} tokens vs GPT-3 ≈ 300 "
                            f"billion tokens (published). Caption: same game, predict the next token. Illustration: a base "
                            f"model continuing 'Q: What is the capital of France?' with more questions instead of answering.",
                      chapter="Pretraining"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax, axl, bars, rt, unpub)), ReplacementTransform(hdr, hdr2), run_time=0.5)
            self.play(FadeIn(ax2), FadeIn(axl2), FadeIn(same), run_time=0.5)
            self.sync("GPT-3 saw", 0.1)
            self.play(FadeIn(tb[1][0]), GrowFromEdge(tb[1][1], LEFT), FadeIn(tb[1][2]), run_time=0.7)
            self.mark("tok-gpt3", tb[1], "GPT-3 was trained on about 300 billion tokens (Brown et al. 2020)")
            self.sync("TinyGPT's world", 0.1)
            self.play(FadeIn(tb[0][0]), GrowFromEdge(tb[0][1], LEFT), FadeIn(tb[0][2]), run_time=0.6)
            self.mark("tok-tiny", tb[0], f"TinyGPT's whole training corpus: {tiny_tokens:,} tokens")
            self.mark("same-game", same, "pretraining objective = next-token prediction, same cross-entropy loss")
            self.sync("Ask it something", 0.1)
            self.play(FadeOut(same), FadeIn(docg, shift=UP * 0.2), run_time=0.6)
            self.mark("base-doc", docg, "illustration: a base model may continue a question with more questions")

        # ============================================================ SFT
        hdr3 = self.header("Stage 2 · Instruction tuning (SFT)")

        def bubble(who, text, col, w=8.6):
            t = self.T(text, 28, FG)
            wl = self.T(who, 22, col, BOLD)
            body = VGroup(wl, t).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            b = RoundedRectangle(corner_radius=0.15, width=max(w, body.width + 0.5), height=body.height + 0.4,
                                 stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.08)
            body.move_to(b).align_to(b, LEFT).shift(RIGHT * 0.25)
            return VGroup(b, body)
        u = bubble("PERSON", "Explain photosynthesis to a ten-year-old.", BLUE)
        a = bubble("ASSISTANT", "Plants catch sunlight and use it to turn air and water into food…", TEAL)
        conv = VGroup(u, a).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(UP * 0.6 + LEFT * 0.3)
        under = Line(a[0].get_corner(DL) + RIGHT * 0.2 + DOWN * 0.12, a[0].get_corner(DR) + LEFT * 0.2 + DOWN * 0.12,
                     color=YELLOW, stroke_width=5)
        ul = self.T("loss is usually computed on the reply tokens only", 24, YELLOW).next_to(under, DOWN, buff=0.15)
        ex_l = self.T("example written by a person (illustration)", 22, MUTED).next_to(conv, UP, buff=0.2).align_to(conv, LEFT)
        with self.say("Stage two: supervised fine-tuning. Same loss, now on example conversations written by people: "
                      "a request, and a good response. The model learns to play the assistant.",
                      scene="Illustrative training example: PERSON 'Explain photosynthesis to a ten-year-old.' / ASSISTANT "
                            "'Plants catch sunlight…'. The reply is underlined: the loss is usually computed on the reply "
                            "tokens only.", chapter="Instruction tuning"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ax2, axl2, tb, docg)), ReplacementTransform(hdr2, hdr3), run_time=0.5)
            self.play(FadeIn(ex_l), FadeIn(u, shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeIn(a, shift=RIGHT * 0.2), run_time=0.6)
            self.mark("sft-person", u, "an instruction (prompt) in a supervised fine-tuning example")
            self.mark("sft-reply", a, "a demonstration reply written by a person")
            self.sync("a good response", 0.1)
            self.play(Create(under), FadeIn(ul), run_time=0.6)
            self.mark("sft-loss", VGroup(under, ul), "SFT: same next-token loss, typically only on the assistant's reply")

        # ============================================================ RLHF
        hdr4 = self.header("Stage 3 · Preference tuning (RLHF)")

        def card(title, text, col):
            t = VGroup(self.T(title, 26, col, BOLD), *[self.T(x, 24, FG) for x in text.split("\n")]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            b = RoundedRectangle(corner_radius=0.14, width=t.width + 0.5, height=t.height + 0.4, stroke_color=col,
                                 stroke_width=2.5, fill_color=col, fill_opacity=0.08)
            return VGroup(b, t.move_to(b))
        cA = card("answer A", "clear, correct, kind", TEAL)
        cB = card("answer B", "rambling, partly wrong", MUTED)
        ab = VGroup(cA, cB).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([-5.0, 0.7, 0])
        tick = self.T("✓ person prefers A", 26, TEAL, BOLD).next_to(ab, DOWN, buff=0.3)
        rm = card("reward model", "learns to predict\nthe picks", YELLOW).move_to([-0.7, 0.7, 0])
        pol = card("the chat model", "nudged toward\nhigh reward", PINK).move_to([4.4, 0.7, 0])
        a1 = Arrow(ab.get_right(), rm.get_left(), buff=0.15, color=MUTED, stroke_width=3)
        a2 = Arrow(rm.get_right(), pol.get_left(), buff=0.15, color=MUTED, stroke_width=3)
        a2l = VGroup(self.T("reinforcement", 22, MUTED), self.T("learning", 22, MUTED)).arrange(DOWN, buff=0.06)
        a2l.next_to(a2, UP, buff=0.22)
        leash = VGroup(self.T("+ a penalty for drifting", 24, PINK), self.T("far from the SFT model", 24, PINK)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).next_to(pol, DOWN, buff=0.3).align_to(pol, LEFT)
        cite = self.T("RLHF recipe as in InstructGPT (Ouyang et al., 2022)", 22, MUTED).to_edge(DOWN, buff=1.25)
        with self.say("Stage three, preference tuning. A person picks the better of two answers. A reward model learns "
                      "those picks, and reinforcement learning nudges the chat model toward high reward, "
                      "with a penalty for drifting too far. That's RLHF, as in InstructGPT, 2022.",
                      scene="RLHF diagram: two answers A and B, a person prefers A → a reward model learns to predict "
                            "preferences → reinforcement learning nudges the chat model toward high reward, plus a penalty "
                            "(KL) for drifting far from the SFT model. Citation: InstructGPT, Ouyang et al. 2022.",
                      chapter="Preference tuning"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(conv, under, ul, ex_l)), ReplacementTransform(hdr3, hdr4), run_time=0.5)
            self.play(FadeIn(ab, shift=RIGHT * 0.2), run_time=0.6)
            self.mark("ans-a", cA, "candidate answer A (illustration)")
            self.mark("ans-b", cB, "candidate answer B (illustration)")
            self.sync("A person picks", 0.1)
            self.play(FadeIn(tick, scale=1.2), run_time=0.4)
            self.mark("pref", tick, "human preference label: A is better than B")
            self.sync("A reward model", 0.1)
            self.play(GrowArrow(a1), FadeIn(rm), run_time=0.6)
            self.mark("rm", rm, "reward model trained on human comparisons")
            self.sync("and reinforcement", 0.1)
            self.play(GrowArrow(a2), FadeIn(a2l), FadeIn(pol), run_time=0.6)
            self.mark("policy", pol, "the chat model (policy), optimised with RL (PPO in InstructGPT)")
            self.sync("with a penalty", 0.1)
            self.play(FadeIn(leash), run_time=0.4)
            self.mark("kl", leash, "KL penalty keeps the tuned model close to the supervised (SFT) model")
            self.sync("That's RLHF", 0.1)
            self.play(FadeIn(cite), run_time=0.4)

        res = VGroup(
            VGroup(self.T("1.3B InstructGPT", 30, PINK, BOLD), self.T("preferred by labelers over", 26, FG),
                   self.T("175B GPT-3", 30, ORANGE, BOLD)).arrange(RIGHT, buff=0.3),
            self.T("despite having over 100× fewer parameters", 26, MUTED)).arrange(DOWN, buff=0.2).move_to(DOWN * 2.05)
        dpo = self.T("newer variants such as DPO (2023) skip the separate reward model", 24, MUTED).to_edge(DOWN, buff=0.3)
        self.fitw(dpo, 13.0)
        with self.say("In that paper, labelers preferred answers from a 1.3 billion parameter InstructGPT over the "
                      "175 billion GPT-3.",
                      scene="InstructGPT result (Ouyang et al. 2022): outputs of the 1.3B InstructGPT were preferred by "
                            "labelers over the 175B GPT-3, despite over 100× fewer parameters. Footnote: newer methods such "
                            "as DPO (2023) skip the separate reward model."):
            self.play(FadeOut(cite), FadeIn(res[0], shift=UP * 0.2), run_time=0.6)
            self.mark("result", res, "InstructGPT paper: 1.3B InstructGPT outputs preferred over 175B GPT-3 outputs")
            self.sync("175 billion", 0.1)
            self.play(FadeIn(res[1]), FadeIn(dpo), run_time=0.5)
            self.mark("dpo", dpo, "Direct Preference Optimization (Rafailov et al. 2023) skips the reward model")

        # ============================================================ chat format
        hdr5 = self.header("Chat = one long document")
        lines = [[("<|im_start|>", "sp"), ("system", "role")], [("You are a helpful assistant.", "t"), ("<|im_end|>", "sp")],
                 [("<|im_start|>", "sp"), ("user", "role")], [("What color is the sky?", "t"), ("<|im_end|>", "sp")],
                 [("<|im_start|>", "sp"), ("assistant", "role")], [("Usually blue, because air scatters blue light most.", "gen"),
                                                                    ("<|im_end|>", "spgen")]]
        rows_ = VGroup()
        for ln in lines:
            parts_ = VGroup()
            for s, kind in ln:
                if kind.startswith("sp"):
                    parts_.add(self.special(s, PINK if kind == "sp" else TEAL))
                elif kind == "role":
                    parts_.add(self.M(s, 26, YELLOW))
                else:
                    parts_.add(self.M(s, 26, TEAL if kind == "gen" else FG))
            rows_.add(parts_.arrange(RIGHT, buff=0.15))
        rows_.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(UP * 0.35)
        self.fitw(rows_, 12.6)
        fmt_l = self.T("ChatML-style template · one example; every model family defines its own special tokens", 22, MUTED)
        fmt_l.to_edge(DOWN, buff=0.45)
        gen_l = self.T("the model writes this part, token by token, until it predicts the end-of-turn token", 24, TEAL)
        gen_l.next_to(rows_, DOWN, buff=0.4)
        with self.say("And a chat is still one long document. Special tokens mark who's speaking, and the model keeps "
                      "predicting next tokens until it predicts the end-of-turn token itself.",
                      scene="A conversation laid out as one token stream in a ChatML-style template: <|im_start|>system … "
                            "<|im_end|> <|im_start|>user What color is the sky?<|im_end|> <|im_start|>assistant, then the "
                            "model's reply (teal) generated until it emits <|im_end|>.", chapter="Chat format"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ab, tick, rm, pol, a1, a2, a2l, leash, res, dpo)), ReplacementTransform(hdr4, hdr5),
                      run_time=0.5)
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows_[:5]], lag_ratio=0.15), FadeIn(fmt_l),
                      run_time=1.2)
            for k, r in enumerate(rows_[:5]):
                self.mark(f"chat-{k}", r, "chat template line: special tokens (pink) mark turns and roles")
            self.sync("and the model keeps", 0.1)
            gen = rows_[5]
            self.play(LaggedStart(*[FadeIn(c) for c in gen[0]], lag_ratio=0.03), run_time=1.2)
            self.play(FadeIn(gen[1], scale=1.3), FadeIn(gen_l), run_time=0.5)
            self.mark("chat-gen", gen, "the assistant's reply is generated token by token; <|im_end|> ends the turn")

        # ============================================================ KV cache
        hdr6 = self.header("KV cache")
        n0, n_new = kv["n_prompt"], kv["n_new"]
        cs = 0.16

        def grid(cached):
            g = VGroup()
            for r in range(n_new):
                row = VGroup()
                n = n0 + r
                for c in range(n):
                    computed = (not cached) or (r == 0) or (c == n - 1)
                    is_new = c == n - 1
                    sq = Square(cs, stroke_width=1.2 if not computed else 0,
                                stroke_color=BLUE, fill_color=YELLOW if is_new else (FG if computed else BLUE),
                                fill_opacity=0.9 if is_new else (0.45 if computed else 0.12))
                    row.add(sq)
                row.arrange(RIGHT, buff=0.03)
                g.add(row)
            g.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            return g
        g1, g2 = grid(False), grid(True)
        VGroup(g1, g2).arrange(RIGHT, buff=1.0).move_to(DOWN * 0.1)
        t1 = self.T("without cache: recompute everything", 26, FG).next_to(g1, UP, buff=0.3).align_to(g1, LEFT)
        t2 = self.T("with KV cache: only the new token", 26, YELLOW).next_to(g2, UP, buff=0.3).align_to(g2, LEFT)
        c1 = self.T(f"{kv['no_cache']} positions computed", 28, FG, BOLD).next_to(g1, DOWN, buff=0.35).align_to(g1, LEFT)
        c2 = self.T(f"{kv['with_cache']} positions computed", 28, YELLOW, BOLD).next_to(g2, DOWN, buff=0.35).align_to(g2, LEFT)
        rows_l = self.T(f"each row = one generation step · prompt = {n0} tokens · {n_new} new tokens", 22, MUTED)
        rows_l.to_corner(UR, buff=0.45)
        same_l = self.T(f"same logits either way (max difference {kv['max_diff']:.0e}) · real TinyGPT check", 24, MUTED)
        same_l.to_edge(DOWN, buff=0.35)

        def sw(fill, op, stroke):
            return Square(0.22, fill_color=fill, fill_opacity=op, stroke_color=BLUE, stroke_width=stroke)
        legend = VGroup(VGroup(sw(YELLOW, 0.9, 0), self.T("new token", 22, MUTED)).arrange(RIGHT, buff=0.12),
                        VGroup(sw(FG, 0.45, 0), self.T("computed", 22, MUTED)).arrange(RIGHT, buff=0.12),
                        VGroup(sw(BLUE, 0.12, 1.5), self.T("cached keys & values, reused", 22, MUTED)).arrange(RIGHT, buff=0.12)
                        ).arrange(RIGHT, buff=0.6).next_to(same_l, UP, buff=0.35)
        with self.say(f"Generation has a speed trick: the KV cache. Earlier tokens' keys and values never change, so keep "
                      f"them, and compute only the new token. For {n_new} tokens after our prompt, that's "
                      f"{kv['with_cache']} positions instead of {kv['no_cache']}, same result.",
                      scene=f"Two grids, one row per generation step ({n_new} new tokens after the {n0}-token probe prompt). "
                            f"Without cache every step recomputes all positions ({kv['no_cache']} total); with the KV cache "
                            f"earlier keys/values are reused (blue outlines) and only the new token is computed "
                            f"({kv['with_cache']} total). Real check on TinyGPT: logits match to {kv['max_diff']:.1e}.",
                      chapter="KV cache"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rows_, fmt_l, gen_l)), ReplacementTransform(hdr5, hdr6), run_time=0.5)
            self.play(FadeIn(t1), FadeIn(rows_l), LaggedStart(*[FadeIn(r) for r in g1], lag_ratio=0.15), run_time=1.0)
            self.play(FadeIn(c1), FadeIn(legend), run_time=0.3)
            self.mark("nocache", VGroup(g1, c1), f"without a cache: {kv['no_cache']} token positions pushed through the network")
            self.sync("so keep", 0.1)
            self.play(FadeIn(t2), LaggedStart(*[FadeIn(r) for r in g2], lag_ratio=0.15), run_time=1.0)
            self.mark("cache", VGroup(g2, c2), f"with a KV cache: {kv['with_cache']} positions (prompt once, then 1 per new token)")
            self.sync("For 8", 0.1)
            self.play(FadeIn(c2, scale=1.2), run_time=0.4)
            self.sync("same result", 0.1)
            self.play(FadeIn(same_l), run_time=0.4)
            self.mark("same", same_l, f"cached and uncached generation give the same tokens ('{kv['text'].strip()[:20]}…'), "
                                      f"max logit difference {kv['max_diff']:.1e}")

        # ============================================================ sampling
        hdr7 = self.header("Sampling knobs")
        show = [" red", " white", " purple", " black"]
        ids_ = [VOCAB.index(t) for t in show]
        Tt = ValueTracker(1.0)

        def probs(T):
            z = logits / T
            e = np.exp(z - z.max())
            return e / e.sum()
        ys = [1.6 - 0.75 * k for k in range(len(show))]
        labs = VGroup(*[self.M(self.show_tok(t), 30, COL[t], t2c={"␣": MUTED}).move_to([-3.4, y, 0], aligned_edge=RIGHT)
                        for t, y in zip(show, ys)])

        def bars_now():
            p = probs(Tt.get_value())
            g = VGroup()
            for t, i, y in zip(show, ids_, ys):
                w = max(0.03, 7.0 * p[i])
                r = Rectangle(width=w, height=0.48, stroke_width=0, fill_color=COL[t], fill_opacity=0.85)
                r.move_to([-3.1, y, 0], aligned_edge=LEFT)
                v = Text(f"{100 * p[i]:.1f}%", font=MONO, font_size=26, color=COL[t]).next_to(r, RIGHT, buff=0.15)
                g.add(VGroup(r, v))
            return g
        live_bars = always_redraw(bars_now)
        src = self.T("real TinyGPT probabilities after “…now tom has the”", 24, MUTED).move_to(DOWN * 1.85)
        tl = self.live(lambda: f"temperature {Tt.get_value():.1f}", lambda t: t.move_to([0.4, 2.6, 0]), size=32, color=YELLOW)
        p1 = probs(1.0)
        cum = np.cumsum(sorted(p1, reverse=True))
        k_keep = int(np.searchsorted(cum, 0.9) + 1)
        keep = [VOCAB[i] for i in np.argsort(-p1)[:k_keep]]
        nuc = self.T(f"top-p 0.9 keeps the fewest tokens covering ≥ 90%: {' + '.join(t.strip() for t in keep)} = "
                     f"{100 * cum[k_keep - 1]:.1f}%", 28, TEAL).to_edge(DOWN, buff=0.5)
        self.fitw(nuc, 12.8)
        with self.say(f"Then the sampling knobs. Temperature sharpens or flattens the odds. Top-p keeps the fewest "
                      f"tokens that together cover, say, 90 percent: here, just red and white.",
                      scene=f"Real TinyGPT probabilities after the probe prompt as temperature changes: at T = 0.7 ' red' "
                            f"{100 * probs(0.7)[ids_[0]]:.1f}%, at T = 1.0 {100 * p1[ids_[0]]:.1f}%, at T = 1.3 "
                            f"{100 * probs(1.3)[ids_[0]]:.1f}%. Top-p 0.9 keeps ' red' + ' white' = {100 * cum[k_keep - 1]:.1f}%.",
                      chapter="Sampling"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(g1, g2, t1, t2, c1, c2, rows_l, same_l, legend)), ReplacementTransform(hdr6, hdr7), run_time=0.5)
            self.play(FadeIn(labs), FadeIn(live_bars), FadeIn(tl), FadeIn(src), run_time=0.6)
            for t, l in zip(show, labs):
                self.mark("samp-" + t.strip(), l, f'"{t}": {100 * p1[VOCAB.index(t)]:.1f}% at temperature 1')
            self.sync("sharpens", 0.1)
            self.play(Tt.animate.set_value(0.7), run_time=0.8)
            self.sync("flattens", 0.1)
            self.play(Tt.animate.set_value(1.3), run_time=0.9)
            self.play(Tt.animate.set_value(1.0), run_time=0.5)
            self.sync("Top-p", 0.1)
            self.play(FadeIn(nuc, shift=UP * 0.2), run_time=0.5)
            self.mark("topp", nuc, f"nucleus (top-p 0.9) set: {keep}, cumulative {cum[k_keep - 1]:.4f}")
            self.sync("here, just", 0.1)
            self.play(*[labs[k].animate.set_opacity(0.3) for k in range(k_keep, len(show))], run_time=0.4)
        live_bars.clear_updaters()
        tl.clear_updaters()

        # ============================================================ outro
        pipe2, parts2 = pipeline(self, n_layers="× 96 layers (GPT-3)")
        pipe2.scale(0.9).move_to(UP * 1.7)
        recap = VGroup(self.T("+ a million times the weights", 28, FG), self.T("+ web-scale pretraining", 28, FG),
                       self.T("+ instruction & preference tuning", 28, FG), self.T("+ chat tokens, KV cache, sampling knobs", 28, FG)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(pipe2, DOWN, buff=0.45)
        cta = self.T("open the X-ray: the engine is all there →", 36, YELLOW, BOLD).to_edge(DOWN, buff=0.4)
        with self.say("Same engine: a million times bigger, fed far more text, then taught to be an assistant. "
                      "Everything inside, you've already seen in TinyGPT.",
                      scene="Recap: the same pipeline (tokens → embed → [attention → MLP] × 96 layers for GPT-3 → "
                            "probabilities → pick one) plus: a million times the weights, web-scale pretraining, instruction "
                            "& preference tuning, chat tokens, KV cache and sampling knobs. CTA: open the X-ray.",
                      chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(*[m for m in self.mobjects])), run_time=0.4)
            self.play(FadeIn(pipe2), run_time=0.8)
            self.mark("pipe-96", pipe2, "same loop and blocks; GPT-3 stacks 96 layers instead of 3")
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in recap], lag_ratio=0.3), run_time=1.4)
            self.mark("recap", recap, "what's different: scale, data, post-training, and inference tricks")
            self.sync("Everything inside", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "next: explore TinyGPT's internals in the X-ray")
        self.wait(0.6)
