import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vscene import *  # noqa
from model import VOCAB, MERGES, tok, bpe_steps, pretokenize

PAIR_COUNTS = [(" ", "t", 13558), ("h", "e", 11868), (" ", "a", 10413), ("t", "h", 10372)]  # first 400k chars, step 0


def tc(t):
    """Same token -> same color, everywhere in this video."""
    return CYCLE[VOCAB.index(t) % len(CYCLE)] if t in VOCAB else MUTED


class Tokens(VScene):
    VIDEO_ID = "tokens"

    def tchip(self, t, size=34, show_id=True):
        return self.chip(t, tc(t), size, id_=VOCAB.index(t) if show_id else None, id_size=22, pad=0.13)

    def row_of(self, toks, size=34, buff=0.14):
        return VGroup(*[self.tchip(t, size) for t in toks]).arrange(RIGHT, buff=buff, aligned_edge=UP)

    def construct(self):
        word = " wednesday"
        steps, applied = bpe_steps(word)

        # ------------------------------------------------------------ hook
        big = self.T("wednesday", 96, FG, BOLD).move_to(UP * 1.2)
        model = RoundedRectangle(corner_radius=0.25, width=3.6, height=1.5, fill_color=PANEL, fill_opacity=1,
                                 stroke_color=BLUE, stroke_width=2.5).move_to(DOWN * 1.8)
        mt = self.T("the model", 30, BLUE, BOLD).move_to(model.get_center() + UP * 0.22)
        ms = self.T("only eats numbers", 24, MUTED).move_to(model.get_center() + DOWN * 0.3)
        arr = Arrow(big.get_bottom(), model.get_top(), color=MUTED, buff=0.25, stroke_width=4)
        qs = self.M("[ ?, ?, ? ]", 36, YELLOW).next_to(arr, RIGHT, buff=0.4)
        with self.say("Models can't read letters. They only eat numbers. So how does a word like Wednesday get in?",
                      scene="The word 'wednesday' in big letters above a box labelled 'the model — only eats numbers', "
                            "with an arrow and '[ ?, ?, ? ]' between them.", chapter="The problem"):
            self.play(FadeIn(big, scale=1.2), run_time=0.5)
            self.play(FadeIn(model), FadeIn(mt), FadeIn(ms), run_time=0.5)
            self.sync("So how")
            self.play(GrowArrow(arr), FadeIn(qs, shift=LEFT * 0.2), run_time=0.6)
            self.mark("word", big, 'the word "wednesday"')
            self.mark("model", VGroup(model, mt, ms), "TinyGPT only accepts integer token ids (0–255)")

        # ------------------------------------------------------------ characters
        base = VOCAB[:35]
        grid = VGroup(*[self.chip(t if t not in ("<unk>",) else "<unk>", MUTED if t in ("<unk>", "\n", " ") else
                                  (TEAL if t.isalpha() else ORANGE), 28, id_=i, id_size=20, pad=0.12)
                        for i, t in enumerate(base)])
        grid.arrange_in_grid(rows=4, buff=(0.2, 0.28), col_widths=[0.85] * 9).move_to(DOWN * 0.25)
        hdr = self.header("Characters")
        with self.say("The simplest trick is one number per character. Our model starts with 35 base symbols. "
                      "26 letters, space, newline, six punctuation marks, and an unknown token.",
                      scene="A grid of the 35 base tokens with their ids: <unk> (0), newline (1), space (2), a–z (3–28), "
                            "and . , ? ! : ' (29–34).", chapter="Characters"):
            self.unmark("word", "model")
            self.play(FadeOut(VGroup(model, mt, ms, arr, qs)), FadeOut(big, shift=UP), FadeIn(hdr), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(g, scale=0.6) for g in grid], lag_ratio=0.04), run_time=1.8)
            self.mark("base-unk", grid[0], "token 0: <unk> (unknown character)")
            self.mark("base-nl", grid[1], "token 1: newline")
            self.mark("base-space", grid[2], "token 2: space")
            self.mark("base-letters", VGroup(*grid[3:29]), "tokens 3–28: the letters a–z")
            self.mark("base-punct", VGroup(*grid[29:35]), "tokens 29–34: . , ? ! : '")

        chars = steps[0]
        crow = self.row_of(chars, 40, 0.16).move_to(UP * 0.4)
        with self.say("But then Wednesday costs ten tokens, and the model has to re-spell every common word from "
                      "scratch. Slow, and wasteful.",
                      scene="' wednesday' (with its leading space, shown as ␣) split into 10 single-character chips "
                            "with ids 2, 25, 7, 6, 16, 7, 21, 6, 3, 27. Flash: '10 tokens for one word?!'"):
            self.unmark("base-unk", "base-nl", "base-space", "base-letters", "base-punct")
            self.play(FadeOut(grid), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.3) for c in crow], lag_ratio=0.08), run_time=1.2)
            for k, (c, t) in enumerate(zip(crow, chars)):
                self.mark(f"ch-{k}", c, f'character token "{t}" (id {VOCAB.index(t)})')
            self.sync("re-spell")
            self.flash("10 tokens for one word?!", PINK, 54, hold=0.9, pos=DOWN * 2.2)

        # ------------------------------------------------------------ BPE training
        hdr2 = self.header("Byte-pair encoding")
        SNIP = "tom has a blue boat. tom gives"
        snippet = self.M(SNIP, 34, FG).move_to(UP * 2.45)
        words = pretokenize(SNIP)
        wchips = VGroup(*[self.chip(w, MUTED, 30, fill=0.08) for w in words]).arrange(RIGHT, buff=0.3).move_to(UP * 2.45)
        seps = VGroup(*[Line(UP * 0.36, DOWN * 0.36, color=YELLOW, stroke_width=3).move_to(
            (wchips[i].get_right() + wchips[i + 1].get_left()) / 2) for i in range(len(words) - 1)])
        wlab = self.T("merges never cross a word boundary", 26, YELLOW).next_to(wchips, DOWN, buff=0.35)
        bars = VGroup()
        mx = PAIR_COUNTS[0][2]
        for a, b, n in PAIR_COUNTS:
            lab = self.M(f"{self.show_tok(a)} + {self.show_tok(b)}", 32, FG, t2c={"␣": MUTED})
            bar = Rectangle(width=6.5 * n / mx, height=0.52, fill_color=BLUE, fill_opacity=0.8, stroke_width=0)
            val = self.T(f"{n:,}", 28, BLUE)
            bars.add(VGroup(lab, bar, val))
        for n, r in enumerate(bars):
            y = 0.55 - n * 0.85
            r[0].move_to([-3.8, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-3.4, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.25)
        cap = self.T("most common pairs inside words (first 400k characters)", 26, MUTED).next_to(bars, UP, buff=0.4)
        cap.align_to(bars[0][0], LEFT)
        with self.say("Byte-pair encoding fixes this. First, chop the training text into words, each keeping its "
                      "leading space. Merges never cross from one word into the next.",
                      scene="The corpus snippet 'tom has a blue boat. tom gives' is split into word pieces (tom · ␣has · "
                            "␣a · ␣blue · ␣boat · . · ␣tom · ␣gives), with yellow separators: merges never cross a word "
                            "boundary.", chapter="Byte-pair encoding"):
            self.unmark(*[f"ch-{k}" for k in range(10)])
            self.play(FadeOut(crow), ReplacementTransform(hdr, hdr2), run_time=0.4)
            self.play(FadeIn(snippet, shift=DOWN * 0.2), run_time=0.5)
            self.sync("into words", 0.1)
            self.play(FadeTransform(snippet, wchips), run_time=0.9)
            self.play(LaggedStart(*[Create(x) for x in seps], lag_ratio=0.1), run_time=0.6)
            self.sync("Merges never", 0.1)
            self.play(FadeIn(wlab, shift=UP * 0.15), run_time=0.4)
            for k, w in enumerate(words):
                self.mark(f"word-{k}", wchips[k], f'pre-token "{w}": merges only happen inside it')
            self.mark("word-rule", wlab, "BPE splits text into words first (leading space attached); merges never cross words")

        with self.say("Then count every pair of neighbors inside those words, across the first 400 thousand characters. "
                      "The winner is space t, 13,558 times. Merge it into a brand new token, and repeat.",
                      scene="Bars of the most frequent adjacent pairs inside words, in the first 400k characters of the "
                            "corpus (the part BPE was trained on): '␣+t' 13,558, 'h+e' 11,868, '␣+a' 10,413, 't+h' 10,372. "
                            "The winner becomes new token #35 '␣t'."):
            self.unmark("word-rule")
            self.play(FadeOut(wlab), FadeIn(cap), run_time=0.5)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                    for r in bars], lag_ratio=0.2), run_time=1.6)
            for (a, b, n), r in zip(PAIR_COUNTS, bars):
                self.mark(f"pair-{(a + b).strip() or 'sp'}", r, f'pair "{a}" + "{b}" appears {n:,} times inside words (first 400k characters)')
            self.sync("The winner", 0.1)
            box = SurroundingRectangle(bars[0], color=YELLOW, buff=0.12, corner_radius=0.1)
            self.play(Create(box), bars[0][1].animate.set_fill(YELLOW), run_time=0.5)
            self.sync("Merge it")
            newtok = self.tchip(" t", 40).next_to(bars[0], RIGHT, buff=0.7).shift(DOWN * 0.08)
            if newtok.get_right()[0] > 6.6:
                newtok.shift(LEFT * (newtok.get_right()[0] - 6.6))
            self.play(TransformFromCopy(bars[0][0], newtok), run_time=0.7)
            self.mark("new-t", newtok, 'merge #1 creates token 35: "␣t"')

        # ------------------------------------------------------------ the first rules build " the"
        rules = VGroup()
        for n, (a, b) in enumerate(MERGES[:4]):
            r = VGroup(self.T(f"{n + 1}.", 28, MUTED),
                       self.M(f"{self.show_tok(a)} + {self.show_tok(b)}", 30, FG, t2c={"␣": MUTED}),
                       self.T("→", 28, MUTED), self.tchip(a + b, 30, show_id=False),
                       self.T(f"id {35 + n}", 24, tc(a + b))).arrange(RIGHT, buff=0.25)
            rules.add(r)
        rules.arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to(LEFT * 3.4 + DOWN * 0.2)
        rt = self.T("merge rules, in learned order", 26, MUTED).next_to(rules, UP, buff=0.45).align_to(rules, LEFT)
        st_the = [list(" the"), [" t", "h", "e"], [" t", "he"], [" the"]]
        rows_the = [self.row_of(s, 40).move_to(RIGHT * 3.6 + DOWN * 0.2) for s in st_the]
        with self.say("The first four merges already build 'the', with its space. Each merge is a rule, and the "
                      "rules are kept in the order they were learned.",
                      scene="Left: the first four merge rules (␣+t→␣t id 35, h+e→he id 36, ␣+a→␣a id 37, ␣t+he→␣the id 38). "
                            "Right: '␣the' fusing from 4 characters into one token, id 38."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(wchips, seps, cap, bars, box)), FadeOut(newtok), run_time=0.4)
            self.play(FadeIn(rt), FadeIn(rows_the[0]), run_time=0.4)
            cur = rows_the[0]
            for n in range(4):
                if n > 0:
                    self.wait(0.25)
                self.play(FadeIn(rules[n], shift=RIGHT * 0.2), run_time=0.35)
                if n != 2:
                    nxt = rows_the[[1, 2, None, 3][n]]
                    self.play(ReplacementTransform(cur, nxt), run_time=0.5)
                    cur = nxt
            self.play(Indicate(cur, color=YELLOW, scale_factor=1.15), run_time=0.6)
            for n, (a, b) in enumerate(MERGES[:4]):
                self.mark(f"rule-{n + 1}", rules[n], f'merge rule #{n + 1}: "{a}" + "{b}" → "{a + b}" (id {35 + n})')
            self.mark("the", cur, 'token " the" = id 38, built by rules 1, 2 and 4')

        eq = VGroup(self.T("35", 96, TEAL, BOLD), self.T("+", 80, MUTED), self.T("221", 96, ORANGE, BOLD),
                    self.T("=", 80, MUTED), self.T("256", 96, YELLOW, BOLD)).arrange(RIGHT, buff=0.5).move_to(UP * 0.3)
        eql = VGroup(self.T("base symbols", 26, TEAL), self.T("merges", 26, ORANGE), self.T("vocab size", 26, YELLOW))
        for l, m in zip(eql, [eq[0], eq[2], eq[4]]):
            l.next_to(m, DOWN, buff=0.3)
        with self.say("Do that 221 times, and you get 35 plus 221. A vocabulary of exactly 256 tokens.",
                      scene="Big equation: 35 base symbols + 221 merges = 256 vocab size."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(rules, rt, cur)), run_time=0.4)
            self.play(FadeIn(eq[0]), FadeIn(eql[0]), run_time=0.3)
            self.sync("plus 221", 0.3)
            self.play(FadeIn(eq[1]), FadeIn(eq[2]), FadeIn(eql[1]), run_time=0.4)
            self.sync("A vocabulary", 0.2)
            self.play(FadeIn(eq[3]), FadeIn(eq[4], scale=1.4), FadeIn(eql[2]), run_time=0.5)
            self.mark("eq-base", VGroup(eq[0], eql[0]), "35 base symbols")
            self.mark("eq-merges", VGroup(eq[2], eql[1]), "221 learned merge rules")
            self.mark("eq-vocab", VGroup(eq[4], eql[2]), "vocab size 256 (fits in one byte)")

        # ------------------------------------------------------------ encode wednesday
        hdr3 = self.header("Encoding a word")
        rows = [self.row_of(s, 40, 0.16).move_to(UP * 0.9) for s in steps]
        panel = RoundedRectangle(corner_radius=0.18, width=8.4, height=1.4, fill_color=PANEL, fill_opacity=1,
                                 stroke_color=LINE, stroke_width=2).move_to(DOWN * 1.75)
        with self.say("To encode new text, start from characters and replay the merges in the order they were "
                      "learned. Space w goes first, then a y, then e d. Lowest rule number first, every time.",
                      scene="' wednesday' as 10 character chips; below, a panel shows the merge being applied. Merges "
                            "fire in learned order: ␣+w → ␣w (rank 9), a+y → ay (rank 10), e+d → ed (rank 13).",
                      chapter="Encoding a word"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(eq, eql)), ReplacementTransform(hdr2, hdr3), run_time=0.4)
            self.play(FadeIn(rows[0], shift=UP * 0.2), FadeIn(panel), run_time=0.6)
            self.cur = rows[0]
            self.rule_txt = None
            self._merge_anims(rows, applied, steps, [0, 1, 2], ["Space w", "then a y", "then e d"], panel)
            self.sync("Lowest rule", 0.1)
            self.play(Indicate(self.rule_txt, color=YELLOW, scale_factor=1.08), run_time=0.6)

        with self.say("Then e s. Then d plus ay makes day, and es plus day makes esday. Ten characters, four tokens.",
                      scene="Remaining merges: e+s → es, d+ay → day, es+day → esday. Final result: ␣w (43) · ed (47) · "
                            "n (16) · esday (202) — 4 tokens.", gap=0.3):
            self._merge_anims(rows, applied, steps, [3, 4, 5], ["Then e s", "d plus ay", "es plus day"], panel)
            self.sync("Ten characters", 0.1)
            four = self.T("10 characters → 4 tokens", 40, YELLOW, BOLD).move_to(panel)
            self.play(FadeOut(self.rule_txt), FadeIn(four, scale=1.2), run_time=0.4)
            for k, (c, t) in enumerate(zip(self.cur, steps[-1])):
                self.mark(f"wed-{k}", c, f'token "{t}" (id {VOCAB.index(t)})')
            self.mark("wed-count", four, '" wednesday" → 4 tokens: " w", "ed", "n", "esday"')

        # ------------------------------------------------------------ weird splits
        hdr4 = self.header("Weird splits")
        ex = [" tuesday", " jumping", " red"]
        lines = VGroup()
        for w in ex:
            pieces = [VOCAB[i] for i in tok(w)]
            src = self.M(self.show_tok(w), 36, FG, t2c={"␣": MUTED})
            arr2 = self.T("→", 36, MUTED)
            r = self.row_of(pieces, 36)
            lines.add(VGroup(src, arr2, r))
        for n, l in enumerate(lines):
            y = 1.6 - n * 1.55
            l[0].move_to([-2.6, y, 0], aligned_edge=RIGHT)
            l[1].move_to([-1.9, y, 0])
            l[2].move_to([-1.3, y + 0.0, 0], aligned_edge=LEFT)
            l[2].shift(UP * (y - l[2][0][0].get_y()))
        with self.say("Notice the splits ignore grammar. 'esday' exists because Tuesday and Wednesday share it. "
                      "Jumping becomes space, j, um, p, ing. The model never sees inside a token.",
                      scene="Examples: ' tuesday' → ␣t · u · esday (shares 'esday', id 202, with wednesday); "
                            "' jumping' → ␣ · j · um · p · ing.", chapter="Weird splits"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(self.cur, panel, four)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(FadeIn(lines[0][0]), FadeIn(lines[0][1]), LaggedStart(*[FadeIn(c) for c in lines[0][2]],
                                                                            lag_ratio=0.15), run_time=0.8)
            self.play(Indicate(lines[0][2][-1], color=YELLOW, scale_factor=1.2), run_time=0.6)
            self.sync("Jumping", 0.1)
            self.play(FadeIn(lines[1][0]), FadeIn(lines[1][1]), LaggedStart(*[FadeIn(c) for c in lines[1][2]],
                                                                            lag_ratio=0.15), run_time=0.8)
            for n, w in enumerate(ex[:2]):
                ps = [VOCAB[i] for i in tok(w)]
                self.mark(f"split-{w.strip()}", lines[n],
                          f'"{w}" → ' + " · ".join(f'"{p}" ({VOCAB.index(p)})' for p in ps))

        with self.say("Frequent words get a token all to themselves. Space red is just ID 157, one single chunk.",
                      scene="' red' → a single token, id 157."):
            self.play(FadeIn(lines[2][0]), FadeIn(lines[2][1]), FadeIn(lines[2][2], scale=1.3), run_time=0.7)
            self.sync("157", 0.3)
            self.play(Indicate(lines[2][2], color=PINK, scale_factor=1.25), run_time=0.7)
            self.mark("split-red", lines[2], '" red" → one token, id 157')

        # ------------------------------------------------------------ capitals -> <unk>; real tokenizers use bytes
        cap_ids = tok("Wednesday")
        cap_src = self.M("Wednesday", 36, FG, t2c={"W": PINK}).move_to([-2.6, 0, 0], aligned_edge=RIGHT)
        cap_arr = self.T("→", 36, MUTED).move_to([-1.9, 0, 0])
        cap_row = VGroup(*[self.chip(VOCAB[i], PINK if i == 0 else tc(VOCAB[i]), 36, id_=i, id_size=22, pad=0.13)
                           for i in cap_ids]).arrange(RIGHT, buff=0.14, aligned_edge=UP)
        cap_row.move_to([-1.3, 0, 0], aligned_edge=LEFT)
        cap_row.shift(UP * (0 - cap_row[0][0].get_y()))
        cap_line = VGroup(cap_src, cap_arr, cap_row).shift(DOWN * 0.2)
        byte_note = VGroup(self.T("most real tokenizers start from the 256 possible bytes", 30, TEAL),
                           self.T("so any text can be encoded, and nothing is ever unknown", 26, MUTED)).arrange(
            DOWN, buff=0.15).move_to(DOWN * 2.3)
        with self.say("One catch. This tiny world is all lowercase, so a capital W becomes the unknown token. Most real "
                      "tokenizers start from the 256 possible bytes instead, so nothing is ever unknown.",
                      scene="'Wednesday' with a capital W encodes as <unk> (id 0) · ed · n · esday: TinyGPT's vocab is "
                            "lowercase only. Note: most real tokenizers start from the 256 possible bytes, so nothing is "
                            "ever unknown."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(lines), run_time=0.4)
            self.play(FadeIn(cap_src), FadeIn(cap_arr), LaggedStart(*[FadeIn(c) for c in cap_row], lag_ratio=0.15),
                      run_time=0.8)
            self.play(Indicate(cap_row[0], color=PINK, scale_factor=1.2), run_time=0.6)
            self.mark("cap-unk", cap_row[0], 'capital "W" is not in the lowercase vocab → <unk> (id 0)')
            self.mark("cap-rest", VGroup(*cap_row[1:]), '"ednesday" still encodes normally: "ed" (47) · "n" (16) · "esday" (202)')
            self.sync("Most real", 0.1)
            self.play(FadeIn(byte_note, shift=UP * 0.2), run_time=0.6)
            self.mark("bytes", byte_note, "byte-level BPE (e.g. GPT-2's) starts from 256 byte tokens, so it never needs <unk>")

        # ------------------------------------------------------------ scale
        hdr5 = self.header("Scale")
        data = [("TinyGPT", 256, TEAL), ("GPT-2", 50257, BLUE), ("newer LLMs", 100000, ORANGE)]
        sbars = VGroup()
        for n, (name, v, col) in enumerate(data):
            lab = self.T(name, 30, col, BOLD)
            bar = Rectangle(width=max(0.025, 8.0 * v / 100000), height=0.7, fill_color=col, fill_opacity=0.85,
                            stroke_width=0)
            if n == 2:  # "and up": a fading arrow past the 100k mark instead of guessing a number
                tip = Arrow(bar.get_right() + LEFT * 0.05, bar.get_right() + RIGHT * 0.75, buff=0, color=col,
                            stroke_width=6, max_tip_length_to_length_ratio=0.45)
                bar = VGroup(bar, tip)
            val = self.T(f"{v:,}" if n < 2 else "100k+", 30, col)
            sbars.add(VGroup(lab, bar, val))
        for n, r in enumerate(sbars):
            y = 1.2 - n * 1.25
            r[0].move_to([-4.2, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-3.8, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.3)
        with self.say("Our vocab has 256 entries. GPT-2 had 50,257. Newer models go past 100 thousand. "
                      "Same algorithm, bigger table.",
                      scene="Vocab-size bars to scale: TinyGPT 256 (a sliver), GPT-2 50,257, newer LLMs drawn at "
                            "100,000 with an arrow for 'and up'. "
                            "Flash: 'same algorithm, bigger table'.", chapter="Scale"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(cap_line, byte_note)), ReplacementTransform(hdr4, hdr5), run_time=0.4)
            for n, ph in enumerate(["Our vocab", "GPT-2", "Newer models"]):
                self.sync(ph, 0.1)
                self.play(FadeIn(sbars[n][0]), GrowFromEdge(sbars[n][1], LEFT), FadeIn(sbars[n][2]), run_time=0.6)
                self.mark(f"vocab-{n}", sbars[n], f"{data[n][0]} vocab size: {data[n][1]:,}" if n < 2
                          else "newer LLMs: vocabularies of 100,000 tokens and up (bar drawn at 100,000)")
            self.sync("Same algorithm")
            self.flash("same algorithm, bigger table", YELLOW, 52, hold=0.8, pos=DOWN * 2.6)

        left = VGroup(self.T("bigger vocab", 34, YELLOW, BOLD),
                      self.T("↓ fewer tokens per sentence", 30, TEAL),
                      self.T("↑ bigger embedding table", 30, PINK)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        left.move_to(UP * 1.7)
        tbl = VGroup(self.T("embedding table = vocab × width", 26, MUTED),
                     self.M("TinyGPT   256 × 64  =     16,384", 28, TEAL),
                     self.M("GPT-2  50,257 × 768 = 38,597,376", 28, BLUE)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        tbl.next_to(left, DOWN, buff=0.8)
        knob = self.T("a knob you'll turn when you build your own", 34, YELLOW).to_edge(DOWN, buff=0.8)
        with self.say("A bigger vocab means shorter sequences, but a bigger embedding table. It's a knob you'll "
                      "get to turn when you build your own.",
                      scene="Trade-off: bigger vocab → fewer tokens per sentence but a bigger embedding table "
                            "(TinyGPT 256×64 = 16,384 numbers; GPT-2 50,257×768 = 38.6 million)."):
            self.unmark(*list(self._marks))
            self.play(FadeOut(sbars), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.2) for m in left], lag_ratio=0.3), run_time=1.0)
            self.play(FadeIn(tbl, shift=LEFT * 0.2), run_time=0.6)
            self.mark("tradeoff", left, "trade-off: bigger vocab = shorter sequences but more embedding parameters")
            self.mark("emb-tiny", tbl[1], "TinyGPT embedding table: 256 × 64 = 16,384 numbers")
            self.mark("emb-gpt2", tbl[2], "GPT-2 (small) embedding table: 50,257 × 768 = 38,597,376 numbers")
            self.sync("It's a knob", 0.2)
            self.play(FadeIn(knob, shift=UP * 0.2), run_time=0.5)
        self.wait(0.5)

    def _merge_anims(self, rows, applied, steps, which, phrases, panel):
        for n, ph in zip(which, phrases):
            m = applied[n]
            i = m["pos"]
            cur = self.cur
            pair = VGroup(cur[i], cur[i + 1])
            hl = SurroundingRectangle(pair, color=YELLOW, buff=0.08, corner_radius=0.1)
            a, b = m["pair"]
            txt = VGroup(self.T(f"rule #{m['rank'] + 1}:", 30, MUTED),
                         self.M(f"{self.show_tok(a)} + {self.show_tok(b)}", 34, FG, t2c={"␣": MUTED}),
                         self.T("→", 32, MUTED),
                         self.M(self.show_tok(m['new']), 34, tc(m['new']), t2c={"␣": MUTED}),
                         self.T(f"(id {m['id']})", 28, tc(m['new']))).arrange(RIGHT, buff=0.25).move_to(panel)
            try:
                self.sync(ph, 0.1)
            except ValueError:
                pass
            anims = [Create(hl)]
            anims.append(FadeOut(self.rule_txt) if self.rule_txt is not None else Wait(0))
            self.play(*anims, FadeIn(txt), run_time=0.35)
            nxt = rows[n + 1]
            # keep untouched chips in place, fuse the pair
            mapping = []
            for k in range(len(nxt)):
                if k < i:
                    mapping.append(ReplacementTransform(cur[k], nxt[k]))
                elif k == i:
                    mapping.append(ReplacementTransform(VGroup(cur[i], cur[i + 1]), nxt[k]))
                else:
                    mapping.append(ReplacementTransform(cur[k + 1], nxt[k]))
            self.play(*mapping, FadeOut(hl), run_time=0.45)
            self.cur = nxt
            self.rule_txt = txt
            self.mark("rule-now", txt, f'merge rule #{m["rank"] + 1}: "{a}" + "{b}" → "{m["new"]}" (id {m["id"]})')
            for k, (c, t) in enumerate(zip(nxt, steps[n + 1])):
                self.mark(f"wed-{k}", c, f'token "{t}" (id {VOCAB.index(t)})')
            for k in range(len(nxt), 11):
                self.unmark(f"wed-{k}")
