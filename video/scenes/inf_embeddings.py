import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vscene import *  # noqa
from lib_inference import (VOCAB, E, P, GROUPS, tid, neighbours, cos_matrix, pca2, group_nn_accuracy, params_count)

VIDEO = dict(id="embeddings", cls="Embeddings", title="Embeddings: words as directions",
             subtitle="The lookup table that turns token ids into 64 numbers, and the meaning hiding in them",
             relates="xray", track="inference", order=30, poster=("The map", 1, 0.9))

GCOL = {"names": TEAL, "colors": PINK, "animals": ORANGE, "objects": YELLOW, "places": BLUE}


def div_rgb(v, vmax):
    """Diverging colour (blue negative, orange positive) as an (r, g, b) uint8 array."""
    a = np.clip(v / vmax, -1, 1)
    bg = np.array(color_to_rgb(BG)); o = np.array(color_to_rgb(ORANGE)); b = np.array(color_to_rgb(BLUE))
    t = (np.abs(a) ** 0.8)[..., None]
    tgt = np.where(a[..., None] > 0, o, b)
    return (255 * (bg + (tgt - bg) * t)).astype(np.uint8)


def div_color(v, vmax):
    a = float(np.clip(v / vmax, -1, 1))
    return interpolate_color(ManimColor(BG), ManimColor(ORANGE if a > 0 else BLUE), abs(a) ** 0.8)


class Embeddings(VScene):
    VIDEO_ID = "embeddings"

    def heat(self, M, vmax, w, h):
        img = ImageMobject(div_rgb(np.asarray(M), vmax))
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.stretch_to_fit_width(w).stretch_to_fit_height(h)
        return img

    def strip(self, vec, vmax, cw=0.16, ch=0.5):
        cells = VGroup(*[Rectangle(width=cw, height=ch, stroke_width=0.6, stroke_color=BG,
                                   fill_color=div_color(v, vmax), fill_opacity=1) for v in vec])
        return cells.arrange(RIGHT, buff=0)

    def construct(self):
        En, Pn = E.numpy(), P.numpy()
        vmax = float(np.quantile(np.abs(En), 0.98))
        RED = tid(" red")
        e_red = En[RED]

        # ------------------------------------------------------------ hook: an id is just a number
        chip = self.chip(" red", PINK, 48, id_=RED, id_size=34).move_to(UP * 0.4)
        q = self.T("what does 157 mean?", 36, MUTED).next_to(chip, DOWN, buff=0.7)
        with self.say(f"To this model, red is just token {RED}. A row number. "
                      f"So how does a number come to mean anything?",
                      scene=f"A big token chip '␣red' with its id {RED} underneath and the question 'what does {RED} mean?'.",
                      chapter="Just a number"):
            self.play(FadeIn(chip, scale=1.2), run_time=0.6)
            self.mark("chip", chip, f'token " red" has id {RED} in the 256-token vocabulary')
            self.sync("So how", 0.1)
            self.play(FadeIn(q, shift=UP * 0.2), run_time=0.5)

        # ------------------------------------------------------------ lookup table
        hdr = self.header("1 · The lookup table")
        tw, th = 4.6, 6.0
        table = self.heat(En, vmax, tw, th).move_to(LEFT * 3.2 + DOWN * 0.35)
        frame_t = SurroundingRectangle(table, buff=0.02, stroke_color=LINE, stroke_width=2)
        rows_l = self.T("256 rows · one per token", 24, MUTED).rotate(PI / 2).next_to(table, LEFT, buff=0.25)
        cols_l = self.T("64 numbers per row", 24, MUTED).next_to(table, UP, buff=0.18)
        ry = table.get_top()[1] - th * (RED + 0.5) / 256
        hl = Rectangle(width=tw + 0.2, height=th / 256 * 3, stroke_color=PINK, stroke_width=3).move_to([table.get_x(), ry, 0])
        s_red = self.strip(e_red, vmax, cw=0.095, ch=0.6).move_to(RIGHT * 3.4 + UP * 0.6)
        s_lab = self.M(f'row {RED}: "␣red"', 26, PINK).next_to(s_red, UP, buff=0.25)
        nums = VGroup(*[self.M(f"{v:+.2f}", 24, FG) for v in e_red[:4]], self.M("…", 24, MUTED),
                      self.M(f"{e_red[-1]:+.2f}", 24, FG)).arrange(RIGHT, buff=0.22).next_to(s_red, DOWN, buff=0.3)
        n_tab = En.size
        with self.say(f"The very first step is a lookup table. 256 rows, one per token, 64 numbers in each row. "
                      f"Token {RED} just grabs row {RED}. That row is its embedding.",
                      scene=f"The real 256×64 embedding table as a heatmap (orange positive, blue negative). Row {RED} "
                            f"is outlined and slides out as a strip of 64 cells; its first values are "
                            f"{', '.join(f'{v:+.2f}' for v in e_red[:4])} … {e_red[-1]:+.2f}.",
                      chapter="The lookup table"):
            self.play(FadeOut(q), chip.animate.scale(0.6).move_to(RIGHT * 3.4 + UP * 2.6), FadeIn(hdr), run_time=0.6)
            self.play(FadeIn(table), Create(frame_t), FadeIn(rows_l), FadeIn(cols_l), run_time=0.9)
            self.mark("table", Group(table, frame_t), f"the real token-embedding table: 256 rows × 64 columns = {n_tab:,} learned numbers")
            self.sync(f"grabs row", 0.1)
            self.play(Create(hl), run_time=0.4)
            self.play(TransformFromCopy(hl, s_red), FadeIn(s_lab), run_time=0.9)
            self.play(FadeIn(nums, shift=DOWN * 0.15), run_time=0.4)
            self.mark("row-red", VGroup(s_red, s_lab, nums),
                      f'embedding of " red" (row {RED}): 64 numbers from {e_red.min():+.2f} to {e_red.max():+.2f}, length {np.linalg.norm(e_red):.2f}')

        with self.say(f"All {n_tab:,} numbers started as random noise. And the same table is reused at the very end, "
                      f"to score every possible next token. So training pulls tokens that fit the same slots toward similar rows.",
                      scene=f"Same view; captions: 'started random → shaped by training' and 'reused at the end to score every "
                            f"next token (weight tying)'. The table holds {n_tab:,} of the model's {params_count():,} parameters."):
            cap = self.T("started random → shaped by training", 26, YELLOW).next_to(nums, DOWN, buff=0.7)
            cap2 = VGroup(self.T("also reused at the end to score", 26, FG),
                          self.T("every next token (weight tying)", 26, FG)).arrange(DOWN, buff=0.1).next_to(cap, DOWN, buff=0.35)
            self.play(FadeIn(cap, shift=UP * 0.2), run_time=0.5)
            self.sync("And the same", 0.1)
            self.play(FadeIn(cap2, shift=UP * 0.2), run_time=0.5)
            self.mark("tying", cap2, "weight tying: logits = final vector · every row of this same embedding table")
            self.sync("So training", 0.1)
            self.play(Indicate(table, scale_factor=1.03, color=YELLOW), run_time=0.8)

        # ------------------------------------------------------------ what a vector means: cosine
        hdr2 = self.header("2 · Meaning = direction")
        U = 1.1
        org = LEFT * 3.6 + DOWN * 0.5
        ax = VGroup(Line(org + LEFT * 2.2 * U, org + RIGHT * 2.2 * U, color=LINE, stroke_width=2),
                    Line(org + DOWN * 2.0 * U, org + UP * 2.2 * U, color=LINE, stroke_width=2))
        toy = self.T("toy 2-D picture", 24, MUTED).next_to(ax, UP, buff=0.2)

        def arr(ang, col, L=2.0):
            return Arrow(org, org + U * L * np.array([np.cos(ang), np.sin(ang), 0]), buff=0, color=col,
                         stroke_width=6, max_tip_length_to_length_ratio=0.15)
        a1 = arr(0.5, PINK); a2 = arr(0.75, "#b48cff"); a3 = arr(2.4, TEAL)
        l1 = self.M("␣red", 26, PINK).next_to(a1.get_end(), RIGHT, buff=0.1)
        l2 = self.M("␣purple", 26, "#b48cff").next_to(a2.get_end(), UP, buff=0.1)
        l3 = self.M("␣tom", 26, TEAL).next_to(a3.get_end(), LEFT, buff=0.1)
        form = VGroup(self.T("cosine similarity", 36, FG, BOLD),
                      self.T("= how aligned two arrows are", 28, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        scale = VGroup(self.M("+1  same way", 28, TEAL), self.M(" 0  unrelated", 28, MUTED),
                       self.M("−1  opposite", 28, ORANGE)).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        fs = VGroup(form, scale).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(RIGHT * 2.8 + DOWN * 0.3)
        S = cos_matrix(En)
        glob = float(S[~np.eye(256, dtype=bool)].mean())
        with self.say("No single number means red. What matters is the direction the whole row points. "
                      "We compare directions with cosine similarity: one means same way, zero means unrelated.",
                      scene="Toy 2-D picture: arrows for ' red' and ' purple' point almost the same way, ' tom' points "
                            "elsewhere. Text: cosine similarity = how aligned two arrows are (+1 same way, 0 unrelated, −1 opposite).",
                      chapter="Meaning = direction"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(table, frame_t, rows_l, cols_l, hl, s_red, s_lab, nums, cap, cap2, chip)),
                      ReplacementTransform(hdr, hdr2), run_time=0.6)
            self.play(Create(ax), FadeIn(toy), GrowArrow(a1), FadeIn(l1), run_time=0.7)
            self.sync("direction", 0.1)
            self.play(GrowArrow(a2), FadeIn(l2), GrowArrow(a3), FadeIn(l3), run_time=0.7)
            self.mark("toy", VGroup(ax, a1, a2, a3, l1, l2, l3, toy), "toy 2-D illustration of embedding directions (not real values)")
            self.sync("We compare", 0.1)
            self.play(FadeIn(form, shift=UP * 0.2), run_time=0.5)
            self.sync("one means", 0.1)
            self.play(FadeIn(scale, lag_ratio=0.3), run_time=0.8)
            self.mark("cos", fs, "cosine similarity: +1 same direction, 0 unrelated, −1 opposite")

        # real neighbours
        probes = [(" red", PINK), (" tom", TEAL), (" cat", ORANGE), (" park", BLUE)]
        cols = VGroup()
        for t, c in probes:
            head = self.chip(t, c, 30)
            rows = VGroup()
            for w, s in neighbours(t, 4):
                rows.add(VGroup(self.M(self.show_tok(w), 26, FG, t2c={"␣": MUTED}),
                                self.M(f"{s:.2f}", 26, c)).arrange(RIGHT, buff=0.3))
            rows.arrange(DOWN, aligned_edge=RIGHT, buff=0.22)
            cols.add(VGroup(head, rows).arrange(DOWN, buff=0.35))
        cols.arrange(RIGHT, buff=0.75, aligned_edge=UP).move_to(DOWN * 0.15)
        sub = self.T("nearest neighbours by cosine similarity (real, all 64 dimensions)", 26, MUTED).next_to(cols, UP, buff=0.5)
        foot = self.T(f"average pair of tokens: {glob:.2f}", 26, MUTED).next_to(cols, DOWN, buff=0.55)
        nr = neighbours(" red", 3)
        with self.say(f"Now the real thing. Red's nearest neighbours: purple at {nr[0][1]:.2f}, white, black. "
                      f"Tom's neighbours are names. Cat's are animals. Park's are places. "
                      f"A random pair of tokens scores about {glob:.1f}.",
                      scene="Four real nearest-neighbour lists (cosine similarity over all 64 dims): "
                            + "; ".join(f"'{t.strip()}' → " + ", ".join(f"{w.strip()} {s:.2f}" for w, s in neighbours(t, 4))
                                        for t, _ in probes) + f". Average token pair: {glob:.2f}."):
            self.unmark("toy", "cos")
            self.play(FadeOut(VGroup(ax, toy, a1, a2, a3, l1, l2, l3, fs)), run_time=0.5)
            self.play(FadeIn(sub), FadeIn(cols[0], shift=UP * 0.2), run_time=0.6)
            for k, (t, c) in enumerate(probes):
                if k:
                    self.sync(["", "Tom", "Cat", "Park"][k], 0.1)
                    self.play(FadeIn(cols[k], shift=UP * 0.2), run_time=0.4)
                self.mark(f"nn-{t.strip()}", cols[k], f'nearest neighbours of "{t}": ' +
                          ", ".join(f'"{w}" {s:.3f}' for w, s in neighbours(t, 4)))
            self.sync("A random", 0.1)
            self.play(FadeIn(foot, shift=UP * 0.2), run_time=0.4)
            self.mark("avg", foot, f"mean cosine similarity over all pairs of the 256 tokens = {glob:.3f}")

        # ------------------------------------------------------------ PCA map
        hdr3 = self.header("3 · The map")
        Y, var = pca2(En)
        Y = Y / np.abs(Y).max()
        pw, ph = 6.4, 5.2
        ctr = RIGHT * 2.1 + DOWN * 0.1
        frame = RoundedRectangle(corner_radius=0.2, width=pw + 0.7, height=ph + 0.6, stroke_color=LINE, stroke_width=2,
                                 fill_color=PANEL, fill_opacity=0.6).move_to(ctr)

        def Pt(y):
            return ctr + RIGHT * y[0] * pw / 2 + UP * y[1] * ph / 2

        gof = {tid(w): g for g, l in GROUPS.items() for w in l}
        bg_dots = VGroup(*[Dot(Pt(Y[i]), radius=0.04, color=MUTED, fill_opacity=0.5) for i in range(256) if i not in gof])
        gdots = {g: VGroup(*[Dot(Pt(Y[tid(w)]), radius=0.08, color=GCOL[g]) for w in l]) for g, l in GROUPS.items()}
        key = VGroup(*[VGroup(Dot(radius=0.09, color=GCOL[g]), self.T(f"{g} ({len(GROUPS[g])})", 26, GCOL[g])).arrange(RIGHT, buff=0.18)
                       for g in GROUPS], VGroup(Dot(radius=0.06, color=MUTED), self.T("other tokens", 26, MUTED)).arrange(RIGHT, buff=0.2))
        key.arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.8).shift(DOWN * 0.3)
        ttl = self.T("all 256 embeddings, squashed from 64-D to 2-D (PCA)", 24, MUTED).next_to(frame, UP, buff=0.15)
        vlab = self.T(f"these 2 axes keep {100 * (var[0] + var[1]):.0f}% of the spread", 24, MUTED).next_to(frame, DOWN, buff=0.15)
        names_c = Y[[tid(w) for w in GROUPS["names"]]].mean(0)
        with self.say("Squash all 256 rows from 64 dimensions down to two, with PCA, and plot them. "
                      "The names split off into their own island. Colors, animals, objects and places crowd together.",
                      scene=f"Real 2-D PCA of all 256 token embeddings (axes keep {100 * (var[0] + var[1]):.1f}% of the variance). "
                            "Names (teal) form a separate island at the bottom; colors, animals, objects and places bunch "
                            "together on the right; other tokens are faint gray dots.",
                      chapter="The map"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(cols, sub, foot)), ReplacementTransform(hdr2, hdr3), run_time=0.5)
            self.play(FadeIn(frame), FadeIn(ttl), FadeIn(vlab),
                      LaggedStart(*[FadeIn(d, scale=0.3) for d in bg_dots], lag_ratio=0.003), run_time=1.1)
            self.play(LaggedStart(*[FadeIn(d, scale=0.3) for g in GROUPS for d in gdots[g]], lag_ratio=0.01),
                      run_time=0.8)
            self.mark("pca", frame, f"2-D PCA of the 256 real token embeddings ({100 * (var[0] + var[1]):.1f}% of variance)")
            self.sync("names split", 0.1)
            self.play(FadeIn(key[0]), *[Indicate(d, color=TEAL, scale_factor=1.6) for d in gdots["names"]], run_time=0.8)
            self.mark("pca-names", gdots["names"], "the 12 name tokens (tom, mia, zoe, …) form their own island")
            self.sync("Colors", 0.1)
            self.play(FadeIn(key[1:]), run_time=0.6)
            for g in ("colors", "animals", "objects", "places"):
                self.mark(f"pca-{g}", gdots[g], f"{len(GROUPS[g])} {g} tokens in the 2-D PCA map")

        # zoom: PCA of only the 48 group words
        words = [(w, g) for g, l in GROUPS.items() for w in l]
        Z, zvar = pca2(En[[tid(w) for w, _ in words]])
        Z = Z / np.abs(Z).max()
        zmid = (Z.max(0) + Z.min(0)) / 2
        zs = 0.72 * min(2 / (Z.max(0) - Z.min(0)))
        zpos = [Pt((z - zmid) * zs) for z in Z]
        zl = {}
        gi = 0
        fx0, fx1 = frame.get_left()[0] + 0.15, frame.get_right()[0] - 0.15
        fy0, fy1 = frame.get_bottom()[1] + 0.15, frame.get_top()[1] - 0.15
        boxes = []                      # occupied rectangles (x0, y0, x1, y1): clusters + placed labels
        clus = {}
        for g, l in GROUPS.items():
            pts = np.array(zpos[gi:gi + len(l)]); gi += len(l)
            clus[g] = pts
            boxes.append((pts[:, 0].min() - 0.1, pts[:, 1].min() - 0.1, pts[:, 0].max() + 0.1, pts[:, 1].max() + 0.1))

        def free(bx):
            if bx[0] < fx0 or bx[2] > fx1 or bx[1] < fy0 or bx[3] > fy1:
                return False
            return all(bx[2] < o[0] or bx[0] > o[2] or bx[3] < o[1] or bx[1] > o[3] for o in boxes)

        for g, pts in clus.items():
            lab = self.T(g, 28, GCOL[g], BOLD)
            w2, h2 = lab.width / 2, lab.height / 2
            x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cands = [(cx, y1 + 0.18 + h2), (cx, y0 - 0.18 - h2), (x1 + 0.18 + w2, cy), (x0 - 0.18 - w2, cy),
                     (x1 + 0.1 + w2, y1 + 0.1 + h2), (x0 - 0.1 - w2, y1 + 0.1 + h2),
                     (x1 + 0.1 + w2, y0 - 0.1 - h2), (x0 - 0.1 - w2, y0 - 0.1 - h2)]
            for k in range(1, 4):
                cands += [(cx, y1 + 0.18 + h2 + 0.3 * k), (cx, y0 - 0.18 - h2 - 0.3 * k)]
            pos = next(((x, y) for x, y in cands if free((x - w2, y - h2, x + w2, y + h2))), cands[0])
            boxes.append((pos[0] - w2, pos[1] - h2, pos[0] + w2, pos[1] + h2))
            lab.move_to([pos[0], pos[1], 0])
            zl[g] = lab
        ok, tot = group_nn_accuracy()
        ttl2 = self.T(f"zoom: PCA of just these {tot} words", 26, MUTED).next_to(frame, UP, buff=0.15)
        vlab2 = self.T(f"2 axes keep {100 * (zvar[0] + zvar[1]):.0f}% of their spread", 24, MUTED).next_to(frame, DOWN, buff=0.15)
        verdict = VGroup(self.T(f"{ok} / {tot}", 44, YELLOW, BOLD),
                         VGroup(*[self.T(t_, 24, FG) for t_ in ("have their nearest", "neighbour in their", "own group (64-D)")]
                                ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        with self.say(f"Re-run PCA on just these {tot} words. Names sit far away. Colors and places get "
                      f"their own patches. Animals and objects overlap: they fill the same slot, nouns after 'the', and often after a color. "
                      f"And in the full 64 dimensions, {ok} out of {tot} words have their nearest neighbour in their own group.",
                      scene=f"Real PCA of only the {tot} group words (names, colors, animals, objects, places): names far off, colors "
                            f"and places in their own patches, animals and objects overlapping (both fill the same noun slot: after 'the', often after a color). Verdict: {ok}/{tot} words have their nearest "
                            "cosine neighbour (in 64-D) in their own group."):
            self.unmark("pca-names", "pca-colors", "pca-animals", "pca-objects", "pca-places")
            moves = []
            k = 0
            for g, l in GROUPS.items():
                for j, w in enumerate(l):
                    moves.append(gdots[g][j].animate.move_to(zpos[k]).scale(0.85))
                    k += 1
            self.play(FadeOut(bg_dots), ReplacementTransform(ttl, ttl2), ReplacementTransform(vlab, vlab2),
                      FadeOut(key[-1]), *moves, run_time=1.4)
            self.play(LaggedStart(*[FadeIn(zl[g], scale=1.2) for g in GROUPS], lag_ratio=0.15), run_time=0.8)
            for g, l in GROUPS.items():
                self.mark(f"zoom-{g}", VGroup(gdots[g], zl[g]), f"{g}: " + ", ".join(w.strip() for w in l))
            self.sync("And in the full", 0.1)
            verdict.next_to(key[:-1], DOWN, buff=0.5).align_to(key, LEFT)
            self.play(FadeIn(verdict, shift=UP * 0.2), run_time=0.5)
            self.mark("verdict", verdict, f"leave-one-out: {ok} of {tot} words have their nearest cosine neighbour in the same group")
            self.play(Indicate(verdict[0], color=YELLOW, scale_factor=1.3), run_time=0.7)

        # ------------------------------------------------------------ position embeddings
        hdr4 = self.header("4 · Position")
        k_pos = 3
        e_pos = Pn[k_pos]
        e_sum = e_red + e_pos
        s1 = self.strip(e_red, vmax, cw=0.1, ch=0.45)
        s2 = self.strip(e_pos, vmax, cw=0.1, ch=0.45)
        s3 = self.strip(e_sum, vmax, cw=0.1, ch=0.45)
        g3 = VGroup(s1, s2, s3).arrange(DOWN, buff=0.42).move_to(RIGHT * 1.9 + UP * 1.35)
        lab1 = self.M(f'token "␣red"  length {np.linalg.norm(e_red):.2f}', 24, PINK).next_to(s1, LEFT, buff=0.3)
        lab2 = self.M(f"+ position {k_pos}  length {np.linalg.norm(e_pos):.2f}", 24, MUTED).next_to(s2, LEFT, buff=0.3)
        lab3 = self.M("= what layer 1 sees", 24, FG).next_to(s3, LEFT, buff=0.3)
        tnorm = float(np.linalg.norm(En, axis=1).mean()); pnorm = float(np.linalg.norm(Pn, axis=1).mean())
        with self.say(f"But red at position 3 and red at position 20 would look identical. "
                      f"So a second table holds one learned vector per position, and it's simply added on.",
                      scene=f"Three real 64-cell strips: the ' red' token embedding (length {np.linalg.norm(e_red):.2f}) + the "
                            f"learned position-{k_pos} embedding (length {np.linalg.norm(e_pos):.2f}) = the vector layer 1 actually sees.",
                      chapter="Position"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(frame, ttl2, vlab2, key[:-1], verdict, *gdots.values(), *zl.values())),
                      ReplacementTransform(hdr3, hdr4), run_time=0.5)
            self.play(FadeIn(lab1), LaggedStart(*[FadeIn(c) for c in s1], lag_ratio=0.01), run_time=0.7)
            self.mark("pos-tok", VGroup(lab1, s1), f'token embedding of " red" (length {np.linalg.norm(e_red):.3f})')
            self.sync("So a second", 0.1)
            self.play(FadeIn(lab2), LaggedStart(*[FadeIn(c) for c in s2], lag_ratio=0.01), run_time=0.7)
            self.mark("pos-vec", VGroup(lab2, s2), f"learned position embedding for slot {k_pos} (length {np.linalg.norm(e_pos):.3f})")
            self.sync("and it's", 0.1)
            self.play(FadeIn(lab3), TransformFromCopy(VGroup(s1, s2), s3), run_time=0.8)
            self.mark("pos-sum", VGroup(lab3, s3), f'input to layer 1 for " red" at position {k_pos} = token + position')

        SP = cos_matrix(Pn)
        hm = self.heat(SP, 1.0, 3.2, 3.2).move_to(LEFT * 3.6 + DOWN * 1.75)
        hm_f = SurroundingRectangle(hm, buff=0.02, stroke_color=LINE, stroke_width=2)
        hm_t = self.T("position vs position (cosine), 0 → 63", 24, MUTED).next_to(hm, UP, buff=0.12)
        nb = float(np.mean([SP[i, i + 1] for i in range(63)]))
        far = float(SP[0].min()); farj = int(SP[0].argmin())
        facts = VGroup(self.T(f"next-door positions: {nb:.2f} on average", 28, ORANGE),
                       self.T(f"position 0 vs position {farj}: {far:+.2f}", 28, BLUE),
                       self.T(f"average length: {pnorm:.2f} (position) vs {tnorm:.2f} (token)", 26, MUTED)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(hm, RIGHT, buff=0.7).shift(DOWN * 0.1)
        with self.say(f"Compare all 64 position vectors with each other. Next-door positions point almost the same way, "
                      f"about {nb:.2f}. Far-apart ones drift apart, even opposite. A smooth, learned sense of where.",
                      scene=f"Real 64×64 cosine-similarity heatmap of the learned position embeddings: a bright diagonal band "
                            f"(neighbours average {nb:.2f}) fading to blue far from it (position 0 vs {farj}: {far:+.2f}). "
                            f"Position vectors average length {pnorm:.2f} vs {tnorm:.2f} for token vectors."):
            self.play(VGroup(lab1, s1, lab2, s2, lab3, s3).animate.shift(UP * 0.35), run_time=0.4)
            self.play(FadeIn(hm), Create(hm_f), FadeIn(hm_t), run_time=0.8)
            self.mark("pos-heat", Group(hm, hm_f), f"cosine similarity between the 64 learned position embeddings; neighbours ≈ {nb:.2f}")
            self.sync("Next-door", 0.1)
            self.play(FadeIn(facts[0], shift=UP * 0.2), run_time=0.4)
            self.sync("Far-apart", 0.1)
            self.play(FadeIn(facts[1], shift=UP * 0.2), FadeIn(facts[2], shift=UP * 0.2), run_time=0.5)
            self.mark("pos-facts", facts, f"neighbouring positions {nb:.3f}; position 0 vs {farj} = {far:.3f}; "
                                          f"mean |position vector| {pnorm:.3f} vs |token vector| {tnorm:.3f}")

        # ------------------------------------------------------------ scale
        hdr5 = self.header("5 · Bigger models")
        small = VGroup(*[Square(0.11, stroke_width=0, fill_color=TEAL, fill_opacity=0.9) for _ in range(64)])
        small.arrange_in_grid(rows=1, cols=64, buff=0.02)
        big = VGroup(*[Square(0.11, stroke_width=0, fill_color=YELLOW, fill_opacity=0.8) for _ in range(64 * 30)])
        big.arrange_in_grid(rows=30, cols=64, buff=0.02)
        small_l = VGroup(self.T("TinyGPT", 30, TEAL, BOLD), self.M("64 numbers per token", 26, TEAL)).arrange(RIGHT, buff=0.35)
        big_l = VGroup(self.T("GPT-3 (largest)", 30, YELLOW, BOLD), self.M("12,288 per token", 26, YELLOW)).arrange(RIGHT, buff=0.35)
        col = VGroup(VGroup(small_l, small).arrange(DOWN, aligned_edge=LEFT, buff=0.2),
                     VGroup(big_l, big).arrange(DOWN, aligned_edge=LEFT, buff=0.2)).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        col.move_to(DOWN * 0.1)
        more = self.T(f"…and {12288 // 64 - 30} more rows like these", 24, MUTED).next_to(big, DOWN, buff=0.15).align_to(big, LEFT)
        with self.say("Sixty-four numbers is tiny. GPT-3's largest version uses 12,288 per token, 192 times the room. "
                      "More directions means more distinctions it can keep apart at once.",
                      scene="TinyGPT's 64-number embedding as one row of 64 squares vs GPT-3's (largest) 12,288: 30 visible rows "
                            "of 64 squares and a note that 162 more rows would be needed (192× more room).",
                      chapter="Bigger models"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(Group(hm, hm_f, hm_t, facts, lab1, s1, lab2, s2, lab3, s3)),
                      ReplacementTransform(hdr4, hdr5), run_time=0.5)
            self.play(FadeIn(small_l), FadeIn(small, lag_ratio=0.02), run_time=0.6)
            self.mark("tiny", VGroup(small_l, small), "TinyGPT: d_model = 64 numbers per token")
            self.sync("GPT-3", 0.1)
            self.play(FadeIn(big_l), FadeIn(big, lag_ratio=0.0005), run_time=1.0)
            self.play(FadeIn(more), run_time=0.3)
            self.mark("gpt3", VGroup(big_l, big, more), "GPT-3 (175B version): 12,288 numbers per token = 192 × 64")
            self.sync("192", 0.1)
            self.flash("192× the room", YELLOW, 56, hold=0.7, pos=DOWN * 0.3)

        # ------------------------------------------------------------ outro
        cta = self.T("now find them in the X-ray →", 46, YELLOW, BOLD).move_to(DOWN * 2.3)
        recap = VGroup(self.chip(" red", PINK, 34, id_=RED, id_size=26), self.T("→", 40, MUTED),
                       self.strip(e_red, vmax, cw=0.08, ch=0.5), self.T("+ position →", 30, MUTED),
                       self.T("a direction", 34, FG, BOLD)).arrange(RIGHT, buff=0.3).move_to(UP * 0.4)
        with self.say("An embedding is a learned row of numbers whose meaning lives in its direction. "
                      "Go find them in the X-ray.",
                      scene="Recap: token id 157 → its 64-number row (+ position) → a direction in space. Card: 'now find them in the X-ray'.",
                      chapter="Recap"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(small_l, small, big_l, big, more, hdr5)), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.2) for m in recap], lag_ratio=0.2), run_time=1.2)
            self.mark("recap", recap, "token id → embedding row (64 numbers) + position vector → a direction in 64-D space")
            self.sync("Go find", 0.1)
            self.play(FadeIn(cta, scale=1.2), run_time=0.4)
            self.mark("cta", cta, "open the X-ray view to inspect embeddings live")
        self.wait(0.6)
