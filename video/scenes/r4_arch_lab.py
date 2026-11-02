import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vscene import *  # noqa
from lib_research import load, Plot, legend, pill, panel, illus, fmt_n
import torch
from model import forward, tok, VOCAB

VIDEO = dict(id="arch-lab", cls="ArchLab", title="Architecture lab: break it to understand it",
             subtitle="Remove attention, MLPs, positions, residuals or LayerNorm, retrain, and watch what breaks (real runs)",
             relates="lab", track="design", order=20, poster=("Leaderboard", 0, 0.95))

AB = load("ablations")
RUNS = {r["id"]: r for r in AB["runs"]}
STEPS = AB["steps"]
PROBE = AB["probe_prompt"]
BASE = RUNS["baseline"]
NG = AB["baselines"]
SHIPPED_RED = float(torch.softmax(forward(tok(PROBE))[-1], -1)[VOCAB.index(" red")])   # the shipped model, same prompt


def acc(r):
    return round(r["probe_set"]["top1_acc"] * r["probe_set"]["n"])
COL = {"no-attention": BLUE, "no-mlp": ORANGE, "no-positions": "#b48cff", "no-residual": PINK,
       "no-layernorm": YELLOW, "1-layer": "#5fd0ff", "6-layer": TEAL, "baseline": FG}


def pct(p):
    return f"{p * 100:.0f}%" if p >= 0.095 else f"{p * 100:.1f}%"


def top1(r):
    return r["probe_top"][0]


def p_of(r, tok):
    for t in r["probe_top"]:
        if t["token"] == tok:
            return t["p"]
    return None


def spoken_tok(t):
    named = {".": "a period", "\n": "a line break", ",": "a comma", "?": "a question mark", "!": "an exclamation mark"}
    return named.get(t.strip() if t.strip() else t, t.strip() or "a space")


SPOKEN = {"no-attention": "No attention", "no-mlp": "No MLP", "no-positions": "No positions",
          "no-residual": "No residuals", "no-layernorm": "No LayerNorm", "1-layer": "One layer", "6-layer": "Six layers"}


# one short, data-driven line per ablation (the insight phrase is the only hand-written part)
INSIGHT = {
    "no-attention": f"Every token is alone. It's exactly a bigram model: loss {RUNS['no-attention']['final_val']:.2f}, "
                    f"bigram baseline {NG['bigram']:.2f}.",
    "no-mlp": f"Attention still moves information, so copying survives. Loss {RUNS['no-mlp']['final_val']:.2f}.",
    "no-positions": f"Surprise: it barely matters. The causal mask, which only lets each word see earlier ones, already "
                    f"leaks word order. Loss {RUNS['no-positions']['final_val']:.2f}.",
    "no-residual": f"It only learned which words are common: {RUNS['no-residual']['final_val']:.2f}, the unigram level.",
    "no-layernorm": f"It learns the language, loss {RUNS['no-layernorm']['final_val']:.2f}, but rarely the trick.",
    "1-layer": f"Loss {RUNS['1-layer']['final_val']:.2f}, but copying like this typically needs two layers working together.",
    "6-layer": "Our one test prompt says it failed. But across all the stories, it often gets it right. One prompt can "
               "mislead.",
}
ORDER = ["no-attention", "no-positions", "no-mlp", "no-residual", "no-layernorm", "1-layer", "6-layer"]


class ArchLab(VScene):
    VIDEO_ID = "arch-lab"

    def probe_view(self, r, col, width=5.2):
        """Top-3 next-token bars for the probe prompt + the model's sampled continuation."""
        rows = VGroup()
        for t in r["probe_top"][:3]:
            c = TEAL if t["token"] == " red" else col
            lab = self.chip(t["token"], c, 24)
            bar = Rectangle(width=max(0.04, width * 0.62 * t["p"]), height=0.34, stroke_width=0, fill_color=c, fill_opacity=0.8)
            pv = self.M(pct(t["p"]), 22, FG)
            row = VGroup(lab, bar, pv)
            lab.move_to(ORIGIN, aligned_edge=RIGHT)
            bar.next_to(lab, RIGHT, buff=0.2)
            pv.next_to(bar, RIGHT, buff=0.15)
            rows.add(row)
        rows.arrange(DOWN, buff=0.18)
        for row in rows:
            row.shift(RIGHT * (rows[0][0].get_right()[0] - row[0].get_right()[0]))
        head = self.M("…now tom has the →", 24, MUTED)
        text = r["samples"][0]["text"].split("\n")[0][:26]
        samp = VGroup(self.T("one random sample (temperature 0.8):", 20, MUTED), self.M(text if text.strip() else "(nothing)", 22, col)) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        a = acc(r)
        score = VGroup(self.M(f"{a}/100", 34, TEAL if a >= 90 else (YELLOW if a >= 50 else PINK)),
                       self.T("stories: right colour on top", 22, FG)).arrange(RIGHT, buff=0.2)
        g = VGroup(head, rows, samp, score).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        return g

    def mini_plot(self, r, col, ymax):
        p = Plot([0, STEPS], [0, ymax], width=5.6, height=3.4, xticks=[0, STEPS // 2, STEPS],
                 yticks=list(range(0, int(ymax) + 1)), xlabel="training step · gray = full model", ylabel="val loss")
        lb = p.line(BASE["curve"]["step"], BASE["curve"]["val"], MUTED, 3)
        la = p.line(r["curve"]["step"], r["curve"]["val"], col, 4)
        return p, lb, la

    def construct(self):
        # ------------------------------------------------------------ hook
        parts = [("attention", BLUE), ("positions", "#b48cff"), ("MLP", ORANGE),
                 ("residual +", PINK), ("LayerNorm", YELLOW), ("1 layer", "#5fd0ff"), ("6 layers", TEAL)]
        pills = VGroup(*[pill(t, c, 30) for t, c in parts]).arrange_in_grid(rows=2, buff=(0.35, 0.45)).move_to(DOWN * 0.3)
        ttl = self.T("what does each part actually do?", 46, FG, BOLD).move_to(UP * 2.3)
        sub_ = self.T("five cuts + two resizes · each retrained from scratch with the shipped model's recipe", 24, MUTED) \
            .next_to(pills, DOWN, buff=0.5)
        with self.say(f"What does each part of a transformer actually do? Remove it, retrain from scratch, and see what "
                      f"breaks. Five cuts and two resizes, each trained with the shipped model's recipe.",
                      scene="Seven pills: five parts to remove (attention, positions, MLP, residual +, LayerNorm) and two "
                            "resizes (1 layer, 6 layers). Each variant is retrained from scratch with the shipped recipe.",
                      chapter="Break it"):
            self.play(FadeIn(ttl, shift=DOWN * 0.2), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(p, scale=0.9) for p in pills], lag_ratio=0.1), FadeIn(sub_), run_time=1.2)
            for (t, _), p in zip(parts, pills):
                self.mark("part-" + t.split()[0], p, f"model part: {t}")
            self.sync("remove it", 0.1)
            self.play(*[Wiggle(p, scale_value=1.08) for p in pills], run_time=0.8)

        # ------------------------------------------------------------ baseline
        hdr = self.header("The full model")
        tb = top1(BASE)
        ymax = 5
        pv = self.probe_view(BASE, TEAL)
        pv.move_to([0.6, -0.3, 0], aligned_edge=LEFT)
        cut = PROBE.rfind(" ", 0, len(PROBE) // 2 + 4)
        prompt = VGroup(self.M(PROBE[:cut], 24, MUTED), self.M(PROBE[cut + 1:], 24, MUTED)).arrange(DOWN, buff=0.12)
        prompt.move_to(UP * 2.55)
        pb, lb, _ = self.mini_plot(BASE, FG, ymax)
        lb_full = pb.line(BASE["curve"]["step"], BASE["curve"]["val"], FG, 4)
        VGroup(pb).move_to(LEFT * 3.3 + DOWN * 0.4)
        lb_full = pb.line(BASE["curve"]["step"], BASE["curve"]["val"], FG, 4)
        fvl = self.M(f"final {BASE['final_val']:.3f}", 24, FG).next_to(pb.p(STEPS, BASE["final_val"]), UP, buff=0.2) \
            .align_to(pb.p(STEPS, 0), RIGHT)
        pr = p_of(BASE, " red")
        with self.say(f"First the control: the full model. Loss {BASE['final_val']:.2f}. On our test prompt, Mia gives Tom "
                      f"the red ball, it says red, {pct(pr or 0)}. The shipped model says {SHIPPED_RED * 100:.0f}: same "
                      f"recipe, different batch order. Runs vary. So we also score 100 fresh stories like it: "
                      f"{acc(BASE)} of 100 right.",
                      scene=f"Baseline (full TinyGPT, {BASE['params']:,} params, {STEPS} steps): validation loss {BASE['final_val']:.3f}; "
                            f"on the probe prompt its top next tokens are "
                            + ", ".join(f"{t['token']!r} {pct(t['p'])}" for t in BASE["probe_top"][:3])
                            + f". The shipped model (same recipe, different batch order) gives ' red' {SHIPPED_RED * 100:.1f}%. "
                              f"On 100 generated trade stories the baseline puts the right colour on top {acc(BASE)}/100 "
                              f"(mean p {BASE['probe_set']['mean_p_correct']:.2f}).",
                      chapter="The control"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(ttl, pills, sub_)), FadeIn(hdr), run_time=0.4)
            self.play(FadeIn(pb), Create(lb_full), run_time=1.0)
            self.play(FadeIn(fvl), run_time=0.3)
            self.mark("base-curve", VGroup(lb_full, fvl), f"baseline validation loss → {BASE['final_val']:.3f}")
            self.sync("Our test prompt", 0.1)
            self.play(FadeIn(prompt), FadeIn(pv, shift=LEFT * 0.2), run_time=0.6)
            self.mark("base-probe", pv[1], "baseline next-token probabilities: "
                      + ", ".join(f"{t['token']!r} {t['p']:.3f}" for t in BASE["probe_top"][:5]))
            self.mark("prompt", prompt, f"probe prompt: {PROBE}")
            self.sync("The shipped", 0.1)
            ship = self.T(f"shipped model, same recipe: ' red' {SHIPPED_RED * 100:.0f}%", 24, MUTED).next_to(pv, DOWN, buff=0.3).align_to(pv, LEFT)
            self.play(FadeIn(ship), run_time=0.3)
            self.mark("shipped", ship, f"the shipped TinyGPT (same recipe, different batch order): p(' red') = {SHIPPED_RED:.3f}")

        # ------------------------------------------------------------ each ablation
        cur = VGroup(pb, lb_full, fvl, pv, ship)
        cur_hdr = hdr
        first = True
        for rid in ORDER:
            r = RUNS[rid]
            col = COL[rid]
            ymax_r = 5
            p, lbase, lab_ = self.mini_plot(r, col, ymax_r)
            p.move_to(LEFT * 3.3 + DOWN * 0.4)
            lbase = p.line(BASE["curve"]["step"], BASE["curve"]["val"], MUTED, 3)
            lab_ = p.line(r["curve"]["step"], r["curve"]["val"], col, 4)
            fl = VGroup(self.M(f"{r['final_val']:.2f}", 24, col), self.T("vs full model", 22, MUTED),
                        self.M(f"{BASE['final_val']:.2f}", 24, MUTED)).arrange(RIGHT, buff=0.15)
            fl.next_to(p.ax, UP, buff=0.18).align_to(p.ax, RIGHT)
            pv2 = self.probe_view(r, col)
            pv2.move_to([0.6, -0.4, 0], aligned_edge=LEFT)
            h2 = self.header(r["label"], col)
            ln_s = f"{SPOKEN[rid]}. {INSIGHT[rid]} Right colour on {acc(r)} of 100 stories."
            with self.say(ln_s, scene=f"Ablation '{r['label']}' ({r['params']:,} params): {r['what']} Real validation curve "
                                      f"(colored) vs baseline (gray): final {r['final_val']:.3f} vs {BASE['final_val']:.3f}. "
                                      f"Probe top tokens: " + ", ".join(f"{t['token']!r} {pct(t['p'])}" for t in r["probe_top"][:3])
                                      + f". On 100 generated stories: right colour on top {acc(r)}/100 (mean p "
                                        f"{r['probe_set']['mean_p_correct']:.2f}). One random sample (temperature 0.8): "
                                        f"{r['samples'][0]['text'][:40]!r}.",
                          chapter="Seven variants" if first else None):
                self.unmark(*list(self._marks))
                self.play(FadeOut(cur), FadeOut(prompt) if first else Wait(0), ReplacementTransform(cur_hdr, h2),
                          run_time=0.4)
                self.play(FadeIn(p), Create(lbase), run_time=0.5)
                self.play(Create(lab_), FadeIn(fl), run_time=0.9)
                self.mark(f"{rid}-curve", VGroup(lab_, fl), f"{r['label']}: val loss {r['final_val']:.3f} "
                          f"(baseline {BASE['final_val']:.3f})")
                self.mark(f"{rid}-base", lbase, f"baseline curve for comparison ({BASE['final_val']:.3f})")
                self.play(FadeIn(pv2, shift=LEFT * 0.2), run_time=0.5)
                self.mark(f"{rid}-probe", pv2, f"{r['label']} on the probe: " + ", ".join(
                    f"{t['token']!r} {t['p']:.3f}" for t in r["probe_top"][:5]) + f"; right colour on {acc(r)}/100 stories; one "
                    f"random sample: {r['samples'][0]['text'][:40]!r}")
            cur = VGroup(p, lbase, lab_, fl, pv2)
            cur_hdr = h2
            first = False

        # ------------------------------------------------------------ leaderboard
        hdr3 = self.header("Leaderboard")
        order = sorted(AB["runs"], key=lambda r: r["final_val"])
        top = max(r["final_val"] for r in order)
        bars = VGroup()
        for k, r in enumerate(order):
            col = COL[r["id"]]
            name = self.T(r["label"], 26, col, BOLD)
            w = 5.5 * r["final_val"] / top
            b = Rectangle(width=w, height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.8)
            v = self.M(f"{r['final_val']:.2f}", 24, FG)
            a_ = acc(r)
            rd = self.M(f"{a_:>3}/100 right", 22, TEAL if a_ >= 90 else (YELLOW if a_ >= 50 else PINK))
            bars.add(VGroup(name, b, v, rd))
        y0 = 2.35
        for k, row in enumerate(bars):
            y = y0 - k * 0.6
            row[0].move_to([-3.6, y, 0], aligned_edge=RIGHT)
            row[1].move_to([-3.35, y, 0], aligned_edge=LEFT)
            row[2].next_to(row[1], RIGHT, buff=0.15)
            row[3].next_to(row[2], RIGHT, buff=0.35)
        cap = VGroup(self.T(f"final validation loss (lower is better) · colour-copy test on 100 generated stories", 22, MUTED),
                     self.T("one run each · same learning rate and steps for all, not tuned per variant · loss noise ≈ ±0.01",
                            22, MUTED)).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.3)
        worst = order[-1]
        n_right = sum(1 for r in AB["runs"] if r["id"] != "baseline" and top1(r)["token"] == " red")
        by_loss = sorted([r for r in AB["runs"] if r["id"] != "baseline"], key=lambda r: -r["final_val"])
        rest = [r for r in order if r["final_val"] < 1.0]
        a_hi, a_lo = max(acc(r) for r in rest), min(acc(r) for r in rest)
        with self.say(f"Line them up. Cutting {by_loss[0]['label'].replace('no ', '')}s or {by_loss[1]['label'].replace('no ', '')} "
                      f"wrecks it. The rest look alike by loss, yet their copy scores run from {a_hi} down to {a_lo} out of "
                      f"100. Loss can hide skills. And each is just one run.",
                      scene="Leaderboard of real final validation losses (lower = better) with each model's colour-copy score "
                            "on 100 generated stories: " + ", ".join(f"{r['label']} {r['final_val']:.3f} ({acc(r)}/100)"
                                                                       for r in order) + ". One run each; same LR and steps "
                            "for every variant.",
                      chapter="Leaderboard"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(cur), ReplacementTransform(cur_hdr, hdr3), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in bars], lag_ratio=0.08), FadeIn(cap), run_time=1.4)
            for r, b in zip(order, bars):
                self.mark("lb-" + r["id"], b, f"{r['label']}: final val {r['final_val']:.3f}, colour-copy test "
                          f"{acc(r)}/100 (mean p {r['probe_set']['mean_p_correct']:.2f})")

        # ------------------------------------------------------------ takeaway
        hdr4 = self.header("What each part does")
        roles = [("attention", "moves information between tokens", BLUE),
                 ("positions", "word order (the causal mask leaks some)", "#b48cff"),
                 ("MLP", "per-token processing and memory", ORANGE),
                 ("residual + LayerNorm", "keep it trainable", PINK),
                 ("depth", "more steps to compose patterns", TEAL)]
        rows = VGroup()
        for a, b, c in roles:
            rows.add(VGroup(self.T(a, 30, c, BOLD), self.T(b, 28, FG)).arrange(RIGHT, buff=0.4))
        for row in rows:
            row[0].move_to([-1.2, 0, 0], aligned_edge=RIGHT); row[1].next_to(row[0], RIGHT, buff=0.4)
        rows.arrange(DOWN, buff=0.35).move_to(DOWN * 0.1)
        for row in rows:
            row[0].align_to([-1.2, 0, 0], RIGHT); row[1].next_to(row[0], RIGHT, buff=0.4)
        rows.shift(RIGHT * -rows.get_center()[0])
        cta = self.T("try your own cuts in the Architecture lab →", 30, YELLOW, BOLD).to_edge(DOWN, buff=0.5)
        with self.say("So: attention moves information, positions add order, MLPs process, residuals and LayerNorm keep "
                      "it trainable, and depth composes. Now go break one yourself.",
                      scene="Roles: attention = moves information between tokens; positions = word order; MLP = per-token "
                            "processing/memory; residual + LayerNorm = keep it trainable; depth = more steps to compose. "
                            "CTA: try your own cuts in the Architecture lab.", chapter="What each part does"):
            self.unmark(*list(self._marks))
            self.play(FadeOut(VGroup(bars, cap)), ReplacementTransform(hdr3, hdr4), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(r_, shift=RIGHT * 0.2) for r_ in rows], lag_ratio=0.25), run_time=1.6)
            for (a, b, _), r_ in zip(roles, rows):
                self.mark("role-" + a.split()[0], r_, f"{a}: {b}")
            self.sync("Now go", 0.1)
            self.play(FadeIn(cta, scale=1.1), run_time=0.4)
            self.mark("cta", cta, "open the Architecture lab")
        self.wait(0.8)
