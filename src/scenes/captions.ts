import type { TinyGPT } from "../model/gpt";
import type { Step } from "../model/generate";
import type { SceneKey, View } from "./types";
import { pct, show } from "../lib/viz";

/** Narration for each scene: an intuition line and a "show me the math" line. */
export function caption(key: SceneKey, m: TinyGPT, s: Step, v: View, math: boolean): string {
  const tok = (id: number) => `“${show(m.meta.vocab[id])}”`;
  const T = s.ctx.length, last = s.ctx[T - 1];
  const top = s.dist[0];
  switch (key) {
    case "tokens": return math
      ? `BPE tokenizer: start from single characters, apply ${m.meta.merges.length} learned merges in priority order. Output: a list of ${T} integers in [0, ${m.cfg.vocab_size}).`
      : `Computers can't read. The text is chopped into ${T} familiar pieces (tokens) and each piece becomes a number. Common words are one token; rare words get split.`;
    case "vectors": return math
      ? `x₀[i] = E_tok[id_i] + E_pos[i]. Two lookup tables (${m.cfg.vocab_size}×${m.cfg.d_model} and ${m.cfg.n_ctx}×${m.cfg.d_model}), no arithmetic yet besides one addition.`
      : `Each token number is swapped for a list of ${m.cfg.d_model} numbers that capture its "meaning", plus a second list that says where it sits in the sentence.`;
    case "attention": {
      const a = s.trace.layers[v.layer].attn[v.head][v.query ?? T - 1];
      const j = a.indexOf(Math.max(...a));
      return math
        ? `For each head: q=W_q·x, k=W_k·x, v=W_v·x. weight(i→j) = softmax_j(q_i·k_j / √${m.cfg.d_model / m.cfg.n_head}) for j ≤ i (causal mask). Output = Σ weight·v_j.`
        : `Tokens look back at earlier tokens to gather context. Right now, in layer ${v.layer + 1} head ${v.head + 1}, ${tok(s.ctx[v.query ?? T - 1])} looks mostly at ${tok(s.ctx[j])} (${pct(a[j])}).`;
    }
    case "layers": {
      const firstHit = [s.trace.lens0, ...s.trace.layers.flatMap((l) => [l.lensMid, l.lensPost])].findIndex((p) => p[0].id === top.id);
      return math
        ? `x ← x + Attn(LN(x)); x ← x + MLP(LN(x)), ${m.cfg.n_layer}×. Logit lens = softmax(E · LN_f(x)) applied to intermediate x.`
        : `Each layer adds a little update to the last token's vector (${tok(last)}). The final answer ${tok(top.id)} ${firstHit <= 0 ? "is there almost from the start" : `first becomes the top guess at step ${firstHit} of 6`}.`;
    }
    case "probs": return math
      ? `logits = E · LN_f(x_last) (weight tying: the same embedding table reads out). p = softmax(logits / T).`
      : `The model scores all ${m.cfg.vocab_size} possible next tokens. Its favourite is ${tok(top.id)} at ${pct(top.p)}.`;
    case "pick": return math
      ? `Sample from top-k after temperature: inverse CDF with r ~ U(0,1) = ${s.r.toFixed(3)}. Append, re-run. Cost: one full forward pass per token.`
      : `A weighted dice roll picks ${tok(s.chosen)}. It's appended, and the whole thing runs again for the next token.`;
  }
}

/** Plain-text description of what's on screen, sent to the tutor. */
export function describe(key: SceneKey, m: TinyGPT, steps: Step[], si: number, v: View, settings: { temperature: number; topK: number }): string {
  const s = steps[si], T = s.ctx.length, w = (id: number) => `"${show(m.meta.vocab[id])}"`;
  const lines = [
    `Generating token ${si + 1} of ${steps.length}. Settings: temperature ${settings.temperature}, top-k ${settings.topK}.`,
    `Current context (${T} tokens): ${s.ctx.map((id, i) => `${i}:${w(id)}`).join(" ")}`,
    `Model's top next-token guesses: ${s.dist.slice(0, 5).map((d) => `${w(d.id)} ${pct(d.p)}`).join(", ")}; sampled: ${w(s.chosen)}.`,
    `Generated so far: ${steps.slice(0, si + 1).map((x) => w(x.chosen)).join(" ")}`,
  ];
  const q = v.query ?? T - 1;
  if (key === "attention") {
    const a = s.trace.layers[v.layer].attn[v.head][q];
    lines.push(`Showing attention layer ${v.layer + 1} head ${v.head + 1}, looking from token #${q} ${w(s.ctx[q])}. Top attended: ${a.map((p, j) => ({ p, j })).sort((x, y) => y.p - x.p).slice(0, 4).map((x) => `#${x.j} ${w(s.ctx[x.j])} ${pct(x.p)}`).join(", ")}.`);
  }
  if (key === "layers") {
    const st = [["input", s.trace.lens0], ...s.trace.layers.flatMap((l, i) => [[`L${i + 1}+attn`, l.lensMid], [`L${i + 1}+mlp`, l.lensPost]])] as [string, { id: number; p: number }[]][];
    lines.push(`Logit lens on last token: ${st.map(([n, p]) => `${n}→${w(p[0].id)} ${pct(p[0].p)}`).join("; ")}. MLP neurons shown for layer ${v.mlpLayer + 1}.`);
  }
  if (key === "pick") lines.push(`Random number r=${s.r.toFixed(3)} over cumulative distribution ${s.dist.map((d) => `${w(d.id)} ${pct(d.p)}`).join(", ")}.`);
  return lines.join("\n");
}
