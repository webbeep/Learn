import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useMemo } from "react";
import { norm, type Pred } from "../model/gpt";
import { div, pct, phase, show, tokColor } from "../lib/viz";
import type { SceneProps } from "./types";

export function LayersScene({ m, step, t, view, setView }: SceneProps) {
  const tr = step.trace, T = step.ctx.length, last = T - 1;
  const tokName = `"${show(m.meta.vocab[step.ctx[last]])}"`;
  const stages: { name: string; preds: Pred[]; delta?: number; kind: string }[] = [{ name: "Input vector", preds: tr.lens0, kind: "embed" }];
  tr.layers.forEach((l, i) => {
    stages.push({ name: `L${i + 1} + attention`, preds: l.lensMid, delta: norm(l.attnOut[last]), kind: "attn" });
    stages.push({ name: `L${i + 1} + MLP`, preds: l.lensPost, delta: norm(l.mlpOut[last]), kind: "mlp" });
  });
  const lt = tr.layers[view.mlpLayer];
  const acts = lt.mlpHidden[last];
  const topN = useMemo(() => Array.from(acts, (a, n) => ({ a, n })).sort((x, y) => Math.abs(y.a) - Math.abs(x.a)).slice(0, 5), [acts]);
  const promotes = (layer: number, n: number) => {
    const Wo = m.W[`blocks.${layer}.mlp_out.weight`], M = m.cfg.d_mlp, D = m.cfg.d_model, E = m.W["tok.weight"];
    const s = Array.from({ length: m.cfg.vocab_size }, (_, v) => { let x = 0; for (let d = 0; d < D; d++) x += E[v * D + d] * Wo[d * M + n]; return { v, x }; });
    return s.sort((a, b) => b.x - a.x).slice(0, 3).map((o) => `"${show(m.meta.vocab[o.v])}"`);
  };

  return (
    <div className="scene layers-scene">
      <div className="lane-label">Its guess after each layer</div>
      <div className="stream">
        {stages.map((s, k) => (
          <div key={k} className={`stage-col ${s.kind}`} style={{ opacity: 0.15 + 0.85 * phase(t, k * 0.1, k * 0.1 + 0.15) }}>
            <div className="stage-name">{s.name}</div>
            {s.delta !== undefined && <div className="delta small muted" data-ask-id={`resid-${k}`} data-ask={`${s.name}: the vector added to the residual stream has length ${s.delta.toFixed(2)}`}>+Δ {s.delta.toFixed(1)}</div>}
            {s.preds.slice(0, 4).map((p, r) => (
              <div key={r} className="lens-bar" data-ask-id={`lens-${k}-${r}`} data-ask={`logit lens after "${s.name}": guess #${r + 1} is "${show(m.meta.vocab[p.id])}" at ${pct(p.p)}`}>
                <span className="fill" style={{ width: `${Math.max(2, p.p * 100)}%`, background: tokColor(p.id, 45) }} />
                <span className="mono">{show(m.meta.vocab[p.id])}</span><span className="small">{pct(p.p)}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="mlp-row" style={{ opacity: phase(t, 0.65, 0.85) }}>
        <div>
          <div className="controls"><span className="text-xs text-muted-foreground">MLP neurons of layer</span>
            <ToggleGroup type="single" variant="outline" size="sm" value={String(view.mlpLayer)} onValueChange={(v) => v && setView({ mlpLayer: +v })}>
              {tr.layers.map((_, l) => <ToggleGroupItem key={l} value={String(l)} className="w-8">{l + 1}</ToggleGroupItem>)}
            </ToggleGroup>
          </div>
          <div className="neuron-grid">
            {Array.from(acts, (a, n) => (
              <i key={n} style={{ background: div(a, 1.2) }} data-ask-id={`mlp-L${view.mlpLayer + 1}:n${n}`} data-ask={`layer ${view.mlpLayer + 1} MLP neuron ${n}: activation ${a.toFixed(2)} for ${tokName}`} title={`neuron ${n}: ${a.toFixed(2)}`} />
            ))}
          </div>
          <div className="small muted">orange = firing</div>
        </div>
        <div className="top-neurons">
          <div className="lane-label">Strongest neurons</div>
          {topN.map(({ a, n }) => {
            const pr = promotes(view.mlpLayer, n);
            return (
              <div key={n} className="neuron-line" data-ask-id={`mlp-top-L${view.mlpLayer + 1}-n${n}`} data-ask={`layer ${view.mlpLayer + 1} neuron ${n} fires ${a.toFixed(2)}; when it fires it pushes the prediction toward ${pr.join(", ")} (approx.)`}>
                <span className="mono">#{n}</span>
                <span className="bar"><i style={{ width: `${Math.min(100, Math.abs(a) * 40)}%`, background: div(a, 1) }} /></span>
                <span className="small">{a.toFixed(2)} → pushes {pr.join(" ")}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
