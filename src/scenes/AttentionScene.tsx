import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { pct, phase, seq, show, tokColor } from "../lib/viz";
import type { SceneProps } from "./types";

export function AttentionScene({ m, step, t, view, setView }: SceneProps) {
  const { n_layer: L, n_head: H } = m.cfg;
  const T = step.ctx.length, q = view.query ?? T - 1;
  const lt = step.trace.layers[view.layer];
  const A = lt.attn[view.head], S = lt.scores[view.head];
  const w = A[q];
  const name = (i: number) => `"${show(m.meta.vocab[step.ctx[i]])}"(#${i})`;
  const top = w.map((p, j) => ({ p, j })).sort((a, b) => b.p - a.p).slice(0, 3).map((x) => x.j);

  // arc diagram geometry
  const W = 1000, pad = 20, dx = (W - 2 * pad) / Math.max(T, 1), y0 = 150, fs = Math.min(15, dx * 0.9);
  const cx = (i: number) => pad + dx * (i + 0.5);
  const draw = phase(t, 0.1, 0.7);

  // heatmap geometry
  const cell = Math.min(16, 360 / T), hm = cell * T;
  return (
    <div className="scene attention-scene">
      <div className="controls">
        <span className="text-xs text-muted-foreground">Layer</span>
        <ToggleGroup type="single" variant="outline" size="sm" value={String(view.layer)} onValueChange={(v) => v && setView({ layer: +v })}>
          {Array.from({ length: L }, (_, l) => <ToggleGroupItem key={l} value={String(l)} className="w-8">{l + 1}</ToggleGroupItem>)}
        </ToggleGroup>
        <span className="ml-2 text-xs text-muted-foreground">Head</span>
        <ToggleGroup type="single" variant="outline" size="sm" value={String(view.head)} onValueChange={(v) => v && setView({ head: +v })}>
          {Array.from({ length: H }, (_, h) => <ToggleGroupItem key={h} value={String(h)} className="w-8">{h + 1}</ToggleGroupItem>)}
        </ToggleGroup>
        
        {view.query !== null && <Button variant="ghost" size="xs" onClick={() => setView({ query: null })}>Reset to last token</Button>}
      </div>

      <div className="arc-wrap"><svg viewBox={`0 0 ${W} 190`} className="arc-svg">
        {w.map((p, j) => {
          if (j === q) return null;
          const x1 = cx(q), x2 = cx(j), h = 14 + 120 * Math.pow(Math.abs(x1 - x2) / W, 0.6);
          const len = Math.abs(x1 - x2) * 1.6 + h;
          return (
            <path key={j} d={`M${x1},${y0 - 18} C${x1},${y0 - 18 - h} ${x2},${y0 - 18 - h} ${x2},${y0 - 18}`}
              fill="none" stroke={tokColor(step.ctx[j])} strokeOpacity={0.25 + 0.75 * p} strokeWidth={0.5 + 14 * p}
              strokeDasharray={len} strokeDashoffset={len * (1 - draw)} strokeLinecap="round"
              data-ask-id={`att-arc-${j}`} data-ask={`layer ${view.layer + 1} head ${view.head + 1}: ${name(q)} attends to ${name(j)} with weight ${pct(p)} (raw score ${S[q][j].toFixed(2)})`} />
          );
        })}
        {step.ctx.map((id, i) => (
          <g key={i} onClick={() => setView({ query: i === T - 1 ? null : i })} style={{ cursor: "pointer" }} opacity={i > q ? 0.25 : 1}
            data-ask-id={`att-tok-${i}`} data-ask={`token ${name(i)}; ${i === q ? "the token we are looking FROM (the query)" : i < q ? `receives ${pct(w[i])} of ${name(q)}'s attention in L${view.layer + 1}H${view.head + 1}` : "in the future relative to the query — masked, it cannot be seen"}`}>
            <rect x={cx(i) - dx / 2 + 1} y={y0 - 16} width={dx - 2} height={30} rx={4} fill={i === q ? "var(--viz-a)" : "var(--muted)"} opacity={i === q ? 0.35 : 1} />
            <text x={cx(i)} y={y0 + 4} textAnchor="middle" fontSize={fs} className="mono" fill={tokColor(id, 78)}>{show(m.meta.vocab[id]).slice(0, Math.max(2, Math.floor(dx / (fs * 0.62))))}</text>
            {i <= q && top.includes(i) && i !== q && <text x={cx(i)} y={y0 + 30} textAnchor="middle" fontSize={12} fill="var(--foreground)" opacity={draw}>{pct(w[i])}</text>}
          </g>
        ))}
      </svg></div>

      <div className="att-bottom">
        <div>
          <div className="lane-label">Layer {view.layer + 1}, head {view.head + 1}</div>
          <svg viewBox={`-70 -70 ${hm + 80} ${hm + 80}`} className="heatmap" style={{ maxWidth: hm + 80 }}>
            {step.ctx.map((id, i) => (
              <g key={i}>
                <text x={-4} y={i * cell + cell * 0.7} fontSize={Math.min(10, cell * 0.8)} textAnchor="end" fill={tokColor(id, 75)} className="mono">{show(m.meta.vocab[id]).slice(0, 7)}</text>
                <text transform={`translate(${i * cell + cell * 0.7},-4) rotate(-60)`} fontSize={Math.min(10, cell * 0.8)} fill={tokColor(id, 75)} className="mono">{show(m.meta.vocab[id]).slice(0, 7)}</text>
              </g>
            ))}
            {A.map((row, i) => row.map((p, j) => (
              <rect key={`${i}-${j}`} x={j * cell} y={i * cell} width={cell - 0.5} height={cell - 0.5} fill={seq(p)} opacity={i === q ? 1 : 0.85}
                data-ask-id={`att-L${view.layer + 1}H${view.head + 1}:${i}-${j}`} data-ask={`layer ${view.layer + 1} head ${view.head + 1}: ${name(i)} → ${name(j)} weight ${pct(p)}, score ${S[i][j].toFixed(2)}`} />
            )))}
            <rect x={-1} y={q * cell - 1} width={(q + 1) * cell + 1} height={cell + 1} fill="none" stroke="var(--viz-a)" strokeWidth={1.5} />
          </svg>
        </div>
        <div>
          <div className="lane-label">All {L * H} heads</div>
          <div className="heads-grid" style={{ gridTemplateColumns: `repeat(${H}, 1fr)` }}>
            {step.trace.layers.map((lt2, l) => lt2.attn.map((a, h) => {
              const best = a[q].map((p, j) => ({ p, j })).sort((x, y) => y.p - x.p)[0];
              return (
                <button key={`${l}-${h}`} className={cn("head-thumb", l === view.layer && h === view.head && "on")} onClick={() => setView({ layer: l, head: h })}
                  data-ask-id={`att-head-L${l + 1}H${h + 1}`} data-ask={`layer ${l + 1} head ${h + 1}: from ${name(q)} it mostly looks at ${name(best.j)} (${pct(best.p)})`}>
                  <svg viewBox={`0 0 ${T} ${T}`} preserveAspectRatio="none">
                    {a.map((row, i) => row.map((p, j) => p > 0.04 ? <rect key={`${i}-${j}`} x={j} y={i} width={1} height={1} fill={seq(p)} /> : null))}
                  </svg>
                  <small>L{l + 1}·H{h + 1}</small>
                </button>
              );
            }))}
          </div>
        </div>
      </div>
    </div>
  );
}
