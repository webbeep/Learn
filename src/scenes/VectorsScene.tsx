import { useState } from "react";
import { div, f2, phase, show, tokColor } from "../lib/viz";
import type { SceneProps } from "./types";

const MAXROWS = 14;
const robustScale = (pts: [number, number][]) => {
  const a = pts.flatMap(([x, y]) => [Math.abs(x), Math.abs(y)]).sort((p, q) => p - q);
  return 1 / (a[Math.floor(a.length * 0.97)] || 1);
};

function Strip({ v, id, what, tok, scale = 0.25, op = 1 }: { v: Float32Array; id: string; what: string; tok: string; scale?: number; op?: number }) {
  return (
    <div className="strip" style={{ opacity: op }} data-ask-id={id} data-ask={`${what} of ${tok}: 64 numbers, e.g. d0=${f2(v[0])}, d1=${f2(v[1])}, max=${f2(Math.max(...v))}, min=${f2(Math.min(...v))}`}>
      {Array.from(v, (x, d) => <i key={d} style={{ background: div(x, scale) }} data-ask-id={`${id}:d${d}`} data-ask={`dimension ${d} of the ${what} of ${tok} = ${x.toFixed(3)}`} />)}
    </div>
  );
}

export function VectorsScene({ m, step, t, extra }: SceneProps) {
  const T = step.ctx.length, start = Math.max(0, T - MAXROWS);
  const rows = step.ctx.slice(start).map((id, k) => ({ id, pos: start + k }));
  const [hover, setHover] = useState<number | null>(null);
  const inCtx = new Set(step.ctx);
  const sc = robustScale(m.meta.pca);
  return (
    <div className="scene vectors-scene">
      <div className="vec-table">
        <div className="vec-head"><span>token</span><span>token embedding <em>(what)</em></span><span /><span>position embedding <em>(where)</em></span><span /><span>input vector</span></div>
        {start > 0 && <div className="muted small">…{start} earlier tokens hidden</div>}
        {rows.map(({ id, pos }) => {
          const tok = `"${show(m.meta.vocab[id])}" (position ${pos})`;
          return (
            <div className="vec-row" key={pos}>
              <span className="mono tchip" style={{ color: tokColor(id, 75) }}>{show(m.meta.vocab[id])}</span>
              <Strip v={step.trace.tokEmb[pos]} id={`emb-tok-${pos}`} what="token embedding" tok={tok} op={phase(t, 0.05, 0.35)} />
              <b style={{ opacity: phase(t, 0.3, 0.4) }}>+</b>
              <Strip v={step.trace.posEmb[pos]} id={`emb-pos-${pos}`} what="position embedding" tok={tok} scale={0.12} op={phase(t, 0.3, 0.6)} />
              <b style={{ opacity: phase(t, 0.6, 0.7) }}>=</b>
              <Strip v={step.trace.x0[pos]} id={`emb-sum-${pos}`} what="input vector (token+position)" tok={tok} op={phase(t, 0.6, 0.9)} />
            </div>
          );
        })}
        <div className="legend small muted"><i style={{ background: div(-1) }} /> negative <i style={{ background: div(0) }} /> zero <i style={{ background: div(1) }} /> positive · each square is one of 64 numbers (GPT-3 uses 12,288)</div>
      </div>
      <div className="pca">
        <div className="lane-label">Map of all {m.cfg.vocab_size} token vectors (64-D squashed to 2-D)</div>
        <svg viewBox="-1.12 -1.12 2.24 2.24" className="pca-svg">
          {m.meta.pca.map(([x0, y0], id) => {
            const x = Math.max(-1.05, Math.min(1.05, x0 * sc)), y = Math.max(-1.05, Math.min(1.05, y0 * sc));
            const on = inCtx.has(id), hv = hover === id || (hover !== null && extra.neighbours[hover].includes(id));
            const label = `token "${show(m.meta.vocab[id])}" (id ${id}) on the embedding map at (${x.toFixed(2)}, ${y.toFixed(2)}); most similar tokens: ${extra.neighbours[id].map((j) => `"${show(m.meta.vocab[j])}"`).join(", ")}`;
            return (
              <g key={id} data-ask-id={`pca-${id}`} data-ask={label} onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)}>
                <circle cx={x} cy={-y} r={on || hv ? 0.035 : 0.02} fill={tokColor(id)} opacity={on || hv ? 1 : 0.5} />
                {(on || hv) && <text x={x + 0.04} y={-y + 0.015} fontSize="0.065" fill="currentColor">{show(m.meta.vocab[id])}</text>}
              </g>
            );
          })}
        </svg>
        <div className="small muted">Hover a dot to see its nearest neighbours. Similar words end up near each other — nobody told the model that; it learned it.</div>
      </div>
    </div>
  );
}
