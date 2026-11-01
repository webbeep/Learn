import { pct, phase, show, tokColor } from "../lib/viz";
import type { SceneProps } from "./types";

export function PickScene({ m, step, steps, si, t, settings, regenerate }: SceneProps) {
  const W = 1000, needle = phase(t, 0.05, 0.55);
  let acc = 0;
  const segs = step.dist.map((d) => { const s = { ...d, x: acc }; acc += d.p; return s; });
  const chosenIdx = segs.findIndex((s) => s.id === step.chosen);
  const landed = t > 0.55;
  const text = m.meta.vocab;
  const greedy = settings.temperature <= 0.01;
  return (
    <div className="scene pick-scene">
      <div className="lane-label">{greedy ? "Temperature 0: always take the top token (greedy)." : `Roll a random number between 0 and 1 (here r = ${step.r.toFixed(3)}) and see which token's slice it lands in. Only the top ${settings.topK} tokens are allowed (top-k).`}</div>
      <svg viewBox={`0 -40 ${W} 120`} className="pick-svg">
        {segs.map((s, i) => (
          <g key={s.id} data-ask-id={`pick-seg-${s.id}`} data-ask={`slice for "${show(text[s.id])}" from ${s.x.toFixed(3)} to ${(s.x + s.p).toFixed(3)} (${pct(s.p)})${s.id === step.chosen ? " — the random number landed here, CHOSEN" : ""}`}>
            <rect x={s.x * W} y={0} width={Math.max(1, s.p * W - 1)} height={44} fill={tokColor(s.id, i === chosenIdx && landed ? 55 : 32)} />
            {s.p * W > 38 && <text x={s.x * W + 5} y={27} fontSize={15} className="mono" fill="#fff">{show(text[s.id])}</text>}
          </g>
        ))}
        <g transform={`translate(${(greedy ? segs[0].p / 2 : step.r) * W * needle},0)`} data-ask-id="pick-needle" data-ask={`random number r = ${step.r.toFixed(3)} → picks "${show(text[step.chosen])}"`}>
          <path d="M0,-6 L-9,-26 L9,-26 Z" fill="var(--pin)" />
          <line x1={0} x2={0} y1={-6} y2={52} stroke="var(--pin)" strokeWidth={3} />
          <text x={0} y={-30} textAnchor={step.r < 0.05 ? "start" : step.r > 0.95 ? "end" : "middle"} fontSize={14} fill="var(--pin)">r={step.r.toFixed(2)}</text>
        </g>
        <text x={0} y={70} fontSize={12} fill="var(--muted)">0</text><text x={W} y={70} fontSize={12} textAnchor="end" fill="var(--muted)">1</text>
      </svg>
      <div className={`picked ${landed ? "in" : ""}`}>
        <span className="muted">next token:</span>
        <span className="tok big" style={{ borderColor: tokColor(step.chosen) }} data-ask-id="pick-chosen" data-ask={`the chosen next token "${show(text[step.chosen])}" (it had ${pct(step.dist[chosenIdx]?.p ?? 0)} probability)`}>
          <span className="mono" style={{ color: tokColor(step.chosen, 75) }}>{show(text[step.chosen])}</span></span>
      </div>
      <div className={`loop-text mono ${landed ? "in" : ""}`}>
        {m.tok.decode(step.ctx)}<span className="fresh">{text[step.chosen]}</span>
      </div>
      <div className="loop-note" style={{ opacity: phase(t, 0.7, 0.9) }}>
        ↺ The new token is glued onto the text and <b>the whole pipeline runs again</b> from step 1 to get the next one.
        That's all "generation" is — {si + 1} of {steps.length} loops done.
        {chosenIdx > 0 && <span> Note: it did <i>not</i> pick the most likely token — that's sampling randomness.</span>}
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={() => regenerate({ seed: settings.seed + 1 }, si)}>🎲 Re-roll from here</button>
          <button onClick={() => regenerate({ temperature: 0 }, si)}>Always pick the top (temperature 0)</button>
        </div>
      </div>
    </div>
  );
}
