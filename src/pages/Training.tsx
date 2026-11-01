import { useEffect, useState } from "react";
import { addWish, setScreen } from "../lib/store";

interface Log { steps: number[]; train: number[]; val: number[]; snapshots: { step: number; samples: string[] }[]; prompts: string[]; params: number; tokens: number; batch: number; lr: number }

export function Training({ corpus }: { corpus: string }) {
  const [log, setLog] = useState<Log | null>(null);
  const [snap, setSnap] = useState(0);
  useEffect(() => { fetch("/model/training.json").then((r) => r.json()).then((l: Log) => { setLog(l); setSnap(l.snapshots.length - 1); }); }, []);
  useEffect(() => {
    if (!log) return;
    const s = log.snapshots[snap];
    setScreen("Training peek", `checkpoint step ${s.step}`, [
      `Training run: ${log.params.toLocaleString()} parameters, ${log.tokens.toLocaleString()} training tokens, batch ${log.batch}×64 tokens, AdamW lr ${log.lr}, 3000 steps.`,
      `Loss went from ${log.train[0]} to ${log.train[log.train.length - 1]} (ln(256)=5.55 is pure guessing).`,
      `Showing samples from checkpoint at step ${s.step}: ${s.samples.map((x, i) => `[prompt "${log.prompts[i]}"] → "${x}"`).join(" | ")}`,
    ].join("\n"));
  }, [log, snap]);
  if (!log) return <div className="loading">Loading training log…</div>;

  const W = 700, H = 300, maxS = log.steps[log.steps.length - 1], maxL = Math.max(...log.train);
  const X = (s: number) => 40 + Math.sqrt(s / maxS) * (W - 60), Y = (l: number) => 10 + (1 - l / maxL) * (H - 40);
  const line = (arr: number[]) => arr.map((l, i) => `${i ? "L" : "M"}${X(log.steps[i]).toFixed(1)},${Y(l).toFixed(1)}`).join("");
  const cur = log.snapshots[snap];
  return (
    <div className="page training">
      <h1>Where did all those numbers come from?</h1>
      <p className="lead">Every number you saw in the X-ray started as random noise. Training nudged all {log.params.toLocaleString()} of them, 3000 times, to make the right next token a bit more likely each time. Here is that whole process — it took 80 seconds on a laptop CPU.</p>
      <div className="two-col">
        <div className="card">
          <div className="lane-label">Loss = how surprised the model is by the real next token (lower is better)</div>
          <svg viewBox={`0 0 ${W} ${H}`} className="loss-svg">
            {[0, 1, 2, 3, 4, 5].map((l) => <g key={l}><line x1={40} x2={W - 20} y1={Y(l)} y2={Y(l)} stroke="var(--grid)" /><text x={32} y={Y(l) + 4} fontSize={11} textAnchor="end" fill="var(--muted)">{l}</text></g>)}
            <path d={line(log.train)} fill="none" stroke="var(--accent)" strokeWidth={2} />
            <path d={line(log.val)} fill="none" stroke="var(--accent2)" strokeWidth={1.5} strokeDasharray="4 3" />
            {log.steps.map((s, i) => i % 4 === 0 && (
              <circle key={s} cx={X(s)} cy={Y(log.train[i])} r={5} fill="transparent" data-ask-id={`loss-${s}`} data-ask={`training step ${s}: train loss ${log.train[i]}, validation loss ${log.val[i]}`} />
            ))}
            {log.snapshots.map((s, i) => (
              <g key={s.step} onClick={() => setSnap(i)} style={{ cursor: "pointer" }} data-ask-id={`snap-${s.step}`} data-ask={`checkpoint at step ${s.step}; samples: ${s.samples.join(" | ")}`}>
                <line x1={X(s.step)} x2={X(s.step)} y1={10} y2={H - 30} stroke={i === snap ? "var(--pin)" : "var(--grid)"} strokeWidth={i === snap ? 2 : 1} />
                <circle cx={X(s.step)} cy={H - 22} r={7} fill={i === snap ? "var(--pin)" : "var(--panel2)"} />
              </g>
            ))}
            {[100, 500, 1000, 2000, 3000].map((s) => <text key={s} x={X(s)} y={H - 6} fontSize={10} textAnchor="middle" fill="var(--muted)">{s}</text>)}
            <text x={44} y={H - 6} fontSize={10} fill="var(--muted)">step (√ scale) →</text>
          </svg>
          <div className="small muted"><span style={{ color: "var(--accent)" }}>━ training loss</span> · <span style={{ color: "var(--accent2)" }}>┅ validation loss (text it never trained on)</span> · click the dots on the bottom to time-travel</div>
        </div>
        <div className="card">
          <div className="controls">
            <span className="muted small">Checkpoint</span>
            {log.snapshots.map((s, i) => <button key={s.step} className={i === snap ? "on" : ""} onClick={() => setSnap(i)}>step {s.step}</button>)}
          </div>
          <input type="range" min={0} max={log.snapshots.length - 1} value={snap} onChange={(e) => setSnap(+e.target.value)} style={{ width: "100%" }} />
          <div className="lane-label">What the model writes at step {cur.step}</div>
          {cur.samples.map((s, i) => (
            <div key={i} className="sample mono" data-ask-id={`snap-${cur.step}-${i}`} data-ask={`at step ${cur.step}, given "${log.prompts[i]}" the model wrote "${s.slice(log.prompts[i].length)}"`}>
              <span className="muted">{log.prompts[i]}</span>{s.slice(log.prompts[i].length)}
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <div className="lane-label">A peek at the training text (synthetic “tiny world” stories, 5.2M characters)</div>
        <pre className="corpus mono" data-ask-id="corpus" data-ask="a sample of the synthetic training corpus">{corpus.slice(0, 900)}</pre>
      </div>
      <div className="card next-level">
        <b>Next level (coming as you ask for it): train it yourself, in the browser.</b> Watch gradients flow backwards through the same picture you saw in the X-ray, change the data, the size, the learning rate — and break it on purpose.
        <button onClick={() => addWish("Level 2: train TinyGPT live in the browser, with gradients visualized flowing back through the X-ray scenes", "Training page")}>⭐ I want this next</button>
      </div>
    </div>
  );
}
