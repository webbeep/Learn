import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TinyGPT } from "../model/gpt";
import { generate, neighbours, type GenSettings, type Step } from "../model/generate";
import { ask, learner, setScreen, useStore } from "../lib/store";
import { show, tokColor } from "../lib/viz";
import { SCENES, type View, type SceneProps } from "../scenes/types";
import { caption, describe } from "../scenes/captions";
import { TokensScene } from "../scenes/TokensScene";
import { VectorsScene } from "../scenes/VectorsScene";
import { AttentionScene } from "../scenes/AttentionScene";
import { LayersScene } from "../scenes/LayersScene";
import { ProbsScene } from "../scenes/ProbsScene";
import { PickScene } from "../scenes/PickScene";

const PRESETS = [
  "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the",
  "zoe has a yellow shoe. q: what color is the shoe? a: the shoe is",
  "once upon a time, there was a little",
  "q: what does the owl say? a:",
  "monday tuesday wednesday",
  "the sky is",
];
const COMPONENTS = { tokens: TokensScene, vectors: VectorsScene, attention: AttentionScene, layers: LayersScene, probs: ProbsScene, pick: PickScene };
const SCENE_MS = 4200;

export function XRay({ m }: { m: TinyGPT }) {
  const [prompt, setPrompt] = useState(PRESETS[0]);
  const [settings, setSettings] = useState<GenSettings>({ n: 6, temperature: 0.8, topK: 8, seed: 7 });
  const [steps, setSteps] = useState<Step[]>(() => generate(m, m.tok.encode(PRESETS[0]).map((t) => t.id), { n: 6, temperature: 0.8, topK: 8, seed: 7 }));
  const [pos, setPos] = useState(0); // continuous: step*6 + scene + fraction
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [view, setViewState] = useState<View>({ layer: 1, head: 3, query: null, mlpLayer: 2 });
  const depth = useStore(learner, (s) => s.depth);
  const extra = useMemo(() => ({ neighbours: neighbours(m) }), [m]);
  const N = SCENES.length, total = steps.length * N;
  const si = Math.min(steps.length - 1, Math.floor(pos / N)), sc = Math.floor(pos) % N, t = pos - Math.floor(pos);
  const step = steps[si], key = SCENES[sc].key;
  const setView = useCallback((v: Partial<View>) => setViewState((o) => ({ ...o, ...v })), []);

  const run = useCallback((text: string, s: GenSettings, autoplay = true) => {
    setSteps(generate(m, m.tok.encode(text).map((x) => x.id), s));
    setViewState((v) => ({ ...v, query: null })); setPos(0); setPlaying(autoplay);
  }, [m]);

  const regenerate = useCallback((patch: Partial<GenSettings>, from = 0) => {
    const s = { ...settings, ...patch }; setSettings(s);
    const ids = [...m.tok.encode(prompt).map((x) => x.id), ...steps.slice(0, from).map((x) => x.chosen)];
    setSteps([...steps.slice(0, from), ...generate(m, ids, { ...s, n: s.n - from })]);
    setPos(from * N + (patch.temperature !== undefined && patch.seed === undefined ? 4 : 5)); setPlaying(false);
  }, [settings, steps, prompt, m, N]);

  // playback clock
  const last = useRef(0);
  useEffect(() => {
    if (!playing) return;
    let raf = 0; last.current = performance.now();
    const tick = (now: number) => {
      const dt = now - last.current; last.current = now;
      setPos((p) => { const np = p + (dt * speed) / SCENE_MS; if (np >= total - 0.001) { setPlaying(false); return total - 0.001; } return np; });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [playing, speed, total]);

  // clicking/selecting anything pauses the "video"
  useEffect(() => ask.subscribe(() => { if (ask.get().selection.length) setPlaying(false); }), []);
  // [[try: ...]] from the tutor
  const tryPrompt = useStore(ask, (s) => s.tryPrompt);
  useEffect(() => { if (tryPrompt) { setPrompt(tryPrompt); run(tryPrompt, settings); ask.set({ tryPrompt: null }); } }, [tryPrompt]); // eslint-disable-line

  // keyboard: space = play/pause, ←/→ = scene
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea")) return;
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      if (e.key === "ArrowRight") setPos((p) => Math.min(total - 0.001, Math.floor(p) + 1));
      if (e.key === "ArrowLeft") setPos((p) => Math.max(0, Math.ceil(p) - 1 - (p % 1 > 0 ? 0 : 0)));
    };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, [total]);

  // tell the tutor what's on screen
  useEffect(() => { setScreen("X-ray", SCENES[sc].title, describe(key, m, steps, si, view, settings)); }, [key, si, steps, view, settings, m, sc]);

  const props: SceneProps = { m, steps, si, step, t: playing ? t : Math.max(t, 1), view, setView, settings, regenerate, extra };
  const Scene = COMPONENTS[key];
  const promptLen = steps[0] ? steps[0].ctx.length : 0;

  return (
    <div className="xray">
      <form className="prompt-bar" onSubmit={(e) => { e.preventDefault(); run(prompt, settings); }}>
        <textarea value={prompt} rows={2} onChange={(e) => setPrompt(e.target.value)} placeholder="Type a prompt (lowercase tiny-world English)…"
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(prompt, settings); } }} />
        <div className="prompt-tools">
          <select value="" onChange={(e) => { if (e.target.value) { setPrompt(e.target.value); run(e.target.value, settings); } }}>
            <option value="">Try a preset…</option>
            {PRESETS.map((p) => <option key={p} value={p}>{p.length > 48 ? "…" + p.slice(-46) : p}</option>)}
          </select>
          <label title="How many tokens to generate">tokens <input type="number" min={1} max={30} value={settings.n} onChange={(e) => setSettings({ ...settings, n: Math.max(1, Math.min(30, +e.target.value)) })} /></label>
          <label title="0 = always pick the top token">temp <input type="number" min={0} max={3} step={0.1} value={settings.temperature} onChange={(e) => setSettings({ ...settings, temperature: +e.target.value })} /></label>
          <label title="Only sample among the k most likely tokens">top-k <input type="number" min={1} max={50} value={settings.topK} onChange={(e) => setSettings({ ...settings, topK: Math.max(1, +e.target.value) })} /></label>
          <button className="primary" type="submit">▶ Run</button>
        </div>
      </form>

      <div className="output-line mono">
        <span className="muted">{m.tok.decode(steps[0].ctx.slice(0, promptLen))}</span>
        {steps.map((s, i) => (
          <span key={i} onClick={() => { setPos(i * N + 5); setPlaying(false); }} className={`gen ${i === si ? "cur" : ""} ${i > si ? "future" : ""}`}
            style={{ background: i <= si ? tokColor(s.chosen, 25) : undefined }} title={`token ${i + 1}: click to jump`}>{m.meta.vocab[s.chosen]}</span>
        ))}
      </div>

      <nav className="pipeline">
        {SCENES.map((s, i) => (
          <button key={s.key} className={i === sc ? "on" : i < sc ? "done" : ""} onClick={() => { setPos(si * N + i); setPlaying(false); }}>
            <span className="ico">{s.icon}</span><span>{i + 1}. {s.title}</span>
            {i === sc && <span className="prog" style={{ width: `${t * 100}%` }} />}
          </button>
        ))}
      </nav>

      <div className="stage" key={`${key}`}>
        <Scene {...props} />
      </div>

      <div className="caption">
        <span className="cap-text">{caption(key, m, step, view, depth === "math")}</span>
        <button className="ghost small" onClick={() => learner.set({ depth: depth === "math" ? "intuition" : "math" })}>{depth === "math" ? "🧸 intuition" : "∑ show the math"}</button>
      </div>

      <div className="player">
        <button onClick={() => { setPos(Math.max(0, (si - 1) * N)); }} title="Previous token">⏮</button>
        <button onClick={() => setPos((p) => Math.max(0, Math.floor(p) - 1))} title="Previous scene (←)">◀</button>
        <button className="play" onClick={() => { if (pos >= total - 0.01) setPos(0); setPlaying(!playing); }} title="Play / pause (space)">{playing ? "⏸" : "▶"}</button>
        <button onClick={() => setPos((p) => Math.min(total - 0.001, Math.floor(p) + 1))} title="Next scene (→)">▶︎|</button>
        <button onClick={() => setPos(Math.min(total - 0.001, (si + 1) * N))} title="Next token">⏭</button>
        <div className="scrub">
          <input type="range" min={0} max={total - 0.001} step={0.01} value={pos} onChange={(e) => { setPos(+e.target.value); setPlaying(false); }} />
          <div className="marks">{steps.map((s, i) => <span key={i} style={{ left: `${((i + 0.5) / steps.length) * 100}%` }} className={i === si ? "cur" : ""}>{show(m.meta.vocab[s.chosen])}</span>)}</div>
        </div>
        <select value={speed} onChange={(e) => setSpeed(+e.target.value)} title="Speed">
          {[0.5, 1, 1.5, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        <button onClick={() => ask.set({ regionMode: true })} title="Drag a box over anything to ask about it">✏️ Highlight</button>
        <button className="primary" onClick={() => { setPlaying(false); ask.set({ open: true }); }}>💬 Ask</button>
      </div>
    </div>
  );
}
