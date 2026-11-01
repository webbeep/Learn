import { useEffect, useState } from "react";
import { learner, setPrefs, useStore, type Prefs } from "../lib/store";
import { englishVoices, hasBrowserSTT, speakText } from "../lib/voice";

const NATURAL = [["luna", "Luna (warm, female)"], ["asteria", "Asteria (clear, female)"], ["thalia", "Thalia (bright, female)"], ["helena", "Helena (calm, female)"],
  ["orion", "Orion (male)"], ["arcas", "Arcas (deep, male)"], ["apollo", "Apollo (casual, male)"], ["zeus", "Zeus (male)"]];

function Seg<K extends keyof Prefs>({ k, opts }: { k: K; opts: [Prefs[K], string][] }) {
  const v = useStore(learner, (s) => s.prefs[k]);
  return <div className="seg">{opts.map(([val, label]) => <button key={String(val)} className={v === val ? "on" : ""} onClick={() => setPrefs({ [k]: val } as Partial<Prefs>)}>{label}</button>)}</div>;
}

export function VoiceSettings() {
  const p = useStore(learner, (s) => s.prefs);
  const [voices, setVoices] = useState(englishVoices());
  useEffect(() => { const f = () => setVoices(englishVoices()); speechSynthesis?.addEventListener?.("voiceschanged", f); return () => speechSynthesis?.removeEventListener?.("voiceschanged", f); }, []);
  return (
    <div className="settings">
      <h4>When I highlight something</h4>
      <Seg k="onHighlight" opts={[["manual", "Show options"], ["listen", "🎙 Start listening"], ["explain", "✨ Explain right away"]]} />
      <p className="muted small">Applies to text you highlight and boxes you drag (H). A plain click always shows the options.</p>

      <h4>🎙 Talking to it</h4>
      <Seg k="stt" opts={[["auto", "Auto"], ["browser", "Browser (live words)"], ["whisper", "Whisper (most accurate)"]]} />
      <p className="muted small">{hasBrowserSTT ? "Your browser can transcribe live as you talk. Whisper runs on Cloudflare and works in every browser." : "This browser has no built-in speech recognition, so Whisper on Cloudflare will be used."} Press <kbd>V</kbd> to talk and stop when you're done (it also stops on silence), or hold <kbd>V</kbd> for push-to-talk.</p>
      <label className="check"><input type="checkbox" checked={p.handsFree} onChange={(e) => setPrefs({ handsFree: e.target.checked })} /> Hands-free conversation: start listening again after each spoken answer</label>

      <h4>🔊 Talking back</h4>
      <Seg k="speak" opts={[["off", "Off"], ["voice", "When I speak"], ["always", "Always"]]} />
      <Seg k="ttsEngine" opts={[["natural", "Natural voice (Cloudflare)"], ["browser", "Browser voice"]]} />
      {p.ttsEngine === "natural" ? (
        <select value={p.naturalVoice} onChange={(e) => setPrefs({ naturalVoice: e.target.value })}>{NATURAL.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
      ) : (
        <select value={p.browserVoice} onChange={(e) => setPrefs({ browserVoice: e.target.value })}>
          <option value="">Best available</option>{voices.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
        </select>
      )}
      <label className="row small">Speed {p.rate.toFixed(2)}× <input type="range" min={0.7} max={1.6} step={0.05} value={p.rate} onChange={(e) => setPrefs({ rate: +e.target.value })} /></label>
      <button onClick={() => speakText("Hi! Highlight anything, press V, and ask me about it. I can see what's in your window.")}>▶ Test voice</button>
      <p className="muted small">Press <kbd>M</kbd> to mute or unmute, and <kbd>Esc</kbd> to stop talking.</p>
    </div>
  );
}
