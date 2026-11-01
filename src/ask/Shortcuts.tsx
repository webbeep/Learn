import { useEffect, useState } from "react";
import { ask, learner, setPrefs, useStore } from "../lib/store";
import { convo, listenAndAsk, sendQuestion, stopAll } from "../lib/conversation";
import { cancelListening, setHold, stopListening, stopSpeaking, voice } from "../lib/voice";

const KEYS: [string, string][] = [
  ["V", "Talk: press, speak, and it sends when you go quiet. Hold V for push-to-talk."],
  ["H", "Highlight: drag a box over anything"],
  ["E", "Explain whatever is selected, or whatever your mouse is over"],
  ["/", "Type a question"],
  ["A", "Open or close the tutor panel"],
  ["M", "Mute or unmute spoken answers"],
  ["Esc", "Stop listening or talking, cancel a highlight"],
  ["Space", "Play or pause the X-ray"],
  ["← →", "Previous or next scene"],
  ["Shift-click", "Select several things"],
  ["?", "Show this list"],
];

export function toast(msg: string) { window.dispatchEvent(new CustomEvent("toast", { detail: msg })); }

/** Global keyboard shortcuts. */
export function useShortcuts(setHelp: (f: (h: boolean) => boolean) => void) {
  useEffect(() => {
    let vDown = 0;
    const typing = (e: KeyboardEvent) => !!(e.target as HTMLElement).closest?.("input, textarea, select, [contenteditable]");
    const down = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const v = voice.get();
        if (v.listening) cancelListening(); else if (v.speaking || convo.get().busy) stopAll();
        else if (ask.get().regionMode) ask.set({ regionMode: false });
        setHelp(() => false); return;
      }
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "v") {
        e.preventDefault(); if (e.repeat) return;
        if (voice.get().listening) { stopListening(); return; }
        vDown = performance.now(); listenAndAsk({ hold: true });
      } else if (k === "h") { ask.set({ regionMode: !ask.get().regionMode }); }
      else if (k === "e") { ask.set({ open: true }); sendQuestion(ask.get().selection.length ? "Explain this." : "Explain what my mouse is pointing at.", "text"); }
      else if (k === "/") { e.preventDefault(); ask.set({ open: true }); setTimeout(() => window.dispatchEvent(new Event("focus-ask")), 30); }
      else if (k === "a") { ask.set({ open: !ask.get().open }); }
      else if (k === "m") {
        const p = learner.get().prefs;
        if (p.speak === "off") { setPrefs({ speak: "voice" }); toast("🔊 Spoken answers on (when you talk)"); }
        else { setPrefs({ speak: "off" }); stopSpeaking(); toast("🔇 Spoken answers off"); }
      } else if (e.key === "?") { setHelp((h) => !h); }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== "v" || !vDown) return;
      const held = performance.now() - vDown; vDown = 0;
      if (held > 450) stopListening(); // push-to-talk: release to send
      else setHold(false); // quick tap: keep listening until silence or the next V
    };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [setHelp]);
}

export function ShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row between"><h3>Keyboard & voice shortcuts</h3><button className="ghost" onClick={onClose}>✕</button></div>
        <table className="keys"><tbody>{KEYS.map(([k, d]) => <tr key={k}><td><kbd>{k}</kbd></td><td>{d}</td></tr>)}</tbody></table>
        <p className="muted small">Change what happens after you highlight, and how it talks back, under ⚙ in the tutor panel.</p>
      </div>
    </div>
  );
}

/** Floating indicator for listening / transcribing / speaking + toasts. */
export function VoiceHUD() {
  const v = useStore(voice, (s) => s);
  const open = useStore(ask, (s) => s.open);
  const [msg, setMsg] = useState("");
  useEffect(() => {
    let t = 0;
    const f = (e: Event) => { setMsg((e as CustomEvent).detail); clearTimeout(t); t = window.setTimeout(() => setMsg(""), 2200); };
    window.addEventListener("toast", f); return () => window.removeEventListener("toast", f);
  }, []);
  useEffect(() => { if (v.error) { toast(`⚠️ ${v.error}`); voice.set({ error: "" }); } }, [v.error]);
  const show = v.listening || v.transcribing || v.speaking;
  return (<>
    {show && (
      <div className={`voice-hud ${open ? "with-panel" : ""}`}>
        {v.listening ? (<>
          <span className="orb" style={{ transform: `scale(${1 + v.level * 0.6})` }} />
          <span className="hud-text">{v.interim || "Listening… ask about what you're looking at"}</span>
          <button className="primary" onClick={stopListening}>Send</button>
          <button className="ghost" onClick={cancelListening}>Cancel</button>
        </>) : v.transcribing ? (<><span className="orb busy" /><span className="hud-text">Transcribing…</span></>) : (<>
          <span className="eq"><i /><i /><i /><i /></span><span className="hud-text">Speaking…</span>
          <button className="ghost" onClick={stopSpeaking}>Stop (Esc)</button>
        </>)}
      </div>
    )}
    {msg && <div className="toast">{msg}</div>}
  </>);
}
