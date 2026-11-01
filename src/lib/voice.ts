// Voice in (speech → text) and voice out (text → speech).
//  - Speech recognition: the browser's built-in engine (live words as you talk) or
//    Whisper on Cloudflare Workers AI (records audio, more accurate, works in every browser).
//  - Speech output: the browser's voices, or a natural Deepgram Aura voice from Workers AI.
import { createStore, learner } from "./store";

export interface VoiceState { listening: boolean; transcribing: boolean; interim: string; speaking: boolean; error: string; engine: string; level: number }
export const voice = createStore<VoiceState>({ listening: false, transcribing: false, interim: "", speaking: false, error: "", engine: "", level: 0 });

/* eslint-disable @typescript-eslint/no-explicit-any */
const SR: any = typeof window !== "undefined" ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition : null;
export const hasBrowserSTT = !!SR;
export const hasMic = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

interface Session { finish: () => void; cancel: () => void; hold: boolean }
let session: Session | null = null;

/** Start listening. onFinal gets the full transcript when you stop (or go quiet). */
export function startListening(onFinal: (text: string) => void, opts: { hold?: boolean } = {}) {
  if (session) return;
  stopSpeaking();
  const pref = learner.get().prefs.stt;
  const useBrowser = pref === "browser" ? hasBrowserSTT : pref === "whisper" ? false : hasBrowserSTT;
  voice.set({ error: "", interim: "", engine: useBrowser ? "browser" : "whisper" });
  if (useBrowser) browserSession(onFinal, !!opts.hold); else whisperSession(onFinal, !!opts.hold);
}
export function stopListening() { session?.finish(); }
export function cancelListening() { session?.cancel(); }
export function setHold(h: boolean) { if (session) session.hold = h; }

function browserSession(onFinal: (t: string) => void, hold: boolean) {
  const r = new SR(); r.lang = "en-US"; r.interimResults = true; r.continuous = true;
  let finalText = "", interim = "", cancelled = false, timer = 0;
  const s: Session = { hold, finish: () => r.stop(), cancel: () => { cancelled = true; r.abort(); } };
  session = s;
  const arm = (ms: number) => { clearTimeout(timer); timer = window.setTimeout(() => { if (!s.hold) r.stop(); else arm(ms); }, ms); };
  arm(8000);
  r.onresult = (e: any) => {
    interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += t; else interim += t;
    }
    voice.set({ interim: (finalText + interim).trim(), level: Math.random() * 0.5 + 0.5 });
    arm(1800); // stop after ~2s of silence
  };
  r.onerror = (e: any) => { if (e.error !== "aborted" && e.error !== "no-speech") voice.set({ error: e.error === "not-allowed" ? "Microphone permission denied" : `Speech recognition: ${e.error}` }); };
  r.onend = () => {
    clearTimeout(timer); session = null;
    const text = (finalText + " " + interim).trim();
    voice.set({ listening: false, interim: "", level: 0 });
    if (!cancelled && text) onFinal(text);
  };
  try { r.start(); voice.set({ listening: true }); } catch (e) { session = null; voice.set({ error: String(e) }); }
}

function whisperSession(onFinal: (t: string) => void, hold: boolean) {
  if (!hasMic) { voice.set({ error: "No microphone access in this browser" }); return; }
  let cancelled = false, rec: MediaRecorder | null = null, stream: MediaStream | null = null, raf = 0, ctx: AudioContext | null = null;
  const chunks: Blob[] = [];
  const s: Session = { hold, finish: () => rec?.state === "recording" && rec.stop(), cancel: () => { cancelled = true; rec?.state === "recording" ? rec.stop() : cleanup(); } };
  session = s;
  const cleanup = () => { cancelAnimationFrame(raf); stream?.getTracks().forEach((t) => t.stop()); ctx?.close().catch(() => {}); session = null; voice.set({ listening: false, level: 0 }); };
  voice.set({ listening: true, interim: "" });
  navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }).then((st) => {
    if (cancelled) { st.getTracks().forEach((t) => t.stop()); session = null; voice.set({ listening: false }); return; }
    stream = st;
    const type = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
    rec = new MediaRecorder(st, type ? { mimeType: type } : undefined);
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      cleanup();
      if (cancelled || !chunks.length) return;
      voice.set({ transcribing: true, interim: "Transcribing…" });
      try {
        const res = await fetch("/api/transcribe", { method: "POST", headers: { "content-type": rec!.mimeType || "audio/webm" }, body: new Blob(chunks, { type: rec!.mimeType }) });
        if (!res.ok) throw new Error(await res.text());
        const { text } = await res.json();
        voice.set({ transcribing: false, interim: "" });
        if (text) onFinal(text); else voice.set({ error: "Didn't catch that — try again" });
      } catch (e) { voice.set({ transcribing: false, interim: "", error: String((e as Error).message).slice(0, 140) }); }
    };
    rec.start(250);
    // voice-activity detection: stop after ~1.8s of quiet once you've started talking
    ctx = new AudioContext(); const an = ctx.createAnalyser(); an.fftSize = 1024;
    ctx.createMediaStreamSource(st).connect(an);
    const data = new Float32Array(an.fftSize); let heard = false, quietSince = performance.now(); const t0 = performance.now();
    const loop = () => {
      an.getFloatTimeDomainData(data);
      let rms = 0; for (const v of data) rms += v * v; rms = Math.sqrt(rms / data.length);
      const level = Math.min(1, rms * 12); voice.set({ level });
      const now = performance.now();
      if (rms > 0.02) { heard = true; quietSince = now; }
      if (!s.hold && ((heard && now - quietSince > 1800) || (!heard && now - t0 > 8000) || now - t0 > 60000)) { s.finish(); return; }
      raf = requestAnimationFrame(loop);
    };
    loop();
  }).catch(() => { session = null; voice.set({ listening: false, error: "Microphone permission denied" }); });
}

// ---------------------------------------------------------------- speech out
/** Turn tutor markdown into something pleasant to hear. */
export function speakable(md: string) {
  return md
    .replace(/\[\[\s*try:\s*([^\]]+?)\s*\]\]/gi, "")
    .replace(/\[\[[^\]]*\]\]/g, "")
    .split("\n").filter((l) => !/^\s*\|/.test(l) && !/^\s*```/.test(l)).join("\n")
    .replace(/`([^`]*)`/g, "$1").replace(/[*_#>]+/g, "").replace(/␣/g, "")
    .replace(/→/g, " to ").replace(/≈/g, " about ").replace(/×/g, " times ").replace(/\s+%/g, "%")
    .replace(/[ \t]+/g, " ");
}

let queue: string[] = [], spokenUpTo = 0, done = false, playing = false, gen = 0;
let audio: HTMLAudioElement | null = null;
let onIdle: (() => void) | null = null;

/** Feed the growing answer text while it streams; whole sentences are spoken as they complete. */
export function speakStream(fullClean: string, final: boolean) {
  const rest = fullClean.slice(spokenUpTo);
  let cut = -1;
  if (final) cut = rest.length;
  else { const re = /[.!?:](?=\s)|\n/g; let m; while ((m = re.exec(rest))) cut = m.index + 1; }
  if (cut > 0) {
    const chunk = rest.slice(0, cut).trim(); spokenUpTo += cut;
    // group short sentences so natural TTS makes fewer requests
    for (const s of chunk.split(/(?<=[.!?])\s+|\n+/).map((x) => x.trim()).filter((x) => /[a-z0-9]/i.test(x))) {
      if (queue.length && queue[queue.length - 1].length + s.length < 160) queue[queue.length - 1] += " " + s; else queue.push(s);
    }
  }
  if (final) done = true;
  pump();
}
export function beginSpeech(idle?: () => void) { stopSpeaking(); spokenUpTo = 0; done = false; onIdle = idle ?? null; }
export function stopSpeaking() {
  gen++; queue = []; done = true; playing = false; onIdle = null;
  try { speechSynthesis.cancel(); } catch { /* none */ }
  if (audio) { audio.pause(); audio = null; }
  voice.set({ speaking: false });
}
export function speakText(text: string) { beginSpeech(); speakStream(speakable(text), true); }

const prefetch = new Map<string, Promise<string>>();
function naturalUrl(text: string) {
  const p = learner.get().prefs;
  const key = p.naturalVoice + "|" + text;
  if (!prefetch.has(key)) prefetch.set(key, fetch("/api/speak", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, voice: p.naturalVoice }) })
    .then(async (r) => { if (!r.ok) throw new Error(await r.text()); return URL.createObjectURL(await r.blob()); }));
  return prefetch.get(key)!;
}

function pump() {
  if (playing) return;
  const myGen = gen;
  const next = queue.shift();
  if (!next) {
    if (done) { voice.set({ speaking: false }); const f = onIdle; onIdle = null; f?.(); }
    return;
  }
  playing = true; voice.set({ speaking: true });
  const p = learner.get().prefs;
  const after = () => { if (myGen !== gen) return; playing = false; pump(); };
  if (p.ttsEngine === "natural") {
    if (queue[0]) naturalUrl(queue[0]).catch(() => {}); // prefetch the next chunk
    naturalUrl(next).then((url) => {
      if (myGen !== gen) return;
      audio = new Audio(url); audio.playbackRate = p.rate;
      audio.onended = after; audio.onerror = after;
      audio.play().catch(() => { browserSay(next, after); });
    }).catch(() => browserSay(next, after)); // fall back to the browser voice
  } else browserSay(next, after);
}

function browserSay(text: string, after: () => void) {
  if (typeof speechSynthesis === "undefined") return after();
  const u = new SpeechSynthesisUtterance(text);
  const p = learner.get().prefs;
  const v = speechSynthesis.getVoices().find((x) => x.name === p.browserVoice) ?? bestVoice();
  if (v) u.voice = v;
  u.rate = p.rate; u.onend = after; u.onerror = after;
  speechSynthesis.speak(u);
}
export function englishVoices() { return typeof speechSynthesis === "undefined" ? [] : speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en")); }
function bestVoice() {
  const vs = englishVoices();
  return vs.find((v) => /natural|neural|online/i.test(v.name)) ?? vs.find((v) => /google us|samantha|aria|jenny/i.test(v.name)) ?? vs[0];
}
