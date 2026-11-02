// The tutor conversation, shared by the side panel, the popover, keyboard shortcuts and voice.
import { ask, createStore, learner, select, type AskItem } from "./store";
import { parseAnswer, recordQA, streamAsk, type Turn } from "./api";
import { beginSpeech, cancelListening, speakStream, speakable, startListening, stopSpeaking } from "./voice";

export const convo = createStore<{ thread: Turn[]; busy: boolean }>({ thread: [], busy: false });
let abort: AbortController | null = null;

// ---- what the learner is pointing at ------------------------------------
let pointer: { id: string; label: string; ts: number } | null = null;
export function trackPointer() {
  const over = (e: MouseEvent) => {
    const t = (e.target as Element).closest?.("main [data-ask-id]");
    if (t) pointer = { id: t.getAttribute("data-ask-id")!, label: t.getAttribute("data-ask") ?? "", ts: Date.now() };
  };
  document.addEventListener("mouseover", over);
  return () => document.removeEventListener("mouseover", over);
}
export function currentPointer(maxAgeS = 10) {
  if (!pointer) return undefined;
  const ago = Math.round((Date.now() - pointer.ts) / 1000);
  return ago <= maxAgeS ? { id: pointer.id, label: pointer.label, ago } : undefined;
}

/** Text highlighted with the mouse (in the app or in an answer) becomes part of the selection. */
export function trackTextSelection() {
  const up = (e: MouseEvent) => {
    if ((e.target as Element | null)?.closest?.("textarea, input, .popover, .region-layer, [data-slot=card].fixed")) return;
    setTimeout(() => {
      const sel = window.getSelection(); const text = sel?.toString().trim() ?? "";
      if (text.length < 2 || text.length > 3000) return;
      const where = (sel!.anchorNode?.parentElement?.closest(".ask-panel") ? "in the tutor's answer" : "on the page");
      const item: AskItem = { id: "text-selection", label: `highlighted text ${where}: “${text.length > 400 ? text.slice(0, 400) + "…" : text}”` };
      const rest = ask.get().selection.filter((s) => s.id !== "text-selection");
      select([item, ...rest.filter(() => e.shiftKey)], "", false);
      const r = sel!.getRangeAt(0).getBoundingClientRect();
      afterHighlight({ x: r.left + r.width / 2, y: r.bottom });
    }, 10);
  };
  document.addEventListener("mouseup", up);
  // Touch devices: long-press selection never fires mouseup, so watch selection changes instead.
  let t = 0;
  const onSel = () => {
    if (!matchMedia("(pointer: coarse)").matches) return;
    clearTimeout(t);
    t = window.setTimeout(() => {
      const sel = window.getSelection(); const text = sel?.toString().trim() ?? "";
      if (text.length < 2 || text.length > 3000 || !sel?.anchorNode?.parentElement?.closest("main")) return;
      up(new MouseEvent("mouseup", { bubbles: true }) as MouseEvent & { target: Element });
    }, 700);
  };
  document.addEventListener("selectionchange", onSel);
  return () => { document.removeEventListener("mouseup", up); document.removeEventListener("selectionchange", onSel); clearTimeout(t); };
}

/** Called after a region / text highlight — does whatever the learner chose in settings. */
export function afterHighlight(pos: { x: number; y: number }) {
  const mode = learner.get().prefs.onHighlight;
  if (mode === "listen") { ask.set({ open: true }); listenAndAsk(); }
  else if (mode === "explain") { ask.set({ open: true }); sendQuestion("Explain what I just highlighted.", "text"); }
  else if (!ask.get().open) window.dispatchEvent(new CustomEvent("ask-popover", { detail: pos }));
}

// ---- asking ---------------------------------------------------------------
export function listenAndAsk(opts: { hold?: boolean } = {}) {
  startListening((text) => { ask.set({ open: true }); sendQuestion(text, "voice"); }, opts);
}

export async function sendQuestion(q: string, via: "text" | "voice") {
  q = q.trim();
  if (!q) return;
  if (convo.get().busy) abort?.abort();
  const sel = ask.get().selection;
  const prior = convo.get().thread.filter((t) => t.content);
  const pointerNow = sel.length ? undefined : currentPointer(via === "voice" ? 15 : 6);
  convo.set({ thread: [...prior, { role: "user", content: q, selection: sel.length ? sel : pointerNow ? [{ id: pointerNow.id, label: `(pointing at) ${pointerNow.label}` }] : [], via }, { role: "assistant", content: "" }], busy: true });
  const prefs = learner.get().prefs;
  const speak = prefs.speak === "always" || (prefs.speak === "voice" && via === "voice");
  if (speak) beginSpeech(() => { if (via === "voice" && learner.get().prefs.handsFree) setTimeout(() => listenAndAsk(), 350); });
  else stopSpeaking();
  abort = new AbortController();
  const setLast = (t: Partial<Turn>) => convo.set((s) => ({ thread: [...s.thread.slice(0, -1), { ...s.thread[s.thread.length - 1], ...t }] }));
  try {
    const full = await streamAsk(q, prior, (t) => { setLast({ content: t }); if (speak) speakStream(speakable(parseAnswer(t).body), false); },
      abort.signal, { mode: via, pointer: pointerNow });
    const { followups, body } = parseAnswer(full);
    setLast({ content: full, followups });
    if (speak) speakStream(speakable(body), true);
    recordQA(q, full, sel);
  } catch (e) {
    const aborted = (e as Error).name === "AbortError";
    setLast({ content: aborted ? "(stopped)" : `⚠️ ${(e as Error).message}`, error: true });
    if (speak && !aborted) stopSpeaking();
  } finally { convo.set({ busy: false }); }
}
export function stopAll() { abort?.abort(); stopSpeaking(); cancelListening(); }
export function newConversation() { stopAll(); convo.set({ thread: [] }); }
