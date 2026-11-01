// Tiny global store + persisted learner model (localStorage).
import { useSyncExternalStore } from "react";

export function createStore<T>(initial: T) {
  let state = initial;
  const subs = new Set<() => void>();
  return {
    get: () => state,
    set(patch: Partial<T> | ((s: T) => Partial<T>)) {
      state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
      subs.forEach((f) => f());
    },
    subscribe(f: () => void) { subs.add(f); return () => { subs.delete(f); }; },
  };
}
export function useStore<T, S>(store: ReturnType<typeof createStore<T>>, sel: (s: T) => S): S {
  return useSyncExternalStore(store.subscribe, () => sel(store.get()));
}

// ---------- learner -------------------------------------------------------
export interface QA { id: string; ts: number; where: string; selection: string[]; q: string; a: string }
export interface Profile {
  summary: string; knows: string[]; confusions: string[]; interests: string[]; style: string;
  next: { title: string; why: string; where: string; try?: string }[]; build: string[]; updated: number;
}
export interface Wish { id: string; ts: number; text: string; source: string; done?: boolean }
export interface Learner {
  goal: string;
  depth: "intuition" | "balanced" | "math";
  length: "short" | "medium" | "long";
  history: QA[];
  profile: Profile | null;
  wishes: Wish[];
  visited: Record<string, number>;
}
const KEY = "topdown-ai-learner-v1";
const DEFAULT: Learner = {
  goal: "Understand LLMs top-down, then train models from scratch and invent a brand-new kind of AI model/training framework.",
  depth: "balanced", length: "medium", history: [], profile: null, wishes: [], visited: {},
};
function load(): Learner {
  try { return { ...DEFAULT, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { return DEFAULT; }
}
export const learner = createStore<Learner>(load());
learner.subscribe(() => { try { localStorage.setItem(KEY, JSON.stringify(learner.get())); } catch { /* private mode */ } });

export const uid = () => Math.random().toString(36).slice(2, 10);

export function addWish(text: string, source: string) {
  const t = text.trim(); if (!t) return;
  learner.set((s) => s.wishes.some((w) => w.text.toLowerCase() === t.toLowerCase()) ? {} :
    { wishes: [...s.wishes, { id: uid(), ts: Date.now(), text: t, source }] });
}
export function markVisited(page: string) {
  learner.set((s) => ({ visited: { ...s.visited, [page]: (s.visited[page] ?? 0) + 1 } }));
}

// ---------- ask / selection ------------------------------------------------
export interface AskItem { id: string; label: string }
export interface AskState {
  open: boolean;
  page: string; scene: string; summary: string; // what's on screen (set by each page/scene)
  selection: AskItem[]; region: string;
  highlight: string | null; // element id being pointed at by the tutor
  regionMode: boolean;
  draft: string; // text to prefill
  tryPrompt: string | null; // [[try: ...]] request for the X-ray
}
export const ask = createStore<AskState>({
  open: false, page: "", scene: "", summary: "", selection: [], region: "", highlight: null,
  regionMode: false, draft: "", tryPrompt: null,
});
export function setScreen(page: string, scene: string, summary: string) {
  const s = ask.get();
  if (s.page !== page || s.scene !== scene || s.summary !== summary) ask.set({ page, scene, summary });
}
export function select(items: AskItem[], region = "", openPanel = true) {
  ask.set((s) => ({ selection: items, region, open: openPanel || s.open }));
}
