// Per-chapter teaching content. Filled in by the curriculum author; keyed by chapter id (see curriculum.ts).
export interface QuizQuestion { q: string; options: string[]; answer: number; why: string }
export interface ChapterContent {
  /** One sentence: what you'll understand after this chapter. */
  promise: string;
  /** 3–5 visual-first "key idea" cards. */
  keyIdeas: { title: string; body: string; tryIt?: string }[];
  /** 3 multiple-choice checks; `answer` is the index of the correct option. */
  quiz: QuizQuestion[];
  /** For people who want the mechanism: plain-text math (no LaTeX) and a short PyTorch snippet. */
  deeper: { math: string; code: string; codeCaption: string };
  /** Common misconception and the correction. */
  pitfall?: { myth: string; truth: string };
  /** Next steps beyond this app (well-known resources only). */
  further?: { title: string; url: string; note: string }[];
}
export const CONTENT: Record<string, ChapterContent> = {};
