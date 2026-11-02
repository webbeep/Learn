// The curriculum: tracks → chapters. Structure lives here; per-chapter teaching content
// (key ideas, quiz, deeper math & code) lives in ./chapters-content.ts.

export type Level = "beginner" | "intermediate" | "advanced";
export interface Track { id: "inference" | "training" | "design" | "research"; title: string; blurb: string }
export interface Chapter {
  id: string; track: Track["id"]; title: string; level: Level; minutes: number;
  video?: string; // id in public/videos/index.json
  lab?: { href: string; label: string }; // interactive page to try it
}

export const TRACKS: Track[] = [
  { id: "inference", title: "How a model thinks", blurb: "Follow one prediction through a real GPT, from text to the next word." },
  { id: "training", title: "How a model learns", blurb: "Where every number came from: loss, gradients, optimizers, and training your own." },
  { id: "design", title: "How models are designed", blurb: "Why the architecture looks the way it does, what scale buys you, and how a tiny GPT becomes ChatGPT." },
  { id: "research", title: "Invent your own", blurb: "The research loop and the open frontier, so you can try your own ideas." },
];

export const CHAPTERS: Chapter[] = [
  { id: "big-picture", track: "inference", title: "The big picture: GPT in 100 seconds", level: "beginner", minutes: 6, video: "gpt-in-100s", lab: { href: "#/xray", label: "Open the X-ray" } },
  { id: "tokens", track: "inference", title: "Tokens: how a model reads", level: "beginner", minutes: 6, video: "tokens", lab: { href: "#/xray/tokens", label: "See your text tokenized" } },
  { id: "embeddings", track: "inference", title: "Embeddings: words as vectors", level: "beginner", minutes: 7, video: "embeddings", lab: { href: "#/xray/vectors", label: "Explore the meaning map" } },
  { id: "attention", track: "inference", title: "Attention: looking back for context", level: "intermediate", minutes: 8, video: "attention", lab: { href: "#/xray/attention", label: "Inspect all 12 heads" } },
  { id: "mlp", track: "inference", title: "MLPs: where knowledge lives", level: "intermediate", minutes: 7, video: "mlp", lab: { href: "#/xray/layers", label: "Watch neurons fire" } },
  { id: "residual-stream", track: "inference", title: "The residual stream & logit lens", level: "intermediate", minutes: 7, video: "residual-stream", lab: { href: "#/xray/layers", label: "Read the logit lens" } },
  { id: "next-token", track: "inference", title: "Next token: logits, softmax & sampling", level: "beginner", minutes: 6, video: "next-token", lab: { href: "#/xray/probs", label: "Play with temperature" } },

  { id: "how-it-learned", track: "training", title: "How it learned, in 100 seconds", level: "beginner", minutes: 6, video: "training", lab: { href: "#/training", label: "Time-travel through training" } },
  { id: "loss", track: "training", title: "Loss: measuring surprise", level: "beginner", minutes: 6, video: "loss", lab: { href: "#/training", label: "See the real loss curve" } },
  { id: "backprop", track: "training", title: "Backpropagation, visually", level: "intermediate", minutes: 8, video: "backprop", lab: { href: "#/backprop", label: "Step through backprop" } },
  { id: "optimizers", track: "training", title: "Gradient descent, momentum & Adam", level: "intermediate", minutes: 8, video: "optimizers", lab: { href: "#/train", label: "Try learning rates live" } },
  { id: "train-your-own", track: "training", title: "Train your own: data, batches & overfitting", level: "intermediate", minutes: 8, video: "train-your-own", lab: { href: "#/train", label: "Train a model in your browser" } },

  { id: "depth", track: "design", title: "Residuals & LayerNorm: why depth works", level: "advanced", minutes: 7, video: "residuals-layernorm", lab: { href: "#/lab", label: "Remove them and retrain" } },
  { id: "arch-lab", track: "design", title: "What each part does: ablations", level: "intermediate", minutes: 7, video: "arch-lab", lab: { href: "#/lab", label: "Open the architecture lab" } },
  { id: "scaling", track: "design", title: "Scaling laws: what bigger buys you", level: "intermediate", minutes: 8, video: "scaling", lab: { href: "#/scaling", label: "Find the scaling law" } },
  { id: "chatgpt", track: "design", title: "From TinyGPT to ChatGPT", level: "beginner", minutes: 8, video: "tiny-to-chatgpt" },

  { id: "research-loop", track: "research", title: "The research loop", level: "intermediate", minutes: 7, video: "research-loop", lab: { href: "#/invent", label: "Run your own experiment" } },
  { id: "frontier", track: "research", title: "The frontier: ideas to build on", level: "advanced", minutes: 9, video: "frontier", lab: { href: "#/invent", label: "Pick an idea to try" } },
];

export const chapterById = (id: string) => CHAPTERS.find((c) => c.id === id);
