import type { TinyGPT } from "../model/gpt";
import type { GenSettings, Step } from "../model/generate";

export interface View { layer: number; head: number; query: number | null; mlpLayer: number }
export interface SceneProps {
  m: TinyGPT; steps: Step[]; si: number; step: Step; t: number;
  view: View; setView: (v: Partial<View>) => void;
  settings: GenSettings; regenerate: (s: Partial<GenSettings>, fromStep?: number) => void;
  extra: { neighbours: number[][] };
}
export const SCENES = [
  { key: "tokens", icon: "✂️", title: "Tokens" },
  { key: "vectors", icon: "🔢", title: "Vectors" },
  { key: "attention", icon: "👀", title: "Attention" },
  { key: "layers", icon: "🧠", title: "Layers" },
  { key: "probs", icon: "📊", title: "Probabilities" },
  { key: "pick", icon: "🎲", title: "Pick next" },
] as const;
export type SceneKey = (typeof SCENES)[number]["key"];
