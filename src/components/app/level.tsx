import { GraduationCap } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { CHAPTERS, type Chapter, type Level } from "@/content/curriculum";
import { learner, useStore, type Learner } from "@/lib/store";

export const LEVEL_LABEL: Record<Level, string> = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" };
export const LEVEL_RANK: Record<Level, number> = { beginner: 0, intermediate: 1, advanced: 2 };

/** The next chapter to do, honoring the learner's level (advanced folks can skip beginner chapters). */
export function nextChapter(l: Pick<Learner, "level" | "progress">): Chapter {
  const rank = LEVEL_RANK[l.level ?? "beginner"];
  return CHAPTERS.find((c) => !l.progress[c.id]?.done && (rank < 2 || LEVEL_RANK[c.level] > 0)) ?? CHAPTERS.find((c) => !l.progress[c.id]?.done) ?? CHAPTERS[0];
}

export function LevelSelect({ className }: { className?: string }) {
  const level = useStore(learner, (s) => s.level);
  return (
    <Select value={level ?? "beginner"} onValueChange={(v) => setLevel(v as Level)}>
      <SelectTrigger size="sm" className={cn("w-[150px]", className)}><GraduationCap /><SelectValue /></SelectTrigger>
      <SelectContent>{(Object.keys(LEVEL_LABEL) as Level[]).map((k) => <SelectItem key={k} value={k}>{LEVEL_LABEL[k]}</SelectItem>)}</SelectContent>
    </Select>
  );
}
export function setLevel(level: Level) {
  learner.set({ level, depth: level === "beginner" ? "intuition" : level === "advanced" ? "math" : "balanced" });
}

