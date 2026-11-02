import { useState } from "react";
import { Baby, Brain, Rocket } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { learner, useStore } from "@/lib/store";
import { setLevel } from "@/pages/Learn";
import type { Level } from "@/content/curriculum";

const OPTIONS: { id: Level; icon: typeof Baby; title: string; body: string }[] = [
  { id: "beginner", icon: Baby, title: "New to this", body: "I use AI tools but don't know how they work inside. Plain words and pictures, please." },
  { id: "intermediate", icon: Brain, title: "I know some basics", body: "I code, and I've heard of neural nets, gradients or attention. Show me the mechanisms." },
  { id: "advanced", icon: Rocket, title: "I've trained models", body: "I know PyTorch and backprop. Give me the math, the shapes and the code." },
];

/** First-visit question: tailors chapter order, explanation depth and the tutor. */
export function Onboarding() {
  const level = useStore(learner, (s) => s.level);
  const [pick, setPick] = useState<Level>("beginner");
  const [open, setOpen] = useState(level === null);
  const done = (l: Level) => { setLevel(l); setOpen(false); };
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) done(level ?? pick); }}>
      <DialogContent className="max-h-[90svh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Welcome to Top-Down AI</DialogTitle>
          <DialogDescription>You'll learn from a real model running right here. How familiar are you with how AI works? You can change this any time.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-2">
          {OPTIONS.map((o) => (
            <button key={o.id} onClick={() => setPick(o.id)}
              className={cn("flex items-start gap-3 rounded-xl border p-3 text-left transition-colors hover:bg-muted/50", pick === o.id && "border-foreground/40 bg-muted ring-1 ring-foreground/10")}>
              <div className={cn("flex size-9 shrink-0 items-center justify-center rounded-lg border", pick === o.id ? "bg-primary text-primary-foreground" : "bg-background")}><o.icon className="size-4" /></div>
              <div><div className="font-medium">{o.title}</div><div className="text-sm text-muted-foreground">{o.body}</div></div>
            </button>
          ))}
        </div>
        <DialogFooter><Button className="w-full sm:w-auto" onClick={() => done(pick)}>Let's go</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
