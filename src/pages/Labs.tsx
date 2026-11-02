import { useEffect } from "react";
import { Boxes, Cpu, FlaskConical, GitBranch, Lightbulb, LineChart, ScanEye } from "lucide-react";
import { setScreen } from "../lib/store";

export const LABS = [
  { href: "#/xray", icon: ScanEye, title: "X-ray", hint: "Watch it pick the next word" },
  { href: "#/train", icon: Cpu, title: "Train a model", hint: "From scratch, in your browser" },
  { href: "#/backprop", icon: GitBranch, title: "Backprop", hint: "See how it learns from mistakes" },
  { href: "#/training", icon: LineChart, title: "Training replay", hint: "The real model, step by step" },
  { href: "#/lab", icon: FlaskConical, title: "Remove parts", hint: "What breaks without each piece" },
  { href: "#/scaling", icon: Boxes, title: "Scaling", hint: "What bigger buys you" },
  { href: "#/invent", icon: Lightbulb, title: "Your experiment", hint: "Test your own idea" },
];

export function Labs() {
  useEffect(() => setScreen("Labs", "", `Labs: ${LABS.map((l) => `${l.title} (${l.hint})`).join(", ")}.`), []);
  return (
    <div className="mx-auto max-w-3xl space-y-6 pb-10">
      <h1 className="text-3xl font-semibold tracking-tight">Labs</h1>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {LABS.map((x) => (
          <a key={x.href} href={x.href} className="group flex flex-col gap-3 rounded-xl border bg-card p-4 transition-colors hover:bg-muted/50"
            data-ask-id={`lab-${x.href.slice(2)}`} data-ask={`Lab: ${x.title}. ${x.hint}.`}>
            <x.icon className="size-6 text-muted-foreground transition-colors group-hover:text-foreground" />
            <div><div className="font-medium">{x.title}</div><div className="text-sm text-muted-foreground">{x.hint}</div></div>
          </a>
        ))}
      </div>
    </div>
  );
}
