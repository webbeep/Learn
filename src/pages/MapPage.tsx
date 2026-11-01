import { useEffect } from "react";
import { ArrowRight, Boxes, Clapperboard, Cpu, FlaskConical, GitBranch, Lightbulb, LineChart, ScanEye, Sparkles, Star, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, CardAction } from "@/components/ui/card";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";
import { cn } from "@/lib/utils";
import { addWish, learner, setScreen, useStore } from "../lib/store";

export const LEVELS = [
  { n: 0, href: "#/xray", icon: ScanEye, title: "Watch a real model think", sub: "Inference X-ray", status: "open",
    blurb: "Type a sentence, press play, and watch a real tiny GPT turn it into the next word: tokens, vectors, attention, layers, probabilities, and a dice roll." },
  { n: 1, href: "#/training", icon: LineChart, title: "Where the numbers came from", sub: "Training peek", status: "open",
    blurb: "The loss curve of the exact model you just watched, and what it wrote at each stage of its 80-second life." },
  { n: 2, icon: Cpu, title: "Train it yourself", sub: "In-browser training", status: "next",
    blurb: "Press train, watch the loss fall live, and see gradients flow backward through the same X-ray. Change the data and watch it learn something new." },
  { n: 3, icon: GitBranch, title: "Backprop, visualized", sub: "Dive down when you need it", status: "later",
    blurb: "Why does each weight move the way it does? The chain rule as a picture, only when you get curious." },
  { n: 4, icon: FlaskConical, title: "Architecture lab", sub: "Rewire the model", status: "later",
    blurb: "Remove attention. Add layers. Swap the MLP. Retrain and compare to learn design by breaking things." },
  { n: 5, icon: Boxes, title: "Scaling", sub: "Why bigger works", status: "later",
    blurb: "Train 5 sizes, plot loss against compute, and find the scaling law yourself." },
  { n: 6, icon: Lightbulb, title: "Invent your own", sub: "A new kind of model", status: "later",
    blurb: "Your idea, your training loop, your framework, with everything above as your toolbox." },
];

export function MapPage() {
  const wishes = useStore(learner, (s) => s.wishes);
  const profile = useStore(learner, (s) => s.profile);
  useEffect(() => setScreen("Journey", "", `The learning journey has levels: ${LEVELS.map((l) => `${l.n}. ${l.title} (${l.status})`).join("; ")}.`), []);
  return (
    <div className="space-y-10 pb-10">
      <section className="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-viz-a/15 via-card to-card px-6 py-10 md:px-10 md:py-14">
        <div className="pointer-events-none absolute -top-24 -right-24 size-72 rounded-full bg-viz-b/20 blur-3xl" />
        <Badge variant="outline" className="mb-4 gap-1.5 bg-background/60"><Sparkles className="size-3" />A real GPT runs in your browser</Badge>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-balance md:text-5xl">Learn AI top-down.</h1>
        <p className="mt-4 max-w-2xl text-lg text-muted-foreground text-pretty">
          Start with a real model working in front of you. Poke it, break it, and ask about anything you see.
          Every pixel can be clicked, and the tutor sees exactly what you're pointing at. Dive into the fundamentals only when you hit a question.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Button size="lg" asChild><a href="#/xray"><ScanEye />Watch a model think</a></Button>
          <Button size="lg" variant="outline" asChild><a href="#/videos"><Clapperboard />Watch the videos</a></Button>
        </div>
      </section>

      {profile?.next?.length ? (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight">Suggested for you</h2>
          <ItemGroup className="grid gap-2 md:grid-cols-3">
            {profile.next.slice(0, 3).map((n, i) => (
              <Item key={i} variant="outline" asChild>
                <a href={n.where === "training" ? "#/training" : n.where === "watch" ? "#/videos" : "#/xray"}>
                  <ItemContent><ItemTitle>{n.title}</ItemTitle><ItemDescription>{n.why}</ItemDescription></ItemContent>
                  <ItemActions><ArrowRight className="size-4" /></ItemActions>
                </a>
              </Item>
            ))}
          </ItemGroup>
        </section>
      ) : null}

      <section className="space-y-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">The journey</h2>
          <p className="text-sm text-muted-foreground">Levels open as you ask for them. Your questions decide what gets built next.</p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {LEVELS.map((l) => {
            const wanted = wishes.some((w) => w.text.startsWith(`Level ${l.n}:`));
            return (
              <Card key={l.n} className={cn("transition-colors", l.status === "open" && "hover:border-foreground/20", l.status === "later" && "opacity-80")}
                data-ask-id={`level-${l.n}`} data-ask={`Level ${l.n}: ${l.title}. ${l.blurb} (${l.status})`}>
                <CardHeader>
                  <div className={cn("mb-2 flex size-9 items-center justify-center rounded-lg border", l.status === "open" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}><l.icon className="size-4" /></div>
                  <CardDescription>Level {l.n} · {l.sub}</CardDescription>
                  <CardTitle>{l.title}</CardTitle>
                  <CardAction>
                    <Badge variant={l.status === "open" ? "default" : l.status === "next" ? "secondary" : "outline"}>{l.status === "open" ? "Open" : l.status === "next" ? "Up next" : "Later"}</Badge>
                  </CardAction>
                </CardHeader>
                <CardContent className="flex-1 text-sm text-muted-foreground">{l.blurb}</CardContent>
                <CardFooter>
                  {l.href ? <Button asChild className="w-full"><a href={l.href}>Open<ArrowRight /></a></Button> :
                    <Button variant="outline" className="w-full" disabled={wanted} onClick={() => addWish(`Level ${l.n}: ${l.title}: ${l.blurb}`, "journey map")}>
                      {wanted ? <><Check />On your build list</> : <><Star />I want this next</>}
                    </Button>}
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
