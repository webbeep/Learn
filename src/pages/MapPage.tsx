import { useEffect } from "react";
import { ArrowRight, Boxes, Clapperboard, Cpu, FlaskConical, GitBranch, GraduationCap, Lightbulb, LineChart, ScanEye, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";
import { Progress } from "@/components/ui/progress";
import { CHAPTERS, TRACKS } from "../content/curriculum";
import { learner, setScreen, useStore } from "../lib/store";
import { LEVEL_LABEL, LevelSelect, nextChapter } from "./Learn";

export const LABS = [
  { href: "#/xray", icon: ScanEye, title: "Inference X-ray", blurb: "Watch a real GPT predict the next word, step by step." },
  { href: "#/training", icon: LineChart, title: "Training run", blurb: "Time-travel through the real model's 80-second training." },
  { href: "#/train", icon: Cpu, title: "Train it yourself", blurb: "Train a small GPT from scratch in your browser." },
  { href: "#/backprop", icon: GitBranch, title: "Backprop", blurb: "Step gradients backwards through a graph you can poke." },
  { href: "#/lab", icon: FlaskConical, title: "Architecture lab", blurb: "Remove a part, retrain, and see what breaks." },
  { href: "#/scaling", icon: Boxes, title: "Scaling", blurb: "Find the scaling law and budget your own training run." },
  { href: "#/invent", icon: Lightbulb, title: "Invent your own", blurb: "Design an experiment and export a PyTorch script." },
];

export function MapPage() {
  const l = useStore(learner, (s) => s);
  const done = CHAPTERS.filter((c) => l.progress[c.id]?.done).length;
  const next = nextChapter(l);
  useEffect(() => setScreen("Home", "", `Home page. Tracks: ${TRACKS.map((t) => t.title).join(", ")}. Labs: ${LABS.map((x) => x.title).join(", ")}. Learner level ${l.level ?? "not set"}, ${done}/${CHAPTERS.length} chapters done; suggested next: ${next.title}.`), [l.level, done, next.title]);
  return (
    <div className="@container space-y-10 pb-10">
      <section className="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-viz-a/15 via-card to-card px-5 py-8 @2xl:px-10 @2xl:py-12">
        <div className="pointer-events-none absolute -top-24 -right-24 size-72 rounded-full bg-viz-b/20 blur-3xl" />
        <Badge variant="outline" className="mb-4 gap-1.5 bg-background/60"><Sparkles className="size-3" />A real GPT runs in your browser</Badge>
        <h1 className="max-w-3xl text-3xl font-semibold tracking-tight text-balance @2xl:text-5xl">Learn AI top-down.</h1>
        <p className="mt-3 max-w-2xl text-pretty text-muted-foreground @2xl:mt-4 @2xl:text-lg">
          Start with a real model working in front of you. Poke it, break it, and ask about anything you see.
          Dive into the fundamentals when you hit a question, then train your own models and try your own ideas.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Button size="lg" asChild><a href={`#/learn/${next.id}`}><GraduationCap />{done ? "Continue learning" : "Start learning"}</a></Button>
          <Button size="lg" variant="outline" asChild><a href="#/xray"><ScanEye />Open the X-ray</a></Button>
        </div>
      </section>

      <a href={`#/learn/${next.id}`} className="group block">
        <Card className="flex-row items-center gap-4 p-4 transition-colors group-hover:bg-muted/40 @xl:p-5">
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground"><span className="font-medium text-pin">{done ? "Up next" : "Start here"}</span>·<span>{done}/{CHAPTERS.length} chapters done</span>{l.level && <>·<span>{LEVEL_LABEL[l.level]} path</span></>}</div>
            <div className="font-semibold leading-snug @xl:text-lg">{next.title}</div>
            <Progress value={(done / CHAPTERS.length) * 100} className="max-w-md" />
          </div>
          <ArrowRight className="size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
        </Card>
      </a>

      {l.profile?.next?.length ? (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight">Suggested from your questions</h2>
          <ItemGroup className="grid gap-2 @3xl:grid-cols-3">
            {l.profile.next.slice(0, 3).map((n, i) => (
              <Item key={i} variant="outline" asChild>
                <a href={n.where === "watch" ? "#/videos" : n.where?.match(/^(xray|training|train|backprop|lab|scaling|invent|learn(\/[\w-]+)?)$/) ? `#/${n.where}` : "#/xray"}>
                  <ItemContent><ItemTitle>{n.title}</ItemTitle><ItemDescription>{n.why}</ItemDescription></ItemContent>
                  <ItemActions><ArrowRight className="size-4" /></ItemActions>
                </a>
              </Item>
            ))}
          </ItemGroup>
        </section>
      ) : null}

      <section className="space-y-3">
        <div className="flex items-end justify-between gap-3">
          <div><h2 className="text-lg font-semibold tracking-tight">The curriculum</h2><p className="text-sm text-muted-foreground">Four tracks, from watching a model think to inventing your own.</p></div>
          <LevelSelect className="hidden @lg:flex" />
        </div>
        <div className="grid gap-3 @2xl:grid-cols-2">
          {TRACKS.map((t, i) => {
            const chs = CHAPTERS.filter((c) => c.track === t.id);
            const d = chs.filter((c) => l.progress[c.id]?.done).length;
            const first = chs.find((c) => !l.progress[c.id]?.done) ?? chs[0];
            return (
              <a key={t.id} href={`#/learn/${first.id}`} className="group" data-ask-id={`track-${t.id}`} data-ask={`Track ${i + 1}: ${t.title}. ${t.blurb} Chapters: ${chs.map((c) => c.title).join("; ")}`}>
                <Card className="h-full gap-3 p-5 transition-colors group-hover:bg-muted/40">
                  <div className="flex items-center justify-between text-xs text-muted-foreground"><span>Track {i + 1} · {chs.length} chapters</span><span className="tabular-nums">{d}/{chs.length}</span></div>
                  <div className="text-lg font-semibold">{t.title}</div>
                  <p className="text-sm text-muted-foreground">{t.blurb}</p>
                  <Progress value={(d / chs.length) * 100} className="h-1" />
                </Card>
              </a>
            );
          })}
        </div>
        <Button variant="outline" asChild><a href="#/learn">See all {CHAPTERS.length} chapters<ArrowRight /></a></Button>
      </section>

      <section className="space-y-3">
        <div><h2 className="text-lg font-semibold tracking-tight">Labs</h2><p className="text-sm text-muted-foreground">Hands-on tools. Every one is clickable and the tutor can explain anything in them.</p></div>
        <div className="grid grid-cols-1 gap-3 @lg:grid-cols-2 @4xl:grid-cols-3">
          {LABS.map((x) => (
            <a key={x.href} href={x.href} className="group flex items-start gap-3 rounded-xl border bg-card p-4 transition-colors hover:bg-muted/40" data-ask-id={`lab-${x.href.slice(2)}`} data-ask={`Lab: ${x.title}. ${x.blurb}`}>
              <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border bg-background"><x.icon className="size-4" /></div>
              <div className="min-w-0"><div className="font-medium">{x.title}</div><div className="text-sm text-muted-foreground">{x.blurb}</div></div>
            </a>
          ))}
          <a href="#/videos" className="group flex items-start gap-3 rounded-xl border bg-card p-4 transition-colors hover:bg-muted/40">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border bg-background"><Clapperboard className="size-4" /></div>
            <div className="min-w-0"><div className="font-medium">All videos</div><div className="text-sm text-muted-foreground">Every explainer in one place, plus further watching.</div></div>
          </a>
        </div>
      </section>
    </div>
  );
}
