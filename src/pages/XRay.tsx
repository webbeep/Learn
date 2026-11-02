import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TinyGPT } from "../model/gpt";
import { generate, neighbours, type GenSettings, type Step } from "../model/generate";
import { ask, learner, setScreen, useStore } from "../lib/store";
import { show, tokColor } from "../lib/viz";
import { Binary, BarChart3, ChevronLeft, ChevronRight, Dices, Eye, Layers, MessageSquare, Pause, Play, Scissors, Settings2, SkipBack, SkipForward, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { SCENES, type View, type SceneProps } from "../scenes/types";
import { caption, describe } from "../scenes/captions";
import { TokensScene } from "../scenes/TokensScene";
import { VectorsScene } from "../scenes/VectorsScene";
import { AttentionScene } from "../scenes/AttentionScene";
import { LayersScene } from "../scenes/LayersScene";
import { ProbsScene } from "../scenes/ProbsScene";
import { PickScene } from "../scenes/PickScene";

const PRESETS = [
  "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the",
  "zoe has a yellow shoe. q: what color is the shoe? a: the shoe is",
  "once upon a time, there was a little",
  "q: what does the owl say? a:",
  "monday tuesday wednesday",
  "the sky is",
];
const COMPONENTS = { tokens: TokensScene, vectors: VectorsScene, attention: AttentionScene, layers: LayersScene, probs: ProbsScene, pick: PickScene };
const SCENE_MS = 4200;
const ICONS = { tokens: Scissors, vectors: Binary, attention: Eye, layers: Layers, probs: BarChart3, pick: Dices };

export function XRay({ m, initialScene }: { m: TinyGPT; initialScene?: string }) {
  const [prompt, setPrompt] = useState(PRESETS[0]);
  const [settings, setSettings] = useState<GenSettings>({ n: 6, temperature: 0.8, topK: 8, seed: 7 });
  const [steps, setSteps] = useState<Step[]>(() => generate(m, m.tok.encode(PRESETS[0]).map((t) => t.id), { n: 6, temperature: 0.8, topK: 8, seed: 7 }));
  const [pos, setPos] = useState(() => Math.max(0, SCENES.findIndex((s) => s.key === initialScene))); // continuous: step*6 + scene + fraction
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showSettings, setShowSettings] = useState(false);
  const [view, setViewState] = useState<View>({ layer: 1, head: 3, query: null, mlpLayer: 2 });
  const depth = useStore(learner, (s) => s.depth);
  const extra = useMemo(() => ({ neighbours: neighbours(m) }), [m]);
  const N = SCENES.length, total = steps.length * N;
  const si = Math.min(steps.length - 1, Math.floor(pos / N)), sc = Math.floor(pos) % N, t = pos - Math.floor(pos);
  const step = steps[si], key = SCENES[sc].key;
  const activeStep = useRef<HTMLButtonElement>(null);
  useEffect(() => { // keep the active step visible in the swipeable row without scrolling the page
    const el = activeStep.current, row = el?.parentElement; if (!el || !row || row.scrollWidth <= row.clientWidth) return;
    row.scrollTo({ left: el.offsetLeft - row.clientWidth / 2 + el.clientWidth / 2, behavior: "smooth" });
  }, [sc]);
  const setView = useCallback((v: Partial<View>) => setViewState((o) => ({ ...o, ...v })), []);

  const run = useCallback((text: string, s: GenSettings, autoplay = true) => {
    setSteps(generate(m, m.tok.encode(text).map((x) => x.id), s));
    setViewState((v) => ({ ...v, query: null })); setPos(0); setPlaying(autoplay);
  }, [m]);

  const regenerate = useCallback((patch: Partial<GenSettings>, from = 0) => {
    const s = { ...settings, ...patch }; setSettings(s);
    const ids = [...m.tok.encode(prompt).map((x) => x.id), ...steps.slice(0, from).map((x) => x.chosen)];
    setSteps([...steps.slice(0, from), ...generate(m, ids, { ...s, n: s.n - from })]);
    setPos(from * N + (patch.temperature !== undefined && patch.seed === undefined ? 4 : 5)); setPlaying(false);
  }, [settings, steps, prompt, m, N]);

  // playback clock
  const last = useRef(0);
  useEffect(() => {
    if (!playing) return;
    let raf = 0; last.current = performance.now();
    const tick = (now: number) => {
      const dt = now - last.current; last.current = now;
      setPos((p) => { const np = p + (dt * speed) / SCENE_MS; if (np >= total - 0.001) { setPlaying(false); return total - 0.001; } return np; });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [playing, speed, total]);

  // clicking/selecting anything pauses the "video"
  useEffect(() => ask.subscribe(() => { if (ask.get().selection.length) setPlaying(false); }), []);
  // [[try: ...]] from the tutor
  const tryPrompt = useStore(ask, (s) => s.tryPrompt);
  useEffect(() => { if (tryPrompt) { setPrompt(tryPrompt); run(tryPrompt, settings); ask.set({ tryPrompt: null }); } }, [tryPrompt]); // eslint-disable-line

  // keyboard: space = play/pause, ←/→ = scene
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea")) return;
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      if (e.key === "ArrowRight") setPos((p) => Math.min(total - 0.001, Math.floor(p) + 1));
      if (e.key === "ArrowLeft") setPos((p) => Math.max(0, Math.ceil(p) - 1 - (p % 1 > 0 ? 0 : 0)));
    };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, [total]);

  // tell the tutor what's on screen
  useEffect(() => { setScreen("X-ray", SCENES[sc].title, describe(key, m, steps, si, view, settings)); }, [key, si, steps, view, settings, m, sc]);

  const props: SceneProps = { m, steps, si, step, t: playing ? t : Math.max(t, 1), view, setView, settings, regenerate, extra };
  const Scene = COMPONENTS[key];
  const promptLen = steps[0] ? steps[0].ctx.length : 0;

  return (
    <div className="@container space-y-4 pb-6">
      <Card className="gap-3 py-4">
        <CardContent className="space-y-3 px-4">
          <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); run(prompt, settings); }}>
            <div className="flex gap-2">
              <Textarea value={prompt} rows={2} onChange={(e) => setPrompt(e.target.value)} placeholder="Type a sentence…"
                className="min-h-16 flex-1 resize-none font-mono text-sm"
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(prompt, settings); } }} />
              <div className="flex flex-col gap-2">
                <Button type="submit"><Play />Run</Button>
                <Button type="button" variant="ghost" size="sm" onClick={() => setShowSettings(!showSettings)} aria-expanded={showSettings}><Settings2 />More</Button>
              </div>
            </div>
            {showSettings && (
              <div className="flex flex-wrap items-end gap-2">
                <Select value="" onValueChange={(v) => { setPrompt(v); run(v, settings); }}>
                  <SelectTrigger className="w-full @xl:w-72"><SelectValue placeholder="Example sentences" /></SelectTrigger>
                  <SelectContent>{PRESETS.map((p) => <SelectItem key={p} value={p} className="font-mono text-xs">{p.length > 52 ? "…" + p.slice(-50) : p}</SelectItem>)}</SelectContent>
                </Select>
                {([["Words", "n", 1, 30, 1, "How many tokens to generate"], ["Temperature", "temperature", 0, 3, 0.1, "0 = always pick the top token"], ["Top-k", "topK", 1, 50, 1, "Only sample among the k most likely tokens"]] as const).map(([label, key, min, max, step, tip]) => (
                  <div key={key} className="grid w-[96px] gap-1" title={tip}>
                    <Label className="text-xs text-muted-foreground">{label}</Label>
                    <Input type="number" min={min} max={max} step={step} value={settings[key]} className="h-8 tabular-nums"
                      onChange={(e) => setSettings({ ...settings, [key]: Math.max(min, Math.min(max, +e.target.value)) })} />
                  </div>
                ))}
              </div>
            )}
          </form>
          <div className="rounded-lg border bg-muted/40 px-3 py-2.5 font-mono text-sm leading-7 whitespace-pre-wrap">
            <span className="text-muted-foreground">{m.tok.decode(steps[0].ctx.slice(0, promptLen))}</span>
            {steps.map((s, i) => (
              <span key={i} onClick={() => { setPos(i * N + 5); setPlaying(false); }}
                className={cn("cursor-pointer rounded-sm transition-opacity", i === si && "ring-2 ring-pin", i > si && "opacity-30")}
                style={{ background: i <= si ? tokColor(s.chosen, 25) : undefined }} title={`Generated token ${i + 1}. Click to jump here.`}>{m.meta.vocab[s.chosen]}</span>
            ))}
          </div>
        </CardContent>
      </Card>

      <nav className="-mx-1 grid snap-x auto-cols-[minmax(108px,1fr)] grid-flow-col gap-2 overflow-x-auto px-1 pb-1 @3xl:mx-0 @3xl:grid-flow-row @3xl:grid-cols-6 @3xl:overflow-visible @3xl:px-0 @3xl:pb-0">
        {SCENES.map((s, i) => {
          const Icon = ICONS[s.key];
          return (
            <button key={s.key} ref={i === sc ? activeStep : undefined} onClick={() => { setPos(si * N + i); setPlaying(false); }}
              className={cn("relative flex snap-start items-center gap-2.5 overflow-hidden rounded-lg border bg-card px-3 py-2 text-left transition-colors hover:bg-muted/60 @3xl:py-2.5",
                i === sc && "border-foreground/30 bg-muted ring-1 ring-foreground/10", i < sc && "text-muted-foreground")}>
              <span className={cn("hidden size-7 shrink-0 items-center justify-center rounded-md border text-xs @xl:flex", i === sc ? "bg-primary text-primary-foreground" : "bg-background")}><Icon className="size-3.5" /></span>
              <span className="min-w-0 truncate text-sm font-medium">{s.title}</span>
              {i === sc && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-pin" style={{ width: `${t * 100}%` }} />}
            </button>
          );
        })}
      </nav>

      <Card className="py-0">
        <div className="stage p-4 md:p-5" key={key}><Scene {...props} /></div>
        <div className="flex flex-wrap items-start gap-x-3 gap-y-2 border-t bg-muted/30 px-4 py-3 md:px-5">
          <Sparkles className="mt-0.5 size-4 shrink-0 text-pin" />
          <p className="min-w-0 flex-1 basis-[calc(100%-2rem)] text-[15px] leading-relaxed text-pretty @2xl:basis-0">{caption(key, m, step, view, depth === "math")}</p>
          <div className="ml-7 flex shrink-0 items-center gap-2 @2xl:ml-0">
            <Label htmlFor="math" className="text-xs text-muted-foreground">Show the math</Label>
            <Switch id="math" checked={depth === "math"} onCheckedChange={(c) => learner.set({ depth: c ? "math" : "intuition" })} />
          </div>
        </div>
      </Card>

      <div className="sticky bottom-[4.5rem] z-20 flex justify-center md:bottom-3">
        <div className="flex w-full max-w-4xl items-center gap-1.5 rounded-xl border bg-popover/90 p-2 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-popover/75">
          <ButtonGroup>
            <Button variant="outline" size="icon-sm" onClick={() => setPos(Math.max(0, (si - 1) * N))} title="Previous token"><SkipBack /></Button>
            <Button variant="outline" size="icon-sm" onClick={() => setPos((p) => Math.max(0, Math.floor(p) - 1))} title="Previous scene (←)"><ChevronLeft /></Button>
          </ButtonGroup>
          <Button size="icon" className="rounded-full" onClick={() => { if (pos >= total - 0.01) setPos(0); setPlaying(!playing); }} title="Play / pause (Space)">
            {playing ? <Pause className="fill-current" /> : <Play className="fill-current" />}
          </Button>
          <ButtonGroup>
            <Button variant="outline" size="icon-sm" onClick={() => setPos((p) => Math.min(total - 0.001, Math.floor(p) + 1))} title="Next scene (→)"><ChevronRight /></Button>
            <Button variant="outline" size="icon-sm" onClick={() => setPos(Math.min(total - 0.001, (si + 1) * N))} title="Next token"><SkipForward /></Button>
          </ButtonGroup>
          <div className="relative mx-2 flex-1 pb-4">
            <Slider min={0} max={total - 0.001} step={0.01} value={[pos]} onValueChange={([v]) => { setPos(v); setPlaying(false); }} />
            <div className="absolute inset-x-0 bottom-0 hidden h-3 @xl:block">{steps.map((s, i) => (
              <span key={i} className={cn("absolute -translate-x-1/2 font-mono text-[10px] whitespace-nowrap", i === si ? "text-pin" : "text-muted-foreground")} style={{ left: `${((i + 0.5) / steps.length) * 100}%` }}>{show(m.meta.vocab[s.chosen])}</span>
            ))}</div>
          </div>
          <Select value={String(speed)} onValueChange={(v) => setSpeed(+v)}>
            <SelectTrigger size="sm" className="w-[68px]"><SelectValue /></SelectTrigger>
            <SelectContent>{[0.5, 1, 1.5, 2, 4].map((s) => <SelectItem key={s} value={String(s)}>{s}×</SelectItem>)}</SelectContent>
          </Select>
          <Button variant="secondary" size="sm" className="hidden sm:inline-flex" onClick={() => { setPlaying(false); ask.set({ open: true }); }}><MessageSquare />Ask</Button>
        </div>
      </div>
    </div>
  );
}
