import { useEffect, useState } from "react";
import { Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { setScreen } from "../lib/store";

interface Log { steps: number[]; train: number[]; val: number[]; snapshots: { step: number; samples: string[] }[]; prompts: string[]; params: number; tokens: number; batch: number; lr: number }

export function Training({ corpus }: { corpus: string }) {
  const [log, setLog] = useState<Log | null>(null);
  const [snap, setSnap] = useState(0);
  useEffect(() => { fetch("/model/training.json").then((r) => r.json()).then((l: Log) => { setLog(l); setSnap(l.snapshots.length - 1); }); }, []);
  useEffect(() => {
    if (!log) return;
    const s = log.snapshots[snap];
    setScreen("Training", `Checkpoint step ${s.step}`, [
      `Training run: ${log.params.toLocaleString()} parameters, ${log.tokens.toLocaleString()} training tokens, batch ${log.batch}×64 tokens, AdamW lr ${log.lr}, 3000 steps.`,
      `Loss went from ${log.train[0]} to ${log.train[log.train.length - 1]} (ln(256)=5.55 is pure guessing).`,
      `Showing samples from checkpoint at step ${s.step}: ${s.samples.map((x, i) => `[prompt "${log.prompts[i]}"] → "${x}"`).join(" | ")}`,
    ].join("\n"));
  }, [log, snap]);
  if (!log) return null;

  const W = 720, H = 300, maxS = log.steps[log.steps.length - 1], maxL = Math.max(...log.train);
  const X = (s: number) => 44 + Math.sqrt(s / maxS) * (W - 64), Y = (l: number) => 12 + (1 - l / maxL) * (H - 44);
  const line = (arr: number[]) => arr.map((l, i) => `${i ? "L" : "M"}${X(log.steps[i]).toFixed(1)},${Y(l).toFixed(1)}`).join("");
  const cur = log.snapshots[snap];
  const stats: [string, string][] = [["Parameters", log.params.toLocaleString()], ["Training tokens", `${(log.tokens / 1e6).toFixed(1)}M`], ["Steps", "3,000"], ["Final loss", log.train[log.train.length - 1].toFixed(2)]];
  return (
    <div className="space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">How it learned</h1>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {stats.map(([k, v]) => <Card key={k} className="gap-1 py-4"><CardContent className="px-4"><div className="text-xs text-muted-foreground">{k}</div><div className="text-2xl font-semibold tabular-nums">{v}</div></CardContent></Card>)}
      </div>
      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader><CardTitle>Loss</CardTitle></CardHeader>
          <CardContent>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
              {[0, 1, 2, 3, 4, 5].map((l) => <g key={l}><line x1={44} x2={W - 16} y1={Y(l)} y2={Y(l)} stroke="var(--viz-grid)" /><text x={36} y={Y(l) + 4} fontSize={11} textAnchor="end" className="fill-muted-foreground">{l}</text></g>)}
              <path d={line(log.val)} fill="none" stroke="var(--viz-b)" strokeWidth={1.5} strokeDasharray="4 3" />
              <path d={line(log.train)} fill="none" stroke="var(--viz-a)" strokeWidth={2.2} />
              {log.steps.map((s, i) => i % 4 === 0 && (
                <circle key={s} cx={X(s)} cy={Y(log.train[i])} r={6} fill="transparent" data-ask-id={`loss-${s}`} data-ask={`training step ${s}: train loss ${log.train[i]}, validation loss ${log.val[i]}`} />
              ))}
              {log.snapshots.map((s, i) => (
                <g key={s.step} onClick={() => setSnap(i)} className="cursor-pointer" data-ask-id={`snap-${s.step}`} data-ask={`checkpoint at step ${s.step}; samples: ${s.samples.join(" | ")}`}>
                  <line x1={X(s.step)} x2={X(s.step)} y1={12} y2={H - 32} stroke={i === snap ? "var(--viz-pin)" : "var(--viz-grid)"} strokeWidth={i === snap ? 2 : 1} />
                  <circle cx={X(s.step)} cy={H - 24} r={6} fill={i === snap ? "var(--viz-pin)" : "var(--muted)"} stroke="var(--border)" />
                </g>
              ))}
              {[100, 500, 1000, 2000, 3000].map((s) => <text key={s} x={X(s)} y={H - 4} fontSize={10} textAnchor="middle" className="fill-muted-foreground">{s}</text>)}
              <text x={48} y={H - 4} fontSize={10} className="fill-muted-foreground">step (√ scale)</text>
            </svg>
            <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-viz-a" />training</span>
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 border-t-2 border-dashed border-viz-b" />held-out text</span>
            </div>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>What it wrote at step {cur.step}</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <ToggleGroup type="single" variant="outline" size="sm" className="w-full flex-wrap" value={String(snap)} onValueChange={(v) => v && setSnap(+v)}>
              {log.snapshots.map((s, i) => <ToggleGroupItem key={s.step} value={String(i)} className="flex-1 tabular-nums">{s.step}</ToggleGroupItem>)}
            </ToggleGroup>
            <Slider min={0} max={log.snapshots.length - 1} step={1} value={[snap]} onValueChange={([v]) => setSnap(v)} />
            <div className="space-y-2">
              {cur.samples.map((s, i) => (
                <div key={i} className="rounded-lg border bg-muted/40 p-3 font-mono text-[13px] leading-relaxed whitespace-pre-wrap"
                  data-ask-id={`snap-${cur.step}-${i}`} data-ask={`at step ${cur.step}, given "${log.prompts[i]}" the model wrote "${s.slice(log.prompts[i].length)}"`}>
                  <span className="text-muted-foreground">{log.prompts[i]}</span>{s.slice(log.prompts[i].length)}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader><CardTitle>The training text</CardTitle></CardHeader>
        <CardContent><ScrollArea className="h-48 rounded-md border bg-muted/30"><pre className="p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground" data-ask-id="corpus" data-ask="a sample of the synthetic training corpus">{corpus.slice(0, 1400)}</pre></ScrollArea></CardContent>
      </Card>
      <Button asChild className="w-full sm:w-auto"><a href="#/train"><Cpu />Train your own</a></Button>
    </div>
  );
}
