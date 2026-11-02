// Idea board + experiment-card builder (saved to localStorage) + PyTorch script export.
import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ClipboardCopy, Download, ExternalLink, FileCode, Pencil, Play, Plus, Save, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { uid } from "../lib/store";
import { IDEAS, type Idea } from "./ideas";
import { CHANGES, browserLink, browserSupports, makeScript, scriptName, type ChangeId, type ExperimentCard } from "./pyscript";

const KEY = "topdown-ai-experiments-v1";
const METRICS = ["validation loss (lower is better)", "training loss", "speed (ms per step)", "color-copy probe: p(\" red\")", "sample quality (read the samples)"];

function loadCards(): ExperimentCard[] {
  try { const v = JSON.parse(localStorage.getItem(KEY) || "[]"); return Array.isArray(v) ? v : []; } catch { return []; }
}
function saveCards(c: ExperimentCard[]) { try { localStorage.setItem(KEY, JSON.stringify(c)); } catch { /* private mode */ } }
const blank = (): ExperimentCard => ({ id: uid(), created: Date.now(), updated: Date.now(), title: "", hypothesis: "", change: "no-pos", custom: "", metric: METRICS[0], seeds: 3, steps: 1500, notes: "", results: [] });

function download(card: ExperimentCard) {
  const blob = new Blob([makeScript(card)], { type: "text/x-python" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = scriptName(card); a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast.success(`Downloaded ${scriptName(card)}`, { description: "Run: python3 train.py --data tiny_world.txt" });
}

export function IdeaBoard({ onUse }: { onUse: (idea: Idea) => void }) {
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold tracking-tight">Ideas</h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {IDEAS.map((it) => (
          <Card key={it.id} size="sm" className="gap-2" data-ask-id={`idea-${it.id}`} data-ask={`frontier idea "${it.title}": ${it.idea} Problem: ${it.problem} Tiny version: ${it.tiny} Papers: ${it.refs.map((r) => r.label).join(", ")}.`}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><it.icon className="size-4 shrink-0 text-muted-foreground" />{it.title}</CardTitle>
            </CardHeader>
            <CardContent className="flex-1 text-sm text-pretty text-muted-foreground">{it.idea}</CardContent>
            <CardFooter className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="h-8" onClick={() => onUse(it)}><Plus />Try it</Button>
              <span className="ml-auto flex gap-3">
                {it.refs.map((r, i) => <a key={r.url} href={r.url} target="_blank" rel="noreferrer" title={r.label} className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground hover:underline"><ExternalLink className="size-3 shrink-0" />Paper{it.refs.length > 1 ? ` ${i + 1}` : ""}</a>)}
              </span>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function CardBuilder({ seed, onSummary }: { seed: { idea: Idea; n: number } | null; onSummary: (s: string) => void }) {
  const [cards, setCards] = useState<ExperimentCard[]>(loadCards);
  const [draft, setDraft] = useState<ExperimentCard>(blank);
  const [editing, setEditing] = useState(false);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => saveCards(cards), [cards]);
  useEffect(() => {
    if (!seed) return;
    const it = seed.idea;
    setDraft({ ...blank(), title: it.title, hypothesis: it.hypothesis, change: it.change, custom: it.custom ?? "" });
    setEditing(false); setOpen(true);
    setTimeout(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  }, [seed]);
  const set = (p: Partial<ExperimentCard>) => setDraft((d) => ({ ...d, ...p }));
  const ch = CHANGES.find((c) => c.id === draft.change)!;
  const valid = draft.hypothesis.trim().length > 0 && (draft.change !== "custom" || draft.custom.trim().length > 0);
  const save = () => {
    const card = { ...draft, title: draft.title.trim() || ch.label, updated: Date.now() };
    setCards((cs) => (cs.some((c) => c.id === card.id) ? cs.map((c) => (c.id === card.id ? card : c)) : [card, ...cs]));
    toast.success(editing ? "Card updated" : "Card saved");
    setDraft(blank()); setEditing(false);
  };
  useEffect(() => {
    onSummary(`Experiment card builder. Draft: title "${draft.title}", hypothesis "${draft.hypothesis}", change "${draft.change === "custom" ? draft.custom : ch.label}", metric "${draft.metric}", ${draft.seeds} seeds, ${draft.steps} steps. ` +
      `Saved cards (${cards.length}): ${cards.map((c) => `"${c.title}" [${c.change}]${c.results.length ? ` results ${c.results.map((r) => `seed ${r.seed}: baseline ${r.baseline} vs variant ${r.variant}`).join("; ")}` : ""}`).join(" | ") || "none"}.`);
  }, [draft, cards, ch, onSummary]);

  return (
    <Collapsible ref={ref} open={open} onOpenChange={setOpen} className="scroll-mt-20 space-y-4">
      <CollapsibleTrigger className="flex w-full items-center gap-2 rounded-xl bg-card px-4 py-3.5 text-left text-base font-medium ring-1 ring-foreground/10 [&[data-state=open]>svg]:rotate-180">
        Plan your own experiment{cards.length > 0 && <Badge variant="secondary">{cards.length} saved</Badge>}
        <ChevronDown className="ml-auto size-4 text-muted-foreground transition-transform" />
      </CollapsibleTrigger>
      <CollapsibleContent className="grid grid-cols-1 gap-4 lg:grid-cols-5 [&>*]:min-w-0">
        <Card className="@container lg:col-span-2">
          <CardHeader>
            <CardTitle>{editing ? "Edit card" : "New card"}</CardTitle>
            {(editing || draft.hypothesis) && <CardAction><Button variant="ghost" size="sm" onClick={() => { setDraft(blank()); setEditing(false); }}><X />Clear</Button></CardAction>}
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1.5"><Label htmlFor="ec-title">Name</Label>
              <Input id="ec-title" value={draft.title} placeholder={ch.label} onChange={(e) => set({ title: e.target.value })} className="h-9" /></div>
            <div className="space-y-1.5"><Label htmlFor="ec-hyp">Hypothesis</Label>
              <Textarea id="ec-hyp" rows={3} value={draft.hypothesis} placeholder="If I …, then val loss will … because …" onChange={(e) => set({ hypothesis: e.target.value })} /></div>
            <div className="space-y-1.5"><Label>The one change</Label>
              <Select value={draft.change} onValueChange={(v) => set({ change: v as ChangeId })}>
                <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{CHANGES.map((c) => <SelectItem key={c.id} value={c.id}>{c.label}</SelectItem>)}</SelectContent>
              </Select>
              {draft.change === "custom" && <Textarea rows={2} value={draft.custom} placeholder="Describe the change" onChange={(e) => set({ custom: e.target.value })} />}
            </div>
            <div className="space-y-1.5"><Label>Metric</Label>
              <Select value={draft.metric} onValueChange={(v) => set({ metric: v })}>
                <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{METRICS.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent>
              </Select></div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5"><Label>Seeds</Label>
                <ToggleGroup type="single" variant="outline" value={String(draft.seeds)} onValueChange={(v) => v && set({ seeds: +v })} className="w-full">
                  {[1, 3, 5].map((n) => <ToggleGroupItem key={n} value={String(n)} className="h-9 flex-1">{n}</ToggleGroupItem>)}
                </ToggleGroup></div>
              <div className="space-y-1.5"><Label>Steps</Label>
                <ToggleGroup type="single" variant="outline" value={String(draft.steps)} onValueChange={(v) => v && set({ steps: +v })} className="w-full">
                  {[500, 1500, 3000].map((n) => <ToggleGroupItem key={n} value={String(n)} className="h-9 flex-1 px-1 tabular-nums">{n >= 1000 ? `${n / 1000}k` : n}</ToggleGroupItem>)}
                </ToggleGroup></div>
            </div>
          </CardContent>
          <CardFooter className="flex-col items-stretch gap-2">
            <Button className="h-9" disabled={!valid} onClick={save}><Save />{editing ? "Update card" : "Save card"}</Button>
            <div className="grid grid-cols-2 gap-2">
              <Button variant="outline" className="h-9" asChild><a href={browserLink(draft)} title={browserSupports(draft.change) ? "Run in the browser trainer" : "Opens the baseline"}><Play />{browserSupports(draft.change) ? "Run in browser" : "Baseline in browser"}</a></Button>
              <Button variant="outline" className="h-9" onClick={() => download({ ...draft, title: draft.title || ch.label })}><Download />Export .py</Button>
            </div>
          </CardFooter>
        </Card>
        <div className="space-y-3 lg:col-span-3">
          {cards.length === 0 ? (
            <Empty className="h-full min-h-32 border border-dashed">
              <EmptyHeader><EmptyTitle>No cards yet</EmptyTitle><EmptyDescription>Fill in the form or pick an idea.</EmptyDescription></EmptyHeader>
            </Empty>
          ) : cards.map((c) => (
            <SavedCard key={c.id} card={c} onChange={(nc) => setCards((cs) => cs.map((x) => (x.id === nc.id ? nc : x)))}
              onEdit={() => { setDraft(c); setEditing(true); setOpen(true); ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }}
              onDelete={() => { if (confirm(`Delete "${c.title}"?`)) setCards((cs) => cs.filter((x) => x.id !== c.id)); }} />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function SavedCard({ card, onChange, onEdit, onDelete }: { card: ExperimentCard; onChange: (c: ExperimentCard) => void; onEdit: () => void; onDelete: () => void }) {
  const ch = CHANGES.find((x) => x.id === card.change)!;
  const script = useMemo(() => makeScript(card), [card]);
  const res = card.results.filter((r) => r.baseline !== "" && r.variant !== "" && Number.isFinite(+r.baseline) && Number.isFinite(+r.variant));
  const diffs = res.map((r) => +r.variant - +r.baseline);
  const md = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : NaN;
  const sd = diffs.length > 1 ? Math.sqrt(diffs.reduce((s, d) => s + (d - md) ** 2, 0) / (diffs.length - 1)) : NaN;
  const setRow = (i: number, p: Partial<ExperimentCard["results"][number]>) => onChange({ ...card, updated: Date.now(), results: card.results.map((r, j) => (j === i ? { ...r, ...p } : r)) });
  return (
    <Card className="gap-3 @container" data-ask-id={`card-${card.id}`} data-ask={`experiment card "${card.title}": hypothesis "${card.hypothesis}", change ${card.change === "custom" ? card.custom : ch.label}, metric ${card.metric}, ${card.seeds} seeds`}>
      <CardHeader>
        <CardTitle className="leading-snug">{card.title}</CardTitle>
        <CardAction className="flex gap-0.5">
          <Button variant="ghost" size="icon-sm" onClick={onEdit} aria-label="Edit"><Pencil /></Button>
          <Button variant="ghost" size="icon-sm" onClick={onDelete} aria-label="Delete"><Trash2 /></Button>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-pretty"><span className="text-muted-foreground">Hypothesis: </span>{card.hypothesis}</p>
        {card.change === "custom" && <p className="text-pretty"><span className="text-muted-foreground">Change: </span>{card.custom}</p>}
        <div className="flex flex-wrap gap-2">
          <Button size="sm" className="h-9" asChild><a href={browserLink(card)}><Play />{browserSupports(card.change) ? "Run in browser" : "Baseline in browser"}</a></Button>
          <Button size="sm" variant="outline" className="h-9" onClick={() => download(card)}><Download />Export .py</Button>
        </div>
        <Collapsible className="group/res rounded-lg border">
          <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium">
            Log results {res.length > 0 && <Badge variant="outline" className="font-mono">Δ {md >= 0 ? "+" : ""}{md.toFixed(3)}{Number.isFinite(sd) && ` ± ${sd.toFixed(3)}`}</Badge>}
            <ChevronDown className="ml-auto size-4 text-muted-foreground transition-transform group-data-[state=open]/res:rotate-180" />
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 border-t p-3">
            <div className="grid grid-cols-[3rem_1fr_1fr_2rem] items-center gap-2 text-xs text-muted-foreground"><span>seed</span><span>baseline</span><span>with change</span><span /></div>
            {card.results.map((r, i) => (
              <div key={i} className="grid grid-cols-[3rem_1fr_1fr_2rem] items-center gap-2">
                <span className="font-mono text-sm">{r.seed}</span>
                <Input inputMode="decimal" className="h-9 font-mono" value={r.baseline} placeholder="0.000" onChange={(e) => setRow(i, { baseline: e.target.value })} aria-label={`baseline seed ${r.seed}`} />
                <Input inputMode="decimal" className="h-9 font-mono" value={r.variant} placeholder="0.000" onChange={(e) => setRow(i, { variant: e.target.value })} aria-label={`variant seed ${r.seed}`} />
                <Button variant="ghost" size="icon-sm" aria-label="Remove row" onClick={() => onChange({ ...card, results: card.results.filter((_, j) => j !== i) })}><X /></Button>
              </div>
            ))}
            <Button variant="outline" size="sm" className="h-9" onClick={() => onChange({ ...card, results: [...card.results, { seed: card.results.length, baseline: "", variant: "", }] })}><Plus />Add a seed</Button>
            {res.length > 0 && <Verdict md={md} sd={sd} n={res.length} metric={card.metric} />}
          </CollapsibleContent>
        </Collapsible>
        <Collapsible className="group/code rounded-lg border">
          <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium">
            <FileCode className="size-4" />{scriptName(card)} <span className="text-xs font-normal text-muted-foreground">{script.split("\n").length} lines</span>
            <ChevronDown className="ml-auto size-4 text-muted-foreground transition-transform group-data-[state=open]/code:rotate-180" />
          </CollapsibleTrigger>
          <CollapsibleContent className="border-t">
            <div className="flex justify-end gap-1 p-1.5">
              <Button variant="ghost" size="sm" onClick={() => { navigator.clipboard.writeText(script); toast.success("Script copied"); }}><ClipboardCopy />Copy</Button>
            </div>
            <pre className="max-h-96 overflow-auto px-3 pb-3 font-mono text-[11.5px] leading-relaxed">
              {script.split("\n").map((l, i) => <div key={i} className={cn("min-w-max", l.includes("<<< EXPERIMENT") && "-mx-3 bg-[color-mix(in_oklch,var(--viz-pin)_18%,transparent)] px-3")}>{l || " "}</div>)}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}

function Verdict({ md, sd, n, metric }: { md: number; sd: number; n: number; metric: string }) {
  const lowerBetter = !/probe|quality/.test(metric);
  const better = lowerBetter ? md < 0 : md > 0;
  if (n < 3 || !Number.isFinite(sd)) return <p className="text-xs text-muted-foreground">Mean change {md >= 0 ? "+" : ""}{md.toFixed(3)} over {n} seed{n > 1 ? "s" : ""}. Log 3+ seeds.</p>;
  const z = Math.abs(md) / (sd / Math.sqrt(n) || 1e-9);
  const clear = z > 3;
  return (
    <p className={cn("rounded-md p-2 text-xs", clear ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" : "bg-muted text-muted-foreground")}>
      Mean change {md >= 0 ? "+" : ""}{md.toFixed(3)} ± {(sd / Math.sqrt(n)).toFixed(3)} over {n} seeds. {clear ? `Clear effect: ${better ? "better" : "worse"}.` : "Within the noise so far."}
    </p>
  );
}
