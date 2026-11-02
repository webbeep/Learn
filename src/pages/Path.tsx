import { useEffect, useState } from "react";
import { ChevronDown, ClipboardCopy, Download, Play, RefreshCw, Trash2, Upload, Wrench } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { addWish, ask, learner, setScreen, useStore } from "../lib/store";
import { refreshProfile } from "../lib/api";
import { Markdown } from "../ask/Markdown";
import { LevelSelect } from "@/components/app/level";

export function Path() {
  const l = useStore(learner, (s) => s);
  const [busy, setBusy] = useState(false);
  const [wish, setWish] = useState("");
  useEffect(() => setScreen("My path", "", `Learner has asked ${l.history.length} questions. Profile: ${l.profile?.summary ?? "none yet"}. Build list: ${l.wishes.map((w) => w.text).join("; ")}`), [l]);
  const p = l.profile;
  const exportPrompt = [
    "Here is my learner profile and feature wishlist from my Top-Down AI app. Please evolve the app (webbeep/Learn) to build the top items, keeping the same style:",
    "", `Goal: ${l.goal}`, p ? `Profile: ${p.summary}\nStyle: ${p.style}\nConfusions: ${p.confusions.join("; ")}\nInterests: ${p.interests.join("; ")}` : "",
    "", "Wishlist:", ...l.wishes.filter((w) => !w.done).map((w) => `- ${w.text}  (from: ${w.source})`),
    "", "Recent questions:", ...l.history.slice(-12).map((h) => `- [${h.where}] ${h.q}`),
  ].join("\n");
  const update = async () => { setBusy(true); try { await refreshProfile(); toast.success("Profile updated"); } catch (e) { toast.error(String((e as Error).message)); } setBusy(false); };

  return (
    <div className="space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">My path</h1>
        <p className="mt-1 max-w-2xl text-muted-foreground">The tutor keeps a picture of how you think, built from your questions. It uses this to tailor every answer, and turns what you wish you could see into a build list for the next version of this app.</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>How the tutor sees you</CardTitle>
            <CardDescription>{p ? `Updated ${new Date(p.updated).toLocaleString()}` : "Builds itself after a few questions"}</CardDescription>
            <CardAction><Button variant="outline" size="sm" disabled={busy || !l.history.length} onClick={update}>{busy ? <Spinner /> : <RefreshCw />}Update</Button></CardAction>
          </CardHeader>
          <CardContent className="space-y-5">
            {p ? (<>
              <p className="text-pretty">{p.summary}</p>
              <p className="text-sm text-muted-foreground"><span className="font-medium text-foreground">Explain to me like: </span>{p.style}</p>
              {([["You've got", p.knows, "border-emerald-500/40 text-emerald-600 dark:text-emerald-400"], ["Still fuzzy", p.confusions, "border-amber-500/40 text-amber-600 dark:text-amber-400"], ["You keep coming back to", p.interests, "border-sky-500/40 text-sky-600 dark:text-sky-400"]] as const).map(([t, arr, c]) => arr.length ? (
                <div key={t} className="space-y-2"><div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{t}</div>
                  <div className="flex flex-wrap gap-1.5">{arr.map((x) => <Badge key={x} variant="outline" className={`h-auto py-0.5 whitespace-normal ${c}`}>{x}</Badge>)}</div></div>
              ) : null)}
            </>) : (
              <Empty className="border border-dashed"><EmptyHeader><EmptyTitle>No profile yet</EmptyTitle><EmptyDescription>Ask a few questions anywhere in the app. Click any visual, or press V and talk.</EmptyDescription></EmptyHeader></Empty>
            )}
            <div className="flex items-center justify-between gap-3"><Label>My experience</Label><LevelSelect /></div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2"><Label>Explanation depth</Label>
                <ToggleGroup type="single" variant="outline" size="sm" value={l.depth} onValueChange={(v) => v && learner.set({ depth: v as typeof l.depth })} className="w-full">
                  {["intuition", "balanced", "math"].map((d) => <ToggleGroupItem key={d} value={d} className="flex-1 capitalize">{d}</ToggleGroupItem>)}
                </ToggleGroup></div>
              <div className="space-y-2"><Label>Answer length</Label>
                <ToggleGroup type="single" variant="outline" size="sm" value={l.length} onValueChange={(v) => v && learner.set({ length: v as typeof l.length })} className="w-full">
                  {["short", "medium", "long"].map((d) => <ToggleGroupItem key={d} value={d} className="flex-1 capitalize">{d}</ToggleGroupItem>)}
                </ToggleGroup></div>
            </div>
            <div className="space-y-2"><Label htmlFor="goal">My goal</Label><Textarea id="goal" value={l.goal} rows={3} onChange={(e) => learner.set({ goal: e.target.value })} /></div>
          </CardContent>
        </Card>

        <div className="space-y-4 lg:col-span-2">
          <Card>
            <CardHeader><CardTitle>Next for you</CardTitle><CardDescription>Picked from your questions</CardDescription></CardHeader>
            <CardContent>
              {p?.next?.length ? <ItemGroup className="gap-2">{p.next.map((n, i) => (
                <Item key={i} variant="muted" size="sm">
                  <ItemContent><ItemTitle>{n.title}</ItemTitle><ItemDescription>{n.why}</ItemDescription></ItemContent>
                  {n.try && <ItemActions><Button size="icon-sm" variant="outline" title={`Try "${n.try}"`} onClick={() => { ask.set({ tryPrompt: n.try! }); location.hash = "#/xray"; }}><Play /></Button></ItemActions>}
                </Item>
              ))}</ItemGroup> : <p className="text-sm text-muted-foreground">Appears after a few questions.</p>}
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Wrench className="size-4" />Build list</CardTitle><CardDescription>How this app evolves. These come from your questions, the tutor, and the ⭐ buttons.</CardDescription></CardHeader>
            <CardContent className="space-y-1">
              {l.wishes.length === 0 && <p className="text-sm text-muted-foreground">Empty for now.</p>}
              {l.wishes.map((w) => (
                <div key={w.id} className="group flex items-start gap-3 rounded-md px-1 py-2 hover:bg-muted/50">
                  <Checkbox className="mt-0.5" checked={!!w.done} onCheckedChange={() => learner.set({ wishes: l.wishes.map((x) => x.id === w.id ? { ...x, done: !x.done } : x) })} />
                  <div className={`min-w-0 flex-1 text-sm ${w.done ? "text-muted-foreground line-through" : ""}`}>{w.text}<div className="text-xs text-muted-foreground">from {w.source}</div></div>
                  <Button variant="ghost" size="icon-xs" className="opacity-0 group-hover:opacity-100" onClick={() => learner.set({ wishes: l.wishes.filter((x) => x.id !== w.id) })}><Trash2 /></Button>
                </div>
              ))}
              <form className="flex gap-2 pt-2" onSubmit={(e) => { e.preventDefault(); addWish(wish, "you"); setWish(""); }}>
                <Input value={wish} onChange={(e) => setWish(e.target.value)} placeholder="I wish I could see…" />
                <Button variant="outline" type="submit">Add</Button>
              </form>
            </CardContent>
            <CardFooter><Button className="w-full" onClick={() => { navigator.clipboard.writeText(exportPrompt); toast.success("Copied", { description: "Paste it into a Claude Code session on this repo." }); }}><ClipboardCopy />Copy as a request for Claude Code</Button></CardFooter>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>Every question you've asked</CardTitle><CardDescription>{l.history.length} so far</CardDescription>
          <CardAction className="flex gap-1.5">
            <Button variant="outline" size="sm" onClick={() => { const b = new Blob([JSON.stringify(l, null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "topdown-ai-learner.json"; a.click(); }}><Download />Export</Button>
            <Button variant="outline" size="sm" asChild><label className="cursor-pointer"><Upload />Import<input type="file" accept="application/json" hidden onChange={async (e) => { const f = e.target.files?.[0]; if (f) { learner.set(JSON.parse(await f.text())); toast.success("Imported"); } }} /></label></Button>
            <Button variant="ghost" size="sm" onClick={() => { if (confirm("Erase your history, profile and build list?")) learner.set({ history: [], profile: null, wishes: [] }); }}>Reset</Button>
          </CardAction>
        </CardHeader>
        <CardContent className="divide-y">
          {l.history.length === 0 && <p className="text-sm text-muted-foreground">Nothing yet.</p>}
          {[...l.history].reverse().map((h) => (
            <Collapsible key={h.id} className="group/qa py-1">
              <CollapsibleTrigger className="flex w-full items-center gap-3 rounded-md px-1 py-2 text-left text-sm hover:bg-muted/50">
                <span className="flex-1 font-medium">{h.q}</span>
                <Badge variant="outline" className="hidden sm:inline-flex">{h.where}</Badge>
                <span className="text-xs text-muted-foreground">{new Date(h.ts).toLocaleDateString()}</span>
                <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[state=open]/qa:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className="mb-2 ml-1 border-l-2 pl-4 text-sm">
                {h.selection.length > 0 && <div className="mb-2 text-xs text-muted-foreground">about: {h.selection.join(" · ")}</div>}
                <Markdown text={h.a} />
              </CollapsibleContent>
            </Collapsible>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
