import { useEffect, useState } from "react";
import { ChevronDown, ClipboardCopy, Download, RefreshCw, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { addWish, learner, setScreen, useStore } from "../lib/store";
import { refreshProfile } from "../lib/api";
import { Markdown } from "../ask/Markdown";
import { LevelSelect } from "@/components/app/level";

function Section({ title, children, action, open = true }: { title: string; children: React.ReactNode; action?: React.ReactNode; open?: boolean }) {
  return (
    <Collapsible defaultOpen={open} className="group/sec border-b pb-4">
      <div className="flex items-center justify-between gap-2 py-2">
        <CollapsibleTrigger className="flex items-center gap-2 text-lg font-semibold tracking-tight">
          {title}<ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[state=open]/sec:rotate-180" />
        </CollapsibleTrigger>
        {action}
      </div>
      <CollapsibleContent className="space-y-3 pt-1">{children}</CollapsibleContent>
    </Collapsible>
  );
}

export function Path() {
  const l = useStore(learner, (s) => s);
  const [busy, setBusy] = useState(false);
  const [wish, setWish] = useState("");
  useEffect(() => setScreen("My notes", "", `Learner has asked ${l.history.length} questions. Profile: ${l.profile?.summary ?? "none yet"}. Wish list: ${l.wishes.map((w) => w.text).join("; ")}`), [l]);
  const p = l.profile;
  const exportPrompt = [
    "Here is my learner profile and feature wishlist from my Top-Down AI app. Please evolve the app (webbeep/Learn) to build the top items, keeping the same style:",
    "", `Goal: ${l.goal}`, p ? `Profile: ${p.summary}\nStyle: ${p.style}\nConfusions: ${p.confusions.join("; ")}\nInterests: ${p.interests.join("; ")}` : "",
    "", "Wishlist:", ...l.wishes.filter((w) => !w.done).map((w) => `- ${w.text}  (from: ${w.source})`),
    "", "Recent questions:", ...l.history.slice(-12).map((h) => `- [${h.where}] ${h.q}`),
  ].join("\n");
  const update = async () => { setBusy(true); try { await refreshProfile(); toast.success("Updated"); } catch (e) { toast.error(String((e as Error).message)); } setBusy(false); };

  return (
    <div className="mx-auto max-w-3xl space-y-4 pb-10">
      <h1 className="text-3xl font-semibold tracking-tight">My notes</h1>

      <Section title="About you" action={<Button variant="ghost" size="sm" disabled={busy || !l.history.length} onClick={update}>{busy ? <Spinner /> : <RefreshCw />}Update</Button>}>
        {p ? (<>
          <p className="text-pretty">{p.summary}</p>
          {p.confusions.length > 0 && <p className="text-sm text-muted-foreground"><span className="font-medium text-foreground">Still fuzzy: </span>{p.confusions.join(" · ")}</p>}
          {p.next.length > 0 && <p className="text-sm text-muted-foreground"><span className="font-medium text-foreground">Try next: </span>{p.next.map((n) => n.title).join(" · ")}</p>}
        </>) : <p className="text-muted-foreground">Ask a few questions and this fills in.</p>}
      </Section>

      <Section title="Questions" open={l.history.length > 0}>
        {l.history.length === 0 && <p className="text-muted-foreground">None yet.</p>}
        <div className="divide-y">
          {[...l.history].reverse().map((h) => (
            <Collapsible key={h.id} className="group/qa">
              <CollapsibleTrigger className="flex w-full items-center gap-3 py-2 text-left text-sm hover:text-foreground">
                <span className="flex-1 font-medium">{h.q}</span>
                <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[state=open]/qa:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className="mb-2 border-l-2 pl-4 text-sm"><Markdown text={h.a} /></CollapsibleContent>
            </Collapsible>
          ))}
        </div>
      </Section>

      <Section title="Wish list" open={l.wishes.length > 0}
        action={l.wishes.length > 0 && <Button variant="ghost" size="sm" onClick={() => { navigator.clipboard.writeText(exportPrompt); toast.success("Copied"); }}><ClipboardCopy />Copy</Button>}>
        {l.wishes.map((w) => (
          <div key={w.id} className="group flex items-start gap-3">
            <Checkbox className="mt-0.5" checked={!!w.done} onCheckedChange={() => learner.set({ wishes: l.wishes.map((x) => x.id === w.id ? { ...x, done: !x.done } : x) })} />
            <span className={`flex-1 text-sm ${w.done ? "text-muted-foreground line-through" : ""}`}>{w.text}</span>
            <Button variant="ghost" size="icon-xs" className="opacity-0 group-hover:opacity-100" onClick={() => learner.set({ wishes: l.wishes.filter((x) => x.id !== w.id) })}><Trash2 /></Button>
          </div>
        ))}
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); addWish(wish, "you"); setWish(""); }}>
          <Input value={wish} onChange={(e) => setWish(e.target.value)} placeholder="I wish I could see…" />
          <Button variant="outline" type="submit">Add</Button>
        </form>
      </Section>

      <Section title="Settings" open={false}>
        <div className="flex items-center justify-between gap-3"><Label>Experience</Label><LevelSelect /></div>
        <div className="flex items-center justify-between gap-3"><Label>Answers</Label>
          <ToggleGroup type="single" variant="outline" size="sm" value={l.length} onValueChange={(v) => v && learner.set({ length: v as typeof l.length })}>
            {["short", "medium", "long"].map((d) => <ToggleGroupItem key={d} value={d} className="capitalize">{d}</ToggleGroupItem>)}
          </ToggleGroup>
        </div>
        <div className="space-y-2"><Label htmlFor="goal">Goal</Label><Textarea id="goal" value={l.goal} rows={2} onChange={(e) => learner.set({ goal: e.target.value })} /></div>
        <div className="flex flex-wrap gap-2 pt-2">
          <Button variant="outline" size="sm" onClick={() => { const b = new Blob([JSON.stringify(l, null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "topdown-ai-notes.json"; a.click(); }}><Download />Export</Button>
          <Button variant="outline" size="sm" asChild><label className="cursor-pointer"><Upload />Import<input type="file" accept="application/json" hidden onChange={async (e) => { const f = e.target.files?.[0]; if (f) { learner.set(JSON.parse(await f.text())); toast.success("Imported"); } }} /></label></Button>
          <Button variant="ghost" size="sm" onClick={() => { if (confirm("Erase your history, profile and wish list?")) learner.set({ history: [], profile: null, wishes: [] }); }}>Reset</Button>
        </div>
      </Section>
    </div>
  );
}
