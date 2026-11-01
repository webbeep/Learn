import { useEffect, useRef, useState } from "react";
import { ArrowUp, Eye, Mic, RotateCcw, Settings2, Sparkles, Square, SquareDashedMousePointer, Volume2, X, Keyboard as KeyboardIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Kbd } from "@/components/ui/kbd";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Item, ItemContent, ItemDescription, ItemTitle } from "@/components/ui/item";
import { cn } from "@/lib/utils";
import { ask, learner, useStore } from "../lib/store";
import { compressSelection, parseAnswer, viewportNote, visibleElements } from "../lib/api";
import { convo, currentPointer, listenAndAsk, newConversation, sendQuestion, stopAll } from "../lib/conversation";
import { speakText, stopListening, voice } from "../lib/voice";
import { Markdown } from "./Markdown";
import { suggestionsFor } from "./suggest";
import { VoiceSettings } from "./VoiceSettings";

export function TutorPanel({ className }: { className?: string }) {
  const scene = useStore(ask, (s) => `${s.page}${s.scene ? " › " + s.scene : ""}`);
  const [tab, setTab] = useState("chat");
  const thread = useStore(convo, (s) => s.thread);
  useEffect(() => {
    const f = () => setTab("chat");
    window.addEventListener("focus-ask", f); return () => window.removeEventListener("focus-ask", f);
  }, []);
  return (
    <div className={cn("flex h-full min-h-0 flex-col bg-sidebar text-sidebar-foreground", className)}>
      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col gap-0">
        <div className="flex items-center gap-2 border-b px-3 py-2.5">
          <div className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground"><Sparkles className="size-3.5" /></div>
          <div className="min-w-0 flex-1 leading-tight">
            <div className="text-sm font-semibold">Tutor</div>
            <div className="truncate text-xs text-muted-foreground">{scene || "Looking at your screen"}</div>
          </div>
          {thread.length > 0 && <Button variant="ghost" size="icon-sm" onClick={newConversation} title="New conversation"><RotateCcw /></Button>}
          <Button variant="ghost" size="icon-sm" onClick={() => ask.set({ open: false })} aria-label="Close"><X /></Button>
        </div>
        <div className="border-b px-3 py-2">
          <TabsList className="w-full">
            <TabsTrigger value="chat"><Sparkles />Chat</TabsTrigger>
            <TabsTrigger value="sees"><Eye />What it sees</TabsTrigger>
            <TabsTrigger value="voice"><Settings2 />Voice</TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="chat" className="flex min-h-0 flex-1 flex-col"><Chat /></TabsContent>
        <TabsContent value="sees" className="min-h-0 flex-1"><ScrollArea className="h-full"><WhatItSees /></ScrollArea></TabsContent>
        <TabsContent value="voice" className="min-h-0 flex-1"><ScrollArea className="h-full"><VoiceSettings /></ScrollArea></TabsContent>
      </Tabs>
    </div>
  );
}

function Chat() {
  const selection = useStore(ask, (s) => s.selection);
  const region = useStore(ask, (s) => s.region);
  const regionMode = useStore(ask, (s) => s.regionMode);
  const draft = useStore(ask, (s) => s.draft);
  const thread = useStore(convo, (s) => s.thread);
  const busy = useStore(convo, (s) => s.busy);
  const listening = useStore(voice, (s) => s.listening || s.transcribing);
  const speaking = useStore(voice, (s) => s.speaking);
  const interim = useStore(voice, (s) => s.interim);
  const [input, setInput] = useState("");
  const end = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (draft) { setInput(draft); ask.set({ draft: "" }); setTimeout(() => inputRef.current?.focus(), 0); } }, [draft]);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [thread, interim]);
  useEffect(() => {
    const f = () => setTimeout(() => inputRef.current?.focus(), 40);
    window.addEventListener("focus-ask", f); return () => window.removeEventListener("focus-ask", f);
  }, []);

  const send = (q: string) => { if (!q.trim()) return; setInput(""); sendQuestion(q, "text"); };
  const last = thread[thread.length - 1];
  const chips = last?.followups?.length ? last.followups : suggestionsFor(selection.map((s) => s.id));

  return (<>
    <div className="space-y-2 border-b px-3 py-2.5">
      {selection.length ? (
        <div className="flex flex-wrap gap-1.5">
          {selection.slice(0, 5).map((s) => (
            <Badge key={s.id} variant="secondary" className="h-auto max-w-full gap-1 py-1 text-left font-normal whitespace-normal"
              onMouseEnter={() => ask.set({ highlight: s.id })} onMouseLeave={() => ask.set({ highlight: null })}>
              <span className="line-clamp-2">{s.label}</span>
              <button className="opacity-60 hover:opacity-100" onClick={() => ask.set({ selection: selection.filter((x) => x.id !== s.id) })}><X className="size-3" /></button>
            </Badge>
          ))}
          {selection.length > 5 && <Badge variant="outline">+{selection.length - 5}</Badge>}
        </div>
      ) : (
        <p className="text-xs leading-relaxed text-muted-foreground">Click anything, highlight text, or drag a box. With nothing selected, I use whatever your mouse is over and everything visible in your window.</p>
      )}
      {region && <p className="text-xs text-muted-foreground">▭ {region}</p>}
      <div className="flex gap-1.5">
        <Button variant={regionMode ? "secondary" : "outline"} size="xs" onClick={() => ask.set({ regionMode: !regionMode })}>
          <SquareDashedMousePointer />{regionMode ? "Drag over the visual…" : "Highlight region"}<Kbd>H</Kbd>
        </Button>
        {selection.length > 0 && <Button variant="ghost" size="xs" onClick={() => ask.set({ selection: [], region: "" })}>Clear</Button>}
      </div>
    </div>

    <ScrollArea className="min-h-0 flex-1">
      <div className="space-y-4 p-3">
        {thread.length === 0 && <Intro />}
        {thread.map((t, i) => t.role === "user" ? (
          <div key={i} className="ml-8 flex flex-col items-end gap-1">
            {t.selection?.length ? <div className="text-right text-[11px] text-muted-foreground line-clamp-2">about: {t.selection.slice(0, 2).map((s) => s.label.slice(0, 80)).join(" · ")}</div> : null}
            <div className="rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">{t.via === "voice" && <Mic className="mr-1 inline size-3.5" />}{t.content}</div>
          </div>
        ) : (
          <div key={i} className={cn("text-sm", t.error && "text-destructive")}>
            {t.content ? (<>
              <Markdown text={parseAnswer(t.content).body} />
              {!t.error && !(busy && i === thread.length - 1) && (
                <Button variant="ghost" size="xs" className="-ml-2 text-muted-foreground" onClick={() => speakText(parseAnswer(t.content).body)}><Volume2 />Read aloud</Button>
              )}
            </>) : <div className="flex gap-1 py-2">{[0, 1, 2].map((d) => <span key={d} className="size-1.5 animate-bounce rounded-full bg-muted-foreground" style={{ animationDelay: `${d * 120}ms` }} />)}</div>}
          </div>
        ))}
        {listening && <div className="ml-8 rounded-2xl rounded-br-sm border border-dashed border-destructive/60 px-3 py-2 text-right text-sm text-muted-foreground"><Mic className="mr-1 inline size-3.5 text-destructive" />{interim || "Listening…"}</div>}
        <div ref={end} />
      </div>
    </ScrollArea>

    <div className="space-y-2 border-t p-3">
      <div className="flex flex-wrap gap-1.5">
        {chips.slice(0, 3).map((c) => (
          <Button key={c} variant="outline" size="xs" className="h-auto max-w-full rounded-full py-1 text-left whitespace-normal" onClick={() => sendQuestion(c, "text")}>{c}</Button>
        ))}
      </div>
      <form className="rounded-xl border bg-background shadow-xs focus-within:ring-3 focus-within:ring-ring/30" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <Textarea ref={inputRef} value={input} rows={2} placeholder="Ask anything, or press V and talk…"
          className="min-h-0 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } if (e.key === "Escape") (e.target as HTMLElement).blur(); }} />
        <div className="flex items-center gap-1.5 px-2 pb-2">
          <Button type="button" variant={listening ? "destructive" : "ghost"} size="icon-sm" className={cn(listening && "animate-pulse")} title="Talk (V)"
            onClick={() => (listening ? stopListening() : listenAndAsk())}><Mic /></Button>
          <span className="flex-1 truncate text-[11px] text-muted-foreground"><Kbd>Enter</Kbd> send · <Kbd>V</Kbd> talk · <Kbd>E</Kbd> explain</span>
          {busy || speaking
            ? <Button type="button" size="icon-sm" variant="secondary" onClick={stopAll} title="Stop"><Square className="fill-current" /></Button>
            : <Button type="submit" size="icon-sm" disabled={!input.trim()} title="Send"><ArrowUp /></Button>}
        </div>
      </form>
    </div>
  </>);
}

function Intro() {
  const p = useStore(learner, (s) => s.profile);
  return (
    <Card className="gap-2 p-4 text-sm shadow-none">
      <div className="font-medium">I can see what you see.</div>
      <p className="text-muted-foreground">Your window, what you've selected or highlighted, what your mouse is over, and the exact numbers inside the model. Type, or press <Kbd>V</Kbd> and just talk. I can answer out loud too.</p>
      <p className="text-muted-foreground">I point at things with <span className="ptr-chip">◎</span> (hover to light them up) and suggest <span className="try-chip">▶ experiments</span>.</p>
      {p?.summary && <p className="border-t pt-2 text-xs text-muted-foreground"><span className="font-medium text-foreground">What I know about you: </span>{p.summary}</p>}
    </Card>
  );
}

/** Transparency: show exactly what context the tutor receives. */
function WhatItSees() {
  const a = useStore(ask, (s) => s);
  const [, tick] = useState(0);
  useEffect(() => { const iv = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(iv); }, []);
  const els = visibleElements(), ptr = currentPointer();
  const rows: [string, string][] = [
    ["Page", `${a.page}${a.scene ? ` › ${a.scene}` : ""}`],
    ["Window", viewportNote()],
    ["Your selection", a.selection.length ? compressSelection(a.selection).map((s) => s.label).join(" · ") : "Nothing selected"],
    ["Mouse is over", ptr ? `${ptr.label} (${ptr.ago}s ago)` : "Nothing recently"],
  ];
  return (
    <div className="space-y-2 p-3 text-sm">
      <p className="text-xs text-muted-foreground">This is sent with every question and updates live as you scroll, click and move the mouse.</p>
      {rows.map(([t, d]) => (
        <Item key={t} variant="outline" size="sm"><ItemContent><ItemTitle>{t}</ItemTitle><ItemDescription className="line-clamp-none">{d}</ItemDescription></ItemContent></Item>
      ))}
      <Item variant="outline" size="sm"><ItemContent><ItemTitle>On screen</ItemTitle><pre className="max-h-48 overflow-auto font-mono text-[11px] whitespace-pre-wrap text-muted-foreground">{a.summary || "—"}</pre></ItemContent></Item>
      <div className="pt-2 text-xs font-medium">{els.length} visible things it can point at</div>
      <ul className="space-y-1 text-xs text-muted-foreground">{els.slice(0, 40).map((e) => <li key={e.id}><code className="text-pin">{e.id}</code> {e.label.slice(0, 100)}</li>)}</ul>
    </div>
  );
}

/** Floating "ask about this" card shown after a click / highlight when the panel is closed. */
export function AskPopover() {
  const sel = useStore(ask, (s) => s.selection);
  const open = useStore(ask, (s) => s.open);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    const h = (e: Event) => setPos((e as CustomEvent).detail);
    window.addEventListener("ask-popover", h); return () => window.removeEventListener("ask-popover", h);
  }, []);
  useEffect(() => { if (open || !sel.length) setPos(null); }, [open, sel.length]);
  if (!pos || open || !sel.length) return null;
  const x = Math.min(pos.x, window.innerWidth - 330), y = Math.min(pos.y + 12, window.innerHeight - 220);
  const go = (fn: () => void) => { setPos(null); ask.set({ open: true }); fn(); };
  return (
    <Card className="fixed z-50 w-80 gap-3 p-3 shadow-lg animate-in fade-in-0 zoom-in-95" style={{ left: Math.max(8, x), top: y }}
      onMouseUp={(e) => e.stopPropagation()} onClick={(e) => e.stopPropagation()}>
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1 text-sm font-medium line-clamp-3">{sel[0].label}</div>
        {sel.length > 1 && <Badge variant="secondary">+{sel.length - 1}</Badge>}
        <Button variant="ghost" size="icon-xs" onClick={() => { setPos(null); ask.set({ selection: [] }); }}><X /></Button>
      </div>
      <div className="flex flex-col gap-1">
        {suggestionsFor(sel.map((s) => s.id)).slice(0, 2).map((q) => (
          <Button key={q} variant="ghost" size="sm" className="justify-start text-left whitespace-normal h-auto py-1.5" onClick={() => go(() => sendQuestion(q, "text"))}>{q}</Button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-1.5">
        <Button size="sm" onClick={() => go(() => listenAndAsk())}><Mic />Talk</Button>
        <Button size="sm" variant="outline" onClick={() => go(() => window.dispatchEvent(new Event("focus-ask")))}><KeyboardIcon />Type</Button>
        <Button size="sm" variant="outline" onClick={() => go(() => sendQuestion("Explain this.", "text"))}><Sparkles />Explain</Button>
      </div>
    </Card>
  );
}
