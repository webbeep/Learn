import { useEffect } from "react";
import { Mic, Square, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Kbd } from "@/components/ui/kbd";
import { cn } from "@/lib/utils";
import { ask, learner, setPrefs, useStore } from "../lib/store";
import { convo, listenAndAsk, sendQuestion, stopAll } from "../lib/conversation";
import { cancelListening, setHold, stopListening, stopSpeaking, voice } from "../lib/voice";

const KEYS: [string[], string][] = [
  [["V"], "Talk: press, speak, and it sends when you go quiet. Hold V to talk and release to send."],
  [["H"], "Highlight: drag a box over anything"],
  [["E"], "Explain what's selected, or what your mouse is over"],
  [["/"], "Type a question"],
  [["A"], "Open or close the tutor"],
  [["M"], "Mute or unmute spoken answers"],
  [["Esc"], "Stop listening or talking, cancel a highlight"],
  [["Space"], "Play or pause the X-ray or a video"],
  [["←", "→"], "Previous or next scene"],
  [["Shift", "Click"], "Select several things"],
  [["?"], "Show this list"],
];

/** Global keyboard shortcuts. */
export function useShortcuts(setHelp: (f: (h: boolean) => boolean) => void) {
  useEffect(() => {
    let vDown = 0;
    const typing = (e: KeyboardEvent) => !!(e.target as HTMLElement).closest?.("input, textarea, select, [contenteditable], [role=slider], [role=combobox]");
    const down = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const v = voice.get();
        if (v.listening) cancelListening(); else if (v.speaking || convo.get().busy) stopAll();
        else if (ask.get().regionMode) ask.set({ regionMode: false });
        return;
      }
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "v") {
        e.preventDefault(); if (e.repeat) return;
        if (voice.get().listening) { stopListening(); return; }
        vDown = performance.now(); listenAndAsk({ hold: true });
      } else if (k === "h") ask.set({ regionMode: !ask.get().regionMode });
      else if (k === "e") { ask.set({ open: true }); sendQuestion(ask.get().selection.length ? "Explain this." : "Explain what my mouse is pointing at.", "text"); }
      else if (k === "/") { e.preventDefault(); ask.set({ open: true }); setTimeout(() => window.dispatchEvent(new Event("focus-ask")), 30); }
      else if (k === "a") ask.set({ open: !ask.get().open });
      else if (k === "m") {
        if (learner.get().prefs.speak === "off") { setPrefs({ speak: "voice" }); toast("Spoken answers on", { description: "The tutor will talk back when you talk to it." }); }
        else { setPrefs({ speak: "off" }); stopSpeaking(); toast("Spoken answers muted"); }
      } else if (e.key === "?") setHelp((h) => !h);
    };
    const up = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== "v" || !vDown) return;
      const held = performance.now() - vDown; vDown = 0;
      if (held > 450) stopListening(); else setHold(false);
    };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [setHelp]);
}

export function ShortcutHelp({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard & voice shortcuts</DialogTitle>
          <DialogDescription>Change what happens after you highlight, and how the tutor talks back, in the tutor's Voice tab.</DialogDescription>
        </DialogHeader>
        <div className="divide-y">
          {KEYS.map(([ks, d]) => (
            <div key={d} className="flex items-start justify-between gap-4 py-2 text-sm">
              <span className="text-muted-foreground">{d}</span>
              <span className="flex shrink-0 gap-1">{ks.map((k) => <Kbd key={k}>{k}</Kbd>)}</span>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** Floating indicator while listening / transcribing / speaking. */
export function VoiceHUD() {
  const v = useStore(voice, (s) => s);
  useEffect(() => { if (v.error) { toast.error(v.error); voice.set({ error: "" }); } }, [v.error]);
  if (!(v.listening || v.transcribing || v.speaking)) return null;
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-36 z-50 flex justify-center px-4 md:bottom-24">
      <div className="pointer-events-auto flex max-w-xl items-center gap-3 rounded-full border bg-popover/95 py-2 pr-2 pl-4 text-popover-foreground shadow-xl backdrop-blur animate-in fade-in-0 slide-in-from-bottom-2">
        {v.listening ? (<>
          <span className="relative flex size-3"><span className="absolute inline-flex size-full animate-ping rounded-full bg-destructive opacity-60" /><span className="relative inline-flex size-3 rounded-full bg-destructive" style={{ transform: `scale(${1 + v.level * 0.5})` }} /></span>
          <span className="min-w-0 truncate text-sm">{v.interim || "Listening… ask about what you're looking at"}</span>
          <Button size="sm" onClick={stopListening}><Mic />Send</Button>
          <Button size="icon-sm" variant="ghost" onClick={cancelListening}><X /></Button>
        </>) : v.transcribing ? (<>
          <span className="size-3 animate-pulse rounded-full bg-primary" /><span className="text-sm">Transcribing…</span>
        </>) : (<>
          <span className="flex h-4 items-end gap-0.5">{[0, 1, 2, 3].map((i) => <span key={i} className={cn("w-1 animate-pulse rounded-full bg-viz-b")} style={{ height: `${6 + ((i * 5) % 10)}px`, animationDelay: `${i * 150}ms` }} />)}</span>
          <span className="text-sm">Speaking…</span>
          <Button size="sm" variant="ghost" onClick={stopSpeaking}><Square className="fill-current" />Stop</Button>
        </>)}
      </div>
    </div>
  );
}
