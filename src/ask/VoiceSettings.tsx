import { useEffect, useState } from "react";
import { Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { learner, setPrefs, useStore, type Prefs } from "../lib/store";
import { englishVoices, hasBrowserSTT, speakText } from "../lib/voice";

const NATURAL = [["luna", "Luna · warm"], ["asteria", "Asteria · clear"], ["thalia", "Thalia · bright"], ["helena", "Helena · calm"],
  ["orion", "Orion · male"], ["arcas", "Arcas · deep male"], ["apollo", "Apollo · casual male"], ["zeus", "Zeus · male"]];

function Seg<K extends keyof Prefs>({ k, opts }: { k: K; opts: [Prefs[K], string][] }) {
  const v = useStore(learner, (s) => s.prefs[k]);
  return (
    <ToggleGroup type="single" variant="outline" size="sm" className="w-full" value={String(v)}
      onValueChange={(val) => { if (val) setPrefs({ [k]: opts.find(([o]) => String(o) === val)![0] } as Partial<Prefs>); }}>
      {opts.map(([val, label]) => <ToggleGroupItem key={String(val)} value={String(val)} className="flex-1 text-xs">{label}</ToggleGroupItem>)}
    </ToggleGroup>
  );
}
function Section({ title, hint, children }: { title: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return <div className="space-y-2"><Label className="text-sm">{title}</Label>{children}{hint && <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>}</div>;
}

export function VoiceSettings() {
  const p = useStore(learner, (s) => s.prefs);
  const [voices, setVoices] = useState(englishVoices());
  useEffect(() => { const f = () => setVoices(englishVoices()); speechSynthesis?.addEventListener?.("voiceschanged", f); return () => speechSynthesis?.removeEventListener?.("voiceschanged", f); }, []);
  return (
    <div className="space-y-5 p-4">
      <Section title="When I highlight something" hint="Applies to highlighted text and dragged boxes. A plain click always shows the options.">
        <Seg k="onHighlight" opts={[["manual", "Show options"], ["listen", "Start listening"], ["explain", "Explain now"]]} />
      </Section>
      <Separator />
      <Section title="Speech recognition" hint={<>{hasBrowserSTT ? "The browser shows your words live as you talk. Whisper runs on Cloudflare and is more accurate." : "This browser can't recognize speech itself, so Whisper on Cloudflare is used."} Press <Kbd>V</Kbd> to talk (it stops when you go quiet), or hold <Kbd>V</Kbd> to talk and release to send.</>}>
        <Seg k="stt" opts={[["auto", "Auto"], ["browser", "Browser"], ["whisper", "Whisper"]]} />
      </Section>
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1"><Label htmlFor="hf">Hands-free conversation</Label><p className="text-xs text-muted-foreground">Listen again after each spoken answer.</p></div>
        <Switch id="hf" checked={p.handsFree} onCheckedChange={(v) => setPrefs({ handsFree: v })} />
      </div>
      <Separator />
      <Section title="Read answers aloud" hint={<>Press <Kbd>M</Kbd> to mute or unmute, and <Kbd>Esc</Kbd> to stop talking.</>}>
        <Seg k="speak" opts={[["off", "Off"], ["voice", "When I talk"], ["always", "Always"]]} />
      </Section>
      <Section title="Voice">
        <Seg k="ttsEngine" opts={[["natural", "Natural (Cloudflare)"], ["browser", "Browser"]]} />
        {p.ttsEngine === "natural" ? (
          <Select value={p.naturalVoice} onValueChange={(v) => setPrefs({ naturalVoice: v })}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{NATURAL.map(([v, l]) => <SelectItem key={v} value={v}>{l}</SelectItem>)}</SelectContent>
          </Select>
        ) : (
          <Select value={p.browserVoice || "__best"} onValueChange={(v) => setPrefs({ browserVoice: v === "__best" ? "" : v })}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent><SelectItem value="__best">Best available</SelectItem>{voices.map((v) => <SelectItem key={v.name} value={v.name}>{v.name}</SelectItem>)}</SelectContent>
          </Select>
        )}
        <div className="flex items-center gap-3 pt-1">
          <span className="w-20 text-xs text-muted-foreground">Speed {p.rate.toFixed(2)}×</span>
          <Slider min={0.7} max={1.6} step={0.05} value={[p.rate]} onValueChange={([v]) => setPrefs({ rate: v })} className="flex-1" />
        </div>
        <Button variant="outline" size="sm" onClick={() => speakText("Hi! Highlight anything, press V, and ask me about it. I can see what's in your window.")}><Play />Test voice</Button>
      </Section>
    </div>
  );
}
