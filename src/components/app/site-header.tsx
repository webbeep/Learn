import { MessageSquare, Mic } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { ask, useStore } from "@/lib/store";
import { voice, stopListening } from "@/lib/voice";
import { listenAndAsk } from "@/lib/conversation";
import { cn } from "@/lib/utils";

export function SiteHeader({ title }: { title: string }) {
  const open = useStore(ask, (s) => s.open);
  const listening = useStore(voice, (s) => s.listening || s.transcribing);
  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-2 border-b bg-background/85 px-3 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-1 h-4! self-center!" />
      <span className="min-w-0 flex-1 truncate text-sm font-medium">{title}</span>
      <Button variant={listening ? "destructive" : "ghost"} size="icon" title="Ask out loud (V)" aria-label="Talk"
        onClick={() => (listening ? stopListening() : listenAndAsk())} className={cn(listening && "animate-pulse")}><Mic /></Button>
      <Button variant={open ? "secondary" : "default"} size="sm" className="hidden md:inline-flex" onClick={() => ask.set({ open: !open })}>
        <MessageSquare />Ask
      </Button>
    </header>
  );
}
