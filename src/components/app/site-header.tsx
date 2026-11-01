import { MessageSquare, Mic, SquareDashedMousePointer } from "lucide-react";
import { Breadcrumb, BreadcrumbItem, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ask, useStore } from "@/lib/store";
import { voice, stopListening } from "@/lib/voice";
import { listenAndAsk } from "@/lib/conversation";
import { cn } from "@/lib/utils";

export function SiteHeader({ title }: { title: string }) {
  const scene = useStore(ask, (s) => s.scene);
  const open = useStore(ask, (s) => s.open);
  const regionMode = useStore(ask, (s) => s.regionMode);
  const listening = useStore(voice, (s) => s.listening || s.transcribing);
  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-2 border-b bg-background/85 px-3 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-1 h-4! self-center!" />
      <Breadcrumb className="min-w-0 flex-1">
        <BreadcrumbList className="flex-nowrap">
          <BreadcrumbItem className="truncate"><BreadcrumbPage className={cn(scene && "text-muted-foreground")}>{title}</BreadcrumbPage></BreadcrumbItem>
          {scene && <><BreadcrumbSeparator className="hidden sm:block" /><BreadcrumbItem className="hidden truncate sm:block"><BreadcrumbPage>{scene}</BreadcrumbPage></BreadcrumbItem></>}
        </BreadcrumbList>
      </Breadcrumb>
      <div className="flex items-center gap-1.5">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant={regionMode ? "secondary" : "ghost"} size="sm" onClick={() => ask.set({ regionMode: !regionMode })}>
              <SquareDashedMousePointer /><span className="hidden md:inline">Highlight</span><Kbd className="hidden md:inline-flex">H</Kbd>
            </Button>
          </TooltipTrigger>
          <TooltipContent>Drag a box over anything to ask about it</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant={listening ? "destructive" : "ghost"} size="sm" onClick={() => (listening ? stopListening() : listenAndAsk())} className={cn(listening && "animate-pulse")}>
              <Mic /><span className="hidden md:inline">{listening ? "Listening…" : "Talk"}</span><Kbd className="hidden md:inline-flex">V</Kbd>
            </Button>
          </TooltipTrigger>
          <TooltipContent>Ask out loud about what you're looking at</TooltipContent>
        </Tooltip>
        <Button variant={open ? "secondary" : "default"} size="sm" onClick={() => ask.set({ open: !open })}>
          <MessageSquare /><span className="hidden sm:inline">Tutor</span>
        </Button>
      </div>
    </header>
  );
}
