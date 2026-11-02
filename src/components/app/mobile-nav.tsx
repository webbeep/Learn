import { Clapperboard, GraduationCap, Home, MessageSquare, ScanEye } from "lucide-react";
import { cn } from "@/lib/utils";
import { ask, useStore } from "@/lib/store";
import { isActive } from "./app-sidebar";

const ITEMS = [
  { href: "#/", label: "Home", icon: Home },
  { href: "#/learn", label: "Learn", icon: GraduationCap },
  { href: "#/xray", label: "X-ray", icon: ScanEye },
  { href: "#/videos", label: "Videos", icon: Clapperboard },
];

/** Bottom tab bar for phones (the sidebar is still available from the header). */
export function MobileNav({ hash }: { hash: string }) {
  const open = useStore(ask, (s) => s.open);
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 border-t bg-background/90 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden">
      <div className="grid h-14 grid-cols-5">
        {ITEMS.map((it) => {
          const on = isActive(it.href, hash);
          return (
            <a key={it.href} href={it.href} className={cn("flex flex-col items-center justify-center gap-0.5 text-[11px]", on ? "text-foreground" : "text-muted-foreground")}>
              <it.icon className={cn("size-5", on && "stroke-[2.4]")} />{it.label}
            </a>
          );
        })}
        <button onClick={() => ask.set({ open: !open })} className={cn("flex flex-col items-center justify-center gap-0.5 text-[11px]", open ? "text-foreground" : "text-muted-foreground")}>
          <MessageSquare className="size-5" />Tutor
        </button>
      </div>
    </nav>
  );
}
