import { Boxes, Clapperboard, Compass, Cpu, FlaskConical, GitBranch, GraduationCap, Keyboard, Lightbulb, Map as MapIcon, Moon, ScanEye, Sun, Telescope, TrendingDown, Laptop } from "lucide-react";
import { useTheme } from "next-themes";
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarGroupLabel, SidebarHeader,
  SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarRail,
} from "@/components/ui/sidebar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { learner, useStore } from "@/lib/store";

export const NAV = [
  { href: "#/", label: "Home", icon: MapIcon, group: "Learn" },
  { href: "#/learn", label: "Curriculum", icon: GraduationCap, group: "Learn" },
  { href: "#/videos", label: "Videos", icon: Clapperboard, group: "Learn" },
  { href: "#/xray", label: "Inference X-ray", icon: ScanEye, group: "Labs" },
  { href: "#/training", label: "Training run", icon: TrendingDown, group: "Labs" },
  { href: "#/train", label: "Train it yourself", icon: Cpu, group: "Labs" },
  { href: "#/backprop", label: "Backprop", icon: GitBranch, group: "Labs" },
  { href: "#/lab", label: "Architecture lab", icon: FlaskConical, group: "Labs" },
  { href: "#/scaling", label: "Scaling", icon: Boxes, group: "Labs" },
  { href: "#/invent", label: "Invent your own", icon: Lightbulb, group: "Labs" },
  { href: "#/path", label: "My path", icon: Compass, group: "You" },
] as const;

export function isActive(href: string, hash: string) {
  if (href === "#/") return hash === "#/" || hash === "";
  return hash === href || hash.startsWith(href + "/") || hash.startsWith(href + "?");
}

export function AppSidebar({ hash, onShortcuts }: { hash: string; onShortcuts: () => void }) {
  const wishes = useStore(learner, (s) => s.wishes.filter((w) => !w.done).length);
  const asked = useStore(learner, (s) => s.history.length);
  const { setTheme } = useTheme();
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="#/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <Telescope className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">Top-Down AI</span>
                  <span className="truncate text-xs text-muted-foreground">Learn by poking a real model</span>
                </div>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {(["Learn", "Labs", "You"] as const).map((g) => (
          <SidebarGroup key={g}>
            <SidebarGroupLabel>{g}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV.filter((n) => n.group === g).map((n) => (
                  <SidebarMenuItem key={n.href}>
                    <SidebarMenuButton asChild isActive={isActive(n.href, hash)} tooltip={n.label}>
                      <a href={n.href}><n.icon /><span>{n.label}</span></a>
                    </SidebarMenuButton>
                    {n.href === "#/path" && wishes > 0 && <SidebarMenuBadge>{wishes}</SidebarMenuBadge>}
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={onShortcuts} tooltip="Keyboard & voice shortcuts">
              <Keyboard /><span>Shortcuts</span><span className="ml-auto text-xs text-muted-foreground">?</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton tooltip="Theme">
                  <Sun className="dark:hidden" /><Moon className="hidden dark:block" /><span>Theme</span>
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="right" align="end">
                <DropdownMenuItem onClick={() => setTheme("light")}><Sun /> Light</DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTheme("dark")}><Moon /> Dark</DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTheme("system")}><Laptop /> System</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <div className="px-2 py-1 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">{asked} question{asked === 1 ? "" : "s"} asked so far</div>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
