import { Compass, FlaskConical, GraduationCap, Keyboard, Laptop, Moon, PlayCircle, Sun, Telescope } from "lucide-react";
import { useTheme } from "next-themes";
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarHeader,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarRail,
} from "@/components/ui/sidebar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export const NAV = [
  { href: "#/", label: "Watch", icon: PlayCircle },
  { href: "#/learn", label: "Learn", icon: GraduationCap },
  { href: "#/labs", label: "Labs", icon: FlaskConical },
] as const;
const LAB_ROUTES = ["#/labs", "#/xray", "#/train", "#/training", "#/backprop", "#/lab", "#/scaling", "#/invent"];

export function isActive(href: string, hash: string) {
  const on = (b: string) => hash === b || hash.startsWith(b + "/") || hash.startsWith(b + "?");
  if (href === "#/") return hash === "#/" || hash === "" || on("#/watch");
  if (href === "#/labs") return LAB_ROUTES.some(on);
  return on(href);
}

export function AppSidebar({ hash, onShortcuts }: { hash: string; onShortcuts: () => void }) {
  const { setTheme } = useTheme();
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="#/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground"><Telescope className="size-4" /></div>
                <span className="truncate font-semibold">Top-Down AI</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map((n) => (
                <SidebarMenuItem key={n.href}>
                  <SidebarMenuButton asChild isActive={isActive(n.href, hash)} tooltip={n.label} size="lg">
                    <a href={n.href}><n.icon /><span>{n.label}</span></a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={isActive("#/path", hash)} tooltip="My notes">
              <a href="#/path"><Compass /><span>My notes</span></a>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={onShortcuts} tooltip="Shortcuts"><Keyboard /><span>Shortcuts</span></SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton tooltip="Theme"><Sun className="dark:hidden" /><Moon className="hidden dark:block" /><span>Theme</span></SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="right" align="end">
                <DropdownMenuItem onClick={() => setTheme("light")}><Sun /> Light</DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTheme("dark")}><Moon /> Dark</DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTheme("system")}><Laptop /> System</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
