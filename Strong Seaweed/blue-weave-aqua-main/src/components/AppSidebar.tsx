import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  FlaskConical,
  MapPin,
  Brain,
  FileText,
  CloudSun,
  MessageSquare,
  Settings,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import BrandLogo from "@/components/BrandLogo";

const navSections = [
  {
    label: "Overview",
    items: [{ title: "Live Dashboard", hint: "Farm pulse", url: "/dashboard", icon: LayoutDashboard }],
  },
  {
    label: "Operations",
    items: [
      { title: "Check My Location", hint: "Suitability", url: "/predict", icon: FlaskConical },
      { title: "Top Farming Areas", hint: "Site explorer", url: "/sites", icon: MapPin },
      { title: "Weather & Season", hint: "Risk radar", url: "/forecast", icon: CloudSun },
      { title: "My Saved Results", hint: "History", url: "/reports", icon: FileText },
    ],
  },
  {
    label: "AI Workspace",
    items: [
      { title: "Task Assistants", hint: "Automation", url: "/agents", icon: Brain },
      { title: "Chat with AI", hint: "Copilot", url: "/chat", icon: MessageSquare },
      { title: "My Settings", hint: "Preferences", url: "/settings", icon: Settings },
    ],
  },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();

  const isActive = (url: string) => location.pathname === url || location.pathname.startsWith(`${url}/`);

  return (
    <Sidebar collapsible="icon" className="border-r-0 bg-transparent">
      <div className="m-3 mb-2 rounded-3xl border border-white/65 bg-white/65 backdrop-blur-xl shadow-[0_18px_36px_-24px_rgba(8,47,73,0.55)] p-3 flex items-center gap-2.5">
        <div className="rounded-xl bg-gradient-to-br from-cyan-500/15 to-blue-500/15 p-1.5">
          <BrandLogo size="md" showWordmark={false} />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-[0.16em] text-cyan-700">Navigation Hub</p>
            <div className="flex items-center gap-1.5">
              <span className="text-base font-bold text-foreground whitespace-nowrap">BlueWeave</span>
              <span className="inline-flex items-center rounded-full bg-cyan-500/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-cyan-700">
                AI
              </span>
            </div>
          </div>
        )}
      </div>
      <SidebarContent>
        {navSections.map((section) => (
          <SidebarGroup key={section.label}>
            {!collapsed && (
              <SidebarGroupLabel className="px-4 pb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                {section.label}
              </SidebarGroupLabel>
            )}
            <SidebarGroupContent>
              <SidebarMenu>
                {section.items.map((item) => {
                  const active = isActive(item.url);
                  return (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton asChild>
                        <NavLink
                          to={item.url}
                          end={item.url === "/dashboard"}
                          className={`group rounded-2xl px-2.5 py-2 transition-all duration-200 ${
                            active
                              ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-[0_14px_28px_-18px_rgba(14,116,144,0.95)]"
                              : "text-slate-600 hover:bg-white/70 hover:text-slate-900"
                          }`}
                          activeClassName=""
                        >
                          <item.icon className={`h-4 w-4 shrink-0 ${collapsed ? "mx-auto" : "mr-2.5"}`} />
                          {!collapsed && (
                            <div className="min-w-0 flex-1">
                              <p className="truncate text-sm font-semibold leading-tight">{item.title}</p>
                              <p
                                className={`truncate text-[11px] leading-tight ${
                                  active ? "text-white/80" : "text-slate-500 group-hover:text-slate-600"
                                }`}
                              >
                                {item.hint}
                              </p>
                            </div>
                          )}
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
        {!collapsed && (
          <div className="mx-3 mb-3 mt-auto space-y-2.5">
            <div className="rounded-2xl border border-white/65 bg-white/65 p-3 text-xs text-slate-600 backdrop-blur-xl">
              <p className="font-semibold text-slate-900">Live Status</p>
              <p className="mt-1">Models online and advisors ready.</p>
            </div>
            <div className="rounded-2xl border border-cyan-200/80 bg-gradient-to-br from-cyan-50 to-blue-50 p-3">
              <div className="flex items-center gap-2 text-cyan-800">
                <Sparkles className="h-3.5 w-3.5" />
                <p className="text-xs font-semibold">Quick start</p>
              </div>
              <NavLink
                to="/predict"
                className="mt-2 inline-flex w-full items-center justify-between rounded-xl bg-white/90 px-2.5 py-2 text-xs font-semibold text-slate-800 transition-colors hover:bg-white"
              >
                Run new location check
                <ArrowRight className="h-3.5 w-3.5" />
              </NavLink>
            </div>
          </div>
        )}
      </SidebarContent>
    </Sidebar>
  );
}
