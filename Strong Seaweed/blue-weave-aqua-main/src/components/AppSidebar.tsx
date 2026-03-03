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
  ChevronLeft,
  ChevronRight,
  Waves,
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
  const { state, toggleSidebar } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();

  const isActive = (url: string) => location.pathname === url || location.pathname.startsWith(`${url}/`);

  return (
    <Sidebar
      collapsible="icon"
      className="border-r-0 bg-transparent
      [&>[data-sidebar=sidebar]]:relative [&>[data-sidebar=sidebar]]:m-2 [&>[data-sidebar=sidebar]]:h-[calc(100%-1rem)]
      [&>[data-sidebar=sidebar]]:overflow-hidden [&>[data-sidebar=sidebar]]:rounded-[24px]
      [&>[data-sidebar=sidebar]]:border [&>[data-sidebar=sidebar]]:border-white/10
      [&>[data-sidebar=sidebar]]:bg-[linear-gradient(180deg,#0F2E47_0%,#123E63_52%,#0B2236_100%)]
      [&>[data-sidebar=sidebar]]:shadow-[18px_0_36px_-28px_rgba(11,34,54,0.85),inset_0_1px_0_rgba(255,255,255,0.06)]"
    >
      <div className="pointer-events-none absolute inset-0 opacity-[0.065] [background-image:radial-gradient(rgba(255,255,255,0.95)_0.35px,transparent_0.35px)] [background-size:3px_3px]" />
      <div className="pointer-events-none absolute -right-16 top-10 h-40 w-40 rounded-full bg-cyan-300/15 blur-3xl" />
      <div className="m-3 mb-2 rounded-2xl border border-white/10 bg-white/[0.045] p-3 backdrop-blur-xl shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] transition-all duration-200 hover:-translate-y-0.5 hover:bg-white/[0.07] hover:shadow-[0_14px_24px_-20px_rgba(22,163,233,0.85),inset_0_1px_0_rgba(255,255,255,0.1)]">
        <div className="flex items-center gap-2.5">
          <div className="relative rounded-2xl bg-gradient-to-br from-[#1DA1F2] to-[#0B6CB8] p-1.5 shadow-[0_0_20px_rgba(29,161,242,0.45)]">
            <div className="absolute inset-0 -z-10 rounded-2xl bg-cyan-300/50 blur-lg" />
            <BrandLogo size={collapsed ? "sm" : "md"} showWordmark={false} />
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <p className="text-[11px] uppercase tracking-[0.16em] text-[#7FA9C4]">Marine Intelligence</p>
              <div className="flex items-center gap-1.5">
                <span className="text-base font-semibold text-white whitespace-nowrap">BlueWave</span>
                <span className="inline-flex items-center rounded-full border border-cyan-200/30 bg-cyan-400/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-cyan-100 shadow-[0_0_12px_rgba(34,211,238,0.35)]">
                  AI
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
      <SidebarContent className="overflow-y-auto pb-2 [scrollbar-color:rgba(125,183,221,0.55)_transparent] [scrollbar-width:thin]">
        <div className="space-y-1">
          {navSections.map((section, sectionIdx) => (
            <SidebarGroup key={section.label} className="px-3 py-2">
              {!collapsed && (
                <SidebarGroupLabel className="mb-1 h-auto px-1 pb-1 text-[11px] font-medium uppercase tracking-[1.5px] text-[#7FA9C4]">
                  <div className="w-full">
                    <p>{section.label}</p>
                    <div className="mt-1.5 h-px w-full bg-gradient-to-r from-cyan-200/25 via-cyan-200/10 to-transparent" />
                  </div>
                </SidebarGroupLabel>
              )}
              <SidebarGroupContent>
                <SidebarMenu className="gap-1.5">
                  {section.items.map((item, itemIdx) => {
                    const active = isActive(item.url);
                    return (
                      <SidebarMenuItem
                        key={item.title}
                        className="animate-fade-in-up"
                        style={{ animationDelay: `${sectionIdx * 80 + itemIdx * 45}ms` }}
                      >
                        <SidebarMenuButton asChild tooltip={item.title} className="h-auto p-0 group-data-[collapsible=icon]:!size-10 group-data-[collapsible=icon]:!p-0">
                          <NavLink
                            to={item.url}
                            end={item.url === "/dashboard"}
                            className={`group relative h-12 overflow-hidden rounded-xl border transition-all duration-200 ${
                              active
                                ? "border-cyan-200/25 bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] text-white shadow-[0_12px_24px_-14px_rgba(14,165,233,0.95)]"
                                : "border-transparent bg-white/[0.02] text-[#CFE9FF] hover:translate-x-1 hover:border-cyan-200/20 hover:bg-white/[0.08] hover:shadow-[0_10px_20px_-16px_rgba(34,211,238,0.75)]"
                            }`}
                            activeClassName=""
                          >
                            <span className={`absolute left-0 top-1/2 h-8 -translate-y-1/2 rounded-r-full bg-white/90 transition-all duration-200 ${active ? "w-[3px] opacity-100" : "w-0 opacity-0"}`} />
                            <span className="pointer-events-none absolute left-1/2 top-1/2 h-20 w-20 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/25 opacity-0 scale-0 transition-all duration-300 group-active:scale-100 group-active:opacity-30" />
                            <div className={`flex h-full items-center ${collapsed ? "justify-center" : "px-3"}`}>
                              <item.icon
                                className={`h-4 w-4 shrink-0 transition-colors ${
                                  collapsed ? "" : "mr-2.5"
                                } ${active ? "text-white" : "text-cyan-200 group-hover:text-cyan-100"}`}
                              />
                              {!collapsed && (
                                <div className="min-w-0 flex-1">
                                  <p className="truncate text-[14px] font-medium leading-tight">{item.title}</p>
                                  <p className={`truncate text-[11px] leading-tight ${active ? "text-white/85" : "text-[#7FA9C4] group-hover:text-[#b8dcf5]"}`}>
                                    {item.hint}
                                  </p>
                                </div>
                              )}
                            </div>
                          </NavLink>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    );
                  })}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          ))}
        </div>
        <div className="mx-3 mb-2 mt-auto space-y-2.5">
          {!collapsed ? (
            <div className="rounded-2xl border border-white/12 bg-white/[0.06] p-3 backdrop-blur-xl shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-white">Live Status</p>
                <div className="flex items-center gap-1.5">
                  <span className="relative inline-flex h-2.5 w-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-300 opacity-75" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
                  </span>
                  <span className="text-[11px] text-emerald-100">Models online</span>
                </div>
              </div>
              <div className="mt-2 flex items-end gap-1 h-4">
                <span className="w-1 rounded-full bg-cyan-200/80 sidebar-waveform" style={{ animationDelay: "0ms" }} />
                <span className="w-1 rounded-full bg-cyan-200/80 sidebar-waveform" style={{ animationDelay: "120ms" }} />
                <span className="w-1 rounded-full bg-cyan-200/80 sidebar-waveform" style={{ animationDelay: "240ms" }} />
                <span className="w-1 rounded-full bg-cyan-200/80 sidebar-waveform" style={{ animationDelay: "360ms" }} />
                <span className="w-1 rounded-full bg-cyan-200/80 sidebar-waveform" style={{ animationDelay: "480ms" }} />
              </div>
              <NavLink
                to="/predict"
                className="mt-3 inline-flex w-full items-center justify-between rounded-xl border border-cyan-100/20 bg-cyan-400/10 px-2.5 py-2 text-xs font-semibold text-cyan-50 transition-all hover:bg-cyan-300/15"
              >
                Run new location check
                <ArrowRight className="h-3.5 w-3.5" />
              </NavLink>
            </div>
          ) : (
            <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl border border-white/15 bg-white/[0.06] text-cyan-100">
              <Waves className="h-4 w-4" />
            </div>
          )}
          <button
            type="button"
            onClick={toggleSidebar}
            className="group flex h-10 w-full items-center justify-center rounded-xl border border-white/12 bg-white/[0.05] text-[#CFE9FF] transition-all duration-200 hover:bg-white/[0.1]"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.12em]">
                <Sparkles className="h-3.5 w-3.5 text-cyan-200" />
                Collapse
                <ChevronLeft className="h-4 w-4 transition-transform group-hover:-translate-x-0.5" />
              </div>
            )}
          </button>
          {collapsed && (
            <p className="pb-1 text-center text-[10px] uppercase tracking-[0.18em] text-[#7FA9C4]">Ctrl/Cmd + B</p>
          )}
        </div>
      </SidebarContent>
    </Sidebar>
  );
}
