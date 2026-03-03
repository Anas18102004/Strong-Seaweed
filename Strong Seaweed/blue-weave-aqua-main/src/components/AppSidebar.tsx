import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
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
} from "lucide-react";
import BrandLogo from "@/components/BrandLogo";

const navItems = [
  { title: "Live Dashboard", url: "/dashboard", icon: LayoutDashboard },
  { title: "Check My Location", url: "/predict", icon: FlaskConical },
  { title: "Top Farming Areas", url: "/sites", icon: MapPin },
  { title: "Task Assistants", url: "/agents", icon: Brain },
  { title: "My Saved Results", url: "/reports", icon: FileText },
  { title: "Weather & Season", url: "/forecast", icon: CloudSun },
  { title: "Chat with AI", url: "/chat", icon: MessageSquare },
  { title: "My Settings", url: "/settings", icon: Settings },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();

  return (
    <Sidebar collapsible="icon" className="border-r-0 bg-transparent">
      <div className="m-3 mb-2 rounded-2xl border border-white/60 bg-white/55 backdrop-blur-xl shadow-[0_12px_28px_-20px_rgba(12,74,110,0.45)] p-3 flex items-center gap-2.5">
        <BrandLogo size="md" showWordmark={false} />
        {!collapsed && (
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-[0.14em] text-cyan-700">Control Center</p>
            <span className="text-lg font-bold text-foreground whitespace-nowrap">
              BlueWeave<span className="gradient-text"> AI</span>
            </span>
          </div>
        )}
      </div>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            {!collapsed && (
              <p className="px-4 pb-2 text-[11px] uppercase tracking-[0.15em] text-slate-500">Navigation</p>
            )}
            <SidebarMenu>
              {navItems.map((item) => {
                const active = location.pathname === item.url;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink
                        to={item.url}
                        end
                        className={`rounded-xl transition-all duration-200 ${
                          active
                            ? "gradient-primary text-primary-foreground shadow-[0_10px_24px_-14px_rgba(2,132,199,0.9)]"
                            : "text-slate-600 hover:bg-white/70 hover:text-slate-900"
                        }`}
                        activeClassName=""
                      >
                        <item.icon className="mr-2.5 h-4 w-4" />
                        {!collapsed && <span className="text-sm font-medium">{item.title}</span>}
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        {!collapsed && (
          <div className="mx-3 mb-3 mt-auto rounded-2xl border border-white/60 bg-white/55 p-3 text-xs text-slate-600 backdrop-blur-xl">
            <p className="font-semibold text-slate-900">AI Status</p>
            <p className="mt-1">Live model + advisor tools ready.</p>
          </div>
        )}
      </SidebarContent>
    </Sidebar>
  );
}
