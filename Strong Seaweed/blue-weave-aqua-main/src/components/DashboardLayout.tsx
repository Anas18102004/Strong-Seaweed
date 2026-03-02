import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Bell, LogOut, LayoutDashboard, FlaskConical, MapPin, Brain, MessageSquare } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { useLocation, useNavigate } from "react-router-dom";
import { NavLink } from "@/components/NavLink";

const mobileNavItems = [
  { title: "Home", url: "/dashboard", icon: LayoutDashboard },
  { title: "Predict", url: "/predict", icon: FlaskConical },
  { title: "Sites", url: "/sites", icon: MapPin },
  { title: "Agents", url: "/agents", icon: Brain },
  { title: "Chat", url: "/chat", icon: MessageSquare },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const pageLabel = (() => {
    const map: Record<string, string> = {
      "/dashboard": "Live Dashboard",
      "/predict": "Check My Location",
      "/results": "Prediction Results",
      "/sites": "Top Farming Areas",
      "/agents": "Task Assistants",
      "/reports": "My Saved Results",
      "/forecast": "Weather & Season",
      "/chat": "AI Chat",
      "/settings": "Settings",
    };
    return map[location.pathname] || "BlueWeave";
  })();

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full gradient-bg relative overflow-hidden">
        <div className="pointer-events-none absolute -top-20 left-[-8rem] h-72 w-72 rounded-full bg-cyan-300/20 blur-3xl" />
        <div className="pointer-events-none absolute top-32 right-[-7rem] h-80 w-80 rounded-full bg-blue-300/20 blur-3xl" />
        <AppSidebar />
        <div className="flex-1 flex flex-col relative z-10">
          <header className="min-h-16 flex items-center justify-between border-b border-white/50 px-3 sm:px-5 py-2.5 glass-strong">
            <div className="flex items-center gap-3 min-w-0">
              <SidebarTrigger className="text-slate-500 hover:text-slate-800" />
              <div className="min-w-0">
                <p className="text-[11px] uppercase tracking-[0.15em] text-cyan-700/90">BlueWeave Workspace</p>
                <h1 className="text-sm sm:text-base font-semibold text-slate-900 truncate">{pageLabel}</h1>
              </div>
            </div>
            <div className="flex items-center gap-2.5">
              <span className="hidden lg:inline-flex text-xs font-medium text-slate-700 glass rounded-full px-3 py-1">Glass UI</span>
              <span className="hidden sm:inline-flex text-xs font-medium text-slate-700 glass rounded-full px-3 py-1">v1.1 Gulf Model</span>
              <button className="w-9 h-9 rounded-xl glass flex items-center justify-center text-slate-500 hover:text-slate-900 transition-colors">
                <Bell className="w-4 h-4" />
              </button>
              <div className="text-xs text-slate-600 hidden sm:block max-w-[11rem] truncate">{user?.name || user?.email || "User"}</div>
              <button
                onClick={() => {
                  signOut();
                  navigate("/signin");
                }}
                className="w-9 h-9 rounded-xl glass flex items-center justify-center text-slate-500 hover:text-slate-900 transition-colors"
                title="Sign out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </header>
          <main className="flex-1 p-3 sm:p-6 pb-24 md:pb-6 overflow-auto">{children}</main>
        </div>
      </div>
      <nav className="md:hidden fixed bottom-0 inset-x-0 z-40 border-t border-border/40 glass-strong px-2 pt-2 pb-[env(safe-area-inset-bottom)]">
        <div className="grid grid-cols-5 gap-1">
          {mobileNavItems.map((item) => {
            const active = location.pathname === item.url;
            return (
              <NavLink
                key={item.title}
                to={item.url}
                className={`min-h-12 rounded-xl flex flex-col items-center justify-center gap-0.5 text-[10px] font-medium transition-colors ${
                  active ? "gradient-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <item.icon className="w-4 h-4" />
                <span>{item.title}</span>
              </NavLink>
            );
          })}
        </div>
      </nav>
    </SidebarProvider>
  );
}
