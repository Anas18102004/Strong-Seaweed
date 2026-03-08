import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Bell, LogOut, LayoutDashboard, FlaskConical, MapPin, Brain, MessageSquare, Activity, ChevronRight } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { useLocation, useNavigate } from "react-router-dom";
import { NavLink } from "@/components/NavLink";
import ElevenLabsConvaiWidget from "@/components/ElevenLabsConvaiWidget";

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
  const userLabel = user?.name || user?.email || "User";
  const userInitial = userLabel.trim().charAt(0).toUpperCase();
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
    return map[location.pathname] || "Akuara";
  })();

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full gradient-bg relative overflow-hidden">
        <div className="pointer-events-none absolute -top-20 left-[-8rem] h-72 w-72 rounded-full bg-cyan-300/20 blur-3xl" />
        <div className="pointer-events-none absolute top-32 right-[-7rem] h-80 w-80 rounded-full bg-blue-300/20 blur-3xl" />
        <AppSidebar />
        <div className="flex-1 flex flex-col relative z-10">
          <header className="relative px-3 sm:px-6 py-3 border-b border-[#9cc3dc]/55 bg-[linear-gradient(180deg,rgba(250,254,255,0.98)_0%,rgba(237,247,253,0.92)_100%)] backdrop-blur-md">
            <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300/35 to-transparent" />
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-cyan-300/40 to-transparent" />
            <div className="flex items-start justify-between gap-4 rounded-2xl border border-[#b5d4e8]/60 bg-white/75 px-3.5 py-3 shadow-[0_14px_30px_-24px_rgba(8,57,92,0.62)]">
              <div className="min-w-0 flex items-start gap-3 sm:gap-4">
                <SidebarTrigger className="mt-0.5 text-slate-500 hover:text-[#123E63] hover:bg-cyan-50/80 transition-all duration-200" />
                <div className="min-w-0 space-y-1.5">
                  <div className="flex items-center gap-2 text-[11px] text-slate-500 tracking-[0.09em] uppercase">
                    <span className="font-semibold text-slate-600">Workspace</span>
                    <ChevronRight className="h-3 w-3 text-cyan-500/70" />
                    <span className="truncate text-slate-500">Marine Operations</span>
                  </div>
                  <h1 className="text-xl sm:text-2xl font-semibold text-[#0F2E47] leading-tight truncate">{pageLabel}</h1>
                  <div className="hidden sm:flex flex-wrap items-center gap-3 text-xs text-slate-600">
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200/60 bg-emerald-50 px-2.5 py-1 text-emerald-700 font-medium">
                      <Activity className="h-3.5 w-3.5" />
                      Operational
                    </span>
                    <span className="h-3.5 w-px bg-slate-300/80" />
                    <span className="font-medium text-slate-700">Marine Core v2</span>
                    <span className="h-3.5 w-px bg-slate-300/80" />
                    <span>Last updated 3 min ago</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 sm:gap-3">
                <button
                  className="relative inline-flex h-10 w-10 items-center justify-center rounded-xl border border-[#b8d4e6] bg-white text-slate-500 transition-all duration-200 hover:-translate-y-0.5 hover:border-cyan-300 hover:text-[#123E63] hover:shadow-[0_10px_18px_-14px_rgba(15,46,71,0.55)]"
                  title="Notifications"
                >
                  <Bell className="w-4 h-4" />
                  <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_0_3px_rgba(125,211,252,0.35)]" />
                </button>
                <div className="hidden sm:flex items-center gap-2 rounded-xl border border-[#b5d4e8] bg-white/95 px-2.5 py-1.5">
                  <div className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[#1DA1F2] to-[#0F2E47] text-white text-sm font-semibold shadow-[0_10px_18px_-12px_rgba(15,46,71,0.85)]">
                    {userInitial}
                  </div>
                  <div className="max-w-[11rem]">
                    <p className="truncate text-sm font-semibold text-[#0F2E47]">{userLabel}</p>
                    <p className="truncate text-[11px] text-slate-500">Enterprise Access</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    signOut();
                    navigate("/signin");
                  }}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-[#b8d4e6] bg-white text-slate-500 transition-all duration-200 hover:-translate-y-0.5 hover:border-rose-300 hover:text-rose-600 hover:shadow-[0_10px_18px_-14px_rgba(244,63,94,0.45)]"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
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
      <ElevenLabsConvaiWidget />
    </SidebarProvider>
  );
}
