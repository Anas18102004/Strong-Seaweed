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

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full gradient-bg">
        <AppSidebar />
        <div className="flex-1 flex flex-col">
          <header className="min-h-14 flex items-center justify-between border-b border-border/40 px-3 sm:px-4 py-2 glass">
            <SidebarTrigger className="text-muted-foreground" />
            <div className="flex items-center gap-3">
              <span className="hidden sm:inline-flex text-xs font-medium text-muted-foreground glass rounded-full px-3 py-1">v1.1 Gulf Model</span>
              <button className="w-9 h-9 rounded-xl glass flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors">
                <Bell className="w-4 h-4" />
              </button>
              <div className="text-xs text-muted-foreground hidden sm:block">{user?.name || user?.email || "User"}</div>
              <button
                onClick={() => {
                  signOut();
                  navigate("/signin");
                }}
                className="w-9 h-9 rounded-xl glass flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
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
