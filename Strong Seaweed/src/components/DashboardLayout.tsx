import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Bell, Waves } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full gradient-bg">
        <AppSidebar />
        <div className="flex-1 flex flex-col">
          <header className="h-14 flex items-center justify-between border-b border-border/40 px-4 glass">
            <SidebarTrigger className="text-muted-foreground" />
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-muted-foreground glass rounded-full px-3 py-1">v1.1 Gulf Model</span>
              <button className="w-9 h-9 rounded-xl glass flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors">
                <Bell className="w-4 h-4" />
              </button>
              <div className="w-9 h-9 rounded-xl gradient-primary flex items-center justify-center text-primary-foreground text-xs font-bold">
                BW
              </div>
            </div>
          </header>
          <main className="flex-1 p-6 overflow-auto">{children}</main>
        </div>
      </div>
    </SidebarProvider>
  );
}
