import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { User, Bell, Shield, Database, Palette } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";

const tabs = [
  { id: "profile", label: "Profile", icon: User },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "security", label: "Security", icon: Shield },
  { id: "data", label: "Data & Models", icon: Database },
  { id: "appearance", label: "Appearance", icon: Palette },
];

export default function Settings() {
  const [activeTab, setActiveTab] = useState("profile");
  const inputClass = "w-full h-12 rounded-2xl glass-strong px-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 transition-shadow";
  const labelClass = "block text-sm font-medium text-foreground mb-1.5";

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Settings</h1>
          <p className="text-muted-foreground text-sm">Manage your account and platform preferences</p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Tabs */}
          <div className="space-y-1">
            {tabs.map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`w-full flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  activeTab === t.id
                    ? "gradient-primary text-primary-foreground shadow-md"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                <t.icon className="w-4 h-4" />
                {t.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="lg:col-span-3">
            <motion.div key={activeTab} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="glass-strong rounded-3xl p-6">
              {activeTab === "profile" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-foreground">Profile Information</h3>
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center text-primary-foreground text-xl font-bold">BW</div>
                    <div>
                      <p className="font-semibold text-foreground">BlueWeave User</p>
                      <p className="text-sm text-muted-foreground">Researcher · Tamil Nadu</p>
                    </div>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div><label className={labelClass}>Full Name</label><input defaultValue="Dr. Priya Sharma" className={inputClass} /></div>
                    <div><label className={labelClass}>Email</label><input defaultValue="priya@blueweave.ai" className={inputClass} /></div>
                    <div><label className={labelClass}>Phone</label><input defaultValue="+91 98765 43210" className={inputClass} /></div>
                    <div><label className={labelClass}>Role</label><input defaultValue="Researcher" className={inputClass} /></div>
                  </div>
                  <Button variant="hero" size="lg">Save Changes</Button>
                </div>
              )}

              {activeTab === "notifications" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-foreground">Notification Preferences</h3>
                  {["Prediction completed", "Risk alerts", "Seasonal advisories", "Report generated", "New model version"].map(n => (
                    <div key={n} className="flex items-center justify-between glass rounded-2xl px-4 py-3">
                      <span className="text-sm text-foreground">{n}</span>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" defaultChecked className="sr-only peer" />
                        <div className="w-9 h-5 rounded-full bg-muted peer-checked:gradient-primary transition-colors after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-4 after:h-4 after:bg-white after:rounded-full after:transition-transform peer-checked:after:translate-x-4" />
                      </label>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === "security" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-foreground">Security Settings</h3>
                  <div><label className={labelClass}>Current Password</label><input type="password" placeholder="••••••••" className={inputClass} /></div>
                  <div><label className={labelClass}>New Password</label><input type="password" placeholder="••••••••" className={inputClass} /></div>
                  <div><label className={labelClass}>Confirm New Password</label><input type="password" placeholder="••••••••" className={inputClass} /></div>
                  <Button variant="hero" size="lg">Update Password</Button>
                </div>
              )}

              {activeTab === "data" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-foreground">Data & Model Settings</h3>
                  <div className="glass rounded-2xl p-4 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">Current Model Version</p>
                      <p className="text-xs text-muted-foreground">Gulf Suitability Model</p>
                    </div>
                    <span className="text-sm font-bold gradient-text">v1.1</span>
                  </div>
                  <div className="glass rounded-2xl p-4 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">Pro Mode</p>
                      <p className="text-xs text-muted-foreground">Enable advanced model parameters</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" className="sr-only peer" />
                      <div className="w-9 h-5 rounded-full bg-muted peer-checked:gradient-primary transition-colors after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-4 after:h-4 after:bg-white after:rounded-full after:transition-transform peer-checked:after:translate-x-4" />
                    </label>
                  </div>
                  <div className="glass rounded-2xl p-4 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">AI Explanation Toggle</p>
                      <p className="text-xs text-muted-foreground">Show detailed reasoning for predictions</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" defaultChecked className="sr-only peer" />
                      <div className="w-9 h-5 rounded-full bg-muted peer-checked:gradient-primary transition-colors after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-4 after:h-4 after:bg-white after:rounded-full after:transition-transform peer-checked:after:translate-x-4" />
                    </label>
                  </div>
                </div>
              )}

              {activeTab === "appearance" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-foreground">Appearance</h3>
                  <div className="glass rounded-2xl p-4">
                    <p className="text-sm font-medium text-foreground mb-3">Theme</p>
                    <div className="flex gap-3">
                      <button className="glass-strong rounded-xl px-4 py-2 text-sm font-medium text-foreground ring-2 ring-primary/30">Light</button>
                      <button className="glass rounded-xl px-4 py-2 text-sm font-medium text-muted-foreground">Dark</button>
                      <button className="glass rounded-xl px-4 py-2 text-sm font-medium text-muted-foreground">System</button>
                    </div>
                  </div>
                  <div className="glass rounded-2xl p-4 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">Confidence Score Badge</p>
                      <p className="text-xs text-muted-foreground">Show model confidence in predictions</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" defaultChecked className="sr-only peer" />
                      <div className="w-9 h-5 rounded-full bg-muted peer-checked:gradient-primary transition-colors after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-4 after:h-4 after:bg-white after:rounded-full after:transition-transform peer-checked:after:translate-x-4" />
                    </label>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
