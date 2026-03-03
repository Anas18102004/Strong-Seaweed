import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { User, Bell, Shield, Database, Palette, Sparkles, CheckCircle2 } from "lucide-react";
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
  const inputClass =
    "w-full h-11 rounded-xl border border-[#c8ddeb] bg-[#f8fcff] px-3.5 text-sm text-slate-800 placeholder:text-slate-500 outline-none transition-all duration-200 focus:border-cyan-300 focus:ring-2 focus:ring-cyan-200/60";
  const labelClass = "mb-1.5 block text-sm font-medium text-[#123e63]";
  const toggle =
    "relative inline-flex h-6 w-11 items-center rounded-full bg-slate-300 transition-colors after:absolute after:left-0.5 after:h-5 after:w-5 after:rounded-full after:bg-white after:transition-transform peer-checked:bg-cyan-600 peer-checked:after:translate-x-5";

  return (
    <DashboardLayout>
      <div className="mx-auto max-w-6xl space-y-5 sm:space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-3xl border border-[#a9cadf]/70 bg-[linear-gradient(135deg,rgba(17,66,98,0.95)_0%,rgba(20,89,132,0.88)_55%,rgba(14,57,91,0.95)_100%)] px-5 py-5 sm:px-6 sm:py-6 shadow-[0_24px_44px_-30px_rgba(11,41,63,0.85)]"
        >
          <div className="pointer-events-none absolute -right-14 -top-12 h-44 w-44 rounded-full bg-cyan-300/20 blur-3xl" />
          <div className="pointer-events-none absolute bottom-0 left-0 h-px w-full bg-gradient-to-r from-transparent via-cyan-200/60 to-transparent" />
          <div className="relative z-10 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.14em] text-cyan-100/80">Control Center</p>
              <h1 className="mt-1 text-2xl sm:text-3xl font-semibold text-white">Settings</h1>
              <p className="mt-2 max-w-2xl text-sm text-cyan-50/85">
                Configure account, security, alerts, and model behavior with production-grade defaults.
              </p>
            </div>
            <div className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-xs text-cyan-50">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
              Policy status: Operational
            </div>
          </div>
        </motion.div>

        <div className="grid gap-4 lg:grid-cols-[250px_minmax(0,1fr)] lg:gap-6">
          <aside className="rounded-2xl border border-[#bfd8e8] bg-white/95 p-2.5 shadow-[0_16px_30px_-24px_rgba(15,74,109,0.92)]">
            <div className="mb-2 hidden items-center gap-2 px-2.5 pt-1 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500 lg:flex">
              <Sparkles className="h-3.5 w-3.5 text-cyan-700" />
              Configuration
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1 lg:block lg:space-y-1 lg:overflow-visible lg:pb-0">
              {tabs.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`w-auto whitespace-nowrap rounded-xl px-4 py-2.5 text-sm font-medium transition-all lg:flex lg:w-full lg:items-center lg:gap-2.5 ${
                    activeTab === t.id
                      ? "bg-gradient-to-r from-[#1da1f2] to-[#0ea5e9] text-white shadow-[0_12px_22px_-16px_rgba(2,132,199,0.95)]"
                      : "text-slate-600 hover:bg-[#f0f7fc] hover:text-[#123e63]"
                  }`}
                >
                  <t.icon className="w-4 h-4" />
                  {t.label}
                </button>
              ))}
            </div>
          </aside>

          <section className="min-w-0">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-3xl border border-[#bfd8e8] bg-white/95 p-4 shadow-[0_24px_38px_-30px_rgba(15,74,109,0.88)] sm:p-6"
            >
              {activeTab === "profile" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Profile Information</h3>
                  <div className="mb-2 flex items-center gap-4 rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-3">
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#1DA1F2] to-[#0F2E47] text-xl font-bold text-white">BW</div>
                    <div>
                      <p className="font-semibold text-[#0f2e47]">BlueWeave User</p>
                      <p className="text-sm text-slate-500">Researcher - Tamil Nadu</p>
                    </div>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div><label className={labelClass}>Full Name</label><input defaultValue="Dr. Priya Sharma" className={inputClass} /></div>
                    <div><label className={labelClass}>Email</label><input defaultValue="priya@blueweave.ai" className={inputClass} /></div>
                    <div><label className={labelClass}>Phone</label><input defaultValue="+91 98765 43210" className={inputClass} /></div>
                    <div><label className={labelClass}>Role</label><input defaultValue="Researcher" className={inputClass} /></div>
                  </div>
                  <Button variant="hero" size="lg" className="min-h-11">Save Changes</Button>
                </div>
              )}

              {activeTab === "notifications" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Notification Preferences</h3>
                  {["Prediction completed", "Risk alerts", "Seasonal advisories", "Report generated", "New model version"].map((n) => (
                    <div key={n} className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] px-4 py-3">
                      <span className="text-sm text-[#123e63]">{n}</span>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" defaultChecked className="sr-only peer" />
                        <div className={toggle} />
                      </label>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === "security" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Security Settings</h3>
                  <div><label className={labelClass}>Current Password</label><input type="password" placeholder="********" className={inputClass} /></div>
                  <div><label className={labelClass}>New Password</label><input type="password" placeholder="********" className={inputClass} /></div>
                  <div><label className={labelClass}>Confirm New Password</label><input type="password" placeholder="********" className={inputClass} /></div>
                  <Button variant="hero" size="lg" className="min-h-11">Update Password</Button>
                </div>
              )}

              {activeTab === "data" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Data & Model Settings</h3>
                  <div className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">Current Model Version</p>
                      <p className="text-xs text-slate-500">Gulf Suitability Model</p>
                    </div>
                    <span className="rounded-full border border-cyan-200 bg-cyan-50 px-2.5 py-1 text-sm font-semibold text-cyan-800">v1.1</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">Pro Mode</p>
                      <p className="text-xs text-slate-500">Enable advanced model parameters</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" className="sr-only peer" />
                      <div className={toggle} />
                    </label>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">AI Explanation Toggle</p>
                      <p className="text-xs text-slate-500">Show detailed reasoning for predictions</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" defaultChecked className="sr-only peer" />
                      <div className={toggle} />
                    </label>
                  </div>
                </div>
              )}

              {activeTab === "appearance" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Appearance</h3>
                  <div className="rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <p className="mb-3 text-sm font-medium text-[#123e63]">Theme</p>
                    <div className="flex gap-3 flex-wrap">
                      <button className="rounded-xl border border-cyan-200 bg-cyan-50 px-4 py-2 text-sm font-semibold text-cyan-800 ring-2 ring-cyan-200">Light</button>
                      <button className="rounded-xl border border-[#c9deec] bg-white px-4 py-2 text-sm font-medium text-slate-600">Dark</button>
                      <button className="rounded-xl border border-[#c9deec] bg-white px-4 py-2 text-sm font-medium text-slate-600">System</button>
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-3 rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">Confidence Score Badge</p>
                      <p className="text-xs text-slate-500">Show model confidence in predictions</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" defaultChecked className="sr-only peer" />
                      <div className={toggle} />
                    </label>
                  </div>
                </div>
              )}
            </motion.div>
          </section>
        </div>
      </div>
    </DashboardLayout>
  );
}
