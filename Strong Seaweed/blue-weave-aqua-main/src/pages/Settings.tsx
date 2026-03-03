import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { User, Bell, Shield, Database, Palette, Sparkles, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useMemo, useState } from "react";
import { api, type SettingsProfile, type UserPreferences } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

const tabs = [
  { id: "profile", label: "Profile", icon: User },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "security", label: "Security", icon: Shield },
  { id: "data", label: "Data & Models", icon: Database },
  { id: "appearance", label: "Appearance", icon: Palette },
];

const defaultPreferences: UserPreferences = {
  notifications: {
    predictionCompleted: true,
    riskAlerts: true,
    seasonalAdvisories: true,
    reportGenerated: true,
    newModelVersion: true,
  },
  dataModels: {
    proMode: false,
    aiExplanation: true,
  },
  appearance: {
    theme: "light",
    confidenceBadge: true,
  },
};

export default function Settings() {
  const { token } = useAuth();
  const [activeTab, setActiveTab] = useState("profile");
  const [profile, setProfile] = useState<SettingsProfile>({
    name: "",
    email: "",
    phone: "",
    state: "",
    role: "",
  });
  const [preferences, setPreferences] = useState<UserPreferences>(defaultPreferences);
  const [modelVersion, setModelVersion] = useState("Marine Core v2");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNext, setPwNext] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const inputClass =
    "w-full h-11 rounded-xl border border-[#c8ddeb] bg-[#f8fcff] px-3.5 text-sm text-slate-800 placeholder:text-slate-500 outline-none transition-all duration-200 focus:border-cyan-300 focus:ring-2 focus:ring-cyan-200/60";
  const labelClass = "mb-1.5 block text-sm font-medium text-[#123e63]";
  const toggle =
    "relative inline-flex h-6 w-11 items-center rounded-full bg-slate-300 transition-colors after:absolute after:left-0.5 after:h-5 after:w-5 after:rounded-full after:bg-white after:transition-transform peer-checked:bg-cyan-600 peer-checked:after:translate-x-5";

  const userInitial = useMemo(() => (profile.name || profile.email || "U").trim().charAt(0).toUpperCase(), [profile.name, profile.email]);
  const toUiError = (err: unknown, fallback: string) => {
    const msg = err instanceof Error ? err.message : fallback;
    const routeMissing = /api route unavailable|cannot (get|put)|\/api\/settings/i.test(msg);
    if (routeMissing) {
      return "Settings API is not deployed on this server yet. Pull latest backend and restart services.";
    }
    return msg || fallback;
  };

  const loadSettings = async (isInitial = false) => {
    if (!token) return;
    if (isInitial) setLoading(true);
    try {
      const out = await api.getSettings(token);
      setProfile(out.profile);
      setPreferences(out.preferences);
      setModelVersion(out.metadata?.modelVersion || "Marine Core v2");
      if (isInitial) setStatusText("");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Could not load settings.";
      const routeMissing = /api route unavailable|\/api\/settings|cannot get/i.test(msg);
      if (routeMissing) {
        try {
          const me = await api.me(token);
          setProfile((prev) => ({
            ...prev,
            name: me.user?.name || prev.name,
            email: me.user?.email || prev.email,
            role: me.user?.role || prev.role,
          }));
          setPreferences(defaultPreferences);
          setStatusText("Settings backend is not deployed on this server yet. Running in profile fallback mode.");
        } catch {
          if (isInitial) setStatusText("Could not load settings.");
        }
      } else if (isInitial) {
        setStatusText(msg);
      }
    } finally {
      if (isInitial) setLoading(false);
    }
  };

  useEffect(() => {
    void loadSettings(true);
    const interval = setInterval(() => {
      void loadSettings(false);
    }, 30000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const saveProfile = async () => {
    if (!token) return;
    setSaving(true);
    setStatusText("");
    try {
      const out = await api.updateSettingsProfile(
        {
          name: profile.name,
          phone: profile.phone,
          state: profile.state,
          role: profile.role,
        },
        token,
      );
      setProfile((prev) => ({ ...prev, ...out.profile }));
      setStatusText("Profile saved.");
    } catch (err) {
      setStatusText(toUiError(err, "Could not save profile."));
    } finally {
      setSaving(false);
    }
  };

  const savePassword = async () => {
    if (!token) return;
    if (!pwCurrent || !pwNext) {
      setStatusText("Enter current and new password.");
      return;
    }
    if (pwNext !== pwConfirm) {
      setStatusText("New password and confirm password do not match.");
      return;
    }

    setSaving(true);
    setStatusText("");
    try {
      await api.updatePassword({ currentPassword: pwCurrent, newPassword: pwNext }, token);
      setPwCurrent("");
      setPwNext("");
      setPwConfirm("");
      setStatusText("Password updated.");
    } catch (err) {
      setStatusText(toUiError(err, "Could not update password."));
    } finally {
      setSaving(false);
    }
  };

  const savePreferences = async (next: UserPreferences) => {
    if (!token) return;
    setPreferences(next);
    try {
      await api.updatePreferences(next, token);
      setStatusText("Preferences saved.");
    } catch (err) {
      setStatusText(toUiError(err, "Could not save preferences."));
    }
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="mx-auto max-w-6xl rounded-2xl border border-[#bfd8e8] bg-white/90 p-6 text-sm text-slate-600">
          Loading settings...
        </div>
      </DashboardLayout>
    );
  }

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

        {statusText ? (
          <div className="rounded-xl border border-cyan-200 bg-cyan-50/70 px-3 py-2 text-sm text-cyan-900">{statusText}</div>
        ) : null}

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
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#1DA1F2] to-[#0F2E47] text-xl font-bold text-white">{userInitial}</div>
                    <div>
                      <p className="font-semibold text-[#0f2e47]">{profile.name || "Akuara User"}</p>
                      <p className="text-sm text-slate-500">{profile.role || "Researcher"} {profile.state ? `- ${profile.state}` : ""}</p>
                    </div>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div><label className={labelClass}>Full Name</label><input value={profile.name} onChange={(e) => setProfile((p) => ({ ...p, name: e.target.value }))} className={inputClass} /></div>
                    <div><label className={labelClass}>Email</label><input value={profile.email} readOnly className={`${inputClass} bg-slate-100`} /></div>
                    <div><label className={labelClass}>Phone</label><input value={profile.phone} onChange={(e) => setProfile((p) => ({ ...p, phone: e.target.value }))} className={inputClass} /></div>
                    <div><label className={labelClass}>Role</label><input value={profile.role} onChange={(e) => setProfile((p) => ({ ...p, role: e.target.value }))} className={inputClass} /></div>
                  </div>
                  <Button variant="hero" size="lg" className="min-h-11" onClick={saveProfile} disabled={saving}>Save Changes</Button>
                </div>
              )}

              {activeTab === "notifications" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Notification Preferences</h3>
                  {[
                    ["Prediction completed", "predictionCompleted"],
                    ["Risk alerts", "riskAlerts"],
                    ["Seasonal advisories", "seasonalAdvisories"],
                    ["Report generated", "reportGenerated"],
                    ["New model version", "newModelVersion"],
                  ].map(([label, key]) => (
                    <div key={key} className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] px-4 py-3">
                      <span className="text-sm text-[#123e63]">{label}</span>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={Boolean(preferences.notifications[key as keyof UserPreferences["notifications"]])}
                          onChange={(e) =>
                            void savePreferences({
                              ...preferences,
                              notifications: {
                                ...preferences.notifications,
                                [key]: e.target.checked,
                              },
                            })
                          }
                          className="sr-only peer"
                        />
                        <div className={toggle} />
                      </label>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === "security" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Security Settings</h3>
                  <div><label className={labelClass}>Current Password</label><input type="password" placeholder="********" value={pwCurrent} onChange={(e) => setPwCurrent(e.target.value)} className={inputClass} /></div>
                  <div><label className={labelClass}>New Password</label><input type="password" placeholder="********" value={pwNext} onChange={(e) => setPwNext(e.target.value)} className={inputClass} /></div>
                  <div><label className={labelClass}>Confirm New Password</label><input type="password" placeholder="********" value={pwConfirm} onChange={(e) => setPwConfirm(e.target.value)} className={inputClass} /></div>
                  <Button variant="hero" size="lg" className="min-h-11" onClick={savePassword} disabled={saving}>Update Password</Button>
                </div>
              )}

              {activeTab === "data" && (
                <div className="space-y-5">
                  <h3 className="text-lg font-semibold text-[#0f2e47]">Data & Model Settings</h3>
                  <div className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">Current Model Version</p>
                      <p className="text-xs text-slate-500">Production inference profile</p>
                    </div>
                    <span className="rounded-full border border-cyan-200 bg-cyan-50 px-2.5 py-1 text-sm font-semibold text-cyan-800">{modelVersion}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">Pro Mode</p>
                      <p className="text-xs text-slate-500">Enable advanced model parameters</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preferences.dataModels.proMode}
                        onChange={(e) =>
                          void savePreferences({
                            ...preferences,
                            dataModels: { ...preferences.dataModels, proMode: e.target.checked },
                          })
                        }
                        className="sr-only peer"
                      />
                      <div className={toggle} />
                    </label>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">AI Explanation Toggle</p>
                      <p className="text-xs text-slate-500">Show detailed reasoning for predictions</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preferences.dataModels.aiExplanation}
                        onChange={(e) =>
                          void savePreferences({
                            ...preferences,
                            dataModels: { ...preferences.dataModels, aiExplanation: e.target.checked },
                          })
                        }
                        className="sr-only peer"
                      />
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
                      {(["light", "dark", "system"] as const).map((theme) => (
                        <button
                          key={theme}
                          onClick={() =>
                            void savePreferences({
                              ...preferences,
                              appearance: { ...preferences.appearance, theme },
                            })
                          }
                          className={`rounded-xl border px-4 py-2 text-sm font-medium ${
                            preferences.appearance.theme === theme
                              ? "border-cyan-200 bg-cyan-50 text-cyan-800 ring-2 ring-cyan-200"
                              : "border-[#c9deec] bg-white text-slate-600"
                          }`}
                        >
                          {theme[0].toUpperCase() + theme.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-3 rounded-2xl border border-[#c9deec] bg-[#f8fcff] p-4">
                    <div>
                      <p className="text-sm font-medium text-[#123e63]">Confidence Score Badge</p>
                      <p className="text-xs text-slate-500">Show model confidence in predictions</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preferences.appearance.confidenceBadge}
                        onChange={(e) =>
                          void savePreferences({
                            ...preferences,
                            appearance: { ...preferences.appearance, confidenceBadge: e.target.checked },
                          })
                        }
                        className="sr-only peer"
                      />
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
