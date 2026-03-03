import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { TrendingUp, Droplets, Activity, MessageSquare, Cpu, ArrowRight, Sparkles, Radar } from "lucide-react";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [apiUp, setApiUp] = useState(false);
  const [sessionsCount, setSessionsCount] = useState(0);
  const [submissions, setSubmissions] = useState<PredictionSubmissionItem[]>([]);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      if (!token) return;
      try {
        const [health, pred, sessions] = await Promise.all([
          api.health(),
          api.mySubmissions(token, 20),
          api.getChatSessions(token),
        ]);
        if (!mounted) return;
        setApiUp((health.status || "").toLowerCase() === "ok");
        setSubmissions(pred.submissions || []);
        setSessionsCount((sessions.sessions || []).length);
      } catch {
        if (!mounted) return;
        setApiUp(false);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void load();
    const t = setInterval(load, 30000);
    return () => {
      mounted = false;
      clearInterval(t);
    };
  }, [token]);

  const topSpecies = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const s of submissions) {
      const name = s.bestSpecies?.displayName || "Unknown";
      counts[name] = (counts[name] || 0) + 1;
    }
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return sorted[0]?.[0] || "-";
  }, [submissions]);

  const avgScore = useMemo(() => {
    const vals = submissions
      .map((s) => s.bestSpecies?.probabilityPercent)
      .filter((v): v is number => typeof v === "number");
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [submissions]);

  const stats = [
    { label: "Live Predictions", value: String(submissions.length), icon: Activity, change: "+12% this cycle" },
    {
      label: "Top Suggested Species",
      value: topSpecies,
      icon: Droplets,
      change: avgScore === null ? "No score yet" : `${avgScore.toFixed(1)}% avg confidence`,
    },
    { label: "Backend API", value: apiUp ? "Online" : "Offline", icon: Cpu, change: apiUp ? "Realtime mode active" : "Check service health" },
    { label: "Chat Sessions", value: String(sessionsCount), icon: MessageSquare, change: "Assistant history retained" },
  ];

  const recentPredictions = submissions.slice(0, 8).map((s) => {
    const p = s.bestSpecies?.probabilityPercent ?? 0;
    const status = p >= 80 ? "Optimal" : p >= 65 ? "Good" : p >= 50 ? "Moderate" : "Fair";
    return {
      location: s.locationName || `${s.lat.toFixed(3)}, ${s.lon.toFixed(3)}`,
      species: s.bestSpecies?.displayName || "Unknown",
      score: Math.max(0, Math.min(100, Math.round(p))),
      status,
    };
  });

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto">
        <div className="ocean-page-shell">
          <motion.div className="ocean-page-header" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="ocean-page-kicker">Overview / Operations</p>
                <h1 className="ocean-title-glow mt-2">
                  Marine <span className="ocean-title-highlight">Intelligence</span> Dashboard
                </h1>
                <p className="mt-3 max-w-2xl text-sm text-[#E6F5FF]">
                  Unified live visibility into prediction performance, model confidence, and advisor activity across your ocean farming operations.
                </p>
                <div className="ocean-header-line" />
              </div>
              <div className="flex items-center gap-2.5">
                <button
                  onClick={() => navigate("/predict")}
                  className="ocean-glass-card rounded-xl px-3.5 py-2 text-xs font-semibold text-[#EAF7FF] transition-colors hover:text-white"
                >
                  New Prediction
                </button>
                <button
                  onClick={() => navigate("/chat")}
                  className="ocean-shine-btn rounded-xl bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] px-3.5 py-2 text-xs font-semibold text-white shadow-[0_10px_24px_-14px_rgba(14,165,233,0.9)]"
                >
                  Open Copilot
                </button>
              </div>
            </div>
            <div className="mt-5 inline-flex items-center gap-2 rounded-full border border-emerald-200/20 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-100">
              <span className="ocean-breathe-dot inline-flex h-2 w-2 rounded-full bg-emerald-300" />
              Live pulse active
            </div>
          </motion.div>

          <div className="px-4 pb-4 sm:px-6 sm:pb-6 space-y-6">
            <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
              {stats.map((s, i) => (
                <motion.div
                  key={s.label}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08, duration: 0.4 }}
                  className="ocean-glass-card ocean-kpi-card rounded-[20px] p-5"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="relative inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400/30 to-blue-500/30">
                      <div className="absolute inset-0 rounded-2xl bg-cyan-300/30 blur-lg" />
                      <s.icon className="relative z-10 h-5 w-5 text-cyan-100" />
                    </div>
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-100">
                      <TrendingUp className="h-3 w-3" />
                      {s.change}
                    </span>
                  </div>
                  <p className="mt-4 text-3xl font-bold text-white tracking-tight">{s.value}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.13em] text-[#A7CCE4]">{s.label}</p>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.4 }}
              className="ocean-glass-card rounded-[22px] p-4 sm:p-6"
            >
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="ocean-page-kicker">Prediction Feed</p>
                  <h2 className="mt-1 text-xl font-semibold text-white">Recent Predictions</h2>
                </div>
                <Radar className="h-5 w-5 text-cyan-200 ocean-weather-float" />
              </div>

              {recentPredictions.length > 0 ? (
                <div className="overflow-x-auto rounded-2xl border border-white/10 bg-white/[0.03]">
                  <table className="w-full text-sm">
                    <thead className="bg-white/[0.06]">
                      <tr className="text-left text-[#E3F2FF]">
                        <th className="px-4 py-3 font-semibold rounded-l-xl">Location</th>
                        <th className="px-4 py-3 font-semibold">Species</th>
                        <th className="px-4 py-3 font-semibold">Score</th>
                        <th className="px-4 py-3 font-semibold rounded-r-xl">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentPredictions.map((p, i) => (
                        <tr key={i} className="border-t border-white/5 text-[#DDEEFF] hover:bg-white/[0.04] transition-colors">
                          <td className="px-4 py-3 font-medium">{p.location}</td>
                          <td className="px-4 py-3 italic text-[#D2E8F8]">{p.species}</td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <div className="h-2 w-20 rounded-full bg-white/10 overflow-hidden">
                                <div className="h-full rounded-full bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9]" style={{ width: `${p.score}%` }} />
                              </div>
                              <span className="text-xs font-semibold text-white">{p.score}%</span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className="inline-flex rounded-full border border-cyan-200/20 bg-cyan-400/10 px-2.5 py-1 text-[11px] font-semibold text-cyan-100">
                              {p.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                !loading && (
                  <div className="ocean-map-placeholder rounded-2xl p-10 text-center">
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-cyan-100/20 bg-cyan-300/10">
                      <Sparkles className="h-6 w-6 text-cyan-100" />
                    </div>
                    <p className="text-white text-base font-semibold">No prediction stream yet</p>
                    <p className="mt-2 text-sm text-[#DDEEFF]">Start your first prediction to unlock live region intelligence.</p>
                    <button
                      onClick={() => navigate("/predict")}
                      className="mt-5 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] px-4 py-2 text-sm font-semibold text-white"
                    >
                      Run First Prediction
                      <ArrowRight className="h-4 w-4" />
                    </button>
                  </div>
                )
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
