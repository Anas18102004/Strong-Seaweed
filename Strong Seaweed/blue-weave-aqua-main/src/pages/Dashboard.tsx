import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { TrendingUp, Droplets, Activity, MessageSquare, Cpu } from "lucide-react";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

export default function Dashboard() {
  const { token } = useAuth();
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
    { label: "Live Predictions", value: String(submissions.length), icon: Activity, change: "from your recent runs" },
    {
      label: "Top Suggested Species",
      value: topSpecies,
      icon: Droplets,
      change: avgScore === null ? "no score yet" : `${avgScore.toFixed(1)}% avg`,
    },
    { label: "Backend API", value: apiUp ? "Online" : "Offline", icon: Cpu, change: apiUp ? "live realtime mode" : "check backend" },
    { label: "Chat Sessions", value: String(sessionsCount), icon: MessageSquare, change: "assistant history saved" },
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
      <div className="max-w-6xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Dashboard</h1>
          <p className="text-muted-foreground text-sm">Live cultivation intelligence from your real backend data</p>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
              className="glass-strong rounded-3xl p-5"
            >
              <div className="flex items-start justify-between mb-3 gap-2">
                <div className="w-10 h-10 rounded-2xl gradient-primary flex items-center justify-center shrink-0">
                  <s.icon className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="text-xs text-muted-foreground font-medium text-right">{s.change}</span>
              </div>
              <p className="text-2xl font-bold text-foreground truncate">{s.value}</p>
              <p className="text-sm text-muted-foreground">{s.label}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="glass-strong rounded-3xl p-6"
        >
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-semibold text-foreground">Recent Predictions</h2>
            <TrendingUp className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted-foreground border-b border-border/50">
                  <th className="pb-3 font-medium">Location</th>
                  <th className="pb-3 font-medium">Species</th>
                  <th className="pb-3 font-medium">Score</th>
                  <th className="pb-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {recentPredictions.map((p, i) => (
                  <tr key={i} className="border-b border-border/30 last:border-0">
                    <td className="py-3 font-medium text-foreground">{p.location}</td>
                    <td className="py-3 text-muted-foreground italic">{p.species}</td>
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div className="h-full rounded-full gradient-primary" style={{ width: `${p.score}%` }} />
                        </div>
                        <span className="font-semibold gradient-text">{p.score}%</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <span
                        className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                          p.status === "Optimal"
                            ? "bg-ocean-100 text-ocean-600"
                            : p.status === "Good"
                              ? "bg-ocean-50 text-ocean-500"
                              : p.status === "Moderate"
                                ? "bg-secondary text-secondary-foreground"
                                : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {p.status}
                      </span>
                    </td>
                  </tr>
                ))}
                {!loading && recentPredictions.length === 0 && (
                  <tr>
                    <td className="py-6 text-muted-foreground" colSpan={4}>
                      No submissions yet. Run your first prediction from "Check My Location".
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}