import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { AlertTriangle, Sun, CloudSun, CloudRain, Wind, Sparkles } from "lucide-react";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useEffect, useMemo, useState } from "react";

type QuarterRow = {
  name: string;
  months: string;
  icon: typeof Sun;
  risk: string;
  riskColor: string;
  riskValue: number;
  summary: string;
  conditions: Array<{ label: string; value: string; color: string }>;
};

function quarterLabel(month: number) {
  if (month <= 3) return "Q1";
  if (month <= 6) return "Q2";
  if (month <= 9) return "Q3";
  return "Q4";
}

export default function SeasonalForecast() {
  const { token } = useAuth();
  const [rows, setRows] = useState<PredictionSubmissionItem[]>([]);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      if (!token) return;
      try {
        const out = await api.mySubmissions(token, 200);
        if (!mounted) return;
        setRows(out.submissions || []);
      } catch {
        if (!mounted) return;
        setRows([]);
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, [token]);

  const quarters = useMemo<QuarterRow[]>(() => {
    const now = new Date();
    const year = now.getFullYear();
    const map: Record<string, number[]> = { Q1: [], Q2: [], Q3: [], Q4: [] };
    for (const r of rows) {
      const d = new Date(r.createdAt);
      const q = quarterLabel(d.getMonth() + 1);
      const p = r.bestSpecies?.probabilityPercent;
      if (typeof p === "number") map[q].push(p);
    }

    const base: Array<{ q: "Q1" | "Q2" | "Q3" | "Q4"; months: string; icon: typeof Sun }> = [
      { q: "Q1", months: "Jan - Mar", icon: Sun },
      { q: "Q2", months: "Apr - Jun", icon: CloudSun },
      { q: "Q3", months: "Jul - Sep", icon: CloudRain },
      { q: "Q4", months: "Oct - Dec", icon: Wind },
    ];

    return base.map((b) => {
      const vals = map[b.q];
      const avg = vals.length ? vals.reduce((a, c) => a + c, 0) / vals.length : 0;
      const risk = avg >= 75 ? "Low" : avg >= 55 ? "Moderate" : "High";
      const riskColor = risk === "Low" ? "text-emerald-300" : risk === "Moderate" ? "text-amber-300" : "text-rose-300";
      const riskValue = risk === "Low" ? 28 : risk === "Moderate" ? 62 : 88;
      return {
        name: `${b.q} ${year}`,
        months: b.months,
        icon: b.icon,
        risk,
        riskColor,
        riskValue,
        summary: vals.length
          ? `Based on ${vals.length} recent runs, average suitability is ${avg.toFixed(1)}%.`
          : "No recent prediction data yet for this quarter.",
        conditions: [
          { label: "Avg Suitability", value: vals.length ? `${avg.toFixed(1)}%` : "-", color: "text-foreground" },
          { label: "Predictions", value: String(vals.length), color: "text-foreground" },
          { label: "Risk Level", value: risk, color: riskColor },
          { label: "Action", value: risk === "Low" ? "Expand" : risk === "Moderate" ? "Monitor" : "Survey", color: riskColor },
        ],
      };
    });
  }, [rows]);

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto">
        <div className="ocean-page-shell">
          <motion.div className="ocean-page-header" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <p className="ocean-page-kicker">Climate / Forecast Engine</p>
            <h1 className="ocean-title-glow mt-2">
              Weather & <span className="ocean-title-highlight">Season</span> Intelligence
            </h1>
            <p className="mt-3 max-w-2xl text-sm text-[#CFE9FF]/80">Quarterly risk outlook generated from your real prediction history and suitability behavior over time.</p>
            <div className="ocean-header-line" />
            <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-amber-200/20 bg-amber-300/10 px-3 py-1 text-xs text-amber-100">
              <AlertTriangle className="h-3.5 w-3.5" />
              Data-driven seasonal risk
            </div>
          </motion.div>

          <div className="px-4 pb-4 sm:px-6 sm:pb-6 space-y-5">
            <div className="overflow-x-auto">
              <div className="flex min-w-[980px] gap-4">
                {quarters.map((q, i) => (
                  <motion.div
                    key={q.name}
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.12 + i * 0.08 }}
                    className={`ocean-glass-card rounded-[22px] p-5 w-[240px] shrink-0 ${q.risk === "High" ? "shadow-[0_0_0_1px_rgba(239,68,68,0.22),0_0_28px_rgba(239,68,68,0.18)]" : ""}`}
                  >
                    <div className="mb-4 flex items-start justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400/30 to-blue-500/35">
                          <q.icon className="h-5 w-5 text-cyan-100 ocean-weather-float" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-white">{q.name}</p>
                          <p className="text-xs text-[#7FA9C4]">{q.months}</p>
                        </div>
                      </div>
                      <span className={`text-xs font-semibold ${q.riskColor}`}>{q.risk} Risk</span>
                    </div>

                    <p className="min-h-[56px] text-xs text-[#CFE9FF]/75 leading-relaxed">{q.summary}</p>

                    <div className="mt-3">
                      <div className="mb-1.5 flex items-center justify-between text-[11px] text-[#7FA9C4]">
                        <span>Risk Level</span>
                        <span className={q.riskColor}>{q.riskValue}%</span>
                      </div>
                      <div className="ocean-risk-meter">
                        <div className="ocean-risk-fill" style={{ width: `${q.riskValue}%` }} />
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-2">
                      {q.conditions.map((c) => (
                        <div key={c.label} className="rounded-xl border border-white/10 bg-white/[0.04] px-2.5 py-2">
                          <p className={`text-sm font-semibold ${c.color}`}>{c.value}</p>
                          <p className="text-[11px] text-[#7FA9C4]">{c.label}</p>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="ocean-glass-card rounded-2xl p-4 flex items-center gap-2.5">
              <Sparkles className="h-4 w-4 text-cyan-200" />
              <p className="text-sm text-[#CFE9FF]/85">Use this timeline to plan deployments, harvest windows, and mitigation actions before high-risk quarters.</p>
            </motion.div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
