import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { AlertTriangle, Sun, CloudSun, CloudRain, Wind } from "lucide-react";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useEffect, useMemo, useState } from "react";

type QuarterRow = {
  name: string;
  months: string;
  icon: typeof Sun;
  risk: string;
  riskColor: string;
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
      const riskColor = risk === "Low" ? "text-ocean-500" : risk === "Moderate" ? "text-yellow-500" : "text-destructive";
      return {
        name: `${b.q} ${year}`,
        months: b.months,
        icon: b.icon,
        risk,
        riskColor,
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
      <div className="max-w-5xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Seasonal Forecast</h1>
          <p className="text-muted-foreground text-sm">Realtime quarterly outlook from your live prediction history</p>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-strong rounded-2xl p-4 flex items-start gap-3 border-l-4 border-yellow-400">
          <AlertTriangle className="w-5 h-5 text-yellow-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-foreground">Data-driven seasonal risk</p>
            <p className="text-xs text-muted-foreground">This page now updates from your real model outcomes instead of fixed static assumptions.</p>
          </div>
        </motion.div>

        <div className="space-y-4">
          {quarters.map((q, i) => (
            <motion.div key={q.name} initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 + i * 0.08 }} className="glass-strong rounded-3xl p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-11 h-11 rounded-2xl gradient-primary flex items-center justify-center">
                    <q.icon className="w-5 h-5 text-primary-foreground" />
                  </div>
                  <div>
                    <h3 className="font-bold text-foreground">{q.name}</h3>
                    <p className="text-xs text-muted-foreground">{q.months}</p>
                  </div>
                </div>
                <span className={`text-sm font-bold ${q.riskColor}`}>{q.risk} Risk</span>
              </div>
              <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{q.summary}</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {q.conditions.map((c) => (
                  <div key={c.label} className="glass rounded-xl p-3 text-center">
                    <p className={`text-sm font-semibold ${c.color}`}>{c.value}</p>
                    <p className="text-xs text-muted-foreground">{c.label}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}