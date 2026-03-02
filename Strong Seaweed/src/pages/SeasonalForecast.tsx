import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { CloudRain, Wind, Thermometer, AlertTriangle, Sun, CloudSun } from "lucide-react";

const quarters = [
  {
    name: "Q1 2026 — Winter",
    months: "Jan – Mar",
    icon: Sun,
    risk: "Low",
    riskColor: "text-ocean-500",
    summary: "Optimal cultivation window. Calm seas, stable temperatures (27–29°C), and minimal storm activity across most regions.",
    conditions: [
      { label: "Cyclone Risk", value: "Very Low", color: "text-ocean-500" },
      { label: "Avg SST", value: "27.8°C", color: "text-foreground" },
      { label: "Rainfall", value: "Low", color: "text-ocean-500" },
      { label: "Current Speed", value: "0.2–0.4 m/s", color: "text-foreground" },
    ],
  },
  {
    name: "Q2 2026 — Pre-Monsoon",
    months: "Apr – Jun",
    icon: CloudSun,
    risk: "Moderate",
    riskColor: "text-yellow-500",
    summary: "Rising temperatures and increasing humidity. Good window for harvest completion. Watch for early monsoon onset in late May.",
    conditions: [
      { label: "Cyclone Risk", value: "Moderate", color: "text-yellow-500" },
      { label: "Avg SST", value: "29.5°C", color: "text-foreground" },
      { label: "Rainfall", value: "Increasing", color: "text-yellow-500" },
      { label: "Current Speed", value: "0.3–0.5 m/s", color: "text-foreground" },
    ],
  },
  {
    name: "Q3 2026 — Monsoon",
    months: "Jul – Sep",
    icon: CloudRain,
    risk: "High",
    riskColor: "text-destructive",
    summary: "Heavy rainfall, rough seas, and cyclone risk. Most cultivation activities should be paused. Focus on infrastructure maintenance.",
    conditions: [
      { label: "Cyclone Risk", value: "High", color: "text-destructive" },
      { label: "Avg SST", value: "28.2°C", color: "text-foreground" },
      { label: "Rainfall", value: "Heavy", color: "text-destructive" },
      { label: "Current Speed", value: "0.5–1.0 m/s", color: "text-foreground" },
    ],
  },
  {
    name: "Q4 2026 — Post-Monsoon",
    months: "Oct – Dec",
    icon: Wind,
    risk: "Moderate",
    riskColor: "text-yellow-500",
    summary: "Retreating monsoon with improving conditions. Best time to begin new cultivation cycles. Northeast monsoon affects east coast.",
    conditions: [
      { label: "Cyclone Risk", value: "Moderate", color: "text-yellow-500" },
      { label: "Avg SST", value: "28.0°C", color: "text-foreground" },
      { label: "Rainfall", value: "Moderate", color: "text-yellow-500" },
      { label: "Current Speed", value: "0.3–0.5 m/s", color: "text-foreground" },
    ],
  },
];

export default function SeasonalForecast() {
  return (
    <DashboardLayout>
      <div className="max-w-5xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Seasonal Forecast</h1>
          <p className="text-muted-foreground text-sm">Quarterly cultivation outlook and climate risk assessment — 2026</p>
        </motion.div>

        {/* Alert banner */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-strong rounded-2xl p-4 flex items-start gap-3 border-l-4 border-yellow-400"
        >
          <AlertTriangle className="w-5 h-5 text-yellow-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-foreground">Early Monsoon Warning</p>
            <p className="text-xs text-muted-foreground">IMD projections suggest monsoon onset may arrive 7–10 days early in 2026. Plan harvest cycles accordingly.</p>
          </div>
        </motion.div>

        {/* Quarterly cards */}
        <div className="space-y-4">
          {quarters.map((q, i) => (
            <motion.div
              key={q.name}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 + i * 0.08 }}
              className="glass-strong rounded-3xl p-6"
            >
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
                {q.conditions.map(c => (
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
