import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { Award, AlertTriangle, CloudRain, Wind, ChevronDown, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";

const speciesResults = [
  { name: "Kappaphycus alvarezii", score: 87, color: "#0077FF", best: true },
  { name: "Gracilaria edulis", score: 81, color: "#00A8E8" },
  { name: "Ulva lactuca", score: 74, color: "#00C6FF" },
  { name: "Sargassum wightii", score: 62, color: "#7DD6FF" },
];

const risks = [
  { label: "Cyclone Risk", value: "Low", icon: Wind, color: "text-ocean-500" },
  { label: "Monsoon Impact", value: "Moderate", icon: CloudRain, color: "text-yellow-500" },
  { label: "Environmental Stress", value: "Low", icon: AlertTriangle, color: "text-ocean-400" },
];

function ScoreRing({ score, color, size = 80 }: { score: number; color: string; size?: number }) {
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;

  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="hsl(var(--muted))" strokeWidth="6" />
      <motion.circle
        cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={circ}
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
      />
    </svg>
  );
}

export default function ResultsPage() {
  const [showWhy, setShowWhy] = useState(false);
  const best = speciesResults[0];

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Prediction Results</h1>
          <p className="text-muted-foreground text-sm">Ecological suitability analysis — Gulf of Mannar</p>
        </motion.div>

        {/* Species Cards */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {speciesResults.map((s, i) => (
            <motion.div
              key={s.name}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 + i * 0.1 }}
              className={`glass-strong rounded-3xl p-5 text-center ${s.best ? "ring-2 ring-primary/30 glow-sm" : ""}`}
            >
              <div className="flex justify-center mb-3 relative">
                <ScoreRing score={s.score} color={s.color} />
                <span className="absolute inset-0 flex items-center justify-center text-xl font-bold text-foreground rotate-0">
                  {s.score}%
                </span>
              </div>
              <p className="text-sm font-medium text-foreground">{s.name}</p>
              {s.best && (
                <span className="inline-flex items-center gap-1 mt-2 text-xs font-medium gradient-primary text-primary-foreground px-2.5 py-0.5 rounded-full">
                  <Award className="w-3 h-3" /> Best Match
                </span>
              )}
            </motion.div>
          ))}
        </div>

        {/* AI Recommendation */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-strong rounded-3xl p-6 border-l-4 border-primary"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-2xl gradient-primary flex items-center justify-center shrink-0">
              <Award className="w-6 h-6 text-primary-foreground" />
            </div>
            <div className="flex-1">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">AI Recommendation</p>
              <h3 className="text-xl font-bold text-foreground mb-2">
                Best Choice: <span className="gradient-text">{best.name}</span>
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                Optimal depth + salinity + current velocity alignment detected. Estimated harvest cycle: <strong className="text-foreground">52 days</strong>. Risk level: <strong className="text-ocean-500">Low</strong>.
              </p>
              <button
                onClick={() => setShowWhy(!showWhy)}
                className="flex items-center gap-1 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
              >
                <ChevronDown className={`w-4 h-4 transition-transform ${showWhy ? "rotate-180" : ""}`} />
                Why this species?
              </button>
              {showWhy && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="text-sm text-muted-foreground mt-3 pl-5 border-l-2 border-border leading-relaxed"
                >
                  Kappaphycus alvarezii thrives in waters with 28–30°C temperatures and 30–35 ppt salinity, both of which match your site. The moderate current velocity (0.3 m/s) supports nutrient flow without mechanical stress. Historical yield data from Gulf of Mannar confirms high productivity in similar conditions.
                </motion.p>
              )}
            </div>
          </div>
        </motion.div>

        {/* Risk Panel */}
        <div className="grid sm:grid-cols-3 gap-4">
          {risks.map((r, i) => (
            <motion.div
              key={r.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.08 }}
              className="glass-strong rounded-3xl p-5 text-center"
            >
              <r.icon className={`w-6 h-6 mx-auto mb-2 ${r.color}`} />
              <p className="text-lg font-bold text-foreground">{r.value}</p>
              <p className="text-xs text-muted-foreground">{r.label}</p>
            </motion.div>
          ))}
        </div>

        <div className="flex justify-end">
          <Button variant="hero-outline" size="lg">
            <Download className="w-4 h-4" /> Export Report
          </Button>
        </div>
      </div>
    </DashboardLayout>
  );
}
