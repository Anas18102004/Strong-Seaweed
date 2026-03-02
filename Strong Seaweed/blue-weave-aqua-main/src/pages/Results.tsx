import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { Award, AlertTriangle, CloudRain, Wind, ChevronDown, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import type { SpeciesPredictionResponse } from "@/lib/api";

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
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={circ}
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
      />
    </svg>
  );
}

function modelBadgeFromReason(reason: string, pending: boolean): { label: string; className: string } {
  if (pending && reason === "out_of_coverage") {
    return { label: "Out of Coverage", className: "bg-amber-500/20 text-amber-200" };
  }
  if (pending && reason === "model_not_trained") {
    return { label: "Pending", className: "bg-slate-500/20 text-slate-200" };
  }
  if (reason === "dedicated_production_model") {
    return { label: "Production", className: "bg-emerald-500/20 text-emerald-200" };
  }
  if (reason.startsWith("genus_proxy_")) {
    return { label: "Genus Proxy", className: "bg-cyan-500/20 text-cyan-100" };
  }
  if (pending) {
    return { label: "Unavailable", className: "bg-slate-500/20 text-slate-200" };
  }
  return { label: "Model", className: "bg-blue-500/20 text-blue-100" };
}

function reasonText(reason: string): string {
  switch (reason) {
    case "dedicated_production_model":
      return "Dedicated, validated species model.";
    case "genus_proxy_positive":
      return "Proxy model indicates ecological suitability.";
    case "genus_proxy_negative":
      return "Proxy model indicates low suitability.";
    case "out_of_coverage":
      return "Point is outside this species model coverage.";
    case "model_not_trained":
      return "Species model is not trained yet.";
    default:
      return reason.replace(/_/g, " ");
  }
}

export default function ResultsPage() {
  const [showWhy, setShowWhy] = useState(false);
  const routerLocation = useLocation();
  const prediction = (routerLocation.state as { prediction?: SpeciesPredictionResponse } | null)?.prediction;

  if (!prediction) {
    return (
      <DashboardLayout>
        <div className="max-w-2xl mx-auto glass-strong rounded-3xl p-8 text-center space-y-3">
          <h1 className="text-2xl font-bold text-foreground">No prediction available</h1>
          <p className="text-sm text-muted-foreground">Run a location check first to view suitability results.</p>
          <Button asChild variant="hero">
            <Link to="/predict">Go to Check My Location</Link>
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  const colorByIndex = ["#0077FF", "#00A8E8", "#00C6FF", "#7DD6FF"];
  const speciesResults = prediction.species.map((species, idx) => ({
    name: species.displayName,
    score: species.probabilityPercent == null ? null : Math.round(species.probabilityPercent),
    color: colorByIndex[idx % colorByIndex.length],
    best: prediction.bestSpecies?.speciesId === species.speciesId,
    pending: !species.ready || species.probabilityPercent == null,
    reason: species.reason,
    badge: modelBadgeFromReason(species.reason, !species.ready || species.probabilityPercent == null),
  }));

  const best = speciesResults.find((item) => item.best) ?? speciesResults[0];

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Prediction Results</h1>
          <p className="text-muted-foreground text-sm">Ecological suitability analysis by species model stack</p>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {speciesResults.map((s, i) => (
            <motion.div
              key={s.name}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 + i * 0.1 }}
              className={`glass-strong rounded-3xl p-5 text-center ${s.best ? "ring-2 ring-primary/30 glow-sm" : ""}`}
            >
              {s.pending ? (
                <div className="h-20 flex flex-col items-center justify-center text-xs text-muted-foreground gap-1">
                  <span>{s.badge.label}</span>
                  <span className="text-[11px]">{reasonText(s.reason)}</span>
                </div>
              ) : (
                <div className="flex justify-center mb-3 relative">
                  <ScoreRing score={s.score ?? 0} color={s.color} />
                  <span className="absolute inset-0 flex items-center justify-center text-xl font-bold text-foreground rotate-0">{s.score}%</span>
                </div>
              )}
              <p className="text-sm font-medium text-foreground">{s.name}</p>
              <span className={`inline-flex mt-2 text-[11px] px-2 py-0.5 rounded-full ${s.badge.className}`}>{s.badge.label}</span>
              {s.best && !s.pending && (
                <span className="inline-flex items-center gap-1 mt-2 text-xs font-medium gradient-primary text-primary-foreground px-2.5 py-0.5 rounded-full">
                  <Award className="w-3 h-3" /> Best Match
                </span>
              )}
            </motion.div>
          ))}
        </div>

        {prediction.warnings.length > 0 && (
          <div className="glass rounded-2xl p-4 border border-amber-400/30 text-amber-100 text-sm">
            <p className="font-semibold mb-1">Model Warnings</p>
            <ul className="space-y-1">
              {prediction.warnings.map((w) => (
                <li key={w}>- {w.replace(/_/g, " ")}</li>
              ))}
            </ul>
          </div>
        )}

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
                Best species score at this site is <strong className="text-foreground">{best.score ?? 0}%</strong>.
                {prediction.nearestGrid && (
                  <>
                    {" "}Nearest model grid is <strong className="text-foreground">{prediction.nearestGrid.distance_km.toFixed(2)} km</strong> away.
                  </>
                )}
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
                  This output combines one dedicated production model (Kappaphycus) and additional genus-proxy species models.
                  Always field-verify locations before deployment decisions.
                </motion.p>
              )}
            </div>
          </div>
        </motion.div>

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
