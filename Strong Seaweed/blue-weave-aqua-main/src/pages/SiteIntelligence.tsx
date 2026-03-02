import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useMemo, useState, useEffect } from "react";
import { MapPin, Droplets, Thermometer } from "lucide-react";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

type RegionItem = {
  name: string;
  score: number;
  level: "high" | "moderate" | "low";
  species: string;
  temp: string;
  salinity: string;
  advisory: string;
};

const levelColors = {
  high: "bg-ocean-500",
  moderate: "bg-ocean-300",
  low: "bg-muted-foreground/50",
};

const levelLabels = {
  high: "Highly Suitable",
  moderate: "Moderate",
  low: "Low Suitability",
};

function regionName(s: PredictionSubmissionItem) {
  if (s.locationName) return s.locationName;
  return `${s.lat.toFixed(2)}, ${s.lon.toFixed(2)}`;
}

export default function SiteIntelligence() {
  const { token } = useAuth();
  const [rows, setRows] = useState<PredictionSubmissionItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      if (!token) return;
      try {
        const out = await api.mySubmissions(token, 100);
        if (!mounted) return;
        setRows(out.submissions || []);
      } catch {
        if (!mounted) return;
        setRows([]);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, [token]);

  const regions = useMemo(() => {
    const by: Record<string, { scores: number[]; species: Record<string, number> }> = {};
    for (const r of rows) {
      const key = regionName(r);
      by[key] = by[key] || { scores: [], species: {} };
      const p = r.bestSpecies?.probabilityPercent;
      if (typeof p === "number") by[key].scores.push(p);
      const sp = r.bestSpecies?.displayName || "Unknown";
      by[key].species[sp] = (by[key].species[sp] || 0) + 1;
    }
    const out: RegionItem[] = Object.entries(by).map(([name, info]) => {
      const avg = info.scores.length ? info.scores.reduce((a, b) => a + b, 0) / info.scores.length : 0;
      const species = Object.entries(info.species).sort((a, b) => b[1] - a[1])[0]?.[0] || "Unknown";
      const level = avg >= 80 ? "high" : avg >= 60 ? "moderate" : "low";
      return {
        name,
        score: Math.round(avg),
        level,
        species,
        temp: "Live from model",
        salinity: "Live from model",
        advisory: avg >= 80 ? "Strong candidate. Prioritize field verification." : avg >= 60 ? "Promising area. Run additional checks." : "Low suitability currently.",
      };
    });
    return out.sort((a, b) => b.score - a.score);
  }, [rows]);

  const [selected, setSelected] = useState<RegionItem | null>(null);
  useEffect(() => {
    if (!selected && regions.length > 0) setSelected(regions[0]);
  }, [regions, selected]);

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Site Intelligence</h1>
          <p className="text-muted-foreground text-sm">Realtime regional suitability from your saved prediction history</p>
        </motion.div>

        <div className="flex flex-wrap gap-3 sm:gap-4 text-xs font-medium">
          {(["high", "moderate", "low"] as const).map((l) => (
            <div key={l} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${levelColors[l]}`} />
              <span className="text-muted-foreground">{levelLabels[l]}</span>
            </div>
          ))}
        </div>

        <div className="grid lg:grid-cols-5 gap-6">
          <div className="lg:col-span-2 space-y-3">
            {regions.map((r, i) => (
              <motion.button
                key={r.name}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => setSelected(r)}
                className={`w-full text-left glass-strong rounded-2xl p-4 transition-all duration-200 ${selected?.name === r.name ? "ring-2 ring-primary/30 glow-sm" : ""}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${levelColors[r.level]}`} />
                    <div>
                      <p className="font-semibold text-foreground text-sm">{r.name}</p>
                      <p className="text-xs text-muted-foreground italic">{r.species}</p>
                    </div>
                  </div>
                  <span className="text-sm font-bold gradient-text">{r.score}%</span>
                </div>
              </motion.button>
            ))}
            {!loading && regions.length === 0 && <p className="text-sm text-muted-foreground">No site data yet. Save some predictions first.</p>}
          </div>

          <div className="lg:col-span-3">
            {selected ? (
              <motion.div key={selected.name} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-strong rounded-2xl sm:rounded-3xl p-4 sm:p-6 space-y-5 lg:sticky lg:top-6">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-xl font-bold text-foreground">{selected.name}</h3>
                    <span className={`inline-block mt-1 text-xs font-medium px-2.5 py-0.5 rounded-full ${selected.level === "high" ? "bg-ocean-100 text-ocean-600" : selected.level === "moderate" ? "bg-secondary text-secondary-foreground" : "bg-muted text-muted-foreground"}`}>
                      {levelLabels[selected.level]}
                    </span>
                  </div>
                  <div className="text-right">
                    <p className="text-3xl font-bold gradient-text">{selected.score}%</p>
                    <p className="text-xs text-muted-foreground">Suitability Score</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="glass rounded-2xl p-3 text-center">
                    <Thermometer className="w-5 h-5 text-primary mx-auto mb-1" />
                    <p className="text-sm font-semibold text-foreground">{selected.temp}</p>
                    <p className="text-xs text-muted-foreground">Temperature</p>
                  </div>
                  <div className="glass rounded-2xl p-3 text-center">
                    <Droplets className="w-5 h-5 text-primary mx-auto mb-1" />
                    <p className="text-sm font-semibold text-foreground">{selected.salinity}</p>
                    <p className="text-xs text-muted-foreground">Salinity</p>
                  </div>
                  <div className="glass rounded-2xl p-3 text-center">
                    <MapPin className="w-5 h-5 text-primary mx-auto mb-1" />
                    <p className="text-sm font-semibold text-foreground italic">{selected.species}</p>
                    <p className="text-xs text-muted-foreground">Top Species</p>
                  </div>
                </div>

                <div className="glass rounded-2xl p-4">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Advisory</p>
                  <p className="text-sm text-foreground">{selected.advisory}</p>
                </div>
              </motion.div>
            ) : (
              <div className="glass-strong rounded-3xl p-12 text-center">
                <MapPin className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <p className="text-muted-foreground">Select a region to view detailed intelligence</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
