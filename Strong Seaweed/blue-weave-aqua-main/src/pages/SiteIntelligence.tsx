import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useMemo, useState, useEffect } from "react";
import { MapPin, Droplets, Thermometer, ArrowRight, Layers } from "lucide-react";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useNavigate } from "react-router-dom";

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
  high: "bg-emerald-300",
  moderate: "bg-cyan-300",
  low: "bg-slate-400",
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
  const navigate = useNavigate();
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
      <div className="max-w-7xl mx-auto">
        <div className="ocean-page-shell">
          <motion.div className="ocean-page-header" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <p className="ocean-page-kicker">Operations / Site Intelligence</p>
            <h1 className="ocean-title-glow mt-2">
              Top Farming <span className="ocean-title-highlight">Areas</span>
            </h1>
            <p className="mt-3 max-w-2xl text-sm text-[#CFE9FF]/80">
              Compare regional suitability using historical inference data and quickly identify high-confidence cultivation zones.
            </p>
            <div className="ocean-header-line" />
          </motion.div>

          <div className="px-4 pb-4 sm:px-6 sm:pb-6">
            <div className="grid xl:grid-cols-5 gap-6">
              <div className="xl:col-span-2 space-y-4">
                <div className="ocean-glass-card rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Region Selector</p>
                  <p className="mt-1 text-sm text-[#CFE9FF]/80">Choose a region to inspect model confidence and advisory signals.</p>
                </div>
                <div className="space-y-3">
                  {regions.map((r, i) => (
                    <motion.button
                      key={r.name}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                      onClick={() => setSelected(r)}
                      className={`w-full text-left rounded-2xl border p-4 transition-all duration-200 ${
                        selected?.name === r.name
                          ? "border-cyan-200/35 bg-gradient-to-r from-cyan-400/15 to-blue-400/20 shadow-[0_10px_22px_-16px_rgba(14,165,233,0.85)]"
                          : "border-white/10 bg-white/[0.04] hover:bg-white/[0.08]"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`h-2.5 w-2.5 rounded-full ${levelColors[r.level]}`} />
                          <div>
                            <p className="text-sm font-semibold text-white">{r.name}</p>
                            <p className="text-xs text-[#9fc6e2] italic">{r.species}</p>
                          </div>
                        </div>
                        <span className="text-sm font-bold text-cyan-100">{r.score}%</span>
                      </div>
                    </motion.button>
                  ))}
                  {!loading && regions.length === 0 && (
                    <div className="ocean-glass-card rounded-2xl p-6 text-center">
                      <MapPin className="mx-auto h-8 w-8 text-cyan-200" />
                      <p className="mt-2 text-sm text-[#CFE9FF]/80">No site data yet. Save prediction runs to unlock ranking.</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="xl:col-span-3 space-y-4">
                <div className="ocean-map-placeholder rounded-[22px] min-h-[280px] p-5">
                  <div className="ocean-map-orbiter" />
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="ocean-page-kicker">Intelligence Map</p>
                      <h3 className="mt-1 text-lg font-semibold text-white">Regional Suitability Layers</h3>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-cyan-100">
                      <span className="rounded-full border border-cyan-100/20 bg-white/10 px-2.5 py-1">Depth</span>
                      <span className="rounded-full border border-cyan-100/20 bg-white/10 px-2.5 py-1">Salinity</span>
                      <span className="rounded-full border border-cyan-100/20 bg-white/10 px-2.5 py-1">Temperature</span>
                    </div>
                  </div>
                  <div className="mt-6 grid grid-cols-3 gap-3">
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-xs text-cyan-100">West Coast Marker</div>
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-xs text-cyan-100">South Bay Marker</div>
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-xs text-cyan-100">Island Marker</div>
                  </div>
                </div>

                {selected ? (
                  <motion.div key={selected.name} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="ocean-glass-card rounded-[22px] p-5 sm:p-6 space-y-5">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="text-2xl font-semibold text-white">{selected.name}</h3>
                        <span className="mt-2 inline-flex rounded-full border border-cyan-200/20 bg-cyan-400/10 px-2.5 py-1 text-xs font-semibold text-cyan-100">
                          {levelLabels[selected.level]}
                        </span>
                      </div>
                      <div className="text-right">
                        <p className="text-3xl font-bold text-cyan-100">{selected.score}%</p>
                        <p className="text-xs text-[#7FA9C4]">Suitability Score</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-center">
                        <Thermometer className="mx-auto mb-1 h-5 w-5 text-cyan-100" />
                        <p className="text-sm font-semibold text-white">{selected.temp}</p>
                        <p className="text-xs text-[#7FA9C4]">Temperature</p>
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-center">
                        <Droplets className="mx-auto mb-1 h-5 w-5 text-cyan-100" />
                        <p className="text-sm font-semibold text-white">{selected.salinity}</p>
                        <p className="text-xs text-[#7FA9C4]">Salinity</p>
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-center">
                        <Layers className="mx-auto mb-1 h-5 w-5 text-cyan-100" />
                        <p className="text-sm font-semibold text-white italic">{selected.species}</p>
                        <p className="text-xs text-[#7FA9C4]">Top Species</p>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#7FA9C4]">Advisory</p>
                      <p className="mt-2 text-sm text-[#CFE9FF]">{selected.advisory}</p>
                    </div>
                  </motion.div>
                ) : (
                  <div className="ocean-glass-card rounded-[22px] p-10 text-center">
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-cyan-100/20 bg-cyan-300/10">
                      <MapPin className="h-6 w-6 text-cyan-100" />
                    </div>
                    <p className="text-base font-semibold text-white">Analyze a Region</p>
                    <p className="mt-2 text-sm text-[#CFE9FF]/70">Select a region from the left panel to generate intelligence insights.</p>
                    <button
                      onClick={() => navigate("/predict")}
                      className="mt-5 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] px-4 py-2 text-sm font-semibold text-white"
                    >
                      Run a Prediction
                      <ArrowRight className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
