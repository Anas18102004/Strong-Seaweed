import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { ArrowRight, ChevronDown, MapPin, Calendar, Ruler, Thermometer, Droplets, Orbit } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

const seasons = ["Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter"];
const locations = [
  "Gulf of Mannar",
  "Palk Bay",
  "Lakshadweep",
  "Andaman Islands",
  "Gulf of Kachchh",
  "Chilika",
  "Ratnagiri",
  "Karwar",
  "Kollam",
];

const locationToCoords: Record<string, { lat: number; lon: number }> = {
  "Gulf of Mannar": { lat: 9.1, lon: 79.3 },
  "Palk Bay": { lat: 9.4, lon: 79.2 },
  Lakshadweep: { lat: 10.5, lon: 72.7 },
  "Andaman Islands": { lat: 11.7, lon: 92.7 },
  "Gulf of Kachchh": { lat: 22.6, lon: 69.8 },
  Chilika: { lat: 19.7, lon: 85.3 },
  Ratnagiri: { lat: 16.9, lon: 73.3 },
  Karwar: { lat: 14.8, lon: 74.1 },
  Kollam: { lat: 8.9, lon: 76.6 },
};

const advancedFields = [
  { key: "ph", label: "pH" },
  { key: "turbidityNtu", label: "Turbidity (NTU)" },
  { key: "currentVelocityMs", label: "Current Velocity (m/s)" },
  { key: "waveHeightM", label: "Wave Height (m)" },
  { key: "rainfallMm", label: "Rainfall (mm)" },
  { key: "tidalAmplitudeM", label: "Tidal Amplitude (m)" },
] as const;

function toOptionalNumber(value: string): number | null {
  if (!value.trim()) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fieldClass(hasValue: boolean) {
  return `w-full h-12 rounded-2xl border border-white/15 bg-white/[0.05] px-10 pr-4 text-sm text-white placeholder:text-[#9dc2dd] focus:outline-none focus:ring-2 focus:ring-cyan-300/35 focus:border-cyan-200/35 transition-all ${hasValue ? "shadow-[0_0_0_1px_rgba(125,183,221,0.18)]" : ""}`;
}

export default function PredictPage() {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [location, setLocation] = useState("");
  const [season, setSeason] = useState("");
  const [depth, setDepth] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [temperatureC, setTemperatureC] = useState("");
  const [salinityPpt, setSalinityPpt] = useState("");
  const [advanced, setAdvanced] = useState<Record<string, string>>({
    ph: "",
    turbidityNtu: "",
    currentVelocityMs: "",
    waveHeightM: "",
    rainfallMm: "",
    tidalAmplitudeM: "",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const { token } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    const latNum = Number(lat);
    const lonNum = Number(lon);
    if (!Number.isFinite(latNum) || !Number.isFinite(lonNum)) {
      setError("Please enter valid latitude and longitude.");
      return;
    }

    const formInput = {
      locationName: location || undefined,
      season: season || undefined,
      depthM: toOptionalNumber(depth),
      overrides: {
        temperatureC: toOptionalNumber(temperatureC),
        salinityPpt: toOptionalNumber(salinityPpt),
      },
      advanced: {
        ph: toOptionalNumber(advanced.ph || ""),
        turbidityNtu: toOptionalNumber(advanced.turbidityNtu || ""),
        currentVelocityMs: toOptionalNumber(advanced.currentVelocityMs || ""),
        waveHeightM: toOptionalNumber(advanced.waveHeightM || ""),
        rainfallMm: toOptionalNumber(advanced.rainfallMm || ""),
        tidalAmplitudeM: toOptionalNumber(advanced.tidalAmplitudeM || ""),
      },
    };

    try {
      setIsLoading(true);
      const prediction = await api.predictSpecies(latNum, lonNum, token || undefined, formInput);
      navigate("/results", {
        state: {
          prediction,
          context: {
            location,
            season,
            depth,
            temperatureC,
            salinityPpt,
            advanced,
          },
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Prediction failed.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto">
        <div className="ocean-page-shell">
          <motion.div className="ocean-page-header" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="ocean-page-kicker">Operations / Suitability Engine</p>
                <h1 className="ocean-title-glow mt-2">
                  Check My <span className="ocean-title-highlight">Location</span>
                </h1>
                <p className="mt-3 max-w-2xl text-sm text-[#CFE9FF]/80">
                  Submit site conditions to run live model inference and species-level suitability scoring.
                </p>
                <div className="ocean-header-line" />
              </div>
              <div className="relative hidden sm:block">
                <Orbit className="h-7 w-7 text-cyan-200 ocean-weather-float" />
                <span className="absolute -right-1 -top-1 inline-flex h-2.5 w-2.5 rounded-full bg-cyan-300 ocean-breathe-dot" />
              </div>
            </div>
          </motion.div>

          <div className="px-4 pb-4 sm:px-6 sm:pb-6">
            <div className="grid xl:grid-cols-5 gap-6">
              <motion.form
                onSubmit={handleSubmit}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.12 }}
                className="ocean-glass-card rounded-[22px] p-5 sm:p-6 space-y-5 xl:col-span-3"
              >
                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Location</label>
                    <div className="relative">
                      <select value={location} onChange={(e) => setLocation(e.target.value)} className={fieldClass(!!location) + " appearance-none cursor-pointer"}>
                        <option value="">Select location...</option>
                        {locations.map((l) => (
                          <option key={l} value={l}>
                            {l}
                          </option>
                        ))}
                      </select>
                      <MapPin className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                    </div>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Season</label>
                    <div className="relative">
                      <select value={season} onChange={(e) => setSeason(e.target.value)} className={fieldClass(!!season) + " appearance-none cursor-pointer"}>
                        <option value="">Select season...</option>
                        {seasons.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </select>
                      <Calendar className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                    </div>
                  </div>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Latitude</label>
                    <div className="relative">
                      <input type="number" step="any" placeholder="e.g. 9.1000" value={lat} onChange={(e) => setLat(e.target.value)} className={fieldClass(!!lat)} />
                      <MapPin className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                    </div>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Longitude</label>
                    <div className="relative">
                      <input type="number" step="any" placeholder="e.g. 79.3000" value={lon} onChange={(e) => setLon(e.target.value)} className={fieldClass(!!lon)} />
                      <MapPin className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                    </div>
                  </div>
                </div>

                {location && locationToCoords[location] && (
                  <button
                    type="button"
                    onClick={() => {
                      setLat(String(locationToCoords[location].lat));
                      setLon(String(locationToCoords[location].lon));
                    }}
                    className="text-xs text-cyan-200 hover:text-cyan-100 transition-colors"
                  >
                    Use default coordinates for {location}
                  </button>
                )}

                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Depth (m)</label>
                    <div className="relative">
                      <input type="number" placeholder="e.g. 5" value={depth} onChange={(e) => setDepth(e.target.value)} className={fieldClass(!!depth)} />
                      <Ruler className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                    </div>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Temperature Override (deg C)</label>
                    <div className="relative">
                      <input
                        type="number"
                        step="any"
                        placeholder="Auto-fetched"
                        value={temperatureC}
                        onChange={(e) => setTemperatureC(e.target.value)}
                        className={fieldClass(!!temperatureC)}
                      />
                      <Thermometer className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                    </div>
                  </div>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Salinity Override (ppt)</label>
                  <div className="relative">
                    <input
                      type="number"
                      step="any"
                      placeholder="Auto-fetched"
                      value={salinityPpt}
                      onChange={(e) => setSalinityPpt(e.target.value)}
                      className={fieldClass(!!salinityPpt)}
                    />
                    <Droplets className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                  </div>
                </div>

                <div className="rounded-2xl border border-white/12 bg-white/[0.04]">
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-[#CFE9FF]"
                  >
                    Advanced Parameters
                    <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                  </button>
                  {showAdvanced && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="border-t border-white/10 px-4 py-4">
                      <div className="grid sm:grid-cols-2 gap-4">
                        {advancedFields.map((field) => (
                          <div key={field.key}>
                            <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">{field.label}</label>
                            <input
                              type="number"
                              step="any"
                              placeholder="-"
                              value={advanced[field.key] || ""}
                              onChange={(e) => setAdvanced((prev) => ({ ...prev, [field.key]: e.target.value }))}
                              className={fieldClass(!!advanced[field.key])}
                            />
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </div>

                {error && <p className="text-sm text-rose-300">{error}</p>}

                <Button type="submit" size="lg" className="ocean-shine-btn w-full bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] text-white hover:opacity-95" disabled={isLoading}>
                  {isLoading ? "Running model..." : "Run Suitability Model"}
                  <ArrowRight className="h-5 w-5" />
                </Button>
              </motion.form>

              <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="xl:col-span-2 space-y-4">
                <div className="ocean-map-placeholder rounded-[22px] min-h-[340px] p-5">
                  <div className="ocean-map-orbiter" />
                  <p className="ocean-page-kicker">Live Preview</p>
                  <h3 className="mt-1 text-xl font-semibold text-white">Suitability Visualizer</h3>
                  <p className="mt-2 max-w-xs text-sm text-[#CFE9FF]/75">Live suitability preview will appear here after model execution.</p>
                  <div className="mt-8 grid grid-cols-2 gap-2 text-xs">
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-cyan-100">Temperature Layer</div>
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-cyan-100">Salinity Layer</div>
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-cyan-100">Depth Layer</div>
                    <div className="rounded-xl border border-cyan-100/15 bg-white/[0.06] px-3 py-2 text-cyan-100">Current Layer</div>
                  </div>
                </div>
                <div className="ocean-glass-card rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.14em] text-[#7FA9C4]">Execution Notes</p>
                  <p className="mt-2 text-sm text-[#CFE9FF]/80">Use this panel to validate coordinates and environmental overrides before triggering high-confidence species recommendations.</p>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
