import DashboardLayout from "@/components/DashboardLayout";
import FallbackAdvisoryCard from "@/components/FallbackAdvisoryCard";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, ChevronDown, MapPin, Calendar, Ruler, Thermometer, Droplets, Radar, LocateFixed, ShieldCheck, ExternalLink } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api, SpeciesPredictionResponse } from "@/lib/api";
import { sanitizeAdvisoryText } from "@/lib/advisory";
import { useAuth } from "@/context/AuthContext";
import { CircleMarker, MapContainer, TileLayer, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const defaultSeasons = ["Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter"];
const defaultLocations = [
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
  return `w-full h-11 rounded-xl border border-white/15 bg-white/[0.05] px-10 pr-4 text-sm text-white placeholder:text-[#c7e1f3] focus:outline-none focus:ring-2 focus:ring-cyan-300/35 focus:border-cyan-200/35 transition-all ${hasValue ? "shadow-[0_0_0_1px_rgba(125,183,221,0.18)]" : ""}`;
}

function riskBand(score: number) {
  if (score >= 78) return "Low Risk";
  if (score >= 60) return "Moderate Risk";
  return "High Risk";
}

function confidenceLabel(score: number, completeness: number) {
  if (score >= 80 && completeness >= 0.6) return "High";
  if (score >= 65 && completeness >= 0.45) return "Medium";
  return "Developing";
}

function recommendedSpecies(score: number, season: string, liveName?: string, hasLivePrediction = false) {
  if (liveName) return liveName;
  if (hasLivePrediction) return "No species passes threshold";
  if (season.toLowerCase().includes("monsoon")) return "Sargassum wightii";
  if (score >= 75) return "Kappaphycus alvarezii";
  if (score >= 62) return "Gracilaria edulis";
  return "Ulva lactuca";
}

function actionabilityLabel(value?: string | null) {
  const v = String(value || "insufficient_data").toLowerCase();
  if (v === "recommended") return "Recommended";
  if (v === "test_pilot_only") return "Pilot Only";
  if (v === "not_recommended") return "Not Recommended";
  return "Insufficient Data";
}

function normalizeReason(value?: string | null) {
  const raw = String(value || "").trim();
  if (!raw) return "No reason provided";
  return raw.replace(/_/g, " ");
}

function splitModelRelease(modelRelease?: string | null): { primary: string; secondary: string | null } {
  const text = String(modelRelease || "").trim();
  if (!text) return { primary: "v2.0 marine-core", secondary: null };
  const [first, ...rest] = text.split("+");
  return { primary: first || "v2.0 marine-core", secondary: rest.length ? rest.join("+") : null };
}

function MapClickHandler({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click: (e) => {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function PredictPage() {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [locations, setLocations] = useState<string[]>(defaultLocations);
  const [seasons, setSeasons] = useState<string[]>(defaultSeasons);
  const [location, setLocation] = useState("");
  const [season, setSeason] = useState("");
  const [depth, setDepth] = useState("");
  const [coords, setCoords] = useState<{ lat: number; lon: number } | null>(null);
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
  const [lastPrediction, setLastPrediction] = useState<SpeciesPredictionResponse | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [envSource, setEnvSource] = useState<string>("");
  const resultPanelRef = useRef<HTMLDivElement | null>(null);
  const navigate = useNavigate();
  const { token } = useAuth();

  useEffect(() => {
    let mounted = true;
    const loadReference = async () => {
      if (!token) return;
      try {
        const out = await api.predictReference(token);
        if (!mounted) return;
        if (Array.isArray(out.locations) && out.locations.length) setLocations(out.locations);
        if (Array.isArray(out.seasons) && out.seasons.length) setSeasons(out.seasons);
        if (!season && out.currentSeason) setSeason(out.currentSeason);
      } catch {
        // keep defaults
      }
    };
    void loadReference();
    return () => {
      mounted = false;
    };
  }, [token, season]);

  useEffect(() => {
    let mounted = true;
    const loadEnvironment = async () => {
      if (!token || !coords) return;
      try {
        const env = await api.predictEnvironment(coords.lat, coords.lon, token);
        if (!mounted) return;
        setTemperatureC(env.temperatureC === null ? "" : String(env.temperatureC));
        setSalinityPpt(env.salinityPpt === null ? "" : String(env.salinityPpt));
        setEnvSource(env.provider || "");
      } catch {
        if (!mounted) return;
        setEnvSource("");
      }
    };
    void loadEnvironment();
    return () => {
      mounted = false;
    };
  }, [coords, token]);

  const handlePickCoords = (lat: number, lon: number) => {
    setCoords({ lat, lon });
    if (!location) setLocation("Map selected location");
  };

  const handleAutoDetect = () => {
    setError("");
    if (!navigator.geolocation) {
      setError("Geolocation is not supported in this browser.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setCoords({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        if (!location) setLocation("Auto-detected location");
      },
      (geoErr) => {
        const code = geoErr?.code;
        if (code === 1) setError("Location permission denied in browser. Allow location access and try again.");
        else if (code === 2) setError("Location service unavailable. Turn on device GPS/network location.");
        else if (code === 3) setError("Location request timed out. Try again or click the map.");
        else setError("Unable to auto-detect your location. Please click on map.");
      },
      { enableHighAccuracy: true, timeout: 12000 },
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!coords) {
      setError("Select location from map or auto-detect first.");
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
      const prediction = await api.predictSpecies(coords.lat, coords.lon, token || undefined, formInput);
      setLastPrediction(prediction);
      setLastUpdated(new Date().toLocaleString());
      window.setTimeout(() => {
        resultPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Prediction failed.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const metrics = useMemo(() => {
    const completenessInputs = [coords, season, depth, temperatureC, salinityPpt].filter(Boolean).length;
    const completeness = Math.min(1, completenessInputs / 5);

    const liveScore =
      lastPrediction?.finalRecommendation?.probabilityPercent ??
      lastPrediction?.bestSpecies?.probabilityPercent ??
      null;
    let heuristic = 55;
    const temp = Number(temperatureC);
    const sal = Number(salinityPpt);
    const dep = Number(depth);
    if (Number.isFinite(temp)) heuristic += temp >= 24 && temp <= 32 ? 8 : -6;
    if (Number.isFinite(sal)) heuristic += sal >= 27 && sal <= 36 ? 10 : -7;
    if (Number.isFinite(dep)) heuristic += dep >= 2 && dep <= 12 ? 6 : -3;
    if (season.toLowerCase().includes("post")) heuristic += 4;
    if (coords) heuristic += 6;
    const score = Math.max(0, Math.min(99, Math.round(liveScore ?? heuristic)));

    return {
      score,
      confidence: confidenceLabel(score, completeness),
      risk: riskBand(score),
      species: recommendedSpecies(
        score,
        season,
        lastPrediction?.finalRecommendation?.displayName || lastPrediction?.bestSpecies?.displayName || undefined,
        Boolean(lastPrediction),
      ),
    };
  }, [coords, season, depth, temperatureC, salinityPpt, lastPrediction]);

  const predictionView = useMemo(() => {
    if (!lastPrediction) return null;
    const scored = [...(lastPrediction.species || [])]
      .filter((item) => Number.isFinite(Number(item?.probabilityPercent)))
      .sort((a, b) => Number(b?.probabilityPercent || 0) - Number(a?.probabilityPercent || 0));
    const topScored = scored[0] || null;
    const chosen =
      lastPrediction.finalRecommendation && lastPrediction.finalRecommendation.speciesId !== "insufficient_data"
        ? lastPrediction.finalRecommendation
        : lastPrediction.bestSpecies && lastPrediction.bestSpecies.speciesId !== "insufficient_data"
        ? lastPrediction.bestSpecies
        : topScored;
    const chosenId = chosen?.speciesId || topScored?.speciesId || "";
    return { chosen, chosenId };
  }, [lastPrediction]);
  const modelReleaseInfo = useMemo(() => splitModelRelease(lastPrediction?.modelRelease), [lastPrediction?.modelRelease]);

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto">
        <div className="ocean-page-shell">
          <motion.div className="ocean-page-header" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="ocean-page-kicker">Operations / Suitability Engine</p>
                <h1 className="ocean-title-glow mt-2">
                  Marine <span className="ocean-title-highlight">Suitability</span> Command
                </h1>
                <p className="mt-3 max-w-2xl text-sm text-[#E6F5FF]">
                  Select a location directly on the map, enrich environmental signals, and run model-grade species suitability forecasting.
                </p>
                <div className="ocean-header-line" />
              </div>
                <div className="ocean-glass-card rounded-xl px-3 py-2 text-xs text-[#EAF7FF]">
                <p className="truncate" title={modelReleaseInfo.primary}>Model version: {modelReleaseInfo.primary}</p>
                {modelReleaseInfo.secondary ? (
                  <p className="mt-1 truncate" title={modelReleaseInfo.secondary}>Multi release: {modelReleaseInfo.secondary}</p>
                ) : null}
                <p className="mt-1">Data source: {lastPrediction?.source || "Hybrid climate + hydro layers"}</p>
                <p className="mt-1">Last updated: {lastUpdated || "Just now"}</p>
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
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[#A7CCE4] mb-2">1. Location</p>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div>
                      <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Location Label</label>
                      <div className="relative">
                        <select
                          value={location}
                          onChange={(e) => {
                            const next = e.target.value;
                            setLocation(next);
                            if (locationToCoords[next]) setCoords(locationToCoords[next]);
                          }}
                          className={fieldClass(!!location) + " appearance-none cursor-pointer"}
                        >
                          <option value="" className="text-slate-900 bg-white">Select location...</option>
                          {locations.map((l) => (
                            <option key={l} value={l} className="text-slate-900 bg-white">
                              {l}
                            </option>
                          ))}
                        </select>
                        <MapPin className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                      </div>
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Season</label>
                      <div className="relative">
                        <select value={season} onChange={(e) => setSeason(e.target.value)} className={fieldClass(!!season) + " appearance-none cursor-pointer"}>
                          <option value="" className="text-slate-900 bg-white">Select season...</option>
                          {seasons.map((s) => (
                            <option key={s} value={s} className="text-slate-900 bg-white">
                              {s}
                            </option>
                          ))}
                        </select>
                        <Calendar className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <Button type="button" onClick={handleAutoDetect} className="rounded-full bg-white/10 border border-white/20 text-cyan-100 hover:bg-white/15">
                      <LocateFixed className="w-4 h-4 mr-1" />
                      Auto-detect my location
                    </Button>
                    <span className="text-xs text-[#D7EEFF]">
                      {coords ? `Selected: ${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)}${envSource ? ` | Live env: ${envSource}` : ""}` : "No coordinates selected yet"}
                    </span>
                  </div>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[#A7CCE4] mb-2">2. Environmental Parameters</p>
                  <div className="grid sm:grid-cols-3 gap-4">
                    <div>
                      <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Depth (m)</label>
                      <div className="relative">
                        <input type="number" placeholder="e.g. 5" value={depth} onChange={(e) => setDepth(e.target.value)} className={fieldClass(!!depth)} />
                        <Ruler className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                      </div>
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Temperature (deg C)</label>
                      <div className="relative">
                        <input type="number" step="any" placeholder="Auto-fetched" value={temperatureC} onChange={(e) => setTemperatureC(e.target.value)} className={fieldClass(!!temperatureC)} />
                        <Thermometer className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                      </div>
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Salinity (ppt)</label>
                      <div className="relative">
                        <input type="number" step="any" placeholder="Auto-fetched" value={salinityPpt} onChange={(e) => setSalinityPpt(e.target.value)} className={fieldClass(!!salinityPpt)} />
                        <Droplets className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-cyan-200" />
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[#A7CCE4] mb-2">3. Advanced Controls</p>
                  <div className="rounded-2xl border border-white/12 bg-white/[0.04]">
                    <button
                      type="button"
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-[#E6F5FF]"
                    >
                      Tune advanced hydrology parameters
                      <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                    </button>
                    {showAdvanced && (
                      <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="border-t border-white/10 px-4 py-4">
                        <div className="grid sm:grid-cols-2 gap-4">
                          {advancedFields.map((field) => (
                            <div key={field.key}>
                              <label className="mb-1.5 block text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">{field.label}</label>
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
                </div>

                {error && <p className="text-sm text-rose-300">{error}</p>}

                <Button type="submit" size="lg" className="ocean-shine-btn w-full min-h-14 text-base bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] text-white shadow-[0_18px_36px_-18px_rgba(14,165,233,0.9)] hover:opacity-95" disabled={isLoading}>
                  {isLoading ? "Running Suitability Model..." : "Run Marine Suitability Model"}
                  <ArrowRight className="h-5 w-5" />
                </Button>
              </motion.form>

              <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="xl:col-span-2 space-y-4">
                <div className="rounded-[22px] overflow-hidden border border-white/12 shadow-[0_18px_34px_-24px_rgba(10,41,65,0.9)]">
                  <div className="h-[270px] w-full bg-slate-900">
                    <MapContainer center={[13.5, 79]} zoom={4} scrollWheelZoom className="h-full w-full">
                      <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                      <MapClickHandler onPick={handlePickCoords} />
                      {coords && (
                        <CircleMarker center={[coords.lat, coords.lon]} radius={8} pathOptions={{ color: "#22d3ee", fillColor: "#1DA1F2", fillOpacity: 0.85, weight: 2 }} />
                      )}
                    </MapContainer>
                  </div>
                </div>

                <div className="ocean-glass-card rounded-2xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Dynamic Suitability</p>
                    <Radar className={`h-4 w-4 text-cyan-200 ${isLoading ? "animate-spin" : ""}`} />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
                      <p className="text-[11px] text-[#A7CCE4] uppercase">Score</p>
                      <p className="text-2xl font-semibold text-white mt-1">{metrics.score}%</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
                      <p className="text-[11px] text-[#A7CCE4] uppercase">Confidence</p>
                      <p className="text-base font-semibold text-cyan-100 mt-1">{metrics.confidence}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
                      <p className="text-[11px] text-[#A7CCE4] uppercase">Risk Band</p>
                      <p className="text-sm font-semibold text-emerald-100 mt-1">{metrics.risk}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.04] p-3">
                      <p className="text-[11px] text-[#A7CCE4] uppercase">Recommended</p>
                      <p className="text-sm font-semibold text-white mt-1">{metrics.species}</p>
                    </div>
                  </div>
                  {lastPrediction && (
                    <Button
                      type="button"
                      onClick={() => navigate("/results", { state: { prediction: lastPrediction, context: { location, season, depth, temperatureC, salinityPpt, advanced } } })}
                      className="mt-3 w-full rounded-xl bg-white/10 border border-white/20 text-cyan-100 hover:bg-white/15"
                    >
                      View Detailed Result
                    </Button>
                  )}
                </div>

                {!lastPrediction && (
                  <div className="ocean-glass-card rounded-2xl p-5 relative overflow-hidden">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_30%,rgba(20,184,166,0.18),transparent_35%),radial-gradient(circle_at_70%_60%,rgba(29,161,242,0.22),transparent_40%)] animate-pulse" />
                    <p className="relative text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Live Preview</p>
                    <p className="relative mt-2 text-sm text-[#E1F2FF]">Animated heatmap placeholder will convert into model-derived suitability surface after execution.</p>
                    <div className="relative mt-4 h-20 rounded-xl border border-cyan-100/15 bg-gradient-to-r from-teal-400/20 via-cyan-300/25 to-blue-500/20" />
                  </div>
                )}

                <div className="ocean-glass-card rounded-2xl p-4">
                  <div className="flex items-center gap-2 text-cyan-100">
                    <ShieldCheck className="h-4 w-4" />
                    <p className="text-xs uppercase tracking-[0.14em]">Trust Indicators</p>
                  </div>
                  <p className="text-xs text-[#D6ECFB] mt-2">Model version: {lastPrediction?.modelRelease || "v2.0 marine-core"} | Source: {lastPrediction?.source || "Akuara climate mesh"} | Last updated: {lastUpdated || "Awaiting first run"}</p>
                </div>
              </motion.div>
            </div>

            {lastPrediction && predictionView && (
              <motion.div
                ref={resultPanelRef}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.12 }}
                className="mt-6 ocean-glass-card rounded-[22px] p-5 sm:p-6"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.14em] text-[#A7CCE4]">Latest Prediction Result</p>
                    <h2 className="mt-1 text-xl font-semibold text-[#EAF7FF]">
                      {predictionView.chosen?.displayName || "No clear candidate"}
                    </h2>
                    <p className="mt-1 text-sm text-[#CFE8F8]">
                      Actionability: {actionabilityLabel(predictionView.chosen?.actionability)}
                      {typeof predictionView.chosen?.probabilityPercent === "number"
                        ? ` | Score: ${predictionView.chosen.probabilityPercent.toFixed(2)}%`
                        : ""}
                    </p>
                  </div>
                  <Button
                    type="button"
                    onClick={() =>
                      navigate("/results", {
                        state: { prediction: lastPrediction, context: { location, season, depth, temperatureC, salinityPpt, advanced } },
                      })
                    }
                    className="rounded-xl bg-white/10 border border-white/20 text-cyan-100 hover:bg-white/15"
                  >
                    Open Detailed Result <ExternalLink className="h-4 w-4" />
                  </Button>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="min-w-0 rounded-xl border border-white/10 bg-white/[0.04] p-3">
                    <p className="text-[11px] uppercase text-[#A7CCE4]">Model Release</p>
                    <p className="mt-1 truncate text-[13px] leading-snug text-[#EAF7FF]" title={modelReleaseInfo.primary}>{modelReleaseInfo.primary}</p>
                    {modelReleaseInfo.secondary ? (
                      <p className="mt-0.5 truncate text-[12px] leading-snug text-[#D3EDFF]" title={modelReleaseInfo.secondary}>+ {modelReleaseInfo.secondary}</p>
                    ) : null}
                  </div>
                  <div className="min-w-0 rounded-xl border border-white/10 bg-white/[0.04] p-3">
                    <p className="text-[11px] uppercase text-[#A7CCE4]">Source</p>
                    <p className="mt-1 break-words text-sm leading-snug text-[#EAF7FF]">{lastPrediction.source}</p>
                  </div>
                  <div className="min-w-0 rounded-xl border border-white/10 bg-white/[0.04] p-3">
                    <p className="text-[11px] uppercase text-[#A7CCE4]">Verification</p>
                    <p className="mt-1 break-words text-sm leading-snug text-[#EAF7FF]">
                      {lastPrediction.verification?.verdict || "unknown"} ({lastPrediction.verification?.confidenceScore ?? 0}%)
                    </p>
                  </div>
                  <div className="min-w-0 rounded-xl border border-white/10 bg-white/[0.04] p-3">
                    <p className="text-[11px] uppercase text-[#A7CCE4]">Nearest Grid</p>
                    <p className="mt-1 text-sm text-[#EAF7FF]">
                      {typeof lastPrediction.nearestGrid?.distance_km === "number"
                        ? `${lastPrediction.nearestGrid.distance_km.toFixed(2)} km`
                        : "N/A"}
                    </p>
                  </div>
                </div>

                <div className="mt-4 overflow-hidden rounded-xl border border-white/10">
                  <div className="grid grid-cols-[minmax(0,1.5fr)_auto_auto] gap-3 bg-white/[0.05] px-4 py-2 text-[11px] uppercase tracking-[0.12em] text-[#A7CCE4]">
                    <p>Species</p>
                    <p>Score</p>
                    <p>Actionability</p>
                  </div>
                  <div className="divide-y divide-white/10">
                    {lastPrediction.species.map((item) => {
                      const isChosen = item.speciesId === predictionView.chosenId;
                      return (
                        <div
                          key={item.speciesId}
                          className={`grid grid-cols-[minmax(0,1.5fr)_auto_auto] gap-3 px-4 py-3 ${isChosen ? "bg-cyan-400/10" : "bg-transparent"}`}
                        >
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium text-[#EAF7FF]">
                              {item.displayName}
                              {isChosen ? " (selected)" : ""}
                            </p>
                            <p className="mt-0.5 text-xs text-[#C2DEEF]">{normalizeReason(item.reason)}</p>
                          </div>
                          <p className="text-sm text-[#EAF7FF]">
                            {typeof item.probabilityPercent === "number" ? `${item.probabilityPercent.toFixed(2)}%` : "N/A"}
                          </p>
                          <p className="text-sm text-[#EAF7FF]">{actionabilityLabel(item.actionability)}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {Array.isArray(lastPrediction.warnings) && lastPrediction.warnings.length > 0 && (
                  <div className="mt-4 rounded-xl border border-amber-200/25 bg-amber-400/10 p-3">
                    <p className="text-xs uppercase tracking-[0.12em] text-amber-100">Warnings</p>
                    <p className="mt-1 text-sm text-amber-50">{lastPrediction.warnings.map((w) => normalizeReason(w)).join(" | ")}</p>
                  </div>
                )}

                {lastPrediction.fallbackAdvisory?.answer && (
                  <FallbackAdvisoryCard
                    text={sanitizeAdvisoryText(lastPrediction.fallbackAdvisory.answer)}
                    variant="dark"
                    className="mt-4"
                  />
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

