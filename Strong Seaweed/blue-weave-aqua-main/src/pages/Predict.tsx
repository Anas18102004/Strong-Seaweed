import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { ArrowRight, ChevronDown, MapPin } from "lucide-react";
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

  const inputClass =
    "w-full h-12 rounded-2xl glass-strong px-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 transition-shadow";
  const labelClass = "block text-sm font-medium text-foreground mb-1.5";

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Check My Location</h1>
          <p className="text-muted-foreground text-sm mb-8">Enter site details and get species-wise suitability output.</p>
        </motion.div>

        <motion.form
          onSubmit={handleSubmit}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="glass-strong rounded-3xl p-8 space-y-6"
        >
          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Location</label>
              <div className="relative">
                <select value={location} onChange={(e) => setLocation(e.target.value)} className={inputClass + " appearance-none cursor-pointer"}>
                  <option value="">Select location...</option>
                  {locations.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
                <MapPin className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
              </div>
            </div>
            <div>
              <label className={labelClass}>Season</label>
              <select value={season} onChange={(e) => setSeason(e.target.value)} className={inputClass + " appearance-none cursor-pointer"}>
                <option value="">Select season...</option>
                {seasons.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Latitude</label>
              <input type="number" step="any" placeholder="e.g. 9.1000" value={lat} onChange={(e) => setLat(e.target.value)} className={inputClass} />
            </div>
            <div>
              <label className={labelClass}>Longitude</label>
              <input type="number" step="any" placeholder="e.g. 79.3000" value={lon} onChange={(e) => setLon(e.target.value)} className={inputClass} />
            </div>
          </div>

          {location && locationToCoords[location] && (
            <button
              type="button"
              onClick={() => {
                setLat(String(locationToCoords[location].lat));
                setLon(String(locationToCoords[location].lon));
              }}
              className="text-xs text-primary hover:text-primary/80 transition-colors"
            >
              Use default coordinates for {location}
            </button>
          )}

          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Depth (m)</label>
              <input type="number" placeholder="e.g. 5" value={depth} onChange={(e) => setDepth(e.target.value)} className={inputClass} />
            </div>
            <div>
              <label className={labelClass}>Temperature Override (deg C)</label>
              <input
                type="number"
                step="any"
                placeholder="Auto-fetched"
                value={temperatureC}
                onChange={(e) => setTemperatureC(e.target.value)}
                className={inputClass}
              />
            </div>
          </div>

          <div>
            <label className={labelClass}>Salinity Override (ppt)</label>
            <input
              type="number"
              step="any"
              placeholder="Auto-fetched"
              value={salinityPpt}
              onChange={(e) => setSalinityPpt(e.target.value)}
              className={inputClass}
            />
          </div>

          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
          >
            <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
            Advanced Parameters
          </button>

          {showAdvanced && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="space-y-6 border-t border-border/40 pt-6">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Physical Parameters</p>
              <div className="grid sm:grid-cols-2 gap-5">
                {advancedFields.map((field) => (
                  <div key={field.key}>
                    <label className={labelClass}>{field.label}</label>
                    <input
                      type="number"
                      step="any"
                      placeholder="-"
                      value={advanced[field.key] || ""}
                      onChange={(e) => setAdvanced((prev) => ({ ...prev, [field.key]: e.target.value }))}
                      className={inputClass}
                    />
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {error && <p className="text-sm text-red-500">{error}</p>}

          <Button type="submit" variant="hero" size="lg" className="w-full" disabled={isLoading}>
            {isLoading ? "Running model..." : "Run Suitability Model"} <ArrowRight className="w-5 h-5" />
          </Button>
        </motion.form>
      </div>
    </DashboardLayout>
  );
}
