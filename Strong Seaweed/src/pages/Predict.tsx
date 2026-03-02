import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { ArrowRight, ChevronDown, MapPin } from "lucide-react";
import { useNavigate } from "react-router-dom";

const seasons = ["Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter"];
const locations = [
  "Gulf of Mannar", "Palk Bay", "Lakshadweep", "Andaman Islands",
  "Gulf of Kachchh", "Chilika", "Ratnagiri", "Karwar", "Kollam"
];

export default function PredictPage() {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [location, setLocation] = useState("");
  const [season, setSeason] = useState("");
  const [depth, setDepth] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate("/results");
  };

  const inputClass = "w-full h-12 rounded-2xl glass-strong px-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 transition-shadow";
  const labelClass = "block text-sm font-medium text-foreground mb-1.5";

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">New Prediction</h1>
          <p className="text-muted-foreground text-sm mb-8">Enter site parameters to run the ecological suitability model</p>
        </motion.div>

        <motion.form
          onSubmit={handleSubmit}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="glass-strong rounded-3xl p-8 space-y-6"
        >
          {/* Basic fields */}
          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Location</label>
              <div className="relative">
                <select value={location} onChange={e => setLocation(e.target.value)} className={inputClass + " appearance-none cursor-pointer"}>
                  <option value="">Select location...</option>
                  {locations.map(l => <option key={l} value={l}>{l}</option>)}
                </select>
                <MapPin className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
              </div>
            </div>
            <div>
              <label className={labelClass}>Season</label>
              <select value={season} onChange={e => setSeason(e.target.value)} className={inputClass + " appearance-none cursor-pointer"}>
                <option value="">Select season...</option>
                {seasons.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-5">
            <div>
              <label className={labelClass}>Depth (m)</label>
              <input type="number" placeholder="e.g. 5" value={depth} onChange={e => setDepth(e.target.value)} className={inputClass} />
            </div>
            <div>
              <label className={labelClass}>Temperature Override (°C)</label>
              <input type="number" placeholder="Auto-fetched" className={inputClass} />
            </div>
          </div>

          <div>
            <label className={labelClass}>Salinity Override (ppt)</label>
            <input type="number" placeholder="Auto-fetched" className={inputClass} />
          </div>

          {/* Advanced toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
          >
            <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
            Advanced Parameters
          </button>

          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="space-y-6 border-t border-border/40 pt-6"
            >
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Physical Parameters</p>
              <div className="grid sm:grid-cols-2 gap-5">
                {["pH", "Turbidity (NTU)", "Current Velocity (m/s)", "Wave Height (m)", "Rainfall (mm)", "Tidal Amplitude (m)"].map(f => (
                  <div key={f}>
                    <label className={labelClass}>{f}</label>
                    <input type="number" placeholder="—" className={inputClass} />
                  </div>
                ))}
              </div>

              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Nutrient Parameters</p>
              <div className="grid sm:grid-cols-2 gap-5">
                {["Nitrate (µg/L)", "Phosphate (µg/L)", "Ammonium (µg/L)", "Silicate (µg/L)"].map(f => (
                  <div key={f}>
                    <label className={labelClass}>{f}</label>
                    <input type="number" placeholder="—" className={inputClass} />
                  </div>
                ))}
              </div>

              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Pollution Indicators</p>
              <div className="grid sm:grid-cols-2 gap-5">
                {["BOD (mg/L)", "DO (mg/L)", "Heavy Metals Level"].map(f => (
                  <div key={f}>
                    <label className={labelClass}>{f}</label>
                    <input type="number" placeholder="—" className={inputClass} />
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          <Button type="submit" variant="hero" size="lg" className="w-full">
            Run Suitability Model <ArrowRight className="w-5 h-5" />
          </Button>
        </motion.form>
      </div>
    </DashboardLayout>
  );
}
