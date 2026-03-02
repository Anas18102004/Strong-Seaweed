import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useState } from "react";
import { MapPin, Droplets, Thermometer, Wind, ChevronRight } from "lucide-react";

const regions = [
  { name: "Gulf of Mannar", score: 92, level: "high" as const, species: "Kappaphycus alvarezii", temp: "28.5°C", salinity: "33 ppt", advisory: "Optimal cultivation window: Oct–Mar", lat: 78, top: 72 },
  { name: "Palk Bay", score: 85, level: "high" as const, species: "Gracilaria edulis", temp: "29°C", salinity: "32 ppt", advisory: "Good year-round potential", lat: 82, top: 68 },
  { name: "Lakshadweep", score: 80, level: "high" as const, species: "Kappaphycus alvarezii", temp: "28°C", salinity: "34 ppt", advisory: "Expanding cultivation zone", lat: 22, top: 62 },
  { name: "Andaman Islands", score: 72, level: "moderate" as const, species: "Ulva lactuca", temp: "27.5°C", salinity: "31 ppt", advisory: "Seasonal limitations in monsoon", lat: 95, top: 50 },
  { name: "Gulf of Kachchh", score: 68, level: "moderate" as const, species: "Sargassum wightii", temp: "26°C", salinity: "37 ppt", advisory: "High salinity may limit species", lat: 18, top: 35 },
  { name: "Chilika Lake", score: 55, level: "low" as const, species: "Gracilaria edulis", temp: "27°C", salinity: "22 ppt", advisory: "Brackish water conditions", lat: 72, top: 42 },
  { name: "Ratnagiri", score: 64, level: "moderate" as const, species: "Ulva lactuca", temp: "27°C", salinity: "34 ppt", advisory: "Post-monsoon window recommended", lat: 30, top: 52 },
  { name: "Karwar", score: 60, level: "moderate" as const, species: "Sargassum wightii", temp: "27.5°C", salinity: "33 ppt", advisory: "Limited to calm season", lat: 28, top: 56 },
  { name: "Kollam", score: 76, level: "high" as const, species: "Gracilaria edulis", temp: "28°C", salinity: "33 ppt", advisory: "Strong backwater potential", lat: 35, top: 75 },
];

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

export default function SiteIntelligence() {
  const [selected, setSelected] = useState<typeof regions[0] | null>(null);

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Site Intelligence</h1>
          <p className="text-muted-foreground text-sm">Regional suitability analysis across India's coastline</p>
        </motion.div>

        {/* Legend */}
        <div className="flex gap-4 text-xs font-medium">
          {(["high", "moderate", "low"] as const).map(l => (
            <div key={l} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${levelColors[l]}`} />
              <span className="text-muted-foreground">{levelLabels[l]}</span>
            </div>
          ))}
        </div>

        <div className="grid lg:grid-cols-5 gap-6">
          {/* Region list */}
          <div className="lg:col-span-2 space-y-3">
            {regions.map((r, i) => (
              <motion.button
                key={r.name}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => setSelected(r)}
                className={`w-full text-left glass-strong rounded-2xl p-4 hover:-translate-y-0.5 transition-all duration-200 ${
                  selected?.name === r.name ? "ring-2 ring-primary/30 glow-sm" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${levelColors[r.level]}`} />
                    <div>
                      <p className="font-semibold text-foreground text-sm">{r.name}</p>
                      <p className="text-xs text-muted-foreground italic">{r.species}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold gradient-text">{r.score}%</span>
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  </div>
                </div>
              </motion.button>
            ))}
          </div>

          {/* Detail panel */}
          <div className="lg:col-span-3">
            {selected ? (
              <motion.div
                key={selected.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-strong rounded-3xl p-6 space-y-5 sticky top-6"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-xl font-bold text-foreground">{selected.name}</h3>
                    <span className={`inline-block mt-1 text-xs font-medium px-2.5 py-0.5 rounded-full ${
                      selected.level === "high" ? "bg-ocean-100 text-ocean-600" :
                      selected.level === "moderate" ? "bg-secondary text-secondary-foreground" :
                      "bg-muted text-muted-foreground"
                    }`}>
                      {levelLabels[selected.level]}
                    </span>
                  </div>
                  <div className="text-right">
                    <p className="text-3xl font-bold gradient-text">{selected.score}%</p>
                    <p className="text-xs text-muted-foreground">Suitability Score</p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
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
                    <p className="text-sm font-semibold text-foreground italic">{selected.species.split(" ")[0]}</p>
                    <p className="text-xs text-muted-foreground">Top Species</p>
                  </div>
                </div>

                <div className="glass rounded-2xl p-4">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Seasonal Advisory</p>
                  <p className="text-sm text-foreground">{selected.advisory}</p>
                </div>

                <div className="glass rounded-2xl p-4">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Recommended Species</p>
                  <p className="text-sm font-medium text-foreground italic">{selected.species}</p>
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
