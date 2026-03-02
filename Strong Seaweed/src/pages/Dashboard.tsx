import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { TrendingUp, Droplets, Thermometer, Wind, MapPin, Activity } from "lucide-react";

const stats = [
  { label: "Active Predictions", value: "24", icon: Activity, change: "+3 today" },
  { label: "Top Species", value: "Kappaphycus", icon: Droplets, change: "87% avg" },
  { label: "Avg Temperature", value: "28.4°C", icon: Thermometer, change: "Optimal" },
  { label: "Sites Analyzed", value: "142", icon: MapPin, change: "+12 this week" },
];

const recentPredictions = [
  { location: "Gulf of Mannar", species: "Kappaphycus alvarezii", score: 87, status: "Optimal" },
  { location: "Palk Bay", species: "Gracilaria edulis", score: 81, status: "Good" },
  { location: "Lakshadweep", species: "Ulva lactuca", score: 74, status: "Moderate" },
  { location: "Andaman Islands", species: "Sargassum wightii", score: 62, status: "Fair" },
  { location: "Gulf of Kachchh", species: "Kappaphycus alvarezii", score: 79, status: "Good" },
];

export default function Dashboard() {
  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Dashboard</h1>
          <p className="text-muted-foreground text-sm">Cultivation intelligence overview</p>
        </motion.div>

        {/* Stats */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
              className="glass-strong rounded-3xl p-5"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="w-10 h-10 rounded-2xl gradient-primary flex items-center justify-center">
                  <s.icon className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="text-xs text-muted-foreground font-medium">{s.change}</span>
              </div>
              <p className="text-2xl font-bold text-foreground">{s.value}</p>
              <p className="text-sm text-muted-foreground">{s.label}</p>
            </motion.div>
          ))}
        </div>

        {/* Recent predictions */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="glass-strong rounded-3xl p-6"
        >
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-semibold text-foreground">Recent Predictions</h2>
            <TrendingUp className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted-foreground border-b border-border/50">
                  <th className="pb-3 font-medium">Location</th>
                  <th className="pb-3 font-medium">Species</th>
                  <th className="pb-3 font-medium">Score</th>
                  <th className="pb-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {recentPredictions.map((p, i) => (
                  <tr key={i} className="border-b border-border/30 last:border-0">
                    <td className="py-3 font-medium text-foreground">{p.location}</td>
                    <td className="py-3 text-muted-foreground italic">{p.species}</td>
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full gradient-primary"
                            style={{ width: `${p.score}%` }}
                          />
                        </div>
                        <span className="font-semibold gradient-text">{p.score}%</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                        p.status === "Optimal" ? "bg-ocean-100 text-ocean-600" :
                        p.status === "Good" ? "bg-ocean-50 text-ocean-500" :
                        p.status === "Moderate" ? "bg-secondary text-secondary-foreground" :
                        "bg-muted text-muted-foreground"
                      }`}>
                        {p.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}
