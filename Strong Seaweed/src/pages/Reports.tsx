import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { FileText, Download, Calendar, Building2, TrendingUp, Leaf } from "lucide-react";
import { Button } from "@/components/ui/button";

const reports = [
  { title: "Cultivation Suitability Report", desc: "Full species suitability analysis with environmental parameters", icon: Leaf, date: "Feb 25, 2026", type: "PDF" },
  { title: "Government Compliance Report", desc: "Environmental compliance and regulatory readiness assessment", icon: Building2, date: "Feb 22, 2026", type: "PDF" },
  { title: "Investment Viability Report", desc: "Financial projections, ROI analysis, and market opportunity", icon: TrendingUp, date: "Feb 20, 2026", type: "PDF" },
  { title: "Seasonal Forecast Summary", desc: "Quarterly cultivation forecast with risk indicators", icon: Calendar, date: "Feb 18, 2026", type: "PDF" },
];

const recentExports = [
  { name: "Gulf_of_Mannar_Suitability_Feb2026.pdf", date: "Feb 25, 2026", size: "2.4 MB" },
  { name: "Palk_Bay_Investment_Analysis.pdf", date: "Feb 22, 2026", size: "1.8 MB" },
  { name: "Q1_2026_Seasonal_Forecast.pdf", date: "Feb 18, 2026", size: "3.1 MB" },
];

export default function Reports() {
  return (
    <DashboardLayout>
      <div className="max-w-5xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Reports</h1>
          <p className="text-muted-foreground text-sm">Generate and export cultivation intelligence reports</p>
        </motion.div>

        {/* Report templates */}
        <div className="grid sm:grid-cols-2 gap-4">
          {reports.map((r, i) => (
            <motion.div
              key={r.title}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              className="glass-strong rounded-3xl p-6 flex gap-4"
            >
              <div className="w-12 h-12 rounded-2xl gradient-primary flex items-center justify-center shrink-0">
                <r.icon className="w-6 h-6 text-primary-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-foreground mb-1">{r.title}</h3>
                <p className="text-xs text-muted-foreground mb-3">{r.desc}</p>
                <Button variant="hero-outline" size="sm">
                  <Download className="w-3.5 h-3.5" /> Generate {r.type}
                </Button>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Recent exports */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-strong rounded-3xl p-6"
        >
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" /> Recent Exports
          </h2>
          <div className="space-y-3">
            {recentExports.map((f, i) => (
              <div key={i} className="flex items-center justify-between glass rounded-2xl px-4 py-3">
                <div className="flex items-center gap-3 min-w-0">
                  <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{f.name}</p>
                    <p className="text-xs text-muted-foreground">{f.date} · {f.size}</p>
                  </div>
                </div>
                <Button variant="ghost" size="icon" className="shrink-0">
                  <Download className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}
