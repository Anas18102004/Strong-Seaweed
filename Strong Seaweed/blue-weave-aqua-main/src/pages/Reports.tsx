import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { FileText, Download, Calendar, TrendingUp, Leaf, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, PredictionSubmissionItem } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useEffect, useMemo, useState } from "react";

function toCsv(rows: PredictionSubmissionItem[]) {
  const headers = ["createdAt", "locationName", "lat", "lon", "bestSpecies", "probabilityPercent", "priority"];
  const lines = rows.map((r) => [
    r.createdAt,
    r.locationName || "",
    String(r.lat),
    String(r.lon),
    r.bestSpecies?.displayName || "",
    String(r.bestSpecies?.probabilityPercent ?? ""),
    r.bestSpecies?.priority || "",
  ]);
  return [headers.join(","), ...lines.map((x) => x.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","))].join("\n");
}

export default function Reports() {
  const { token } = useAuth();
  const [submissions, setSubmissions] = useState<PredictionSubmissionItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      if (!token) return;
      try {
        const out = await api.mySubmissions(token, 100);
        if (!mounted) return;
        setSubmissions(out.submissions || []);
      } catch {
        if (!mounted) return;
        setSubmissions([]);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, [token]);

  const avgProb = useMemo(() => {
    const vals = submissions.map((s) => s.bestSpecies?.probabilityPercent).filter((v): v is number => typeof v === "number");
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [submissions]);

  const exportCsv = () => {
    const csv = toCsv(submissions);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `blueweave_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <DashboardLayout>
      <div className="max-w-5xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">Reports</h1>
          <p className="text-muted-foreground text-sm">Realtime exports and summaries from your saved predictions</p>
        </motion.div>

        <div className="grid sm:grid-cols-3 gap-4">
          <div className="glass-strong rounded-2xl p-4">
            <p className="text-xs text-muted-foreground">Saved Predictions</p>
            <p className="text-2xl font-bold text-foreground">{submissions.length}</p>
          </div>
          <div className="glass-strong rounded-2xl p-4">
            <p className="text-xs text-muted-foreground">Average Confidence</p>
            <p className="text-2xl font-bold text-foreground">{avgProb === null ? "-" : `${avgProb.toFixed(1)}%`}</p>
          </div>
          <div className="glass-strong rounded-2xl p-4">
            <p className="text-xs text-muted-foreground">Export Type</p>
            <p className="text-2xl font-bold text-foreground">CSV</p>
          </div>
        </div>

        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="glass-strong rounded-3xl p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Leaf className="w-5 h-5 text-primary" /> Generate Realtime Export
          </h2>
          <div className="flex flex-wrap gap-3">
            <Button variant="hero-outline" onClick={exportCsv} disabled={submissions.length === 0}>
              <Download className="w-3.5 h-3.5" /> Download CSV
            </Button>
            <Button variant="ghost" disabled>
              <FileText className="w-3.5 h-3.5" /> PDF export (next)
            </Button>
          </div>
          {submissions.length === 0 && !loading && <p className="text-xs text-muted-foreground mt-3">No data yet. Run predictions first.</p>}
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-strong rounded-3xl p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Database className="w-5 h-5 text-primary" /> Latest Saved Results
          </h2>
          <div className="space-y-3">
            {submissions.slice(0, 10).map((s, i) => (
              <div key={s.id} className="flex items-center justify-between glass rounded-2xl px-4 py-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{s.locationName || `${s.lat.toFixed(3)}, ${s.lon.toFixed(3)}`}</p>
                  <p className="text-xs text-muted-foreground">{new Date(s.createdAt).toLocaleString()}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-foreground">{s.bestSpecies?.displayName || "-"}</p>
                  <p className="text-xs text-muted-foreground">{s.bestSpecies?.probabilityPercent ?? "-"}%</p>
                </div>
              </div>
            ))}
            {!loading && submissions.length === 0 && <p className="text-sm text-muted-foreground">No exports yet.</p>}
          </div>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}