import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { FileText, Download, Calendar, TrendingUp, Leaf, Database, MapPin, Waves } from "lucide-react";
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

  const topSpecies = useMemo(() => {
    const counts = new Map<string, number>();
    for (const s of submissions) {
      const name = s.bestSpecies?.displayName?.trim();
      if (!name) continue;
      counts.set(name, (counts.get(name) || 0) + 1);
    }
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    return sorted[0]?.[0] || "-";
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
      <div className="max-w-6xl mx-auto space-y-5 sm:space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-3xl border border-[#a9cadf]/70 bg-[linear-gradient(135deg,rgba(17,66,98,0.95)_0%,rgba(20,89,132,0.88)_55%,rgba(14,57,91,0.95)_100%)] px-5 py-5 sm:px-6 sm:py-6 shadow-[0_24px_44px_-30px_rgba(11,41,63,0.85)]"
        >
          <div className="pointer-events-none absolute -right-14 -top-12 h-44 w-44 rounded-full bg-cyan-300/20 blur-3xl" />
          <div className="pointer-events-none absolute bottom-0 left-0 h-px w-full bg-gradient-to-r from-transparent via-cyan-200/60 to-transparent" />
          <div className="relative z-10 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.14em] text-cyan-100/80">Saved Results</p>
              <h1 className="mt-1 text-2xl sm:text-3xl font-semibold text-white">Prediction Reports</h1>
              <p className="mt-2 max-w-2xl text-sm text-cyan-50/85">
                Review performance history, inspect top-performing species, and export your prediction archive for external workflows.
              </p>
            </div>
            <div className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-xs text-cyan-50">
              <Calendar className="h-3.5 w-3.5 text-cyan-200" />
              Last sync {new Date().toLocaleString()}
            </div>
          </div>
        </motion.div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-2xl border border-[#bfd8e8] bg-white/95 p-4 shadow-[0_16px_28px_-24px_rgba(15,74,109,0.9)]">
            <p className="text-xs text-slate-500">Saved Predictions</p>
            <p className="mt-1 text-2xl font-semibold text-[#0f2e47]">{submissions.length}</p>
            <p className="mt-1 text-[11px] text-slate-500">Historical suitability records</p>
          </div>
          <div className="rounded-2xl border border-[#bfd8e8] bg-white/95 p-4 shadow-[0_16px_28px_-24px_rgba(15,74,109,0.9)]">
            <p className="text-xs text-slate-500">Average Confidence</p>
            <p className="mt-1 text-2xl font-semibold text-[#0f2e47]">{avgProb === null ? "-" : `${avgProb.toFixed(1)}%`}</p>
            <p className="mt-1 text-[11px] text-slate-500">Across all completed runs</p>
          </div>
          <div className="rounded-2xl border border-[#bfd8e8] bg-white/95 p-4 shadow-[0_16px_28px_-24px_rgba(15,74,109,0.9)]">
            <p className="text-xs text-slate-500">Top Recommended Species</p>
            <p className="mt-1 text-xl font-semibold text-[#0f2e47]">{topSpecies}</p>
            <p className="mt-1 text-[11px] text-slate-500">Highest recurrence in reports</p>
          </div>
          <div className="rounded-2xl border border-[#bfd8e8] bg-white/95 p-4 shadow-[0_16px_28px_-24px_rgba(15,74,109,0.9)]">
            <p className="text-xs text-slate-500">Export Type</p>
            <p className="mt-1 text-2xl font-semibold text-[#0f2e47]">CSV</p>
            <p className="mt-1 text-[11px] text-slate-500">API-ready structured output</p>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-3xl border border-[#b9d3e4] bg-white/95 p-5 sm:p-6 shadow-[0_24px_38px_-30px_rgba(15,74,109,0.88)]"
        >
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-[#0f2e47]">
            <Leaf className="h-5 w-5 text-cyan-700" /> Export Center
          </h2>
          <div className="flex flex-wrap gap-2.5 sm:gap-3">
            <Button variant="hero-outline" onClick={exportCsv} disabled={submissions.length === 0} className="min-h-11">
              <Download className="w-3.5 h-3.5" /> Download CSV
            </Button>
            <Button variant="ghost" disabled className="min-h-11 border border-[#c9deec]">
              <FileText className="w-3.5 h-3.5" /> PDF export (next)
            </Button>
          </div>
          {submissions.length === 0 && !loading && <p className="mt-3 text-xs text-slate-500">No data yet. Run predictions first.</p>}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-3xl border border-[#b9d3e4] bg-white/95 p-5 sm:p-6 shadow-[0_24px_38px_-30px_rgba(15,74,109,0.88)]"
        >
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-[#0f2e47]">
            <Database className="h-5 w-5 text-cyan-700" /> Latest Saved Results
          </h2>

          <div className="grid gap-2.5">
            {submissions.slice(0, 12).map((s) => (
              <div
                key={s.id}
                className="grid gap-2.5 rounded-2xl border border-[#c9deec] bg-[linear-gradient(180deg,rgba(248,252,255,0.98)_0%,rgba(240,248,253,0.94)_100%)] px-4 py-3 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-center"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-[#123e63]">{s.locationName || `${s.lat.toFixed(3)}, ${s.lon.toFixed(3)}`}</p>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                    <span className="inline-flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      {s.lat.toFixed(3)}, {s.lon.toFixed(3)}
                    </span>
                    <span className="h-3 w-px bg-slate-300" />
                    <span className="inline-flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {new Date(s.createdAt).toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="sm:text-right">
                  <p className="text-sm font-semibold text-[#0f2e47]">{s.bestSpecies?.displayName || "-"}</p>
                  <p className="text-[11px] text-slate-500">Recommended species</p>
                </div>
                <div className="inline-flex items-center gap-1 rounded-full border border-cyan-200 bg-cyan-50 px-2.5 py-1 text-xs font-semibold text-cyan-800 sm:justify-self-end">
                  <TrendingUp className="h-3.5 w-3.5" />
                  {s.bestSpecies?.probabilityPercent ?? "-"}%
                </div>
              </div>
            ))}
            {!loading && submissions.length === 0 && (
              <div className="rounded-2xl border border-dashed border-cyan-200/80 bg-cyan-50/40 px-4 py-8 text-center">
                <Waves className="mx-auto mb-2 h-5 w-5 text-cyan-700" />
                <p className="text-sm font-medium text-[#123e63]">No saved results yet</p>
                <p className="mt-1 text-xs text-slate-500">Run a prediction from Check My Location to populate this ledger.</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}
