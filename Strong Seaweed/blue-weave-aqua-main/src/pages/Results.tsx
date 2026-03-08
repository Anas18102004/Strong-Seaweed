import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { Award, AlertTriangle, CloudRain, Wind, ChevronDown, Download, MapPin, Database, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import type { SpeciesPredictionResponse } from "@/lib/api";

const risks = [
  { label: "Cyclone Risk", value: "Low", icon: Wind, color: "text-ocean-500" },
  { label: "Monsoon Impact", value: "Moderate", icon: CloudRain, color: "text-yellow-500" },
  { label: "Environmental Stress", value: "Low", icon: AlertTriangle, color: "text-ocean-400" },
];

function ScoreRing({ score, color, size = 80 }: { score: number; color: string; size?: number }) {
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;

  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(190, 214, 229, 0.82)" strokeWidth="6" />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={circ}
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
      />
    </svg>
  );
}

function modelBadgeFromReason(reason: string, pending: boolean): { label: string; className: string } {
  if (pending && reason === "out_of_coverage") {
    return { label: "Out of Coverage", className: "border border-amber-200 bg-amber-50 text-amber-800" };
  }
  if (pending && reason === "model_not_trained") {
    return { label: "Pending", className: "border border-slate-200 bg-slate-100 text-slate-700" };
  }
  if (reason === "dedicated_production_model") {
    return { label: "Production", className: "border border-emerald-200 bg-emerald-50 text-emerald-800" };
  }
  if (reason.startsWith("genus_proxy_")) {
    return { label: "Genus Proxy", className: "border border-cyan-200 bg-cyan-50 text-cyan-800" };
  }
  if (pending) {
    return { label: "Unavailable", className: "border border-slate-200 bg-slate-100 text-slate-700" };
  }
  return { label: "Model", className: "border border-blue-200 bg-blue-50 text-blue-800" };
}

function reasonText(reason: string): string {
  switch (reason) {
    case "dedicated_production_model":
      return "Dedicated, validated species model.";
    case "genus_proxy_positive":
      return "Proxy model indicates ecological suitability.";
    case "genus_proxy_negative":
      return "Proxy model indicates low suitability.";
    case "out_of_coverage":
      return "Point is outside this species model coverage.";
    case "model_not_trained":
      return "Species model is not trained yet.";
    default:
      return reason.replace(/_/g, " ");
  }
}

function sanitizeAdvisoryText(text: string): string {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => {
      const t = line.trim();
      if (!t) return true;
      if (/^[-•]?\s*ask\s*["']?e["']?\s*$/i.test(t)) return false;
      if (/^[-•]?\s*ask\s*["']?expand["']?.*$/i.test(t)) return false;
      return true;
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function splitModelRelease(modelRelease: string): { primary: string; secondary: string | null } {
  const text = String(modelRelease || "").trim();
  if (!text) return { primary: "unknown", secondary: null };
  const [first, ...rest] = text.split("+");
  return { primary: first || "unknown", secondary: rest.length ? rest.join("+") : null };
}

function downloadTextFile(content: string, fileName: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function toReportMarkdown(prediction: SpeciesPredictionResponse) {
  const lines: string[] = [];
  lines.push("# Akuara Prediction Report");
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Location: ${prediction.input.lat}, ${prediction.input.lon}`);
  lines.push(`Model release: ${prediction.modelRelease}`);
  lines.push(`Source: ${prediction.source}`);
  lines.push("");
  lines.push("## Final Recommendation");
  const final = prediction.finalRecommendation;
  if (final) {
    lines.push(`- Species: ${final.displayName}`);
    lines.push(`- Probability: ${final.probabilityPercent ?? "n/a"}%`);
    lines.push(`- Actionability: ${final.actionability}`);
    lines.push(`- Decision source: ${final.source}`);
    lines.push(`- Verification: ${final.verificationVerdict} (${final.verificationConfidenceScore}%)`);
  } else if (prediction.bestSpecies) {
    lines.push(`- Species: ${prediction.bestSpecies.displayName}`);
    lines.push(`- Probability: ${prediction.bestSpecies.probabilityPercent ?? "n/a"}%`);
    lines.push(`- Actionability: ${prediction.bestSpecies.actionability || "insufficient_data"}`);
  } else {
    lines.push("- No recommendation available.");
  }
  lines.push("");
  lines.push("## Species Scores");
  for (const s of prediction.species || []) {
    lines.push(`- ${s.displayName}: ${s.probabilityPercent ?? "n/a"}% | ${s.actionability || "insufficient_data"} | ${s.reason}`);
  }
  if (Array.isArray(prediction.warnings) && prediction.warnings.length > 0) {
    lines.push("");
    lines.push("## Warnings");
    for (const w of prediction.warnings) lines.push(`- ${w}`);
  }
  if (prediction.fallbackAdvisory?.answer) {
    lines.push("");
    lines.push("## Fallback Advisory");
    lines.push(sanitizeAdvisoryText(prediction.fallbackAdvisory.answer));
  }
  return lines.join("\n");
}

export default function ResultsPage() {
  const [showWhy, setShowWhy] = useState(false);
  const routerLocation = useLocation();
  const prediction = (routerLocation.state as { prediction?: SpeciesPredictionResponse } | null)?.prediction;

  if (!prediction) {
    return (
      <DashboardLayout>
        <div className="max-w-2xl mx-auto glass-strong rounded-3xl p-8 text-center space-y-3">
          <h1 className="text-2xl font-bold text-foreground">No prediction available</h1>
          <p className="text-sm text-muted-foreground">Run a location check first to view suitability results.</p>
          <Button asChild variant="hero">
            <Link to="/predict">Go to Check My Location</Link>
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  const colorByIndex = ["#0077FF", "#00A8E8", "#00C6FF", "#7DD6FF"];
  const finalRecommendation = prediction.finalRecommendation || null;
  const preferredSpeciesId =
    finalRecommendation && finalRecommendation.speciesId !== "insufficient_data"
      ? finalRecommendation.speciesId
      : prediction.bestSpecies?.speciesId;
  const speciesResults = prediction.species.map((species, idx) => ({
    name: species.displayName,
    score: species.probabilityPercent == null ? null : Math.round(species.probabilityPercent),
    color: colorByIndex[idx % colorByIndex.length],
    best: preferredSpeciesId === species.speciesId,
    pending: !species.ready || species.probabilityPercent == null,
    reason: species.reason,
    badge: modelBadgeFromReason(species.reason, !species.ready || species.probabilityPercent == null),
    actionability: species.actionability || "insufficient_data",
  }));

  const scoredByRank = speciesResults
    .filter((item): item is (typeof speciesResults)[number] & { score: number } => typeof item.score === "number")
    .sort((a, b) => b.score - a.score);
  const modelBest = speciesResults.find((item) => item.best) ?? scoredByRank[0] ?? speciesResults[0];
  const hasInsufficientData =
    (finalRecommendation?.actionability || prediction.bestSpecies?.actionability || "insufficient_data") ===
    "insufficient_data";
  const best = hasInsufficientData ? scoredByRank[0] ?? modelBest : modelBest;
  const bestActionability = best?.actionability || finalRecommendation?.actionability || "insufficient_data";
  const bestIsRecommended = bestActionability === "recommended" && !hasInsufficientData;
  const bestIsPilot = bestActionability === "test_pilot_only" || hasInsufficientData;
  const bestHasScore = typeof best?.score === "number";
  const bestScoreText =
    typeof best?.score === "number"
      ? best.score
      : Number.isFinite(Number(finalRecommendation?.probabilityPercent))
      ? Math.round(Number(finalRecommendation?.probabilityPercent))
      : 0;
  const topSpeciesLabel = best?.name || "No clear candidate";
  const coordText = `${prediction.input.lat.toFixed(4)}, ${prediction.input.lon.toFixed(4)}`;
  const modelReleaseInfo = splitModelRelease(prediction.modelRelease);
  const fallbackAnswer = sanitizeAdvisoryText(prediction.fallbackAdvisory?.answer || "");

  const handleExportReport = () => {
    const slug = `${prediction.input.lat.toFixed(4)}_${prediction.input.lon.toFixed(4)}`.replace(/[^\d_.-]/g, "_");
    const fileName = `akuara_report_${slug}_${new Date().toISOString().slice(0, 10)}.md`;
    downloadTextFile(toReportMarkdown(prediction), fileName);
  };

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-[30px] border border-[#9cc7e3]/75 bg-[linear-gradient(135deg,rgba(13,63,96,0.96)_0%,rgba(19,105,152,0.9)_58%,rgba(14,73,112,0.96)_100%)] px-5 py-5 text-[#ecf8ff] shadow-[0_28px_52px_-34px_rgba(10,46,70,0.88)] sm:px-6 sm:py-6"
        >
          <div className="pointer-events-none absolute -right-12 -top-10 h-40 w-40 rounded-full bg-cyan-300/25 blur-3xl" />
          <div className="pointer-events-none absolute left-10 bottom-0 h-20 w-32 bg-gradient-to-r from-cyan-200/20 to-transparent blur-2xl" />
          <div className="relative z-10">
            <p className="text-[11px] uppercase tracking-[0.14em] text-cyan-100/80">Suitability Engine Output</p>
            <h1 className="mt-1 text-2xl font-semibold leading-tight sm:text-3xl">Prediction Results</h1>
            <p className="mt-2 max-w-3xl text-sm text-cyan-50/90">
              Model output is now fully visible here with species ranking, warnings, and recommendation context. No need to switch pages.
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-white/20 bg-white/10 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.12em] text-cyan-100/85">Top Species</p>
                <p className="mt-1 text-sm font-semibold text-white">{topSpeciesLabel}</p>
              </div>
              <div className="rounded-xl border border-white/20 bg-white/10 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.12em] text-cyan-100/85">Input Coordinates</p>
                <p className="mt-1 inline-flex items-center gap-1 text-sm font-semibold text-white">
                  <MapPin className="h-3.5 w-3.5" /> {coordText}
                </p>
              </div>
              <div className="rounded-xl border border-white/20 bg-white/10 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.12em] text-cyan-100/85">Model Release</p>
                <div className="mt-1 flex min-w-0 items-start gap-1 text-sm font-semibold text-white">
                  <Database className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <div className="min-w-0">
                    <p className="truncate text-[13px] leading-snug" title={modelReleaseInfo.primary}>
                      {modelReleaseInfo.primary}
                    </p>
                    {modelReleaseInfo.secondary ? (
                      <p className="truncate text-[12px] leading-snug text-cyan-100/90" title={modelReleaseInfo.secondary}>
                        + {modelReleaseInfo.secondary}
                      </p>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {speciesResults.map((s, i) => (
            <motion.div
              key={s.name}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 + i * 0.1 }}
              className={`rounded-3xl border bg-[linear-gradient(180deg,rgba(255,255,255,0.96)_0%,rgba(241,249,255,0.93)_100%)] p-5 text-center shadow-[0_18px_30px_-26px_rgba(14,62,94,0.82)] ${
                s.best ? "border-cyan-300 ring-2 ring-cyan-200/80" : "border-[#c2d9ea]"
              }`}
            >
              {s.pending ? (
                <div className="h-20 flex flex-col items-center justify-center text-xs text-[#527085] gap-1">
                  <span>{s.badge.label}</span>
                  <span className="text-[11px] leading-relaxed">{reasonText(s.reason)}</span>
                </div>
              ) : (
                <div className="flex justify-center mb-3 relative">
                  <ScoreRing score={s.score ?? 0} color={s.color} />
                  <span className="absolute inset-0 flex items-center justify-center text-xl font-bold text-[#0f2e45] rotate-0">{s.score}%</span>
                </div>
              )}
              <p className="text-[1.18rem] leading-tight font-semibold text-[#112f45] min-h-[2.25rem]">{s.name}</p>
              <span className={`inline-flex mt-2 text-[11px] px-2.5 py-0.5 rounded-full font-semibold ${s.badge.className}`}>{s.badge.label}</span>
              {s.best && !s.pending && (
                <span className="inline-flex items-center gap-1 mt-2 text-xs font-medium bg-gradient-to-r from-[#1da1f2] to-[#0ea5e9] text-white px-2.5 py-0.5 rounded-full">
                  <Award className="w-3 h-3" /> Best Match
                </span>
              )}
            </motion.div>
          ))}
        </div>

        {prediction.warnings.length > 0 && (
          <div className="rounded-2xl border border-amber-300 bg-[linear-gradient(180deg,rgba(255,251,235,0.98)_0%,rgba(254,243,199,0.92)_100%)] p-4 text-sm text-amber-900 shadow-[0_14px_24px_-20px_rgba(120,53,15,0.6)]">
            <p className="font-semibold mb-1">Model Warnings</p>
            <ul className="space-y-1">
              {prediction.warnings.map((w) => (
                <li key={w}>- {w.replace(/_/g, " ")}</li>
              ))}
            </ul>
          </div>
        )}

        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="rounded-3xl border border-[#b7d4e8] bg-[linear-gradient(180deg,rgba(255,255,255,0.97)_0%,rgba(241,249,255,0.94)_100%)] p-6 shadow-[0_24px_36px_-30px_rgba(13,63,96,0.9)]"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#1da1f2] to-[#0e7fd1] flex items-center justify-center shrink-0 shadow-[0_14px_24px_-16px_rgba(8,84,139,0.75)]">
              <Award className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <p className="text-xs font-semibold text-[#5c7f96] uppercase tracking-wider mb-1">AI Recommendation</p>
              <h3 className="text-3xl font-bold text-[#102f46] leading-tight mb-2">
                {bestIsRecommended
                  ? <>Recommended for cultivation: <span className="gradient-text">{best.name}</span></>
                  : bestIsPilot
                  ? <>Pilot-only candidate: <span className="gradient-text">{best?.name || "No clear candidate"}</span></>
                  : <>No cultivation recommendation at this location</>}
              </h3>
              <p className="text-[1.1rem] text-[#3f637a] leading-relaxed mb-3">
                {bestIsRecommended
                  ? <>Best species score at this site is <strong className="text-[#0f2e47]">{bestScoreText}%</strong>.</>
                  : bestIsPilot && bestHasScore
                  ? <>Top available species score is <strong className="text-[#0f2e47]">{best.score}%</strong>. Confidence is limited, so start with a small pilot and verify in field conditions.</>
                  : <>Top ranked model score is <strong className="text-[#0f2e47]">{bestScoreText}%</strong>, below strict cultivation threshold.</>}
                {prediction.nearestGrid && (
                  <>
                    {" "}Nearest model grid is <strong className="text-[#0f2e47]">{prediction.nearestGrid.distance_km.toFixed(2)} km</strong> away.
                  </>
                )}
              </p>
              {prediction.verification && (
                <p className="mb-3 inline-flex items-center gap-1 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-800">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  Verification: {prediction.verification.verdict} ({prediction.verification.confidenceScore}%)
                </p>
              )}
              {finalRecommendation && (
                <p className="mb-3 text-xs text-[#5e7f95]">
                  Decision source: {finalRecommendation.source.replace(/_/g, " ")}
                  {finalRecommendation.selectionReason ? ` | ${finalRecommendation.selectionReason.replace(/_/g, " ")}` : ""}
                  {finalRecommendation.consensusTier ? ` | ${finalRecommendation.consensusTier.replace(/_/g, " ")}` : ""}
                  {finalRecommendation.tieResolved ? " | tie resolved with guardrail" : ""}
                  {finalRecommendation.tieDetected && !finalRecommendation.tieResolved ? " | tie detected" : ""}
                  {finalRecommendation.disagreementWithAgent ? " | model and advisory conflict auto-resolved by verification" : ""}.
                </p>
              )}
              {fallbackAnswer && (
                <div className="mb-3 rounded-xl border border-cyan-200 bg-cyan-50 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-cyan-800">Fallback Advisory</p>
                  <p className="mt-1 whitespace-pre-line text-sm text-cyan-900">{fallbackAnswer}</p>
                </div>
              )}
              <button
                onClick={() => setShowWhy(!showWhy)}
                className="flex items-center gap-1 text-sm font-semibold text-[#0f8adf] hover:text-[#0b6db1] transition-colors"
              >
                <ChevronDown className={`w-4 h-4 transition-transform ${showWhy ? "rotate-180" : ""}`} />
                Why this species?
              </button>
              {showWhy && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="text-sm text-[#4c6e84] mt-3 pl-5 border-l-2 border-[#c4dced] leading-relaxed"
                >
                  This output combines one dedicated production model (Kappaphycus) and additional genus-proxy species models.
                  Always field-verify locations before deployment decisions.
                </motion.p>
              )}
            </div>
          </div>
        </motion.div>

        <div className="grid sm:grid-cols-3 gap-4">
          {risks.map((r, i) => (
            <motion.div
              key={r.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.08 }}
              className="rounded-3xl border border-[#c2daeb] bg-[linear-gradient(180deg,rgba(255,255,255,0.96)_0%,rgba(240,248,253,0.92)_100%)] p-5 text-center shadow-[0_18px_30px_-26px_rgba(14,62,94,0.82)]"
            >
              <r.icon className={`w-6 h-6 mx-auto mb-2 ${r.color}`} />
              <p className="text-4xl font-bold leading-none text-[#0f2e47]">{r.value}</p>
              <p className="mt-1 text-sm text-[#5c7f96]">{r.label}</p>
            </motion.div>
          ))}
        </div>

        <div className="flex justify-end">
          <Button
            size="lg"
            onClick={handleExportReport}
            className="rounded-xl bg-gradient-to-r from-[#1da1f2] to-[#0ea5e9] text-white shadow-[0_18px_30px_-20px_rgba(14,126,187,0.8)] hover:opacity-95"
          >
            <Download className="w-4 h-4" /> Export Report
          </Button>
        </div>
      </div>
    </DashboardLayout>
  );
}
