import express from "express";
import { authRequired } from "../middleware/auth.js";
import { PredictionSubmission } from "../models/PredictionSubmission.js";
import { ChatSession } from "../models/ChatSession.js";
import { config } from "../config.js";

const router = express.Router();

function withTimeout(ms, promise) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

async function probeJson(url) {
  if (!url) return { ok: false, detail: "not_configured" };
  try {
    const res = await withTimeout(config.aiTimeoutMs || 12000, fetch(url));
    if (!res.ok) return { ok: false, detail: `http_${res.status}` };
    const out = await res.json();
    return { ok: true, detail: "ok", payload: out };
  } catch (err) {
    return { ok: false, detail: err instanceof Error ? err.message : "probe_failed" };
  }
}

function topScoredSpecies(prediction = null) {
  const species = Array.isArray(prediction?.species) ? prediction.species : [];
  return (
    species
      .filter((s) => Number.isFinite(Number(s?.probabilityPercent)))
      .sort((a, b) => Number(b?.probabilityPercent || 0) - Number(a?.probabilityPercent || 0))[0] || null
  );
}

function effectiveRecommendation(prediction = null) {
  const final = prediction?.finalRecommendation || null;
  if (final && String(final?.speciesId || "").toLowerCase() !== "insufficient_data") {
    return final;
  }
  const best = prediction?.bestSpecies || null;
  const top = topScoredSpecies(prediction);
  return best?.actionability === "insufficient_data" ? top || best : best || top;
}

router.get("/summary", authRequired, async (req, res) => {
  const userId = req.user.id;
  const [submissions, sessionsCount] = await Promise.all([
    PredictionSubmission.find({ userId }).sort({ createdAt: -1 }).limit(200).lean(),
    ChatSession.countDocuments({ userId }),
  ]);

  const avgValues = submissions
    .map((s) => effectiveRecommendation(s.prediction)?.probabilityPercent)
    .filter((v) => typeof v === "number");
  const avgConfidence = avgValues.length
    ? avgValues.reduce((a, b) => a + b, 0) / avgValues.length
    : null;

  const speciesCounts = {};
  for (const s of submissions) {
    const name = effectiveRecommendation(s.prediction)?.displayName || "Unknown";
    speciesCounts[name] = (speciesCounts[name] || 0) + 1;
  }
  const topSpecies = Object.entries(speciesCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "-";

  const last24h = Date.now() - 24 * 60 * 60 * 1000;
  const predictions24h = submissions.filter((s) => new Date(s.createdAt).getTime() >= last24h).length;

  return res.json({
    totals: {
      predictions: submissions.length,
      predictions24h,
      sessions: sessionsCount,
    },
    metrics: {
      avgConfidence,
      topSpecies,
    },
    updatedAt: new Date().toISOString(),
  });
});

router.get("/activity", authRequired, async (req, res) => {
  const userId = req.user.id;
  const submissions = await PredictionSubmission.find({ userId }).sort({ createdAt: -1 }).limit(20).lean();

  const predictions = submissions.map((s) => {
    const chosen = effectiveRecommendation(s.prediction);
    const prob = Number(chosen?.probabilityPercent || 0);
    const score = Math.max(0, Math.min(100, Math.round(prob)));
    const status = score >= 80 ? "Optimal" : score >= 65 ? "Good" : score >= 50 ? "Moderate" : "Fair";
    return {
      id: s._id.toString(),
      createdAt: s.createdAt,
      location: s.locationName || `${Number(s.lat).toFixed(3)}, ${Number(s.lon).toFixed(3)}`,
      species: chosen?.displayName || "Unknown",
      score,
      status,
    };
  });

  return res.json({
    predictions,
    updatedAt: new Date().toISOString(),
  });
});

router.get("/health", authRequired, async (_req, res) => {
  const modelProbe = await probeJson(`${String(config.modelApiUrl || "").replace(/\/$/, "")}/health`);
  const aiProbe = await probeJson(`${String(config.langGraphApiUrl || "").replace(/\/$/, "")}/health`);
  const backend = { ok: true, detail: "ok" };

  return res.json({
    backend,
    modelApi: { ok: modelProbe.ok, detail: modelProbe.detail },
    aiGateway: { ok: aiProbe.ok, detail: aiProbe.detail },
    updatedAt: new Date().toISOString(),
  });
});

export default router;
