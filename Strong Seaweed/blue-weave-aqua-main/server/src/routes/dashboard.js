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

router.get("/summary", authRequired, async (req, res) => {
  const userId = req.user.id;
  const [submissions, sessionsCount] = await Promise.all([
    PredictionSubmission.find({ userId }).sort({ createdAt: -1 }).limit(200).lean(),
    ChatSession.countDocuments({ userId }),
  ]);

  const avgValues = submissions
    .map((s) => s.prediction?.bestSpecies?.probabilityPercent)
    .filter((v) => typeof v === "number");
  const avgConfidence = avgValues.length
    ? avgValues.reduce((a, b) => a + b, 0) / avgValues.length
    : null;

  const speciesCounts = {};
  for (const s of submissions) {
    const name = s.prediction?.bestSpecies?.displayName || "Unknown";
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
    const prob = Number(s.prediction?.bestSpecies?.probabilityPercent || 0);
    const score = Math.max(0, Math.min(100, Math.round(prob)));
    const status = score >= 80 ? "Optimal" : score >= 65 ? "Good" : score >= 50 ? "Moderate" : "Fair";
    return {
      id: s._id.toString(),
      createdAt: s.createdAt,
      location: s.locationName || `${Number(s.lat).toFixed(3)}, ${Number(s.lon).toFixed(3)}`,
      species: s.prediction?.bestSpecies?.displayName || "Unknown",
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
