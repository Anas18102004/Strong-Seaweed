import express from "express";
import { authRequired } from "../middleware/auth.js";
import { predictSpeciesAtPoint } from "../services/predictService.js";
import { PredictionSubmission } from "../models/PredictionSubmission.js";

const router = express.Router();

function toOptionalNumber(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function normalizeFormInput(input = {}) {
  const overrides = input.overrides || {};
  const advanced = input.advanced || {};
  return {
    locationName: String(input.locationName || "").trim(),
    season: String(input.season || "").trim(),
    depthM: toOptionalNumber(input.depthM),
    overrides: {
      temperatureC: toOptionalNumber(overrides.temperatureC),
      salinityPpt: toOptionalNumber(overrides.salinityPpt),
    },
    advanced: {
      ph: toOptionalNumber(advanced.ph),
      turbidityNtu: toOptionalNumber(advanced.turbidityNtu),
      currentVelocityMs: toOptionalNumber(advanced.currentVelocityMs),
      waveHeightM: toOptionalNumber(advanced.waveHeightM),
      rainfallMm: toOptionalNumber(advanced.rainfallMm),
      tidalAmplitudeM: toOptionalNumber(advanced.tidalAmplitudeM),
    },
  };
}

router.post("/species", authRequired, async (req, res) => {
  const lat = Number(req.body?.lat);
  const lon = Number(req.body?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return res.status(400).json({ error: "numeric lat and lon are required" });
  }

  try {
    const formInput = normalizeFormInput(req.body?.formInput);
    const out = await predictSpeciesAtPoint(lat, lon);

    await PredictionSubmission.create({
      userId: req.user.id,
      lat,
      lon,
      ...formInput,
      prediction: out,
    });

    return res.json(out);
  } catch (err) {
    return res.status(500).json({ error: err instanceof Error ? err.message : "prediction failed" });
  }
});

router.get("/submissions/me", authRequired, async (req, res) => {
  const limit = Math.min(100, Math.max(1, Number(req.query.limit) || 20));
  const submissions = await PredictionSubmission.find({ userId: req.user.id })
    .sort({ createdAt: -1 })
    .limit(limit)
    .lean();
  res.json({
    total: submissions.length,
    submissions: submissions.map((s) => ({
      id: s._id.toString(),
      lat: s.lat,
      lon: s.lon,
      locationName: s.locationName || "",
      season: s.season || "",
      createdAt: s.createdAt,
      bestSpecies: s.prediction?.bestSpecies || null,
    })),
  });
});

export default router;
