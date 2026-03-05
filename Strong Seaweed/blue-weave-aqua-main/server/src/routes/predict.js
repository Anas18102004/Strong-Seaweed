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

function detectSeason(now = new Date()) {
  const month = now.getUTCMonth() + 1;
  if (month >= 3 && month <= 5) return "Pre-Monsoon";
  if (month >= 6 && month <= 9) return "Monsoon";
  if (month >= 10 && month <= 11) return "Post-Monsoon";
  return "Winter";
}

async function fetchEnvironment(lat, lon) {
  const safeLat = Number(lat);
  const safeLon = Number(lon);
  if (!Number.isFinite(safeLat) || !Number.isFinite(safeLon)) {
    throw new Error("invalid_coordinates");
  }

  const url = `https://api.open-meteo.com/v1/forecast?latitude=${safeLat}&longitude=${safeLon}&current=temperature_2m`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`open_meteo_http_${res.status}`);
  const out = await res.json();
  const temp = Number(out?.current?.temperature_2m);

  // Salinity is not available from open public weather APIs directly.
  return {
    temperatureC: Number.isFinite(temp) ? temp : null,
    salinityPpt: null,
    provider: "open-meteo",
    fetchedAt: new Date().toISOString(),
    notes: Number.isFinite(temp)
      ? ["Temperature fetched live.", "Salinity requires marine sensor/model source."]
      : ["Live temperature unavailable for this location.", "Salinity requires marine sensor/model source."],
  };
}

router.get("/reference", authRequired, async (req, res) => {
  const defaultLocations = [
    "Gulf of Mannar",
    "Palk Bay",
    "Lakshadweep",
    "Andaman Islands",
    "Gulf of Kachchh",
    "Chilika",
    "Ratnagiri",
    "Karwar",
    "Kollam",
  ];

  const seasons = ["Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter"];
  const recent = await PredictionSubmission.find({ userId: req.user.id })
    .select({ locationName: 1 })
    .sort({ createdAt: -1 })
    .limit(50)
    .lean();
  const seen = new Set(defaultLocations);
  const merged = [...defaultLocations];
  for (const row of recent) {
    const name = String(row.locationName || "").trim();
    if (!name || seen.has(name)) continue;
    merged.push(name);
    seen.add(name);
  }

  res.json({
    locations: merged,
    seasons,
    currentSeason: detectSeason(new Date()),
    fetchedAt: new Date().toISOString(),
  });
});

router.get("/environment", authRequired, async (req, res) => {
  const lat = Number(req.query.lat);
  const lon = Number(req.query.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return res.status(400).json({ error: "numeric lat and lon are required" });
  }
  try {
    const out = await fetchEnvironment(lat, lon);
    return res.json(out);
  } catch (err) {
    return res.status(502).json({
      error: err instanceof Error ? err.message : "environment_fetch_failed",
      temperatureC: null,
      salinityPpt: null,
      provider: "open-meteo",
      fetchedAt: new Date().toISOString(),
      notes: ["Could not fetch live environment data."],
    });
  }
});

router.post("/species", authRequired, async (req, res) => {
  const lat = Number(req.body?.lat);
  const lon = Number(req.body?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return res.status(400).json({ error: "numeric lat and lon are required" });
  }

  try {
    const formInput = normalizeFormInput(req.body?.formInput);
    const out = await predictSpeciesAtPoint(lat, lon, formInput);

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
