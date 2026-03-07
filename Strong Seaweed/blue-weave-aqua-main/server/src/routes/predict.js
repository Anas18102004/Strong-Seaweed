import express from "express";
import { authRequired } from "../middleware/auth.js";
import { predictSpeciesAtPoint } from "../services/predictService.js";
import { runChat } from "../services/aiService.js";
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

function summarizeSpeciesForPrompt(speciesRows = []) {
  if (!Array.isArray(speciesRows) || speciesRows.length === 0) return "No species scores available.";
  return speciesRows
    .slice()
    .sort((a, b) => Number(b?.probabilityPercent || 0) - Number(a?.probabilityPercent || 0))
    .map((s) => {
      const pct = Number.isFinite(Number(s?.probabilityPercent)) ? `${Number(s.probabilityPercent).toFixed(2)}%` : "n/a";
      return `${s?.displayName || s?.speciesId || "unknown"}: score=${pct}, actionability=${s?.actionability || "unknown"}, reason=${s?.reason || "unknown"}`;
    })
    .join("; ");
}

function advisoryQuestionFromContext({ lat, lon, formInput, prediction, environment }) {
  const best = prediction?.bestSpecies || null;
  const nearestKm = prediction?.nearestGrid?.distance_km;
  const nearestText = Number.isFinite(Number(nearestKm)) ? Number(nearestKm).toFixed(2) : "n/a";
  const tempText = Number.isFinite(Number(environment?.temperatureC)) ? `${Number(environment.temperatureC).toFixed(1)} C` : "n/a";
  const salText = Number.isFinite(Number(environment?.salinityPpt)) ? `${Number(environment.salinityPpt).toFixed(1)} ppt` : "n/a";
  const speciesSummary = summarizeSpeciesForPrompt(prediction?.species || []);
  const warnings = Array.isArray(prediction?.warnings) ? prediction.warnings.join(", ") : "none";
  const cop = prediction?.copernicusContext || null;
  const copSummary = cop
    ? [
        `Copernicus salinity(mean/std/grad)=${cop?.salinity?.mean ?? "n/a"}/${cop?.salinity?.std ?? "n/a"}/${cop?.salinity?.gradient ?? "n/a"}`,
        `currents(mean/p90/grad)=${cop?.currents?.speedMean ?? "n/a"}/${cop?.currents?.speedP90 ?? "n/a"}/${cop?.currents?.gradient ?? "n/a"}`,
        `waves(mean/p95/grad)=${cop?.waves?.heightMean ?? "n/a"}/${cop?.waves?.heightP95 ?? "n/a"}/${cop?.waves?.gradient ?? "n/a"}`,
        `featureTimestamp=${cop?.featureTimestamp || "n/a"}`,
      ].join("; ")
    : "Copernicus context unavailable";

  return [
    "You are a seaweed aquaculture decision-support specialist.",
    "Use provided model/context signals as primary evidence. Do not claim certainty.",
    `Location: lat=${lat}, lon=${lon}, name=${formInput?.locationName || "unknown"}, season=${formInput?.season || "unknown"}.`,
    `Best model species: ${best?.displayName || "none"}; score=${best?.probabilityPercent ?? "n/a"}%; actionability=${best?.actionability || "unknown"}; decisionSource=${prediction?.decisionSource || "unknown"}.`,
    `Nearest model grid distance: ${nearestText} km.`,
    `Environment hints: temperature=${tempText}, salinity=${salText}, provider=${environment?.provider || "unknown"}.`,
    `Copernicus signals: ${copSummary}.`,
    `Model warnings: ${warnings}.`,
    `Species ranking summary: ${speciesSummary}`,
    "Respond in 5 bullets: 1) cultivation recommendation level, 2) why, 3) top risks, 4) field checks before farming, 5) next 7-day action plan.",
    "If confidence is low or warnings exist, explicitly say pilot-only or not recommended.",
    "Also include latest current seaweed policy/regulatory/market context from web grounding when available, and label it as external context.",
  ].join(" ");
}

function shouldGenerateFallbackAdvisory(prediction) {
  const best = prediction?.bestSpecies || null;
  const actionability = String(best?.actionability || prediction?.actionability || "insufficient_data").toLowerCase();
  const score = Number(best?.probabilityPercent);
  const warnings = Array.isArray(prediction?.warnings) ? prediction.warnings.map((w) => String(w).toLowerCase()) : [];

  if (["insufficient_data", "not_recommended", "test_pilot_only"].includes(actionability)) return true;
  if (!Number.isFinite(score)) return true;
  if (score < 45) return true;
  if (
    warnings.some((w) =>
      [
        "no_species_meets_suitability_threshold",
        "no_species_with_ready_scores",
        "no_species_meets_threshold_using_screening_fallback",
        "no_species_meets_threshold_using_ranking_fallback",
        "best_species_low_confidence",
        "best_species_ultra_low_confidence",
        "environmental_features_stale",
        "low_override_mapping_coverage",
      ].includes(w),
    )
  ) {
    return true;
  }

  return false;
}

function topScoredSpecies(prediction) {
  const list = Array.isArray(prediction?.species) ? prediction.species : [];
  return (
    list
      .filter((s) => Number.isFinite(Number(s?.probabilityPercent)))
      .sort((a, b) => Number(b?.probabilityPercent || 0) - Number(a?.probabilityPercent || 0))[0] || null
  );
}

function deterministicAdvisoryText(prediction, environment) {
  const best = prediction?.bestSpecies || null;
  const top = topScoredSpecies(prediction);
  const chosen = best?.speciesId === "insufficient_data" ? top : best || top;
  const speciesName = chosen?.displayName || "No strong species candidate";
  const scoreText = Number.isFinite(Number(chosen?.probabilityPercent)) ? `${Number(chosen.probabilityPercent).toFixed(2)}%` : "n/a";
  const tempText = Number.isFinite(Number(environment?.temperatureC)) ? `${Number(environment.temperatureC).toFixed(1)} C` : "n/a";
  const salText = Number.isFinite(Number(environment?.salinityPpt)) ? `${Number(environment.salinityPpt).toFixed(1)} ppt` : "n/a";
  const warnings = Array.isArray(prediction?.warnings) && prediction.warnings.length > 0 ? prediction.warnings.join(", ") : "none";
  return [
    `Recommendation level: pilot-only for ${speciesName} (score ${scoreText}).`,
    `Why: model confidence is limited; decision source=${prediction?.decisionSource || "unknown"}, warnings=${warnings}.`,
    `Top risks: uncertain local coverage and possible environmental mismatch.`,
    `Field checks before farming: validate salinity/temperature/depth locally (temp ${tempText}, salinity ${salText}).`,
    "Next 7 days: run a small pilot line, monitor growth/fouling daily, then rerun prediction with updated observations.",
  ].join("\n");
}

async function generateFallbackAdvisory({ userId, lat, lon, formInput, prediction, environment }) {
  const question = advisoryQuestionFromContext({ lat, lon, formInput, prediction, environment });
  try {
    const aiOut = await runChat(question, {
      userId,
      context: {
        mode: "predict_advisory",
        locationName: formInput.locationName,
        season: formInput.season,
        lat,
        lon,
        depthM: formInput.depthM,
        overrides: formInput.overrides,
        advanced: formInput.advanced,
        prediction,
        environment,
      },
    });
    const answer = String(aiOut?.answer || "").trim();
    const unusable = !answer || /live ai model is unavailable/i.test(answer);
    if (unusable) throw new Error("ai_advisory_unavailable");
    return {
      answer,
      model: aiOut?.model || "unknown",
      provider: aiOut?.provider || "unknown",
      status: aiOut?.status || "live",
      routedAgent: aiOut?.routedAgent || "copilot",
      source: "llm_plus_context",
    };
  } catch {
    return {
      answer: deterministicAdvisoryText(prediction, environment),
      model: "deterministic-fallback",
      provider: "rule-fallback",
      status: "fallback",
      routedAgent: "copilot",
      source: "rule_fallback",
    };
  }
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
    const prediction = await predictSpeciesAtPoint(lat, lon, formInput);
    let fallbackAdvisory = null;
    let advisoryFallbackUsed = false;

    if (shouldGenerateFallbackAdvisory(prediction)) {
      const environment = await fetchEnvironment(lat, lon).catch(() => ({
        temperatureC: null,
        salinityPpt: null,
        provider: "unavailable",
        fetchedAt: new Date().toISOString(),
        notes: ["Live environment data unavailable during fallback advisory generation."],
      }));
      fallbackAdvisory = await generateFallbackAdvisory({
        userId: req.user.id,
        lat,
        lon,
        formInput,
        prediction,
        environment,
      });
      advisoryFallbackUsed = true;
    }

    const out = {
      ...prediction,
      advisoryFallbackUsed,
      fallbackAdvisory,
    };

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

router.post("/species/advisory", authRequired, async (req, res) => {
  const lat = Number(req.body?.lat);
  const lon = Number(req.body?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return res.status(400).json({ error: "numeric lat and lon are required" });
  }

  try {
    const formInput = normalizeFormInput(req.body?.formInput);
    const prediction = await predictSpeciesAtPoint(lat, lon, formInput);
    const environment = await fetchEnvironment(lat, lon).catch(() => ({
      temperatureC: null,
      salinityPpt: null,
      provider: "unavailable",
      fetchedAt: new Date().toISOString(),
      notes: ["Live environment data unavailable during advisory generation."],
    }));
    const advisory = await generateFallbackAdvisory({
      userId: req.user?.id,
      lat,
      lon,
      formInput,
      prediction,
      environment,
    });

    return res.json({
      input: { lat, lon },
      formInput,
      prediction,
      environment,
      advisory,
      generatedAt: new Date().toISOString(),
    });
  } catch (err) {
    return res.status(502).json({ error: err instanceof Error ? err.message : "advisory_generation_failed" });
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
      topCandidate: topScoredSpecies(s.prediction || null),
      advisoryFallbackUsed: Boolean(s.prediction?.advisoryFallbackUsed),
      fallbackAdvisory: s.prediction?.fallbackAdvisory
        ? {
            summary: String(s.prediction.fallbackAdvisory.answer || "").replace(/\s+/g, " ").trim().slice(0, 220),
            provider: s.prediction.fallbackAdvisory.provider || "unknown",
            status: s.prediction.fallbackAdvisory.status || "unknown",
          }
        : null,
    })),
  });
});

export default router;
