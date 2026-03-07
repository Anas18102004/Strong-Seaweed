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

function advisoryQuestionFromContext({ lat, lon, formInput, prediction, environment, verification = null }) {
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
  const verificationSummary = verification
    ? `Verification verdict=${verification?.verdict || "unknown"} (confidence=${verification?.confidenceScore ?? "n/a"}), candidate=${
        verification?.candidate?.displayName || "n/a"
      }, evidence=model:${verification?.evidence?.modelStrength || "n/a"}|copernicus:${
        verification?.evidence?.copernicusSupport || "n/a"
      }|web:${verification?.evidence?.occurrenceSupport || "n/a"}, nearestOccurrenceKm=${
        verification?.evidence?.occurrenceNearestKm ?? "n/a"
      }.`
    : "Verification summary unavailable.";

  return [
    "You are a seaweed aquaculture decision-support specialist.",
    "Use provided model/context signals as primary evidence. Do not claim certainty.",
    `Location: lat=${lat}, lon=${lon}, name=${formInput?.locationName || "unknown"}, season=${formInput?.season || "unknown"}.`,
    `Best model species: ${best?.displayName || "none"}; score=${best?.probabilityPercent ?? "n/a"}%; actionability=${best?.actionability || "unknown"}; decisionSource=${prediction?.decisionSource || "unknown"}.`,
    `Nearest model grid distance: ${nearestText} km.`,
    `Environment hints: temperature=${tempText}, salinity=${salText}, provider=${environment?.provider || "unknown"}.`,
    `Copernicus signals: ${copSummary}.`,
    verificationSummary,
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

function toRadians(x) {
  return (Number(x) * Math.PI) / 180;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const safe = [lat1, lon1, lat2, lon2].map((v) => Number(v));
  if (safe.some((v) => !Number.isFinite(v))) return null;
  const [aLat, aLon, bLat, bLon] = safe;
  const dLat = toRadians(bLat - aLat);
  const dLon = toRadians(bLon - aLon);
  const h =
    Math.sin(dLat / 2) ** 2 + Math.cos(toRadians(aLat)) * Math.cos(toRadians(bLat)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
  return 6371.0 * c;
}

function obisNameCandidates(species = null) {
  const names = new Set();
  const add = (v) => {
    const s = String(v || "").trim();
    if (s) names.add(s);
  };
  add(species?.canonicalName);
  add(species?.displayName);
  if (Array.isArray(species?.synonyms)) {
    for (const s of species.synonyms) add(s);
  }
  return Array.from(names);
}

function obisGeometryBox(lat, lon, radiusDeg = 1.0) {
  const la = Number(lat);
  const lo = Number(lon);
  const d = Number(radiusDeg);
  if (![la, lo, d].every(Number.isFinite)) return null;
  const lat1 = la - d;
  const lat2 = la + d;
  const lon1 = lo - d;
  const lon2 = lo + d;
  return `POLYGON((${lon1} ${lat1},${lon2} ${lat1},${lon2} ${lat2},${lon1} ${lat2},${lon1} ${lat1}))`;
}

async function fetchJsonWithTimeout(url, timeoutMs = 6000) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ac.signal });
    if (!res.ok) throw new Error(`http_${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

async function fetchObisEvidence({ lat, lon, species }) {
  const names = obisNameCandidates(species);
  if (!names.length) {
    return { support: "weak", recordCount: 0, nearestKm: null, speciesMatched: null, queryName: null, notes: ["no_species_name_for_obis"] };
  }
  const geometry = obisGeometryBox(lat, lon, 1.0);
  if (!geometry) {
    return { support: "weak", recordCount: 0, nearestKm: null, speciesMatched: null, queryName: null, notes: ["invalid_geometry"] };
  }

  let best = null;
  for (const name of names) {
    const url = `https://api.obis.org/v3/occurrence?scientificname=${encodeURIComponent(name)}&geometry=${encodeURIComponent(
      geometry,
    )}&size=80`;
    try {
      const data = await fetchJsonWithTimeout(url, 6000);
      const rows = Array.isArray(data?.results) ? data.results : [];
      let nearest = null;
      for (const r of rows) {
        const d = haversineKm(lat, lon, r?.decimalLatitude, r?.decimalLongitude);
        if (!Number.isFinite(d)) continue;
        if (nearest === null || d < nearest) nearest = d;
      }
      const candidate = {
        support: "weak",
        recordCount: Number(data?.total || rows.length || 0),
        nearestKm: nearest === null ? null : Number(nearest.toFixed(2)),
        speciesMatched: rows[0]?.species || null,
        queryName: name,
        sampleDate: rows[0]?.eventDate || null,
        notes: [],
      };
      if (candidate.recordCount > 0 && candidate.nearestKm !== null && candidate.nearestKm <= 75) candidate.support = "strong";
      else if (candidate.recordCount > 0 && candidate.nearestKm !== null && candidate.nearestKm <= 200) candidate.support = "moderate";
      else if (candidate.recordCount > 0) candidate.support = "moderate";

      if (!best) {
        best = candidate;
      } else if (candidate.support === "strong" && best.support !== "strong") {
        best = candidate;
      } else if (candidate.support === best.support && candidate.recordCount > best.recordCount) {
        best = candidate;
      } else if (
        candidate.support === best.support &&
        candidate.recordCount === best.recordCount &&
        candidate.nearestKm !== null &&
        (best.nearestKm === null || candidate.nearestKm < best.nearestKm)
      ) {
        best = candidate;
      }
    } catch {
      // Keep trying other names.
    }
  }

  return best || { support: "weak", recordCount: 0, nearestKm: null, speciesMatched: null, queryName: names[0], notes: ["obis_unavailable"] };
}

function classifyCopernicusSupport(species, cop = null) {
  const name = String(species?.speciesId || "").toLowerCase();
  const sal = Number(cop?.salinity?.mean);
  const wave = Number(cop?.waves?.heightMean);
  const cur = Number(cop?.currents?.speedMean);
  if (![sal, wave, cur].every(Number.isFinite)) return "unknown";

  const envelopes = {
    kappaphycus_alvarezii: { sal: [28, 36], wave: [0.2, 1.3], cur: [0.05, 0.45] },
    gracilaria_edulis: { sal: [20, 36], wave: [0.1, 1.5], cur: [0.03, 0.55] },
    ulva_lactuca: { sal: [18, 38], wave: [0.05, 2.2], cur: [0.02, 0.9] },
    sargassum_wightii: { sal: [24, 38], wave: [0.2, 2.0], cur: [0.05, 0.9] },
  };
  const env = envelopes[name];
  if (!env) return "unknown";
  const inRange = (v, r) => v >= r[0] && v <= r[1];
  const passed = [inRange(sal, env.sal), inRange(wave, env.wave), inRange(cur, env.cur)].filter(Boolean).length;
  if (passed >= 3) return "strong";
  if (passed >= 2) return "moderate";
  return "weak";
}

function modelStrength(prediction, candidate) {
  const score = Number(candidate?.probabilityPercent);
  const action = String(candidate?.actionability || "").toLowerCase();
  const conf = String(candidate?.confidenceBand || "").toLowerCase();
  if (action === "recommended" && Number.isFinite(score) && score >= 70) return "strong";
  if (action === "recommended" || action === "test_pilot_only") return conf === "high" || conf === "medium" ? "moderate" : "weak";
  if (Number.isFinite(score) && score >= 40) return "moderate";
  if (Number.isFinite(score) && score >= 20) return "weak";
  return "weak";
}

async function verifyPredictionEvidence({ lat, lon, prediction }) {
  const best = prediction?.bestSpecies || null;
  const top = topScoredSpecies(prediction);
  const candidate = best?.speciesId === "insufficient_data" ? top : best || top;
  if (!candidate) {
    return {
      verdict: "weak",
      confidenceScore: 0,
      candidate: null,
      evidence: {
        modelStrength: "weak",
        copernicusSupport: "unknown",
        occurrenceSupport: "weak",
        occurrenceRecordCount: 0,
        occurrenceNearestKm: null,
      },
      notes: ["no_candidate_species"],
      checkedAt: new Date().toISOString(),
    };
  }

  const copSupport = classifyCopernicusSupport(candidate, prediction?.copernicusContext || null);
  const modelSupport = modelStrength(prediction, candidate);
  const obis = await fetchObisEvidence({ lat, lon, species: candidate });
  const occSupport = obis?.support || "weak";

  const scoreMap = { weak: 0.4, moderate: 0.8, strong: 1.2, unknown: 0.5 };
  const modelMap = { weak: 0.8, moderate: 1.3, strong: 1.8 };
  const confidenceScore = Number(
    (
      (modelMap[modelSupport] || 0.8) +
      (scoreMap[copSupport] || 0.5) +
      (scoreMap[occSupport] || 0.4)
    ).toFixed(2)
  );
  const verdict = confidenceScore >= 3.2 ? "strong" : confidenceScore >= 2.2 ? "moderate" : "weak";

  const notes = [];
  notes.push(`model=${modelSupport}`);
  notes.push(`copernicus=${copSupport}`);
  notes.push(`occurrence=${occSupport}`);
  if (Number.isFinite(Number(candidate?.probabilityPercent))) notes.push(`model_score=${candidate.probabilityPercent}%`);
  if (obis?.recordCount !== undefined) notes.push(`occurrence_records=${obis.recordCount}`);
  if (obis?.nearestKm !== null && obis?.nearestKm !== undefined) notes.push(`occurrence_nearest_km=${obis.nearestKm}`);

  return {
    verdict,
    confidenceScore,
    candidate: {
      speciesId: candidate?.speciesId || null,
      displayName: candidate?.displayName || null,
      canonicalName: candidate?.canonicalName || candidate?.displayName || null,
      probabilityPercent: candidate?.probabilityPercent ?? null,
      actionability: candidate?.actionability || null,
    },
    evidence: {
      modelStrength: modelSupport,
      copernicusSupport: copSupport,
      occurrenceSupport: occSupport,
      occurrenceRecordCount: obis?.recordCount ?? 0,
      occurrenceNearestKm: obis?.nearestKm ?? null,
      occurrenceSpeciesMatched: obis?.speciesMatched || null,
      occurrenceQueryName: obis?.queryName || null,
      occurrenceSampleDate: obis?.sampleDate || null,
    },
    notes,
    checkedAt: new Date().toISOString(),
  };
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

async function generateFallbackAdvisory({ userId, lat, lon, formInput, prediction, environment, verification = null }) {
  const question = advisoryQuestionFromContext({ lat, lon, formInput, prediction, environment, verification });
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
        verification,
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
    const verification = await verifyPredictionEvidence({ lat, lon, prediction }).catch(() => ({
      verdict: "weak",
      confidenceScore: 0,
      candidate: null,
      evidence: {
        modelStrength: "weak",
        copernicusSupport: "unknown",
        occurrenceSupport: "weak",
        occurrenceRecordCount: 0,
        occurrenceNearestKm: null,
      },
      notes: ["verification_unavailable"],
      checkedAt: new Date().toISOString(),
    }));
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
        verification,
      });
      advisoryFallbackUsed = true;
    }

    const out = {
      ...prediction,
      verification,
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
    const verification = await verifyPredictionEvidence({ lat, lon, prediction }).catch(() => ({
      verdict: "weak",
      confidenceScore: 0,
      candidate: null,
      evidence: {
        modelStrength: "weak",
        copernicusSupport: "unknown",
        occurrenceSupport: "weak",
        occurrenceRecordCount: 0,
        occurrenceNearestKm: null,
      },
      notes: ["verification_unavailable"],
      checkedAt: new Date().toISOString(),
    }));
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
      verification,
    });

    return res.json({
      input: { lat, lon },
      formInput,
      prediction,
      verification,
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
      verification: s.prediction?.verification
        ? {
            verdict: s.prediction.verification.verdict || "unknown",
            confidenceScore: Number(s.prediction.verification.confidenceScore || 0),
            candidateDisplayName: s.prediction.verification?.candidate?.displayName || "",
          }
        : null,
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
