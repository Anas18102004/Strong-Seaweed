import { config } from "../config.js";

function timeoutSignal(ms = 6000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(t) };
}

export async function predictSpeciesAtPoint(lat, lon, formInput = null) {
  const speciesUrl = `${config.modelApiUrl}/predict/species`;
  const kappaUrl = `${config.modelApiUrl}/predict/kappaphycus`;
  let orchestrated = null;
  let kapp = null;
  let modelError = "";
  const payload = formInput ? { lat, lon, formInput } : { lat, lon };

  try {
    const t = timeoutSignal();
    const res = await fetch(speciesUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: t.signal,
    });
    t.clear();

    if (!res.ok) {
      modelError = `Species API error: ${res.status}`;
    } else {
      orchestrated = await res.json();
    }
  } catch (err) {
    modelError = err instanceof Error ? err.message : "Species API unreachable";
  }

  if (orchestrated && Array.isArray(orchestrated.species)) {
    return orchestrated;
  }

  // Fallback to legacy kappa-only API behavior.
  try {
    const t = timeoutSignal();
    const res = await fetch(kappaUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: t.signal,
    });
    t.clear();

    if (!res.ok) {
      modelError = `Kappaphycus API error: ${res.status}`;
    } else {
      kapp = await res.json();
    }
  } catch (err) {
    modelError = err instanceof Error ? err.message : "Kappaphycus API unreachable";
  }

  const kappScore = kapp?.kappaphycus?.probability_percent;

  const species = [
    {
      speciesId: "kappaphycus_alvarezii",
      displayName: "Kappaphycus alvarezii",
      ready: Boolean(kapp),
      probabilityPercent: Number.isFinite(kappScore) ? Number(kappScore) : null,
      priority: kapp?.kappaphycus?.priority || "unknown",
      reason: kapp ? "live_production_model" : modelError || "model_not_available",
    },
    {
      speciesId: "gracilaria_edulis",
      displayName: "Gracilaria edulis",
      ready: false,
      probabilityPercent: null,
      priority: "pending",
      reason: "model_training_pending",
    },
    {
      speciesId: "ulva_lactuca",
      displayName: "Ulva lactuca",
      ready: false,
      probabilityPercent: null,
      priority: "pending",
      reason: "model_training_pending",
    },
    {
      speciesId: "sargassum_wightii",
      displayName: "Sargassum wightii",
      ready: false,
      probabilityPercent: null,
      priority: "pending",
      reason: "model_training_pending",
    },
  ];

  const best = species
    .filter((s) => s.ready && s.probabilityPercent !== null)
    .sort((a, b) => (b.probabilityPercent || 0) - (a.probabilityPercent || 0))[0] || null;

  return {
    input: { lat, lon },
    source: "species-orchestrator",
    modelRelease: kapp?.model?.release_tag || "kappaphycus_unavailable",
    nearestGrid: kapp?.nearest_grid || null,
    species,
    bestSpecies: best,
    warnings: modelError ? [modelError] : [],
  };
}
