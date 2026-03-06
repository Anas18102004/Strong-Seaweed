import { config } from "../config.js";

function timeoutSignal(ms = Math.max(12000, Number(config.aiTimeoutMs || 12000))) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(t) };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postModelJson(url, payload, attempts = 2) {
  let lastErr = "";
  for (let i = 0; i < attempts; i += 1) {
    const t = timeoutSignal();
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: t.signal,
      });
      if (!res.ok) {
        lastErr = `HTTP ${res.status}`;
      } else {
        const data = await res.json();
        return { ok: true, data, error: "" };
      }
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err || "request_failed");
    } finally {
      t.clear();
    }

    if (i < attempts - 1) {
      await sleep(250);
    }
  }
  return { ok: false, data: null, error: lastErr || "request_failed" };
}

export async function predictSpeciesAtPoint(lat, lon, formInput = null) {
  const speciesUrl = `${config.modelApiUrl}/predict/species`;
  const kappaUrl = `${config.modelApiUrl}/predict/kappaphycus`;
  let orchestrated = null;
  let kapp = null;
  let modelError = "";
  const payload = formInput ? { lat, lon, formInput } : { lat, lon };

  const orchestratedReq = await postModelJson(speciesUrl, payload, 2);
  if (orchestratedReq.ok) {
    orchestrated = orchestratedReq.data;
  } else {
    modelError = `Species API error: ${orchestratedReq.error}`;
  }

  if (orchestrated && Array.isArray(orchestrated.species)) {
    return orchestrated;
  }

  // Fallback to legacy kappa-only API behavior.
  const kappaReq = await postModelJson(kappaUrl, payload, 2);
  if (kappaReq.ok) {
    kapp = kappaReq.data;
  } else {
    modelError = `Kappaphycus API error: ${kappaReq.error}`;
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
