import { config } from "../config.js";
import { predictSpeciesAtPoint } from "./predictService.js";

const CANNED = {
  cultivation:
    "For beginners: start with stable salinity and moderate wave zones. Focus on line quality, weekly fouling checks, and harvest timing discipline.",
  risk:
    "Top risks are monsoon wave spikes, sudden salinity shifts, and storm windows. Prepare contingency lines and pause operations under high-energy sea states.",
  yield:
    "To increase yield: maintain spacing discipline, clean biofouling every 7-10 days, and avoid delayed harvest beyond peak biomass.",
  site:
    "For expansion: shortlist shallow, manageable-wave areas with low conflict and strong consistency over multiple months.",
  market:
    "For market timing: align harvest quality, processor demand windows, and logistics costs before scaling volume.",
  copilot:
    "I can help with cultivation planning, risk alerts, yield improvement, site expansion, and market timing.",
};

const STACK_MAP = {
  cultivation: ["LangChain", "Domain Rules"],
  risk: ["LangGraph", "Risk Flow"],
  yield: ["CrewAI", "Optimization Crew"],
  site: ["LangChain", "Geo Retrieval"],
  market: ["CrewAI", "Market Crew"],
  copilot: ["LangGraph", "Orchestrator"],
};

const AGENT_PROVIDER_PREF = {
  cultivation: "langchain",
  site: "langchain",
  risk: "langgraph",
  copilot: "langgraph",
  yield: "crewai",
  market: "crewai",
};

const PROVIDER_URLS = {
  langchain: config.langChainApiUrl,
  langgraph: config.langGraphApiUrl,
  crewai: config.crewAiApiUrl,
};

const MODEL_LOCATION_COORDS = {
  "gulf of mannar": { lat: 9.1, lon: 79.3 },
  "palk bay": { lat: 9.4, lon: 79.2 },
  lakshadweep: { lat: 10.5, lon: 72.7 },
  "andaman islands": { lat: 11.7, lon: 92.7 },
  "gulf of kachchh": { lat: 22.6, lon: 69.8 },
  chilika: { lat: 19.7, lon: 85.3 },
  ratnagiri: { lat: 16.9, lon: 73.3 },
  karwar: { lat: 14.8, lon: 74.1 },
  kollam: { lat: 8.9, lon: 76.6 },
};

const CACHE_TTL_MS = 60_000;
const CACHE_MAX_ITEMS = 1000;
const aiCache = new Map();
const DEFAULT_VOICE_MIME = "audio/mpeg";
const TTS_PROVIDER_BROWSER = "browser";
const TTS_PROVIDER_ELEVENLABS = "elevenlabs";
const MAX_UI_ANSWER_CHARS = 1200;

function normalize(text = "") {
  return String(text || "").trim().toLowerCase();
}

function stableStringify(value) {
  const seen = new WeakSet();
  const walk = (v) => {
    if (v === null || typeof v !== "object") return v;
    if (seen.has(v)) return null;
    seen.add(v);
    if (Array.isArray(v)) return v.map(walk);
    return Object.keys(v)
      .sort()
      .reduce((acc, k) => {
        acc[k] = walk(v[k]);
        return acc;
      }, {});
  };
  return JSON.stringify(walk(value));
}

function getCached(key) {
  const item = aiCache.get(key);
  if (!item) return null;
  if (item.expiresAt <= Date.now()) {
    aiCache.delete(key);
    return null;
  }
  return { ...item.value, cached: true };
}

function setCached(key, value) {
  if (aiCache.size >= CACHE_MAX_ITEMS) {
    const first = aiCache.keys().next().value;
    if (first) aiCache.delete(first);
  }
  aiCache.set(key, {
    expiresAt: Date.now() + CACHE_TTL_MS,
    value: { ...value, cached: false },
  });
}

function routeQuestion(question) {
  // Delegate intent routing to the LLM copilot instead of keyword heuristics.
  if (!String(question || "").trim()) return "copilot";
  return "copilot";
}

function unavailableAnswer(agentId = "copilot") {
  const topic = String(agentId || "copilot");
  return `Live AI model is unavailable right now for ${topic}. Please check provider status/config and try again.`;
}

function toNum(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function extractLatLonFromQuestion(question = "") {
  const m = String(question).match(/(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)/);
  if (!m) return null;
  const lat = toNum(m[1]);
  const lon = toNum(m[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function shouldUseModelGrounding(question = "", ctx = {}) {
  const q = String(question || "").toLowerCase();
  const hasPredictIntent = [
    "predict",
    "suitability",
    "which species",
    "best species",
    "what should i grow",
    "what to grow",
    "can i cultivate",
    "is this location good",
    "farm here",
    "kappaphycus",
    "gracilaria",
    "ulva",
    "sargassum",
  ].some((k) => q.includes(k));

  const c = ctx && typeof ctx === "object" ? ctx : {};
  const hasCoords = Number.isFinite(toNum(c.lat)) && Number.isFinite(toNum(c.lon));
  const hasLocation = String(c.locationName || "").trim().length > 0;
  const hasQuestionCoords = Boolean(extractLatLonFromQuestion(question));
  return hasPredictIntent && (hasCoords || hasLocation || hasQuestionCoords);
}

function resolveCoords(question = "", ctx = {}) {
  const c = ctx && typeof ctx === "object" ? ctx : {};
  const lat = toNum(c.lat);
  const lon = toNum(c.lon);
  if (Number.isFinite(lat) && Number.isFinite(lon)) return { lat, lon, source: "context" };

  const fromText = extractLatLonFromQuestion(question);
  if (fromText) return { ...fromText, source: "question" };

  const loc = String(c.locationName || "").trim().toLowerCase();
  if (loc && MODEL_LOCATION_COORDS[loc]) {
    return { ...MODEL_LOCATION_COORDS[loc], source: "location_lookup" };
  }

  return null;
}

function buildModelFormInput(ctx = {}) {
  const c = ctx && typeof ctx === "object" ? ctx : {};
  const adv = c.advanced && typeof c.advanced === "object" ? c.advanced : {};
  const ov = c.overrides && typeof c.overrides === "object" ? c.overrides : {};
  return {
    locationName: String(c.locationName || "").trim(),
    season: String(c.season || "").trim(),
    depthM: toNum(c.depthM),
    overrides: {
      temperatureC: toNum(ov.temperatureC),
      salinityPpt: toNum(ov.salinityPpt),
    },
    advanced: {
      ph: toNum(adv.ph),
      turbidityNtu: toNum(adv.turbidityNtu),
      currentVelocityMs: toNum(adv.currentVelocityMs),
      waveHeightM: toNum(adv.waveHeightM),
      rainfallMm: toNum(adv.rainfallMm),
      tidalAmplitudeM: toNum(adv.tidalAmplitudeM),
    },
  };
}

function formatModelGroundedAnswer(pred, coordsMeta = null) {
  const best = pred?.bestSpecies || null;
  const species = Array.isArray(pred?.species) ? pred.species : [];
  const top = [...species]
    .filter((s) => Number.isFinite(s?.probabilityPercent))
    .sort((a, b) => (b.probabilityPercent || 0) - (a.probabilityPercent || 0))
    .slice(0, 3);
  const lines = [];
  if (best && Number.isFinite(best?.probabilityPercent)) {
    lines.push(
      `Model-grounded recommendation: ${best.displayName} (${best.probabilityPercent.toFixed(2)}% suitability, ${best.priority}).`,
    );
  } else {
    lines.push("Model result: no species currently meets the suitability threshold for this context.");
  }
  if (top.length > 0) {
    lines.push(`Top species scores: ${top.map((s) => `${s.displayName} ${Number(s.probabilityPercent).toFixed(2)}%`).join(" | ")}.`);
  }
  if (pred?.nearestGrid?.distance_km !== undefined && pred?.nearestGrid?.distance_km !== null) {
    lines.push(`Nearest model grid distance: ${Number(pred.nearestGrid.distance_km).toFixed(2)} km.`);
  }
  if (coordsMeta?.source) {
    lines.push(`Location source: ${coordsMeta.source}.`);
  }
  if (Array.isArray(pred?.warnings) && pred.warnings.length > 0) {
    lines.push(`Warnings: ${pred.warnings.join(", ")}.`);
  }
  return lines.join("\n");
}

async function maybeRunModelGrounding(question, context = {}) {
  const rawCtx = context?.context && typeof context.context === "object" ? context.context : context;
  if (!shouldUseModelGrounding(question, rawCtx)) return null;
  const coords = resolveCoords(question, rawCtx);
  if (!coords) return null;

  try {
    const pred = await predictSpeciesAtPoint(coords.lat, coords.lon, buildModelFormInput(rawCtx));
    return {
      answer: normalizeUiAnswer(formatModelGroundedAnswer(pred, coords)),
      model: String(pred?.modelRelease || "species-orchestrator"),
      stack: ["Species Orchestrator", "Model Grounding"],
      routedAgent: routeQuestion(question),
      provider: "model-api",
      status: "live",
      modelGrounded: true,
      prediction: pred,
    };
  } catch (err) {
    console.error(`[aiService] model grounding failed: ${err instanceof Error ? err.message : "unknown_error"}`);
    return null;
  }
}

function shouldUseWebGrounding(question = "") {
  const q = String(question || "").toLowerCase();
  const hasSeaweedContext = ["seaweed", "kappaphycus", "gracilaria", "ulva", "sargassum", "aquaculture"].some((k) =>
    q.includes(k),
  );
  const hasRecencyIntent = ["latest", "today", "current", "news", "price", "market", "policy", "regulation"].some((k) =>
    q.includes(k),
  );
  return hasSeaweedContext && hasRecencyIntent;
}

async function fetchWebSnapshot(question = "") {
  const q = encodeURIComponent(String(question || "").slice(0, 180));
  const url = `https://api.duckduckgo.com/?q=${q}&format=json&no_redirect=1&no_html=1`;
  try {
    const data = await fetchJson(url, { method: "GET" }, 4500);
    const abstract = String(data?.AbstractText || "").trim();
    const heading = String(data?.Heading || "").trim();
    const related = Array.isArray(data?.RelatedTopics) ? data.RelatedTopics : [];
    const firstRelated = related
      .map((x) => (x && typeof x === "object" ? String(x.Text || "").trim() : ""))
      .find(Boolean);
    const lines = [];
    if (heading && abstract) lines.push(`${heading}: ${abstract}`);
    else if (abstract) lines.push(abstract);
    if (firstRelated) lines.push(`Related: ${firstRelated}`);
    return lines.join("\n").trim();
  } catch {
    return "";
  }
}

function normalizeUiAnswer(input) {
  let text = String(input || "").replace(/\r\n/g, "\n");

  // Remove markdown heading markers.
  text = text.replace(/^\s{0,3}#{1,6}\s+/gm, "");

  // Convert markdown bullets to normal bullets.
  text = text.replace(/^\s*[*•]\s+/gm, "- ");

  // Flatten markdown table separator rows.
  text = text.replace(/^\s*\|?[-:| ]+\|?\s*$/gm, "");

  // Flatten table rows into readable inline text.
  text = text.replace(/^\s*\|(.+)\|\s*$/gm, (_m, row) => {
    return row
      .split("|")
      .map((cell) => cell.trim())
      .filter(Boolean)
      .join(" - ");
  });

  // Keep emphasis markers minimal for UI renderer but remove single-star noise.
  text = text.replace(/\*(?!\*)(.*?)\*(?!\*)/g, "$1");

  // Collapse excess blank lines/spaces.
  text = text.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();

  // Keep responses medium by default.
  if (text.length > MAX_UI_ANSWER_CHARS) {
    text = `${text.slice(0, MAX_UI_ANSWER_CHARS).trim()}\n\n- Ask \"expand\" if you want deeper detail.`;
  }

  return text;
}

function providerOrder(agentId) {
  const preferred = AGENT_PROVIDER_PREF[agentId] || "langgraph";
  return [preferred, "langchain", "langgraph", "crewai"].filter((x, i, arr) => arr.indexOf(x) === i);
}

async function fetchJson(url, options = {}, timeoutMs = config.aiTimeoutMs) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: ac.signal });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

async function fetchBuffer(url, options = {}, timeoutMs = config.aiTimeoutMs) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: ac.signal });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    const ab = await res.arrayBuffer();
    return { buffer: Buffer.from(ab), contentType: res.headers.get("content-type") || DEFAULT_VOICE_MIME };
  } finally {
    clearTimeout(t);
  }
}

function normalizeElevenLabsError(err) {
  const raw = err instanceof Error ? err.message : String(err || "");
  const lower = raw.toLowerCase();
  if (lower.includes("paid_plan_required") || lower.includes("payment_required")) {
    return "ElevenLabs rejected this voice: paid library voice. Use your own/free voice ID or upgrade ElevenLabs plan.";
  }
  if (lower.includes("invalid_api_key")) {
    return "ElevenLabs API key is invalid. Check ELEVENLABS_API_KEY in server/.env.";
  }
  if (lower.includes("voice_not_found") || lower.includes("voice not found")) {
    return "ElevenLabs voice ID not found. Check ELEVENLABS_VOICE_ID_FEMALE.";
  }
  return raw || "ElevenLabs TTS failed.";
}

function resolveVoiceId(voiceProfile = "default") {
  const profile = String(voiceProfile || "default").toLowerCase();
  if (profile === "female" && config.elevenLabsVoiceIdFemale) return config.elevenLabsVoiceIdFemale;
  if (profile === "male" && config.elevenLabsVoiceIdMale) return config.elevenLabsVoiceIdMale;
  return config.elevenLabsVoiceId || config.elevenLabsVoiceIdFemale || config.elevenLabsVoiceIdMale || "";
}

function activeTtsProvider() {
  const provider = String(config.voiceTtsProvider || "").toLowerCase();
  if (provider === TTS_PROVIDER_ELEVENLABS) return TTS_PROVIDER_ELEVENLABS;
  return TTS_PROVIDER_BROWSER;
}

async function generateElevenLabsAudio(ttsText, voiceProfile = "default") {
  if (activeTtsProvider() !== TTS_PROVIDER_ELEVENLABS) {
    return null;
  }
  const voiceId = resolveVoiceId(voiceProfile);
  const text = String(ttsText || "").slice(0, 2500);
  if (!config.elevenLabsApiKey || !voiceId) {
    console.warn(
      `[aiService][elevenlabs] skipped keyConfigured=${Boolean(config.elevenLabsApiKey)} voiceIdConfigured=${Boolean(voiceId)} voiceProfile=${voiceProfile}`,
    );
    return null;
  }
  console.info(
    `[aiService][elevenlabs] tts start voiceProfile=${voiceProfile} voiceId=${voiceId} chars=${text.length} model=${config.elevenLabsModelId}`,
  );
  const body = {
    text,
    model_id: config.elevenLabsModelId,
    output_format: config.elevenLabsOutputFormat,
    voice_settings: {
      stability: 0.45,
      similarity_boost: 0.75,
      style: 0.1,
      use_speaker_boost: true,
    },
  };
  const { buffer, contentType } = await fetchBuffer(
    `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voiceId)}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": config.elevenLabsApiKey,
      },
      body: JSON.stringify(body),
    },
    Math.max(5000, Math.min(config.aiTimeoutMs, 15000)),
  );
  console.info(
    `[aiService][elevenlabs] tts success voiceProfile=${voiceProfile} voiceId=${voiceId} bytes=${buffer.length} mime=${contentType}`,
  );
  return {
    audioBase64: buffer.toString("base64"),
    audioMime: contentType,
    voiceProvider: "elevenlabs",
    voiceProfile,
  };
}

async function attachElevenLabsOrThrow(out, voiceProfile) {
  let tts = null;
  try {
    tts = await generateElevenLabsAudio(out.ttsText, voiceProfile);
  } catch (err) {
    const msg = normalizeElevenLabsError(err);
    throw new Error(msg);
  }
  if (tts) {
    Object.assign(out, tts);
    return out;
  }
  if (config.elevenLabsStrictVoice && activeTtsProvider() === TTS_PROVIDER_ELEVENLABS) {
    const voiceId = resolveVoiceId(voiceProfile);
    if (!config.elevenLabsApiKey) {
      throw new Error("ELEVENLABS_STRICT_VOICE is enabled but ELEVENLABS_API_KEY is missing.");
    }
    if (!voiceId) {
      throw new Error(
        `ELEVENLABS_STRICT_VOICE is enabled but no voice id is configured for profile '${voiceProfile}'.`,
      );
    }
    throw new Error("ELEVENLABS_STRICT_VOICE is enabled but ElevenLabs TTS failed.");
  }
  return out;
}

async function tryProvider(provider, path, payload) {
  const base = String(PROVIDER_URLS[provider] || "").trim();
  if (!base) {
    console.warn(`[aiService] provider=${provider} path=${path} not configured`);
    return { ok: false, provider, error: "provider_not_configured" };
  }
  try {
    const out = await fetchJson(`${base}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return { ok: true, provider, out };
  } catch (err) {
    const error = err instanceof Error ? err.message : "provider_call_failed";
    console.error(`[aiService] provider=${provider} path=${path} failed: ${error}`);
    return { ok: false, provider, error };
  }
}

export async function runAgent(agent, question, context = {}) {
  const id = normalize(agent) || routeQuestion(question);
  const stacks = STACK_MAP[id] || STACK_MAP.copilot;
  const cacheKey = `agent:${id}:${normalize(question)}:${stableStringify(context)}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  for (const provider of providerOrder(id)) {
    const tried = await tryProvider(provider, "/agent", { agent: id, question, context });
    if (tried.ok) {
      const out = {
        agent: id,
        answer: normalizeUiAnswer(tried.out?.answer || ""),
        stack: tried.out?.stack || stacks,
        provider,
        status: tried.out?.status || "live",
        latencyMs: Number(tried.out?.latencyMs || 0) || undefined,
      };
      if (!out.answer.trim()) out.answer = unavailableAnswer(id);
      setCached(cacheKey, out);
      return out;
    }
  }

  const out = {
    agent: id,
    answer: normalizeUiAnswer(unavailableAnswer(id)),
    stack: stacks,
    provider: "fallback",
    status: "fallback",
  };
  setCached(cacheKey, out);
  return out;
}

export async function runChat(question, context = {}) {
  const routedAgent = routeQuestion(question);
  const cacheKey = `chat:${routedAgent}:${normalize(question)}:${stableStringify(context)}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const grounded = await maybeRunModelGrounding(question, context);
  if (grounded) {
    setCached(cacheKey, grounded);
    return grounded;
  }

  if (config.aiRouterMode === "provider-first") {
    for (const provider of ["langgraph", "langchain", "crewai"]) {
      const tried = await tryProvider(provider, "/chat", { question, context, routedAgent });
      if (tried.ok) {
        const out = {
          answer: normalizeUiAnswer(tried.out?.answer || CANNED.copilot),
          model: String(tried.out?.model || `provider-${provider}`),
          stack: tried.out?.stack || (STACK_MAP[routedAgent] || STACK_MAP.copilot),
          routedAgent,
          provider,
          status: tried.out?.status || "live",
          latencyMs: Number(tried.out?.latencyMs || 0) || undefined,
        };
        if (shouldUseWebGrounding(question)) {
          const web = await fetchWebSnapshot(question);
          if (web) out.answer = normalizeUiAnswer(`${out.answer}\n\nLive web snapshot:\n${web}`);
        }
        setCached(cacheKey, out);
        return out;
      }
    }
  }

  const routed = await runAgent(routedAgent, question, context);
  const fallbackModelName = routed.provider === "fallback" ? "akuara-assistant" : `hybrid-${routed.provider}`;
  const answer = routed.provider === "fallback" ? unavailableAnswer(routedAgent) : routed.answer;
  const out = {
    answer: normalizeUiAnswer(answer),
    model: fallbackModelName,
    stack: routed.stack,
    routedAgent,
    provider: routed.provider,
    status: routed.status,
    latencyMs: routed.latencyMs,
  };
  if (shouldUseWebGrounding(question)) {
    const web = await fetchWebSnapshot(question);
    if (web) out.answer = normalizeUiAnswer(`${out.answer}\n\nLive web snapshot:\n${web}`);
  }
  setCached(cacheKey, out);
  return out;
}

export async function runVoice(question, context = {}) {
  const routedAgent = routeQuestion(question);
  const cacheKey = `voice:${routedAgent}:${normalize(question)}:${stableStringify(context)}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const grounded = await maybeRunModelGrounding(question, context);
  if (grounded) {
    const voiceOut = {
      ...grounded,
      ttsText: normalizeUiAnswer(grounded.answer),
      voiceProfile: String(context?.voiceProfile || "female").toLowerCase(),
    };
    try {
      await attachElevenLabsOrThrow(voiceOut, voiceOut.voiceProfile);
    } catch (err) {
      if (config.elevenLabsStrictVoice) throw err;
    }
    const cacheSafe = { ...voiceOut };
    delete cacheSafe.audioBase64;
    setCached(cacheKey, cacheSafe);
    return voiceOut;
  }

  const voiceProfile = String(context?.voiceProfile || "female").toLowerCase();

  for (const provider of ["langgraph", "langchain", "crewai"]) {
    const tried = await tryProvider(provider, "/voice/respond", {
      question,
      routedAgent,
      context,
      locale: String(context?.locale || "en-US"),
    });
    if (tried.ok) {
      const out = {
        answer: normalizeUiAnswer(tried.out?.answer || CANNED.copilot),
        ttsText: normalizeUiAnswer(tried.out?.ttsText || tried.out?.answer || CANNED.copilot),
        model: String(tried.out?.model || `voice-${provider}`),
        stack: tried.out?.stack || (STACK_MAP[routedAgent] || STACK_MAP.copilot),
        routedAgent,
        provider,
        status: tried.out?.status || "live",
        latencyMs: Number(tried.out?.latencyMs || 0) || undefined,
        voiceProfile,
      };
      try {
        await attachElevenLabsOrThrow(out, voiceProfile);
      } catch (err) {
        console.error(
          `[aiService][elevenlabs] strict-mode provider-response attach failed voiceProfile=${voiceProfile}: ${
            err instanceof Error ? err.message : "unknown_error"
          }`,
        );
        if (config.elevenLabsStrictVoice) throw err;
        // voice synthesis fallback handled client-side when not strict.
      }
      const cacheSafe = { ...out };
      delete cacheSafe.audioBase64;
      setCached(cacheKey, cacheSafe);
      return out;
    }
  }

  const out = await runChat(question, context);
  const fallback = {
    ...out,
    ttsText: normalizeUiAnswer(out.answer),
    voiceProfile,
  };
  try {
    await attachElevenLabsOrThrow(fallback, voiceProfile);
  } catch (err) {
    console.error(
      `[aiService][elevenlabs] strict-mode fallback attach failed voiceProfile=${voiceProfile}: ${
        err instanceof Error ? err.message : "unknown_error"
      }`,
    );
    if (config.elevenLabsStrictVoice) throw err;
    // voice synthesis fallback handled client-side when not strict.
  }
  const cacheSafe = { ...fallback };
  delete cacheSafe.audioBase64;
  setCached(cacheKey, cacheSafe);
  return fallback;
}

export async function getAiStatus() {
  const providers = [];
  for (const provider of ["langchain", "langgraph", "crewai"]) {
    const base = String(PROVIDER_URLS[provider] || "").trim();
    if (!base) {
      providers.push({ provider, configured: false, online: false, detail: "not_configured" });
      continue;
    }
    try {
      const out = await fetchJson(`${base}/health`, { method: "GET" }, Math.min(config.aiTimeoutMs, 4000));
      providers.push({
        provider,
        configured: true,
        online: true,
        detail: out?.status || "ok",
        voiceSupported: Boolean(out?.voice_supported),
      });
    } catch (err) {
      providers.push({
        provider,
        configured: true,
        online: false,
        detail: err instanceof Error ? err.message : "health_failed",
      });
    }
  }
  return {
    routerMode: config.aiRouterMode,
    providers,
    voice: {
      ttsProvider: activeTtsProvider(),
      elevenLabsConfigured: Boolean(config.elevenLabsApiKey && resolveVoiceId("female")),
      elevenLabsFemaleConfigured: Boolean(config.elevenLabsApiKey && resolveVoiceId("female")),
      elevenLabsMaleConfigured: Boolean(config.elevenLabsApiKey && resolveVoiceId("male")),
      elevenLabsModelId: config.elevenLabsModelId,
      elevenLabsOutputFormat: config.elevenLabsOutputFormat,
      elevenLabsStrictVoice: config.elevenLabsStrictVoice,
    },
    cache: {
      enabled: true,
      ttlMs: CACHE_TTL_MS,
      size: aiCache.size,
    },
  };
}
