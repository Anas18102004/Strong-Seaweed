import express from "express";
import { authRequired } from "../middleware/auth.js";
import { getAiStatus, runAgent, runChat, runVoice } from "../services/aiService.js";
import { config } from "../config.js";
import { ChatSession } from "../models/ChatSession.js";
import { ChatMessage } from "../models/ChatMessage.js";

const router = express.Router();

function logAiStart(route, userId, extra = {}) {
  const start = Date.now();
  console.info(`[AI][START] route=${route} user=${userId}`, extra);
  return start;
}

function logAiEnd(route, userId, start, ok = true, extra = {}) {
  const latencyMs = Date.now() - start;
  console.info(`[AI][END] route=${route} user=${userId} ok=${ok} latencyMs=${latencyMs}`, extra);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sanitizeContext(raw = {}) {
  const input = raw && typeof raw === "object" ? raw : {};
  const asNum = (v) => {
    if (v === null || v === undefined || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const asStr = (v, max = 80) => String(v || "").trim().slice(0, max);
  const advancedIn = input.advanced && typeof input.advanced === "object" ? input.advanced : {};
  const overridesIn = input.overrides && typeof input.overrides === "object" ? input.overrides : {};
  return {
    mode: asStr(input.mode, 24),
    locationName: asStr(input.locationName, 120),
    speciesHint: asStr(input.speciesHint, 80),
    season: asStr(input.season, 40),
    lat: asNum(input.lat),
    lon: asNum(input.lon),
    depthM: asNum(input.depthM),
    overrides: {
      temperatureC: asNum(overridesIn.temperatureC),
      salinityPpt: asNum(overridesIn.salinityPpt),
    },
    advanced: {
      ph: asNum(advancedIn.ph),
      turbidityNtu: asNum(advancedIn.turbidityNtu),
      currentVelocityMs: asNum(advancedIn.currentVelocityMs),
      waveHeightM: asNum(advancedIn.waveHeightM),
      rainfallMm: asNum(advancedIn.rainfallMm),
      tidalAmplitudeM: asNum(advancedIn.tidalAmplitudeM),
    },
  };
}

function deepgramLanguageForLocale(locale = "hi-IN") {
  const value = String(locale || "").toLowerCase();
  if (value.startsWith("en")) return "en";
  if (value.startsWith("hi")) return "hi";
  if (value.startsWith("gu")) return "gu";
  return "hi";
}

async function transcribeWithDeepgram({ audioBase64, mimeType, locale }) {
  if (!config.deepgramApiKey) {
    throw new Error("deepgram_not_configured");
  }

  const base64 = String(audioBase64 || "").trim();
  if (!base64) {
    throw new Error("audio_missing");
  }

  const audio = Buffer.from(base64, "base64");
  if (!audio.length) {
    throw new Error("audio_empty");
  }

  const model = encodeURIComponent(String(config.deepgramSttModel || "flux-general-en"));
  const language = encodeURIComponent(deepgramLanguageForLocale(locale));
  const url = `${String(config.deepgramSttApiUrl || "https://api.deepgram.com/v1/listen")}?model=${model}&language=${language}&smart_format=true&punctuate=true`;
  const contentType = String(mimeType || "audio/webm");

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Token ${config.deepgramApiKey}`,
      "Content-Type": contentType,
    },
    body: audio,
  });

  const raw = await response.text();
  if (!response.ok) {
    throw new Error(`deepgram_http_${response.status}: ${raw.slice(0, 260)}`);
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("deepgram_parse_error");
  }

  const alt = parsed?.results?.channels?.[0]?.alternatives?.[0] || {};
  const transcript = String(alt?.transcript || "").trim();
  const confidence = Number(alt?.confidence);
  return {
    transcript,
    confidence: Number.isFinite(confidence) ? confidence : null,
    provider: "deepgram",
    model: String(config.deepgramSttModel || "flux-general-en"),
    locale: deepgramLanguageForLocale(locale),
  };
}

async function loadRecentConversation(sessionId, userId, limit = 8) {
  const rows = await ChatMessage.find({ sessionId, userId })
    .sort({ createdAt: -1 })
    .limit(limit)
    .lean();
  return rows
    .reverse()
    .map((m) => ({ role: m.role, content: m.content }))
    .slice(-limit);
}

async function resolveSession(userId, sessionIdInput, fallbackTitle) {
  let session = null;
  const input = String(sessionIdInput || "").trim();
  if (input) {
    session = await ChatSession.findOne({ _id: input, userId });
  }
  if (!session) {
    session = await ChatSession.create({
      userId,
      title: String(fallbackTitle || "New Chat").slice(0, 80),
      lastMessageAt: new Date(),
    });
  }
  return session;
}

router.post("/chat", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/chat", req.user?.id, { hasSessionId: Boolean(req.body?.sessionId) });
  const question = String(req.body?.question || "").trim();
  const sessionIdInput = String(req.body?.sessionId || "").trim();
  if (!question) {
    logAiEnd("/api/ai/chat", req.user?.id, start, false, { error: "question_missing" });
    return res.status(400).json({ error: "question is required" });
  }

  try {
    const session = await resolveSession(req.user.id, sessionIdInput, question);

    await ChatMessage.create({
      sessionId: session._id,
      userId: req.user.id,
      role: "user",
      content: question,
    });

    const conversation = await loadRecentConversation(session._id, req.user.id);
    const out = await runChat(question, {
      userId: req.user.id,
      sessionId: session._id.toString(),
      conversation,
      context: sanitizeContext(req.body?.context || {}),
    });

    await ChatMessage.create({
      sessionId: session._id,
      userId: req.user.id,
      role: "assistant",
      content: out.answer,
      meta: {
        model: out.model,
        stack: out.stack,
        routedAgent: out.routedAgent,
        provider: out.provider,
        status: out.status,
        latencyMs: out.latencyMs,
        cached: out.cached,
      },
    });

    session.lastMessageAt = new Date();
    await session.save();

    logAiEnd("/api/ai/chat", req.user?.id, start, true, {
      routedAgent: out.routedAgent,
      provider: out.provider,
      status: out.status,
    });
    res.json({ ...out, sessionId: session._id.toString() });
  } catch (error) {
    logAiEnd("/api/ai/chat", req.user?.id, start, false, { error: error instanceof Error ? error.message : "unknown_error" });
    return res.status(502).json({
      error: error instanceof Error ? error.message : "chat_failed",
    });
  }
});

router.post("/chat/stream", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/chat/stream", req.user?.id, { hasSessionId: Boolean(req.body?.sessionId) });
  const question = String(req.body?.question || "").trim();
  const sessionIdInput = String(req.body?.sessionId || "").trim();
  if (!question) {
    logAiEnd("/api/ai/chat/stream", req.user?.id, start, false, { error: "question_missing" });
    return res.status(400).json({ error: "question is required" });
  }

  const session = await resolveSession(req.user.id, sessionIdInput, question);
  await ChatMessage.create({
    sessionId: session._id,
    userId: req.user.id,
    role: "user",
    content: question,
  });

  const conversation = await loadRecentConversation(session._id, req.user.id);
  const out = await runChat(question, {
    userId: req.user.id,
    sessionId: session._id.toString(),
    conversation,
    context: sanitizeContext(req.body?.context || {}),
  });

  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const text = String(out.answer || "");
  const chunks = text.match(/.{1,14}(\s|$)/g) || [text];
  let disconnected = false;
  req.on("close", () => {
    disconnected = true;
  });

  for (const part of chunks) {
    if (disconnected) break;
    res.write(`data: ${JSON.stringify({ type: "delta", token: part })}\n\n`);
    // Small delay so users can perceive stream effect.
    await sleep(18);
  }

  if (!disconnected) {
    await ChatMessage.create({
      sessionId: session._id,
      userId: req.user.id,
      role: "assistant",
      content: text,
      meta: {
        model: out.model,
        stack: out.stack,
        routedAgent: out.routedAgent,
        provider: out.provider,
        status: out.status,
        latencyMs: out.latencyMs,
        cached: out.cached,
      },
    });

    session.lastMessageAt = new Date();
    await session.save();

    res.write(
      `data: ${JSON.stringify({
        type: "done",
        answer: text,
        sessionId: session._id.toString(),
        model: out.model,
        provider: out.provider,
        status: out.status,
      })}\n\n`,
    );
  }

  logAiEnd("/api/ai/chat/stream", req.user?.id, start, !disconnected, {
    routedAgent: out.routedAgent,
    provider: out.provider,
    status: out.status,
    disconnected,
  });
  res.end();
});

router.post("/voice/transcribe", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/voice/transcribe", req.user?.id);
  const audioBase64 = String(req.body?.audioBase64 || "").trim();
  const mimeType = String(req.body?.mimeType || "audio/webm").trim() || "audio/webm";
  const locale = String(req.body?.locale || "hi-IN").trim() || "hi-IN";

  if (!audioBase64) {
    logAiEnd("/api/ai/voice/transcribe", req.user?.id, start, false, { error: "audio_missing" });
    return res.status(400).json({ error: "audioBase64 is required" });
  }

  // Guard against oversized uploads.
  if (audioBase64.length > 12_000_000) {
    logAiEnd("/api/ai/voice/transcribe", req.user?.id, start, false, { error: "audio_too_large" });
    return res.status(413).json({ error: "audio payload too large" });
  }

  try {
    const out = await transcribeWithDeepgram({ audioBase64, mimeType, locale });
    logAiEnd("/api/ai/voice/transcribe", req.user?.id, start, true, {
      provider: out.provider,
      model: out.model,
      transcriptChars: out.transcript.length,
      locale: out.locale,
    });
    return res.json(out);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "transcribe_failed";
    logAiEnd("/api/ai/voice/transcribe", req.user?.id, start, false, { error: detail });
    return res.status(502).json({ error: detail });
  }
});

router.post("/voice/respond", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/voice/respond", req.user?.id, { hasSessionId: Boolean(req.body?.sessionId) });
  const question = String(req.body?.question || "").trim();
  const sessionIdInput = String(req.body?.sessionId || "").trim();
  const locale = String(req.body?.locale || "hi-IN").trim() || "hi-IN";
  const voiceProfile = String(req.body?.voiceProfile || "female").trim().toLowerCase();
  if (!question) {
    logAiEnd("/api/ai/voice/respond", req.user?.id, start, false, { error: "question_missing" });
    return res.status(400).json({ error: "question is required" });
  }

  try {
    const session = await resolveSession(req.user.id, sessionIdInput, question);

    await ChatMessage.create({
      sessionId: session._id,
      userId: req.user.id,
      role: "user",
      content: `[voice:${locale}] ${question}`,
    });

    const conversation = await loadRecentConversation(session._id, req.user.id);
    let out;
    let voiceFallbackUsed = false;
    try {
      out = await runVoice(question, {
        userId: req.user.id,
        sessionId: session._id.toString(),
        conversation,
        locale,
        voiceProfile,
        context: sanitizeContext(req.body?.context || {}),
      });
    } catch (voiceErr) {
      console.warn(
        `[AI][VOICE_FALLBACK] user=${req.user?.id} reason=${voiceErr instanceof Error ? voiceErr.message : "voice_failed"}`,
      );
      const chatOut = await runChat(question, {
        userId: req.user.id,
        sessionId: session._id.toString(),
        conversation,
        context: sanitizeContext(req.body?.context || {}),
      });
      out = {
        ...chatOut,
        ttsText: String(chatOut?.answer || ""),
        voiceProfile,
        voiceProvider: "browser",
        status: "voice_fallback_text_only",
      };
      voiceFallbackUsed = true;
    }

    await ChatMessage.create({
      sessionId: session._id,
      userId: req.user.id,
      role: "assistant",
      content: out.answer,
      meta: {
        model: out.model,
        stack: out.stack,
        routedAgent: out.routedAgent,
        provider: out.provider,
        status: out.status,
        latencyMs: out.latencyMs,
        cached: out.cached,
        voice: true,
        ttsText: out.ttsText,
        locale,
        voiceProfile,
        voiceFallbackUsed,
      },
    });

    session.lastMessageAt = new Date();
    await session.save();

    logAiEnd("/api/ai/voice/respond", req.user?.id, start, true, {
      routedAgent: out.routedAgent,
      provider: out.provider,
      status: out.status,
      voiceProfile,
      locale,
      voiceFallbackUsed,
    });
    res.json({ ...out, sessionId: session._id.toString() });
  } catch (error) {
    logAiEnd("/api/ai/voice/respond", req.user?.id, start, false, { error: error instanceof Error ? error.message : "unknown_error" });
    return res.status(502).json({
      error: error instanceof Error ? error.message : "voice_failed",
    });
  }
});

router.post("/agent", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/agent", req.user?.id);
  const agent = String(req.body?.agent || "").trim();
  const question = String(req.body?.question || "").trim();
  if (!agent || !question) {
    logAiEnd("/api/ai/agent", req.user?.id, start, false, { error: "agent_or_question_missing" });
    return res.status(400).json({ error: "agent and question are required" });
  }

  try {
    const out = await runAgent(agent, question, { userId: req.user.id });
    logAiEnd("/api/ai/agent", req.user?.id, start, true, {
      agent: out.agent,
      provider: out.provider,
      status: out.status,
    });
    res.json(out);
  } catch (error) {
    logAiEnd("/api/ai/agent", req.user?.id, start, false, { error: error instanceof Error ? error.message : "unknown_error" });
    return res.status(502).json({
      error: error instanceof Error ? error.message : "agent_failed",
    });
  }
});

router.get("/status", authRequired, async (_req, res) => {
  const out = await getAiStatus();
  res.json(out);
});

router.get("/chat/sessions", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/chat/sessions", req.user?.id);
  const sessions = await ChatSession.find({ userId: req.user.id })
    .sort({ lastMessageAt: -1 })
    .limit(30)
    .lean();

  logAiEnd("/api/ai/chat/sessions", req.user?.id, start, true, { count: sessions.length });
  res.json({
    sessions: sessions.map((s) => ({
      id: s._id.toString(),
      title: s.title,
      lastMessageAt: s.lastMessageAt,
      createdAt: s.createdAt,
    })),
  });
});

router.get("/chat/sessions/:id/messages", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/chat/sessions/:id/messages", req.user?.id, { sessionId: req.params.id });
  const session = await ChatSession.findOne({ _id: req.params.id, userId: req.user.id }).lean();
  if (!session) {
    logAiEnd("/api/ai/chat/sessions/:id/messages", req.user?.id, start, false, { error: "session_not_found" });
    return res.status(404).json({ error: "session not found" });
  }

  const messages = await ChatMessage.find({ sessionId: session._id, userId: req.user.id })
    .sort({ createdAt: 1 })
    .lean();

  logAiEnd("/api/ai/chat/sessions/:id/messages", req.user?.id, start, true, { messageCount: messages.length });
  res.json({
    session: {
      id: session._id.toString(),
      title: session.title,
      lastMessageAt: session.lastMessageAt,
    },
    messages: messages.map((m) => ({
      id: m._id.toString(),
      role: m.role,
      content: m.content,
      createdAt: m.createdAt,
    })),
  });
});

router.delete("/chat/sessions/:id", authRequired, async (req, res) => {
  const start = logAiStart("/api/ai/chat/sessions/:id", req.user?.id, { sessionId: req.params.id });
  const session = await ChatSession.findOne({ _id: req.params.id, userId: req.user.id }).lean();
  if (!session) {
    logAiEnd("/api/ai/chat/sessions/:id", req.user?.id, start, false, { error: "session_not_found" });
    return res.status(404).json({ error: "session not found" });
  }

  await ChatMessage.deleteMany({ sessionId: session._id, userId: req.user.id });
  await ChatSession.deleteOne({ _id: session._id, userId: req.user.id });

  logAiEnd("/api/ai/chat/sessions/:id", req.user?.id, start, true, { deleted: true });
  return res.json({ success: true });
});

export default router;
