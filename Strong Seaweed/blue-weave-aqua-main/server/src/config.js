import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT || 4000),
  mongoUri: process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/blueweave",
  jwtSecret: process.env.JWT_SECRET || "change-me",
  corsOrigins: (process.env.CORS_ORIGINS || "http://127.0.0.1:8080,http://localhost:8080")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean),
  modelApiUrl: process.env.MODEL_API_URL || "http://127.0.0.1:8000",
  aiTimeoutMs: Number(process.env.AI_TIMEOUT_MS || 12000),
  aiRouterMode: process.env.AI_ROUTER_MODE || "hybrid",
  langChainApiUrl: process.env.LANGCHAIN_API_URL || "http://127.0.0.1:8101",
  langGraphApiUrl: process.env.LANGGRAPH_API_URL || "http://127.0.0.1:8101",
  crewAiApiUrl: process.env.CREWAI_API_URL || "http://127.0.0.1:8101",
  deepgramApiKey: process.env.DEEPGRAM_API_KEY || "",
  deepgramSttModel: process.env.DEEPGRAM_STT_MODEL || "flux-general-en",
  deepgramSttApiUrl: process.env.DEEPGRAM_STT_API_URL || "https://api.deepgram.com/v1/listen",
  deepgramTtsModel: process.env.DEEPGRAM_TTS_MODEL || "aura-2-thalia-en",
  deepgramTtsModelHi: process.env.DEEPGRAM_TTS_MODEL_HI || "",
  deepgramTtsModelGu: process.env.DEEPGRAM_TTS_MODEL_GU || "",
  deepgramTtsModelFemale: process.env.DEEPGRAM_TTS_MODEL_FEMALE || "aura-2-thalia-en",
  deepgramTtsModelMale: process.env.DEEPGRAM_TTS_MODEL_MALE || "aura-2-orion-en",
  deepgramTtsApiUrl: process.env.DEEPGRAM_TTS_API_URL || "https://api.deepgram.com/v1/speak",
  deepgramTtsStrictVoice: String(process.env.DEEPGRAM_TTS_STRICT_VOICE || "false").toLowerCase() === "true",
  elevenLabsApiKey: process.env.ELEVENLABS_API_KEY || "",
  elevenLabsVoiceId: process.env.ELEVENLABS_VOICE_ID || "",
  elevenLabsVoiceIdFemale: process.env.ELEVENLABS_VOICE_ID_FEMALE || "",
  elevenLabsVoiceIdMale: process.env.ELEVENLABS_VOICE_ID_MALE || "",
  elevenLabsModelId: process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2",
  elevenLabsOutputFormat: process.env.ELEVENLABS_OUTPUT_FORMAT || "mp3_44100_128",
  elevenLabsStrictVoice: String(process.env.ELEVENLABS_STRICT_VOICE || "false").toLowerCase() === "true",
  voiceTtsProvider: String(process.env.VOICE_TTS_PROVIDER || "deepgram").toLowerCase(),
};
