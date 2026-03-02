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
  elevenLabsApiKey: process.env.ELEVENLABS_API_KEY || "",
  elevenLabsVoiceId: process.env.ELEVENLABS_VOICE_ID || "",
  elevenLabsVoiceIdFemale: process.env.ELEVENLABS_VOICE_ID_FEMALE || "",
  elevenLabsVoiceIdMale: process.env.ELEVENLABS_VOICE_ID_MALE || "",
  elevenLabsModelId: process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2",
  elevenLabsOutputFormat: process.env.ELEVENLABS_OUTPUT_FORMAT || "mp3_44100_128",
  elevenLabsStrictVoice: String(process.env.ELEVENLABS_STRICT_VOICE || "false").toLowerCase() === "true",
  voiceTtsProvider: String(process.env.VOICE_TTS_PROVIDER || "browser").toLowerCase(),
};
