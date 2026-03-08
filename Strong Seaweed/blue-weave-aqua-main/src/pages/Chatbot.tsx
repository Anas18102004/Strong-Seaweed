import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  History,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Radio,
  PhoneOff,
  Plus,
  Clock3,
  Menu,
  Trash2,
  Search,
  SlidersHorizontal,
  Paperclip,
  BrainCircuit,
  ShieldCheck,
  Database,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import BrandLogo from "@/components/BrandLogo";
import { useLocation } from "react-router-dom";

type Message = {
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  context?: string;
  model?: string;
};

type InputMode = "ask" | "analyze" | "forecast" | "compare";

type SpeechRecognitionAlternativeLite = { transcript: string };
type SpeechRecognitionResultLite = {
  isFinal: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlternativeLite;
};
type SpeechRecognitionResultListLite = {
  length: number;
  [index: number]: SpeechRecognitionResultLite;
};
type SpeechRecognitionEventLite = {
  resultIndex: number;
  results: SpeechRecognitionResultListLite;
};

type SpeechRecognitionErrorEventLite = {
  error?: string;
  message?: string;
};

type BrowserSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLite) => void) | null;
  onresult: ((event: SpeechRecognitionEventLite) => void) | null;
  start: () => void;
  stop: () => void;
};

declare global {
  interface Window {
    webkitSpeechRecognition?: new () => BrowserSpeechRecognition;
    SpeechRecognition?: new () => BrowserSpeechRecognition;
  }
}

const suggestedPrompts = [
  "What should I grow in Gulf of Mannar?",
  "Explain current risk in simple terms",
  "How can I improve my yield this month?",
  "Which new location is worth testing next?",
];

const locationToCoords: Record<string, { lat: number; lon: number }> = {
  "Gulf of Mannar": { lat: 9.1, lon: 79.3 },
  "Palk Bay": { lat: 9.4, lon: 79.2 },
  Lakshadweep: { lat: 10.5, lon: 72.7 },
  "Andaman Islands": { lat: 11.7, lon: 92.7 },
  "Gulf of Kachchh": { lat: 22.6, lon: 69.8 },
  Chilika: { lat: 19.7, lon: 85.3 },
  Ratnagiri: { lat: 16.9, lon: 73.3 },
  Karwar: { lat: 14.8, lon: 74.1 },
  Kollam: { lat: 8.9, lon: 76.6 },
};

const locationPresets = Object.keys(locationToCoords);
const speciesPresets = ["Kappaphycus", "Gracilaria", "Sargassum", "Ulva", "Any"];
const seasonPresets = ["Current", "Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter", "Any"];

function resolveContextCoords(input: string): { lat: number; lon: number } | undefined {
  const raw = String(input || "").trim();
  if (!raw) return undefined;

  const direct = locationToCoords[raw];
  if (direct) return direct;

  const lower = raw.toLowerCase();
  const byName = Object.entries(locationToCoords).find(([name]) => name.toLowerCase() === lower)?.[1];
  if (byName) return byName;

  const match = raw.match(/(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)/);
  if (!match) return undefined;
  const lat = Number(match[1]);
  const lon = Number(match[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return undefined;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return undefined;
  return { lat, lon };
}

function renderInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={idx} className="font-semibold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={idx}>{part}</span>;
  });
}

function renderAssistantText(raw: string) {
  const lines = String(raw || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter((line, idx, arr) => !(line === "" && arr[idx - 1] === ""));

  return (
    <div className="space-y-2">
      {lines.map((line, idx) => {
        if (/^#{1,3}\s+/.test(line)) {
          const clean = line.replace(/^#{1,3}\s+/, "");
          return (
            <p key={idx} className="font-semibold text-slate-900">
              {renderInline(clean)}
            </p>
          );
        }

        if (/^[-*]\s+/.test(line) || /^\u2022\s+/.test(line)) {
          const clean = line.replace(/^[-*]\s+/, "").replace(/^\u2022\s+/, "");
          return (
            <div key={idx} className="flex items-start gap-2 text-slate-800">
              <span className="mt-1 text-cyan-700">-</span>
              <p className="leading-relaxed">{renderInline(clean)}</p>
            </div>
          );
        }

        if (line.includes("|")) {
          const clean = line
            .replace(/\|/g, " ")
            .replace(/\s{2,}/g, " ")
            .trim();
          if (!clean || /^[-: ]+$/.test(clean)) return null;
          return (
            <p key={idx} className="leading-relaxed text-slate-800">
              {renderInline(clean)}
            </p>
          );
        }

        return (
          <p key={idx} className="leading-relaxed text-slate-800">
            {renderInline(line)}
          </p>
        );
      })}
    </div>
  );
}

function getDateGroupLabel(value?: string) {
  if (!value) return "No Date";
  const now = new Date();
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "No Date";
  const startNow = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startD = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diffDays = Math.floor((startNow - startD) / 86400000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays <= 7) return "Last 7 Days";
  return "Earlier";
}

function getExecutionSteps() {
  return ["Analyzing marine context", "Validating constraints", "Running suitability heuristics"];
}

const DEEPGRAM_CLIP_MS = 4200;

export default function Chatbot() {
  const welcomeMessage: Message = {
    role: "assistant",
    content: "Hi. I am your Seaweed AI helper. Ask a practical question and I will answer in plain language.",
    confidence: 94,
    context: "Gulf Zone",
    model: "Marine Core v2",
  };
  const [messages, setMessages] = useState<Message[]>([
    welcomeMessage,
  ]);
  const [input, setInput] = useState("");
  const [inputMode, setInputMode] = useState<InputMode>("ask");
  const [contextLocation, setContextLocation] = useState("Gulf of Mannar");
  const [contextSpecies, setContextSpecies] = useState("Kappaphycus");
  const [contextSeason, setContextSeason] = useState("Current");
  const [sessionSearch, setSessionSearch] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [sessions, setSessions] = useState<{ id: string; title: string; lastMessageAt?: string }[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [showSidebar, setShowSidebar] = useState(false);
  const [isDraftNewChat, setIsDraftNewChat] = useState(false);

  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [voiceMode, setVoiceMode] = useState(false);
  const [voiceSessionActive, setVoiceSessionActive] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [voiceError, setVoiceError] = useState("");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [streamingAssistant, setStreamingAssistant] = useState("");
  const [sttLocale, setSttLocale] = useState("hi-IN");
  const [orbOffset, setOrbOffset] = useState({ x: 0, y: 0 });

  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const lastRecognitionErrorRef = useRef("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaChunksRef = useRef<BlobPart[]>([]);
  const deepgramStopTimerRef = useRef<number | null>(null);
  const voiceResumeTimerRef = useRef<number | null>(null);
  const deepgramEnabledRef = useRef(true);
  const voiceSessionActiveRef = useRef(false);
  const voiceModeRef = useRef(false);
  const isTypingRef = useRef(false);
  const listeningRef = useRef(false);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { token } = useAuth();
  const location = useLocation();

  useEffect(() => {
    const browserLang = (navigator.language || "").toLowerCase();
    if (browserLang.startsWith("hi")) {
      setSttLocale("hi-IN");
      return;
    }
    if (browserLang.startsWith("gu")) {
      setSttLocale("gu-IN");
      return;
    }
    if (browserLang.startsWith("en")) {
      setSttLocale("en-IN");
      return;
    }
    setSttLocale("hi-IN");
  }, []);

  useEffect(() => {
    const prefill = (location.state as { prefill?: string } | null)?.prefill;
    if (prefill && !input.trim()) {
      setInput(prefill);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  useEffect(() => {
    if (voiceMode) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping, voiceMode]);

  useEffect(() => {
    voiceSessionActiveRef.current = voiceSessionActive;
  }, [voiceSessionActive]);

  useEffect(() => {
    voiceModeRef.current = voiceMode;
  }, [voiceMode, voiceSupported]);

  useEffect(() => {
    isTypingRef.current = isTyping;
  }, [isTyping]);

  useEffect(() => {
    listeningRef.current = listening;
  }, [listening]);

  const normalizeVoiceLocale = (locale: string) => {
    const lower = String(locale || "").toLowerCase();
    if (lower.startsWith("hi")) return "hi-IN";
    if (lower.startsWith("gu")) return "gu-IN";
    if (lower.startsWith("en")) return "en-IN";
    return "hi-IN";
  };

  const localeVoiceCandidates = (locale: string) => {
    const normalized = normalizeVoiceLocale(locale).toLowerCase();
    if (normalized.startsWith("gu")) return ["gu", "hi", "en"];
    if (normalized.startsWith("hi")) return ["hi", "en"];
    return ["en", "hi"];
  };

  const pickFemaleVoice = (locale: string) => {
    if (!("speechSynthesis" in window)) return null;
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;
    const langPriority = localeVoiceCandidates(locale);
    const byName = voices.find((v) => {
      if (!/(female|woman|samantha|aria|ava|lisa|susan|zira|heera|veena|pooja|kajal|lekha)/i.test(v.name)) {
        return false;
      }
      const lang = String(v.lang || "").toLowerCase();
      return langPriority.some((prefix) => lang.startsWith(prefix));
    });
    if (byName) return byName;
    const byLang = voices.find((v) => {
      const lang = String(v.lang || "").toLowerCase();
      return langPriority.some((prefix) => lang.startsWith(prefix));
    });
    return byLang || voices[0] || null;
  };

  const speak = (text: string, locale = sttLocale) => {
    if (!voiceEnabled || !("speechSynthesis" in window)) return;
    const clean = String(text || "").replace(/\s+/g, " ").trim();
    if (!clean) return;

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.lang = normalizeVoiceLocale(locale);
      utterance.rate = 1.02;
      utterance.pitch = 1.07;
      const female = pickFemaleVoice(utterance.lang);
      if (female) utterance.voice = female;
      window.speechSynthesis.speak(utterance);
    } catch {
      // no-op
    }
  };

  const stopVoiceOutput = () => {
    try {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    } catch {
      // no-op
    }
    try {
      const a = activeAudioRef.current;
      if (a) {
        a.pause();
        a.currentTime = 0;
      }
      activeAudioRef.current = null;
    } catch {
      // no-op
    }
  };

  const playAudioBase64 = (audioBase64?: string, audioMime?: string) => {
    if (!audioBase64) return false;
    try {
      const src = `data:${audioMime || "audio/mpeg"};base64,${audioBase64}`;
      const audio = new Audio(src);
      activeAudioRef.current = audio;
      audio.onended = () => {
        if (activeAudioRef.current === audio) activeAudioRef.current = null;
      };
      void audio.play();
      return true;
    } catch {
      return false;
    }
  };

  const clearDeepgramTimer = () => {
    if (deepgramStopTimerRef.current !== null) {
      window.clearTimeout(deepgramStopTimerRef.current);
      deepgramStopTimerRef.current = null;
    }
  };

  const clearVoiceResumeTimer = () => {
    if (voiceResumeTimerRef.current !== null) {
      window.clearTimeout(voiceResumeTimerRef.current);
      voiceResumeTimerRef.current = null;
    }
  };

  const scheduleVoiceResume = (delayMs = 260, retries = 8) => {
    clearVoiceResumeTimer();
    const attempt = (remaining: number) => {
      if (!voiceModeRef.current || !voiceSessionActiveRef.current) return;
      if (listeningRef.current) return;
      if (isTypingRef.current) {
        if (remaining <= 0) return;
        voiceResumeTimerRef.current = window.setTimeout(() => attempt(remaining - 1), 320);
        return;
      }
      void startListening();
      if (remaining <= 0) return;
      voiceResumeTimerRef.current = window.setTimeout(() => {
        if (!listeningRef.current && voiceModeRef.current && voiceSessionActiveRef.current) {
          attempt(remaining - 1);
        }
      }, 480);
    };
    voiceResumeTimerRef.current = window.setTimeout(() => attempt(retries), delayMs);
  };

  const stopDeepgramRecorder = () => {
    clearDeepgramTimer();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        // no-op
      }
    }
    const stream = mediaStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  };

  const blobToBase64 = async (blob: Blob) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const raw = String(reader.result || "");
        const base64 = raw.includes(",") ? raw.split(",")[1] : "";
        if (!base64) {
          reject(new Error("audio_encode_failed"));
          return;
        }
        resolve(base64);
      };
      reader.onerror = () => reject(new Error("audio_encode_failed"));
      reader.readAsDataURL(blob);
    });

  const pickRecorderMimeType = () => {
    if (typeof MediaRecorder === "undefined") return "";
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    return candidates.find((x) => MediaRecorder.isTypeSupported(x)) || "";
  };

  const startDeepgramCapture = async () => {
    if (!token) throw new Error("Authentication required for Deepgram voice transcription.");
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      throw new Error("Media recorder is not supported in this browser.");
    }
    if (isTypingRef.current || listeningRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;
    const mimeType = pickRecorderMimeType();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    mediaRecorderRef.current = recorder;
    mediaChunksRef.current = [];

    recorder.onstart = () => {
      setListening(true);
      setVoiceError("");
      setLiveTranscript("Listening...");
    };
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        mediaChunksRef.current.push(event.data);
      }
    };
    recorder.onerror = () => {
      setListening(false);
      setVoiceError("Audio capture failed. Please retry.");
      stopDeepgramRecorder();
    };
    recorder.onstop = async () => {
      clearDeepgramTimer();
      setListening(false);

      const streamRef = mediaStreamRef.current;
      if (streamRef) {
        streamRef.getTracks().forEach((t) => t.stop());
        mediaStreamRef.current = null;
      }

      const chunks = [...mediaChunksRef.current];
      mediaChunksRef.current = [];
      mediaRecorderRef.current = null;

      // User explicitly stopped voice mode.
      if (!voiceSessionActiveRef.current || !voiceModeRef.current) return;
      if (!chunks.length) {
        setVoiceError("No audio captured. Please try again.");
        scheduleVoiceResume(260);
        return;
      }

      const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
      try {
        const audioBase64 = await blobToBase64(blob);
        const stt = await api.voiceTranscribe(audioBase64, blob.type || "audio/webm", sttLocale, token || undefined);
        const transcript = String(stt?.transcript || "").trim();
        if (!transcript) {
          setVoiceError("No speech detected. Please speak clearly and try again.");
          scheduleVoiceResume(260);
          return;
        }

        setLiveTranscript(transcript);
        await sendMessage(transcript, "voice", { autoResumeVoice: true });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Deepgram transcription failed.";
        if (message.includes("deepgram_not_configured") || message.includes("401")) {
          deepgramEnabledRef.current = false;
          setVoiceError("Deepgram STT is not configured on server. Falling back to browser STT.");
          return;
        }
        if (message.includes("deepgram_http_400") || message.toLowerCase().includes("language")) {
          deepgramEnabledRef.current = false;
          setVoiceError(`Deepgram STT does not support ${sttLocale} for this setup. Falling back to browser STT.`);
          return;
        }
        setVoiceError(`Deepgram STT failed. ${message}`);
      } finally {
        scheduleVoiceResume(260);
      }
    };

    recorder.start();
    deepgramStopTimerRef.current = window.setTimeout(() => {
      const activeRecorder = mediaRecorderRef.current;
      if (activeRecorder && activeRecorder.state === "recording") {
        try {
          activeRecorder.stop();
        } catch {
          // no-op
        }
      }
    }, DEEPGRAM_CLIP_MS);
  };

  const stopListening = () => {
    clearVoiceResumeTimer();
    stopDeepgramRecorder();
    setListening(false);
    const recognition = recognitionRef.current;
    if (!recognition) return;
    try {
      recognition.stop();
    } catch {
      // no-op
    }
  };

  const startListening = async () => {
    if (!voiceModeRef.current || !voiceSessionActiveRef.current) return;
    if (isTypingRef.current || listeningRef.current) return;

    // Prefer Deepgram STT to avoid browser SpeechRecognition network instability.
    if (deepgramEnabledRef.current && token && navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined") {
      try {
        await startDeepgramCapture();
        return;
      } catch (error) {
        setVoiceError(error instanceof Error ? error.message : "Could not start Deepgram STT.");
      }
    }

    const recognition = recognitionRef.current;
    if (!recognition) {
      setVoiceError("Voice input is not supported in this browser.");
      return;
    }
    try {
      // Prefer explicit permission prompt when supported.
      if (navigator.mediaDevices?.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((t) => t.stop());
      }

      setVoiceError("");
      recognition.start();
    } catch (error) {
      setVoiceError(error instanceof Error ? error.message : "Microphone could not start.");
    }
  };

  const loadSessions = async () => {
    if (!token) return;
    try {
      const res = await api.getChatSessions(token);
      const items = (res.sessions || []).map((s) => ({ id: s.id, title: s.title, lastMessageAt: s.lastMessageAt }));
      setSessions(items);
      if (!sessionId && !isDraftNewChat && items.length > 0) {
        await openSession(items[0].id);
      }
    } catch {
      setSessions([]);
    } finally {
      setLoadingSessions(false);
    }
  };

  const startNewChat = () => {
    setSessionId(undefined);
    setIsDraftNewChat(true);
    setMessages([welcomeMessage]);
    setInput("");
    setLiveTranscript("");
    setShowSidebar(false);
  };

  const formatTime = (value?: string) => {
    if (!value) return "";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  const openSession = async (id: string) => {
    if (!token) return;
    try {
      const out = await api.getChatMessages(id, token);
      setSessionId(id);
      setIsDraftNewChat(false);
      setShowSidebar(false);
      const list = out.messages.map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content } as Message));
      const hydrated = list.map((m) =>
        m.role === "assistant" ? { ...m, confidence: 91, context: contextLocation, model: "Marine Core v2" } : m,
      );
      setMessages(
        hydrated.length
          ? hydrated
          : [{ role: "assistant", content: "This session is empty. Ask your first question.", confidence: 93, model: "Marine Core v2" }],
      );
    } catch {
      setMessages([{ role: "assistant", content: "Could not load this session.", confidence: 0, model: "Marine Core v2" }]);
    }
  };

  const deleteSession = async (id: string) => {
    if (!token) return;
    try {
      await api.deleteChatSession(id, token);
      const isCurrent = sessionId === id;
      if (isCurrent) {
        startNewChat();
      }
      await loadSessions();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: err instanceof Error ? err.message : "Could not delete this chat." },
      ]);
    }
  };

  useEffect(() => {
    void loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const sendMessage = async (
    text?: string,
    mode: "text" | "voice" = "text",
    options: { autoResumeVoice?: boolean } = {},
  ) => {
    const base = String(text || input || "").trim();
    const msg = base;
    if (!msg.trim()) return;
    const selectedCoords = resolveContextCoords(contextLocation);
    const aiContext = {
      mode: inputMode,
      locationName: contextLocation,
      speciesHint: contextSpecies,
      season: contextSeason,
      lat: selectedCoords?.lat,
      lon: selectedCoords?.lon,
    };

    setMessages((prev) => [...prev, { role: "user", content: base }]);
    setInput("");
    setLiveTranscript("");
    setVoiceError("");
    setIsTyping(true);
    setStreamingAssistant("");

    try {
      if (mode === "voice") {
        const res = await api.voiceRespond(msg, token || undefined, sessionId, sttLocale, "female", aiContext);
        if (res.sessionId) {
          setSessionId(res.sessionId);
          setIsDraftNewChat(false);
        }
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: res.answer, confidence: 92, context: contextLocation, model: "Marine Core v2" },
        ]);

        const played = playAudioBase64(res.audioBase64, res.audioMime);
        if (!played) speak(res.ttsText || res.answer, sttLocale);
      } else {
        try {
          let built = "";
          const res = await api.chatStream(msg, token || undefined, sessionId, aiContext, (chunk) => {
            built += chunk;
            setStreamingAssistant(built);
          });
          if (res.sessionId) {
            setSessionId(res.sessionId);
            setIsDraftNewChat(false);
          }
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: (res.answer || built).trim(),
              confidence: 92,
              context: contextLocation,
              model: "Marine Core v2",
            },
          ]);
          setStreamingAssistant("");
        } catch {
          setStreamingAssistant("");
          const res = await api.chat(msg, token || undefined, sessionId, aiContext);
          if (res.sessionId) {
            setSessionId(res.sessionId);
            setIsDraftNewChat(false);
          }
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: res.answer, confidence: 92, context: contextLocation, model: "Marine Core v2" },
          ]);
        }
      }

      await loadSessions();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Chat request failed.";
      if (mode === "voice") {
        try {
          const selectedCoords = resolveContextCoords(contextLocation);
          const aiContext = {
            mode: inputMode,
            locationName: contextLocation,
            speciesHint: contextSpecies,
            season: contextSeason,
            lat: selectedCoords?.lat,
            lon: selectedCoords?.lon,
          };
          const fallback = await api.chat(msg, token || undefined, sessionId, aiContext);
          if (fallback.sessionId) {
            setSessionId(fallback.sessionId);
            setIsDraftNewChat(false);
          }
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: fallback.answer,
              confidence: 88,
              context: contextLocation,
              model: "Marine Core v2",
            },
          ]);
          setVoiceError("Voice pipeline unavailable. Switched to text response with browser female TTS fallback.");
          speak(fallback.answer, sttLocale);
          await loadSessions();
          return;
        } catch {
          setVoiceError(message);
        }
      }
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: message, confidence: 0, context: contextLocation, model: "Marine Core v2" },
      ]);
    } finally {
      setIsTyping(false);
      if (options.autoResumeVoice && voiceModeRef.current && voiceSupported && voiceSessionActiveRef.current) {
        scheduleVoiceResume(260);
      }
    }
  };

  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const hasDeepgramCapture = Boolean(navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined");
    const hasBrowserStt = Boolean(SR);
    setVoiceSupported(hasDeepgramCapture || hasBrowserStt);

    if (!SR) {
      recognitionRef.current = null;
      return;
    }
    const recognition = new SR();
    recognition.lang = sttLocale;
    recognition.interimResults = true;
    recognition.continuous = false;
    try {
      (recognition as BrowserSpeechRecognition & { maxAlternatives?: number }).maxAlternatives = 3;
    } catch {
      // no-op for browsers that don't support this flag
    }

    recognition.onstart = () => {
      setListening(true);
      lastRecognitionErrorRef.current = "";
    };
    recognition.onend = () => {
      setListening(false);
      const lastErr = lastRecognitionErrorRef.current;
      const blockAutoResume =
        lastErr === "network" || lastErr === "not-allowed" || lastErr === "service-not-allowed";
      if (blockAutoResume) {
        voiceSessionActiveRef.current = false;
        setVoiceSessionActive(false);
        return;
      }
      if (voiceModeRef.current && voiceSessionActiveRef.current && voiceSupported) {
        scheduleVoiceResume(220);
      }
    };
    recognition.onerror = (event: SpeechRecognitionErrorEventLite) => {
      setListening(false);
      const code = String(event?.error || "").toLowerCase();
      lastRecognitionErrorRef.current = code;
      if (code === "not-allowed" || code === "service-not-allowed") {
        setVoiceError("Microphone permission denied. Please allow mic access in browser settings.");
        return;
      }
      if (code === "no-speech") {
        setVoiceError("No speech detected. Please speak clearly and try again.");
        return;
      }
      if (code === "network") {
        if (sttLocale !== "en-IN") {
          setSttLocale("en-IN");
          setVoiceError("Speech recognition network error. Switched speech locale to English (India). Press Start Voice again.");
          return;
        }
        setVoiceError("Speech recognition network error. Voice session paused. Use Chrome/Edge over HTTPS and press Start Voice.");
        return;
      }
      setVoiceError(event?.message || "Voice input failed. Try again.");
    };
    recognition.onresult = (event: SpeechRecognitionEventLite) => {
      let interim = "";
      let finalized = "";

      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const r = event.results[i];
        const segment = String(r?.[0]?.transcript || "").trim();
        if (!segment) continue;
        if (r.isFinal) finalized += `${segment} `;
        else interim += `${segment} `;
      }

      setLiveTranscript(interim.trim());

      const finalText = finalized.trim();
      if (!finalText || isTyping) return;
      stopListening();
      void sendMessage(finalText, "voice", { autoResumeVoice: false });
    };

    recognitionRef.current = recognition;

    return () => {
      try {
        recognition.stop();
      } catch {
        // no-op
      }
      recognitionRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTyping, voiceMode, voiceSupported, sttLocale, voiceSessionActive]);

  useEffect(() => {
    if (!voiceMode) {
      voiceSessionActiveRef.current = false;
      setVoiceSessionActive(false);
      stopListening();
      stopVoiceOutput();
      setLiveTranscript("");
      setVoiceError("");
      return;
    }
    if (!voiceSupported) {
      setVoiceError("Voice input is not supported in this browser.");
      return;
    }
    voiceSessionActiveRef.current = true;
    setVoiceSessionActive(true);
    setVoiceError("");
    setLiveTranscript("Starting voice channel...");
    scheduleVoiceResume(180, 10);
  }, [voiceMode, voiceSupported]);

  useEffect(() => {
    if (!voiceEnabled) {
      stopVoiceOutput();
    }
  }, [voiceEnabled]);

  useEffect(
    () => () => {
      stopListening();
      stopVoiceOutput();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const filteredSessions = sessions.filter((s) => s.title.toLowerCase().includes(sessionSearch.toLowerCase()));
  const groupedSessions = filteredSessions.reduce<Record<string, { id: string; title: string; lastMessageAt?: string }[]>>((acc, s) => {
    const label = getDateGroupLabel(s.lastMessageAt);
    if (!acc[label]) acc[label] = [];
    acc[label].push(s);
    return acc;
  }, {});

  const sessionGroupOrder = ["Today", "Yesterday", "Last 7 Days", "Earlier", "No Date"];
  const modeOptions: { id: InputMode; label: string }[] = [
    { id: "ask", label: "Ask" },
    { id: "analyze", label: "Analyze" },
    { id: "forecast", label: "Forecast" },
    { id: "compare", label: "Compare" },
  ];

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-1 sm:px-0 h-[calc(100dvh-8.6rem)] sm:h-[calc(100dvh-7rem)] relative pb-2">
        <div className="pointer-events-none absolute -top-12 -left-10 h-60 w-60 rounded-full bg-cyan-300/25 blur-3xl" />
        <div className="pointer-events-none absolute top-24 right-[-2rem] h-72 w-72 rounded-full bg-blue-400/25 blur-3xl" />

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-3 sm:mb-4 rounded-[20px] border border-[#a8c7dc]/70 bg-white/85 p-4 sm:p-5 shadow-[0_14px_34px_-26px_rgba(13,72,110,0.62)]"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-[0.14em] text-cyan-800 mb-1.5">
                <BrainCircuit className="w-3.5 h-3.5" />
                Marine Intelligence Copilot
              </p>
              <h1 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1">AI Operations Command</h1>
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700"><ShieldCheck className="h-3.5 w-3.5" /> Operational</span>
                <span className="h-3.5 w-px bg-slate-300/90" />
                <span>Model: Marine Core v2</span>
                <span className="h-3.5 w-px bg-slate-300/90" />
                <span>Streaming: Live</span>
                <span className="h-3.5 w-px bg-slate-300/90" />
                <span>Context Window: 128k</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="glass"
                size="icon"
                className="w-10 h-10 md:hidden"
                onClick={() => setShowSidebar((v) => !v)}
                title="Toggle chat history"
              >
                <Menu className="w-4 h-4" />
              </Button>
              <Button
                type="button"
                variant={voiceMode ? "hero" : "glass"}
                onClick={() =>
                  setVoiceMode((v) => {
                    const next = !v;
                    if (!next) {
                      voiceSessionActiveRef.current = false;
                      setVoiceSessionActive(false);
                      stopListening();
                      stopVoiceOutput();
                    }
                    return next;
                  })
                }
                className="min-h-10"
                title="Switch voice mode"
              >
                <Radio className="w-4 h-4" /> {voiceMode ? "Voice Channel Active" : "Open Voice Channel"}
              </Button>
              <Button
                type="button"
                variant="glass"
                size="icon"
                onClick={() => setVoiceEnabled((v) => !v)}
                className="w-10 h-10"
                title={voiceEnabled ? "Disable spoken replies" : "Enable spoken replies"}
              >
                {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </Button>
            </div>
          </div>
        </motion.div>

        {voiceMode ? (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="voice-immersive-shell rounded-[30px] p-3 sm:p-4 h-[calc(100%-5.2rem)] sm:h-[calc(100%-5.6rem)]">
            <div
              className="voice-mode-panel rounded-[28px] p-6 sm:p-8 text-center relative overflow-hidden h-full flex flex-col"
              onMouseMove={(e) => {
                const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width - 0.5) * 12;
                const y = ((e.clientY - rect.top) / rect.height - 0.5) * 12;
                setOrbOffset({ x, y });
              }}
              onMouseLeave={() => setOrbOffset({ x: 0, y: 0 })}
            >
              <div className="voice-grid-glow" />
              <div className="voice-noise" />
              <div className="voice-particle-layer" />

              <div className="relative z-10 flex h-full flex-col">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-6">
                  <div className="voice-chip bg-cyan-300/10 border-cyan-200/25">
                    <span className={`voice-chip-dot ${listening ? "voice-chip-dot-live" : ""}`} />
                    AI Voice Core
                  </div>
                  <div className="voice-chip text-cyan-100/80 bg-white/5">Voice: Female</div>
                  <label className="voice-chip text-cyan-100/90 bg-white/5">
                    STT:
                    <select
                      value={sttLocale}
                      onChange={(e) => setSttLocale(e.target.value)}
                      className="ml-2 bg-transparent outline-none"
                      aria-label="Speech recognition locale"
                    >
                      <option value="hi-IN">Hindi (India)</option>
                      <option value="gu-IN">Gujarati (India)</option>
                      <option value="en-IN">English (India)</option>
                    </select>
                  </label>
                </div>

                <div className={`voice-orb ${listening ? "voice-orb-live" : ""} ${isTyping ? "voice-orb-thinking" : ""}`} style={{ transform: `translate(${orbOffset.x}px, ${orbOffset.y}px)` }}>
                  <div className="voice-orb-ring" />
                  <div className="voice-orb-ring voice-orb-ring-2" />
                  <div className="voice-orb-ring voice-orb-ring-3" />
                  <div className="voice-orb-core">
                    <div className="voice-monogram" role="img" aria-label="Akuara Voice Monogram">
                      <span className="voice-monogram-text">AK</span>
                      <span className="voice-monogram-wave">AI</span>
                      <span className="voice-monogram-glint" />
                    </div>
                  </div>
                  <div className="voice-bars" aria-hidden="true">
                    {Array.from({ length: 7 }).map((_, i) => (
                      <span key={i} className="voice-bar" style={{ animationDelay: `${i * 0.12}s` }} />
                    ))}
                  </div>
                </div>

                <div className="mt-auto" />
                <div className="voice-status-bar mt-8">
                  <p className="text-sm sm:text-base text-cyan-50 font-medium">
                    {isTyping
                      ? "Thinking and preparing your voice response..."
                      : listening
                        ? "Listening live. Speak naturally."
                        : voiceSessionActive
                          ? "Voice standby. Listening will auto-resume."
                          : voiceSupported
                            ? "Voice channel ready."
                            : "Voice input is not supported in this browser."}
                  </p>
                  {isTyping && (
                    <div className="voice-thinking-dots mt-2">
                      <span />
                      <span />
                      <span />
                    </div>
                  )}
                </div>

                <div className="voice-transcript mt-3">
                  <p className="text-xs text-cyan-100/90 min-h-5">{liveTranscript || "Waiting for your voice..."}</p>
                  {voiceError ? <p className="text-xs text-red-200 mt-2">{voiceError}</p> : null}
                </div>

                <div className="mt-6 flex flex-wrap justify-center gap-3">
                  <Button
                    type="button"
                    variant={listening ? "hero-outline" : "hero"}
                    className="voice-mic-btn min-h-12 min-w-44 rounded-full"
                    onClick={() => {
                      if (voiceSessionActive || listening) {
                        voiceSessionActiveRef.current = false;
                        setVoiceSessionActive(false);
                        stopListening();
                      } else {
                        voiceSessionActiveRef.current = true;
                        setVoiceSessionActive(true);
                        scheduleVoiceResume(120, 10);
                      }
                    }}
                    disabled={!voiceSupported || isTyping}
                  >
                    {voiceSessionActive || listening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />} {voiceSessionActive || listening ? "Stop Voice" : "Start Voice"}
                  </Button>
                  <Button
                    type="button"
                    className="min-h-12 min-w-44 rounded-full border border-white/20 bg-white/10 text-cyan-100 hover:bg-white/15"
                    onClick={() => {
                      voiceSessionActiveRef.current = false;
                      setVoiceSessionActive(false);
                      stopListening();
                      stopVoiceOutput();
                      setVoiceMode(false);
                    }}
                  >
                    <PhoneOff className="w-4 h-4" /> Exit Voice Mode
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <>
            {showSidebar && <div className="fixed inset-0 bg-slate-900/25 z-30 md:hidden" onClick={() => setShowSidebar(false)} />}

            <div className="grid md:grid-cols-[320px_minmax(0,1fr)] gap-3 sm:gap-4 h-[calc(100%-5.2rem)] sm:h-[calc(100%-5.6rem)]">
          <aside
            className={`${showSidebar ? "block" : "hidden"} md:block fixed md:static inset-y-0 left-0 z-40 md:z-auto w-[88vw] max-w-[336px] md:w-auto rounded-r-[20px] md:rounded-[18px] border border-[#b7d1e1] bg-[#f5fbff] md:bg-[#f6fbff]/96 shadow-[0_20px_40px_-28px_rgba(15,74,109,0.65)] p-3 h-full overflow-hidden`}
          >
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-cyan-800" />
                <p className="text-sm font-semibold text-slate-900">Session Ledger</p>
              </div>
              <Button variant="hero" size="sm" className="h-8 px-2.5" onClick={startNewChat}>
                <Plus className="w-3.5 h-3.5 mr-1" /> New
              </Button>
            </div>
            <p className="text-xs text-slate-600 mb-3">Search, resume, and filter mission conversations.</p>
            <div className="flex items-center gap-2 mb-3">
              <div className="relative flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />
                <input
                  value={sessionSearch}
                  onChange={(e) => setSessionSearch(e.target.value)}
                  placeholder="Search conversations"
                  className="h-9 w-full rounded-lg border border-[#c2d8e8] bg-white pl-8 pr-2 text-xs text-slate-900 placeholder:text-slate-500 outline-none transition-all duration-200 focus:border-cyan-300 focus:ring-2 focus:ring-cyan-200/50"
                />
              </div>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[#c2d8e8] bg-white text-slate-600 transition-all duration-200 hover:-translate-y-0.5 hover:text-cyan-800"
                title="Filter"
              >
                <SlidersHorizontal className="h-4 w-4" />
              </button>
            </div>
            <div className="space-y-3 overflow-y-auto h-[calc(100%-8.5rem)] pr-1">
              {loadingSessions && <p className="text-xs text-muted-foreground">Loading sessions...</p>}
              {!loadingSessions && filteredSessions.length === 0 && (
                <div className="rounded-xl border border-[#c5d9e8] bg-white p-3 text-xs text-slate-600">No matching history. Start with New Chat.</div>
              )}
              {sessionGroupOrder.map((group) => {
                const items = groupedSessions[group] || [];
                if (!items.length) return null;
                return (
                  <div key={group} className="space-y-2">
                    <p className="px-1 text-[11px] font-semibold uppercase tracking-[0.11em] text-slate-500">{group}</p>
                    {items.map((s) => (
                      <div
                        key={s.id}
                        onClick={() => void openSession(s.id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            void openSession(s.id);
                          }
                        }}
                        role="button"
                        tabIndex={0}
                        className={`group relative w-full rounded-lg border p-2.5 pr-10 text-left transition-all duration-200 ${
                          sessionId === s.id
                            ? "border-cyan-300/70 bg-gradient-to-r from-[#1da1f2]/12 to-[#0ea5e9]/20 shadow-[0_10px_18px_-14px_rgba(2,132,199,0.9)]"
                            : "border-[#c6dbea] bg-white hover:-translate-y-0.5 hover:border-cyan-300/60 hover:bg-cyan-50/40"
                        }`}
                      >
                        <p className={`truncate text-xs font-semibold ${sessionId === s.id ? "text-cyan-900" : "text-slate-800"}`}>{s.title}</p>
                        <p className="mt-1 flex items-center gap-1 text-[11px] text-slate-500">
                          <Clock3 className="w-3 h-3" /> {formatTime(s.lastMessageAt)}
                        </p>
                        <span className="pointer-events-none absolute inset-y-2 left-0 w-0.5 rounded-r bg-cyan-500 opacity-0 transition-opacity group-hover:opacity-70 data-[active=true]:opacity-100" data-active={sessionId === s.id} />
                        <button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            void deleteSession(s.id);
                          }}
                          className="absolute right-2 top-1/2 inline-flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-md border border-transparent bg-white/70 text-slate-400 opacity-0 transition-all duration-200 hover:border-rose-200 hover:text-rose-600 group-hover:opacity-100 focus-visible:opacity-100"
                          title="Delete chat"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </aside>

          <div className="min-h-0 flex flex-col">
            <div className="flex-1 overflow-y-auto space-y-4 mb-3 sm:mb-4 pr-1 rounded-[18px] border border-[#b8d2e3] bg-[#f8fcff]/98 shadow-[0_20px_38px_-30px_rgba(15,74,109,0.72)] p-3 sm:p-5">
              {messages.map((m, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex">
                  {m.role === "user" ? (
                    <div className="ml-auto max-w-[92%] sm:max-w-[84%] rounded-xl border border-cyan-300/30 bg-gradient-to-r from-[#1da1f2] to-[#0ea5e9] px-4 py-3 text-sm text-white shadow-[0_12px_26px_-16px_rgba(2,132,199,0.92)]">
                      {m.content}
                    </div>
                  ) : (
                    <div className="w-full max-w-[98%] rounded-xl border border-[#c4d9e7] bg-white px-4 py-3 shadow-[0_14px_26px_-22px_rgba(15,74,109,0.62)]">
                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                        <BrandLogo size="sm" showWordmark={false} className="shrink-0" />
                        <span className="font-semibold text-slate-800">Marine Intelligence Assistant</span>
                        <span className="h-3 w-px bg-slate-300" />
                        <span>{m.model || "Marine Core v2"}</span>
                        <span className="h-3 w-px bg-slate-300" />
                        <span>Confidence {m.confidence ?? 92}%</span>
                        <span className="h-3 w-px bg-slate-300" />
                        <span>Data Context {m.context || "Regional mesh"}</span>
                      </div>
                      <div className="mt-3">
                        <p className="mb-2 text-[11px] uppercase tracking-[0.12em] text-slate-500">Answer</p>
                        <div className="text-sm text-slate-800">{renderAssistantText(m.content)}</div>
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
              {streamingAssistant && (
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex">
                  <div className="w-full rounded-xl border border-cyan-200/60 bg-gradient-to-r from-cyan-50/90 to-white px-4 py-3">
                    <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-cyan-800">Executing Response Pipeline</div>
                    <div className="grid gap-1.5 sm:grid-cols-3 text-xs text-slate-600">
                      {getExecutionSteps().map((step, idx) => (
                        <div key={step} className="rounded-md border border-cyan-100 bg-white/80 px-2.5 py-1.5">
                          <span className="text-cyan-700 font-semibold mr-1">{idx + 1}.</span>
                          {step}
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 text-sm text-slate-800">{renderAssistantText(streamingAssistant)}</div>
                  </div>
                </motion.div>
              )}
              {isTyping && (
                <p className="text-xs text-slate-500 inline-flex items-center gap-2">
                  <Database className="h-3.5 w-3.5 text-cyan-700" />
                  AI execution in progress...
                </p>
              )}
              <div ref={bottomRef} />
            </div>

            {messages.length <= 1 && !voiceMode && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="flex flex-wrap gap-2 mb-4">
                {suggestedPrompts.map((p) => (
                  <button
                    key={p}
                    onClick={() => sendMessage(p, "text")}
                    className="glass rounded-full px-3.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-primary/5 transition-colors flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3 h-3" /> {p}
                  </button>
                ))}
              </motion.div>
            )}

            {!voiceMode && (
              <div className="rounded-[18px] border border-[#b8d2e3] bg-white/95 shadow-[0_14px_28px_-20px_rgba(15,74,109,0.72)] sticky bottom-0 pb-[env(safe-area-inset-bottom)]">
                <div className="flex flex-wrap items-center gap-2 border-b border-[#d6e6f1] px-3 py-2">
                  {modeOptions.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setInputMode(opt.id)}
                      className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all duration-200 ${
                        inputMode === opt.id
                          ? "bg-cyan-600 text-white shadow-[0_8px_16px_-12px_rgba(8,145,178,0.92)]"
                          : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                  <span className="ml-auto text-[11px] text-slate-500">Structured prompt mode</span>
                </div>
                <div className="grid gap-2 px-3 py-2 sm:grid-cols-3">
                  <input
                    value={contextLocation}
                    onChange={(e) => setContextLocation(e.target.value)}
                    className="h-9 rounded-lg border border-[#c6dbea] bg-white px-2 text-xs text-slate-700 outline-none focus:border-cyan-300"
                    aria-label="Context location"
                    list="context-location-options"
                    placeholder="Any location or lat,lon"
                  />
                  <datalist id="context-location-options">
                    {locationPresets.map((name) => (
                      <option key={name} value={name} />
                    ))}
                  </datalist>
                  <input
                    value={contextSpecies}
                    onChange={(e) => setContextSpecies(e.target.value)}
                    className="h-9 rounded-lg border border-[#c6dbea] bg-white px-2 text-xs text-slate-700 outline-none focus:border-cyan-300"
                    aria-label="Context species"
                    list="context-species-options"
                    placeholder="Any species"
                  />
                  <datalist id="context-species-options">
                    {speciesPresets.map((name) => (
                      <option key={name} value={name} />
                    ))}
                  </datalist>
                  <input
                    value={contextSeason}
                    onChange={(e) => setContextSeason(e.target.value)}
                    className="h-9 rounded-lg border border-[#c6dbea] bg-white px-2 text-xs text-slate-700 outline-none focus:border-cyan-300"
                    aria-label="Context season"
                    list="context-season-options"
                    placeholder="Any season/time"
                  />
                  <datalist id="context-season-options">
                    {seasonPresets.map((name) => (
                      <option key={name} value={name} />
                    ))}
                  </datalist>
                </div>
                <div className="flex items-center gap-2 sm:gap-3 p-2.5">
                  <button
                    type="button"
                    className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-[#c6dbea] text-slate-600 transition-all duration-200 hover:-translate-y-0.5 hover:text-cyan-800"
                    title="Attach data"
                  >
                    <Paperclip className="h-4 w-4" />
                  </button>
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && sendMessage(undefined, "text")}
                    placeholder="Ask, analyze, compare, or forecast with marine context"
                    className="flex-1 bg-transparent px-2 min-h-11 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none"
                  />
                  <Button variant="glass" size="icon" onClick={() => setVoiceMode(true)} className="rounded-lg w-11 h-11 shrink-0" disabled={!voiceSupported || isTyping}>
                    <Mic className="w-4 h-4" />
                  </Button>
                  <Button variant="hero" size="icon" onClick={() => sendMessage(undefined, "text")} className="rounded-lg w-11 h-11 shrink-0" disabled={isTyping}>
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
          </>
        )}
      </div>
    </DashboardLayout>
  );
}
