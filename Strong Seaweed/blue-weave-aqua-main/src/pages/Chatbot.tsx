import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import { Send, Waves, Sparkles, History, Mic, MicOff, Volume2, VolumeX, Radio, PhoneOff, Plus, Clock3, Menu, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

type Message = { role: "user" | "assistant"; content: string };

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

export default function Chatbot() {
  const welcomeMessage: Message = {
    role: "assistant",
    content: "Hi. I am your Seaweed AI helper. Ask a practical question and I will answer in plain language.",
  };
  const [messages, setMessages] = useState<Message[]>([
    welcomeMessage,
  ]);
  const [input, setInput] = useState("");
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
  const [sttLocale, setSttLocale] = useState("en-IN");

  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { token } = useAuth();

  useEffect(() => {
    const browserLang = (navigator.language || "").toLowerCase();
    if (browserLang.startsWith("en-in") || browserLang.startsWith("hi-in")) {
      setSttLocale("en-IN");
      return;
    }
    if (browserLang.startsWith("en-us")) {
      setSttLocale("en-US");
      return;
    }
    if (browserLang.startsWith("en-gb")) {
      setSttLocale("en-GB");
    }
  }, []);

  useEffect(() => {
    if (voiceMode) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping, voiceMode]);

  const pickFemaleVoice = () => {
    if (!("speechSynthesis" in window)) return null;
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;
    const byName = voices.find((v) => /(female|woman|samantha|aria|ava|lisa|susan|zira)/i.test(v.name));
    if (byName) return byName;
    const byLang = voices.find((v) => /^en[-_]/i.test(v.lang));
    return byLang || voices[0] || null;
  };

  const speak = (text: string) => {
    if (!voiceEnabled || !("speechSynthesis" in window)) return;
    const clean = String(text || "").replace(/\s+/g, " ").trim();
    if (!clean) return;

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.lang = "en-US";
      utterance.rate = 1.02;
      utterance.pitch = 1.07;
      const female = pickFemaleVoice();
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

  const stopListening = () => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    try {
      recognition.stop();
    } catch {
      // no-op
    }
  };

  const startListening = async () => {
    const recognition = recognitionRef.current;
    if (!recognition || isTyping || listening) return;
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
      setMessages(list.length ? list : [{ role: "assistant", content: "This session is empty. Ask your first question." }]);
    } catch {
      setMessages([{ role: "assistant", content: "Could not load this session." }]);
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
    const msg = text || input;
    if (!msg.trim()) return;

    setMessages((prev) => [...prev, { role: "user", content: msg }]);
    setInput("");
    setLiveTranscript("");
    setVoiceError("");
    setIsTyping(true);
    setStreamingAssistant("");

    try {
      if (mode === "voice") {
        const res = await api.voiceRespond(msg, token || undefined, sessionId, sttLocale, "female");
        if (res.sessionId) {
          setSessionId(res.sessionId);
          setIsDraftNewChat(false);
        }
        setMessages((prev) => [...prev, { role: "assistant", content: res.answer }]);

        const played = playAudioBase64(res.audioBase64, res.audioMime);
        if (!played) speak(res.ttsText || res.answer);
      } else {
        try {
          let built = "";
          const res = await api.chatStream(msg, token || undefined, sessionId, (chunk) => {
            built += chunk;
            setStreamingAssistant(built);
          });
          if (res.sessionId) {
            setSessionId(res.sessionId);
            setIsDraftNewChat(false);
          }
          setMessages((prev) => [...prev, { role: "assistant", content: (res.answer || built).trim() }]);
          setStreamingAssistant("");
        } catch {
          setStreamingAssistant("");
          const res = await api.chat(msg, token || undefined, sessionId);
          if (res.sessionId) {
            setSessionId(res.sessionId);
            setIsDraftNewChat(false);
          }
          setMessages((prev) => [...prev, { role: "assistant", content: res.answer }]);
        }
      }

      await loadSessions();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Chat request failed.";
      if (mode === "voice") {
        setVoiceError(message);
      }
      setMessages((prev) => [...prev, { role: "assistant", content: message }]);
    } finally {
      setIsTyping(false);
      if (options.autoResumeVoice && voiceMode && voiceSupported && voiceSessionActive) {
        setTimeout(() => startListening(), 250);
      }
    }
  };

  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setVoiceSupported(false);
      return;
    }
    setVoiceSupported(true);
    const recognition = new SR();
    recognition.lang = sttLocale;
    recognition.interimResults = false;
    recognition.continuous = false;
    try {
      (recognition as BrowserSpeechRecognition & { maxAlternatives?: number }).maxAlternatives = 3;
    } catch {
      // no-op for browsers that don't support this flag
    }

    recognition.onstart = () => setListening(true);
    recognition.onend = () => {
      setListening(false);
      if (voiceMode && voiceSessionActive && voiceSupported && !isTyping) {
        setTimeout(() => {
          void startListening();
        }, 220);
      }
    };
    recognition.onerror = (event: SpeechRecognitionErrorEventLite) => {
      setListening(false);
      const code = String(event?.error || "").toLowerCase();
      if (code === "not-allowed" || code === "service-not-allowed") {
        setVoiceError("Microphone permission denied. Please allow mic access in browser settings.");
        return;
      }
      if (code === "no-speech") {
        setVoiceError("No speech detected. Please speak clearly and try again.");
        return;
      }
      if (code === "network") {
        setVoiceError("Speech recognition network error. Check internet and retry.");
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
      setVoiceSessionActive(false);
      stopListening();
      stopVoiceOutput();
      setLiveTranscript("");
      setVoiceError("");
      return;
    }
    // Keep explicit user control: start recognition only on Start Voice click.
  }, [voiceMode]);

  useEffect(() => {
    if (!voiceEnabled) {
      stopVoiceOutput();
    }
  }, [voiceEnabled]);

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-1 sm:px-0 h-[calc(100dvh-8.6rem)] sm:h-[calc(100dvh-7rem)] relative pb-2">
        <div className="pointer-events-none absolute -top-12 -left-10 h-60 w-60 rounded-full bg-cyan-300/25 blur-3xl" />
        <div className="pointer-events-none absolute top-24 right-[-2rem] h-72 w-72 rounded-full bg-blue-400/25 blur-3xl" />

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-3 sm:mb-4 rounded-[28px] sm:rounded-[30px] border border-white/55 bg-gradient-to-br from-cyan-100/75 via-white/65 to-blue-100/65 p-4 sm:p-6 backdrop-blur-2xl shadow-[0_22px_56px_-28px_rgba(13,72,110,0.52)]"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-[0.14em] text-cyan-700 mb-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                BlueWeave Chat
              </p>
              <h1 className="text-xl sm:text-3xl font-semibold text-slate-900 mb-1">Conversation Studio</h1>
              <p className="text-slate-600 text-sm sm:text-base">Unified chat workspace with voice mode, history, and live assistant streaming.</p>
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
                <Radio className="w-4 h-4" /> {voiceMode ? "Voice Mode On" : "Voice Mode"}
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
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="voice-shell rounded-[30px] p-3 sm:p-4 mb-4"
          >
            <div className="voice-mode-panel rounded-[28px] p-6 sm:p-8 text-center relative overflow-hidden">
              <div className="voice-grid-glow" />
              <div className="voice-noise" />

              <div className="relative z-10">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-6">
                  <div className="voice-chip">
                    <span className={`voice-chip-dot ${listening ? "voice-chip-dot-live" : ""}`} />
                    BlueWeave Voice Assistant
                  </div>
                  <div className="voice-chip text-cyan-100/80">
                    Voice: Female
                  </div>
                  <label className="voice-chip text-cyan-100/90">
                    STT:
                    <select
                      value={sttLocale}
                      onChange={(e) => setSttLocale(e.target.value)}
                      className="ml-2 bg-transparent outline-none"
                      aria-label="Speech recognition locale"
                    >
                      <option value="en-IN">English (India)</option>
                      <option value="en-US">English (US)</option>
                      <option value="en-GB">English (UK)</option>
                      <option value="hi-IN">Hindi (India)</option>
                    </select>
                  </label>
                </div>

                <div className={`voice-orb ${listening ? "voice-orb-live" : ""} ${isTyping ? "voice-orb-thinking" : ""}`}>
                  <div className="voice-orb-ring" />
                  <div className="voice-orb-ring voice-orb-ring-2" />
                  <div className="voice-orb-ring voice-orb-ring-3" />
                  <div className="voice-orb-core">
                    <div className="voice-monogram" role="img" aria-label="BlueWeave Voice Monogram">
                      <span className="voice-monogram-text">BW</span>
                      <span className="voice-monogram-wave">~~~</span>
                      <span className="voice-monogram-glint" />
                    </div>
                  </div>
                  <div className="voice-bars" aria-hidden="true">
                    {Array.from({ length: 7 }).map((_, i) => (
                      <span
                        key={i}
                        className="voice-bar"
                        style={{ animationDelay: `${i * 0.12}s` }}
                      />
                    ))}
                  </div>
                </div>

                <p className="mt-8 text-sm sm:text-base text-cyan-50 font-medium">
                  {isTyping
                    ? "Thinking and preparing your voice response..."
                    : listening
                      ? "Listening live. Speak naturally."
                      : voiceSessionActive
                        ? "Voice standby. Listening will auto-resume."
                      : voiceSupported
                        ? "Tap Start Voice to begin."
                        : "Voice input is not supported in this browser."}
                </p>

                <div className="voice-transcript mt-3">
                  <p className="text-xs text-cyan-100/90 min-h-5">{liveTranscript || "Waiting for your voice..."}</p>
                  {voiceError ? <p className="text-xs text-red-200 mt-2">{voiceError}</p> : null}
                </div>

                <div className="mt-6 flex flex-wrap justify-center gap-3">
                  <Button
                    type="button"
                    variant={listening ? "hero-outline" : "hero"}
                    className="min-h-11 min-w-40"
                    onClick={() => {
                      if (voiceSessionActive || listening) {
                        setVoiceSessionActive(false);
                        stopListening();
                      } else {
                        setVoiceSessionActive(true);
                        void startListening();
                      }
                    }}
                    disabled={!voiceSupported || isTyping}
                  >
                    {voiceSessionActive || listening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />} {voiceSessionActive || listening ? "Stop Voice" : "Start Voice"}
                  </Button>
                  <Button
                    type="button"
                    variant="glass"
                    className="min-h-11 min-w-40"
                    onClick={() => {
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
        ) : null}
        {showSidebar && <div className="fixed inset-0 bg-slate-900/25 z-30 md:hidden" onClick={() => setShowSidebar(false)} />}

        <div className="grid md:grid-cols-[320px_minmax(0,1fr)] gap-3 sm:gap-4 h-[calc(100%-5.2rem)] sm:h-[calc(100%-5.6rem)]">
          <aside
            className={`${showSidebar ? "block" : "hidden"} md:block fixed md:static inset-y-0 left-0 z-40 md:z-auto w-[88vw] max-w-[336px] md:w-auto rounded-r-[24px] md:rounded-[28px] border border-white/55 bg-white/82 md:bg-white/55 backdrop-blur-xl shadow-[0_20px_42px_-24px_rgba(15,74,109,0.55)] p-3 sm:p-4 h-full overflow-hidden`}
          >
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-primary" />
                <p className="text-sm font-semibold text-slate-900">Chats</p>
              </div>
              <Button variant="hero" size="sm" className="h-8 px-2.5" onClick={startNewChat}>
                <Plus className="w-3.5 h-3.5 mr-1" /> New
              </Button>
            </div>
            <p className="text-xs text-slate-600 mb-3">Start a new chat or reopen any previous conversation.</p>
            <div className="space-y-2 overflow-y-auto h-[calc(100%-4.5rem)] pr-1">
              {loadingSessions && <p className="text-xs text-muted-foreground">Loading sessions...</p>}
              {!loadingSessions && sessions.length === 0 && (
                <div className="rounded-2xl border border-white/70 bg-white/70 p-3 text-xs text-slate-600">No chat history yet. Start with New Chat.</div>
              )}
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={`w-full rounded-2xl p-2 border transition ${
                    sessionId === s.id
                      ? "gradient-primary text-primary-foreground border-cyan-300/30 shadow-[0_12px_24px_-16px_rgba(2,132,199,0.9)]"
                      : "border-white/75 bg-white/78 hover:bg-white/90"
                  }`}
                >
                  <button onClick={() => openSession(s.id)} className="w-full text-left px-1 py-1">
                    <p className={`text-xs font-medium truncate ${sessionId === s.id ? "text-primary-foreground" : "text-foreground"}`}>{s.title}</p>
                    <p className={`text-[11px] mt-1 flex items-center gap-1 ${sessionId === s.id ? "text-primary-foreground/90" : "text-muted-foreground"}`}>
                      <Clock3 className="w-3 h-3" /> {formatTime(s.lastMessageAt)}
                    </p>
                  </button>
                  <div className="flex justify-end">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteSession(s.id);
                      }}
                      className={`text-[11px] px-2 py-1 rounded-lg flex items-center gap-1 ${
                        sessionId === s.id ? "bg-white/20 text-primary-foreground" : "bg-red-50 text-red-600"
                      }`}
                      title="Delete chat"
                    >
                      <Trash2 className="w-3 h-3" /> Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </aside>

          <div className="min-h-0 flex flex-col">
            <div className="flex-1 overflow-y-auto space-y-4 mb-3 sm:mb-4 pr-1 rounded-[26px] sm:rounded-[30px] border border-white/55 bg-white/55 backdrop-blur-xl shadow-[0_20px_42px_-24px_rgba(15,74,109,0.55)] p-3 sm:p-5">
              {messages.map((m, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`flex items-start gap-2.5 max-w-[92%] sm:max-w-[85%] ${m.role === "user" ? "flex-row-reverse" : ""}`}>
                    {m.role === "assistant" && (
                      <div className="w-8 h-8 rounded-xl gradient-primary flex items-center justify-center shrink-0">
                        <Waves className="w-4 h-4 text-primary-foreground" />
                      </div>
                    )}
                    <div className={`whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed border ${
                      m.role === "user"
                        ? "gradient-primary text-primary-foreground border-cyan-300/30 shadow-[0_10px_24px_-16px_rgba(2,132,199,0.9)]"
                        : "bg-white/90 text-slate-800 border-white/80"
                    }`}>
                      {m.role === "assistant" ? renderAssistantText(m.content) : m.content}
                    </div>
                  </div>
                </motion.div>
              ))}
              {streamingAssistant && (
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex justify-start">
                  <div className="flex items-start gap-2.5 max-w-[92%] sm:max-w-[85%]">
                    <div className="w-8 h-8 rounded-xl gradient-primary flex items-center justify-center shrink-0">
                      <Waves className="w-4 h-4 text-primary-foreground" />
                    </div>
                    <div className="whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed bg-white/90 text-slate-800 border border-white/80">
                      {renderAssistantText(streamingAssistant)}
                    </div>
                  </div>
                </motion.div>
              )}
              {isTyping && <p className="text-xs text-muted-foreground">AI is thinking...</p>}
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
              <div className="rounded-[22px] sm:rounded-[24px] border border-white/60 bg-white/65 backdrop-blur-xl shadow-[0_16px_36px_-22px_rgba(15,74,109,0.5)] flex items-center gap-2 sm:gap-3 p-2.5 sticky bottom-0 pb-[env(safe-area-inset-bottom)]">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && sendMessage(undefined, "text")}
                  placeholder="Ask a question about cultivation, risk, yield, weather, or market decisions"
                  className="flex-1 bg-transparent px-2 min-h-11 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none"
                />
                <Button variant="glass" size="icon" onClick={() => setVoiceMode(true)} className="rounded-xl w-11 h-11 shrink-0" disabled={!voiceSupported || isTyping}>
                  <Mic className="w-4 h-4" />
                </Button>
                <Button variant="hero" size="icon" onClick={() => sendMessage(undefined, "text")} className="rounded-xl w-11 h-11 shrink-0" disabled={isTyping}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
