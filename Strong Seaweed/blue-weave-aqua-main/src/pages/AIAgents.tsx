import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import {
  Brain,
  TrendingUp,
  MapPin,
  DollarSign,
  Send,
  ShieldCheck,
  Cpu,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

type AgentId = "cultivation" | "risk" | "yield" | "site" | "market" | "copilot";

type AgentCard = {
  id: AgentId;
  name: string;
  icon: typeof Brain;
  short: string;
  helperText: string;
  prompts: string[];
};

const agentCatalog: AgentCard[] = [
  {
    id: "cultivation",
    name: "Start a Seaweed Farm",
    icon: Brain,
    short: "Beginner setup, species selection, and first steps.",
    helperText: "Best if you are starting from scratch or confused about where to begin.",
    prompts: [
      "What species should I start with in Gulf of Mannar?",
      "Give a 30-day beginner cultivation plan.",
      "What setup mistakes should I avoid in month one?",
    ],
  },
  {
    id: "risk",
    name: "Reduce Farm Risk",
    icon: ShieldCheck,
    short: "Storm, disease, and operational risk guidance.",
    helperText: "Use before deployment, monsoon periods, or uncertain weather windows.",
    prompts: [
      "Top risks for this month in Tamil Nadu coast?",
      "Give a cyclone preparedness checklist.",
      "When should I pause field operations?",
    ],
  },
  {
    id: "yield",
    name: "Improve Yield",
    icon: TrendingUp,
    short: "Increase growth, quality, and harvest consistency.",
    helperText: "Use when your output is low or inconsistent.",
    prompts: [
      "How can I improve yield in next 2 weeks?",
      "Give fouling-control routine for better growth.",
      "How should I optimize harvest timing?",
    ],
  },
  {
    id: "site",
    name: "Choose Better Location",
    icon: MapPin,
    short: "Compare locations and find where farming is more suitable.",
    helperText: "Use before expansion, investment, or field trials.",
    prompts: [
      "Which nearby coast is best for expansion?",
      "Compare two candidate locations for suitability.",
      "What location constraints matter most?",
    ],
  },
  {
    id: "market",
    name: "Sell at the Right Time",
    icon: DollarSign,
    short: "Harvest and sales timing support.",
    helperText: "Use when deciding when/how much to sell.",
    prompts: [
      "When should I sell dry biomass this month?",
      "How to align harvest with demand windows?",
      "What is the safer sell strategy for small farmers?",
    ],
  },
  {
    id: "copilot",
    name: "General AI Helper",
    icon: Cpu,
    short: "Ask anything and get clear practical next steps.",
    helperText: "Use this if you are not sure which section fits your question.",
    prompts: [
      "I am new. What should I do first this week?",
      "Create a weekly operations plan from risk to market.",
      "Summarize best actions for my current stage.",
    ],
  },
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

function renderAgentText(raw: string) {
  const lines = String(raw || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter((line, idx, arr) => !(line === "" && arr[idx - 1] === ""));

  return (
    <div className="space-y-2">
      {lines.map((line, idx) => {
        // Strip markdown heading markers but keep emphasis.
        if (/^#{1,3}\s+/.test(line)) {
          const clean = line.replace(/^#{1,3}\s+/, "");
          return (
            <p key={idx} className="font-semibold text-slate-900">
              {renderInline(clean)}
            </p>
          );
        }

        // Turn markdown bullets into styled bullets.
        if (/^[-*•]\s+/.test(line)) {
          const clean = line.replace(/^[-*•]\s+/, "");
          return (
            <div key={idx} className="flex items-start gap-2 text-slate-800">
              <span className="mt-1 text-cyan-700">•</span>
              <p className="leading-relaxed">{renderInline(clean)}</p>
            </div>
          );
        }

        // Flatten markdown table rows to readable text.
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

export default function AIAgents() {
  const [selectedAgent, setSelectedAgent] = useState<AgentId>("copilot");
  const [chatInput, setChatInput] = useState("");
  const [locationHint, setLocationHint] = useState("");
  const [messages, setMessages] = useState<{ role: "user" | "agent"; text: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const { token } = useAuth();

  const active = useMemo(
    () => agentCatalog.find((a) => a.id === selectedAgent) || agentCatalog[0],
    [selectedAgent],
  );

  useEffect(() => {
    setMessages([
      {
        role: "agent",
        text: `Hi, I can help with ${active.name.toLowerCase()}. Ask your question in simple language.`,
      },
    ]);
  }, [active.id, active.name]);

  const sendMessage = async (question?: string) => {
    const base = String(question || chatInput).trim();
    const location = String(locationHint || "").trim();
    const q = location ? `${base}\n\nLocation: ${location}` : base;
    if (!q || !selectedAgent) return;
    setMessages((prev) => [...prev, { role: "user", text: base }]);
    setChatInput("");
    setLoading(true);
    try {
      const res = await api.askAgent(selectedAgent, q, token || undefined);
      setMessages((prev) => [...prev, { role: "agent", text: res.answer }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "agent", text: err instanceof Error ? err.message : "Assistant failed." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto px-1 sm:px-0 space-y-5 sm:space-y-6 pb-4 relative">
        <div className="pointer-events-none absolute -top-10 -left-8 h-52 w-52 rounded-full bg-cyan-300/25 blur-3xl" />
        <div className="pointer-events-none absolute top-20 right-0 h-64 w-64 rounded-full bg-blue-400/20 blur-3xl" />

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-[28px] sm:rounded-[30px] border border-white/45 bg-gradient-to-br from-cyan-100/65 via-white/55 to-blue-100/55 p-4 sm:p-7 backdrop-blur-2xl shadow-[0_18px_55px_-25px_rgba(13,72,110,0.45)]"
        >
          <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-cyan-300/30 blur-2xl" />
          <div className="relative">
            <p className="inline-flex items-center gap-2 rounded-full border border-white/60 bg-white/55 px-3 py-1 text-[11px] uppercase tracking-[0.16em] text-cyan-700">
              <Sparkles className="h-3.5 w-3.5" />
              AI Farm Help Center
            </p>
            <h1 className="mt-2.5 text-xl sm:text-3xl font-semibold text-slate-900">Ask Better, Decide Faster</h1>
            <p className="mt-2 max-w-2xl text-sm sm:text-base text-slate-600">
              Select a help type, add your location, and get practical action steps without technical noise.
            </p>
            <div className="mt-4 grid sm:grid-cols-3 gap-2">
              <div className="rounded-2xl border border-white/70 bg-white/55 px-3 py-2.5 text-xs text-slate-700">1. Choose a help type</div>
              <div className="rounded-2xl border border-white/70 bg-white/55 px-3 py-2.5 text-xs text-slate-700">2. Ask in plain language</div>
              <div className="rounded-2xl border border-white/70 bg-white/55 px-3 py-2.5 text-xs text-slate-700">3. Follow the recommended next steps</div>
            </div>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-4 sm:gap-5">
          <div className="lg:col-span-2">
            <div className="rounded-[26px] border border-white/55 bg-white/45 backdrop-blur-xl shadow-[0_12px_40px_-22px_rgba(16,76,116,0.45)] p-3 sm:p-4 space-y-3">
              <p className="px-1 text-xs uppercase tracking-[0.14em] text-slate-500">Choose Help Type</p>
              {agentCatalog.map((a, idx) => {
                const activeCard = a.id === selectedAgent;
                return (
                  <motion.button
                    key={a.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.03 }}
                    onClick={() => setSelectedAgent(a.id)}
                    className={`group w-full text-left rounded-2xl p-4 border transition-all duration-200 ${
                      activeCard
                        ? "border-cyan-300/70 bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-[0_14px_30px_-20px_rgba(2,132,199,0.85)]"
                        : "border-white/65 bg-white/65 hover:bg-white/80 text-slate-900"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
                          activeCard ? "bg-white/20 text-white" : "bg-cyan-50 text-cyan-600 group-hover:bg-cyan-100"
                        }`}
                      >
                        <a.icon className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <p className={`font-semibold text-sm ${activeCard ? "text-white" : "text-slate-900"}`}>{a.name}</p>
                        <p className={`mt-0.5 text-xs ${activeCard ? "text-white/85" : "text-slate-600"}`}>{a.short}</p>
                      </div>
                    </div>
                  </motion.button>
                );
              })}
            </div>
          </div>

          <div className="lg:col-span-3 space-y-4">
            <div className="rounded-[24px] sm:rounded-[26px] border border-white/55 bg-white/45 backdrop-blur-xl shadow-[0_14px_35px_-24px_rgba(15,74,109,0.5)] p-4 sm:p-5">
              <div className="flex items-center gap-2 mb-1">
                <active.icon className="h-5 w-5 text-cyan-600" />
                <h3 className="text-lg font-semibold text-slate-900">{active.name}</h3>
              </div>
              <p className="text-sm text-slate-600">{active.short}</p>

              <div className="grid sm:grid-cols-2 gap-3 mt-4">
                <div className="rounded-2xl border border-white/65 bg-white/65 p-3">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500 mb-1">Best for</p>
                  <p className="text-sm text-slate-800">{active.helperText}</p>
                </div>
                <div className="rounded-2xl border border-white/65 bg-white/65 p-3">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500 mb-1">Tip</p>
                  <p className="text-sm text-slate-800">Include location and current challenge to get sharper advice.</p>
                </div>
              </div>

              <p className="text-xs uppercase tracking-[0.14em] text-slate-500 mt-4 mb-2">Quick prompts</p>
              <div className="flex flex-wrap gap-2">
                {active.prompts.map((p) => (
                  <button
                    key={p}
                    className="rounded-full border border-white/70 bg-white/75 px-3 py-1.5 text-xs text-slate-700 hover:bg-cyan-50 hover:text-cyan-700 transition-colors"
                    onClick={() => sendMessage(p)}
                    disabled={loading}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-[26px] border border-white/55 bg-white/45 backdrop-blur-xl shadow-[0_14px_35px_-24px_rgba(15,74,109,0.5)] overflow-hidden">
              <div className="px-4 sm:px-5 py-3 border-b border-white/45 bg-white/40">
                <p className="text-sm font-semibold text-slate-900">Ask Your Question</p>
              </div>

              <div className="h-[18rem] sm:h-80 overflow-y-auto p-4 sm:p-5 space-y-3 bg-gradient-to-b from-white/40 to-white/20">
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[92%] sm:max-w-[82%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm border ${
                        m.role === "user"
                          ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white border-cyan-300/30"
                          : "bg-white/75 text-slate-800 border-white/75"
                      }`}
                    >
                      {m.role === "agent" ? renderAgentText(m.text) : m.text}
                    </div>
                  </div>
                ))}
                {loading && <p className="text-xs text-slate-500">Assistant is preparing your answer...</p>}
              </div>

              <div className="px-4 sm:px-5 py-3 border-t border-white/45 flex gap-2 sm:gap-3 bg-white/35">
                <div className="flex-1 space-y-2">
                  <input
                    value={locationHint}
                    onChange={(e) => setLocationHint(e.target.value)}
                    placeholder="Optional location (e.g., Navsari, Gujarat)"
                    className="w-full h-10 rounded-2xl border border-white/70 bg-white/80 px-4 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-300/65"
                  />
                  <input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                    placeholder="Type your question..."
                    className="w-full h-11 rounded-2xl border border-white/70 bg-white/80 px-4 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-300/65"
                  />
                </div>
                <Button
                  variant="hero"
                  size="icon"
                  onClick={() => sendMessage()}
                  className="rounded-2xl w-11 h-11 self-end shadow-[0_12px_24px_-16px_rgba(2,132,199,0.85)]"
                  disabled={loading}
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
              <div className="px-4 sm:px-5 pb-4 bg-white/35">
                <p className="text-[11px] text-slate-500 flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5 text-cyan-600" />
                  User-facing guidance only. Internal model/provider details are hidden.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
