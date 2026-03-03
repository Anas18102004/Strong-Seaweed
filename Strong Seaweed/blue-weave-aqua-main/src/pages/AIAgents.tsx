import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { Brain, TrendingUp, MapPin, DollarSign, ShieldCheck, Sparkles, ArrowRight, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useNavigate } from "react-router-dom";

type AgentId = "cultivation" | "risk" | "yield" | "site" | "market" | "copilot";

type AgentCard = {
  id: AgentId;
  name: string;
  icon: typeof Brain;
  short: string;
  outcome: string;
  defaultPrompt: string;
};

const taskCatalog: AgentCard[] = [
  {
    id: "cultivation",
    name: "Farm Setup Plan",
    icon: Brain,
    short: "Get a step-by-step starter plan for cultivation.",
    outcome: "Structured setup checklist",
    defaultPrompt: "Create a practical seaweed farm setup plan for a beginner.",
  },
  {
    id: "risk",
    name: "Risk Mitigation",
    icon: ShieldCheck,
    short: "Identify key risks and mitigation actions.",
    outcome: "Risk matrix + action plan",
    defaultPrompt: "Analyze current farm risks and give mitigation actions.",
  },
  {
    id: "yield",
    name: "Yield Improvement",
    icon: TrendingUp,
    short: "Find bottlenecks and optimize output.",
    outcome: "Yield optimization steps",
    defaultPrompt: "Suggest actions to improve seaweed yield in the next 30 days.",
  },
  {
    id: "site",
    name: "Site Selection",
    icon: MapPin,
    short: "Evaluate location suitability and tradeoffs.",
    outcome: "Site suitability recommendation",
    defaultPrompt: "Evaluate site suitability and recommend the best location strategy.",
  },
  {
    id: "market",
    name: "Market Timing",
    icon: DollarSign,
    short: "Plan harvest and sell windows.",
    outcome: "Sales timing strategy",
    defaultPrompt: "Recommend best harvest and selling timing based on market behavior.",
  },
  {
    id: "copilot",
    name: "General Strategy",
    icon: Sparkles,
    short: "Use when your problem spans multiple areas.",
    outcome: "Cross-functional operational plan",
    defaultPrompt: "Give a practical end-to-end strategy for current farm operations.",
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

function renderResponse(raw: string) {
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
        if (/^\d+\.\s+/.test(line)) {
          const clean = line.replace(/^\d+\.\s+/, "");
          const n = line.match(/^(\d+\.)/)?.[1] || "";
          return (
            <div key={idx} className="flex items-start gap-2 text-slate-800">
              <span className="mt-0.5 text-cyan-700 font-semibold min-w-[1.1rem]">{n}</span>
              <p className="leading-relaxed">{renderInline(clean)}</p>
            </div>
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
  const [selectedTask, setSelectedTask] = useState<AgentId>("cultivation");
  const [location, setLocation] = useState("");
  const [objective, setObjective] = useState("");
  const [constraints, setConstraints] = useState("");
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const { token } = useAuth();
  const navigate = useNavigate();

  const active = useMemo(
    () => taskCatalog.find((a) => a.id === selectedTask) || taskCatalog[0],
    [selectedTask],
  );

  const runTask = async () => {
    const parts = [
      objective.trim() || active.defaultPrompt,
      location.trim() ? `Location: ${location.trim()}` : "",
      constraints.trim() ? `Constraints: ${constraints.trim()}` : "",
      "Return a concise, actionable task plan with clear next steps.",
    ].filter(Boolean);
    const question = parts.join("\n");

    setLoading(true);
    setOutput("");
    setLastQuestion(question);
    try {
      const res = await api.askAgent(active.id, question, token || undefined);
      setOutput(res.answer || "No response received.");
    } catch (err) {
      setOutput(err instanceof Error ? err.message : "Assistant failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-1 sm:px-0 space-y-5 sm:space-y-6 pb-4">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-[30px] border border-white/55 bg-gradient-to-br from-cyan-100/80 via-white/70 to-blue-100/70 p-5 sm:p-7 backdrop-blur-2xl shadow-[0_24px_58px_-28px_rgba(13,72,110,0.52)]"
        >
          <p className="inline-flex items-center gap-2 rounded-full border border-white/65 bg-white/60 px-3 py-1 text-[11px] uppercase tracking-[0.14em] text-cyan-700">
            <Sparkles className="h-3.5 w-3.5" />
            Task Assistant Studio
          </p>
          <h1 className="mt-2 text-xl sm:text-3xl font-semibold text-slate-900">Run Structured Tasks, Then Continue in Chat</h1>
          <p className="mt-2 text-sm sm:text-base text-slate-600 max-w-3xl">
            This page is for targeted outcomes. Choose one task, run it with context, get one clean action plan, then move to Chat for back-and-forth discussion.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-4 sm:gap-5">
          <div className="lg:col-span-2">
            <div className="rounded-[26px] border border-white/60 bg-white/60 backdrop-blur-xl shadow-[0_18px_42px_-22px_rgba(16,76,116,0.5)] p-3 sm:p-4 space-y-3">
              <p className="px-1 text-xs uppercase tracking-[0.14em] text-slate-500">Choose Task</p>
              {taskCatalog.map((task) => {
                const activeCard = task.id === selectedTask;
                return (
                  <button
                    key={task.id}
                    onClick={() => setSelectedTask(task.id)}
                    className={`w-full text-left rounded-2xl p-4 border transition-all ${
                      activeCard
                        ? "border-cyan-300/70 bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-[0_14px_30px_-20px_rgba(2,132,199,0.85)]"
                        : "border-white/75 bg-white/80 hover:bg-white/92 text-slate-900"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${activeCard ? "bg-white/20 text-white" : "bg-cyan-50 text-cyan-600"}`}>
                        <task.icon className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <p className={`font-semibold text-sm ${activeCard ? "text-white" : "text-slate-900"}`}>{task.name}</p>
                        <p className={`mt-0.5 text-xs ${activeCard ? "text-white/85" : "text-slate-600"}`}>{task.short}</p>
                        <p className={`mt-2 text-[11px] ${activeCard ? "text-white/90" : "text-cyan-700"}`}>Outcome: {task.outcome}</p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="lg:col-span-3 space-y-4">
            <div className="rounded-[26px] border border-white/60 bg-white/60 backdrop-blur-xl shadow-[0_18px_42px_-24px_rgba(15,74,109,0.52)] p-4 sm:p-5">
              <p className="text-sm font-semibold text-slate-900">{active.name}</p>
              <p className="text-xs text-slate-600 mt-1">Give only the context needed for this task.</p>
              <div className="grid sm:grid-cols-2 gap-3 mt-4">
                <input
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Location (optional)"
                  className="w-full h-11 rounded-2xl border border-white/70 bg-white/80 px-4 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-300/65"
                />
                <input
                  value={objective}
                  onChange={(e) => setObjective(e.target.value)}
                  placeholder="Objective (optional)"
                  className="w-full h-11 rounded-2xl border border-white/70 bg-white/80 px-4 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-300/65"
                />
              </div>
              <textarea
                value={constraints}
                onChange={(e) => setConstraints(e.target.value)}
                placeholder="Constraints (budget, timeline, manpower, regulations)"
                className="w-full mt-3 min-h-24 rounded-2xl border border-white/70 bg-white/80 px-4 py-3 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-300/65"
              />
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={runTask} variant="hero" className="min-h-11" disabled={loading}>
                  {loading ? "Running task..." : "Run Task Assistant"} <ArrowRight className="w-4 h-4" />
                </Button>
                <Button
                  onClick={() => navigate("/chat", { state: { prefill: lastQuestion || active.defaultPrompt } })}
                  variant="hero-outline"
                  className="min-h-11"
                >
                  Continue in Chat <MessageSquare className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="rounded-[26px] border border-white/60 bg-white/60 backdrop-blur-xl shadow-[0_18px_42px_-24px_rgba(15,74,109,0.52)] overflow-hidden">
              <div className="px-5 py-3 border-b border-white/45 bg-white/45">
                <p className="text-sm font-semibold text-slate-900">Task Output</p>
              </div>
              <div className="p-5 min-h-[18rem] bg-gradient-to-b from-white/55 to-white/30">
                {!output && !loading && (
                  <p className="text-sm text-slate-600">Run a task to generate a structured action plan here.</p>
                )}
                {loading && <p className="text-sm text-slate-600">Assistant is preparing your plan...</p>}
                {output && <div className="text-sm">{renderResponse(output)}</div>}
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
