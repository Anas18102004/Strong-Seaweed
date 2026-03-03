import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import {
  Brain,
  TrendingUp,
  MapPin,
  DollarSign,
  ShieldCheck,
  Sparkles,
  ArrowRight,
  MessageSquare,
  CircleDot,
  Cpu,
  Activity,
  Clock3,
  WandSparkles,
} from "lucide-react";
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
        <strong key={idx} className="font-semibold text-[#0F2E47]">
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
            <p key={idx} className="font-semibold text-[#0F2E47]">
              {renderInline(clean)}
            </p>
          );
        }
        if (/^[-*]\s+/.test(line) || /^\u2022\s+/.test(line)) {
          const clean = line.replace(/^[-*]\s+/, "").replace(/^\u2022\s+/, "");
          return (
            <div key={idx} className="flex items-start gap-2 text-slate-700">
              <span className="mt-1 text-cyan-700">-</span>
              <p className="leading-relaxed">{renderInline(clean)}</p>
            </div>
          );
        }
        if (/^\d+\.\s+/.test(line)) {
          const clean = line.replace(/^\d+\.\s+/, "");
          const n = line.match(/^(\d+\.)/)?.[1] || "";
          return (
            <div key={idx} className="flex items-start gap-2 text-slate-700">
              <span className="mt-0.5 text-cyan-700 font-semibold min-w-[1.1rem]">{n}</span>
              <p className="leading-relaxed">{renderInline(clean)}</p>
            </div>
          );
        }
        return (
          <p key={idx} className="leading-relaxed text-slate-700">
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
          className="ai-command-hero ai-hero-flow relative overflow-hidden rounded-[24px] border border-cyan-100/20 p-4 sm:p-6 backdrop-blur-2xl"
        >
          <div className="ai-neural-grid ai-neural-mesh" />
          <p className="relative z-10 inline-flex items-center gap-2 rounded-full border border-cyan-100/30 bg-cyan-300/10 px-3 py-1 text-[11px] uppercase tracking-[0.16em] text-cyan-100">
            <Sparkles className="h-3.5 w-3.5" />
            AI Command Studio
          </p>
          <h1 className="relative z-10 mt-3 text-2xl sm:text-[50px] leading-[1.04] font-semibold text-white tracking-[-0.02em]">
            Run Structured <span className="ocean-title-highlight">Intelligence</span>
          </h1>
          <p className="relative z-10 mt-2 text-sm sm:text-base text-[#D8EEFF] max-w-3xl">
            Design targeted execution plans with contextual constraints, then continue deep iterative collaboration in live chat.
          </p>
          <div className="relative z-10 mt-4 flex flex-wrap items-center gap-2.5">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-100/20 bg-white/[0.07] px-3 py-1.5 text-xs text-[#DDF4FF]">
              <Cpu className="h-3.5 w-3.5 text-cyan-200" />
              Model v2.4.1
            </div>
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-100">
              <Activity className="h-3.5 w-3.5 ocean-breathe-dot" />
              Status: Live
            </div>
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-100/20 bg-white/[0.07] px-3 py-1.5 text-xs text-[#CFE9FF]">
              <Clock3 className="h-3.5 w-3.5 text-cyan-200" />
              Last updated: just now
            </div>
          </div>
          <div className="relative z-10 mt-3 inline-flex items-center gap-2 text-xs text-emerald-100">
            <CircleDot className="h-3.5 w-3.5 ocean-breathe-dot" />
            Live AI orchestration online
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-4 sm:gap-6">
          <div className="lg:col-span-2">
            <div className="ocean-glass-card rounded-2xl p-4 space-y-3 border border-[#c2d8e8] bg-white/95">
              <p className="px-1 text-xs uppercase tracking-[0.14em] text-slate-600">Choose Task</p>
              {taskCatalog.map((task) => {
                const activeCard = task.id === selectedTask;
                return (
                  <button
                    key={task.id}
                    onClick={() => setSelectedTask(task.id)}
                    className={`task-select-tile group w-full text-left rounded-xl p-4 border transition-all duration-200 ${
                      activeCard
                        ? "border-[#66ceff]/60 bg-[linear-gradient(135deg,#0f3a59_0%,#145981_55%,#1b6f99_100%)] text-white shadow-[0_20px_34px_-16px_rgba(9,84,133,0.78)]"
                        : "border-[#c6dbea] bg-white text-slate-800 hover:-translate-y-0.5 hover:border-cyan-300/60 hover:shadow-[0_14px_26px_-20px_rgba(15,52,88,0.45)]"
                    }`}
                  >
                    <span className="task-shine" />
                    {activeCard && <span className="task-active-line" />}
                    <div className="flex items-start gap-3">
                      <div className={`relative w-10 h-10 rounded-xl flex items-center justify-center ${activeCard ? "bg-white/20 text-white" : "bg-cyan-100 text-cyan-700"}`}>
                        <span className="absolute inset-0 rounded-xl bg-cyan-300/25 blur-md opacity-0 group-hover:opacity-100 transition-opacity" />
                        <task.icon className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <p className={`font-semibold text-sm ${activeCard ? "text-white" : "text-slate-900"}`}>{task.name}</p>
                        <p className={`mt-0.5 text-xs ${activeCard ? "text-white/95" : "text-slate-600"}`}>{task.short}</p>
                        <p className={`mt-2 text-[11px] ${activeCard ? "text-cyan-100" : "text-cyan-700"}`}>Outcome: {task.outcome}</p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="lg:col-span-3 space-y-4">
            <div className="context-intelligence-panel rounded-2xl p-4 sm:p-6">
              <p className="text-sm font-semibold text-white">{active.name}</p>
              <p className="text-xs text-[#b7dbf4] mt-1">Provide focused context for higher-quality task plans.</p>
              <div className="grid sm:grid-cols-2 gap-3 mt-4">
                <label className="relative block ai-input-shell rounded-xl">
                  <input
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    placeholder=" "
                    className="peer h-11 w-full rounded-xl border border-white/25 bg-[#0f304b]/70 px-4 text-sm text-white outline-none transition-all duration-200 focus:border-cyan-200/45 focus:ring-2 focus:ring-cyan-200/35"
                  />
                  <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-xs text-[#9fc6e2] transition-all peer-focus:top-2 peer-focus:text-[10px] peer-focus:text-cyan-100 peer-[:not(:placeholder-shown)]:top-2 peer-[:not(:placeholder-shown)]:text-[10px]">Location (optional)</span>
                </label>
                <label className="relative block ai-input-shell rounded-xl">
                  <input
                    value={objective}
                    onChange={(e) => setObjective(e.target.value)}
                    placeholder=" "
                    className="peer h-11 w-full rounded-xl border border-white/25 bg-[#0f304b]/70 px-4 text-sm text-white outline-none transition-all duration-200 focus:border-cyan-200/45 focus:ring-2 focus:ring-cyan-200/35"
                  />
                  <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-xs text-[#9fc6e2] transition-all peer-focus:top-2 peer-focus:text-[10px] peer-focus:text-cyan-100 peer-[:not(:placeholder-shown)]:top-2 peer-[:not(:placeholder-shown)]:text-[10px]">Objective (optional)</span>
                </label>
              </div>
              <label className="relative block mt-4 ai-input-shell rounded-xl">
                <textarea
                  value={constraints}
                  onChange={(e) => setConstraints(e.target.value)}
                  placeholder=" "
                  className="peer min-h-24 w-full rounded-xl border border-white/25 bg-gradient-to-b from-[#10324d]/75 to-[#0d2940]/78 px-4 py-3 text-sm text-white outline-none transition-all duration-200 focus:border-cyan-200/45 focus:ring-2 focus:ring-cyan-200/35"
                />
                <span className="pointer-events-none absolute left-4 top-3 text-xs text-[#9fc6e2] transition-all peer-focus:top-2 peer-focus:text-[10px] peer-focus:text-cyan-100 peer-[:not(:placeholder-shown)]:top-2 peer-[:not(:placeholder-shown)]:text-[10px]">Constraints (budget, timeline, manpower, regulations)</span>
              </label>
              <div className="mt-5 flex flex-wrap gap-2">
                <Button
                  onClick={runTask}
                  className="ocean-shine-btn ai-cta-btn min-h-12 rounded-full bg-gradient-to-r from-[#1DA1F2] to-[#0EA5E9] px-5 text-white hover:opacity-95"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="ai-loading-bars">
                        <span />
                        <span />
                        <span />
                      </span>
                      Running task...
                    </>
                  ) : (
                    <>
                      Run Task Assistant <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </Button>
                <Button
                  onClick={() => navigate("/chat", { state: { prefill: lastQuestion || active.defaultPrompt } })}
                  className="min-h-12 rounded-full border border-white/30 bg-white/[0.14] text-white transition-all duration-200 hover:bg-white/[0.2] hover:-translate-y-0.5"
                >
                  Continue in Chat <MessageSquare className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="ocean-glass-card rounded-2xl overflow-hidden border border-[#c2d8e8] bg-white/96">
              <div className="px-5 py-3 border-b border-[#d6e8f3] bg-[#f5fbff]">
                <p className="text-sm font-semibold text-[#0F2E47]">Task Output</p>
              </div>
              <div className="p-5 min-h-[18rem] bg-gradient-to-b from-[#fbfeff] to-[#f5fbff]">
                {!output && !loading && (
                  <div className="ai-output-empty relative overflow-hidden rounded-xl border border-dashed border-cyan-300/50 bg-white p-4">
                    <div className="ai-output-shimmer" />
                    <div className="relative z-10 flex items-center gap-2 text-cyan-800">
                      <WandSparkles className="h-4 w-4" />
                      <p className="text-sm font-medium">AI generation preview</p>
                    </div>
                    <p className="relative z-10 mt-2 text-sm text-slate-600">Run a task to generate a structured action plan here.</p>
                    <div className="relative z-10 mt-3 rounded-lg border border-cyan-100/40 bg-[#f4fbff] p-3 text-xs text-slate-700">
                      <p className="font-semibold text-cyan-800">Example Output</p>
                      <p className="mt-1">1. Site readiness checklist</p>
                      <p>2. 30-day execution milestones</p>
                      <p>3. Risk mitigation actions</p>
                    </div>
                  </div>
                )}
                {loading && <p className="text-sm text-slate-600">Assistant is preparing your plan...</p>}
                {output && <div className="text-sm text-slate-800">{renderResponse(output)}</div>}
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
