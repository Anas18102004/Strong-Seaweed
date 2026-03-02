import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useState } from "react";
import { Brain, Waves, TrendingUp, MapPin, DollarSign, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";

const agents = [
  { id: "cultivation", name: "Cultivation Advisor", icon: Brain, desc: "Species selection, planting schedules, and growth optimization guidance", color: "from-ocean-500 to-ocean-400" },
  { id: "risk", name: "Environmental Risk Agent", icon: Waves, desc: "Real-time environmental risk assessment and early warning alerts", color: "from-ocean-400 to-accent" },
  { id: "yield", name: "Yield Optimization Agent", icon: TrendingUp, desc: "Harvest timing, productivity forecasting, and yield maximization", color: "from-accent to-ocean-300" },
  { id: "site", name: "Site Expansion Agent", icon: MapPin, desc: "New site identification and expansion opportunity analysis", color: "from-ocean-300 to-ocean-200" },
  { id: "market", name: "Market Intelligence Agent", icon: DollarSign, desc: "Pricing trends, demand forecasts, and market opportunity insights", color: "from-ocean-500 to-accent" },
];

const sampleResponses: Record<string, string[]> = {
  cultivation: [
    "Based on your Gulf of Mannar site profile, I recommend Kappaphycus alvarezii for the upcoming post-monsoon season. Water temperatures are optimal at 28.5°C.",
    "The current conditions suggest a 52-day harvest cycle. Monitor salinity levels weekly — any drop below 30 ppt may require intervention.",
  ],
  risk: [
    "No immediate cyclone risk detected for the next 14 days. However, monsoon onset may arrive early this year based on current wind patterns.",
  ],
  yield: [
    "Current growth rate projections indicate 15-18 tonnes/hectare for your Kappaphycus crop at Gulf of Mannar, assuming stable environmental conditions.",
  ],
  site: [
    "Kollam coast shows emerging potential with a 76% suitability score. I recommend a pilot site assessment for Gracilaria edulis cultivation.",
  ],
  market: [
    "Carrageenan demand is projected to grow 8% YoY in 2026. Current farmgate prices for dry Kappaphycus are ₹35-42/kg in Tamil Nadu.",
  ],
};

export default function AIAgents() {
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<{ role: "user" | "agent"; text: string }[]>([]);

  const openChat = (id: string) => {
    setActiveAgent(id);
    setMessages([{ role: "agent", text: sampleResponses[id]?.[0] || "How can I assist you?" }]);
  };

  const sendMessage = () => {
    if (!chatInput.trim() || !activeAgent) return;
    setMessages(prev => [...prev, { role: "user", text: chatInput }]);
    setChatInput("");
    setTimeout(() => {
      const responses = sampleResponses[activeAgent] || [];
      const resp = responses[Math.floor(Math.random() * responses.length)] || "I'll analyze that and get back to you with recommendations.";
      setMessages(prev => [...prev, { role: "agent", text: resp }]);
    }, 800);
  };

  return (
    <DashboardLayout>
      <div className="max-w-5xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-2xl font-bold text-foreground mb-1">AI Agents</h1>
          <p className="text-muted-foreground text-sm">Specialized intelligence agents for cultivation decision-making</p>
        </motion.div>

        {/* Agent cards */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((a, i) => (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              className="glass-strong rounded-3xl p-6 hover:-translate-y-1 transition-all duration-300 cursor-pointer group"
              onClick={() => openChat(a.id)}
            >
              <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${a.color} flex items-center justify-center mb-4`}>
                <a.icon className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="font-bold text-foreground mb-1">{a.name}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{a.desc}</p>
              <p className="text-xs font-medium text-primary mt-3 group-hover:underline">Start conversation →</p>
            </motion.div>
          ))}
        </div>

        {/* Chat panel */}
        {activeAgent && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-strong rounded-3xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-border/40">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-xl gradient-primary flex items-center justify-center">
                  {(() => { const Icon = agents.find(a => a.id === activeAgent)?.icon || Brain; return <Icon className="w-4 h-4 text-primary-foreground" />; })()}
                </div>
                <span className="font-semibold text-foreground">{agents.find(a => a.id === activeAgent)?.name}</span>
              </div>
              <button onClick={() => setActiveAgent(null)} className="text-muted-foreground hover:text-foreground"><X className="w-5 h-5" /></button>
            </div>

            <div className="h-72 overflow-y-auto p-6 space-y-4">
              {messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                    m.role === "user"
                      ? "gradient-primary text-primary-foreground"
                      : "glass text-foreground"
                  }`}>
                    {m.text}
                  </div>
                </div>
              ))}
            </div>

            <div className="px-6 py-4 border-t border-border/40 flex gap-3">
              <input
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendMessage()}
                placeholder="Ask the agent..."
                className="flex-1 h-11 rounded-2xl glass px-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
              <Button variant="hero" size="icon" onClick={sendMessage} className="rounded-2xl w-11 h-11">
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </motion.div>
        )}
      </div>
    </DashboardLayout>
  );
}
