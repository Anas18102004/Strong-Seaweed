import DashboardLayout from "@/components/DashboardLayout";
import { motion } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import { Send, Waves, Mic, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

type Message = { role: "user" | "assistant"; content: string };

const suggestedPrompts = [
  "What's the best species for Gulf of Mannar?",
  "When should I harvest Kappaphycus?",
  "Is Lakshadweep suitable for Gracilaria?",
  "What are the risks during monsoon season?",
];

const responses: Record<string, string> = {
  default: "Based on our ecological models, I can help you with species selection, site analysis, and cultivation timing. What specific region or species are you interested in?",
  "gulf": "Gulf of Mannar is one of India's most productive seaweed cultivation zones with a 92% overall suitability score. Kappaphycus alvarezii thrives here with optimal temperatures (28–30°C) and salinity (32–35 ppt). The best cultivation window is October through March.",
  "harvest": "For Kappaphycus alvarezii in optimal conditions, the typical harvest cycle is 45–60 days. Monitor thallus length — harvest when branches reach 15–20 cm. Avoid harvesting during spring tides for better quality retention.",
  "lakshadweep": "Lakshadweep shows strong potential with an 80% suitability score. Gracilaria edulis can be cultivated in the lagoon areas with controlled depth. Water clarity and consistent salinity make it a promising expansion zone.",
  "monsoon": "During monsoon season (Jul–Sep), major risks include: heavy wave action damaging rafts, reduced salinity from rainfall, increased turbidity reducing photosynthesis, and cyclone damage. I recommend pausing active cultivation and focusing on infrastructure maintenance.",
};

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hello! I'm the BlueWeave AI Assistant. I can help you with seaweed cultivation analysis, species selection, and site intelligence. What would you like to know?" },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const getResponse = (text: string): string => {
    const lower = text.toLowerCase();
    if (lower.includes("gulf") || lower.includes("mannar")) return responses.gulf;
    if (lower.includes("harvest") || lower.includes("kappaphycus")) return responses.harvest;
    if (lower.includes("lakshadweep")) return responses.lakshadweep;
    if (lower.includes("monsoon") || lower.includes("risk")) return responses.monsoon;
    return responses.default;
  };

  const sendMessage = (text?: string) => {
    const msg = text || input;
    if (!msg.trim()) return;
    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setInput("");
    setIsTyping(true);
    setTimeout(() => {
      setMessages(prev => [...prev, { role: "assistant", content: getResponse(msg) }]);
      setIsTyping(false);
    }, 1000);
  };

  return (
    <DashboardLayout>
      <div className="max-w-3xl mx-auto flex flex-col h-[calc(100vh-7rem)]">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-4">
          <h1 className="text-2xl font-bold text-foreground mb-1">AI Assistant</h1>
          <p className="text-muted-foreground text-sm">Ask anything about seaweed cultivation in India</p>
        </motion.div>

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1">
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`flex items-start gap-2.5 max-w-[85%] ${m.role === "user" ? "flex-row-reverse" : ""}`}>
                {m.role === "assistant" && (
                  <div className="w-8 h-8 rounded-xl gradient-primary flex items-center justify-center shrink-0">
                    <Waves className="w-4 h-4 text-primary-foreground" />
                  </div>
                )}
                <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "gradient-primary text-primary-foreground"
                    : "glass-strong text-foreground"
                }`}>
                  {m.content}
                </div>
              </div>
            </motion.div>
          ))}
          {isTyping && (
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl gradient-primary flex items-center justify-center">
                <Waves className="w-4 h-4 text-primary-foreground" />
              </div>
              <div className="glass-strong rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Suggested prompts */}
        {messages.length <= 1 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="flex flex-wrap gap-2 mb-4">
            {suggestedPrompts.map(p => (
              <button
                key={p}
                onClick={() => sendMessage(p)}
                className="glass rounded-full px-3.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-primary/5 transition-colors flex items-center gap-1.5"
              >
                <Sparkles className="w-3 h-3" /> {p}
              </button>
            ))}
          </motion.div>
        )}

        {/* Input */}
        <div className="glass-strong rounded-2xl flex items-center gap-3 p-2">
          <button className="w-10 h-10 rounded-xl glass flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors shrink-0">
            <Mic className="w-4 h-4" />
          </button>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && sendMessage()}
            placeholder="Ask BlueWeave AI..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          <Button variant="hero" size="icon" onClick={() => sendMessage()} className="rounded-xl w-10 h-10 shrink-0">
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </DashboardLayout>
  );
}
