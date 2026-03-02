import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { FloatingSpeciesCards } from "@/components/FloatingSpeciesCards";
import { ArrowRight, Microscope, MapPin, BarChart3, Brain, FileText, Waves } from "lucide-react";
import { useNavigate } from "react-router-dom";

const features = [
  { icon: Microscope, title: "Species Prediction", desc: "ML-powered multi-species suitability scoring" },
  { icon: MapPin, title: "Site Intelligence", desc: "Regional analysis across India's coastline" },
  { icon: BarChart3, title: "Seasonal Forecast", desc: "Monsoon and cyclone risk assessment" },
  { icon: Brain, title: "AI Advisory", desc: "Intelligent cultivation recommendations" },
  { icon: FileText, title: "Export Reports", desc: "PDF reports for stakeholders" },
  { icon: Waves, title: "Environmental Data", desc: "Real-time oceanographic parameters" },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen gradient-bg overflow-hidden relative">
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-ocean-200/30 rounded-full blur-3xl animate-ocean-shift" />
        <div className="absolute top-1/2 -left-40 w-80 h-80 bg-ocean-100/40 rounded-full blur-3xl animate-float-slow" />
        <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-ocean-200/20 rounded-full blur-3xl animate-float-delay-2" />
      </div>

      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-6 lg:px-12 py-5">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl gradient-primary flex items-center justify-center">
            <Waves className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-xl font-bold text-foreground">BlueWeave<span className="gradient-text"> AI</span></span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-muted-foreground">
          <a href="#features" className="hover:text-foreground transition-colors">Features</a>
          <a href="#about" className="hover:text-foreground transition-colors">About</a>
        </div>
        <Button variant="glass" size="sm" onClick={() => navigate("/dashboard")}>
          Launch Platform <ArrowRight className="w-4 h-4" />
        </Button>
      </nav>

      {/* Hero */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 lg:px-12 pt-16 pb-24">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 text-xs font-medium text-muted-foreground mb-6">
              <span className="w-2 h-2 rounded-full gradient-primary animate-glow-pulse" />
              v1.1 Gulf Model — Live
            </div>
            <h1 className="text-5xl lg:text-6xl font-extrabold leading-[1.1] tracking-tight text-foreground mb-6">
              AI-Powered Seaweed<br />
              Cultivation{" "}
              <span className="gradient-text">Intelligence for India</span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-lg mb-8 leading-relaxed">
              Predict. Optimize. Cultivate Smarter — using ecological AI models trained on India's marine ecosystems.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button variant="hero" size="xl" onClick={() => navigate("/predict")}>
                Start Cultivation Analysis <ArrowRight className="w-5 h-5" />
              </Button>
              <Button variant="hero-outline" size="xl" onClick={() => navigate("/dashboard")}>
                Explore Potential Sites
              </Button>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            <FloatingSpeciesCards />
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto px-6 lg:px-12 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-14"
        >
          <h2 className="text-3xl font-bold text-foreground mb-3">Platform Capabilities</h2>
          <p className="text-muted-foreground max-w-md mx-auto">End-to-end intelligence for marine aquaculture decision-making</p>
        </motion.div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
              className="glass-strong rounded-3xl p-6 hover:-translate-y-1 transition-transform duration-300 group cursor-pointer"
            >
              <div className="w-11 h-11 rounded-2xl gradient-primary flex items-center justify-center mb-4 group-hover:glow-sm transition-shadow">
                <f.icon className="w-5 h-5 text-primary-foreground" />
              </div>
              <h3 className="font-semibold text-foreground mb-1">{f.title}</h3>
              <p className="text-sm text-muted-foreground">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border/50 py-8 text-center text-sm text-muted-foreground">
        <p>© 2026 BlueWeave AI — Climate Intelligence Platform</p>
      </footer>
    </div>
  );
}
