import { motion } from "framer-motion";

const species = [
  { name: "Kappaphycus alvarezii", score: 87, color: "from-ocean-500 to-ocean-400" },
  { name: "Gracilaria edulis", score: 81, color: "from-ocean-400 to-accent" },
  { name: "Ulva lactuca", score: 74, color: "from-accent to-ocean-300" },
  { name: "Sargassum wightii", score: 62, color: "from-ocean-300 to-ocean-200" },
];

export function FloatingSpeciesCards() {
  return (
    <div className="relative w-full max-w-sm mx-auto">
      {species.map((s, i) => (
        <motion.div
          key={s.name}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 + i * 0.15, duration: 0.6 }}
          className={`glass-strong rounded-3xl p-5 mb-4 ${
            i === 0 ? "animate-float" :
            i === 1 ? "animate-float-delay-1" :
            i === 2 ? "animate-float-delay-2" :
            "animate-float-delay-3"
          }`}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-ocean-muted">{s.name}</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-20 h-2 rounded-full bg-muted overflow-hidden">
                <motion.div
                  className={`h-full rounded-full bg-gradient-to-r ${s.color}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${s.score}%` }}
                  transition={{ delay: 0.6 + i * 0.15, duration: 0.8, ease: "easeOut" }}
                />
              </div>
              <span className="text-lg font-bold gradient-text">{s.score}%</span>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
