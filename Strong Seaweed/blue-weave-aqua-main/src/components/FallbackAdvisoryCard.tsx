import { AlertTriangle, ListChecks } from "lucide-react";
import { parseFallbackAdvisory } from "@/lib/advisory";

type Variant = "light" | "dark";

type Props = {
  text?: string | null;
  variant?: Variant;
  className?: string;
};

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

const tone: Record<
  Variant,
  {
    shell: string;
    title: string;
    sectionShell: string;
    sectionTitle: string;
    body: string;
    icon: string;
    levelChip: string;
  }
> = {
  light: {
    shell: "rounded-2xl border border-cyan-200 bg-cyan-50/70 p-4",
    title: "text-xs font-semibold uppercase tracking-[0.12em] text-cyan-800",
    sectionShell: "rounded-xl border border-cyan-200/70 bg-white/80 p-3",
    sectionTitle: "text-[11px] font-semibold uppercase tracking-[0.1em] text-cyan-700",
    body: "mt-1 text-sm leading-relaxed text-cyan-950",
    icon: "text-cyan-700",
    levelChip: "inline-flex rounded-full border border-cyan-300 bg-cyan-100 px-3 py-1 text-xs font-semibold text-cyan-900",
  },
  dark: {
    shell: "rounded-2xl border border-cyan-200/25 bg-cyan-400/10 p-4",
    title: "text-xs font-semibold uppercase tracking-[0.12em] text-cyan-100",
    sectionShell: "rounded-xl border border-cyan-200/20 bg-black/10 p-3",
    sectionTitle: "text-[11px] font-semibold uppercase tracking-[0.1em] text-cyan-100",
    body: "mt-1 text-sm leading-relaxed text-cyan-50",
    icon: "text-cyan-100",
    levelChip: "inline-flex rounded-full border border-cyan-200/40 bg-cyan-300/15 px-3 py-1 text-xs font-semibold text-cyan-50",
  },
};

function AdvisoryList({
  title,
  items,
  variant,
}: {
  title: string;
  items: string[];
  variant: Variant;
}) {
  if (!items.length) return null;
  const t = tone[variant];
  return (
    <section className={t.sectionShell}>
      <p className={t.sectionTitle}>{title}</p>
      <ol className={cx(t.body, "list-decimal space-y-1 pl-5")}>
        {items.map((item, index) => (
          <li key={`${title}-${index}`}>{item}</li>
        ))}
      </ol>
    </section>
  );
}

export default function FallbackAdvisoryCard({ text, variant = "light", className }: Props) {
  const parsed = parseFallbackAdvisory(text);
  if (!parsed.hasContent) return null;
  const t = tone[variant];

  return (
    <section className={cx(t.shell, className)}>
      <div className="flex items-center gap-2">
        <ListChecks className={cx("h-4 w-4", t.icon)} />
        <p className={t.title}>Fallback Advisory</p>
      </div>

      {parsed.recommendationLevel && (
        <div className="mt-3">
          <span className={t.levelChip}>{parsed.recommendationLevel}</span>
        </div>
      )}

      {parsed.why && (
        <section className={cx("mt-3", t.sectionShell)}>
          <p className={t.sectionTitle}>Why</p>
          <p className={t.body}>{parsed.why}</p>
        </section>
      )}

      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        <AdvisoryList title="Top Risks" items={parsed.topRisks} variant={variant} />
        <AdvisoryList title="Field Checks" items={parsed.fieldChecks} variant={variant} />
        <AdvisoryList title="7-Day Action Plan" items={parsed.actionPlan} variant={variant} />
      </div>

      {parsed.notes.length > 0 && (
        <section className={cx("mt-3", t.sectionShell)}>
          <div className="flex items-center gap-2">
            <AlertTriangle className={cx("h-3.5 w-3.5", t.icon)} />
            <p className={t.sectionTitle}>Additional Notes</p>
          </div>
          <ul className={cx(t.body, "list-disc space-y-1 pl-5")}>
            {parsed.notes.map((line, idx) => (
              <li key={`note-${idx}`}>{line}</li>
            ))}
          </ul>
        </section>
      )}
    </section>
  );
}
