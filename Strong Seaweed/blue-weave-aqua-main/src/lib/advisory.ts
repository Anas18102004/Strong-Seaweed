export type ParsedAdvisory = {
  recommendationLevel: string | null;
  why: string | null;
  topRisks: string[];
  fieldChecks: string[];
  actionPlan: string[];
  notes: string[];
  hasContent: boolean;
};

const ASK_HINT_PATTERNS = [
  /^[-*]?\s*ask\s*["']?e["']?\s*$/i,
  /^[-*]?\s*ask\s*["']?expand["']?.*$/i,
  /^[-*]?\s*ask\s+for\s+expand.*$/i,
];

const SECTION_PATTERNS = {
  level: /^[-*]?\s*(?:cultivation\s+)?recommendation level\s*:\s*(.*)$/i,
  why: /^[-*]?\s*why\s*:\s*(.*)$/i,
  topRisks: /^[-*]?\s*top risks\s*:?\s*(.*)$/i,
  fieldChecks: /^[-*]?\s*field checks(?: before farming)?\s*:?\s*(.*)$/i,
  actionPlan: /^[-*]?\s*next\s*7(?:[\s-]*day(?:s)?(?:\s*action\s*plan)?)?\s*:?\s*(.*)$/i,
};

function stripListPrefix(line: string): string {
  return String(line || "")
    .replace(/^(\d+[\).\s-]+|[-*]\s+)/, "")
    .trim();
}

function appendText(base: string | null, extra: string): string {
  const clean = extra.trim();
  if (!clean) return base || "";
  return base ? `${base} ${clean}` : clean;
}

export function sanitizeAdvisoryText(text?: string | null): string {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => {
      const t = String(line || "").trim();
      if (!t) return true;
      return !ASK_HINT_PATTERNS.some((pattern) => pattern.test(t));
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function parseFallbackAdvisory(text?: string | null): ParsedAdvisory {
  const clean = sanitizeAdvisoryText(text);
  const parsed: Omit<ParsedAdvisory, "hasContent"> = {
    recommendationLevel: null,
    why: null,
    topRisks: [],
    fieldChecks: [],
    actionPlan: [],
    notes: [],
  };

  if (!clean) {
    return { ...parsed, hasContent: false };
  }

  const lines = clean
    .split("\n")
    .map((line) => String(line || "").trim())
    .filter(Boolean);

  let activeList: "topRisks" | "fieldChecks" | "actionPlan" | null = null;
  let activeText: "recommendationLevel" | "why" | null = null;

  for (const line of lines) {
    let matched = false;

    const levelMatch = line.match(SECTION_PATTERNS.level);
    if (levelMatch) {
      parsed.recommendationLevel = appendText(null, levelMatch[1] || "");
      activeList = null;
      activeText = "recommendationLevel";
      matched = true;
    }

    if (!matched) {
      const whyMatch = line.match(SECTION_PATTERNS.why);
      if (whyMatch) {
        parsed.why = appendText(null, whyMatch[1] || "");
        activeList = null;
        activeText = "why";
        matched = true;
      }
    }

    if (!matched) {
      const topMatch = line.match(SECTION_PATTERNS.topRisks);
      if (topMatch) {
        activeList = "topRisks";
        activeText = null;
        const seed = stripListPrefix(topMatch[1] || "");
        if (seed) parsed.topRisks.push(seed);
        matched = true;
      }
    }

    if (!matched) {
      const checksMatch = line.match(SECTION_PATTERNS.fieldChecks);
      if (checksMatch) {
        activeList = "fieldChecks";
        activeText = null;
        const seed = stripListPrefix(checksMatch[1] || "");
        if (seed) parsed.fieldChecks.push(seed);
        matched = true;
      }
    }

    if (!matched) {
      const planMatch = line.match(SECTION_PATTERNS.actionPlan);
      if (planMatch) {
        activeList = "actionPlan";
        activeText = null;
        const seed = stripListPrefix(planMatch[1] || "");
        if (seed) parsed.actionPlan.push(seed);
        matched = true;
      }
    }

    if (matched) continue;

    const listItem = stripListPrefix(line);
    const hasBulletPrefix = /^(\d+[\).\s-]+|[-*]\s+)/.test(line);

    if (activeList) {
      if (hasBulletPrefix || parsed[activeList].length === 0) {
        if (listItem) parsed[activeList].push(listItem);
      } else {
        const lastIndex = parsed[activeList].length - 1;
        parsed[activeList][lastIndex] = appendText(parsed[activeList][lastIndex], line);
      }
      continue;
    }

    if (activeText) {
      parsed[activeText] = appendText(parsed[activeText], line);
      continue;
    }

    if (listItem) parsed.notes.push(listItem);
  }

  const hasContent = Boolean(
    parsed.recommendationLevel ||
      parsed.why ||
      parsed.topRisks.length ||
      parsed.fieldChecks.length ||
      parsed.actionPlan.length ||
      parsed.notes.length,
  );

  return { ...parsed, hasContent };
}
