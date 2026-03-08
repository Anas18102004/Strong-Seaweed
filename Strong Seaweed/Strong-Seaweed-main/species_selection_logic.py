from __future__ import annotations

from typing import Callable

import numpy as np


def as_float(v):
    try:
        if v is None:
            return None
        out = float(v)
        if np.isfinite(out):
            return out
        return None
    except Exception:
        return None


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def sigmoid(x: float) -> float:
    z = float(np.clip(x, -40.0, 40.0))
    return 1.0 / (1.0 + float(np.exp(-z)))


def species_env_support(species_id: str, feat_vals: dict | None) -> float:
    if not isinstance(feat_vals, dict):
        return 0.5
    sid = str(species_id or "").strip().lower()
    envelopes = {
        "kappaphycus_alvarezii": {"sal": (28.0, 36.0), "wave": (0.2, 1.3), "cur": (0.05, 0.45)},
        "gracilaria_edulis": {"sal": (20.0, 36.0), "wave": (0.1, 1.5), "cur": (0.03, 0.55)},
        "ulva_lactuca": {"sal": (18.0, 38.0), "wave": (0.05, 2.2), "cur": (0.02, 0.90)},
        "sargassum_wightii": {"sal": (24.0, 38.0), "wave": (0.2, 2.0), "cur": (0.05, 0.90)},
    }
    env = envelopes.get(sid)
    if env is None:
        return 0.5

    def score_value(v: float | None, lo: float, hi: float) -> float:
        if v is None:
            return 0.5
        if lo <= v <= hi:
            return 1.0
        span = max(hi - lo, 1e-6)
        slack = 0.25 * span
        if lo - slack <= v <= hi + slack:
            return 0.5
        return 0.0

    sal = as_float(feat_vals.get("so_mean"))
    wave = as_float(feat_vals.get("wave_mean"))
    cur = as_float(feat_vals.get("current_mean"))
    parts = [
        score_value(sal, env["sal"][0], env["sal"][1]),
        score_value(wave, env["wave"][0], env["wave"][1]),
        score_value(cur, env["cur"][0], env["cur"][1]),
    ]
    return float(sum(parts) / max(len(parts), 1))


def species_rank_score(
    species_row: dict,
    env_support: float,
    rank_weight_prob: float,
    rank_weight_margin: float,
    rank_weight_env: float,
) -> float:
    prob_pct = as_float(species_row.get("probabilityPercent"))
    if prob_pct is None:
        return -1.0
    margin_pct = as_float(species_row.get("marginToThresholdPercent"))
    prob_term = clamp01(prob_pct / 100.0)
    margin_term = sigmoid((margin_pct or 0.0) / 8.0)
    env_term = clamp01(env_support)
    score = (
        rank_weight_prob * prob_term
        + rank_weight_margin * margin_term
        + rank_weight_env * env_term
    )
    return float(round(score, 6))


def sort_by_rank(rows: list[dict], default_prior_support: float = 0.5) -> list[dict]:
    return sorted(
        rows,
        key=lambda x: (
            float(x.get("rankScore") or -1.0),
            float(x.get("probabilityPercent") or -1.0),
            float(x.get("marginToThresholdPercent") or -9999.0),
            float(x.get("priorSupportScore") or default_prior_support),
        ),
        reverse=True,
    )


def select_best_species(
    species_rows: list[dict],
    top_candidates: list[dict],
    feat_vals: dict | None,
    sort_by_rank_fn: Callable[[list[dict]], list[dict]],
) -> tuple[dict | None, str]:
    del feat_vals  # Signature compatibility for serving layer.
    eligible = [
        s
        for s in species_rows
        if s["ready"]
        and s["probabilityPercent"] is not None
        and str(s.get("reason", "")).endswith("_positive")
    ]
    if eligible:
        return sort_by_rank_fn(eligible)[0], "ranked_eligible_positive"
    if top_candidates:
        return top_candidates[0], "ranked_top_candidate"
    return None, "none"


def apply_tie_guardrail(
    best: dict | None,
    top_candidates: list[dict],
    selection_diagnostics: dict,
    warnings_out: list[str],
    tie_gap_percent: float,
    tie_force_pilot_only: bool,
    default_prior_support: float,
) -> tuple[dict | None, bool]:
    if len(top_candidates) < 2:
        return best, False

    top1 = top_candidates[0]
    top2 = top_candidates[1]
    top1_prob = float(top1.get("probabilityPercent", 0.0) or 0.0)
    top2_prob = float(top2.get("probabilityPercent", 0.0) or 0.0)
    gap = top1_prob - top2_prob
    tie_detected = gap <= tie_gap_percent
    selection_diagnostics["tieDetected"] = bool(tie_detected)
    selection_diagnostics["tieGapPercent"] = round(gap, 4)
    if not tie_detected:
        return best, False

    tie_rows = sort_by_rank([top1, top2], default_prior_support=default_prior_support)
    tie_winner = tie_rows[0]
    tie_candidates = [str(top1.get("speciesId") or ""), str(top2.get("speciesId") or "")]
    selection_diagnostics["tieCandidates"] = tie_candidates
    selection_diagnostics["tieResolved"] = str(tie_winner.get("speciesId") or "") != tie_candidates[0]
    warnings_out.append("best_species_tie_detected")
    warnings_out.append(f"best_species_tie_candidates={','.join(tie_candidates)}")

    best_after = best
    if best_after is None or str(best_after.get("speciesId") or "") == tie_candidates[0]:
        best_after = tie_winner
        selection_diagnostics["selectionReason"] = "tie_rank_resolved"
        selection_diagnostics["rankScore"] = float(tie_winner.get("rankScore") or 0.0)

    both_recommended = (
        str(top1.get("actionability") or "") == "recommended"
        and str(top2.get("actionability") or "") == "recommended"
    )
    forced = False
    if (
        best_after is not None
        and str(best_after.get("speciesId")) != "insufficient_data"
        and tie_force_pilot_only
        and both_recommended
        and str(best_after.get("actionability", "")) == "recommended"
    ):
        best_after = dict(best_after)
        best_after["actionability"] = "test_pilot_only"
        best_after["reason"] = "tie_forced_pilot_only"
        warnings_out.append("best_species_tie_forced_pilot_only")
        selection_diagnostics["selectionReason"] = "tie_forced_pilot_only"
        forced = True

    return best_after, forced
