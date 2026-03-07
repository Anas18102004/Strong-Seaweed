# ML + LLM Advisory Roadmap (Farm-Ready)

## Objective
Build a decision-support system that combines calibrated species suitability models with grounded advisory intelligence for farmers.

## Principles
- Model scores remain the primary quantitative signal.
- LLM never overrides model outputs; it explains, contextualizes, and suggests safe next actions.
- Every advisory must expose uncertainty, warnings, and required field validation.
- External context (web and Copernicus) must be provenance-tagged.

## Phase 1 (Now) — Safety + Explainability Baseline
- Guarantee non-null `bestSpecies` with explicit actionability labels (`recommended`, `test_pilot_only`, `not_recommended`, `insufficient_data`).
- Add advisory endpoint that packages:
  - prediction outputs and warnings,
  - location and user inputs,
  - available environment hints.
- Force advisory structure:
  1. recommendation level,
  2. why,
  3. risks,
  4. required field checks,
  5. 7-day action plan.

## Phase 2 — Grounded Context Enrichment
- Add structured Copernicus context extraction pipeline for requested coordinates:
  - `download_copernicus_data.py`
  - `process_copernicus_ocean_features.py`
  - `extract_copernicus_features_points.py`
- Attach derived indicators to advisory context:
  - seasonal variability,
  - anomaly markers,
  - wave/current stress proxies,
  - staleness timestamp.
- Add web context retrieval for policy/regulation/news only when user asks recency-sensitive questions.

## Phase 3 — Trust and Governance
- Add confidence policy engine before rendering advisory:
  - if low confidence or out-of-coverage, block strong recommendations,
  - enforce “pilot-only” or “not recommended”.
- Add trace block to each advisory:
  - model release,
  - decision source,
  - threshold margin,
  - context sources and timestamps.
- Persist advisory quality feedback from users (accepted/rejected + outcome) for iterative calibration.

## Phase 4 — Farm Outcome Optimization
- Build farm outcome dataset (yield/survival/contamination events) linked to site + season + advisory.
- Train post-decision risk models (survival and expected yield bands).
- Add localized ops playbooks (seedling density, rope spacing, harvest windows) by species + season + risk level.

## Immediate KPI Targets
- API reliability: >99% prediction response success.
- Null best-species rate: 0%.
- Advisory actionability consistency: 100% aligned with model policy.
- Hard benchmark guardrails (must pass before model promotion):
  - precision >= 0.80,
  - recall >= 0.80,
  - AUC >= 0.85.

## What Not To Do
- Do not let LLM invent suitability scores.
- Do not treat web snippets as equivalent to calibrated model evidence.
- Do not auto-upgrade models without hard benchmark and readiness gate pass.
