# Deployment Status: v1.1r_plus35provisional (Provisional)

## Active Promotion
- Promoted as active artifacts from `releases/v1.1r_plus35provisional/`:
  - `models/realtime/xgboost_realtime_model.json`
  - `models/realtime/xgboost_realtime_ensemble.pkl`
  - `models/realtime/xgboost_realtime_calibrator.pkl`
  - `models/realtime/xgboost_realtime_features.json`
  - `artifacts/reports/xgboost_realtime_report.json`

Backup of previous active artifacts:
- `releases/backups/20260227_153412/`

## Data Basis
- Training dataset: `data/tabular/training_dataset_v1_1_plus11web_plus35provisional.csv`
- Positives: 92
- Negatives: 259

## Core Metrics (Training Report)
- Spatial AUC: 0.7648 (+/- 0.1823)
- OOF calibrated AP: 0.5445
- Recommended threshold: 0.7142857

## Benchmark Runs
- Full-grid score snapshot:
  - `outputs/realtime_ranked_candidates_v1_1r_plus35provisional.csv`
  - `releases/v1.1r_plus35provisional/snapshots/realtime_ranked_candidates_v1_1r_plus35provisional_baseline.csv`
- Hard-50 test:
  - `artifacts/reports/v1.1r_plus35provisional_hard50_eval.json`
- Leakage-aware non-overlap subset:
  - `artifacts/reports/v1.1r_plus35provisional_hard50_eval_nonoverlap.json`

## Governance Note
- This release is **provisional** because the +35 positive additions were not strict-verified (`is_verified=False`, `qa_status=pending` in source candidate workflow).
- Keep this release for experimentation and decision support; do not label as final strict production until external non-overlap positives are verified and re-ingested in strict mode.

