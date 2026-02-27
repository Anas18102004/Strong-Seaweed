# Deployment Status: v1.1r_plus11web

## Active Production Artifacts
- `models/realtime/xgboost_realtime_model.json`
- `models/realtime/xgboost_realtime_ensemble.pkl`
- `models/realtime/xgboost_realtime_calibrator.pkl`
- `models/realtime/xgboost_realtime_features.json`
- `artifacts/reports/xgboost_realtime_report.json`

These were promoted from:
- `releases/v1.1r_plus11web/models/*`
- `releases/v1.1r_plus11web/reports/xgboost_realtime_report_v1.1r_plus11web.json`

Backup of previous active artifacts:
- `releases/backups/20260227_150354/`

## Model Snapshot
- Dataset: `training_dataset_v1_1_merged46_plus_hn30_augmented_plus11web.csv`
- Positives: 57
- Negatives: 259
- Spatial AUC: 0.6955 (+/- 0.1529)
- OOF calibrated AP: 0.4322
- Recommended threshold: 0.667

## Baseline Scoring Output
- Ranked candidates: `outputs/realtime_ranked_candidates_v1_1r_plus11web.csv`
- Release baseline snapshot: `releases/v1.1r_plus11web/snapshots/realtime_ranked_candidates_v1_1r_plus11web_baseline.csv`

## Run Command (Release-Specific)
```powershell
python score_realtime_production.py --release_tag v1.1r_plus11web --input data/tabular/master_feature_matrix_v1_1_augmented.csv --output outputs/realtime_ranked_candidates_v1_1r_plus11web.csv
```

