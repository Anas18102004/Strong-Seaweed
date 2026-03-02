# Deployment Status: Kappaphycus + Species Orchestrator (Current Runtime)

## Runtime State
- Kappaphycus dedicated release in runtime: **kappa_india_gulf_v2_prod_ready_v4_with_v2labels**
- Multi-species release in runtime: **multi_species_cop_india_v5b_rich_relaxed_soft_hn**
- Runtime API server: `serve_species_api.py`

## Active Artifacts
- `releases/kappa_india_gulf_v2_prod_ready_v4_with_v2labels/models/xgboost_realtime_model_kappa_india_gulf_v2_prod_ready_v4_with_v2labels.json`
- `releases/kappa_india_gulf_v2_prod_ready_v4_with_v2labels/models/xgboost_realtime_ensemble_kappa_india_gulf_v2_prod_ready_v4_with_v2labels.pkl`
- `releases/kappa_india_gulf_v2_prod_ready_v4_with_v2labels/models/xgboost_realtime_calibrator_kappa_india_gulf_v2_prod_ready_v4_with_v2labels.pkl`
- `releases/multi_species_cop_india_v5b_rich_relaxed_soft_hn/models/xgb_gracilaria_spp_multi_species_cop_india_v5b_rich_relaxed_soft_hn.pkl`
- `releases/multi_species_cop_india_v5b_rich_relaxed_soft_hn/models/xgb_ulva_spp_multi_species_cop_india_v5b_rich_relaxed_soft_hn.pkl`
- `releases/multi_species_cop_india_v5b_rich_relaxed_soft_hn/models/xgb_sargassum_spp_multi_species_cop_india_v5b_rich_relaxed_soft_hn.pkl`

## Kappaphycus Snapshot (Current)
- Training dataset: `data/tabular/training_dataset_kappa_india_gulf_v2_hardmerge4_augmented_plus_v2labels_2026_02_28.csv`
- Samples: 450
- Positives: 124
- Negatives: 326
- Spatial AUC: 0.7824 (+/- 0.0610)
- OOF calibrated AP: 0.5772
- Recommended threshold: 0.7857142686843872

## Multi-Species Snapshot (Current)
- Data source pipeline:
  - `data/tabular/multispecies_occurrences_multispecies_india_v2_prod.csv`
  - `data/netcdf/india_physics_2025w01.nc`
  - `data/netcdf/india_waves_2025w01.nc`
- Copernicus dataset rows:
  - `gracilaria_spp`: rows=1265, pos=65, neg=1200
  - `ulva_spp`: rows=1285, pos=85, neg=1200
  - `sargassum_spp`: rows=1264, pos=64, neg=1200
  - `kappaphycus_alvarezii`: rows=1205, pos=5, neg=1200 (skipped in multi-species training; dedicated kappa model remains active)
- Spatial CV performance:
  - `gracilaria_spp`: spatial AUC=0.8107, spatial AP=0.3647
  - `ulva_spp`: spatial AUC=0.7749, spatial AP=0.3501
  - `sargassum_spp`: spatial AUC=0.7780, spatial AP=0.3553
- Notes: these remain genus-proxy ecological models with coverage guards; they are not yet hard-farm validated production labels.

## Neighbor Soft-Positive Experiment (Implemented)
- Script added: `build_neighbor_soft_positives.py`
- Training updated to use `label_weight` in XGBoost (`train_multispecies_models.py`).
- Best-performing expansion config on current India grid:
  - distance bands: `4 km -> 0.85`, `8 km -> 0.65`, `12 km -> 0.45`
  - ecological gate: feature quantile box + similarity threshold
  - negative protection buffer: `14 km`
- Generated datasets suffix: `_softpos_v3`
- Report: `artifacts/reports/neighbor_soft_positive_report_softpos_v3.json`

### Soft-Positive Counts Added
- `gracilaria_spp`: +2
- `sargassum_spp`: +2
- `ulva_spp`: +5
- `kappaphycus_alvarezii`: +0

### Before vs After (Weighted Trainer)
- Baseline release: `multi_species_cop_india_v2_prod_wtfix`
- Soft-positive release: `multi_species_cop_india_v2_softpos_v3`
- `gracilaria_spp`: AUC `0.8107 -> 0.8042`, AP `0.3647 -> 0.3629`
- `sargassum_spp`: AUC `0.7780 -> 0.8014`, AP `0.3553 -> 0.4598`
- `ulva_spp`: AUC `0.7749 -> 0.7689`, AP `0.3501 -> 0.3434`

### Decision
- Keep default runtime at `multi_species_cop_india_v2_prod` until strict geo-holdout gate passes.
- Keep `multi_species_cop_india_v2_softpos_v4_grid` as experimental (promising but not gate-passed).

## Neighbor Soft-Positive v4 (Grid-Aware, Promoted)
- Input datasets: `training_dataset_*_multispecies_cop_india_v2_prod.csv`
- Output datasets: `training_dataset_*_multispecies_cop_india_v2_prod_softpos_v4_grid.csv`
- Key change: soft positives are expanded from full Copernicus grid near hard positives, not only sampled negatives.
- Soft positives added:
  - `gracilaria_spp`: +115 (111 from grid candidates)
  - `sargassum_spp`: +106 (102 from grid candidates)
  - `ulva_spp`: +155 (145 from grid candidates)
- Release: `multi_species_cop_india_v2_softpos_v4_grid`
- Spatial CV performance:
  - `gracilaria_spp`: AUC=0.8425, AP=0.6714
  - `sargassum_spp`: AUC=0.8811, AP=0.6769
  - `ulva_spp`: AUC=0.8312, AP=0.6455

## Strict Geographic Holdout Gate (Latest)
- Gate script: `strict_geographic_holdout_gate_multispecies.py`
- Reports:
  - `artifacts/reports/strict_geo_holdout_gate_multispecies_softpos_v4_grid.json`
  - `artifacts/reports/strict_geo_holdout_gate_multispecies_softpos_v4_grid_nolonlat.json`
  - baseline compare: `artifacts/reports/strict_geo_holdout_gate_multispecies_v2prod_baseline_nolonlat.json`
- Decision: **FAIL** for both baseline and soft-positive releases under strict geo holdout.
- Key finding: model overfits train partition (`train_auc~1.0`) and drops strongly on independent spatial test partitions.

## Rich-Feature Recovery (v5b) - Gate PASS
- Rich dataset builder added:
  - `build_multispecies_copernicus_rich_point_datasets.py`
- Hard-negative + soft-positive pipeline:
  - `build_neighbor_soft_positives.py` (`softpos_v7_relaxed_rich`)
  - `build_hard_negatives_multispecies.py` (`hn_v3_rich`)
- Training release:
  - `multi_species_cop_india_v5b_rich_relaxed_soft_hn`
- Strict gate report (PASS):
  - `artifacts/reports/strict_geo_holdout_gate_multispecies_v5b_rich_relaxed_soft_hn_gate.json`
- Gate thresholds used:
  - min test AUC `0.80`
  - min test AP `0.35`
  - max Brier `0.20`
  - max train-test AUC gap `0.15`

### PASS Metrics (Geo Holdout)
- `gracilaria_spp`: test AUC `0.9186`, AP `0.7328`
- `sargassum_spp`: test AUC `0.8424`, AP `0.5013`
- `ulva_spp`: test AUC `0.9256`, AP `0.8049`

## Fix Attempt v3 (Hard Negatives + Strong Regularization)
- Added scripts:
  - `build_hard_negatives_multispecies.py`
  - `strict_geographic_holdout_gate_multispecies.py`
- Updated:
  - `train_multispecies_models.py` (no lon/lat features, weighted training, stronger regularization)
- Dataset variant tested:
  - `training_dataset_*_multispecies_cop_india_v2_prod_softpos_v5_strict_hn_v1.csv`
- Release tested:
  - `multi_species_cop_india_v3_softpos_v5_hn_v1`
- Gate report:
  - `artifacts/reports/strict_geo_holdout_gate_multispecies_v3_softpos_v5_hn_v1_gate.json`
- Decision: **FAIL**
  - gracilaria test AUC/AP: `0.6114 / 0.1025`
  - sargassum test AUC/AP: `0.5590 / 0.0818`
  - ulva test AUC/AP: `0.6854 / 0.1531`

### Practical Conclusion
- Current non-kappa multispecies stack is still not production-gate ready under strict geo holdout.
- Main blocker is not just model tuning; it is **feature/label quality ceiling** from current 5-feature Copernicus point setup.

## API Runtime Command
```powershell
cd d:\Strong Seaweed\Strong-Seaweed-main
$env:PYTHONPATH='d:\Strong Seaweed\Strong-Seaweed-main\.deps_xgb'
python serve_species_api.py
```

## Legacy Scoring Command (Kappaphycus-only batch ranking)
```powershell
python score_realtime_production.py --release_tag kappa_india_gulf_v2_prod_ready_v4_with_v2labels --input data/tabular/master_feature_matrix_kappa_india_gulf_v2_hardmerge4_augmented.csv --output outputs/realtime_ranked_candidates_kappa_india_gulf_v2_prod_ready_v4_with_v2labels.csv
```
