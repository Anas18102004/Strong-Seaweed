# Project Structure

## Keep in Root
- Python scripts and project metadata only.

## Folders
- `data/tabular/`: CSV datasets and plans (`training_dataset.csv`, `master_feature_matrix.csv`, `v1_1_data_plan.csv`, etc.).
- `data/rasters/`: all GeoTIFF inputs/features.
- `data/netcdf/`: all Copernicus netCDF files.
- `data/config/`: supporting JSON config/catalog files.
- `models/realtime/`: deployed realtime model artifacts (`*.pkl`, `*.json` feature/model files).
- `outputs/`: scoring outputs (for example `realtime_ranked_candidates.csv`).
- `docs/`: model cards, QA docs, notes.
- `artifacts/reports/`: metric and run reports (`*.json`, summary CSVs).
- `artifacts/diagnostics/`: SHAP/feature-importance and test diagnostics.
- `artifacts/experiments/`: temporary experiment outputs, seed files, augmented datasets.
- `artifacts/temp/`: temporary/intermediate files.
- `artifacts/legacy_models/`: old model files not used by current runtime.

## Rule
Scripts now resolve structured folders via `project_paths.py` (with root fallback for legacy files).
