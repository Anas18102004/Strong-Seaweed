# Runtime Pack (Inference-Ready)

This repo tracks a runtime-safe subset so GitHub users can run trained predictions immediately.

Included for inference:

- `serve_species_api.py`
- `data/netcdf/india_physics_2025w01.nc`
- `data/netcdf/india_waves_2025w01.nc`
- `data/tabular/master_feature_matrix_kappa_india_gulf_v2_hardmerge4_augmented.csv`
- `releases/kappa_india_gulf_v2_prod_ready_v4_with_v2labels/`
- `releases/kappa_india_gulf_v2_prod_ready_v3/` (fallback)
- `releases/multi_species_cop_india_v5b_rich_relaxed_soft_hn/`
- `releases/multi_species_cop_india_v2_prod/` (fallback)

Not tracked by default:

- large raw/experimental training files
- local virtualenvs and dependency caches
- outputs/snapshots/temp artifacts

Run:

```powershell
cd Strong-Seaweed-main
python -m venv .venv_model_api
.\.venv_model_api\Scripts\python -m pip install -r requirements-model-api.txt
.\.venv_model_api\Scripts\python serve_species_api.py
```
