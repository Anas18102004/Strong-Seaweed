# Strict Geographic Holdout Gate

- Decision: **PASS**
- Dataset glob: `training_dataset_*_wc_v2_20260306_060653_cop_india_rich.csv`

## gracilaria_spp
- Status: **PASS**
- Rows train/val/test: 685/265/348 | Pos train/val/test: 43/31/24
- Test AUC/AP/Brier (cal): 0.9163/0.7139/0.0299
- Train AUC raw: 0.9865
- Failed checks: none

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **PASS**
- Rows train/val/test: 694/265/348 | Pos train/val/test: 49/33/25
- Test AUC/AP/Brier (cal): 0.8646/0.4683/0.0448
- Train AUC raw: 0.9867
- Failed checks: none

## ulva_spp
- Status: **PASS**
- Rows train/val/test: 731/281/374 | Pos train/val/test: 69/45/40
- Test AUC/AP/Brier (cal): 0.9171/0.7788/0.0326
- Train AUC raw: 0.9816
- Failed checks: none
