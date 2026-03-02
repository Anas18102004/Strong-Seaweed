# Strict Geographic Holdout Gate

- Decision: **FAIL**
- Dataset glob: `*v3_rich_softpos_v6_rich_hn_v2_rich.csv`

## gracilaria_spp
- Status: **PASS**
- Rows train/val/test: 908/301/379 | Pos train/val/test: 43/31/24
- Test AUC/AP/Brier (cal): 0.9098/0.6907/0.0262
- Train AUC raw: 0.9995
- Failed checks: none

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **FAIL**
- Rows train/val/test: 953/265/380 | Pos train/val/test: 49/33/25
- Test AUC/AP/Brier (cal): 0.8256/0.5496/0.0352
- Train AUC raw: 0.9991
- Failed checks: auc_gap

## ulva_spp
- Status: **PASS**
- Rows train/val/test: 987/281/409 | Pos train/val/test: 69/45/40
- Test AUC/AP/Brier (cal): 0.8984/0.7468/0.0308
- Train AUC raw: 0.9983
- Failed checks: none
