# Strict Geographic Holdout Gate

- Decision: **FAIL**
- Dataset glob: `*hn_v3_rich.csv`

## gracilaria_spp
- Status: **PASS**
- Rows train/val/test: 873/298/369 | Pos train/val/test: 43/33/24
- Test AUC/AP/Brier (cal): 0.9186/0.7328/0.0258
- Train AUC raw: 0.9994
- Failed checks: none

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **FAIL**
- Rows train/val/test: 913/264/372 | Pos train/val/test: 49/33/25
- Test AUC/AP/Brier (cal): 0.8424/0.5013/0.0383
- Train AUC raw: 0.9989
- Failed checks: auc_gap

## ulva_spp
- Status: **PASS**
- Rows train/val/test: 947/280/399 | Pos train/val/test: 69/46/40
- Test AUC/AP/Brier (cal): 0.9256/0.8049/0.0278
- Train AUC raw: 0.9988
- Failed checks: none
