# Strict Geographic Holdout Gate

- Decision: **PASS**
- Dataset glob: `*hn_v3_rich.csv`

## gracilaria_spp
- Status: **PASS**
- Rows train/val/test: 873/298/369 | Pos train/val/test: 43/33/24
- Test AUC/AP/Brier (cal): 0.9577/0.7833/0.0245
- Train AUC raw: 0.9924
- Failed checks: none

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **PASS**
- Rows train/val/test: 913/264/372 | Pos train/val/test: 49/33/25
- Test AUC/AP/Brier (cal): 0.8665/0.5259/0.0381
- Train AUC raw: 0.9909
- Failed checks: none

## ulva_spp
- Status: **PASS**
- Rows train/val/test: 947/280/399 | Pos train/val/test: 69/46/40
- Test AUC/AP/Brier (cal): 0.9319/0.7691/0.0301
- Train AUC raw: 0.9899
- Failed checks: none
