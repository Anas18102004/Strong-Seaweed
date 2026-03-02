# Strict Geographic Holdout Gate

- Decision: **FAIL**
- Dataset glob: `training_dataset_*_multispecies_cop_india_v2_prod.csv`

## gracilaria_spp
- Status: **FAIL**
- Rows train/val/test: 672/242/351 | Pos train/val/test: 26/17/22
- Test AUC/AP/Brier (cal): 0.6803/0.1197/0.0610
- Train AUC raw: 1.0000
- Failed checks: min_pos_train, auc_test, ap_test, auc_gap

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **FAIL**
- Rows train/val/test: 676/241/347 | Pos train/val/test: 25/18/21
- Test AUC/AP/Brier (cal): 0.6444/0.1236/0.0598
- Train AUC raw: 1.0000
- Failed checks: min_pos_train, auc_test, ap_test, auc_gap

## ulva_spp
- Status: **FAIL**
- Rows train/val/test: 687/248/350 | Pos train/val/test: 32/24/29
- Test AUC/AP/Brier (cal): 0.5887/0.1080/0.0826
- Train AUC raw: 1.0000
- Failed checks: auc_test, ap_test, auc_gap
