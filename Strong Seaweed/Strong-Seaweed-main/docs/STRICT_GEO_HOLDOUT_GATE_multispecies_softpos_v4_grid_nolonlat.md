# Strict Geographic Holdout Gate

- Decision: **FAIL**
- Dataset glob: `training_dataset_*_multispecies_cop_india_v2_prod_softpos_v4_grid.csv`

## gracilaria_spp
- Status: **FAIL**
- Rows train/val/test: 725/269/379 | Pos train/val/test: 81/44/53
- Test AUC/AP/Brier (cal): 0.7145/0.3429/0.1110
- Train AUC raw: 0.9999
- Failed checks: auc_test, ap_test, auc_gap

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **FAIL**
- Rows train/val/test: 724/263/375 | Pos train/val/test: 76/40/52
- Test AUC/AP/Brier (cal): 0.6360/0.2800/0.1219
- Train AUC raw: 1.0000
- Failed checks: auc_test, ap_test, auc_gap

## ulva_spp
- Status: **FAIL**
- Rows train/val/test: 761/284/383 | Pos train/val/test: 110/60/65
- Test AUC/AP/Brier (cal): 0.6379/0.3099/0.1418
- Train AUC raw: 1.0000
- Failed checks: auc_test, ap_test, auc_gap
