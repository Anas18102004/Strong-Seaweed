# Strict Geographic Holdout Gate

- Decision: **FAIL**
- Dataset glob: `training_dataset_*_multispecies_cop_india_v2_prod_softpos_v4_grid.csv`

## gracilaria_spp
- Status: **FAIL**
- Rows train/val/test: 725/269/379 | Pos train/val/test: 81/44/53
- Test AUC/AP/Brier (cal): 0.7496/0.4152/0.1012
- Train AUC raw: 1.0000
- Failed checks: auc_test, auc_gap

## kappaphycus_alvarezii
- Status: **FAIL**
- Rows train/val/test: 654/237/320 | Pos train/val/test: 6/3/2
- Test AUC/AP/Brier (cal): 0.2296/0.0063/0.0070
- Train AUC raw: 1.0000
- Failed checks: min_pos_train, min_pos_val, min_pos_test, auc_test, ap_test, auc_gap

## sargassum_spp
- Status: **FAIL**
- Rows train/val/test: 724/263/375 | Pos train/val/test: 76/40/52
- Test AUC/AP/Brier (cal): 0.6671/0.3018/0.1124
- Train AUC raw: 1.0000
- Failed checks: auc_test, ap_test, auc_gap

## ulva_spp
- Status: **FAIL**
- Rows train/val/test: 761/284/383 | Pos train/val/test: 110/60/65
- Test AUC/AP/Brier (cal): 0.8192/0.5576/0.1020
- Train AUC raw: 1.0000
- Failed checks: auc_gap
