# Strict Geographic Holdout Gate

- Decision: **FAIL**
- Dataset glob: `training_dataset_*hn_v1.csv`

## gracilaria_spp
- Status: **FAIL**
- Rows train/val/test: 892/273/406 | Pos train/val/test: 33/19/30
- Test AUC/AP/Brier (cal): 0.6114/0.1025/0.0741
- Train AUC raw: 0.9883
- Failed checks: auc_test, ap_test, auc_gap

## kappaphycus_alvarezii
- Status: **SKIP**
- Reason: excluded_species

## sargassum_spp
- Status: **FAIL**
- Rows train/val/test: 916/251/405 | Pos train/val/test: 36/21/26
- Test AUC/AP/Brier (cal): 0.5590/0.0818/0.0673
- Train AUC raw: 0.9852
- Failed checks: auc_test, ap_test, auc_gap

## ulva_spp
- Status: **FAIL**
- Rows train/val/test: 942/255/397 | Pos train/val/test: 40/30/35
- Test AUC/AP/Brier (cal): 0.6854/0.1531/0.0956
- Train AUC raw: 0.9795
- Failed checks: auc_test, ap_test, auc_gap
