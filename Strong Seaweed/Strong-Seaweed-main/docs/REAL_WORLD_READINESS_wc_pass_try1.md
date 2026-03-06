# Real-World Readiness Check

- Decision: **FAIL**
- Model version: `wc_pass_try1`

## Key metrics
- Train spatial AUC: 0.9012
- Train OOF AP (cal): 0.5284
- Hard50 AUC / P / R: 0.5816 / 0.0000 / 0.0000
- Independent n / pos / neg: 50 / 25 / 25
- Independent AUC / P / R: 0.5816 / 0.0000 / 0.0000

## Dataset QA
- Rows: 285 | Pos: 37 | Neg: 248
- Label conflicts (same lon/lat, mixed labels): 8 (ratio 0.0281)
- Missing cells: 0 | Rows with missing: 0
- Duplicate lon/lat/label rows: 0

## OOF QA
- OOF file found: True
- OOF missing rows: 0 / 285 (ratio 0.0000)

## Checks
- PASS | train_spatial_auc
- PASS | train_oof_ap_calibrated
- FAIL | hard50_auc
- FAIL | hard50_precision
- FAIL | hard50_recall
- PASS | independent_n
- PASS | independent_pos
- PASS | independent_neg
- FAIL | independent_auc
- FAIL | independent_precision
- FAIL | independent_recall
- PASS | dataset_bad_labels
- PASS | dataset_missing_cells
- PASS | dataset_no_dup_lon_lat_label
- FAIL | dataset_conflict_ratio
- PASS | oof_missing_ratio

## Failed checks
- hard50_auc
- hard50_precision
- hard50_recall
- independent_auc
- independent_precision
- independent_recall
- dataset_conflict_ratio