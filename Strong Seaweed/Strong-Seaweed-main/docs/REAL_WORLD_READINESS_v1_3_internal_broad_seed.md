# Real-World Readiness Check

- Decision: **FAIL**
- Model version: `v1.1r_plus35provisional`

## Key metrics
- Train spatial AUC: 0.7648
- Train OOF AP (cal): 0.5445
- Hard50 AUC / P / R: 0.9288 / 1.0000 / 0.4400
- Independent n / pos / neg: 15 / 1 / 14
- Independent AUC / P / R: 0.6786 / 0.0000 / 0.0000

## Dataset QA
- Rows: 762 | Pos: 127 | Neg: 635
- Label conflicts (same lon/lat, mixed labels): 0 (ratio 0.0000)
- Missing cells: 0 | Rows with missing: 0
- Duplicate lon/lat/label rows: 0

## OOF QA
- OOF file found: True
- OOF missing rows: 90 / 351 (ratio 0.2564)

## Checks
- PASS | train_spatial_auc
- PASS | train_oof_ap_calibrated
- PASS | hard50_auc
- PASS | hard50_precision
- PASS | hard50_recall
- FAIL | independent_n
- FAIL | independent_pos
- FAIL | independent_neg
- FAIL | independent_auc
- FAIL | independent_precision
- FAIL | independent_recall
- PASS | dataset_bad_labels
- PASS | dataset_missing_cells
- PASS | dataset_no_dup_lon_lat_label
- PASS | dataset_conflict_ratio
- FAIL | oof_missing_ratio

## Failed checks
- independent_n
- independent_pos
- independent_neg
- independent_auc
- independent_precision
- independent_recall
- oof_missing_ratio