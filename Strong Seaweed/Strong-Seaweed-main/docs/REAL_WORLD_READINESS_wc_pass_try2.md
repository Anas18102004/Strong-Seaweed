# Real-World Readiness Check

- Decision: **FAIL**
- Model version: `wc_pass_try2`

## Key metrics
- Train spatial AUC: 0.8869
- Train OOF AP (cal): 0.7550
- Hard50 AUC / P / R: 0.7888 / 0.8182 / 0.7200
- Independent n / pos / neg: 44 / 22 / 22
- Independent AUC / P / R: 0.7510 / 0.7895 / 0.6818

## Dataset QA
- Rows: 449 | Pos: 123 | Neg: 326
- Label conflicts (same lon/lat, mixed labels): 0 (ratio 0.0000)
- Missing cells: 0 | Rows with missing: 0
- Duplicate lon/lat/label rows: 0

## OOF QA
- OOF file found: True
- OOF missing rows: 0 / 449 (ratio 0.0000)

## Checks
- PASS | train_spatial_auc
- PASS | train_oof_ap_calibrated
- FAIL | hard50_auc
- PASS | hard50_precision
- PASS | hard50_recall
- PASS | independent_n
- PASS | independent_pos
- PASS | independent_neg
- PASS | independent_auc
- PASS | independent_precision
- PASS | independent_recall
- PASS | dataset_bad_labels
- PASS | dataset_missing_cells
- PASS | dataset_no_dup_lon_lat_label
- PASS | dataset_conflict_ratio
- PASS | oof_missing_ratio

## Failed checks
- hard50_auc