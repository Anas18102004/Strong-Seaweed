# Real-World Readiness Check

- Decision: **FAIL**
- Model version: `wc_v5_20260306_070205`

## Key metrics
- Train spatial AUC: 0.9799
- Train OOF AP (cal): 0.9521
- Hard50 AUC / P / R: 0.5520 / 0.4615 / 0.4800
- Independent n / pos / neg: 50 / 25 / 25
- Independent AUC / P / R: 0.5520 / 0.4615 / 0.4800

## Dataset QA
- Rows: 277 | Pos: 41 | Neg: 236
- Label conflicts (same lon/lat, mixed labels): 0 (ratio 0.0000)
- Missing cells: 0 | Rows with missing: 0
- Duplicate lon/lat/label rows: 0

## OOF QA
- OOF file found: True
- OOF missing rows: 0 / 246 (ratio 0.0000)

## Checks
- PASS | train_spatial_auc
- PASS | train_oof_ap_calibrated
- FAIL | hard50_auc
- FAIL | hard50_precision
- PASS | hard50_recall
- PASS | independent_n
- PASS | independent_pos
- PASS | independent_neg
- FAIL | independent_auc
- FAIL | independent_precision
- FAIL | independent_recall
- PASS | dataset_bad_labels
- PASS | dataset_missing_cells
- PASS | dataset_no_dup_lon_lat_label
- PASS | dataset_conflict_ratio
- PASS | oof_missing_ratio

## Failed checks
- hard50_auc
- hard50_precision
- independent_auc
- independent_precision
- independent_recall