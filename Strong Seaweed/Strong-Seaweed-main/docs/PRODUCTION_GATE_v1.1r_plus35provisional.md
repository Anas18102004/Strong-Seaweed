# Production Gate: v1.1r_plus35provisional

- Result: **WARN**

## Key Metrics
- train_spatial_auc: 0.7648395120134249
- train_oof_ap_calibrated: 0.5445229507600359
- hard50_auc: 0.9288
- hard50_precision: 1.0
- hard50_recall: 0.44
- independent_n: 15
- independent_pos: 1
- independent_neg: 14
- independent_auc: 0.6785714285714286
- independent_precision: 0.0
- independent_recall: 0.0

## Checks
- PASS | train_spatial_auc | value=0.7648395120134249 | threshold=0.7
- PASS | train_oof_ap_calibrated | value=0.5445229507600359 | threshold=0.45
- PASS | hard50_auc | value=0.9288 | threshold=0.85
- PASS | hard50_precision | value=1.0 | threshold=0.8
- PASS | hard50_recall | value=0.44 | threshold=0.4
- FAIL | independent_sample_size | value=15 | threshold=40
- FAIL | independent_positive_count | value=1 | threshold=20
- FAIL | independent_negative_count | value=14 | threshold=20
- FAIL | independent_auc | value=0.6785714285714286 | threshold=0.75
- FAIL | independent_precision | value=0.0 | threshold=0.7
- FAIL | independent_recall | value=0.0 | threshold=0.6

## Decision
Strong in-sample and hard50 performance, but independent leakage-free holdout is insufficient. Treat as provisional until >=40 holdout rows with >=20 positives and >=20 negatives are verified.