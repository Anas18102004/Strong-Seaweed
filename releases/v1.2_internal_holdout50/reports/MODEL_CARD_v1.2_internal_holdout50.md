# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.2_internal_holdout50)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.075171, 79.799937], lat [8.403120, 9.786525]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1.2_internal_train_minus_holdout50_deployable
- Samples: 316
- Positives: 82
- Negatives: 234
- Features: 28
- Spatial folds: 5

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.7513 +/- 0.1966
- Spatial AP: 0.7173
- OOF calibrated AP: 0.5942
- OOF calibrated Brier: 0.1837

## Deployment Policy
- Threshold: 1.000000
- Precision at threshold: 1.0000
- Recall at threshold: 0.0395
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (316 rows, 82 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
