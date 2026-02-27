# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1_1p2_exp)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.075171, 79.799937], lat [8.403120, 9.786525]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1_1p2_exp_plus15_augmented_deployable
- Samples: 320
- Positives: 61
- Negatives: 259
- Features: 29
- Spatial folds: 5

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.6192 +/- 0.1859
- Spatial AP: 0.4241
- OOF calibrated AP: 0.3760
- OOF calibrated Brier: 0.1902

## Deployment Policy
- Threshold: 0.285714
- Precision at threshold: 0.3615
- Recall at threshold: 0.8246
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (320 rows, 61 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
