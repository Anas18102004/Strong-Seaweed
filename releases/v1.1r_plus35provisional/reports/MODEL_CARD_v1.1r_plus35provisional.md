# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.1r_plus35provisional)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.075171, 79.799937], lat [8.403120, 9.786525]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1_1_plus11web_plus35provisional_deployable
- Samples: 351
- Positives: 92
- Negatives: 259
- Features: 29
- Spatial folds: 6

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.7648 +/- 0.1823
- Spatial AP: 0.7441
- OOF calibrated AP: 0.5445
- OOF calibrated Brier: 0.1870

## Deployment Policy
- Threshold: 0.714286
- Precision at threshold: 0.7500
- Recall at threshold: 0.0674
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (351 rows, 92 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
