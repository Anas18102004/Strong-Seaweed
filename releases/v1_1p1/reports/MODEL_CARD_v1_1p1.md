# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.0)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.075171, 79.799937], lat [8.403120, 9.786525]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: training_dataset_v1_1_merged46_plus_hn30_deployable
- Samples: 305
- Positives: 46
- Negatives: 259
- Features: 23
- Spatial folds: 5

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.6365 +/- 0.1900
- Spatial AP: 0.4756
- OOF calibrated AP: 0.3217
- OOF calibrated Brier: 0.1629

## Deployment Policy
- Threshold: 0.284615
- Precision at threshold: 0.2971
- Recall at threshold: 0.9318
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (305 rows, 46 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
