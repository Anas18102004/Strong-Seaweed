# Model Card: Gulf of Mannar Regional Ecological Screening Model (v1.0)

## Scope
- Region: Gulf of Mannar
- Bounding box: lon [78.513331, 79.788938], lat [9.029702, 9.766321]
- Intended use: Regional ecological suitability screening
- Not intended for: Micro-site (<5 km) placement decisions

## Training Data
- Selected dataset: combined_deployable
- Samples: 246
- Positives: 41
- Negatives: 205
- Features: 23
- Spatial folds: 5

## Resolution
- Effective ecological resolution: ~22 km
- Coarsest layer: wave_height (~0.20 degree, ~22 km)
- Secondary ocean physics resolution: ~9.2 km

## Validation Snapshot
- Spatial AUC: 0.9288 +/- 0.1226
- Spatial AP: 0.9042
- OOF calibrated AP: 0.8051
- OOF calibrated Brier: 0.0944

## Deployment Policy
- Threshold: 0.642857
- Precision at threshold: 0.7500
- Recall at threshold: 0.8571
- Priority policy: high >= 0.80, medium >= 0.60

## Known Limitations
- Dataset size is limited (246 rows, 41 positives).
- Geographic coverage is regional (Gulf of Mannar), not pan-India.
- Calibrated outputs are stepwise due to isotonic calibration.
