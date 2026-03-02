# V1.1 Next-12 Positives Checklist

## 1) Fill template
- File: `data/tabular/v1_1_next12_positives_template.csv`
- Required quality:
  - `species = Kappaphycus alvarezii`
  - `confidence_score >= 0.90`
  - `coordinate_precision_km <= 1`
  - `is_verified = true`
  - `qa_status = approved`
  - `notes` includes `verified_dates=YYYY-MM-DD;YYYY-MM-DD;YYYY-MM-DD`

## 2) Source balance target (recommended)
- `literature`: 4-5
- `government`: 4-5
- `satellite_digitized`: 2-3

## 3) Diversity target (recommended)
- Avoid adding points from the same 1-2 km strip.
- Prefer points across:
  - depth 5-8m
  - distance_to_shore 1000-2000m
  - moderate wave exposure
  - west + east edges of Gulf/Palk belt

## 4) Strict ingestion command
```powershell
python ingest_presence_records.py `
  --inputs data/tabular/v1_1_next12_positives_template.csv `
  --master_csv data/tabular/master_feature_matrix_v1_1.csv `
  --training_csv data/tabular/training_dataset_v1_1_merged46_plus_hn30_augmented.csv `
  --max_snap_m 1500 `
  --species_filter kappaphycus `
  --strict_acceptance `
  --strict_min_confidence 0.9 `
  --strict_min_verified_dates 2 `
  --out_csv artifacts/experiments/v1_1_next12_snapped_strict.csv `
  --out_report artifacts/experiments/v1_1_next12_report_strict.json
```

## 5) Acceptance rule
- Proceed to retraining only if `>= 8` unique new snapped positives are accepted.

## 6) Retrain command (comparison release)
```powershell
python train_realtime_production.py --fast --production `
  --release_tag v1.1r_next12_cmp `
  --dataset_paths data/tabular/training_dataset_v1_1_merged46_plus_hn30_augmented.csv `
  --inference_feature_source data/tabular/master_feature_matrix_v1_1_augmented.csv
```

## 7) Promote criteria
- Promote only if all hold:
  - Spatial AUC >= current baseline
  - OOF calibrated AP >= current baseline
  - Threshold behavior becomes more stable (not extreme)

