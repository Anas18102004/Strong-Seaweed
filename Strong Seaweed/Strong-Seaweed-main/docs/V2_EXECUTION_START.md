# V2 Execution Start

## Files Added
- `data/config/v2_dataset_schema.json`
- `data/config/v2_species_configs.json`
- `data/tabular/v2_unified_dataset_template.csv`
- `validate_v2_dataset.py`
- `build_v2_spatial_splits.py`

## Step 1: Fill Dataset
Populate `data/tabular/v2_unified_dataset_template.csv` (or a new CSV) with your real v2 rows.

## Step 2: Validate
```powershell
python validate_v2_dataset.py --input_csv data/tabular/v2_unified_dataset_template.csv
```

## Step 3: Build Splits
```powershell
python build_v2_spatial_splits.py --input_csv data/tabular/v2_unified_dataset_template.csv
```

## Outputs
- Validation report: `artifacts/reports/v2_dataset_validation_report.json`
- Split CSV: `data/tabular/v2_unified_dataset_with_splits.csv`
- Split report: `artifacts/reports/v2_spatial_split_report.json`
