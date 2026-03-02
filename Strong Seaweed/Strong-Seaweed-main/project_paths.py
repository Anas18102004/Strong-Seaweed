from pathlib import Path


BASE = Path(__file__).resolve().parent

DATA_DIR = BASE / "data"
TABULAR_DIR = DATA_DIR / "tabular"
RASTER_DIR = DATA_DIR / "rasters"
NETCDF_DIR = DATA_DIR / "netcdf"

MODELS_DIR = BASE / "models"
REALTIME_MODELS_DIR = MODELS_DIR / "realtime"

OUTPUTS_DIR = BASE / "outputs"
DOCS_DIR = BASE / "docs"
REPORTS_DIR = BASE / "artifacts" / "reports"
DIAGNOSTICS_DIR = BASE / "artifacts" / "diagnostics"
EXPERIMENTS_DIR = BASE / "artifacts" / "experiments"


def ensure_dirs() -> None:
    for d in [
        DATA_DIR,
        TABULAR_DIR,
        RASTER_DIR,
        NETCDF_DIR,
        MODELS_DIR,
        REALTIME_MODELS_DIR,
        OUTPUTS_DIR,
        DOCS_DIR,
        REPORTS_DIR,
        DIAGNOSTICS_DIR,
        EXPERIMENTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def with_legacy(new_path: Path, legacy_name: str) -> Path:
    legacy = BASE / legacy_name
    return new_path if new_path.exists() else legacy
