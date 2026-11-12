"""Shared paths and basic input checks for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "zomato.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def ensure_project_directories() -> None:
    """Create output directories if they do not exist."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def require_dataset(path: Path = RAW_DATA_PATH) -> Path:
    """Return the dataset path or raise a helpful error when it is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the supplied CSV at "
            "data/raw/zomato.csv; this project does not generate replacement data."
        )
    return path
