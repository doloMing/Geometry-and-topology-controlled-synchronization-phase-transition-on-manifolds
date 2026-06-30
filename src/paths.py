from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"


def ensure_directories() -> None:
    """Create output directories used by the notebooks."""
    for path in (DATA_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)

