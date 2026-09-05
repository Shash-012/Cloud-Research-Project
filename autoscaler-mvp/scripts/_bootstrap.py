"""Put src/ on the import path so the scripts run without an install step."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
for _d in (DATA, MODELS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
