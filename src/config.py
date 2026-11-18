from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)

OP_COLS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

RUL_CAP = 125                      # piece-wise linear RUL target
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SEED = 42

# dashboard status thresholds (cycles)
CRITICAL_RUL = 30
WARNING_RUL = 75
