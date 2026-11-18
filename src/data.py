import pandas as pd
from src.config import DATA_DIR, COLS, RUL_CAP


def load_fd(fd: str):
    """Return (train_df, test_df, true_rul_for_test_units)."""
    train = pd.read_csv(DATA_DIR / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(DATA_DIR / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(DATA_DIR / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["RUL"])["RUL"].values
    return train, test, rul


def add_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """RUL = last_cycle - cycle, clipped at `cap` (healthy phase is indistinguishable)."""
    out = train.copy()
    out["RUL"] = (out.groupby("unit")["cycle"].transform("max") - out["cycle"]).clip(upper=cap)
    return out
