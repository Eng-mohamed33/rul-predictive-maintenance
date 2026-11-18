"""Leak-free preprocessing: operating-condition normalisation + per-engine rolling features.

Everything that learns something (KMeans, scalers, dropped sensors) is fit on TRAIN only.
All rolling / diff / ewm features are computed per engine (groupby unit), using past data only.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import OP_COLS, SENSOR_COLS, N_CONDITIONS, SEED


class Preprocessor:
    def __init__(self, fd: str, windows=(5, 15), std_window=5, ewm_span=10):
        self.fd = fd
        self.windows = windows
        self.std_window = std_window
        self.ewm_span = ewm_span

    # ---- fitting -------------------------------------------------------
    def _condition(self, df):
        if self.kmeans is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans.predict(df[OP_COLS].values)

    def fit(self, train: pd.DataFrame):
        k = N_CONDITIONS[self.fd]
        self.kmeans = KMeans(k, n_init=10, random_state=SEED).fit(train[OP_COLS].values) if k > 1 else None
        cond = self._condition(train)
        self.scalers = {c: StandardScaler().fit(train.loc[cond == c, SENSOR_COLS]) for c in range(k)}
        norm = self._normalize_all(train, cond)
        self.sensors = [s for s in SENSOR_COLS if norm[s].std() > 1e-6]   # drop constant sensors
        return self

    # ---- transforming --------------------------------------------------
    def _normalize_all(self, df, cond):
        df = df.reset_index(drop=True).copy()
        df[SENSOR_COLS] = df[SENSOR_COLS].astype(float)
        for c, sc in self.scalers.items():
            m = cond == c
            if m.any():
                df.loc[m, SENSOR_COLS] = sc.transform(df.loc[m, SENSOR_COLS])
        return df

    def normalize(self, df):
        """Condition-normalised sensors (used by the deep-learning models)."""
        return self._normalize_all(df, self._condition(df))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # rolling / ewm / diff assume chronological order inside each engine
        df = df.sort_values(["unit", "cycle"], kind="stable")
        norm = self.normalize(df)
        g = norm.groupby("unit")[self.sensors]
        parts = [norm[["unit", "cycle"]], norm[self.sensors]]
        for w in self.windows:
            parts.append(g.rolling(w, min_periods=1).mean().droplevel(0).add_suffix(f"_mean{w}"))
        parts.append(
            g.rolling(self.std_window, min_periods=2).std().droplevel(0).fillna(0).add_suffix(f"_std{self.std_window}")
        )
        parts.append(g.ewm(span=self.ewm_span, adjust=False).mean().droplevel(0).add_suffix("_ewm"))
        parts.append(g.diff().fillna(0).add_suffix("_diff"))
        out = pd.concat(parts, axis=1)
        self.feature_names_ = [c for c in out.columns if c != "unit"]   # includes `cycle`
        return out
