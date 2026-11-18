"""Train + evaluate classical models on C-MAPSS.   python -m src.train --fd FD001"""
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import MODELS_DIR, REPORTS_DIR, RUL_CAP, SEED
from src.data import load_fd, add_rul
from src.metrics import report_both, rmse
from src.preprocessing import Preprocessor


def get_models():
    return {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.6, n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, n_jobs=-1, random_state=SEED, verbose=-1),
    }


def run(fd: str, cv_folds: int = 3):
    train, test, rul_true = load_fd(fd)
    train = add_rul(train)
    prep = Preprocessor(fd).fit(train)
    ftr, fte = prep.transform(train), prep.transform(test)
    cols = prep.feature_names_
    X, y, groups = ftr[cols], train["RUL"].values, train["unit"].values
    Xt = fte.groupby("unit").tail(1).sort_values("unit")[cols]    # last reading of every test engine
    print(f"[{fd}] train {X.shape}, test engines {len(Xt)}, sensors kept {len(prep.sensors)}")

    rows, preds = [], {}
    for name, model in get_models().items():
        cv = np.nan
        if cv_folds > 1:   # split BY ENGINE, never by row
            scores = []
            for tr, va in GroupKFold(cv_folds).split(X, y, groups):
                m = clone(model).fit(X.iloc[tr], y[tr])
                scores.append(rmse(y[va], np.clip(m.predict(X.iloc[va]), 0, RUL_CAP)))
            cv = float(np.mean(scores))
        model.fit(X, y)
        p = np.clip(model.predict(Xt), 0, RUL_CAP)
        preds[name] = p
        rows.append({"dataset": fd, "model": name, "cv_rmse": cv, **report_both(rul_true, p, RUL_CAP)})
        print(f"  {name:13s} cv_rmse={cv:6.2f}  " + "  ".join(f"{k}={v:.2f}" for k, v in rows[-1].items() if k in ("MAE", "RMSE", "NASA_score", "RMSE_capped")))
        if name in ("LightGBM", "XGBoost"):
            joblib.dump(model, MODELS_DIR / f"{fd}_{name.lower()}.joblib")

    joblib.dump(prep, MODELS_DIR / f"{fd}_prep.joblib")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / f"results_{fd}_trees.csv", index=False)

    best = res.sort_values("RMSE").iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rul_true, preds[best], s=14, alpha=0.7)
    ax.plot([0, 150], [0, 150], "r--", lw=1)
    ax.set(xlabel="True RUL", ylabel="Predicted RUL", title=f"{fd} – {best}")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / f"{fd}_pred_vs_true.png", dpi=130); plt.close(fig)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001", help="FD001..FD004 or ALL")
    ap.add_argument("--cv", type=int, default=3, help="GroupKFold splits (0/1 = skip)")
    a = ap.parse_args()
    fds = ["FD001", "FD002", "FD003", "FD004"] if a.fd.upper() == "ALL" else [a.fd.upper()]
    out = pd.concat([run(f, a.cv) for f in fds])
    if len(fds) > 1:
        out.to_csv(REPORTS_DIR / "results_all_trees.csv", index=False)
    print("\n", out.round(2).to_string(index=False))
