"""SHAP explanations.   python -m src.explain --fd FD001 [--unit 34]"""
import argparse
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_fd, add_rul


def load_assets(fd):
    return joblib.load(MODELS_DIR / f"{fd}_prep.joblib"), joblib.load(MODELS_DIR / f"{fd}_lightgbm.joblib")


def global_summary(fd, n=2000):
    prep, model = load_assets(fd)
    train, _, _ = load_fd(fd)
    X = prep.transform(add_rul(train))[prep.feature_names_].sample(n, random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(X)
    for kind, name in (("dot", "summary"), ("bar", "importance")):
        shap.summary_plot(sv, X, plot_type=kind, max_display=15, show=False)
        plt.tight_layout(); plt.savefig(REPORTS_DIR / f"{fd}_shap_{name}.png", dpi=130, bbox_inches="tight"); plt.close()
    print("saved global SHAP plots for", fd)


def explain_engine(fd, unit, top=12):
    """Return (predicted_rul, DataFrame of top contributing features) for an engine's last cycle."""
    prep, model = load_assets(fd)
    _, test, _ = load_fd(fd)
    feats = prep.transform(test)
    row = feats[feats.unit == unit].tail(1)[prep.feature_names_]
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(row)[0]
    import pandas as pd
    df = pd.DataFrame({"feature": prep.feature_names_, "value": row.values[0], "shap": sv})
    df = df.reindex(df.shap.abs().sort_values(ascending=False).index).head(top)
    return float(np.clip(model.predict(row)[0], 0, 125)), float(np.ravel(ex.expected_value)[0]), df


def engine_plot(fd, unit):
    pred, base, df = explain_engine(fd, unit)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = df.iloc[::-1]
    ax.barh(d.feature, d.shap, color=np.where(d.shap > 0, "#2ca02c", "#d62728"))
    ax.set(title=f"{fd} engine {unit}: predicted RUL {pred:.0f} (baseline {base:.0f})", xlabel="SHAP value (cycles)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / f"{fd}_engine{unit}_shap.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--unit", type=int, default=None)
    a = ap.parse_args()
    global_summary(a.fd)
    if a.unit:
        engine_plot(a.fd, a.unit)
