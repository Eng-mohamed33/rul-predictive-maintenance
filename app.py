"""Fleet health dashboard.   streamlit run app.py"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR, RUL_CAP, CRITICAL_RUL, WARNING_RUL
from src.data import load_fd
from src.explain import explain_engine

st.set_page_config(page_title="Engine RUL Dashboard", page_icon="✈️", layout="wide")


def status(rul):
    return "🔴 Critical" if rul <= CRITICAL_RUL else "🟡 Warning" if rul <= WARNING_RUL else "🟢 Healthy"


@st.cache_resource
def load(fd):
    prep = joblib.load(MODELS_DIR / f"{fd}_prep.joblib")
    model = joblib.load(MODELS_DIR / f"{fd}_lightgbm.joblib")
    _, test, rul_true = load_fd(fd)
    feats = prep.transform(test)
    feats["pred_rul"] = np.clip(model.predict(feats[prep.feature_names_]), 0, RUL_CAP)
    return prep, feats, rul_true


available = [f for f in ("FD001", "FD002", "FD003", "FD004") if (MODELS_DIR / f"{f}_lightgbm.joblib").exists()]
if not available:
    st.error("No trained models found. Run `python -m src.train --fd FD001` first.")
    st.stop()

fd = st.sidebar.selectbox("Dataset", available)
prep, feats, rul_true = load(fd)
last = feats.groupby("unit").tail(1).sort_values("unit").reset_index(drop=True)
last["true_rul"] = rul_true
last["status"] = last.pred_rul.apply(status)
fleet = last[["unit", "cycle", "pred_rul", "true_rul", "status"]].rename(
    columns={"cycle": "cycles_observed", "pred_rul": "predicted_RUL"})
fleet["predicted_RUL"] = fleet.predicted_RUL.round(1)

st.title("✈️ Turbofan Engine Health – Remaining Useful Life")
c = st.columns(4)
c[0].metric("Engines", len(fleet))
for col, s in zip(c[1:], ("🔴 Critical", "🟡 Warning", "🟢 Healthy")):
    col.metric(s, int((fleet.status == s).sum()))

left, right = st.columns([1, 1.4])
with left:
    st.subheader("Fleet status")
    order = st.radio("Sort by", ["Most critical first", "Unit id"], horizontal=True)
    view = fleet.sort_values("predicted_RUL") if order.startswith("Most") else fleet
    st.dataframe(view, hide_index=True, height=480, width="stretch")

with right:
    unit = st.selectbox("Inspect engine", fleet.unit.tolist(),
                        index=int(fleet.predicted_RUL.idxmin()))
    e = feats[feats.unit == unit]
    row = fleet[fleet.unit == unit].iloc[0]
    st.markdown(f"### Engine {unit} – {row.status}  \nPredicted RUL **{row.predicted_RUL:.0f}** cycles "
                f"(true: {row.true_rul:.0f}) after {row.cycles_observed} observed cycles")

    st.caption("Health index = predicted RUL / cap, recomputed at every observed cycle (smoothed)")
    hi = (e.pred_rul.rolling(5, min_periods=1).mean() / RUL_CAP * 100)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(e.cycle, hi, lw=2)
    ax.axhline(CRITICAL_RUL / RUL_CAP * 100, color="r", ls="--", lw=1)
    ax.axhline(WARNING_RUL / RUL_CAP * 100, color="orange", ls="--", lw=1)
    ax.set(ylim=(0, 105), xlabel="cycle", ylabel="health index (%)")
    st.pyplot(fig); plt.close(fig)

    sensors = st.multiselect("Sensors (condition-normalised)", prep.sensors,
                             default=prep.sensors[:4])
    if sensors:
        st.line_chart(e.set_index("cycle")[sensors], height=220)

st.subheader("Why this prediction? (SHAP, last cycle)")
pred, base, df = explain_engine(fd, int(unit))
fig, ax = plt.subplots(figsize=(8, 3.6))
d = df.iloc[::-1]
ax.barh(d.feature, d.shap, color=np.where(d.shap > 0, "#2ca02c", "#d62728"))
ax.set(xlabel=f"effect on RUL in cycles (baseline {base:.0f} → prediction {pred:.0f})")
st.pyplot(fig); plt.close(fig)
st.caption("Green pushes the RUL up (healthier), red pushes it down (more degraded).")

with st.expander("Global model behaviour"):
    for name in ("shap_importance", "pred_vs_true"):
        p = REPORTS_DIR / f"{fd}_{name}.png"
        if p.exists():
            st.image(str(p), width=420)
