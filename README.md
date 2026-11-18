# Predictive Maintenance – Remaining Useful Life (NASA C-MAPSS)

Predicts how many operating cycles a turbofan engine has left before failure, explains each prediction with SHAP, and shows fleet health in a Streamlit dashboard.

## Pipeline
1. **Target**: `RUL = last_cycle − cycle`, clipped at 125 (piece-wise linear RUL).
2. **Preprocessing** (fit on train only): KMeans on operating settings (6 regimes for FD002/FD004) → per-regime sensor standardisation; constant sensors dropped.
3. **Features** (per engine, past data only): rolling mean (5, 15), rolling std (5), EWM, first difference, cycle count → ~100 features.
4. **Models**: Linear (baseline), Random Forest, XGBoost, LightGBM; optional LSTM / 1D-CNN on 30-cycle windows.
5. **Validation**: `GroupKFold` by engine (never by row). **Test**: last reading of each test engine vs. the official RUL file (reported uncapped and capped at 125).
6. **Metrics**: MAE, RMSE, NASA scoring function (penalises late predictions more).
7. **Explainability**: SHAP (global + per-engine). **Dashboard**: fleet table, status (🟢/🟡/🔴), health index, sensor trends, SHAP per engine.

## Results (test set, last cycle per engine)
Training targets are clipped at 125, so the model can never predict more than 125. The official `RUL_FD00x.txt` files are **not** clipped, so every engine with true RUL > 125 counts as an unavoidable error. Both views are reported (`*_capped` = true RUL clipped at 125, common practice in the literature). Best model per dataset by capped RMSE:

| Dataset | Model | MAE | RMSE | NASA score | RMSE (capped) | NASA score (capped) |
|---|---|---|---|---|---|---|
| FD001 | XGBoost | 12.69 | 16.62 | 400 | 15.42 | 358 |
| FD002 | XGBoost | 18.65 | 27.29 | 10313 | 15.31 | 1109 |
| FD003 | LightGBM | 13.78 | 18.42 | 763 | 16.78 | 549 |
| FD004 | XGBoost | 20.67 | 28.08 | 5537 | 16.98 | 1608 |

Full table (all models, MAE/NASA also capped): `reports/results_all_trees.csv`.

Most of the apparent gap on the six-regime sets (FD002/FD004: official RMSE ≈ 27–28 vs CV RMSE ≈ 16) comes from the uncapped true RUL, not from model failure: against capped RUL the test RMSE is ≈ 15–17, in line with CV. `cv_rmse` is computed on all rows of held-out engines (capped target), so it is still not identical to a last-cycle test metric.

Sequence models (`train_dl.py`) are implemented but no LSTM/CNN results are included yet.

## Run (PowerShell)
```powershell
python -m pip install -r requirements.txt
python -m src.download_data           # needs internet; data/ and models/ are git-ignored
python -m src.train --fd ALL          # or FD001
python -m src.explain --fd FD001 --unit 34
python -m src.train_dl --fd FD001 --model lstm   # needs torch
python -m streamlit run app.py
```

## Layout
`src/` (config, data, preprocessing, metrics, train, train_dl, explain) · `app.py` (dashboard) · `reports/` (metrics + plots) · `models/` (git-ignored). The dashboard needs `src.train` to be run first.
