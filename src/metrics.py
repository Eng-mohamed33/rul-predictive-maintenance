import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def nasa_score(y, p):
    """PHM08 scoring: late predictions (p > y) are penalised more than early ones."""
    d = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def report(y, p):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "NASA_score": nasa_score(y, p)}


def report_both(y, p, cap):
    """Metrics against the official (uncapped) RUL and against RUL clipped at `cap`.

    Training targets are clipped at `cap`, so a model can never predict above it. Scoring
    against the uncapped official RUL therefore counts every engine with true RUL > cap as an
    unavoidable error. The *_capped columns remove that artefact (common practice in the literature).
    """
    y = np.asarray(y, dtype=float)
    out = report(y, p)
    out.update({f"{k}_capped": v for k, v in report(np.minimum(y, cap), p).items()})
    return out
