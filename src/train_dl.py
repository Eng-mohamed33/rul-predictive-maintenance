"""LSTM / 1D-CNN on sliding windows.   python -m src.train_dl --fd FD001 --model lstm"""
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import MODELS_DIR, REPORTS_DIR, RUL_CAP, SEED
from src.data import load_fd, add_rul
from src.metrics import report_both
from src.preprocessing import Preprocessor

WIN = 30


def unit_windows(arr, win=WIN):
    """All windows ending at each cycle (front-padded with the first row)."""
    pad = np.vstack([np.repeat(arr[:1], win - 1, axis=0), arr])
    return np.stack([pad[i:i + win] for i in range(len(arr))]).astype(np.float32)


def build(norm, sensors, with_target, last_only=False):
    X, y, u = [], [], []
    for unit, d in norm.groupby("unit"):
        w = unit_windows(d[sensors].values)
        if last_only:
            w = w[-1:]
        X.append(w); u += [unit] * len(w)
        if with_target:
            y.append(d["RUL"].values)
    return np.concatenate(X), (np.concatenate(y).astype(np.float32) if with_target else None), np.array(u)


class LSTMNet(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.rnn = nn.LSTM(f, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


class CNNNet(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * WIN, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


def run(fd, model_name="lstm", epochs=30, bs=256, lr=1e-3):
    torch.manual_seed(SEED); np.random.seed(SEED)
    train, test, rul_true = load_fd(fd)
    train = add_rul(train)
    prep = Preprocessor(fd).fit(train)
    ntr, nte = prep.normalize(train), prep.normalize(test)
    ntr["RUL"] = train["RUL"].values
    S = prep.sensors

    units = ntr.unit.unique()
    rng = np.random.RandomState(SEED); rng.shuffle(units)
    val_units = set(units[: int(0.15 * len(units))])          # validation split BY ENGINE
    Xa, ya, ua = build(ntr, S, True)
    tr, va = ~np.isin(ua, list(val_units)), np.isin(ua, list(val_units))
    Xt, _, _ = build(nte, S, False, last_only=True)

    net = (LSTMNet if model_name == "lstm" else CNNNet)(len(S))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    T = lambda a: torch.from_numpy(a)
    Xtr, ytr, Xva, yva = T(Xa[tr]), T(ya[tr] / RUL_CAP), T(Xa[va]), T(ya[va] / RUL_CAP)
    best, best_state, patience = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xva) - yva) ** 2).mean().sqrt()) * RUL_CAP
        if v < best:
            best, best_state, patience = v, copy.deepcopy(net.state_dict()), 0
        else:
            patience += 1
        print(f"  epoch {ep + 1:02d}  val_rmse={v:.2f}")
        if patience >= 6:
            break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        p = np.clip(net(T(Xt)).numpy() * RUL_CAP, 0, RUL_CAP)
    res = {"dataset": fd, "model": model_name.upper(), "val_rmse": best, **report_both(rul_true, p, RUL_CAP)}
    print(res)
    torch.save(best_state, MODELS_DIR / f"{fd}_{model_name}.pt")
    pd.DataFrame([res]).to_csv(REPORTS_DIR / f"results_{fd}_{model_name}.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--model", default="lstm", choices=["lstm", "cnn"])
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    run(a.fd.upper(), a.model, a.epochs)
