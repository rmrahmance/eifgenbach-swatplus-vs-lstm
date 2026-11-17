"""
STEP 6 - LSTM rainfall-runoff model (deep learning)
====================================================
An LSTM ("Long Short-Term Memory" neural network) reads the weather of the last
365 days and predicts today's discharge. It learns the rainfall -> runoff relation
directly from the data (no physics equations inside).

Two experiments are run one after the other:
  main experiment : rainfall (RADOLAN, 3 subbasins) + ERA5-Land climate + day of year
  rain_only       : rainfall + day of year only (to see how much the climate inputs help)

Input  : Models/data_processed/observed_daily.csv   (RADOLAN rain + observed Q)
         Models/data_processed/climate_daily.csv    (ERA5-Land temperature, radiation, humidity, ET0)
Output : Models/LSTM/lstm_seed*.pt          trained network weights (PyTorch)
         Models/LSTM/lstm_scaler.json       the numbers used to normalise the inputs
         Models/LSTM/lstm_simulation.csv    observed vs simulated discharge, every day
         Models/LSTM/lstm_metrics.csv       NSE, KGE, PBIAS for training / validation / test
         Models/LSTM/lstm_training_log.csv  loss per epoch
         Models/LSTM/experiment_rain_only/  the same files for the rain-only experiment
How to run: open this file in VS Code and press the Run button (20-60 minutes on a laptop CPU).
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from _common import DP, LS, AREA_KM2, start_log, done

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) - 1)))   # more threads are not faster for a small LSTM

# ------------------------------------------------------------------ settings
SEQ_LEN = 365                          # look-back window: 1 year of weather
HIDDEN = 64                            # size of the LSTM "memory"
DROPOUT = 0.3                          # randomly switch off 30 % of connections while training
EPOCHS = 25                            # passes through the training data
BATCH = 128                            # examples per weight update
LR = 1e-3                              # learning rate of the Adam optimiser
SEEDS = [1, 2, 3]                      # 3 networks -> ensemble mean (more robust)
TRAIN = ("2003-01-01", "2010-12-31")   # the network learns here
VALID = ("2011-01-01", "2012-12-31")   # early stopping: keep the epoch that is best here
TEST = ("2013-01-01", "2020-12-31")    # never seen during training = independent test
CLIMATE = ["tmax_C", "tmin_C", "srad_MJm2", "rh_frac", "pet_mm"]


# ------------------------------------------------------------------ the network
class RunoffLSTM(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):                      # x: (batch, 365 days, n_features)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)   # use the memory after the last day


def metrics(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    nse = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    r = np.corrcoef(o, s)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    pbias = 100 * (s.sum() - o.sum()) / o.sum()
    return dict(NSE=round(nse, 3), KGE=round(kge, 3), PBIAS=round(pbias, 1), r=round(r, 3), n=int(m.sum()))


def experiment(rain_only, seeds):
    out_dir = LS / "experiment_rain_only" if rain_only else LS
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n######## LSTM experiment: {'rain only' if rain_only else 'rain + ERA5-Land climate'} ########")

    # ---------------------------------------------------------- 1. data
    obs = pd.read_csv(DP / "observed_daily.csv", parse_dates=["date"], index_col="date")
    df = obs[["P_sub1", "P_sub2", "P_sub3", "Q_mm"]].copy()
    features = ["P_sub1", "P_sub2", "P_sub3"]
    if not rain_only:
        clim = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"], index_col="date")
        for c in CLIMATE:
            df[c] = clim[c]
            features.append(c)
    doy = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)       # day of year as a circle:
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)       # 31 Dec and 1 Jan are neighbours
    features += ["sin_doy", "cos_doy"]
    df = df.loc["2001-01-01":"2020-12-31"]
    df[features] = df[features].interpolate(limit=3).fillna(0.0)     # tiny gaps in the inputs
    print("input features:", features)

    # normalise with mean/std of the TRAINING period only (no information from the future)
    tr = df.loc[TRAIN[0]:TRAIN[1]]
    mu, sd = tr[features].mean(), tr[features].std().replace(0, 1)
    q_mu, q_sd = tr["Q_mm"].mean(), tr["Q_mm"].std()
    json.dump({"features": features, "mean": mu.to_dict(), "std": sd.to_dict(), "q_mean": q_mu, "q_std": q_sd,
               "seq_len": SEQ_LEN}, open(out_dir / "lstm_scaler.json", "w"), indent=1)
    X = ((df[features] - mu) / sd).to_numpy(np.float32)
    y = ((df["Q_mm"] - q_mu) / q_sd).to_numpy(np.float32)
    dates = df.index

    def windows(start, end):
        """days in [start, end] that have an observed Q and a full 365-day history"""
        idx = np.where((dates >= start) & (dates <= end))[0]
        return np.array([t for t in idx if t >= SEQ_LEN - 1 and not np.isnan(y[t])])

    class SeqData(torch.utils.data.Dataset):
        def __init__(self, idx): self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            t = self.idx[i]
            return torch.from_numpy(X[t - SEQ_LEN + 1:t + 1]), torch.tensor(0.0 if np.isnan(y[t]) else y[t])

    def predict(model, idx):
        model.eval()
        res = []
        with torch.no_grad():
            for xb, _ in torch.utils.data.DataLoader(SeqData(idx), batch_size=512):
                res.append(model(xb).numpy())
        return np.concatenate(res) * q_sd + q_mu

    i_tr, i_va, i_te = windows(*TRAIN), windows(*VALID), windows(*TEST)
    i_all = np.arange(SEQ_LEN - 1, len(dates))
    q_obs = df["Q_mm"].to_numpy()
    print(f"training days {len(i_tr)}, validation days {len(i_va)}, test days {len(i_te)}")

    # ---------------------------------------------------------- 2. training (one network per seed)
    log, sims = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = RunoffLSTM(len(features))
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        loader = torch.utils.data.DataLoader(SeqData(i_tr), batch_size=BATCH, shuffle=True)
        best, best_state, t0 = -1e9, None, time.time()
        for ep in range(1, EPOCHS + 1):
            model.train(); tot = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)      # mean squared error
                loss.backward()                                    # gradients of the error
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tot += loss.item() * len(yb)          # nudge all weights
            nse_va = metrics(q_obs[i_va], predict(model, i_va))["NSE"]
            sched.step(-nse_va)
            log.append(dict(seed=seed, epoch=ep, train_mse=tot / len(i_tr), valid_NSE=nse_va))
            if nse_va > best:                                      # early stopping: remember the best epoch
                best, best_state = nse_va, {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 1 == 0:
                print(f"  network {seed}  epoch {ep:3d}  training MSE {tot/len(i_tr):.3f}  validation NSE {nse_va:.3f}"
                      f"  ({time.time()-t0:.0f} s)", flush=True)
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / f"lstm_seed{seed}.pt")
        sims.append(predict(model, i_all))
        print(f"  network {seed}: best validation NSE {best:.3f}")

    # ---------------------------------------------------------- 3. results
    sim_mm = np.clip(np.mean(sims, axis=0), 0, None)          # ensemble mean, no negative flow
    res = pd.DataFrame(index=dates[i_all])
    res["Q_obs_mm"] = q_obs[i_all]
    res["Q_lstm_mm"] = sim_mm
    for k, s in zip(seeds, sims):
        res[f"Q_lstm_seed{k}_mm"] = np.clip(s, 0, None)
    f = AREA_KM2 * 1e6 / 1000 / 86400                          # mm/day -> m3/s
    res["Q_obs_m3s"] = res["Q_obs_mm"] * f
    res["Q_lstm_m3s"] = res["Q_lstm_mm"] * f
    res.index.name = "date"
    res.to_csv(out_dir / "lstm_simulation.csv", float_format="%.4f")
    pd.DataFrame(log).to_csv(out_dir / "lstm_training_log.csv", index=False, float_format="%.4f")
    rows = []
    for name, (a, b) in {"training 2003-2010": TRAIN, "validation 2011-2012": VALID, "test 2013-2020": TEST}.items():
        r = res.loc[a:b]
        rows.append(dict(period=name, **metrics(r["Q_obs_m3s"], r["Q_lstm_m3s"])))
    met = pd.DataFrame(rows)
    met.to_csv(out_dir / "lstm_metrics.csv", index=False)
    print(met.to_string(index=False))


if __name__ == "__main__":
    start_log("06_lstm_model")
    experiment(rain_only=False, seeds=SEEDS)
    experiment(rain_only=True, seeds=SEEDS[:2])
    done("06_lstm_model")
