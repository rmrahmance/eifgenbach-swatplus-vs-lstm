"""
STEP 5 - Calibrate and validate SWAT+ (SUFI-2 style, written in plain Python)
==============================================================================
Calibration = adjusting uncertain model parameters until simulated discharge matches
the observed discharge in the CALIBRATION period (2005-2012).
Validation  = running the calibrated model in a period it has never "seen"
(2013-2020) and checking that it is still good.

Method (the same idea as SUFI-2 in SWAT-CUP):
  iteration 1 : 250 parameter sets drawn by Latin-Hypercube sampling over wide ranges
  iteration 2 : ranges narrowed around the best 10 % of iteration 1, 150 new sets
  best set    : the run with the highest KGE in the calibration period
  uncertainty : 95PPU band (2.5-97.5 % of the best runs), p-factor and r-factor
  sensitivity : multiple regression of KGE on the parameters (t-stat, p-value)
SWAT+ applies the changes itself through its own file  calibration.cal.

How to run: open this file in VS Code and press the Run button (about 45 minutes on a laptop,
depending on the number of processor cores). The numbers of runs can be changed below (N1, N2).
"""
import os, shutil, subprocess, time, tempfile, pathlib
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import qmc, t as t_dist
from _common import MODELS, DP, SW, CALIB, VALID, start_log, done

N1, N2 = 250, 150                        # runs in iteration 1 and iteration 2
TXT = SW / "project" / "Scenarios" / "Default" / "TxtInOut"
CAL = SW / "calibration"
CAL.mkdir(exist_ok=True)
EXE = "swatplus.exe"
# the many test runs are made in a local temporary folder (fast, and not synchronised by Sciebo/OneDrive)
WORK = pathlib.Path(tempfile.gettempdir()) / "w4220_swat_calibration"
WORK.mkdir(exist_ok=True)
N_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 1))   # parallel SWAT+ runs

#            name       change   min     max    meaning (SWAT2012 name)
PARAMS = [("cn2",       "pctchg", -25.0,  15.0, "SCS curve number: how much rain becomes surface runoff (CN2)"),
          ("esco",      "absval",  0.50,  1.00, "soil evaporation compensation factor (ESCO)"),
          ("epco",      "absval",  0.30,  1.00, "plant uptake compensation factor (EPCO)"),
          ("awc",       "pctchg", -30.0,  30.0, "available water capacity of the soil (SOL_AWC)"),
          ("perco",     "absval",  0.10,  0.95, "percolation coefficient: water leaving the soil profile downward"),
          ("latq_co",   "absval",  0.00,  0.80, "lateral (interflow) coefficient"),
          ("lat_ttime", "absval",  0.50,  10.0, "lateral flow travel time [days] (LAT_TTIME)"),
          ("petco",     "absval",  0.85,  1.15, "potential evapotranspiration multiplier"),
          ("alpha",     "absval",  0.005, 0.30, "baseflow recession constant [1/day] (ALPHA_BF)"),
          ("flo_min",   "absval",  0.00,  8.00, "water-table height needed before baseflow starts [m] (GWQMN)"),
          ("revap_co",  "absval",  0.02,  0.20, "groundwater 'revap' coefficient (GW_REVAP)"),
          ("deep_seep", "absval",  0.00,  0.20, "fraction of recharge lost to the deep aquifer (RCHRG_DP)"),
          ("surlag",    "absval",  0.50,  8.00, "surface runoff lag coefficient (SURLAG)"),
          ("chn",       "absval",  0.025, 0.15, "Manning's n of the channels (CH_N2)")]
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[2] for p in PARAMS])
HI = np.array([p[3] for p in PARAMS])

obs = pd.read_csv(DP / "observed_daily.csv", parse_dates=["date"], index_col="date")["Q_m3s"]


# ------------------------------------------------------------------ helpers
def metrics(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    nse = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    r = np.corrcoef(o, s)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    pbias = 100 * (s.sum() - o.sum()) / o.sum()
    return dict(NSE=nse, KGE=kge, PBIAS=pbias, r=r)


def write_cal(folder, values):
    lines = [f"{n:<12}{PARAMS[i][1]:<10}{v:14.6f}  0  0  0  0  0  0  0  0" for i, (n, v) in enumerate(zip(NAMES, values))]
    head = "calibration.cal: written by 05_calibrate_validate_swatplus.py\n" + f"{len(lines)}\n" + \
           "NAME        CHG_TYP            VAL  CONDS LYR1 LYR2 YEAR1 YEAR2 DAY1 DAY2 OBJ_TOT\n"
    (folder / "calibration.cal").write_text(head + "\n".join(lines) + "\n")


def read_flow(folder):
    d = pd.read_csv(folder / "channel_sd_day.txt", sep=r"\s+", skiprows=[0, 2],
                    usecols=["mon", "day", "yr", "gis_id", "flo_out"])
    d = d[d.gis_id == 1]                                   # channel 1 = outlet at gauge SMAR
    d.index = pd.to_datetime(dict(year=d.yr, month=d.mon, day=d.day))
    return d["flo_out"]


def prepare_worker(k):
    w = WORK / f"worker{k}"
    if w.exists():
        shutil.rmtree(w)
    shutil.copytree(TXT, w, ignore=shutil.ignore_patterns("*.txt", "*.out", "*.csv"))
    cio = (w / "file.cio").read_text().splitlines()
    for i, line in enumerate(cio):          # tell SWAT+ to read calibration.cal
        if line.startswith("chg"):
            parts = line.split()
            parts[2] = "calibration.cal"
            cio[i] = "chg               " + "  ".join(f"{p:<16}" for p in parts[1:])
    (w / "file.cio").write_text("\n".join(cio) + "\n")
    return w


def run_one(args):
    k, values = args
    w = WORK / f"worker{k}"
    write_cal(w, values)
    r = subprocess.run([str(w / EXE)], cwd=w, capture_output=True, text=True)
    if "Execution successfully completed" not in r.stdout:
        return None
    return read_flow(w).to_numpy(np.float32)


def run_batch(X, label):
    sims, t0 = [None] * len(X), time.time()
    jobs = [(i % N_WORKERS, X[i]) for i in range(len(X))]
    with Pool(N_WORKERS) as pool:
        for start in range(0, len(jobs), N_WORKERS):
            out = pool.map(run_one, [(j, jobs[start + j][1]) for j in range(min(N_WORKERS, len(jobs) - start))])
            for j, o in enumerate(out):
                sims[start + j] = o
            if (start // N_WORKERS) % 5 == 0:
                print(f"  {label}: {start + len(out)}/{len(X)} runs ({time.time() - t0:.0f} s)", flush=True)
    return sims


def evaluate(sims, X, it, dates):
    rows = []
    for i, s in enumerate(sims):
        row = dict(iteration=it, **dict(zip(NAMES, X[i])))
        if s is not None:
            ser = pd.Series(s, index=dates)
            for per, (a, b) in {"cal": CALIB, "val": VALID}.items():
                for k, v in metrics(obs.loc[a:b], ser.loc[a:b].reindex(obs.loc[a:b].index)).items():
                    row[f"{k}_{per}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    start_log("05_calibrate_validate_swatplus")
    print(f"{N_WORKERS} SWAT+ runs in parallel, working folder {WORK}")
    for old in CAL.glob("worker*"):                     # left-overs of an earlier, interrupted run
        shutil.rmtree(old, ignore_errors=True)
    # ------------------------------------------------------------------ 0. default run (no changes)
    for k in range(N_WORKERS):
        prepare_worker(k)
    w0 = WORK / "worker0"
    (w0 / "calibration.cal").unlink(missing_ok=True)
    subprocess.run([str(w0 / EXE)], cwd=w0, capture_output=True)
    q_default = read_flow(w0)
    dates = q_default.index
    print("default run:", {k: round(v, 3) for k, v in metrics(obs.loc[CALIB[0]:CALIB[1]], q_default.reindex(obs.loc[CALIB[0]:CALIB[1]].index)).items()})

    # ------------------------------------------------------------------ 1. iteration 1 (wide ranges)
    X1 = qmc.scale(qmc.LatinHypercube(d=len(PARAMS), seed=42).random(N1), LO, HI)
    S1 = run_batch(X1, "iteration 1")
    R1 = evaluate(S1, X1, 1, dates)

    # ------------------------------------------------------------------ 2. iteration 2 (narrowed ranges)
    best = R1.nlargest(max(10, N1 // 10), "KGE_cal")
    lo2 = np.maximum(LO, best[NAMES].min().to_numpy() - 0.05 * (HI - LO))
    hi2 = np.minimum(HI, best[NAMES].max().to_numpy() + 0.05 * (HI - LO))
    X2 = qmc.scale(qmc.LatinHypercube(d=len(PARAMS), seed=7).random(N2), lo2, hi2)
    S2 = run_batch(X2, "iteration 2")
    R2 = evaluate(S2, X2, 2, dates)

    runs = pd.concat([R1, R2], ignore_index=True)
    runs.insert(0, "run", np.arange(1, len(runs) + 1))
    runs.to_csv(CAL / "calibration_runs.csv", index=False, float_format="%.5f")
    S = S1 + S2
    pd.DataFrame({"parameter": NAMES, "change_type": [p[1] for p in PARAMS], "meaning": [p[4] for p in PARAMS],
                  "iter1_min": LO, "iter1_max": HI, "iter2_min": lo2, "iter2_max": hi2}).to_csv(
        CAL / "parameter_ranges.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------ 3. best parameter set
    ib = int(runs["KGE_cal"].idxmax())
    xb = runs.loc[ib, NAMES].to_numpy(float)
    q_best = pd.Series(S[ib], index=dates)
    pd.DataFrame({"parameter": NAMES, "change_type": [p[1] for p in PARAMS], "best_value": xb,
                  "meaning": [p[4] for p in PARAMS]}).to_csv(CAL / "best_parameters.csv", index=False, float_format="%.4f")
    write_cal(CAL, xb)                                  # calibrated calibration.cal

    # calibrated model folder = TxtInOut + calibration.cal
    CALTXT = SW / "TxtInOut_calibrated"
    if CALTXT.exists():
        shutil.rmtree(CALTXT)
    shutil.copytree(WORK / "worker0", CALTXT)
    write_cal(CALTXT, xb)
    subprocess.run([str(CALTXT / EXE)], cwd=CALTXT, capture_output=True)

    # ------------------------------------------------------------------ 4. uncertainty (95PPU) + sensitivity
    ok = runs.dropna(subset=["KGE_cal"])
    behav = ok.nlargest(max(20, len(ok) // 20), "KGE_cal").index      # best 5 % = "behavioural" runs
    M = np.vstack([S[i] for i in behav])
    lo95, hi95 = np.percentile(M, 2.5, axis=0), np.percentile(M, 97.5, axis=0)

    res = pd.DataFrame({"Q_obs_m3s": obs.reindex(dates), "Q_swat_default_m3s": q_default.to_numpy(),
                        "Q_swat_calibrated_m3s": q_best.to_numpy(), "ppu95_low_m3s": lo95, "ppu95_high_m3s": hi95},
                       index=dates)
    res.index.name = "date"
    res.loc["2001-01-01":].to_csv(SW / "swatplus_simulation.csv", float_format="%.4f")

    rows = []
    for name, (a, b) in {"calibration 2005-2012": CALIB, "validation 2013-2020": VALID}.items():
        r = res.loc[a:b].dropna(subset=["Q_obs_m3s"])
        inside = ((r.Q_obs_m3s >= r.ppu95_low_m3s) & (r.Q_obs_m3s <= r.ppu95_high_m3s)).mean()
        rfac = (r.ppu95_high_m3s - r.ppu95_low_m3s).mean() / r.Q_obs_m3s.std()
        for lab, col in [("default", "Q_swat_default_m3s"), ("calibrated", "Q_swat_calibrated_m3s")]:
            m = metrics(r.Q_obs_m3s, r[col])
            rows.append(dict(period=name, model=f"SWAT+ {lab}", **{k: round(v, 3) for k, v in m.items()},
                             p_factor=round(inside, 2) if lab == "calibrated" else None,
                             r_factor=round(rfac, 2) if lab == "calibrated" else None))
    met = pd.DataFrame(rows)
    met.to_csv(SW / "swatplus_metrics.csv", index=False)
    print(met.to_string(index=False))

    # global sensitivity (SUFI-2): regress KGE on the parameter values
    Xs = (ok[NAMES] - ok[NAMES].mean()) / ok[NAMES].std()
    A = np.column_stack([np.ones(len(Xs)), Xs.to_numpy()])
    y = ok["KGE_cal"].to_numpy()
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    s2 = resid @ resid / (len(y) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    tstat = beta / se
    pval = 2 * (1 - t_dist.cdf(np.abs(tstat), len(y) - A.shape[1]))
    sens = pd.DataFrame({"parameter": NAMES, "t_stat": tstat[1:], "p_value": pval[1:]})
    sens["abs_t"] = sens.t_stat.abs()
    sens.sort_values("abs_t", ascending=False).drop(columns="abs_t").to_csv(CAL / "sensitivity.csv", index=False, float_format="%.4f")
    print(sens.sort_values("abs_t", ascending=False).to_string(index=False))
    for k in range(N_WORKERS):
        shutil.rmtree(WORK / f"worker{k}", ignore_errors=True)
    done("05_calibrate_validate_swatplus")


if __name__ == "__main__":
    main()
