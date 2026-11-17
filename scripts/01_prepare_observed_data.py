"""
STEP 1 - Prepare the observed data (discharge + radar rainfall)
================================================================
Input  (from the W4220 data folder):
  SMAR_15min/SMAR_15min.csv                       15-min discharge at gauge Markusmühle [m3/s]
  wv_markusmuehle_basin/.../..tzg..csv            hourly RADOLAN rainfall, whole basin [mm/h]
  wv_markusmuehle/.../..tzg..csv                  hourly RADOLAN rainfall, 3 subbasins [mm/h]
Output (Models/data_processed):
  discharge_daily.csv, discharge_hourly.csv, rain_daily.csv, rain_hourly.csv,
  observed_daily.csv (everything on one daily table), data_quality.txt
How to run: open this file in VS Code and press "Run Python File" (no settings needed).
"""
import numpy as np
import pandas as pd
from _common import DATA, DP as OUT, AREA_KM2, start_log, done

start_log("01_prepare_observed_data")
TZG = "radolan_hourly_reproc_2017_002_timeseries_tzg"

# ---------- 1. read discharge (15 min) ----------
q = pd.read_csv(DATA / "SMAR_15min" / "SMAR_15min.csv", sep=";", parse_dates=["Datum"],
                index_col="Datum")["Wert"].rename("Q_m3s")
q = pd.to_numeric(q, errors="coerce")
q[q < 0] = np.nan                                  # negative discharge is impossible

# regular 15-min time axis -> gaps become visible as NaN
full = pd.date_range(q.index.min().ceil("D"), q.index.max().floor("D") - pd.Timedelta("15min"), freq="15min")
q15 = q[~q.index.duplicated()].reindex(full)

# fill SHORT gaps (<= 6 h = 24 steps) by straight-line interpolation; longer gaps stay empty
na = q15.isna()
run_id = (na != na.shift()).cumsum()
run_len = na.groupby(run_id).transform("size")
q15_filled = q15.interpolate(limit_area="inside")
q15_filled[na & (run_len > 24)] = np.nan
n_filled = int((na & q15_filled.notna()).sum())

# hourly and daily means; a day is only kept if >= 80 % of its 96 values exist
qh = q15_filled.resample("h").mean()
cnt = q15_filled.resample("D").count()
qd = q15_filled.resample("D").mean().where(cnt >= 0.8 * 96)

# ---------- 2. read RADOLAN rainfall (hourly sums, time = end of hour, UTC) ----------
rb = pd.read_csv(DATA / "wv_markusmuehle_basin" / TZG / f"{TZG}..csv", sep=";",
                 parse_dates=["Datum"], index_col="Datum").rename(columns={"markus": "P_basin"})
rs = pd.read_csv(DATA / "wv_markusmuehle" / TZG / f"{TZG}..csv", sep=";",
                 parse_dates=["Datum"], index_col="Datum")
rs.columns = ["P_sub1", "P_sub2", "P_sub3"]
rain_h = rb.join(rs).apply(pd.to_numeric, errors="coerce")
rain_h[rain_h < 0] = np.nan
# the stamp HH:50 marks the END of the hourly sum -> label each hour by its start hour
rain_h.index = rain_h.index.floor("h")
rain_h = rain_h[~rain_h.index.duplicated()]
rain_h = rain_h.reindex(pd.date_range(rain_h.index.min(), rain_h.index.max(), freq="h"))
cnt_r = rain_h["P_basin"].resample("D").count()
rain_d = rain_h.resample("D").sum(min_count=1).where(cnt_r >= 22)   # allow 2 missing hours

# ---------- 3. combine to one daily table ----------
area_km2 = AREA_KM2
daily = rain_d.join(qd.rename("Q_m3s"), how="left")
daily["Q_mm"] = daily["Q_m3s"] * 86400 / (area_km2 * 1e6) * 1000   # m3/s -> mm/day over the basin
daily.index.name = "date"

qh.rename("Q_m3s").to_frame().rename_axis("time").to_csv(OUT / "discharge_hourly.csv", float_format="%.4f")
qd.rename("Q_m3s").to_frame().rename_axis("date").to_csv(OUT / "discharge_daily.csv", float_format="%.4f")
rain_h.rename_axis("time").to_csv(OUT / "rain_hourly.csv", float_format="%.2f")
rain_d.rename_axis("date").to_csv(OUT / "rain_daily.csv", float_format="%.2f")
daily.to_csv(OUT / "observed_daily.csv", float_format="%.4f")

# ---------- 4. quality report ----------
both = daily.dropna(subset=["P_basin", "Q_mm"])
yrs = both.groupby(both.index.year).agg(P=("P_basin", "sum"), Q=("Q_mm", "sum"), n=("Q_mm", "size"))
yrs = yrs[yrs.n > 330]
lines = [
    f"Discharge 15-min values: {q15.size}, missing: {q15.isna().sum()} ({q15.isna().mean()*100:.2f} %)",
    f"Filled by interpolation (gaps <= 6 h): {n_filled} values",
    f"Discharge daily: {qd.index.min().date()} .. {qd.index.max().date()}, valid days {qd.notna().sum()} of {qd.size}",
    f"Q daily  mean {qd.mean():.3f}  median {qd.median():.3f}  min {qd.min():.3f}  max {qd.max():.3f} m3/s",
    f"Q 15-min max {q15.max():.2f} m3/s on {q15.idxmax()}",
    f"Rain hourly: {rain_h.index.min()} .. {rain_h.index.max()}, missing hours {rain_h['P_basin'].isna().sum()}",
    f"Rain mean annual (basin, full years): {rain_d['P_basin'].groupby(rain_d.index.year).sum().loc[2001:2024].mean():.0f} mm",
    f"Max daily rain {rain_d['P_basin'].max():.1f} mm on {rain_d['P_basin'].idxmax().date()}",
    f"Max hourly rain {rain_h['P_basin'].max():.1f} mm on {rain_h['P_basin'].idxmax()}",
    f"Mean runoff coefficient Q/P (years with >330 valid days): {(yrs.Q.sum()/yrs.P.sum()):.2f}",
    "", "Year  P[mm]  Q[mm]  Q/P", *[f"{y}  {r.P:6.0f} {r.Q:6.0f}  {r.Q/r.P:.2f}" for y, r in yrs.iterrows()],
]
(OUT / "data_quality.txt").write_text("\n".join(lines))
print("\n".join(lines))
done("01_prepare_observed_data")
