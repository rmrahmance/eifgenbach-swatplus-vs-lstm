"""
STEP 7 - Make all figures of the report (and compare SWAT+ with the LSTM)
==========================================================================
Every figure is saved as PNG in Models/figures. A figure is skipped when its
input files do not exist yet, so the script can be run after any step.
How to run: open this file in VS Code and press "Run Python File".
"""
import sys, pathlib, json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from _common import DATA, MODELS, DP, FIG, SW, LS, CALIB, VALID, start_log, done
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
C_OBS, C_SWAT, C_LSTM, C_RAIN, C_DEF = "#1f2937", "#d97706", "#2563eb", "#0ea5e9", "#9ca3af"
AREA = 23.885

sub = gpd.read_file(DATA / "wv_markusmuehle" / "wv_markusmuehle.shp").to_crs(25832).sort_values("name")
riv = gpd.read_file(DATA / "r_markusmuehle" / "r_markusmuehle" / "r_markusmuehle.shp").to_crs(25832)
r1 = gpd.read_file(DATA / "qgis_analysis" / "reach_R1.gpkg").to_crs(25832)
pts = gpd.read_file(DATA / "qgis_analysis" / "hms_elements.gpkg").to_crs(25832).set_index("name")
basin = sub.union_all()
SUBCOL = {"Subbasin-1": "#f59e0b", "Subbasin-2": "#10b981", "Subbasin-3": "#8b5cf6"}


def have(*paths):
    return all(pathlib.Path(p).exists() for p in paths)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


def metrics(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    nse = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    r = np.corrcoef(o, s)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    return dict(NSE=nse, KGE=kge, PBIAS=100 * (s.sum() - o.sum()) / o.sum(), r=r)


def map_frame(ax):
    xmin, ymin, xmax, ymax = basin.buffer(400).bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#cbd5e1")
    # scale bar 2 km + north arrow
    x0, y0 = xmin + 300, ymin + 300
    ax.plot([x0, x0 + 2000], [y0, y0], color="k", lw=3, solid_capstyle="butt")
    ax.text(x0 + 1000, y0 + 150, "2 km", ha="center", fontsize=8)
    ax.annotate("N", xy=(xmax - 450, ymax - 350), xytext=(xmax - 450, ymax - 1150), ha="center", fontsize=10,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))


def hillshade():
    with rasterio.open(DATA / "dem_markus_5m.tif") as d:
        z = d.read(1, masked=True).astype(float)
        ext = [d.bounds.left, d.bounds.right, d.bounds.bottom, d.bounds.top]
    hs = LightSource(azdeg=315, altdeg=40).hillshade(z.filled(np.nan), vert_exag=2, dx=5, dy=5)
    return z, hs, ext


# =============================================================== Fig 1  study area
def fig_study_area():
    z, hs, ext = hillshade()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.imshow(hs, extent=ext, cmap="gray", alpha=0.55)
    im = ax.imshow(z, extent=ext, cmap="terrain", alpha=0.55, vmin=120, vmax=360)
    for _, s in sub.iterrows():
        gpd.GeoSeries([s.geometry]).boundary.plot(ax=ax, color=SUBCOL[s["name"]], lw=2.2)
        c = s.geometry.representative_point()
        ax.text(c.x, c.y + 600 if s["name"] != "Subbasin-1" else c.y - 900, f"{s['name']}\n{s.geometry.area/1e6:.1f} km²", ha="center",
                fontsize=8.5, fontweight="bold", color="#111827",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SUBCOL[s["name"]], alpha=0.85))
    riv.clip(basin.buffer(30)).plot(ax=ax, color="#1d4ed8", lw=0.9)
    r1.plot(ax=ax, color="#1e3a8a", lw=2.4)
    ax.plot(pts.loc["SMAR"].geometry.x, pts.loc["SMAR"].geometry.y, "v", ms=11, color="#dc2626", mec="white", zorder=5)
    ax.plot(pts.loc["J1"].geometry.x, pts.loc["J1"].geometry.y, "o", ms=8, color="#111827", mec="white", zorder=5)
    ax.annotate("Gauge Markusmühle (SMAR)\n= catchment outlet", (pts.loc["SMAR"].geometry.x, pts.loc["SMAR"].geometry.y),
                xytext=(14, -12), textcoords="offset points", fontsize=8, ha="left", color="#dc2626", fontweight="bold")
    ax.annotate("Junction J1", (pts.loc["J1"].geometry.x, pts.loc["J1"].geometry.y), xytext=(8, -14),
                textcoords="offset points", fontsize=8)
    map_frame(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.set_label("Elevation [m a.s.l.]")
    ax.legend(handles=[Line2D([], [], color="#1d4ed8", lw=1, label="streams"),
                       Line2D([], [], color="#1e3a8a", lw=2.4, label="reach R1 (Eifgenbach)"),
                       Line2D([], [], marker="v", ls="", color="#dc2626", label="discharge gauge")],
              loc="upper left", fontsize=8, frameon=True)
    ax.set_title("Study area: Eifgenbach catchment to gauge Markusmühle (23.9 km²)")
    save(fig, "fig01_study_area.png")


# =============================================================== Fig 2  observed data
def fig_observed():
    d = pd.read_csv(DP / "observed_daily.csv", parse_dates=["date"], index_col="date").loc["2001":"2020"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6]))
    pm = d.P_basin.resample("MS").sum()
    a1.bar(pm.index + pd.Timedelta(days=15), pm, width=25, color=C_RAIN)
    a1.set_ylabel("Rain [mm/month]"); a1.set_title("Observed rainfall (RADOLAN, monthly sums) and discharge (gauge, daily means)")
    a1.invert_yaxis()
    a2.plot(d.index, d.Q_m3s, color=C_OBS, lw=0.6)
    na = d.Q_m3s.isna() & (d.index >= "2002-10-21")
    run = na.groupby((na != na.shift()).cumsum()).transform("size")
    miss = na & (run >= 5)                                           # show only gaps of 5 days or more
    a2.fill_between(d.index, 0, 16, where=miss, color="#fecaca", alpha=0.6, step="mid", label="gap in discharge data (>= 5 days)")
    a2.set_ylim(0, 16); a2.set_ylabel("Discharge Q [m³/s]")
    for (a, b), lab, col in [(("2001-01-01", "2004-12-31"), "warm-up", "#e5e7eb"), (CALIB, "calibration / training", "#fde68a"),
                             (VALID, "validation / test", "#bfdbfe")]:
        a2.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=0.35, lw=0)
        a2.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 15.2, lab, ha="center", fontsize=8.5, fontweight="bold")
    a2.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    save(fig, "fig02_observed_timeseries.png")

    # annual water balance
    y = d.groupby(d.index.year).agg(P=("P_basin", "sum"), Q=("Q_mm", "sum"), n=("Q_mm", "count"))
    y = y.loc[2005:2019]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    x = np.arange(len(y))
    ax.bar(x - 0.2, y.P, 0.4, color=C_RAIN, label="rainfall P (RADOLAN)")
    ax.bar(x + 0.2, y.Q * 365 / y.n, 0.4, color=C_OBS, label="runoff Q (gauge, gap-corrected)")
    ax.set_xticks(x, y.index); ax.set_ylabel("mm / year")
    for i, (p, q, n) in enumerate(zip(y.P, y.Q, y.n)):
        ax.text(i + 0.2, q * 365 / n + 15, f"{q*365/n/p:.2f}", ha="center", fontsize=7)
    ax.set_title("Annual rainfall and runoff (numbers = runoff coefficient Q/P)")
    ax.legend(ncol=2, loc="upper right")
    save(fig, "fig03_annual_water_balance.png")

    # rainfall of the 3 subbasins + monthly regime
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    m = d.groupby(d.index.month)[["P_sub1", "P_sub2", "P_sub3", "P_basin"]].mean() * \
        d.groupby(d.index.month).size().values[:, None] / 20
    for c_, col, lab in [("P_sub1", SUBCOL["Subbasin-1"], "Subbasin-1"), ("P_sub2", SUBCOL["Subbasin-2"], "Subbasin-2"),
                         ("P_sub3", SUBCOL["Subbasin-3"], "Subbasin-3")]:
        a.plot(m.index, m[c_], "o-", color=col, label=lab, ms=3)
    a.set_xticks(range(1, 13), list("JFMAMJJASOND")); a.set_ylabel("mm / month"); a.legend(fontsize=7.5)
    a.set_title("Mean monthly RADOLAN rainfall")
    q = d.groupby(d.index.month).Q_m3s.mean()
    b.bar(q.index, q, color=C_OBS, alpha=0.8); b.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    b.set_ylabel("m³/s"); b.set_title("Mean monthly discharge (regime)")
    save(fig, "fig04_monthly_regime.png")


# =============================================================== Fig 5-7  satellite inputs
WC_COL = {10: ("#006400", "Tree cover"), 20: ("#ffbb22", "Shrubland"), 30: ("#ffff4c", "Grassland"),
          40: ("#f096ff", "Cropland"), 50: ("#fa0000", "Built-up"), 60: ("#b4b4b4", "Bare / sparse"),
          80: ("#0064c8", "Permanent water"), 90: ("#0096a0", "Herbaceous wetland")}


def fig_landcover():
    with rasterio.open(DP / "landcover_worldcover_10m.tif") as s:
        a = s.read(1); b = s.bounds
    codes = sorted(WC_COL)
    cmap = ListedColormap([WC_COL[c][0] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], len(codes))
    a = np.ma.masked_equal(a, 0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax.imshow(a, extent=[b.left, b.right, b.bottom, b.top], cmap=cmap, norm=norm, interpolation="nearest")
    sub.boundary.plot(ax=ax, color="k", lw=1)
    map_frame(ax); ax.set_title("ESA WorldCover 2021 (10 m, Sentinel-1/2)")
    f = pd.read_csv(DP / "landcover_fractions.csv")
    present = [c for c in codes if c in set(f.class_code)]
    ax.legend(handles=[Patch(color=WC_COL[c][0], label=WC_COL[c][1]) for c in present], loc="lower right",
              fontsize=7.5, frameon=True)
    piv = f.pivot_table(index="subbasin", columns="class_code", values="percent", fill_value=0)
    piv = piv.loc[["Subbasin-1", "Subbasin-2", "Subbasin-3", "Basin"]]
    left = np.zeros(len(piv))
    for c in present:
        bx.barh(piv.index, piv[c], left=left, color=WC_COL[c][0], edgecolor="white", lw=0.5)
        for i, v in enumerate(piv[c]):
            if v > 7:
                bx.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5)
        left += piv[c].to_numpy()
    bx.set_xlim(0, 100); bx.set_xlabel("% of area"); bx.invert_yaxis(); bx.set_title("Land-cover shares")
    save(fig, "fig05_landcover.png")


def fig_soil():
    s = pd.read_csv(DP / "soil_profiles_swat.csv")
    fig, axs = plt.subplots(1, 4, figsize=(10, 3.4), sharey=True)
    for name, g in s.groupby("name"):
        z = np.r_[0, g.dp_mm.to_numpy()] / 10
        for ax, col, lab in zip(axs, ["sand", "clay", "awc", "ksat_mmh"],
                                ["Sand [%]", "Clay [%]", "Available water capacity [-]", "Sat. conductivity [mm/h]"]):
            v = g[col].to_numpy()
            ax.step(np.r_[v[0], v], z, where="pre", lw=2, label=f"{name} ({g.hyd_grp.iloc[0]})")
            ax.set_xlabel(lab)
    axs[0].invert_yaxis(); axs[0].set_ylabel("Depth [cm]")
    axs[0].legend(fontsize=7, title="soil (hydrologic group)", title_fontsize=7)
    fig.suptitle("Soil profiles derived from ISRIC SoilGrids (250 m) with the Saxton & Rawls (2006) pedotransfer functions",
                 fontsize=10, fontweight="bold")
    save(fig, "fig06_soil_profiles.png")


def fig_climate():
    c = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"], index_col="date").loc["2001":"2020"]
    o = pd.read_csv(DP / "observed_daily.csv", parse_dates=["date"], index_col="date").loc["2001":"2020"]
    m = c.groupby(c.index.month).mean()
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
    axs[0].fill_between(m.index, m.tmin_C, m.tmax_C, color="#fca5a5", alpha=0.6, label="Tmin - Tmax")
    axs[0].plot(m.index, m.tmean_C, color="#b91c1c", label="Tmean"); axs[0].set_ylabel("°C")
    axs[0].set_title("Air temperature"); axs[0].legend(fontsize=7.5)
    axs[1].plot(m.index, m.srad_MJm2, color="#f59e0b"); axs[1].set_ylabel("MJ/m²/day"); axs[1].set_title("Solar radiation")
    pm = o.P_basin.groupby(o.index.month).sum() / 20
    em = c.pet_mm.groupby(c.index.month).sum() / 20
    axs[2].bar(pm.index - 0.2, pm, 0.4, color=C_RAIN, label="rain (RADOLAN)")
    axs[2].bar(em.index + 0.2, em, 0.4, color="#16a34a", label="ET0 (FAO-56)")
    axs[2].set_ylabel("mm/month"); axs[2].set_title("Water supply vs demand"); axs[2].legend(fontsize=7.5)
    for ax in axs:
        ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    fig.suptitle("ERA5-Land / ERA5 climate of the basin (monthly means 2001-2020)", fontsize=10, fontweight="bold", y=1.03)
    save(fig, "fig07_climate_era5.png")


# =============================================================== Fig 8  HRUs + slope
def fig_hrus():
    h = pd.read_csv(SW / "hru_summary.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.05]))
    fig.subplots_adjust(wspace=0.62)
    with rasterio.open(DATA / "qgis_analysis" / "slope_deg.tif") as s:
        sl = np.tan(np.radians(s.read(1, masked=True))) * 100; b = s.bounds
    cl = np.ma.masked_array(np.digitize(sl.filled(0), [10, 25]), sl.mask)
    ax.imshow(cl, extent=[b.left, b.right, b.bottom, b.top], cmap=ListedColormap(["#fef3c7", "#fbbf24", "#b45309"]),
              interpolation="nearest")
    sub.boundary.plot(ax=ax, color="k", lw=1); map_frame(ax)
    ax.set_title("Slope classes used for the HRUs")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(["#fef3c7", "#fbbf24", "#b45309"],
              ["0-10 %", "10-25 %", "> 25 %"])], loc="upper left", frameon=True, fontsize=8)
    short = {"frst_teof": "forest", "past": "pasture", "urmd": "urban", "agrl": "crop", "rngb_teof": "shrub",
             "watr": "water", "wetn": "wetland", "bsvg": "bare"}
    h["label"] = (h.landuse.map(short).fillna(h.landuse) + " | " + h.soil.str.replace("SG_", "") + " | " +
                  h.slp.str.replace("25-9999", ">25") + " %")
    h = h.sort_values(["lsu", "arslp"], ascending=[True, False])
    cols = [SUBCOL[f"Subbasin-{k}"] for k in h.lsu]
    y = np.arange(len(h))
    bx.barh(y, h.arslp / 100, color=cols)
    bx.set_yticks(y, [f"{i}: {l}" for i, l in zip(h.id, h.label)], fontsize=6.8)
    bx.invert_yaxis(); bx.set_xlabel("HRU area [km²]"); bx.set_title(f"The {len(h)} HRUs (colour = subbasin)")
    bx.grid(axis="y", alpha=0)
    bx.legend(handles=[Patch(color=SUBCOL[k], label=k) for k in SUBCOL], loc="lower right", fontsize=7.5)
    save(fig, "fig08_hrus.png")


# =============================================================== Fig 9-13  SWAT+ results
def fig_calibration():
    runs = pd.read_csv(SW / "calibration" / "calibration_runs.csv")
    rng = pd.read_csv(SW / "calibration" / "parameter_ranges.csv")
    best = runs.loc[runs.KGE_cal.idxmax()]
    names = rng.parameter.tolist()
    fig, axs = plt.subplots(3, 5, figsize=(11, 6.2), sharey=True)
    for ax, p in zip(axs.flat, names):
        for it, col in [(1, "#9ca3af"), (2, C_SWAT)]:
            r = runs[runs.iteration == it]
            ax.scatter(r[p], r.KGE_cal, s=4, color=col, alpha=0.6, lw=0, label=f"iteration {it}")
        ax.scatter([best[p]], [best.KGE_cal], s=40, marker="*", color="#dc2626", zorder=5, label="best run")
        ax.set_title(p, fontsize=9); ax.set_ylim(max(-0.5, runs.KGE_cal.min()), runs.KGE_cal.max() + 0.03)
    axs.flat[-1].axis("off")
    axs.flat[0].legend(fontsize=7, loc="lower left", markerscale=2)
    for ax in axs[:, 0]:
        ax.set_ylabel("KGE (calibration)")
    fig.suptitle("Dotty plots: every dot is one SWAT+ run (parameter value vs. model quality)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig09_dotty_plots.png")

    s = pd.read_csv(SW / "calibration" / "sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    s = s.iloc[::-1]
    ax.barh(s.parameter, s.t_stat.abs(), color=np.where(s.p_value < 0.05, C_SWAT, "#d1d5db"))
    for i, (t, p) in enumerate(zip(s.t_stat, s.p_value)):
        ax.text(abs(t) + 0.3, i, f"p={p:.3f}", va="center", fontsize=7)
    ax.set_xlabel("|t-stat| (bigger = more sensitive)")
    ax.set_title("Global sensitivity of KGE to the parameters (orange: p < 0.05)")
    save(fig, "fig10_sensitivity.png")


def fig_swat_hydrographs():
    r = pd.read_csv(SW / "swatplus_simulation.csv", parse_dates=["date"], index_col="date").loc["2005":"2020"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for ax, (a, b), lab in [(axs[0], CALIB, "Calibration 2005-2012"), (axs[1], VALID, "Validation 2013-2020")]:
        x = r.loc[a:b]
        ax.fill_between(x.index, x.ppu95_low_m3s, x.ppu95_high_m3s, color=C_SWAT, alpha=0.25, lw=0, label="95PPU band")
        ax.plot(x.index, x.Q_obs_m3s, color=C_OBS, lw=0.7, label="observed")
        ax.plot(x.index, x.Q_swat_calibrated_m3s, color=C_SWAT, lw=0.7, label="SWAT+ calibrated")
        m = metrics(x.Q_obs_m3s, x.Q_swat_calibrated_m3s)
        ax.set_title(f"{lab}:  NSE = {m['NSE']:.2f}   KGE = {m['KGE']:.2f}   PBIAS = {m['PBIAS']:.1f} %")
        ax.set_ylabel("Q [m³/s]")
    axs[0].legend(ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, "fig11_swat_hydrograph.png")


def fig_water_balance():
    f = SW / "TxtInOut_calibrated" / "basin_wb_aa.txt"
    d = pd.read_csv(f, sep=r"\s+", skiprows=[0, 2], index_col=False)
    r = d.iloc[-1]
    comp = {"Evapo-\ntranspiration": r.et, "Surface\nrunoff": r.surq_gen, "Lateral flow\n(interflow)": r.latq,
            "Percolation to\ngroundwater": r.perc}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1, 1.2]))
    fig.subplots_adjust(wspace=0.35)
    ax.pie(list(comp.values()), labels=[f"{k}\n{v:.0f} mm" for k, v in comp.items()],
           colors=["#16a34a", "#ef4444", "#f59e0b", "#3b82f6"], startangle=90, wedgeprops=dict(width=0.45, ec="white"),
           textprops=dict(fontsize=8))
    ax.set_title(f"Where the {r.precip:.0f} mm/yr of rain goes\n(calibrated SWAT+, 2001-2020)")
    w = pd.read_csv(SW / "TxtInOut_calibrated" / "basin_wb_mon.txt", sep=r"\s+", skiprows=[0, 2], index_col=False)
    w = w[(w.yr >= 2005)]
    m = w.groupby("mon")[["precip", "et", "surq_gen", "latq", "perc", "sw_ave"]].mean()
    bx.bar(m.index, m.precip, color=C_RAIN, alpha=0.5, label="rain")
    bx.plot(m.index, m.et, "o-", color="#16a34a", ms=3, label="ET")
    bx.plot(m.index, m.surq_gen + m.latq, "o-", color="#ef4444", ms=3, label="surface + lateral flow")
    bx.plot(m.index, m.perc, "o-", color="#3b82f6", ms=3, label="percolation")
    bx.set_xticks(range(1, 13), list("JFMAMJJASOND")); bx.set_ylabel("mm/month"); bx.legend(fontsize=7.5)
    bx.set_title("Monthly water balance (calibrated)")
    save(fig, "fig12_swat_water_balance.png")


# =============================================================== Fig 14-19  LSTM + comparison
def fig_lstm_training():
    lg = pd.read_csv(LS / "lstm_training_log.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    for s, g in lg.groupby("seed"):
        a.plot(g.epoch, g.train_mse, lw=1, label=f"network {s}")
        b.plot(g.epoch, g.valid_NSE, lw=1)
    a.set_xlabel("epoch"); a.set_ylabel("training loss (MSE)"); a.set_title("Learning: error on training data")
    b.set_xlabel("epoch"); b.set_ylabel("NSE on validation years"); b.set_title("Checking: skill on 2011-2012")
    b.set_ylim(max(0, lg.valid_NSE.min()), 1)
    a.legend(fontsize=7)
    save(fig, "fig14_lstm_training.png")


def load_both():
    s = pd.read_csv(SW / "swatplus_simulation.csv", parse_dates=["date"], index_col="date")
    l = pd.read_csv(LS / "lstm_simulation.csv", parse_dates=["date"], index_col="date")
    d = s.join(l[["Q_lstm_m3s"]], how="left")
    return d


def fig_comparison():
    d = load_both()
    rows = []
    for per, (a, b) in {"Calibration / training\n2005-2012": CALIB, "Validation / test\n2013-2020": VALID}.items():
        x = d.loc[a:b]
        for mod, col in [("SWAT+", "Q_swat_calibrated_m3s"), ("LSTM", "Q_lstm_m3s")]:
            rows.append(dict(period=per, model=mod, **metrics(x.Q_obs_m3s, x[col])))
    m = pd.DataFrame(rows)
    m.assign(period=m.period.str.replace("\n", " ")).round(3).to_csv(MODELS / "comparison_metrics.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, k in zip(axs, ["NSE", "KGE", "PBIAS"]):
        piv = m.pivot(index="period", columns="model", values=k)[["SWAT+", "LSTM"]]
        x = np.arange(len(piv))
        for j, (mod, col) in enumerate([("SWAT+", C_SWAT), ("LSTM", C_LSTM)]):
            ax.bar(x + (j - 0.5) * 0.36, piv[mod], 0.36, color=col, label=mod)
            for xi, v in zip(x, piv[mod]):
                ax.text(xi + (j - 0.5) * 0.36, v + (0.02 if v >= 0 else -0.06) * (10 if k == "PBIAS" else 1),
                        f"{v:.2f}" if k != "PBIAS" else f"{v:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(x, piv.index, fontsize=7.5); ax.set_title(k + (" [%]" if k == "PBIAS" else ""))
        if k != "PBIAS":
            ax.set_ylim(min(0, piv.min().min() - 0.1), 1)
        ax.axhline(0, color="k", lw=0.6)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Model quality: SWAT+ (process-based) vs. LSTM (data-driven)", fontsize=10, fontweight="bold", y=1.10)
    save(fig, "fig15_metrics_comparison.png")

    # hydrograph of the test period + zoom on one flood season
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    x = d.loc[VALID[0]:VALID[1]]
    axs[0].plot(x.index, x.Q_obs_m3s, color=C_OBS, lw=0.7, label="observed")
    axs[0].plot(x.index, x.Q_swat_calibrated_m3s, color=C_SWAT, lw=0.7, label="SWAT+")
    axs[0].plot(x.index, x.Q_lstm_m3s, color=C_LSTM, lw=0.7, label="LSTM")
    axs[0].set_ylabel("Q [m³/s]"); axs[0].legend(ncol=3); axs[0].set_title("Validation / test period 2013-2020")
    win = x.Q_obs_m3s.rolling(120, center=True).sum().idxmax()
    z = d.loc[win - pd.Timedelta(days=75): win + pd.Timedelta(days=75)]
    rain = pd.read_csv(DP / "observed_daily.csv", parse_dates=["date"], index_col="date").P_basin.reindex(z.index)
    ax2 = axs[1].twinx()
    ax2.bar(z.index, rain, color=C_RAIN, alpha=0.5, width=1); ax2.invert_yaxis(); ax2.set_ylim(80, 0)
    ax2.set_ylabel("rain [mm/day]"); ax2.grid(False)
    axs[1].plot(z.index, z.Q_obs_m3s, color=C_OBS, lw=1.4, label="observed")
    axs[1].plot(z.index, z.Q_swat_calibrated_m3s, color=C_SWAT, lw=1.2, label="SWAT+")
    axs[1].plot(z.index, z.Q_lstm_m3s, color=C_LSTM, lw=1.2, label="LSTM")
    axs[1].set_ylabel("Q [m³/s]"); axs[1].set_title("Zoom: the wettest 5 months of the test period"); axs[1].legend()
    fig.tight_layout()
    save(fig, "fig16_hydrograph_comparison.png")

    # scatter + flow duration curve
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    lim = x[["Q_obs_m3s", "Q_swat_calibrated_m3s", "Q_lstm_m3s"]].max().max() * 1.05
    a.scatter(x.Q_obs_m3s, x.Q_swat_calibrated_m3s, s=4, color=C_SWAT, alpha=0.5, label="SWAT+")
    a.scatter(x.Q_obs_m3s, x.Q_lstm_m3s, s=4, color=C_LSTM, alpha=0.5, label="LSTM")
    a.plot([0, lim], [0, lim], "k--", lw=0.8); a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("observed Q [m³/s]"); a.set_ylabel("simulated Q [m³/s]"); a.set_title("Observed vs simulated (test)")
    a.legend(markerscale=3)
    for col, c_, lab in [("Q_obs_m3s", C_OBS, "observed"), ("Q_swat_calibrated_m3s", C_SWAT, "SWAT+"), ("Q_lstm_m3s", C_LSTM, "LSTM")]:
        v = np.sort(x.dropna(subset=["Q_obs_m3s"])[col].dropna().to_numpy())[::-1]
        b.plot(np.arange(1, len(v) + 1) / (len(v) + 1) * 100, v, color=c_, lw=1.3, label=lab)
    b.set_yscale("log"); b.set_xlabel("% of time the flow is equalled or exceeded"); b.set_ylabel("Q [m³/s]")
    b.set_title("Flow duration curve (test period)"); b.legend()
    save(fig, "fig17_scatter_fdc.png")

    # mean monthly regime
    obs_mask = d.Q_obs_m3s.notna()
    y = d[obs_mask].loc["2005":"2020"]
    mm = y.groupby(y.index.month)[["Q_obs_m3s", "Q_swat_calibrated_m3s", "Q_lstm_m3s"]].mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(mm.index, mm.Q_obs_m3s, "o-", color=C_OBS, label="observed")
    ax.plot(mm.index, mm.Q_swat_calibrated_m3s, "s-", color=C_SWAT, label="SWAT+")
    ax.plot(mm.index, mm.Q_lstm_m3s, "^-", color=C_LSTM, label="LSTM")
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND")); ax.set_ylabel("mean Q [m³/s]"); ax.legend()
    ax.set_title("Mean seasonal cycle 2005-2020")
    save(fig, "fig18_seasonal_cycle.png")


# =============================================================== diagrams (no data needed)
def box(ax, x, y, w, h, text, fc, ec="#334155", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, p, q, col="#334155", text=None, rad=0.0, dx=0.0, dy=0.12, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha=ha, fontsize=7.5, color=col)


def fig_workflow():
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.1, 4.6, 2.3, 1.1, "Your data folder\nDEM 5 m, subbasins, rivers,\ndischarge (15 min), RADOLAN", "#e0f2fe")
    box(ax, 0.1, 2.9, 2.3, 1.1, "Satellite / reanalysis data\nWorldCover, SoilGrids,\nERA5-Land / ERA5 (internet)", "#dcfce7")
    box(ax, 2.9, 4.6, 2.0, 1.1, "Step 1\nprepare observed data\n01_prepare_observed_data.py", "#f1f5f9")
    box(ax, 2.9, 2.9, 2.0, 1.1, "Steps 2-3\ndownload + process\n02_*.py, 03_*.py", "#f1f5f9")
    box(ax, 5.5, 4.0, 2.0, 1.2, "Step 4: build SWAT+\nHRUs, routing, soils,\nweather → TxtInOut\n04_build_*.py", "#fef3c7")
    box(ax, 5.5, 2.3, 2.0, 1.2, "Step 5: calibrate +\nvalidate SWAT+\n(400 runs, SUFI-2 style)\n05_calibrate_*.py", "#fef3c7")
    box(ax, 5.5, 0.4, 2.0, 1.2, "Step 6: train LSTM\n(365-day memory,\n3-network ensemble)\n06_lstm_model.py", "#dbeafe")
    box(ax, 8.0, 2.3, 1.9, 1.4, "Step 7: compare\nNSE, KGE, PBIAS,\nhydrographs, FDC\n07_make_figures.py", "#ede9fe")
    box(ax, 8.0, 0.4, 1.9, 1.2, "Step 8: this report\n(Word document)", "#fce7f3")
    arrow(ax, (2.4, 5.15), (2.9, 5.15)); arrow(ax, (2.4, 3.45), (2.9, 3.45))
    ax.plot([5.2, 5.2], [1.0, 4.6], color="#334155", lw=2.2)                  # "bus" of prepared inputs
    ax.plot([4.9, 5.2], [5.15, 4.6], color="#334155", lw=1.3); ax.plot([4.9, 5.2], [3.45, 3.45], color="#334155", lw=1.3)
    ax.text(5.08, 2.6, "model inputs", rotation=90, ha="center", va="center", fontsize=7.5, color="#334155",
            bbox=dict(fc="white", ec="none", pad=1))
    arrow(ax, (5.2, 4.6), (5.5, 4.6)); arrow(ax, (5.2, 1.0), (5.5, 1.0))
    arrow(ax, (6.5, 4.0), (6.5, 3.5))
    arrow(ax, (7.5, 2.9), (8.0, 3.0)); arrow(ax, (7.5, 1.0), (8.0, 2.5)); arrow(ax, (8.95, 2.3), (8.95, 1.6))
    ax.text(5, 5.9, "Complete workflow (each step = one Python script in Models/scripts, run in VS Code; SWAT+ 4.0.2 installed)", ha="center",
            fontsize=10.5, fontweight="bold")
    save(fig, "fig00_workflow.png")


def fig_swat_concept():
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    # 1. HRU composition
    ax.text(1.6, 4.35, "1. An HRU = one combination of", ha="center", fontsize=9.5, fontweight="bold")
    for i, (t, c) in enumerate([("land use\n(forest,\npasture ...)", "#bbf7d0"), ("soil\n(silt loam,\nloam ...)", "#fde68a"),
                                ("slope\n(0-10, 10-25,\n>25 %)", "#e5e7eb")]):
        box(ax, 0.05 + i * 1.12, 3.0, 0.92, 1.0, t, c, fs=7.3)
        if i < 2:
            ax.text(1.03 + i * 1.12, 3.5, "×", fontsize=13, ha="center", va="center")
    box(ax, 0.35, 1.9, 2.5, 0.75, "HRU\n(Hydrologic Response Unit)", "#fff7ed", bold=True, fs=8)
    arrow(ax, (1.6, 3.0), (1.6, 2.65))
    ax.text(1.6, 1.25, "Each HRU calculates its own daily\nwater balance; HRUs in one subbasin\nare added up (no position inside it).",
            ha="center", fontsize=7.6, color="#475569")
    # 2. water balance of an HRU
    ax.text(5.15, 4.35, "2. Daily water balance of one HRU", ha="center", fontsize=9.5, fontweight="bold")
    ax.add_patch(plt.Rectangle((3.6, 1.3), 3.1, 1.5, fc="#fde68a", ec="#92400e", lw=1))
    ax.add_patch(plt.Rectangle((3.6, 0.2), 3.1, 0.6, fc="#bfdbfe", ec="#1e40af", lw=1))
    ax.text(5.15, 2.0, "soil water\n(AWC, ESCO, EPCO, PERCO)", ha="center", va="center", fontsize=8)
    ax.text(5.15, 0.5, "shallow aquifer (ALPHA, FLO_MIN)", ha="center", va="center", fontsize=8)
    arrow(ax, (4.0, 4.0), (4.0, 2.85), "#0284c7", "rain", dx=0.08, dy=0.2, ha="left")
    arrow(ax, (4.8, 2.85), (4.8, 4.0), "#16a34a", "evapo-\ntranspiration", dx=0.08, dy=0.15, ha="left")
    arrow(ax, (5.6, 2.95), (7.3, 2.95), "#dc2626", "surface runoff (CN2)")
    arrow(ax, (6.7, 1.9), (7.4, 1.9), "#f59e0b", "lateral flow", dx=-0.28, ha="left")
    arrow(ax, (5.15, 1.3), (5.15, 0.8), "#1d4ed8", "percolation", dx=0.08, dy=-0.05, ha="left")
    arrow(ax, (6.7, 0.5), (7.4, 0.5), "#1d4ed8", "baseflow", dx=-0.28, ha="left")
    # 3. routing
    ax.text(8.65, 4.35, "3. Routing to the outlet", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 7.5, 3.35, 1.0, 0.6, "Sub-2", "#d1fae5", fs=8); box(ax, 8.8, 3.35, 1.0, 0.6, "Sub-3", "#ede9fe", fs=8)
    box(ax, 8.15, 2.1, 1.0, 0.6, "Sub-1\n(reach R1)", "#fef3c7", fs=7.5)
    box(ax, 8.15, 0.9, 1.0, 0.6, "gauge\nSMAR", "#fee2e2", fs=7.5)
    arrow(ax, (8.0, 3.35), (8.45, 2.7)); arrow(ax, (9.3, 3.35), (8.85, 2.7)); arrow(ax, (8.65, 2.1), (8.65, 1.5))
    ax.text(8.65, 3.05, "J1", fontsize=8, ha="center", fontweight="bold")
    save(fig, "fig13_swat_concept.png")


def fig_lstm_concept():
    fig, ax = plt.subplots(figsize=(10, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    ax.text(5, 3.4, "How the LSTM predicts discharge of day t", ha="center", fontsize=10.5, fontweight="bold")
    days = ["t-364", "t-363", "...", "t-2", "t-1", "t"]
    for i, d in enumerate(days):
        x = 0.4 + i * 1.25
        if d != "...":
            box(ax, x, 0.35, 0.95, 0.7, f"day {d}\nrain, T,\nET0, ...", "#e0f2fe", fs=7)
            box(ax, x, 1.55, 0.95, 0.7, "LSTM\ncell", "#dbeafe", bold=True, fs=8)
            arrow(ax, (x + 0.47, 1.05), (x + 0.47, 1.55))
        else:
            ax.text(x + 0.47, 1.9, "• • •", ha="center", fontsize=14)
        if i < len(days) - 1:
            arrow(ax, (x + 0.95, 1.9), (x + 1.25, 1.9), "#7c3aed")
    ax.text(3.6, 2.45, "memory (cell state) is passed from day to day  →  it can 'remember' wet soils and groundwater",
            ha="center", fontsize=7.8, color="#7c3aed")
    box(ax, 7.95, 2.6, 1.6, 0.6, "Q(t) [m³/s]", "#fef3c7", bold=True)
    arrow(ax, (7.4, 2.25), (8.3, 2.6))
    save(fig, "fig19_lstm_concept.png")


if __name__ == "__main__":
    start_log("07_make_figures")
    fig_workflow(); fig_swat_concept(); fig_lstm_concept()
    fig_study_area(); fig_observed()
    if have(DP / "landcover_fractions.csv"):
        fig_landcover()
    if have(DP / "soil_profiles_swat.csv"):
        fig_soil()
    if have(DP / "climate_daily.csv"):
        fig_climate()
    if have(SW / "hru_summary.csv"):
        fig_hrus()
    if have(SW / "calibration" / "calibration_runs.csv"):
        fig_calibration(); fig_swat_hydrographs(); fig_water_balance()
    if have(LS / "lstm_training_log.csv"):
        fig_lstm_training()
    if have(SW / "swatplus_simulation.csv", LS / "lstm_simulation.csv"):
        fig_comparison()
    done("07_make_figures")
