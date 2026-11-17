"""
STEP 4 - Build the SWAT+ model (from GIS + satellite data to the TxtInOut folder)
=================================================================================
This script does what QSWAT+ (the QGIS plug-in) and the SWAT+ Editor normally do
by mouse-clicks, but in Python so every step is visible and repeatable:

  A. put all maps on one 10 m grid (subbasins, elevation, slope, land cover, soil)
  B. cut each subbasin into HRUs  (Hydrologic Response Unit = land use x soil x slope class)
  C. describe subbasins, channels, aquifers and the routing (which object drains into which)
  D. write the soils (from OpenLandMap) and the weather (RADOLAN + ERA5-Land) in SWAT+ format
  E. let the official SWAT+ Editor code (tools/swatplus_editor_api) create the project
     database and write all SWAT+ input text files  ->  SWATplus/TxtInOut
  F. run SWAT+ once with default parameters

How to run: open this file in VS Code and press "Run Python File" (takes about 1 minute).
"""
import sys, os, shutil, sqlite3, subprocess, platform, glob, pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from pyproj import Transformer
from _common import DATA, MODELS, DP, SW, SWATPLUS_HOME, start_log, done

start_log("04_build_swatplus_project")
TOOLS = SW / "tools"
PROJ = SW / "project"                    # SWAT+ project folder (database + Scenarios/Default/TxtInOut)
WEA = PROJ / "weather"                   # weather files written by this script
sys.path.insert(0, str(TOOLS / "swatplus_editor_api"))


def find_swat_exe():
    """the SWAT+ model executable: first the one installed with SWAT+ 4.0.2, otherwise the one in tools/"""
    if platform.system() == "Windows":
        roots = [SWATPLUS_HOME, pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
                 pathlib.Path(os.environ.get("ProgramFiles", "C:/Program Files"))]
        for root in roots:
            hits = [h for h in glob.glob(str(root / "**" / "swat_exe" / "*.exe"), recursive=True)
                    if "rel" in os.path.basename(h).lower() and "debug" not in h.lower()]
            if hits:
                return pathlib.Path(sorted(hits)[-1])
        return TOOLS / "swatplus-62.0.1-windows.exe"
    return TOOLS / "swatplus-62.0.1-linux"


def find_datasets_db():
    """the SWAT+ parameter database installed with SWAT+ (falls back to the copy in tools/)"""
    inst = SWATPLUS_HOME / "Databases" / "swatplus_datasets.sqlite"
    return inst if inst.exists() else TOOLS / "swatplus_datasets.sqlite"


# ------------------------------------------------------------------ settings
CELL = 10.0                              # grid size [m]
# ESA WorldCover class  ->  SWAT+ plant / urban code (datasets database names)
LU_MAP = {10: "frst_teof",   # tree cover          -> mixed forest, temperate oceanic zone
          20: "rngb_teof",   # shrubland           -> range brush
          30: "past",        # grassland           -> pasture / meadow
          40: "agrl",        # cropland            -> generic agriculture
          50: "urmd",        # built-up            -> urban, medium density
          60: "bsvg",        # bare / sparse veg.  -> barren
          80: "watr",        # permanent water
          90: "wetn"}        # herbaceous wetland  -> wetland
SLOPE_BANDS = [(0, 10, "0-10"), (10, 25, "10-25"), (25, 9999, "25-9999")]   # slope classes [%]
THR_LU, THR_SOIL, THR_SLP = 5.0, 10.0, 10.0       # HRU thresholds [% of the parent area]
YEAR_START, YEAR_END = 2001, 2020                 # simulation period (2001-2004 = warm-up)
EDITOR_VERSION = "4.0.1"          # the SWAT+ Editor version installed with SWAT+ 4.0.2
SWAT_VERSION = "62.0.1"
SUB_NAMES = {1: "Subbasin-1", 2: "Subbasin-2", 3: "Subbasin-3"}
# channel network (from the HEC-HMS/QGIS analysis): Subbasin-2 and -3 join at J1,
# reach R1 (Eifgenbach) runs through Subbasin-1 to the gauge SMAR = outlet
DOWNSTREAM = {1: None, 2: 1, 3: 1}

to_wgs = Transformer.from_crs(25832, 4326, always_xy=True)

# ================================================================== A. common 10 m grid
sub = gpd.read_file(DATA / "wv_markusmuehle" / "wv_markusmuehle.shp").to_crs(25832)
sub["sid"] = sub["name"].str[-1].astype(int)
sub = sub.sort_values("sid").reset_index(drop=True)
xmin, ymin, xmax, ymax = sub.total_bounds
xmin, ymin = np.floor(xmin / CELL) * CELL - CELL, np.floor(ymin / CELL) * CELL - CELL
xmax, ymax = np.ceil(xmax / CELL) * CELL + CELL, np.ceil(ymax / CELL) * CELL + CELL
W, H = int((xmax - xmin) / CELL), int((ymax - ymin) / CELL)
T = from_origin(xmin, ymax, CELL, CELL)


def to_grid(path, method):
    """read any raster and resample it onto the common 10 m grid"""
    with rasterio.open(path) as src:
        dst = np.full((H, W), np.nan, np.float32)
        reproject(rasterio.band(src, 1), dst, dst_transform=T, dst_crs="EPSG:25832",
                  resampling=method, src_nodata=src.nodata, dst_nodata=np.nan)
    return dst


sid = features.rasterize([(g, i) for g, i in zip(sub.geometry, sub.sid)], out_shape=(H, W), transform=T, fill=0)
dem = to_grid(DATA / "dem_markus_5m.tif", Resampling.average)
slope = np.tan(np.radians(to_grid(DATA / "qgis_analysis" / "slope_deg.tif", Resampling.average))) * 100  # %
lc = to_grid(DP / "landcover_worldcover_10m.tif", Resampling.nearest)
soil_cls = to_grid(DP / "soil_class_250m.tif", Resampling.nearest)
soil_tab = pd.read_csv(DP / "soil_profiles_swat.csv")
soil_names = soil_tab.drop_duplicates("class_id").set_index("class_id")["name"].to_dict()

inside = sid > 0
# fill tiny holes (no-data pixels) with the most common value of the subbasin
for arr in (lc, soil_cls):
    for s in sub.sid:
        m = (sid == s) & np.isnan(arr)
        if m.any():
            vals = arr[(sid == s) & ~np.isnan(arr)]
            arr[m] = pd.Series(vals).mode().iloc[0]
for arr in (dem, slope):
    arr[inside & np.isnan(arr)] = np.nanmean(arr[inside])

rows, cols = np.nonzero(inside)
xs, ys = rasterio.transform.xy(T, rows, cols)
pix = pd.DataFrame({"sid": sid[rows, cols], "lc": lc[rows, cols].astype(int),
                    "soil": soil_cls[rows, cols].astype(int), "slope": slope[rows, cols],
                    "elev": dem[rows, cols], "x": xs, "y": ys})
pix["lu"] = pix["lc"].map(LU_MAP).fillna("past")
pix["slp"] = pd.cut(pix["slope"], [b[0] for b in SLOPE_BANDS] + [1e9], labels=[b[2] for b in SLOPE_BANDS],
                    right=False).astype(str)
pix["soilname"] = pix["soil"].map(soil_names)
PA = CELL * CELL / 1e4                                  # pixel area in ha

# ================================================================== B. HRUs with thresholds
hru_rows = []
for s, g in pix.groupby("sid"):
    a_sub = len(g) * PA
    lu_a = g.groupby("lu").size() * PA
    lu_keep = lu_a[lu_a / a_sub * 100 >= THR_LU]
    lu_keep = lu_keep / lu_keep.sum() * a_sub          # eliminated land uses are shared out proportionally
    for lu, a_lu in lu_keep.items():
        gl = g[g.lu == lu]
        so_a = gl.groupby("soilname").size() * PA
        so_keep = so_a[so_a / so_a.sum() * 100 >= THR_SOIL]
        so_keep = so_keep / so_keep.sum() * a_lu
        for so, a_so in so_keep.items():
            gs = gl[gl.soilname == so]
            sl_a = gs.groupby("slp").size() * PA
            sl_keep = sl_a[sl_a / sl_a.sum() * 100 >= THR_SLP]
            sl_keep = sl_keep / sl_keep.sum() * a_so
            for sl, a_hru in sl_keep.items():
                gh = gs[gs.slp == sl]
                lon, lat = to_wgs.transform(gh.x.mean(), gh.y.mean())
                hru_rows.append(dict(lsu=s, arsub=a_sub, arlsu=a_sub, landuse=lu, arland=a_lu, soil=so,
                                     arso=a_so, slp=sl, arslp=a_hru, slope=gh.slope.mean(),
                                     lat=lat, lon=lon, elev=gh.elev.mean()))
hrus = pd.DataFrame(hru_rows)
hrus.insert(0, "id", np.arange(1, len(hrus) + 1))
print(f"{len(hrus)} HRUs created")

# ================================================================== C. subbasins, channels, aquifers, routing
rivers = gpd.read_file(DATA / "r_markusmuehle" / "r_markusmuehle" / "r_markusmuehle.shp").to_crs(25832)
r1 = gpd.read_file(DATA / "qgis_analysis" / "reach_R1.gpkg").to_crs(25832)
stats = pd.read_csv(DATA / "qgis_analysis" / "subbasin_stats.csv").set_index("subbasin")
d5 = rasterio.open(DATA / "dem_markus_5m.tif")


def z_at(x, y):
    """lowest DEM value within 15 m of a point (the stream bed)"""
    r, c = d5.index(x, y)
    win = d5.read(1, window=((r - 3, r + 4), (c - 3, c + 4)), masked=True)
    return float(win.min())

sub_rows, lsu_rows, cha_rows, aqu_rows, rout = [], [], [], [], []
area_ha = {s: (pix.sid == s).sum() * PA for s in sub.sid}
for _, r in sub.iterrows():
    s = r.sid
    g = pix[pix.sid == s]
    st = stats.loc[SUB_NAMES[s]]
    c = r.geometry.centroid
    lon, lat = to_wgs.transform(c.x, c.y)
    sll = 121 if g.slope.mean() < 1 else 90 if g.slope.mean() < 3 else 60 if g.slope.mean() < 5 else 30 if g.slope.mean() < 8 else 10
    sub_rows.append((s, area_ha[s], g.slope.mean(), st.flowpath_L_m, sll, lat, lon, g.elev.mean(), g.elev.min(), g.elev.max(), None))

    # main channel of the subbasin
    up_area_ha = area_ha[s] + sum(area_ha[u] for u, d in DOWNSTREAM.items() if d == s)
    a_km2 = up_area_ha / 100
    if s == 1:                                  # reach R1 from junction J1 to gauge SMAR
        line = r1.geometry.union_all()
        length = line.length
        pts = gpd.read_file(DATA / "qgis_analysis" / "hms_elements.gpkg").to_crs(25832).set_index("name").geometry
        zmin, zmax = sorted([z_at(pts["SMAR"].x, pts["SMAR"].y), z_at(pts["J1"].x, pts["J1"].y)])
        mid = line.centroid
    else:                                       # longest flow path of the subbasin
        length = st.flowpath_L_m
        zmin = g.elev.min()
        zmax = zmin + st.flowpath_S * length
        clip = rivers.clip(r.geometry)
        mid = clip.geometry.union_all().centroid if len(clip) else c
    slope_ch = max((zmax - zmin) / length * 100, 0.1)
    width = 1.29 * a_km2 ** 0.6                 # QSWAT+ default channel geometry (Srinivasan 2006)
    depth = 0.13 * a_km2 ** 0.4
    mlon, mlat = to_wgs.transform(mid.x, mid.y)
    strahler = 3 if s == 1 else 2
    cha_rows.append((s, s, up_area_ha, strahler, length, slope_ch, width, depth, zmin, zmax, mlat, mlon))
    lsu_rows.append((s, 0, s, s, area_ha[s], g.slope.mean(), st.flowpath_L_m, slope_ch, width, depth, lat, lon, g.elev.mean()))
    aqu_rows.append((s, 0, s, 0, area_ha[s], lat, lon, g.elev.mean()))
    # routing: landscape unit -> channel (all water) and -> aquifer (recharge); aquifer -> channel
    rout += [(s, "LSU", "tot", s, "CH", 100.0), (s, "LSU", "rhg", s, "AQU", 100.0), (s, "AQU", "tot", s, "CH", 100.0)]
    rout.append((s, "CH", "tot", DOWNSTREAM[s] or 0, "CH" if DOWNSTREAM[s] else "X", 100.0))

pd.DataFrame(cha_rows, columns=["id", "subbasin", "areac_ha", "strahler", "len_m", "slope_pct", "width_m", "depth_m",
                                "elevmin", "elevmax", "midlat", "midlon"]).to_csv(SW / "channels_summary.csv", index=False)
hrus.to_csv(SW / "hru_summary.csv", index=False, float_format="%.4f")

# ================================================================== D1. project database with the GIS tables
if PROJ.exists():
    shutil.rmtree(PROJ)
WEA.mkdir(parents=True)
db = PROJ / "markusmuehle.sqlite"
con = sqlite3.connect(db)
con.executescript("""
CREATE TABLE project_config (id INTEGER PRIMARY KEY NOT NULL DEFAULT (1), project_name TEXT, project_directory TEXT,
 editor_version TEXT, gis_type TEXT, gis_version TEXT, project_db TEXT, reference_db TEXT, wgn_db TEXT,
 wgn_table_name TEXT, weather_data_dir TEXT, weather_data_format TEXT, input_files_dir TEXT,
 input_files_last_written DATETIME, swat_last_run DATETIME, delineation_done BOOLEAN DEFAULT (0) NOT NULL,
 hrus_done BOOLEAN DEFAULT (0) NOT NULL, soil_table TEXT, soil_layer_table TEXT, output_last_imported DATETIME,
 imported_gis BOOLEAN DEFAULT (0) NOT NULL, is_lte BOOLEAN DEFAULT (0) NOT NULL, use_gwflow BOOLEAN DEFAULT (0) NOT NULL);
CREATE TABLE gis_subbasins (id INTEGER PRIMARY KEY UNIQUE NOT NULL, area REAL, slo1 REAL, len1 REAL, sll REAL,
 lat REAL, lon REAL, elev REAL, elevmin REAL, elevmax REAL, waterid INTEGER);
CREATE TABLE gis_lsus (id INTEGER PRIMARY KEY UNIQUE NOT NULL, category INTEGER, channel INTEGER, subbasin INTEGER,
 area REAL, slope REAL, len1 REAL, csl REAL, wid1 REAL, dep1 REAL, lat REAL, lon REAL, elev REAL);
CREATE TABLE gis_channels (id INTEGER PRIMARY KEY UNIQUE NOT NULL, subbasin INTEGER, areac REAL, strahler INTEGER,
 len2 REAL, slo2 REAL, wid2 REAL, dep2 REAL, elevmin REAL, elevmax REAL, midlat REAL, midlon REAL);
CREATE TABLE gis_hrus (id INTEGER PRIMARY KEY UNIQUE NOT NULL, lsu INTEGER, arsub REAL, arlsu REAL, landuse TEXT,
 arland REAL, soil TEXT, arso REAL, slp TEXT, arslp REAL, slope REAL, lat REAL, lon REAL, elev REAL);
CREATE TABLE gis_aquifers (id INTEGER PRIMARY KEY, category INTEGER, subbasin INTEGER, deep_aquifer INTEGER,
 area REAL, lat REAL, lon REAL, elev REAL);
CREATE TABLE gis_deep_aquifers (id INTEGER PRIMARY KEY, subbasin INTEGER, area REAL, lat REAL, lon REAL, elev REAL);
CREATE TABLE gis_water (id INTEGER, wtype TEXT, lsu INTEGER, subbasin INTEGER, area REAL, xpr REAL, ypr REAL,
 lat REAL, lon REAL, elev REAL);
CREATE TABLE gis_points (id INTEGER NOT NULL, subbasin INTEGER, ptype TEXT, xpr REAL, ypr REAL, lat REAL, lon REAL, elev REAL);
CREATE TABLE gis_routing (sourceid INTEGER, sourcecat TEXT, hyd_typ TEXT, sinkid INTEGER, sinkcat TEXT, percent REAL);
CREATE TABLE soils_sol (id INTEGER NOT NULL PRIMARY KEY, name TEXT NOT NULL, hyd_grp TEXT NOT NULL, dp_tot REAL NOT NULL,
 anion_excl REAL NOT NULL, perc_crk REAL NOT NULL, texture TEXT, description TEXT);
CREATE TABLE soils_sol_layer (id INTEGER NOT NULL PRIMARY KEY, soil_id INTEGER NOT NULL, layer_num INTEGER NOT NULL,
 dp REAL NOT NULL, bd REAL NOT NULL, awc REAL NOT NULL, soil_k REAL NOT NULL, carbon REAL NOT NULL, clay REAL NOT NULL,
 silt REAL NOT NULL, sand REAL NOT NULL, rock REAL NOT NULL, alb REAL NOT NULL, usle_k REAL NOT NULL, ec REAL NOT NULL,
 caco3 REAL, ph REAL, FOREIGN KEY (soil_id) REFERENCES soils_sol (id) ON DELETE CASCADE);
""")
con.execute("INSERT INTO project_config (id, project_name, project_directory, gis_type, gis_version, project_db, "
            "reference_db, weather_data_dir, weather_data_format, input_files_dir, delineation_done, hrus_done) "
            "VALUES (1,'markusmuehle','','qgis','4.0','markusmuehle.sqlite','swatplus_datasets.sqlite',"
            "'weather','plus','Scenarios/Default/TxtInOut',1,1)")


def py(rows):
    """numpy numbers -> plain Python numbers (SQLite cannot store numpy types)"""
    return [tuple(float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for v in r)
            for r in rows]


sub_rows, lsu_rows, cha_rows, aqu_rows, rout = py(sub_rows), py(lsu_rows), py(cha_rows), py(aqu_rows), py(rout)
con.executemany("INSERT INTO gis_subbasins VALUES (?,?,?,?,?,?,?,?,?,?,?)", sub_rows)
con.executemany("INSERT INTO gis_lsus VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", lsu_rows)
con.executemany("INSERT INTO gis_channels VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", cha_rows)
con.executemany("INSERT INTO gis_aquifers VALUES (?,?,?,?,?,?,?,?)", aqu_rows)
con.executemany("INSERT INTO gis_routing VALUES (?,?,?,?,?,?)", rout)
con.executemany("INSERT INTO gis_hrus VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                py(hrus[["id", "lsu", "arsub", "arlsu", "landuse", "arland", "soil", "arso", "slp",
                         "arslp", "slope", "lat", "lon", "elev"]].itertuples(index=False)))

# ================================================================== D2. soils (from OpenLandMap, see step 3)
used = sorted(hrus.soil.unique())
lid = 1
for k, (name, prof) in enumerate(soil_tab[soil_tab.name.isin(used)].groupby("name", sort=True), start=1):
    p0 = prof.iloc[0]
    con.execute("INSERT INTO soils_sol VALUES (?,?,?,?,?,?,?,?)",
                py([(k, name, p0.hyd_grp, float(prof.dp_mm.max()), 0.5, 0.5, p0.texture, p0.description)])[0])
    for n, (_, L) in enumerate(prof.sort_values("dp_mm").iterrows(), start=1):
        con.execute("INSERT INTO soils_sol_layer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    py([(lid, k, n, L.dp_mm, L.bd, L.awc, L.ksat_mmh, L.oc_pct, L.clay, L.silt, L.sand, L.rock,
                         L.alb, L.usle_k, 0.0, 0.0, L.ph)])[0])
        lid += 1
con.commit(); con.close()
DATASETS = find_datasets_db()
print("SWAT+ datasets database:", DATASETS)
shutil.copy(DATASETS, PROJ / "swatplus_datasets.sqlite")

# ================================================================== D3. weather files (SWAT+ format)
obs = pd.read_csv(DP / "observed_daily.csv", parse_dates=["date"], index_col="date")
clim = pd.read_csv(DP / "climate_daily.csv", parse_dates=["date"], index_col="date")
days = pd.date_range(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31", freq="D")
obs, clim = obs.reindex(days), clim.reindex(days)


def write_wfile(fname, title, lat, lon, elev, cols):
    with open(WEA / fname, "w") as f:
        f.write(f"{fname}: {title}\n")
        f.write("nbyr     tstep       lat       lon      elev\n")
        f.write(f"{YEAR_END - YEAR_START + 1:4d}         0 {lat:9.4f} {lon:9.4f} {elev:9.2f}\n")
        vals = np.column_stack(cols)
        for d, v in zip(days, vals):
            f.write(f"{d.year} {d.dayofyear:4d} " + " ".join(f"{x:9.3f}" if np.isfinite(x) else "  -99.000" for x in v) + "\n")


files = {k: [] for k in ["pcp", "tmp", "slr", "hmd", "wnd"]}
for s, (lat, lon, elev) in {r[0]: (r[5], r[6], r[7]) for r in sub_rows}.items():
    tag = f"sub{s}"
    write_wfile(f"{tag}.pcp", "RADOLAN radar rainfall [mm/day]", lat, lon, elev, [obs[f"P_sub{s}"].to_numpy()])
    write_wfile(f"{tag}.tmp", "ERA5-Land Tmax Tmin [degC]", lat, lon, elev, [clim.tmax_C.to_numpy(), clim.tmin_C.to_numpy()])
    write_wfile(f"{tag}.slr", "ERA5-Land solar radiation [MJ/m2/day]", lat, lon, elev, [clim.srad_MJm2.to_numpy()])
    write_wfile(f"{tag}.hmd", "ERA5-Land relative humidity [fraction]", lat, lon, elev, [clim.rh_frac.to_numpy()])
    write_wfile(f"{tag}.wnd", "ERA5-Land wind speed [m/s]", lat, lon, elev, [clim.wind_ms.to_numpy()])
    for k in files:
        files[k].append(f"{tag}.{k}")
for k, lst in files.items():
    with open(WEA / f"{k}.cli", "w") as f:
        f.write(f"{k}.cli: {k} file names\nfilename\n" + "\n".join(lst) + "\n")

# weather generator statistics (used by SWAT+ only to fill gaps) from the same observations
wg = pd.DataFrame({"P": obs[["P_sub1", "P_sub2", "P_sub3"]].mean(axis=1), "tx": clim.tmax_C, "tn": clim.tmin_C,
                   "sr": clim.srad_MJm2, "td": clim.tdew_C, "ws": clim.wind_ms}, index=days)
rain_h = pd.read_csv(DP / "rain_hourly.csv", parse_dates=["time"], index_col="time")["P_basin"]
rain_h = rain_h.loc[f"{YEAR_START}":f"{YEAR_END}"]
wet = wg.P >= 0.1
mon_rows = []
for m in range(1, 13):
    gm = wg[wg.index.month == m]
    wm = wet[wet.index.month == m]
    prev = wet.shift(1)[wet.index.month == m]
    n_years = gm.index.year.nunique()
    hhr = rain_h[rain_h.index.month == m].groupby(rain_h[rain_h.index.month == m].index.year).max().mean() * 0.7
    mon_rows.append(dict(month=m, tmp_max_ave=gm.tx.mean(), tmp_min_ave=gm.tn.mean(), tmp_max_sd=gm.tx.std(),
                         tmp_min_sd=gm.tn.std(), pcp_ave=gm.P.sum() / n_years, pcp_sd=gm.P.std(), pcp_skew=gm.P.skew(),
                         wet_dry=(wm & (prev == False)).sum() / max((prev == False).sum(), 1),
                         wet_wet=(wm & (prev == True)).sum() / max((prev == True).sum(), 1),
                         pcp_days=wm.sum() / n_years, pcp_hhr=hhr, slr_ave=gm.sr.mean(), dew_ave=gm.td.mean(),
                         wnd_ave=gm.ws.mean()))
wgn = pd.DataFrame(mon_rows)
wgn.to_csv(SW / "weather_generator_wgn.csv", index=False, float_format="%.3f")

# ================================================================== E. SWAT+ Editor: build project + write TxtInOut
cwd = os.getcwd()
os.chdir(TOOLS / "swatplus_editor_api")
from actions.setup_project import SetupProject
from actions.import_weather import WeatherImport
from actions.write_files import WriteFiles
from database.project.setup import SetupProjectDatabase
from database.project import climate, simulation

SetupProject(str(db), EDITOR_VERSION, "markusmuehle", str(PROJ / "swatplus_datasets.sqlite"))   # + GIS import
SetupProjectDatabase.init(str(db))
blon, blat = to_wgs.transform(*sub.union_all().centroid.coords[0])
w = climate.Weather_wgn_cli.create(name="wgn_markus", lat=blat, lon=blon, elev=float(pix.elev.mean()),
                                   rain_yrs=YEAR_END - YEAR_START + 1)
for r in mon_rows:
    climate.Weather_wgn_cli_mon.create(weather_wgn_cli=w.id, **{k: float(v) for k, v in r.items()})
WeatherImport(str(db), True, True).import_data()          # creates 3 stations and links them to every object
simulation.Time_sim.update(day_start=0, yrc_start=YEAR_START, day_end=0, yrc_end=YEAR_END, step=0).execute()

# print only what we need (keeps the run fast): daily channel flow, water balances
pp = simulation.Print_prt.get()
pp.nyskip = 0; pp.csvout = False; pp.save()
keep = {"channel_sd": (True, True, True, True), "basin_wb": (False, True, True, True),
        "lsunit_wb": (False, False, True, True), "hru_wb": (False, False, False, True),
        "basin_aqu": (False, True, True, True), "aquifer": (False, False, True, True)}
for o in simulation.Print_prt_object.select():
    o.daily, o.monthly, o.yearly, o.avann = keep.get(o.name, (False, False, False, False))
    o.save()
WriteFiles(str(db), SWAT_VERSION).write()
os.chdir(cwd)

TXT = PROJ / "Scenarios" / "Default" / "TxtInOut"
exe = find_swat_exe()
print("SWAT+ executable:", exe)
shutil.copy(exe, TXT / "swatplus.exe")
os.chmod(TXT / "swatplus.exe", 0o755)

# ================================================================== F. first run with default parameters
print("running SWAT+ with default parameters ...")
res = subprocess.run([str(TXT / "swatplus.exe")], cwd=TXT, capture_output=True, text=True)
(SW / "swatplus_default_run.log").write_text(res.stdout[-5000:] + res.stderr[-5000:])
print(res.stdout[-400:])
done("04_build_swatplus_project")
