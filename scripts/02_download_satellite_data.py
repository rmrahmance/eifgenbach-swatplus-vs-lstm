"""
STEP 2 - Download the satellite / reanalysis data for SWAT+ (land cover, soil, climate)
========================================================================================
Everything is downloaded directly from the free, official servers - no account needed:

  1. ESA WorldCover 2021 v200, 10 m land cover (Sentinel-1 + Sentinel-2)
     https://esa-worldcover.s3.eu-central-1.amazonaws.com  (Cloud-Optimised GeoTIFF on AWS)
  2. ISRIC SoilGrids 2.0, 250 m soil properties at 6 depths
     https://files.isric.org/soilgrids/latest/data  (fallback: ISRIC WCS service)
  3. ERA5-Land / ERA5 reanalysis (ECMWF/Copernicus), hourly -> daily climate 2000-2024
     https://archive-api.open-meteo.com  (Open-Meteo historical weather API, models "era5_land" and "era5")

Only the small window around the catchment is read from the big global files
("windowed reading" of Cloud-Optimised GeoTIFFs), so the downloads are only a few MB.

Output (Models/data_raw_satellite):
  worldcover_2021_10m.tif        land-cover classes, EPSG:25832, 10 m
  soilgrids_250m.tif             42 bands: sand, silt, clay, soc, bdod, cfvo, phh2o x 6 depths, EPSG:25832, 250 m
  era5_hourly_<year>.csv         raw hourly ERA5-Land (T, dew point, RH) + ERA5 (radiation, wind, rain) values
  era5land_daily_basin.csv       daily Tmax, Tmin, Tmean, dew point, RH, radiation, wind, rain, ET0
How to run: open this file in VS Code and press "Run Python File". Needs internet (~5-10 min).
"""
import os, time, json, tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import requests
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from _common import DATA, RAW, start_log, done

start_log("02_download_satellite_data")
os.environ.update({"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
                   "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2", "GDAL_HTTP_TIMEOUT": "60"})

# ------------------------------------------------------------------ the catchment and the target grids
basin = gpd.read_file(DATA / "wv_markusmuehle_basin" / "wv_markusmuehle_basin.shp").to_crs(25832)
xmin, ymin, xmax, ymax = basin.total_bounds


def grid(cell, buf):
    """regular grid in ETRS89 / UTM 32N (EPSG:25832) around the catchment"""
    x0, y0 = np.floor((xmin - buf) / cell) * cell, np.floor((ymin - buf) / cell) * cell
    x1, y1 = np.ceil((xmax + buf) / cell) * cell, np.ceil((ymax + buf) / cell) * cell
    w, h = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    return from_origin(x0, y1, cell, cell), w, h


def read_window(url, cell, buf, resampling, nodata=None):
    """read only the catchment window of a remote global raster, re-projected to EPSG:25832"""
    T, w, h = grid(cell, buf)
    with rasterio.open("/vsicurl/" + url) as src:
        with WarpedVRT(src, crs="EPSG:25832", transform=T, width=w, height=h, resampling=resampling,
                       src_nodata=nodata if nodata is not None else src.nodata) as vrt:
            return vrt.read(1), T


def save(path, arrays, T, dtype, nodata, names):
    prof = dict(driver="GTiff", height=arrays[0].shape[0], width=arrays[0].shape[1], count=len(arrays),
                dtype=dtype, crs="EPSG:25832", transform=T, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        for i, (a, n) in enumerate(zip(arrays, names), start=1):
            dst.write(a.astype(dtype), i)
            dst.set_band_description(i, n)


# ================================================================== 1. ESA WorldCover
t0 = time.time()
out_lc = RAW / "worldcover_2021_10m.tif"
if not out_lc.exists():
    lon, lat = gpd.GeoSeries(basin.geometry, crs=25832).to_crs(4326).union_all().centroid.coords[0]
    tile = f"N{int(np.floor(lat / 3) * 3):02d}E{int(np.floor(lon / 3) * 3):03d}"
    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    print("WorldCover tile:", url)
    lc, T = read_window(url, 10, 300, Resampling.nearest, nodata=0)
    save(out_lc, [lc], T, "uint8", 0, ["landcover"])
    v, n = np.unique(lc[lc > 0], return_counts=True)
    print("  classes found:", dict(zip(v.tolist(), n.tolist())), f"({time.time() - t0:.0f} s)")
else:
    print("WorldCover already downloaded:", out_lc)

# ================================================================== 2. SoilGrids 2.0
PROPS = ["sand", "silt", "clay", "soc", "bdod", "cfvo", "phh2o"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
out_soil = RAW / "soilgrids_250m.tif"


def soilgrids_wcs(prop, depth):
    """fallback: ISRIC Web Coverage Service, cut-out in geographic coordinates"""
    T, w, h = grid(250, 1000)
    lon0, lat0, lon1, lat1 = gpd.GeoSeries([basin.buffer(2000).union_all()], crs=25832).to_crs(4326).total_bounds
    url = (f"https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
           f"&COVERAGEID={prop}_{depth}_mean&FORMAT=image/tiff&SUBSET=long({lon0},{lon1})&SUBSET=lat({lat0},{lat1})"
           "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    tmp = os.path.join(tempfile.gettempdir(), f"sg_{prop}_{depth}.tif")
    open(tmp, "wb").write(r.content)
    dst = np.full((h, w), np.nan, np.float32)
    with rasterio.open(tmp) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=T, dst_crs="EPSG:25832", resampling=Resampling.nearest,
                  src_nodata=src.nodata if src.nodata is not None else -32768, dst_nodata=np.nan)
    return dst, T


if not out_soil.exists():
    arrays, names = [], []
    for prop in PROPS:
        for depth in DEPTHS:
            url = f"https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_mean.vrt"
            try:
                a, T = read_window(url, 250, 1000, Resampling.nearest, nodata=-32768)
                a = np.where(a == -32768, np.nan, a).astype(np.float32)
            except Exception as e:  # noqa: BLE001
                print(f"  VRT failed for {prop} {depth} ({e}); trying WCS ...")
                a, T = soilgrids_wcs(prop, depth)
            arrays.append(a); names.append(f"{prop}_{depth}")
            print(f"  SoilGrids {prop:5s} {depth:9s} mean in window = {np.nanmean(a):8.1f}  ({time.time() - t0:.0f} s)")
    save(out_soil, arrays, T, "float32", np.nan, names)
else:
    print("SoilGrids already downloaded:", out_soil)

# ================================================================== 3. ERA5-Land climate (Open-Meteo API)
c = basin.to_crs(4326).union_all().centroid
LAT, LON, ELEV = round(c.y, 4), round(c.x, 4), 254          # catchment centre, mean elevation [m]
LAND_VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m"]          # from ERA5-Land (~9 km)
ATM_VARS = ["shortwave_radiation", "wind_speed_10m", "precipitation", "et0_fao_evapotranspiration"]  # ERA5 (~28 km)
HOURLY = LAND_VARS + ATM_VARS


def open_meteo(year, variables, model):
    params = dict(latitude=LAT, longitude=LON, elevation=ELEV, start_date=f"{year}-01-01", end_date=f"{year}-12-31",
                  hourly=",".join(variables), models=model, wind_speed_unit="ms", timezone="GMT")
    for attempt in range(5):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
        if r.status_code == 200:
            break
        print(f"  {year}: HTTP {r.status_code}, retry in 20 s"); time.sleep(20)
    r.raise_for_status()
    js = r.json()
    return pd.DataFrame(js["hourly"]), js


frames = []
for year in range(2000, 2025):
    f = RAW / f"era5_hourly_{year}.csv"
    if not f.exists():
        land, js = open_meteo(year, LAND_VARS, "era5_land")        # temperature + humidity: ERA5-Land
        atm, _ = open_meteo(year, ATM_VARS, "era5")                # radiation, wind, rain: ERA5 (not in ERA5-Land API)
        h = land.merge(atm, on="time")
        h.to_csv(f, index=False)
        print(f"  ERA5-Land/ERA5 {year}: {len(h)} hours  (ERA5-Land cell {js['latitude']:.3f} N, {js['longitude']:.3f} E)")
        time.sleep(1)
    frames.append(pd.read_csv(f, parse_dates=["time"]))
h = pd.concat(frames).set_index("time").sort_index()

d = pd.DataFrame({
    "tmax_C": h.temperature_2m.resample("D").max(),
    "tmin_C": h.temperature_2m.resample("D").min(),
    "tmean_C": h.temperature_2m.resample("D").mean(),
    "tdew_C": h.dew_point_2m.resample("D").mean(),
    "rh_frac": h.relative_humidity_2m.resample("D").mean() / 100,
    "srad_MJm2": h.shortwave_radiation.resample("D").sum() * 3600 / 1e6,    # W/m2 (hourly mean) -> MJ/m2/day
    "wind_ms": h.wind_speed_10m.resample("D").mean(),
    "pcp_era5_mm": h.precipitation.resample("D").sum(),
    "et0_openmeteo_mm": h.et0_fao_evapotranspiration.resample("D").sum(),
})
d.index.name = "date"
d.to_csv(RAW / "era5land_daily_basin.csv", float_format="%.3f")
print(f"ERA5-Land daily: {d.index.min().date()} .. {d.index.max().date()}, mean T {d.tmean_C.mean():.1f} C, "
      f"mean annual rain {d.pcp_era5_mm.resample('YE').sum().mean():.0f} mm")
json.dump({"latitude": LAT, "longitude": LON, "elevation": ELEV, "variables": HOURLY},
          open(RAW / "era5land_request.json", "w"), indent=1)
done("02_download_satellite_data")
