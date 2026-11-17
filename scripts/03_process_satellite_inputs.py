"""
STEP 3 - Turn the satellite / reanalysis downloads into model inputs
=====================================================================
Input  (Models/data_raw_satellite, from step 2):
  worldcover_2021_10m.tif, soilgrids_250m.tif, era5land_daily_basin.csv
Output (Models/data_processed):
  climate_daily.csv             Tmax, Tmin, Tmean, dew point, RH, solar radiation, wind, FAO-56 ET0
  landcover_worldcover_10m.tif  land cover clipped to the catchment
  landcover_fractions.csv       share of each land-cover class per subbasin
  soil_class_250m.tif           soil class map (USDA texture class of the topsoil)
  soil_profiles_swat.csv        SWAT+ soil table (5 layers; AWC, Ksat ... from pedotransfer functions)
How to run: open this file in VS Code and press "Run Python File".
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from _common import DATA, RAW, DP, start_log, done

start_log("03_process_satellite_inputs")
ELEV, LAT = 254.0, 51.115                         # mean catchment elevation [m], latitude [deg]
sub = gpd.read_file(DATA / "wv_markusmuehle" / "wv_markusmuehle.shp").to_crs(25832).sort_values("name")
basin = sub.union_all()

# ================================================================== 1. CLIMATE (ERA5-Land)
c = pd.read_csv(RAW / "era5land_daily_basin.csv", parse_dates=["date"], index_col="date").sort_index()


def es(T):                                         # saturation vapour pressure [kPa], FAO-56 eq. 11
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


c["wind2m_ms"] = c.wind_ms * 4.87 / np.log(67.8 * 10 - 5.42)        # 10 m -> 2 m wind (FAO-56 eq. 47)

# FAO-56 Penman-Monteith reference evapotranspiration ET0 [mm/day]
J = c.index.dayofyear.to_numpy()
phi = np.radians(LAT)
dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)                         # inverse relative Earth-Sun distance
dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)                     # solar declination
ws = np.arccos(-np.tan(phi) * np.tan(dec))                           # sunset hour angle
Ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
Rso = (0.75 + 2e-5 * ELEV) * Ra                                      # clear-sky radiation
Tx, Tn, Rs, u2 = c.tmax_C.to_numpy(), c.tmin_C.to_numpy(), c.srad_MJm2.to_numpy(), c.wind2m_ms.to_numpy()
ea = es(c.tdew_C.to_numpy())                                         # actual vapour pressure from dew point
esat = (es(Tx) + es(Tn)) / 2
Rnl = 4.903e-9 * (((Tx + 273.16) ** 4 + (Tn + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(Rs / Rso, 0, 1) - 0.35)
Rn = 0.77 * Rs - Rnl                                                 # net radiation (albedo 0.23)
Tm = (Tx + Tn) / 2
delta = 4098 * es(Tm) / (Tm + 237.3) ** 2
gamma = 0.000665 * 101.3 * ((293 - 0.0065 * ELEV) / 293) ** 5.26
c["pet_mm"] = np.clip((0.408 * delta * Rn + gamma * 900 / (Tm + 273) * u2 * (esat - ea)) /
                      (delta + gamma * (1 + 0.34 * u2)), 0, None)
c.to_csv(DP / "climate_daily.csv", float_format="%.3f")
y = c.loc["2001":"2020"]
print(f"climate 2001-2020: mean T {y.tmean_C.mean():.1f} C | ET0 {y.pet_mm.resample('YE').sum().mean():.0f} mm/yr "
      f"(Open-Meteo's own ET0: {y.et0_openmeteo_mm.resample('YE').sum().mean():.0f}) | ERA5-Land rain "
      f"{y.pcp_era5_mm.resample('YE').sum().mean():.0f} mm/yr")

# ================================================================== 2. LAND COVER (ESA WorldCover)
WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse vegetation",
      70: "Snow and ice", 80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
with rasterio.open(RAW / "worldcover_2021_10m.tif") as src:
    arr, tr = mask(src, [basin.buffer(50)], crop=True, nodata=0)
    prof = src.profile
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr, nodata=0, dtype="uint8", compress="lzw")
    with rasterio.open(DP / "landcover_worldcover_10m.tif", "w", **prof) as dst:
        dst.write(arr.astype("uint8"))
    rows = []
    for _, s in sub.iterrows():
        a, _ = mask(src, [s.geometry], crop=True, nodata=0)
        v, n = np.unique(a[a > 0], return_counts=True)
        for vv, nn in zip(v, n):
            rows.append(dict(subbasin=s["name"], class_code=int(vv), class_name=WC.get(int(vv), "?"),
                             area_km2=nn * 100 / 1e6, percent=100 * nn / n.sum()))
lcf = pd.DataFrame(rows)
tot = lcf.groupby(["class_code", "class_name"]).area_km2.sum().reset_index()
tot["subbasin"] = "Basin"
tot["percent"] = 100 * tot.area_km2 / tot.area_km2.sum()
lcf = pd.concat([lcf, tot], ignore_index=True)
lcf.to_csv(DP / "landcover_fractions.csv", index=False, float_format="%.3f")
print(lcf[lcf.subbasin == "Basin"][["class_name", "percent"]].round(1).to_string(index=False))

# ================================================================== 3. SOIL (SoilGrids) -> SWAT+ soil table
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]       # SWAT+ layers = SoilGrids depth intervals
BOTTOM_MM = [50, 150, 300, 600, 1000]
with rasterio.open(RAW / "soilgrids_250m.tif") as src:
    names = list(src.descriptions)
    arr, tr = mask(src, [basin.buffer(250)], crop=True, nodata=np.nan, all_touched=True, filled=True)
    sp = src.profile
B = {n: arr[i] for i, n in enumerate(names)}
# SoilGrids units -> normal units
conv = {"sand": 0.1, "silt": 0.1, "clay": 0.1,        # g/kg -> %
        "soc": 0.01,                                   # dg/kg -> %  (organic carbon)
        "bdod": 0.01,                                  # cg/cm3 -> g/cm3 (bulk density)
        "cfvo": 0.1,                                   # cm3/dm3 -> % (coarse fragments = stones)
        "phh2o": 0.1}                                  # pH x 10 -> pH
for n in list(B):
    B[n] = B[n] * conv[n.split("_")[0]]


def usda_texture(sand, silt, clay):
    """USDA soil texture triangle -> class abbreviation"""
    if silt + 1.5 * clay < 15: return "Sa"
    if silt + 1.5 * clay >= 15 and silt + 2 * clay < 30: return "LoSa"
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2 * clay >= 30): return "SaLo"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52: return "Lo"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12): return "SiLo"
    if silt >= 80 and clay < 12: return "Si"
    if 20 <= clay < 35 and silt < 28 and sand > 45: return "SaClLo"
    if 27 <= clay < 40 and 20 < sand <= 45: return "ClLo"
    if 27 <= clay < 40 and sand <= 20: return "SiClLo"
    if clay >= 35 and sand > 45: return "SaCl"
    if clay >= 40 and silt >= 40: return "SiCl"
    return "Cl"


TEXNAME = {"Sa": "sand", "LoSa": "loamy sand", "SaLo": "sandy loam", "Lo": "loam", "SiLo": "silt loam", "Si": "silt",
           "SaClLo": "sandy clay loam", "ClLo": "clay loam", "SiClLo": "silty clay loam", "SaCl": "sandy clay",
           "SiCl": "silty clay", "Cl": "clay"}
valid = np.isfinite(B["sand_0-5cm"])
tex = np.full(valid.shape, "", dtype=object)
for i, j in zip(*np.nonzero(valid)):
    tex[i, j] = usda_texture(B["sand_0-5cm"][i, j], B["silt_0-5cm"][i, j], B["clay_0-5cm"][i, j])
classes = sorted({t for t in tex[valid]})
code_of = {t: k + 1 for k, t in enumerate(classes)}
cls = np.zeros(valid.shape, "uint8")
for t, k in code_of.items():
    cls[tex == t] = k
sp.update(count=1, dtype="uint8", nodata=0, height=cls.shape[0], width=cls.shape[1], transform=tr, compress="lzw")
with rasterio.open(DP / "soil_class_250m.tif", "w", **sp) as dst:
    dst.write(cls, 1)


def saxton_rawls(S, C, OM):
    """Saxton & Rawls (2006): S, C as fractions, OM in % -> field capacity, wilting point, saturation, Ksat [mm/h]"""
    t1500 = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    wp = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    fc = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    ts33 = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    s33 = ts33 + (0.636 * ts33 - 0.107)
    sat = fc + s33 - 0.097 * S + 0.043
    lam = (np.log(fc) - np.log(wp)) / (np.log(1500) - np.log(33))
    ks = 1930 * (sat - fc) ** (3 - lam)
    return fc, wp, sat, ks


def usle_k(sand, silt, clay, oc):
    """Williams (1995) soil erodibility, as in the SWAT documentation"""
    fcsand = 0.2 + 0.3 * np.exp(-0.256 * sand * (1 - silt / 100))
    fclsi = (silt / (clay + silt)) ** 0.3
    forgc = 1 - 0.25 * oc / (oc + np.exp(3.72 - 2.95 * oc))
    sn = 1 - sand / 100
    fhisand = 1 - 0.7 * sn / (sn + np.exp(-5.51 + 22.9 * sn))
    return fcsand * fclsi * forgc * fhisand


soil_rows = []
for t, code in code_of.items():
    m = cls == code
    layers = []
    for dp, d in zip(BOTTOM_MM, DEPTHS):
        g = lambda p: float(np.nanmean(B[f"{p}_{d}"][m]))
        sand, silt, clay, oc, bd, rock, ph = g("sand"), g("silt"), g("clay"), g("soc"), g("bdod"), g("cfvo"), g("phh2o")
        tot3 = sand + silt + clay
        sand, silt, clay = 100 * sand / tot3, 100 * silt / tot3, 100 * clay / tot3     # make them add up to 100 %
        om = min(oc * 1.724, 8.0)            # organic matter = 1.724 x OC; equations are valid up to 8 % OM
        fc, wp, sat, ks = saxton_rawls(sand / 100, clay / 100, om)
        awc = (fc - wp) * (1 - rock / 100)                                             # stones hold no water
        layers.append(dict(class_id=code, name=f"SG_{t}", texture=t, description=f"SoilGrids {TEXNAME[t]}",
                           dp_mm=dp, bd=round(bd, 3), awc=round(awc, 4), ksat_mmh=round(ks, 2), oc_pct=round(oc, 3),
                           clay=round(clay, 1), silt=round(silt, 1), sand=round(sand, 1), rock=round(rock, 1),
                           alb=round(float(np.clip(0.2227 * np.exp(-1.8672 * oc), 0.05, 0.25)), 3),
                           usle_k=round(usle_k(sand, silt, clay, oc), 3), ph=round(ph, 1),
                           fc=round(fc, 4), wp=round(wp, 4), n_pixels_250m=int(m.sum())))
    kmin = min(L["ksat_mmh"] for L in layers) / 3.6            # least permeable layer, mm/h -> micrometre/s
    hsg = "A" if kmin > 40 else "B" if kmin > 10 else "C" if kmin > 1 else "D"
    for L in layers:
        L["hyd_grp"] = hsg
    soil_rows += layers
soil = pd.DataFrame(soil_rows)
soil.to_csv(DP / "soil_profiles_swat.csv", index=False)
print(soil[["name", "dp_mm", "sand", "silt", "clay", "oc_pct", "bd", "rock", "awc", "ksat_mmh", "hyd_grp", "n_pixels_250m"]].to_string(index=False))
done("03_process_satellite_inputs")
