"""
Shared settings for all scripts (paths, periods, a log file).
You normally do not need to change anything here.

Folder layout that the scripts expect:
    data/                      <- your data folder (DEM, shapefiles, discharge, RADOLAN ...)
    data/Models/               <- everything produced by the scripts
    data/Models/scripts/       <- this file and the numbered scripts
"""
import sys, pathlib, datetime, os

SCRIPTS = pathlib.Path(__file__).resolve().parent
MODELS = SCRIPTS.parent                      # ...\data\Models
DATA = pathlib.Path(os.environ.get("W4220_DATA", MODELS.parent))   # ...\data
DP = MODELS / "data_processed"               # cleaned / processed data
RAW = MODELS / "data_raw_satellite"          # downloaded satellite / reanalysis data
FIG = MODELS / "figures"
SW = MODELS / "SWATplus"
LS = MODELS / "LSTM"
LOGS = MODELS / "logs"
for p in (DP, RAW, FIG, SW, LS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

# where SWAT+ 4.0.2 was installed ("Install for me only"); change if you installed elsewhere
SWATPLUS_HOME = pathlib.Path(os.environ.get("SWATPLUS_HOME", pathlib.Path.home() / "SWATPlus"))

AREA_KM2 = 23.885                            # catchment area at gauge Markusmuehle
WARMUP = ("2001-01-01", "2004-12-31")
CALIB = ("2005-01-01", "2012-12-31")         # SWAT+ calibration / LSTM training (+ early stopping)
VALID = ("2013-01-01", "2020-12-31")         # SWAT+ validation / LSTM test


class _Tee:
    """Write everything that is printed both to the screen and to a log file.

    Behaves like a minimal file object, so it can replace ``sys.stdout`` or
    ``sys.stderr``. Every write is passed on to all wrapped streams.
    """
    def __init__(self, *streams):
        """Keep the streams to copy output to (e.g. the console and an open log file)."""
        self.streams = streams

    def write(self, s):
        """Write the text ``s`` to every stream and flush, so the log is never behind the screen."""
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        """Flush all wrapped streams."""
        for st in self.streams:
            st.flush()


def start_log(name):
    """Start logging a script run to ``logs/<name>.log``.

    Redirects ``sys.stdout`` and ``sys.stderr`` so that all printed output and
    error messages appear on the screen and are also saved in the log file.
    The log file is overwritten on each run. A header with the start time,
    the Python version and the data and Models folders is printed first.

    Parameters
    ----------
    name : str
        Name of the script, used for the log file name and the header line
        (for example ``"01_prepare_observed_data"``).
    """
    f = open(LOGS / f"{name}.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print(f"=== {name}  started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  (Python {sys.version.split()[0]}, {sys.executable})")
    print(f"    data folder  : {DATA}")
    print(f"    Models folder: {MODELS}")


def done(name):
    """Print the closing line with the finish time, so a complete run is easy to spot in the log.

    Parameters
    ----------
    name : str
        The same name that was given to :func:`start_log`.
    """
    print(f"=== {name}  finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
