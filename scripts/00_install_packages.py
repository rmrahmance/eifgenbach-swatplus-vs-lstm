"""
STEP 0 - Install the Python packages needed by all other scripts
=================================================================
Open this file in Visual Studio Code and press the Run button (the triangle, top right).
It installs the packages into the Python interpreter that runs this file and then
checks that they work. Run it only once. Internet connection needed (~500 MB, mostly PyTorch).
"""
import subprocess, sys, importlib
from _common import start_log, done

start_log("00_install_packages")
PACKAGES = ["numpy", "pandas", "scipy", "matplotlib", "geopandas", "pyogrio", "shapely", "pyproj",
            "rasterio", "requests", "peewee"]


def pip(*args):
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off", *args]
    print(">", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:                      # show pip's messages live
        print("   ", line.rstrip(), flush=True)
    return p.wait()


missing = []
for name in PACKAGES:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
print("packages to install:", missing or "none")
if missing:
    pip(*missing)
try:
    import torch  # noqa: F401
    print("PyTorch already installed:", torch.__version__)
except ImportError:
    # CPU-only PyTorch (smaller download, no graphics card needed)
    if pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu") != 0:
        pip("torch")

print("\nChecking the installation:")
ok = True
for name in PACKAGES + ["torch"]:
    try:
        m = importlib.import_module(name)
        print(f"  OK   {name:<12} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  FAIL {name:<12} {e}")
print("\nAll packages are ready." if ok else "\nSome packages failed - see messages above.")
done("00_install_packages")
