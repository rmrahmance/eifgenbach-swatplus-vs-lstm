"""
Checks for the NSE, KGE and PBIAS functions, with values worked out by hand.

The function metrics(o, s) is written out in three scripts (05, 06 and 07).
Those scripts read data as soon as they are imported, so the test pulls only
the metrics function out of each file and runs it on small made-up series.

Run from the repository folder:
    python -m unittest discover tests
"""
import ast
import math
import pathlib
import unittest

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
FILES = ["05_calibrate_validate_swatplus.py", "06_lstm_model.py", "07_make_figures.py"]


def load_metrics(fname):
    """Return the metrics() function defined in one script, without running the rest of it."""
    tree = ast.parse((SCRIPTS / fname).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "metrics")
    ns = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), fname, "exec"), ns)
    return ns["metrics"]


O = [1.0, 2.0, 3.0, 4.0]

# (observed, simulated, NSE, KGE, PBIAS), worked out by hand:
CASES = [
    # perfect simulation
    (O, O, 1.0, 1.0, 0.0),
    # every day 1 too high: sum of squared errors 4, variance sum 5 -> NSE 0.2
    # r = 1, same spread, mean 3.5 / 2.5 = 1.4 -> KGE = 1 - 0.4 = 0.6, PBIAS = 100 * 4 / 10
    (O, [2.0, 3.0, 4.0, 5.0], 0.2, 0.6, 40.0),
    # twice the observed: squared errors 1 + 4 + 9 + 16 = 30 -> NSE = 1 - 30 / 5 = -5
    # r = 1, spread and mean both doubled -> KGE = 1 - sqrt(2), PBIAS = 100
    (O, [2.0, 4.0, 6.0, 8.0], -5.0, 1 - math.sqrt(2), 100.0),
]


class TestMetrics(unittest.TestCase):

    def test_hand_checked_values(self):
        for fname in FILES:
            metrics = load_metrics(fname)
            for o, s, nse, kge, pbias in CASES:
                with self.subTest(script=fname, sim=s):
                    m = metrics(o, s)
                    # 06 rounds NSE and KGE to 3 decimals and PBIAS to 1
                    self.assertAlmostEqual(m["NSE"], nse, places=3)
                    self.assertAlmostEqual(m["KGE"], kge, places=3)
                    self.assertAlmostEqual(m["PBIAS"], pbias, places=1)

    def test_missing_days_are_skipped(self):
        # the NaN day is dropped from both series, leaving the "1 too high" case above
        o = [1.0, 2.0, np.nan, 3.0, 4.0]
        s = [2.0, 3.0, 9.0, 4.0, 5.0]
        for fname in FILES:
            with self.subTest(script=fname):
                m = load_metrics(fname)(o, s)
                self.assertAlmostEqual(m["NSE"], 0.2, places=3)
                self.assertAlmostEqual(m["KGE"], 0.6, places=3)
                self.assertAlmostEqual(m["PBIAS"], 40.0, places=1)

    def test_lstm_metrics_reports_day_count(self):
        m = load_metrics("06_lstm_model.py")([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, np.nan, 5.0])
        self.assertEqual(m["n"], 2)


if __name__ == "__main__":
    unittest.main()
