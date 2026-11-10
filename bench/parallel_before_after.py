"""Old vs new, at the packaged default, on the sizes that matter.

Run the same script against two shared libraries by preloading the .so (see
bench/parallel_ab.py). Every call uses the wrapper's default worker count, so
this is what a caller actually gets.

    LIB_AB_SO=/tmp/lib_old.so       python bench/parallel_before_after.py
    LIB_AB_SO=/tmp/lib_par_probe.so python bench/parallel_before_after.py
"""

import ctypes
import os
import pathlib
import statistics
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_so = os.environ.get("LIB_AB_SO")
if _so:
    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)

import mojo_minhash as mh  # noqa: E402
from mojo_minhash import _lib  # noqa: E402

SIG = [(64, 512), (128, 4096), (256, 4096), (512, 4096)]
JAC = [(400, 64), (700, 64), (900, 128)]
HIST = [7_998_000]


def _med(fn, n, r=7):
    fn()
    b = float("inf")
    for _ in range(r):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        b = min(b, (time.perf_counter() - t) / n)
    return b * 1e3


def main():
    rng = np.random.default_rng(0)
    print(f"so={_so} load={os.getloadavg()[0]:.1f}")
    print(f"{'case':<26}{'ms/call':>10}")
    for h, s in SIG:
        sh = rng.integers(0, 2**32, size=s, dtype=np.uint64)
        co = np.stack(
            [rng.integers(0, 2**32 - 1, size=h, dtype=np.uint64)] * 2, axis=1
        ).astype(np.int64)
        print(f"{f'sig h={h} s={s}':<26}{_med(lambda: mh.calculate_signature(sh, co), 5):>10.3f}")
    for n, l in JAC:
        sigs = rng.integers(0, 64, size=(n, l), dtype=np.int64)
        print(f"{f'jac {n}x{l}':<26}{_med(lambda: mh.score_matrix(sigs), 3):>10.3f}")
    for n in HIST:
        flat = rng.integers(0, 11, size=n).astype(np.float64) / 10.0
        print(f"{f'hist n={n}':<26}{_med(lambda: _lib.score_histogram(flat), 3):>10.3f}")


if __name__ == "__main__":
    main()
