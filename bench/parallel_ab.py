"""Per-worker-count A/B on the shapes the benchmark and real callers use.

Preloads `$LIB_AB_SO` (see bench/parallel_ab.py). Each row is the median of
best-of-7 timings, best-of-7 per run, 5 runs.

    LIB_AB_SO=/tmp/lib_old.so python bench/parallel_ab.py
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

SIG = [(64, 512), (128, 512), (128, 1024), (128, 4096), (256, 4096), (512, 4096)]
JAC = [(100, 64), (200, 64), (700, 64), (900, 128), (3000, 3000)]
WORKERS = (1, 2, 4, 8)


def _one(fn, repeats=7):
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _med(fn, runs=5, repeats=7):
    return statistics.median(_one(fn, repeats) for _ in range(runs))


def main():
    rng = np.random.default_rng(0)

    def coeffs(h):
        return np.stack(
            [rng.integers(0, 2**32 - 1, size=h, dtype=np.uint64)] * 2, axis=1
        ).astype(np.int64)

    print(f"so={_so}  load={os.getloadavg()[0]:.1f}")
    print(f"{'case':<26}{'w=1':>9}{'w=2':>9}{'w=4':>9}{'w=8':>9}")
    for h, s in SIG:
        sh = rng.integers(0, 2**32, size=s, dtype=np.uint64)
        co = coeffs(h)
        row = [_med(lambda: mh.calculate_signature(sh, co, workers=w)) for w in WORKERS]
        print(f"{f'sig h={h} s={s}':<26}" + "".join(f"{v:>9.3f}" for v in row))
    for n, l in JAC:
        sigs = rng.integers(0, 64, size=(n, l), dtype=np.int64)
        reps = 1 if n * l > 10**6 else 5
        row = [
            _med(lambda: mh.score_matrix(sigs, workers=w), runs=3, repeats=reps)
            for w in WORKERS
        ]
        print(f"{f'jac {n}x{l}':<26}" + "".join(f"{v:>9.3f}" for v in row))


if __name__ == "__main__":
    main()
