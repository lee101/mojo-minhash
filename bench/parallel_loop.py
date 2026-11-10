"""10 identical calls in one timed loop -- the cost a real caller pays.

`_time` in the other benches times one call at a time; if a single ctypes ->
Mojo -> `parallelize` round trip is expensive on this host, a per-call timer
charges it to every call anyway. This loop amortises Python's timer and shows
whether the floor is in the call or in the measurement.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_minhash as mh  # noqa: E402

rng = np.random.default_rng(0)
sh = rng.integers(0, 2 ** 32, size=4096, dtype=np.uint64)
co = np.stack([rng.integers(0, 2 ** 32 - 1, size=64, dtype=np.uint64)] * 2, axis=1).astype(np.int64)
sigs = rng.integers(0, 64, size=(400, 64), dtype=np.int64)


def bench(label, fn, n=20):
    fn()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    print(f"{label:<34}{best * 1e3:>9.3f} ms/call")


for w in (1, 2, 4, 8):
    bench(f"signature 64x4096 workers={w}", lambda w=w: mh.calculate_signature(sh, co, workers=w))
for w in (1, 2, 4, 8):
    bench(f"jaccard 400x64 workers={w}", lambda w=w: mh.score_matrix(sigs, workers=w))
