import sys, time
import numpy as np
sys.path.insert(0, 'python')
import mojo_minhash as mh

rng = np.random.default_rng(0)


def med(fn, n=5, r=7):
    fn()
    b = 1e9
    for _ in range(r):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        b = min(b, (time.perf_counter() - t) / n)
    return b * 1e3


sh = rng.integers(0, 2 ** 32, size=4096, dtype=np.uint64)
for h in (64, 128, 256, 512):
    co = np.stack([rng.integers(0, 2 ** 32 - 1, size=h, dtype=np.uint64)] * 2, axis=1).astype(np.int64)
    print(h, *[round(med(lambda w=w: mh.calculate_signature(sh, co, workers=w)), 3) for w in (1, 2, 4, 8)])
