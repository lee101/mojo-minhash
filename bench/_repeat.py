import sys
import time
import numpy as np
sys.path.insert(0, 'python')
import mojo_minhash as mh


def med(fn, n=5, r=9):
    fn()
    vals = []
    for _ in range(r):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        vals.append((time.perf_counter() - t) / n)
    vals.sort()
    return [round(v * 1e3, 3) for v in (vals[0], vals[len(vals) // 2])]


rng = np.random.default_rng(0)
sh = rng.integers(0, 2 ** 32, size=4096, dtype=np.uint64)
for h in (128, 256, 512):
    co = np.stack([rng.integers(0, 2 ** 32 - 1, size=h, dtype=np.uint64)] * 2, axis=1).astype(np.int64)
    out = {}
    for w in (1, 2, 4):
        out[w] = med(lambda: mh.calculate_signature(sh, co, workers=w))
    print(h, {w: out[w] for w in out})
