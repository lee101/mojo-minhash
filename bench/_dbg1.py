import sys, time
sys.path.insert(0, 'python')
import numpy as np
import mojo_minhash as mh

rng = np.random.default_rng(0)
s = rng.integers(0, 2 ** 32, size=4096, dtype=np.uint64)
co = np.stack([rng.integers(0, 2 ** 32 - 1, size=64, dtype=np.uint64)] * 2, axis=1).astype(np.int64)
for w in (1, 1, 2, 2):
    t = time.perf_counter()
    mh.calculate_signature(s, co, workers=w)
    print('sig w', w, round((time.perf_counter() - t) * 1e3, 3))
sigs = rng.integers(0, 64, size=(400, 64), dtype=np.int64)
for w in (1, 1, 2, 2):
    t = time.perf_counter()
    mh.score_matrix(sigs, workers=w)
    print('jac w', w, round((time.perf_counter() - t) * 1e3, 3))
