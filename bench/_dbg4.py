import sys
import numpy as np
sys.path.insert(0, 'python')
import mojo_minhash as mh

h, s = 1024, 4096
p = mh.PRIME
rng = np.random.default_rng(1)


def python_min(shingles, coeffs):
    x = shingles.astype(np.uint64)[None, :]
    a = coeffs[:, 0].astype(np.uint64)[:, None]
    b = coeffs[:, 1].astype(np.uint64)[:, None]
    return ((a * x + b) % np.uint64(p)).min(axis=1).astype(np.int64)


for trial in range(6):
    sh = rng.integers(0, 2 ** 32, size=s, dtype=np.uint64)
    co = np.stack([rng.integers(0, 2 ** 32 - 1, size=h, dtype=np.uint64)] * 2, axis=1).astype(np.int64)
    serial = mh.calculate_signature(sh, co, workers=1)
    par = mh.calculate_signature(sh, co, workers=4)
    py = python_min(sh, co)
    print("trial", trial, "serial==py", np.array_equal(serial, py),
          "par==py", np.array_equal(par, py), "par==serial", np.array_equal(par, serial),
          "ndiff", int((par != serial).sum()))
