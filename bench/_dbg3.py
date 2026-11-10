import sys
import numpy as np
sys.path.insert(0, 'python')
import mojo_minhash as mh

rng = np.random.default_rng(20261110)
h, s = 1024, 4096
sh = rng.integers(0, 2 ** 32, size=s, dtype=np.uint64)
co = np.stack([rng.integers(0, 2 ** 32 - 1, size=h, dtype=np.uint64)] * 2, axis=1).astype(np.int64)
a = mh.calculate_signature(sh, co, workers=1)
b = mh.calculate_signature(sh, co, workers=4)
d = np.nonzero(a != b)[0]
print("ndiff", d.size, "first", d[:10])
for i in d[:5]:
    # python min over suffix from each block start
    res = [(int(co[i, 0]) * int(x) + int(co[i, 1])) % mh.PRIME for x in sh]
    print(i, a[i], b[i], min(res), res.index(min(res)))
