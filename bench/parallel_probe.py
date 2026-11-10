import sys
import statistics
import time
import numpy as np
sys.path.insert(0, 'python')
import mojo_minhash as mh


def _time(fn, repeats):
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _median(fn, repeats, runs=5):
    return statistics.median(_time(fn, repeats) for _ in range(runs))


SIZES = {
    "signature": [(16, 64), (32, 256), (64, 512), (128, 512), (128, 1024),
                  (128, 4096), (256, 4096), (512, 4096), (1024, 4096)],
    "jaccard": [(32, 32), (64, 64), (100, 64), (200, 64), (400, 64), (700, 64), (900, 128)],
    "histogram": [2016, 126_000, 7_998_000],
}


def _coeffs(rng, n_hash):
    return np.stack(
        [rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64)] * 2, axis=1
    ).astype(np.int64)


def main():
    rng = np.random.default_rng(0)
    print(f"load {__import__('os').getloadavg()[0]:.1f}")
    print(f"{'case':<26}{'serial':>10}{'default':>10}{'speedup':>10}")
    for h, s in SIZES["signature"]:
        sh = rng.integers(0, 2**32, size=s, dtype=np.uint64)
        co = _coeffs(rng, h)
        saved = _lib.DEFAULT_WORKERS
        _lib.DEFAULT_WORKERS = 1
        try:
            t1 = _median(lambda: mh.calculate_signature(sh, co), 5)
        finally:
            _lib.DEFAULT_WORKERS = saved
        tp = _median(lambda: mh.calculate_signature(sh, co), 5)
        print(f"{f'signature h={h} s={s}':<26}{t1:>10.3f}{tp:>10.3f}{t1 / tp:>9.2f}x")
    for n, l in SIZES["jaccard"]:
        sigs = rng.integers(0, 64, size=(n, l), dtype=np.int64)
        saved = _lib.DEFAULT_WORKERS
        _lib.DEFAULT_WORKERS = 1
        try:
            t1 = _median(lambda: mh.score_matrix(sigs), 3)
        finally:
            _lib.DEFAULT_WORKERS = saved
        tp = _median(lambda: mh.score_matrix(sigs), 3)
        print(f"{f'jaccard {n}x{l}':<26}{t1:>10.3f}{tp:>10.3f}{t1 / tp:>9.2f}x")
    for n in SIZES["histogram"]:
        flat = rng.integers(0, 11, size=n).astype(np.float64) / 10.0
        t1 = _median(lambda: _lib.score_histogram(flat), 3)
        print(f"{f'histogram n={n}':<26}{t1:>10.3f}{t1:>10.3f}{1.0:>9.2f}x")


if __name__ == "__main__":
    import mojo_minhash._lib as _lib
    main()
