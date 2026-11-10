"""Raw-kernel timing for the unroll phase.

Times the exported kernels through the same ctypes bridge `bench/bench.py` uses,
but at real caller sizes and without any correctness gate, so a kernel change can
be measured on its own. Correctness stays with `bench/bench.py` and pytest.

Usage:
  python bench/kernels_micro.py            # one run, prints medians of best-of-k
  python bench/kernels_micro.py --repeat 5 # five runs, prints median of medians
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_minhash as mh  # noqa: E402
from mojo_minhash import _lib  # noqa: E402

PRIME = mh.PRIME


def _coeffs(rng, n_hash):
    return np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64),
        ],
        axis=1,
    ).astype(np.int64)


def _words(rng, n_tokens):
    return [
        "".join(rng.choice(list("abcdefghijklmnop"), size=int(rng.integers(3, 9))))
        for _ in range(n_tokens)
    ]


def _spans(words):
    chunks, off, tlen, pos = [], [], [], 0
    for w in words:
        blob = w.encode()
        chunks.append(blob)
        off.append(pos)
        tlen.append(len(blob))
        pos += len(blob)
    return (
        np.frombuffer(b"".join(chunks), np.uint8),
        np.array(off, dtype=np.int32),
        np.array(tlen, dtype=np.int32),
    )


def _time(fn, repeats=5):
    """Best wall time over `repeats`, in ms."""
    fn()  # warm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _one_run():
    out = {}

    # --- signature kernel, one worker = the kernel alone -------------------
    for h, s in ((128, 4096), (256, 4096), (64, 512)):
        rng = np.random.default_rng(0)
        shingles = rng.integers(0, 2**32, size=s, dtype=np.uint64)
        coeffs = _coeffs(rng, h)

        def sig():
            mh.calculate_signature(shingles, coeffs, workers=1)

        out[f"signature h={h} s={s}"] = _time(sig)

    # --- jaccard kernel, one worker ---------------------------------------
    for n_docs in (400, 700, 900):
        rng = np.random.default_rng(11)
        sig_len = 128 if n_docs == 900 else 64
        sigs = rng.integers(0, 64, size=(n_docs, sig_len), dtype=np.int64)

        def jac():
            _lib.jaccard_scores(sigs, workers=1)

        out[f"jaccard {n_docs}x{sig_len}"] = _time(jac)

    # --- score histogram, flat ---------------------------------------------
    rng = np.random.default_rng(6)
    flat = rng.integers(0, 11, size=7_998_000).astype(np.float64) / 10.0

    def hist():
        _lib.score_histogram(flat)

    out["histogram n=7998000"] = _time(hist)
    small = flat[:126_000].copy()

    def hist_small():
        _lib.score_histogram(small)

    out["histogram n=126000"] = _time(hist_small)

    # --- shingle adler-32 kernel -------------------------------------------
    for n_tokens in (20000, 200):
        rng = np.random.default_rng(3)
        words = _words(rng, n_tokens)
        buf, off, tlen = _spans(words)

        def sh():
            _lib.adler_shinges_from_spans(buf, off, tlen, 5)

        out[f"shingles n={n_tokens}"] = _time(sh)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    runs = [_one_run() for _ in range(args.repeat)]
    keys = list(runs[0])
    print(f"{'kernel':<28}{'ms':>10}")
    print("-" * 38)
    for k in keys:
        val = statistics.median(r[k] for r in runs)
        print(f"{k:<28}{val:>10.3f}")


if __name__ == "__main__":
    main()
