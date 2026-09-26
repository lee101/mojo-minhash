"""Correctness-gated benchmark for mojo-minhash.

Every case checks the compiled result against the reference before it is timed,
so a kernel regression shows up as a correctness failure rather than as a
suspiciously good number. The reference column is the real `minhash` module
where one exists, and otherwise the fastest vectorised NumPy formulation of
the same computation -- not a Python loop NumPy would never use.

Run: python bench/bench.py
"""

from __future__ import annotations

import pathlib
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import minhash as ref  # noqa: E402
import mojo_minhash as mh  # noqa: E402
from mojo_minhash import _lib  # noqa: E402

PRIME = mh.PRIME


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


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


def bench_shingles(n_tokens: int = 20000, k: int = 5):
    """Shingle hashing against zlib called once per shingle from Python.

    There is no vectorised NumPy adler-32, so `zlib.adler32` in a loop is the
    fastest fair baseline and is exactly what the reference does. Both sides
    get pre-laid-out input: the reference pre-joins its strings, so the Mojo
    side is handed the token spans directly rather than paying to build the
    token offset table.
    """
    rng = np.random.default_rng(3)
    words = _words(rng, n_tokens)
    joined = [" ".join(words[i : i + k]) for i in range(n_tokens - k + 1)]
    buf, off, tlen = _spans(words)

    def theirs():
        return [zlib.adler32(s.encode()) for s in joined]

    got = _lib.adler_shinges_from_spans(buf, off, tlen, k)
    assert got.tolist() == theirs(), "shingle mismatch"
    return (
        f"shingles n={n_tokens} k={k}",
        _time(theirs, 3),
        _time(lambda: _lib.adler_shinges_from_spans(buf, off, tlen, k)),
    )


def bench_shingles_end_to_end(n_tokens: int = 20000, k: int = 5):
    """The whole `generate_shingles` call, token table and all, against zlib."""
    rng = np.random.default_rng(8)
    words = _words(rng, n_tokens)
    joined = [" ".join(words[i : i + k]) for i in range(n_tokens - k + 1)]

    def theirs():
        return [zlib.adler32(s.encode()) for s in joined]

    assert mh.generate_shingles(words, k).tolist() == theirs()
    return (
        f"shingles e2e n={n_tokens}",
        _time(theirs, 3),
        _time(lambda: mh.generate_shingles(words, k)),
    )


def bench_signature(n_hash: int = 128, n_shingles: int = 4096):
    """Signature min-reduce against the vectorised uint64 matrix formulation.

    NumPy cannot do this in int64 -- `a * x` overflows -- but uint64 holds the
    product exactly because a, x < 2^32 implies a * x + b < 2^64. That is the
    fastest fair equivalent and it is what the reference's own numerics allow.
    """
    rng = np.random.default_rng(0)
    shingles = rng.integers(0, 2**32, size=n_shingles, dtype=np.uint64)
    coeffs = _coeffs(rng, n_hash)

    def numpy_sig():
        a = coeffs[:, 0].astype(np.uint64)[:, None]
        b = coeffs[:, 1].astype(np.uint64)[:, None]
        x = shingles.astype(np.uint64)[None, :]
        return ((a * x + b) % np.uint64(PRIME)).min(axis=1)

    got = mh.calculate_signature(shingles, coeffs, workers=1)
    assert np.array_equal(got, numpy_sig().astype(np.int64)), "signature mismatch"
    return (
        f"signature h={n_hash} s={n_shingles}",
        _time(numpy_sig, 3),
        _time(lambda: mh.calculate_signature(shingles, coeffs, workers=1)),
    )


def bench_signature_threads(n_hash: int = 256, n_shingles: int = 4096):
    """The same compute-bound kernel fanned out over eight threads."""
    rng = np.random.default_rng(1)
    shingles = rng.integers(0, 2**32, size=n_shingles, dtype=np.uint64)
    coeffs = _coeffs(rng, n_hash)
    base = mh.calculate_signature(shingles, coeffs, workers=1)
    assert np.array_equal(mh.calculate_signature(shingles, coeffs, workers=8), base)
    return (
        f"signature 8 threads h={n_hash}",
        _time(lambda: mh.calculate_signature(shingles, coeffs, workers=1), 3),
        _time(lambda: mh.calculate_signature(shingles, coeffs, workers=8)),
    )


def bench_signature_reference(n_docs: int = 64, n_hash: int = 64, n_shingles: int = 512):
    """Whole-pipeline signature build against the reference's own Python loops."""
    rng = np.random.default_rng(2)
    pairs = mh.generate_hash_funcs(n_hash, seed=0)
    funcs = [(lambda a, b, c: (lambda x: (a * x + b) % c))(*p, PRIME) for p in pairs]
    shingles = [
        [int(v) for v in rng.integers(0, 2**32, size=n_shingles, dtype=np.uint64)]
        for _ in range(n_docs)
    ]
    sh_arrays = [np.array(s, dtype=np.uint32) for s in shingles]

    def theirs():
        for s in shingles:
            ref.calculate_signature(s, funcs)

    def mine():
        for s in sh_arrays:
            mh.calculate_signature(s, pairs, workers=1)

    got = [mh.calculate_signature(s, pairs, workers=1) for s in sh_arrays]
    expect = [ref.calculate_signature(s, funcs) for s in shingles]
    assert all(np.array_equal(a, b) for a, b in zip(got, expect))
    return (
        f"{n_docs} docs x {n_hash} hashes",
        _time(theirs, 3),
        _time(mine, 3),
    )


def bench_score_matrix(n_docs: int = 400, sig_len: int = 64):
    """Upper-triangular agreement against the reference's per-row count_nonzero."""
    rng = np.random.default_rng(4)
    sigs = rng.integers(0, 64, size=(n_docs, sig_len), dtype=np.int64)

    def theirs():
        return [
            ref.approx_jaccard_score(sigs[i], sigs[i + 1 :], 1) for i in range(n_docs)
        ]

    for a, b in zip(mh.score_matrix(sigs, workers=1), theirs()):
        assert np.array_equal(a, b)
    return (
        f"score matrix {n_docs}x{sig_len}",
        _time(theirs, 3),
        _time(lambda: mh.score_matrix(sigs, workers=1)),
    )

def bench_score_matrix_threads(n_docs: int = 900, sig_len: int = 128):
    """The same scan split over eight threads."""
    rng = np.random.default_rng(5)
    sigs = rng.integers(0, 256, size=(n_docs, sig_len), dtype=np.int64)
    base = mh.score_matrix(sigs, workers=1)
    for a, b in zip(base, mh.score_matrix(sigs, workers=8)):
        assert np.array_equal(a, b)
    return (
        f"score matrix 8 thr {n_docs}x{sig_len}",
        _time(lambda: mh.score_matrix(sigs, workers=1), 3),
        _time(lambda: mh.score_matrix(sigs, workers=8)),
    )



def bench_histogram(n_docs: int = 4000, sig_len: int = 64):
    """Score binning on a pre-flattened score array, against np.histogram.

    Both sides receive the same flat array: the row-to-flat concatenation is a
    NumPy copy and charging it to one side only would not be a comparison of
    the kernel.
    """
    rng = np.random.default_rng(6)
    sigs = rng.integers(0, 64, size=(n_docs, sig_len), dtype=np.int64)
    rows = mh.score_matrix(sigs, workers=1)
    bins = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)
    flat = np.concatenate([(r * 10).astype(int) / 10 for r in rows])

    def theirs():
        return np.histogram(flat, bins)[0]

    got = _lib.score_histogram(flat)
    assert np.array_equal(got, theirs()), "histogram mismatch"
    return (
        f"histogram n={flat.size}",
        _time(theirs, 3),
        _time(lambda: _lib.score_histogram(flat)),
    )


def bench_histogram_from_rows(n_docs: int = 4000, sig_len: int = 64):
    """The same binning, but from the ragged row list `score_matrix` returns.

    The Mojo side pays a NumPy concatenate here that the reference's own
    binning loop also pays, so this is the number a user actually sees.
    """
    rng = np.random.default_rng(7)
    sigs = rng.integers(0, 64, size=(n_docs, sig_len), dtype=np.int64)
    rows = mh.score_matrix(sigs, workers=1)
    bins = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)

    def theirs():
        counts, _ = np.histogram(
            np.concatenate([(r * 10).astype(int) / 10 for r in rows]), bins
        )
        return counts

    assert np.array_equal(mh.score_histogram(rows), theirs())
    return (
        f"histogram from rows n={n_docs}",
        _time(theirs, 3),
        _time(lambda: mh.score_histogram(rows)),
    )


CASES = (
    bench_shingles,
    bench_shingles_end_to_end,
    bench_signature,
    bench_signature_threads,
    bench_signature_reference,
    bench_score_matrix,
    bench_score_matrix_threads,
    bench_histogram,
    bench_histogram_from_rows,
)


def main():
    print(f"{'case':<30}{'reference':>12}{'mojo-minhash':>16}{'ratio':>10}")
    print("-" * 68)
    for fn in CASES:
        label, ref_t, got_t = fn()
        ratio = ref_t / got_t if got_t else float("nan")
        print(f"{label:<30}{ref_t * 1e3:>10.2f}ms{got_t * 1e3:>14.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
