"""Shared scaffolding for the parity tests.

Every public function is checked against the *installed* upstream `minhash`
(0.1.1, `.pixi/envs/default/lib/python3.13/site-packages/minhash.py`) rather
than against a restatement of its formulas, so a reference bug is reproduced
instead of being papered over.

Tolerance: `TOL = 0.0`. That is a written tolerance, not a missing one -- it
says each comparison must agree exactly. It is legitimate here because every
operation on this path is exact: the shingle and signature kernels are integer
arithmetic in uint32/uint64 (the reference does the same in Python's
arbitrary-precision ints), and a score is `count / sig_len` where both sides
divide the same integer by the same integer, so the IEEE double is bit-equal.
A comparison that needed slack would be a parity bug, so `TOL` is applied to
both sides of every numerical assertion via `assert_close`.
"""

import os

import numpy as np
import pytest

import minhash as ref

TOL = 0.0

PRIME = 4294969733


def assert_close(got, expect, atol=TOL, rtol=TOL):
    """Exact-by-default numerical comparison with an explicit tolerance."""
    assert np.allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(expect, dtype=np.float64),
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    ), f"got {got!r}, expected {expect!r}"


@pytest.fixture
def coeffs_json_cleanup():
    """`ref.generate_hash_funcs` writes `coeffs.json` into the cwd; remove it."""
    yield
    if os.path.exists("coeffs.json"):
        os.remove("coeffs.json")


def ref_signature_funcs(pairs):
    """Build the reference's closures from (a, b) pairs.

    This is the same lambda the reference's `func` returns (upstream line 21:
    `return lambda x: (a * x + b) % c`).
    """
    return [(lambda a, b: (lambda x: (a * x + b) % PRIME))(int(a), int(b)) for a, b in pairs]


def ref_signature(shingles, pairs):
    return ref.calculate_signature(shingles, ref_signature_funcs(pairs))


def ref_shingles(words, k):
    return np.array(list(ref.generate_shingles(words, k)), dtype=np.uint32)


def ref_score_rows(sigs):
    """Upstream __main__ line 91: `[approx_jaccard_score(a, sigs[i+1:], 1) ...]`."""
    sigs = np.asarray(sigs)
    return [ref.approx_jaccard_score(a, sigs[i + 1 :], 1) for i, a in enumerate(sigs)]


BINS = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)


def ref_hist(rows):
    """Upstream __main__ lines 95-100, the binning loop, as one count vector."""
    counts = np.zeros(len(BINS) - 1, dtype=np.int64)
    for row in rows:
        counts += np.histogram((np.asarray(row) * 10).astype(int) / 10, BINS)[0]
    return counts


def raises_same_class(exc_type):
    """Assert the port raises the same exception class upstream raised."""
    return pytest.raises(exc_type)
