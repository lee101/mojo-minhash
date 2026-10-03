"""Tests for the signature kernel's blocked inner loop.

`mh_signature` runs HASH_BLOCKS hash columns per shingle so the 64-bit `div`
has independent chains to interleave with. That path has a width-dependent tail:
a hash count that is not a multiple of the block, and a shingle count that is
short, both take the scalar loop. Every assertion here is exact equality,
because the operation is exact integer arithmetic.
"""

import random

import numpy as np
import pytest

import mojo_minhash as mh
from mojo_minhash import _lib

PRIME = mh.PRIME


def _pairs(n_hash, rng):
    return np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64),
        ],
        axis=1,
    ).astype(np.int64)


def _python_min(shingles, coeffs, n_hash):
    return np.array(
        [
            min((int(coeffs[h, 0]) * int(v) + int(coeffs[h, 1])) % PRIME for v in shingles)
            for h in range(n_hash)
        ],
        dtype=np.int64,
    )


@pytest.mark.parametrize("n_hash", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17])
def test_hash_counts_around_the_block_width_are_exact(n_hash):
    # A hash count that is an exact multiple of the block, one more, and one
    # fewer are where a dropped tail would show up.
    rng = np.random.default_rng(n_hash)
    shingles = rng.integers(0, 2**32, size=257, dtype=np.uint64)
    coeffs = _pairs(n_hash, rng)
    assert np.array_equal(
        mh.calculate_signature(shingles, coeffs, workers=1), _python_min(shingles, coeffs, n_hash)
    )


@pytest.mark.parametrize("n_shingles", [1, 2, 3, 4, 5, 8, 9, 16, 17, 33])
def test_shingle_counts_around_the_block_width_are_exact(n_shingles):
    rng = np.random.default_rng(1000 + n_shingles)
    shingles = rng.integers(0, 2**32, size=n_shingles, dtype=np.uint64)
    coeffs = _pairs(8, rng)
    assert np.array_equal(
        mh.calculate_signature(shingles, coeffs, workers=1), _python_min(shingles, coeffs, 8)
    )


def test_extreme_coefficients_are_exact():
    # Coefficients at both ends of the allowed range drive the limb reduction to
    # its extremes: products just below 2**64, and products of zero.
    rng = np.random.default_rng(99)
    shingles = rng.integers(0, 2**32, size=400, dtype=np.uint64)
    for coeffs in (
        np.full((6, 2), 2**32 - 2, dtype=np.int64),
        np.zeros((6, 2), dtype=np.int64),
        np.array([[1, 0], [0, 1], [2**32 - 2, 0], [0, 2**32 - 2]], dtype=np.int64),
        np.array([[1, 2**32 - 2], [2**32 - 2, 1]], dtype=np.int64),
    ):
        got = mh.calculate_signature(shingles, coeffs, workers=1)
        assert np.array_equal(got, _python_min(shingles, coeffs, len(coeffs)))


def test_minimum_is_found_across_the_whole_shingle_range():
    # The winning element of the min sits at the start, the middle and the end,
    # so a truncated inner loop cannot pass.
    rng = np.random.default_rng(4)
    for pos in (0, 1, 63, 64, 127, 511):
        shingles = rng.integers(1, 2**31, size=512, dtype=np.uint64)
        coeffs = _pairs(8, rng)
        coeffs[3] = [1, 0]
        shingles = shingles.astype(np.uint64).copy()
        shingles[pos] = np.uint64(1)
        got = mh.calculate_signature(shingles, coeffs, workers=1)
        assert got[3] == 1
        assert np.array_equal(got, _python_min(shingles, coeffs, 8))


def test_chunked_and_whole_calls_agree():
    # The kernel writes res[h] at a global offset, so a chunk-local offset is a
    # silent corruption that only appears once the hashes are split up.
    rng = np.random.default_rng(11)
    # The kernel reads shingles as UInt32, so the raw buffer handed to it here
    # must be uint32 as well; the wrapper's `np.ascontiguousarray` is what
    # normally guarantees that. A uint64 array would be read as 2n packed words
    # and produce wrong values for every column, which would mask the offset
    # behaviour this test is about.
    shingles = rng.integers(0, 2**32, size=1000, dtype=np.uint32)
    coeffs = _pairs(64, rng)
    whole = mh.calculate_signature(shingles, coeffs, workers=1)

    out = np.zeros(64, dtype=np.uint64)
    for lo in range(0, 64, 7):
        hi = min(lo + 7, 64)
        chunk = np.full(64, 0xDEADBEEF, dtype=np.uint64)
        _lib.lib.mh_signature(
            shingles.ctypes.data, coeffs.ctypes.data, shingles.size, 64, PRIME,
            chunk.ctypes.data, lo, hi,
        )
        # A call only writes [lo, hi); the sentinel below lo is still in
        # `chunk`, so copying the whole prefix would clobber earlier chunks.
        out[lo:hi] = chunk[lo:hi]
    assert np.array_equal(out, whole)

    # The sentinel also proves the kernel wrote exactly the requested columns:
    # nothing below `lo` and nothing at or past `hi` may be touched.
    for lo in (0, 7, 28, 63):
        hi = min(lo + 7, 64)
        chunk = np.full(64, 0xDEADBEEF, dtype=np.uint64)
        _lib.lib.mh_signature(
            shingles.ctypes.data, coeffs.ctypes.data, shingles.size, 64, PRIME,
            chunk.ctypes.data, lo, hi,
        )
        assert np.all(chunk[:lo] == 0xDEADBEEF)
        assert np.all(chunk[hi:] == 0xDEADBEEF)
        assert np.array_equal(chunk[lo:hi], whole[lo:hi])


@pytest.mark.parametrize("workers", [1, 2, 3, 5, 8, 64])
def test_workers_do_not_change_the_result(workers):
    rng = random.Random(3)
    shingles = rng.sample(range(2**32), 500)
    coeffs = _pairs(33, np.random.default_rng(7))
    assert np.array_equal(
        mh.calculate_signature(shingles, coeffs, workers=workers),
        _python_min(np.array(shingles, dtype=np.uint64), coeffs, 33),
    )


def test_reduction_is_exact_for_every_coefficient_class():
    # A sweep over the coefficient magnitude classes: the limb reduction must
    # agree with Python's arbitrary-precision integers in all of them.
    rng = np.random.default_rng(20260930)
    shingles = rng.integers(0, 2**32, size=1500, dtype=np.uint64)
    for shift in (0, 8, 16, 20, 24, 28, 31, 32):
        coeffs = _pairs(6, rng)
        coeffs = np.right_shift(coeffs, 32 - shift).astype(np.int64)
        got = mh.calculate_signature(shingles, coeffs, workers=1)
        assert np.array_equal(got, _python_min(shingles, coeffs, 6))


def test_agreement_kernel_handles_awkward_signature_lengths():
    # `_agree` is a scalar loop over the signature and the score kernel divides
    # by its length, so lengths around a hardware width are the interesting ones.
    for sig_len in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33):
        rng = np.random.default_rng(sig_len)
        sigs = rng.integers(0, 5, size=(6, sig_len), dtype=np.int64)
        got = mh.score_matrix(sigs, workers=1)
        expect = [
            np.count_nonzero(sigs[i] == sigs[i + 1:], axis=1) / sig_len
            for i in range(6)
        ]
        assert len(got) == len(expect)
        for a, b in zip(got, expect):
            assert np.array_equal(a, b)