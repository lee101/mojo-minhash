"""Parity of the compiled signature min-reduce against `minhash.calculate_signature`.

The reference computes `(a * x + b) % prime` in CPython's arbitrary-precision
integers, so parity here is exact, not approximate: the kernel's 16-bit limb
reduction must land on the same residue. Every assertion below uses
`rtol=0, atol=0` because the operation is exact integer arithmetic.
"""

import random

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh
from mojo_minhash import _lib

PRIME = mh.PRIME


def _ref_signature(shingles, pairs):
    funcs = [(lambda a, b, c: (lambda x: (a * x + b) % c))(a, b, PRIME) for a, b in pairs]
    return ref.calculate_signature(shingles, funcs)


def test_matches_reference_on_a_real_document():
    words = "the quick brown fox jumps over the lazy dog".split()
    shingles = list(ref.generate_shingles(words, 2))
    pairs = mh.generate_hash_funcs(24, seed=0)
    assert np.array_equal(
        mh.calculate_signature(shingles, pairs), _ref_signature(shingles, pairs)
    )


def test_coefficient_draw_matches_the_reference_draw():
    random.seed(0)
    funcs = ref.generate_hash_funcs(12)
    import os

    if os.path.exists("coeffs.json"):
        os.remove("coeffs.json")
    recovered = []
    for fn in funcs:
        b = fn(0)
        recovered.append(((fn(1) - b) % PRIME, b))
    assert recovered == mh.generate_hash_funcs(12, seed=0)


def test_single_shingle_pins_the_reduction_exactly():
    # With one shingle the minimum is that one hash value, so this compares the
    # kernel's limb reduction against CPython term for term.
    for x in (0, 1, 2**32 - 2, 2**32 - 1, 4294967295, 123456789):
        for a, b in [(0, 0), (1, 0), (0, 1), (2**32 - 2, 2**32 - 2), (2**32 - 1, 0)]:
            got = mh.calculate_signature([x], [(a, b)], workers=1)
            assert got.tolist() == [(a * x + b) % PRIME]


def test_product_above_2_pow_63_does_not_wrap():
    # a * x alone exceeds int64 here, so an int64 intermediate would wrap to a
    # negative residue and fail loudly.
    a, x, b = 4_000_000_000, 4_000_000_000, 4_000_000_000
    got = mh.calculate_signature([x], [(a, b)], workers=1)
    assert got.tolist() == [(a * x + b) % PRIME]
    assert a * x + b > 2**63


def test_random_shingles_and_coefficients_match_python_int_arithmetic():
    rng = np.random.default_rng(20260926)
    for _ in range(40):
        n_s = int(rng.integers(1, 300))
        n_h = int(rng.integers(1, 20))
        shingles = rng.integers(0, 2**32, size=n_s, dtype=np.uint64)
        pairs = np.stack(
            [
                rng.integers(0, 2**32 - 1, size=n_h, dtype=np.uint64),
                rng.integers(0, 2**32 - 1, size=n_h, dtype=np.uint64),
            ],
            axis=1,
        ).astype(np.int64)
        expect = np.array(
            [min((int(a) * int(v) + int(b)) % PRIME for v in shingles) for a, b in pairs],
            dtype=np.int64,
        )
        assert np.array_equal(mh.calculate_signature(shingles, pairs, workers=1), expect)


@pytest.mark.parametrize("workers", [1, 2, 3, 5, 8, 64])
def test_chunking_does_not_change_the_result(workers):
    # A chunk bound placed one hash early or late would show up here.
    rng = np.random.default_rng(3)
    shingles = rng.integers(0, 2**32, size=97, dtype=np.uint64)
    pairs = np.stack(
        [
            rng.integers(0, 2**32 - 1, size=13, dtype=np.uint64),
            rng.integers(0, 2**32 - 1, size=13, dtype=np.uint64),
        ],
        axis=1,
    ).astype(np.int64)
    base = mh.calculate_signature(shingles, pairs, workers=1)
    assert np.array_equal(mh.calculate_signature(shingles, pairs, workers=workers), base)


def test_minimum_not_maximum():
    # A min-reduce that quietly became a max-reduce still returns *a* residue
    # for every hash, so it has to be pinned against the true minimum.
    rng = np.random.default_rng(11)
    shingles = rng.integers(0, 2**32, size=64, dtype=np.uint64)
    pairs = mh.generate_hash_funcs(8, seed=5)
    got = mh.calculate_signature(shingles, pairs, workers=1)
    for h, (a, b) in enumerate(pairs):
        vals = [(a * int(v) + b) % PRIME for v in shingles]
        assert got[h] == min(vals)
        assert got[h] != max(vals)


def test_empty_shinge_list_raises_like_min():
    with pytest.raises(ValueError):
        mh.calculate_signature([], mh.generate_hash_funcs(2, seed=0))


def test_out_of_range_operands_are_rejected_before_the_kernel():
    pairs = mh.generate_hash_funcs(1, seed=0)
    with pytest.raises(ValueError):
        _lib.signature(np.array([0], dtype=np.uint32), np.array([[2**32, 0]], dtype=np.int64))
    with pytest.raises(ValueError):
        _lib.signature(
            np.array([0], dtype=np.uint32),
            np.array([[-1, 0]], dtype=np.int64),
        )
    assert pairs  # keeps the fixture referenced


def test_output_dtype_matches_the_reference():
    pairs = mh.generate_hash_funcs(4, seed=1)
    got = mh.calculate_signature([1, 2, 3], pairs, workers=1)
    assert got.dtype == np.int64
    assert got.shape == (4,)
