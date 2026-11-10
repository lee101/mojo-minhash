"""Regression tests for the memory-safety and error-path defects found in review.

Each test here corresponds to a specific failure that was reachable through the
public API. They are grouped by the layer that owned the bug, because that is
the layer that has to stay fixed:

* `tests/test_kernels.py` covers the kernels themselves.
* This file covers the ctypes bridge, which is where the inputs are validated
  before any address is handed to compiled code.

The bridge bugs were all of the same shape: the kernel trusts the lengths and
offsets it is given, because there is no way to check them once a raw address
has crossed the ABI. An out-of-range value is an out-of-bounds read, not an
exception, so every one of these has to be caught in Python.
"""

import random

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh
from mojo_minhash import _lib

# --------------------------------------------------------------------------
# Empty inputs must not reach the kernels with a degenerate shape
# --------------------------------------------------------------------------


def test_score_matrix_of_zero_documents_is_empty():
    assert _lib.jaccard_scores(np.zeros((0, 5), dtype=np.int64)) == []


def test_score_matrix_of_one_document_has_no_cells():
    rows = _lib.jaccard_scores(np.zeros((1, 4), dtype=np.int64))
    assert len(rows) == 1
    assert rows[0].size == 0


def test_score_matrix_rejects_a_zero_length_signature():
    # The kernel divides by sig_len. Zero writes an undefined double rather
    # than raising, and the reference's count/len would be NaN.
    with pytest.raises(ValueError, match="signature position"):
        _lib.jaccard_scores(np.zeros((3, 0), dtype=np.int64))


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_signature_of_zero_hashes_is_an_empty_array(workers):
    got = _lib.signature(
        np.array([1, 2, 3], dtype=np.uint32),
        np.zeros((0, 2), dtype=np.int64),
        workers=workers,
    )
    assert got.shape == (0,)
    assert got.dtype == np.int64


def test_histogram_rejects_zero_bins():
    # n_bins == 0 makes the kernel's top-bin index -1, so every count landed
    # one int64 *before* the output buffer. That was a reproducible heap
    # corruption, which glibc only reported later as "corrupted size vs.
    # prev_size" on an unrelated allocation.
    with pytest.raises(ValueError, match="n_bins"):
        _lib.score_histogram(np.array([0.5]), 10.0, 0)


def test_histogram_of_no_scores_is_all_zero():
    assert _lib.score_histogram(np.empty(0)).tolist() == [0] * 11
    assert _lib.score_histogram([]).tolist() == [0] * 11


# --------------------------------------------------------------------------
# Empty documents must follow the reference
# --------------------------------------------------------------------------


def test_empty_document_rejects_k_zero():
    # Verified against the reference: k=0 does not raise there. `deque(x,
    # maxlen=0)` is legal, every join is of an empty deque, so upstream yields
    # adler32(b"") once per token plus one -- [1] for [] and [1, 1] for ["a"].
    #
    # This package treats a zero-width shingle as invalid instead, and
    # `test_shingles.py::test_k_zero_is_rejected` has always pinned that for
    # non-empty documents. The defect was that an *empty* document short-
    # circuited before the check and returned [1], so the same k=0 call raised
    # for one input and succeeded for the other.
    assert list(ref.generate_shingles([], 0)) == [1]
    assert list(ref.generate_shingles(["a"], 0)) == [1, 1]

    with pytest.raises(ValueError, match="k must be >= 1"):
        mh.generate_shingles([], 0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        mh.generate_shingles(["a"], 0)


@pytest.mark.parametrize("words", [[], ["a"], ["a", "b"]])
def test_k_zero_is_rejected_regardless_of_document_length(words):
    with pytest.raises(ValueError, match="k must be >= 1"):
        mh.generate_shingles(words, 0)


def test_empty_document_still_hashes_the_empty_string_for_valid_k():
    import zlib

    assert mh.generate_shingles([], 2).tolist() == [zlib.adler32(b"")]


# --------------------------------------------------------------------------
# Silent dtype narrowing
# --------------------------------------------------------------------------


def test_signature_rejects_shingles_that_do_not_fit_in_uint32():
    # 2**32 truncated to uint32 is 0, which silently changed the residue:
    # (1 * 2**40) % PRIME is 4294345861, not 0.
    with pytest.raises(ValueError, match=r"\[0, 2\*\*32\)"):
        _lib.signature(np.array([2**40], dtype=np.int64), np.array([[1, 0]], dtype=np.int64))


def test_signature_rejects_negative_shingles():
    with pytest.raises(ValueError, match=r"\[0, 2\*\*32\)"):
        _lib.signature(np.array([-1], dtype=np.int64), np.array([[1, 0]], dtype=np.int64))


def test_signature_rejects_out_of_range_coefficients():
    with pytest.raises(ValueError, match=r"\[0, 2\*\*32\)"):
        _lib.signature(
            np.array([0], dtype=np.uint32), np.array([[2**32, 0]], dtype=np.int64)
        )


def test_signature_accepts_the_boundary_values():
    got = _lib.signature(
        np.array([0, 2**32 - 1], dtype=np.uint64),
        np.array([[2**32 - 1, 2**32 - 1]], dtype=np.int64),
    )
    assert got.tolist() == [((2**32 - 1) * (2**32 - 1) + (2**32 - 1)) % mh.PRIME]


# --------------------------------------------------------------------------
# Span tables must describe the buffer they index
# --------------------------------------------------------------------------


def _spans(buf, off, tlen, k=1):
    return _lib.adler_shinges_from_spans(
        np.frombuffer(buf, np.uint8),
        np.array(off, dtype=np.int64),
        np.array(tlen, dtype=np.int64),
        k,
    )


def test_span_offset_past_the_end_is_rejected():
    # off[1] + tlen[1] ran past the 6-byte buffer: an out-of-bounds read.
    with pytest.raises(ValueError, match="past the end"):
        _spans(b"abcdef", [0, 5], [1, 4])


def test_span_offsets_that_overflow_int32_are_rejected():
    # These are int64 inputs, but the kernel reads the table as Int32, so
    # 2**31 wrapped to -2**31 and the kernel dereferenced a wild address --
    # a hard segfault before the check existed.
    with pytest.raises(ValueError, match="int32"):
        _spans(b"abcdefgh", [2**31, 2**31 + 4], [4, 4])


def test_span_length_that_overflows_int32_is_rejected():
    with pytest.raises(ValueError, match="int32"):
        _spans(b"abcdefgh", [0, 0], [2**32, 2])


def test_mismatched_span_table_lengths_are_rejected():
    with pytest.raises(ValueError, match="entries but tlen"):
        _spans(b"abcdef", [0, 2], [2])


def test_negative_span_values_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _spans(b"abcdef", [-1, 2], [2, 2])
    with pytest.raises(ValueError, match="non-negative"):
        _spans(b"abcdef", [0, 2], [-1, 2])


def test_two_dimensional_span_tables_are_rejected():
    buf = np.frombuffer(b"abcdef", np.uint8)
    with pytest.raises(ValueError, match="1-D"):
        _lib.adler_shinges_from_spans(
            buf, np.zeros((1, 2), dtype=np.int64), np.zeros((1, 2), dtype=np.int64), 1
        )


def test_no_token_spans_still_hash_the_empty_string():
    import zlib

    assert _spans(b"", [], [], k=2).tolist() == [zlib.adler32(b"")]


def test_zero_length_tokens_are_hashed_as_empty_strings():
    # A token that hashes to the empty string is legal and must not be
    # confused with a missing span: " ".join of two empty tokens is " ",
    # which is a space, not an empty byte string.
    import zlib

    assert _spans(b"   ", [0, 1, 2], [0, 0, 1], k=1).tolist() == [
        zlib.adler32(b""),
        zlib.adler32(b""),
        zlib.adler32(b" "),
    ]


def test_a_multi_dimensional_buffer_is_flattened_not_rejected():
    buf = np.frombuffer(b"abcdef", np.uint8).reshape(2, 3)
    got = _lib.adler_shinges_from_spans(
        buf,
        np.array([0, 3], dtype=np.int32),
        np.array([3, 3], dtype=np.int32),
        1,
    )
    assert got.tolist() == [
        __import__("zlib").adler32(b"abc"),
        __import__("zlib").adler32(b"def"),
    ]


# --------------------------------------------------------------------------
# approx_jaccard_score is delegated to NumPy, so it needs its own checks
# --------------------------------------------------------------------------


def test_approx_jaccard_score_rejects_empty_and_mismatched_signatures():
    with pytest.raises(ValueError):
        mh.approx_jaccard_score(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError):
        mh.approx_jaccard_score(
            np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64)
        )


def test_approx_jaccard_score_matches_the_reference():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 8, size=17, dtype=np.int64)
    b = rng.integers(0, 8, size=17, dtype=np.int64)
    assert mh.approx_jaccard_score(a, b) == ref.approx_jaccard_score(a, b)


# --------------------------------------------------------------------------
# generate_hash_funcs validates the prime it accepts but does not use
# --------------------------------------------------------------------------


def test_generate_hash_funcs_rejects_an_unusable_prime():
    with pytest.raises(ValueError, match="33 bits"):
        mh.generate_hash_funcs(2, prime=0, seed=0)
    with pytest.raises(ValueError, match="33 bits"):
        mh.generate_hash_funcs(2, prime=2**33, seed=0)


def test_generate_hash_funcs_still_draws_independently_of_the_prime(coeffs_json_cleanup):
    # Documented behaviour: the draw is a plain random.sample, so the prime
    # argument cannot change the coefficients. Checked against upstream's draw
    # whenever the prime is the upstream default, which is the only prime
    # upstream's closures can be rebuilt for.
    assert mh.generate_hash_funcs(4, seed=0) == mh.generate_hash_funcs(
        4, prime=12345, seed=0
    )
    random.seed(0)
    funcs = ref.generate_hash_funcs(4)
    prime = ref.generate_hash_funcs.__defaults__[1]
    assert prime == mh.PRIME
    recovered = [((fn(1) - fn(0)) % prime, int(fn(0))) for fn in funcs]
    assert [(int(a), int(b)) for a, b in mh.generate_hash_funcs(4, seed=0)] == recovered