"""Degenerate inputs, checked against what upstream `minhash` does with them.

Each test below first *runs upstream* on the degenerate input and records the
result, then asserts the port either returns the same value (within `TOL`) or
raises the same exception class. Where the port deliberately reports
something upstream silently produces (a NaN score, or a ValueError instead of
an undefinable empty minimum), the test says so and pins both sides, so the
deviation stays visible instead of drifting.

Upstream line references are to the installed
`.pixi/envs/default/lib/python3.13/site-packages/minhash.py` (0.1.1).

    generate_shingles      12-17   maxlen deque over words[:count]
    generate_hash_funcs    19-25   sample + pop pairing, writes coeffs.json
    calculate_signature    27-28   min(map(hash, shingles)), empty list -> []
    approx_jaccard_score   39-40   count_nonzero(a == b, axis) / len(a)
    binning loop           93-100  (row*10).astype(int)/10 vs BINS
"""

import random

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh
from mojo_minhash import _lib

from parity_helpers import (
    PRIME,
    TOL,
    assert_close,
    coeffs_json_cleanup,
    ref_hist,
    ref_score_rows,
    ref_shingles,
    ref_signature,
)

WORDS = "the quick brown fox jumps over the lazy dog".split()


def _consume_if_generator(result):
    """Upstream `generate_shingles` is lazy; `list(...)` is what makes it run."""
    import types

    if isinstance(result, types.GeneratorType):
        return list(result)
    return result


def _same_exception_upstream(fn_ref, fn_port, *args, **kwargs):
    """Upstream raises here; the port must raise the same class."""
    with pytest.raises(Exception) as upstream:
        _consume_if_generator(fn_ref(*args, **kwargs))
    with pytest.raises(type(upstream.value)):
        fn_port(*args, **kwargs)
    return type(upstream.value)


# --------------------------------------------------------------------------
# generate_shingles: empty, one element, duplicates, invalid k
# --------------------------------------------------------------------------


def test_empty_document_matches_upstream():
    # Upstream line 13: `deque([], maxlen=count)` -> `" ".join([]) == ""`.
    expect = list(ref.generate_shingles([], 2))
    got = mh.generate_shingles([], 2)
    assert expect == [1]
    assert got.tolist() == expect


def test_empty_document_matches_upstream_for_every_k():
    for k in (1, 2, 3, 64):
        assert mh.generate_shingles([], k).tolist() == list(ref.generate_shingles([], k))


def test_one_element_document_matches_upstream():
    for words in (["solo"], ["a"], [""], ["two words"]):
        for k in (1, 2, 3):
            assert mh.generate_shingles(words, k).tolist() == list(
                ref.generate_shingles(words, k)
            )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_duplicate_tokens_match_upstream(k):
    # Upstream hashes positions, not a deduplicated set: ["a", "a"] y-shingles
    # are both adler32(b"a") (for k=1) or adler32(b"a a") (k=2), repeated.
    words = ["a", "a", "a", "a"]
    got = mh.generate_shingles(words, k)
    expect = ref_shingles(words, k)
    assert got.tolist() == expect.tolist()
    if k == 1:
        assert len(set(got.tolist())) == 1


def test_empty_string_tokens_match_upstream():
    # Zero-length tokens are legal and hash as their joined surroundings.
    for words in (["", ""], ["", "a", ""], ["a", "", "b"]):
        for k in (1, 2, 3):
            assert mh.generate_shingles(words, k).tolist() == list(
                ref.generate_shingles(words, k)
            )


def test_negative_count_raises_the_same_class_as_upstream():
    # Upstream line 13: `deque(words[:count], maxlen=-1)` raises
    # `ValueError: maxlen must be non-negative` (verified by running it). A
    # single token keeps the first `" ".join` simple; note that with two or
    # more tokens upstream would raise TypeError from the join first, because
    # `words[:-1]` is still a list of tokens.
    exc = _same_exception_upstream(
        ref.generate_shingles, mh.generate_shingles, ["solo"], -1
    )
    assert exc is ValueError


def test_zero_count_is_a_documented_deviation():
    # Upstream does NOT raise for count=0: `maxlen=0` is legal, every join is
    # the empty string, so it yields adler32(b"") once per token plus one --
    # `[1]` for `[]` and `[1, 1]` for `["a"]` (upstream lines 13-17). The port
    # treats a zero-width shingle as invalid, which
    # `test_shingles.py::test_k_zero_is_rejected` has always pinned. Record both
    # sides so the deviation cannot silently spread.
    assert list(ref.generate_shingles([], 0)) == [1]
    assert list(ref.generate_shingles(["a"], 0)) == [1, 1]
    with pytest.raises(ValueError, match="k must be >= 1"):
        mh.generate_shingles([], 0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        mh.generate_shingles(["a"], 0)


# --------------------------------------------------------------------------
# calculate_signature: empty, one element, duplicates, mismatched shapes
# --------------------------------------------------------------------------


def test_empty_hash_list_matches_upstream():
    # Upstream line 28: the comprehension runs zero times, whatever the
    # shingles, so the result is an empty array -- no ValueError from `min`.
    shingles = list(ref.generate_shingles(WORDS, 2))
    expect = ref.calculate_signature(
        shingles, []
    )
    got = mh.calculate_signature(shingles, [])
    assert expect.shape == got.shape == (0,)
    assert got.size == 0


def test_empty_shingle_list_raises_the_same_class_as_upstream():
    # Upstream line 28: `min(map(hash, []))` raises `ValueError: min()
    # iterable argument is empty` (verified by running it).
    pairs = np.array([[1, 2]], dtype=np.int64)
    exc = _same_exception_upstream(ref_signature, mh.calculate_signature, [], pairs)
    assert exc is ValueError


def test_one_shingle_matches_upstream():
    pairs = np.array([[3, 4], [2**32 - 1, 2**32 - 1]], dtype=np.int64)
    got = mh.calculate_signature([7], pairs)
    expect = ref_signature([7], pairs)
    assert got.shape == expect.shape == (2,)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_all_duplicate_shingles_match_upstream():
    pairs = np.array([[12345, 67], [89, 101112]], dtype=np.int64)
    got = mh.calculate_signature([42] * 50, pairs)
    expect = ref_signature([42] * 50, pairs)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_zero_shingle_matches_upstream():
    # x == 0 gives the bare offset b mod prime, checked upstream's way.
    pairs = np.array([[7, 9], [0, 2**32 - 1]], dtype=np.int64)
    got = mh.calculate_signature([0, 0, 0], pairs)
    expect = ref_signature([0, 0, 0], pairs)
    assert got.tolist() == [9 % PRIME, (2**32 - 1) % PRIME]
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_mismatched_shingle_shape_raises_the_same_class_as_upstream():
    # Upstream hashes the *tokens*, not a nested array: a 2-D array of ints
    # reaches `" ".join` with a row as an element and raises TypeError
    # (verified by running it). The port's `generate_shingles` must raise the
    # same class for that input rather than producing a shingle.
    two_d = np.array([[1, 2], [3, 4]], dtype=np.int64)
    exc = _same_exception_upstream(ref.generate_shingles, mh.generate_shingles, two_d, 2)
    assert exc is TypeError


def test_non_string_tokens_raise_the_same_class_as_upstream():
    # Upstream's `" ".join(words[i : i + count])` raises TypeError on an int
    # token (verified by running it: "expected str instance, int found").
    exc = _same_exception_upstream(
        ref.generate_shingles, mh.generate_shingles, [1, 2, 3], 2
    )
    assert exc is TypeError


def test_ragged_coeff_shape_is_rejected_like_a_mismatched_shape():
    # The port's contract for hash_funcs is an (n, 2) array; upstream's
    # equivalent mismatched call is a wrong-length list of closures, which
    # still yields one output per closure. The port instead rejects anything
    # that is not (n, 2) -- ValueError, the same class upstream raises for the
    # malformed-input calls above.
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        mh.calculate_signature([1, 2, 3], np.zeros((4,), dtype=np.int64))


# --------------------------------------------------------------------------
# approx_jaccard_score: empty, one element, mismatched shapes, NaN
# --------------------------------------------------------------------------


def test_approx_jaccard_score_empty_upstream_is_nan_port_raises():
    # Upstream line 40: `np.count_nonzero(...) / len(a)` is `0 / 0` -> nan,
    # with a RuntimeWarning. The port raises instead, which
    # test_ffi_safety.py::test_approx_jaccard_score_rejects_empty_and_mismatched_signatures
    # already pins. Both sides are asserted here so the deviation is explicit:
    # a caller must not rely on a NaN silently propagating out of the port.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expect = ref.approx_jaccard_score(np.zeros(0), np.zeros(0))
    assert np.isnan(expect)
    with pytest.raises(ValueError, match="at least one position"):
        mh.approx_jaccard_score(np.zeros(0), np.zeros(0))


def test_approx_jaccard_score_mismatched_lengths_raise_the_same_class():
    # Upstream line 40: `a == b` broadcasts and raises
    # `ValueError: operands could not be broadcast together`.
    a = np.zeros(3, dtype=np.int64)
    b = np.zeros(4, dtype=np.int64)
    exc = _same_exception_upstream(ref.approx_jaccard_score, mh.approx_jaccard_score, a, b)
    assert exc is ValueError


def test_approx_jaccard_score_nan_agreement_matches_upstream():
    # Upstream compares positions, and nan == nan is False, so the score is a
    # clean fraction. The port differs in mechanism -- it casts NaN to its
    # int64 signature domain, where NaN becomes INT_MIN -- so *after* the cast
    # the two NaNs are equal and the score is 1.0, not upstream's 0.5. Pinned,
    # not blessed: signatures are integer hashes and cannot hold NaN, so the
    # port's domain is the cast values, and parity must not be claimed for the
    # pre-cast comparison.
    import warnings

    a = np.array([np.nan, 1.0])
    b = np.array([np.nan, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expect = ref.approx_jaccard_score(a, b)
        got = mh.approx_jaccard_score(a, b)
        a_int = a.astype(np.int64)
        b_int = b.astype(np.int64)
        expect_int = ref.approx_jaccard_score(a_int, b_int)
    assert expect == 0.5
    assert a_int[0] == b_int[0] == np.iinfo(np.int64).min
    assert got == 1.0
    # The port does agree with upstream on the *cast* domain, which is the one
    # it declares: an int64 NaN sentinel compared as an integer.
    assert expect_int == got


def test_approx_jaccard_score_one_element_matches_upstream():
    for x, y in ((5, 5), (5, 6)):
        assert_close(
            mh.approx_jaccard_score([x], [y]),
            ref.approx_jaccard_score([x], [y]),
            atol=TOL,
            rtol=TOL,
        )


# --------------------------------------------------------------------------
# score_matrix: empty, one document, mismatched shapes
# --------------------------------------------------------------------------


def test_score_matrix_empty_and_single_document_match_upstream():
    zero = np.zeros((0, 4), dtype=np.int64)
    assert mh.score_matrix(zero, workers=1) == ref_score_rows(zero) == []
    one = np.array([[1, 2, 3, 4]], dtype=np.int64)
    got = mh.score_matrix(one, workers=1)
    expect = ref_score_rows(one)
    assert len(got) == len(expect) == 1 and got[0].size == expect[0].size == 0


def test_score_matrix_zero_length_signature_matches_upstream_shape():
    # An (n, 0) matrix: upstream's `count_nonzero` over an empty axis divided
    # by `len(a) == 0` is NaN per pair (upstream line 40), and the shape of the
    # rows is still n_docs rows of shrinking length. The port refuses the
    # division outright (the kernel would divide by zero into a raw buffer);
    # this is the documented deviation pinned in
    # test_ffi_safety.py::test_score_matrix_rejects_a_zero_length_signature.
    import warnings

    sigs = np.zeros((2, 0), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expect = ref_score_rows(sigs)
    assert [len(r) for r in expect] == [1, 0]
    assert np.isnan(expect[0]).all()
    with pytest.raises(ValueError, match="at least one signature position"):
        mh.score_matrix(sigs, workers=1)


def test_score_matrix_one_dimensional_input_raises_like_a_mismatch():
    # A 1-D array has no documents; upstream's axis=1 call on it broadcasts
    # against an empty slice and returns nothing per document, so the shape
    # mismatch is the port's to reject. ValueError, the class upstream raises
    # for shape mismatches (line 40).
    with pytest.raises(ValueError, match="2-D"):
        mh.score_matrix(np.arange(6, dtype=np.int64), workers=1)


# --------------------------------------------------------------------------
# score_histogram: NaN/inf, -0.0, empty rows
# --------------------------------------------------------------------------


def test_score_histogram_drops_non_finite_scores_exactly_like_np_histogram():
    # Upstream line 98 casts with `.astype(int)`, which maps NaN and +/-inf to
    # INT_MIN; np.histogram then ignores them because they fall left of the
    # first edge. Run it rather than assuming the cast.
    scores = [np.nan, np.inf, -np.inf, 0.0, 0.5, 1.0]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expect = np.histogram(
            (np.asarray(scores) * 10).astype(int) / 10,
            (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42),
        )[0]
    got = mh.score_histogram(np.asarray(scores, dtype=np.float64))
    assert_close(got, expect, atol=TOL, rtol=TOL)
    assert int(expect.sum()) == 3 and int(np.count_nonzero(~np.isfinite(scores))) == 3


def test_score_histogram_negative_zero_matches_np_histogram():
    scores = np.array([-0.0], dtype=np.float64)
    expect = np.histogram(
        (scores * 10).astype(int) / 10,
        (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42),
    )[0]
    got = mh.score_histogram(scores)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_score_histogram_score_below_first_edge_depends_on_truncation():
    # Upstream truncates *before* binning: -0.05 * 10 truncates to 0, so it is
    # counted in bin 0, while -0.5 truncates to -5 and is dropped. Compared
    # against np.histogram's view of the same truncated values.
    scores = np.array([-0.05, -0.5], dtype=np.float64)
    truncated = (scores * 10).astype(int) / 10
    expect = np.histogram(
        truncated, (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)
    )[0]
    assert truncated.tolist() == [-0.0, -0.5]
    got = mh.score_histogram(scores)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_score_histogram_empty_rows_match_the_reference_loop():
    for rows in ([], np.empty(0), [np.empty(0), np.empty(0)]):
        got = mh.score_histogram(rows)
        expect = ref_hist(rows if isinstance(rows, list) else [rows])
        assert_close(got, expect, atol=TOL, rtol=TOL)


def test_score_histogram_of_upstream_rows_matches_upstream_histogram():
    # The full __main__ path on a tiny synthetic corpus: signatures via the
    # port, rows via upstream line 91, histogram on both sides. Tolerance is
    # TOL = 0.0 because each score is the same integer count over the same
    # integer length.
    rng = np.random.default_rng(31)
    sigs = rng.integers(0, 3, size=(9, 11), dtype=np.int64)
    rows_ref = ref_score_rows(sigs)
    rows_port = mh.score_matrix(sigs, workers=1)
    for a, b in zip(rows_port, rows_ref):
        assert_close(a, b, atol=TOL, rtol=TOL)
    assert_close(mh.score_histogram(rows_port), ref_hist(rows_ref), atol=TOL, rtol=TOL)
