"""Per-function parity: every public function against the upstream reference.

`tests/test_shingles.py`, `test_signature.py`, `test_scores.py` already compare
the compiled path with `minhash` on a handful of inputs. This file closes the
gaps that leaves, function by function:

    generate_shingles       every k, multi-byte text, one/duplicate tokens
    generate_hash_funcs     the draw recovered from upstream's own closures,
                            across seeds and counts, including count >= 1 edge
    calculate_signature     upstream's closures, duplicate inputs, dtype
    approx_jaccard_score    axis=0 against upstream's own line
    score_matrix            upstream's axis=1 blocks, the n_docs 0/1 edges
    score_histogram         upstream's `np.histogram` loop, NaN/inf, -0.0

Each numerical comparison goes through `assert_close`, whose tolerance
`TOL = 0.0` is written down and justified in `parity_helpers`.
"""

import random

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh

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

# A document with duplicates, a repeated token and a case pair, which is where
# a shingler that de-duplicated or reordered would diverge from upstream.
DUP_WORDS = "a b a b c a b".split()


# --------------------------------------------------------------------------
# generate_shingles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64])
@pytest.mark.parametrize("words", [WORDS, DUP_WORDS, ["solo"], ["a", "a", "a"]])
def test_generate_shingles_matches_upstream(words, k):
    got = mh.generate_shingles(words, k)
    expect = ref_shingles(words, k)
    assert got.dtype == expect.dtype
    assert got.shape == expect.shape
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_generate_shingles_default_count_matches_upstream():
    # Explicit default-argument parity: upstream's `count=2`.
    import inspect

    assert inspect.signature(ref.generate_shingles).parameters["count"].default == 2
    assert inspect.signature(mh.generate_shingles).parameters["count"].default == 2
    assert_close(mh.generate_shingles(WORDS), ref_shingles(WORDS, 2), atol=TOL, rtol=TOL)


@pytest.mark.parametrize(
    "words",
    [
        [],
        ["solo"],
        ["a", "a"],
        ["a", "a", "a", "a"],
        ["", ""],
        ["x", "", "y"],
        ["ab cd", "ef"],
        ["naïve", "café", "über", "日本語"],
        [" "],
        [" ", " "],
    ],
)
@pytest.mark.parametrize("k", [1, 2, 3, 5, 99])
def test_generate_shingles_matches_upstream_on_degenerate_documents(words, k):
    got = mh.generate_shingles(words, k)
    expect = ref_shingles(words, k)
    assert got.shape == expect.shape
    assert_close(got, expect, atol=TOL, rtol=TOL)


# --------------------------------------------------------------------------
# generate_hash_funcs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 7, 42])
def test_generate_hash_funcs_draw_matches_upstreams_own_closures(count, coeffs_json_cleanup):
    # Upstream returns closures, not pairs. Its own closures are the reference
    # for the draw: b = fn(0), and a * 1 + b == fn(1), so `fn(1) - b` is `a`
    # modulo the prime. Recovering (a, b) from upstream and comparing with the
    # port's pairs is a direct check of `random.sample` range and pop order.
    for seed in (0, 1, 12345):
        random.seed(seed)
        funcs = ref.generate_hash_funcs(count)
        recovered = []
        for fn in funcs:
            b = fn(0)
            recovered.append(((fn(1) - b) % PRIME, int(b)))
        got = [(int(a), int(b)) for a, b in mh.generate_hash_funcs(count, seed=seed)]
        assert got == recovered
        # And the drawn coefficients really do define a hash family, so this
        # is not comparing two empty lists.
        assert len(got) == count
        assert all(0 <= a < 2**32 - 1 and 0 <= b < 2**32 - 1 for a, b in got)


def test_generate_hash_funcs_uses_upstreams_sample_range(coeffs_json_cleanup):
    # Upstream line 22: `random.sample(range(2**32 - 1), count * 2)`. Same seed,
    # same call, same list: compare the raw draw rather than the paired view.
    random.seed(7)
    raw = random.sample(range(2**32 - 1), 2 * 9)
    random.seed(7)
    ref.generate_hash_funcs(9)
    assert len(raw) == 18
    got = mh.generate_hash_funcs(9, seed=7)
    expected_pairs = []
    scratch = list(raw)
    for _ in range(9):
        expected_pairs.append((scratch.pop(), scratch.pop()))
    assert [(int(a), int(b)) for a, b in got] == expected_pairs


def test_generate_hash_funcs_zero_count_matches_upstream(coeffs_json_cleanup):
    # Upstream `generate_hash_funcs(0)` samples nothing and returns `[]`. The
    # port documents `count >= 1` (ValueError) as a deliberate deviation, so
    # the comparison here is of the *reference* behaviour only; see
    # test_degenerate_upstream for the deviation record.
    random.seed(0)
    assert ref.generate_hash_funcs(0) == []
    with pytest.raises(ValueError, match="count must be >= 1"):
        mh.generate_hash_funcs(0, seed=0)


# --------------------------------------------------------------------------
# calculate_signature
# --------------------------------------------------------------------------


def _pairs(n_hash, seed):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
        ],
        axis=1,
    )


@pytest.mark.parametrize("doc_words", [WORDS, DUP_WORDS, ["solo"], ["a", "a", "a"]])
def test_calculate_signature_matches_upstreams_closures(doc_words):
    # Upstream line 28: `np.array([min(map(hash, shingles)) for hash in hash_funcs])`.
    shingles = list(ref.generate_shingles(doc_words, 2))
    pairs = mh.generate_hash_funcs(24, seed=0)
    got = mh.calculate_signature(shingles, pairs)
    expect = ref_signature(shingles, pairs)
    assert got.dtype == expect.dtype
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_calculate_signature_duplicate_shingles_match_upstream():
    # Duplicate positions cannot change a minimum, but a kernel that indexed
    # by position rather than comparing values would notice them.
    shingles = [5, 5, 5, 5, 7, 5]
    pairs = _pairs(9, 1)
    got = mh.calculate_signature(shingles, pairs)
    expect = ref_signature(shingles, pairs)
    assert_close(got, expect, atol=TOL, rtol=TOL)
    # All-equal shingles collapse every hash to that one residue.
    same = mh.calculate_signature([11, 11, 11], pairs)
    expect_same = ref_signature([11, 11, 11], pairs)
    assert_close(same, expect_same, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("value", [0, 2**32 - 1])
def test_calculate_signature_range_endpoints_match_upstream(value):
    pairs = _pairs(7, value + 3)
    got = mh.calculate_signature([value] * 3, pairs)
    expect = ref_signature([value] * 3, pairs)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_calculate_signature_one_shingle_one_hash_matches_upstream():
    # Upstream's smallest non-degenerate call: one hash, one shingle.
    pairs = np.array([[3, 4]], dtype=np.int64)
    got = mh.calculate_signature([2], pairs)
    expect = ref_signature([2], pairs)
    assert got.shape == expect.shape == (1,)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_calculate_signature_empty_hash_list_matches_upstream():
    # Upstream line 28 builds an empty list comprehension over zero hash_funcs,
    # so it returns an empty array without ever touching the shingles. The port
    # must return an empty result, not raise on the degenerate coeff shape.
    got = mh.calculate_signature([1, 2, 3], [])
    expect = ref_signature([1, 2, 3], [])
    assert got.shape == expect.shape
    assert got.size == 0
    assert_close(got, expect, atol=TOL, rtol=TOL)


# --------------------------------------------------------------------------
# approx_jaccard_score
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_approx_jaccard_score_axis0_matches_upstream(seed):
    # Upstream line 40: `np.count_nonzero(a == b, axis) / len(a)`.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    a = rng.integers(0, 3, size=n, dtype=np.int64)
    b = rng.integers(0, 3, size=n, dtype=np.int64)
    got = mh.approx_jaccard_score(a, b)
    expect = ref.approx_jaccard_score(a, b)
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_approx_jaccard_score_axis0_default_matches_upstream():
    # The default axis is the one upstream's own default uses (`axis=0`).
    import inspect

    assert inspect.signature(ref.approx_jaccard_score).parameters["axis"].default == 0
    assert inspect.signature(mh.approx_jaccard_score).parameters["axis"].default == 0
    a = np.array([1, 2, 3, 4], dtype=np.int64)
    b = np.array([1, 9, 3, 9], dtype=np.int64)
    assert_close(
        mh.approx_jaccard_score(a, b),
        ref.approx_jaccard_score(a, b),
        atol=TOL,
        rtol=TOL,
    )


def test_approx_jaccard_score_identical_and_disjoint():
    a = np.arange(16, dtype=np.int64)
    assert_close(
        mh.approx_jaccard_score(a, a), ref.approx_jaccard_score(a, a), atol=TOL, rtol=TOL
    )
    b = a + 1000
    assert_close(
        mh.approx_jaccard_score(a, b),
        ref.approx_jaccard_score(a, b),
        atol=TOL,
        rtol=TOL,
    )


def test_approx_jaccard_score_axis1_is_score_matrix():
    # The port routes the block form to `score_matrix` (documented deviation);
    # the reference's axis=1 is exactly upstream __main__ line 91, which is
    # what `score_matrix` reproduces. Compare through the reference block
    # rather than only against itself.
    rng = np.random.default_rng(12)
    sigs = rng.integers(0, 4, size=(7, 5), dtype=np.int64)
    got = mh.score_matrix(sigs, workers=1)
    expect = ref_score_rows(sigs)
    assert len(got) == len(expect)
    for a, b in zip(got, expect):
        assert_close(a, b, atol=TOL, rtol=TOL)


# --------------------------------------------------------------------------
# score_matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_docs", [2, 3, 5, 16, 17])
def test_score_matrix_matches_upstream_blocks(n_docs):
    rng = np.random.default_rng(n_docs)
    sigs = rng.integers(0, 3, size=(n_docs, 6), dtype=np.int64)
    got = mh.score_matrix(sigs, workers=1)
    expect = ref_score_rows(sigs)
    assert len(got) == len(expect)
    for a, b in zip(got, expect):
        assert_close(a, b, atol=TOL, rtol=TOL)


def test_score_matrix_with_zero_documents_matches_upstream():
    sigs = np.zeros((0, 5), dtype=np.int64)
    assert len(mh.score_matrix(sigs, workers=1)) == len(ref_score_rows(sigs)) == 0


def test_score_matrix_with_one_document_matches_upstream():
    sigs = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    got = mh.score_matrix(sigs, workers=1)
    expect = ref_score_rows(sigs)
    assert len(got) == len(expect) == 1
    assert got[0].size == expect[0].size == 0


# --------------------------------------------------------------------------
# score_histogram
# --------------------------------------------------------------------------


def test_score_histogram_matches_upstream_binning_loop():
    # Upstream __main__ lines 95-100. Build rows upstream's own way (line 91),
    # then bin them upstream's own way.
    rng = np.random.default_rng(21)
    sigs = rng.integers(0, 4, size=(15, 17), dtype=np.int64)
    rows = ref_score_rows(sigs)
    got = mh.score_histogram(rows)
    expect = ref_hist(rows)
    assert got.dtype == np.int64
    assert got.shape == expect.shape
    assert_close(got, expect, atol=TOL, rtol=TOL)


@pytest.mark.parametrize(
    "scores",
    [
        [0.0],
        [1.0],
        [0.05, 0.15, 0.29, 0.3, 0.7, 0.71, 1.0],
        [-0.0],
        [1 / 3, 2 / 3, 1 / 7],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [float("nan"), 0.5, float("inf"), -0.5, 1.0],
        [-0.5, -0.05, 0.0],
    ],
)
def test_score_histogram_on_edge_scores_matches_np_histogram(scores):
    # The reference's bin step is exactly `np.histogram(...)` on a per-row
    # basis, so for a single row it is one call. The cast of NaN/inf warns,
    # and that warning is part of the reference behaviour being reproduced.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expect = np.histogram(
            (np.asarray(scores, dtype=np.float64) * 10).astype(int) / 10,
            (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42),
        )[0]
    got = mh.score_histogram(np.asarray(scores, dtype=np.float64))
    assert_close(got, expect, atol=TOL, rtol=TOL)


def test_score_histogram_counts_duplicate_scores_once_each():
    scores = [0.5] * 7 + [0.25] * 3
    expect = np.histogram(
        (np.asarray(scores) * 10).astype(int) / 10,
        (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42),
    )[0]
    got = mh.score_histogram(np.asarray(scores))
    assert_close(got, expect, atol=TOL, rtol=TOL)
    assert int(got.sum()) == len(scores)
