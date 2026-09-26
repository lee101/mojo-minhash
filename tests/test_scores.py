"""Parity of the upper-triangular agreement scan and the score histogram."""

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh

BINS = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)


def _sig_matrix(n_docs, sig_len, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(n_docs, sig_len), dtype=np.int64)


def _ref_rows(sigs):
    return [
        ref.approx_jaccard_score(sigs[i], sigs[i + 1 :], 1) for i in range(len(sigs))
    ]


def _ref_hist(rows):
    counts = np.zeros(len(BINS) - 1, dtype=np.int64)
    for row in rows:
        counts += np.histogram((np.asarray(row) * 10).astype(int) / 10, BINS)[0]
    return counts


def test_rows_match_the_reference_triangle():
    sigs = _sig_matrix(9, 5, seed=1)
    got = mh.score_matrix(sigs, workers=1)
    expect = _ref_rows(sigs)
    assert len(got) == len(expect) == 9
    for a, b in zip(got, expect):
        # Both sides divide an exact integer count by the same integer, so the
        # IEEE double result is bit-identical, not merely close.
        assert np.array_equal(a, b)


def test_row_lengths_shrink_by_one():
    sigs = _sig_matrix(6, 3, seed=2)
    rows = mh.score_matrix(sigs, workers=1)
    assert [len(r) for r in rows] == [5, 4, 3, 2, 1, 0]


def test_row_i_compares_against_the_next_document_first():
    # Catches an off-by-one that pairs row i with document i instead of i + 1.
    sigs = np.array([[1, 1, 1], [1, 1, 1], [9, 9, 9]], dtype=np.int64)
    rows = mh.score_matrix(sigs, workers=1)
    assert rows[0].tolist() == [1.0, 0.0]
    assert rows[1].tolist() == [0.0]


def test_document_is_never_scored_against_itself():
    sigs = np.array([[5, 6, 7]], dtype=np.int64)
    rows = mh.score_matrix(sigs, workers=1)
    assert len(rows) == 1 and rows[0].size == 0
    sigs = np.array([[5, 6, 7], [5, 6, 7]], dtype=np.int64)
    assert mh.score_matrix(sigs, workers=1)[0].tolist() == [1.0]


def test_partial_agreement_is_the_exact_fraction():
    sigs = np.array([[1, 2, 3, 4], [1, 2, 9, 9]], dtype=np.int64)
    assert mh.score_matrix(sigs, workers=1)[0].tolist() == [0.5]


def test_signature_length_one():
    sigs = np.array([[1], [1], [2], [1]], dtype=np.int64)
    rows = mh.score_matrix(sigs, workers=1)
    assert rows[0].tolist() == [1.0, 0.0, 1.0]
    assert rows[1].tolist() == [0.0, 1.0]
    assert rows[2].tolist() == [0.0]


@pytest.mark.parametrize("workers", [1, 2, 3, 4, 8, 32])
def test_chunking_does_not_change_the_scores(workers):
    # The kernel writes row i at a global offset, so a chunk-local offset is a
    # silent corruption that only shows up once the rows are split up.
    sigs = _sig_matrix(17, 6, seed=4)
    base = mh.score_matrix(sigs, workers=1)
    got = mh.score_matrix(sigs, workers=workers)
    for a, b in zip(base, got):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("n_docs", [2, 3, 8, 33, 64, 65, 127, 128, 129])
def test_triangle_is_exact_at_awkward_document_counts(n_docs):
    sigs = _sig_matrix(n_docs, 4, seed=5)
    got = mh.score_matrix(sigs, workers=1)
    expect = _ref_rows(sigs)
    assert len(got) == n_docs
    for a, b in zip(got, expect):
        assert np.array_equal(a, b)


def test_non_contiguous_input_is_handled():
    sigs = _sig_matrix(5, 3, seed=6)
    strided = np.asfortranarray(sigs)
    got = mh.score_matrix(strided, workers=1)
    for a, b in zip(got, _ref_rows(sigs)):
        assert np.array_equal(a, b)


def test_histogram_matches_the_reference_binning():
    sigs = _sig_matrix(12, 7, seed=7)
    rows = mh.score_matrix(sigs, workers=1)
    assert np.array_equal(mh.score_histogram(rows), _ref_hist(rows))


def test_truncation_is_toward_zero_not_round_half_to_even():
    # 3/20 is exactly 0.15 and 0.15 * 10 is exactly 1.5. NumPy's astype(int)
    # truncates to 1; Mojo's Int64(Float64) cast rounds half to even and would
    # give 2. Only truncation reproduces the reference's bin.
    counts = mh.score_histogram(np.array([3 / 20]))
    assert counts.tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for hits, total in ((7, 20), (11, 20)):
        score = hits / total
        counts = mh.score_histogram(np.array([score]))
        want = min(int((np.float64(score) * 10).astype(int)), 10)
        assert counts.sum() == 1
        assert counts[want] == 1


def test_truncation_separates_0_29_from_0_3():
    # 0.29 * 10 is 2.9 and truncates to 2, while 0.3 * 10 is 3.0000000000000004
    # and truncates to 3, so the two land in adjacent bins. 0.7 and 0.71 both
    # truncate to 7.
    counts = mh.score_histogram(np.array([0.3, 0.7, 0.29, 0.71]))
    assert counts.tolist() == [0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0]


def test_score_of_one_lands_in_the_closed_top_bin():
    counts = mh.score_histogram(np.array([1.0]))
    assert counts.tolist() == [0] * 10 + [1]


def test_zero_score_lands_in_the_first_bin():
    counts = mh.score_histogram(np.array([0.0]))
    assert counts.tolist() == [1] + [0] * 10


def test_histogram_counts_every_score_exactly_once():
    sigs = _sig_matrix(20, 9, seed=8)
    rows = mh.score_matrix(sigs, workers=1)
    n_scores = sum(len(r) for r in rows)
    assert mh.score_histogram(rows).sum() == n_scores
    assert np.array_equal(mh.score_histogram(rows), _ref_hist(rows))


def test_empty_scores_give_an_empty_histogram():
    assert mh.score_histogram(np.empty(0)).tolist() == [0] * 11
    assert mh.score_histogram([]).tolist() == [0] * 11
