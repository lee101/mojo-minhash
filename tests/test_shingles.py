"""Parity of the compiled adler-32 shingler against `minhash.generate_shingles`."""

import random
import zlib

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh

WORDS = "the quick brown fox jumps over the lazy dog".split()


def _ref_shingles(words, k):
    return np.array(list(ref.generate_shingles(words, k)), dtype=np.uint32)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8, 9, 10, 64])
def test_matches_reference_for_every_window_width(k):
    got = mh.generate_shingles(WORDS, k)
    assert np.array_equal(got, _ref_shingles(WORDS, k))
    assert got.size == max(1, len(WORDS) - k + 1)


def test_short_document_yields_one_window():
    # The reference builds a maxlen deque from words[:k] and then iterates
    # words[k:], which is empty, so it yields exactly one shingle.
    words = ["alpha", "beta"]
    assert mh.generate_shingles(words, 5).tolist() == _ref_shingles(words, 5).tolist()
    assert mh.generate_shingles(words, 5).size == 1


def test_empty_document_hashes_the_empty_string():
    assert mh.generate_shingles([], 2).tolist() == [zlib.adler32(b"")]
    assert mh.generate_shingles([], 2).tolist() == [1]


def test_window_width_one_hashes_each_token_alone():
    got = mh.generate_shingles(WORDS, 1)
    assert got.tolist() == [zlib.adler32(w.encode()) for w in WORDS]


def test_window_is_the_joined_string_not_a_byte_slide():
    # A 2-shingle of ["ab", "c"] must equal the 1-shingle of the single token
    # "ab c": the kernel reassembles the joined string. A kernel that slid by
    # one byte instead of one token would produce neither.
    assert mh.generate_shingles(["ab", "c"], 2).tolist() == mh.generate_shingles(
        ["ab c"], 1
    ).tolist()
    assert mh.generate_shingles(["ab", "c"], 2)[0] == zlib.adler32(b"ab c")
    # ...and the 1-shingles stay per-token.
    assert mh.generate_shingles(["ab", "c"], 1).tolist() != mh.generate_shingles(
        ["ab c"], 1
    ).tolist()


def test_separator_is_exactly_one_space():
    got = mh.generate_shingles(["x", "y"], 2)
    assert got.tolist() == [zlib.adler32(b"x y")]
    assert got.tolist() != [zlib.adler32(b"xy")]


def test_multibyte_tokens_are_hashed_as_bytes():
    words = ["naïve", "café", "über", "日本語"]
    for k in (1, 2, 3, 4):
        assert mh.generate_shingles(words, k).tolist() == _ref_shingles(words, k).tolist()
    # a dropped or mis-sliced multi-byte boundary changes the byte string
    assert mh.generate_shingles(words, 2)[0] == zlib.adler32("naïve café".encode())


def test_long_random_documents_match_the_reference():
    rng = random.Random(7)
    for trial in range(12):
        words = [
            "".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 9)))
            for _ in range(rng.randint(1, 60))
        ]
        k = rng.randint(1, 5)
        assert mh.generate_shingles(words, k).tolist() == _ref_shingles(words, k).tolist()


@pytest.mark.parametrize("n_tok", [1, 2, 3, 5, 8, 13, 31, 32, 33, 63, 64, 65, 127])
def test_token_counts_around_vector_widths(n_tok):
    # Windows that straddle a hardware vector width are where a dropped tail
    # would show up; a count that is an exact multiple of the width is not.
    words = [f"w{i}" for i in range(n_tok)]
    assert mh.generate_shingles(words, 2).tolist() == _ref_shingles(words, 2).tolist()


def test_k_zero_is_rejected():
    with pytest.raises(ValueError):
        mh.generate_shingles(WORDS, 0)
