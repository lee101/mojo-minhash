"""Tests for the shingle token-table build (the separator-scan fast path)."""

import random
import zlib

import numpy as np
import pytest

import minhash as ref
import mojo_minhash as mh
from mojo_minhash import _lib


def _ref(words, k):
    return [zlib.adler32(s.encode()) for s in
            (" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1)))]


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_separator_scan_matches_the_reference(k):
    rng = random.Random(17)
    for trial in range(20):
        words = [
            "".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 9)))
            for _ in range(rng.randint(1, 40))
        ]
        assert mh.generate_shingles(words, k).tolist() == _ref(words, k)


def test_a_token_containing_a_space_falls_back_and_still_matches():
    # " ".join cannot tell ["ab cd", "ef"] from ["ab", "cd", "ef"], and neither
    # can the reference; the fast path must detect the extra separator and fall
    # back rather than silently produce the wrong spans.
    words = ["ab cd", "ef", "ghij"]
    for k in (1, 2, 3, 4):
        assert mh.generate_shingles(words, k).tolist() == _ref(words, k)


def test_separator_scan_agrees_with_the_per_token_table():
    rng = random.Random(5)
    for trial in range(20):
        words = [
            "".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 7)))
            for _ in range(rng.randint(2, 30))
        ]
        blobs = [w.encode() for w in words]
        tlen = np.fromiter((len(b) for b in blobs), dtype=np.int32, count=len(blobs))
        off = np.zeros(len(blobs), dtype=np.int32)
        np.cumsum(tlen[:-1], out=off[1:])
        per_token = _lib.adler_shinges_from_spans(
            np.frombuffer(b"".join(blobs), np.uint8), off, tlen, 2
        )
        assert np.array_equal(mh.generate_shingles(words, 2), per_token)


def test_single_token_and_empty_inputs():
    assert mh.generate_shingles(["solo"], 1).tolist() == [zlib.adler32(b"solo")]
    assert mh.generate_shingles([], 2).tolist() == [zlib.adler32(b"")]
    assert mh.generate_shingles(["a", "b"], 2).tolist() == [zlib.adler32(b"a b")]


def test_multibyte_tokens_are_hashed_as_bytes():
    words = ["naïve", "café", "über", "日本語"]
    for k in (1, 2, 3):
        assert mh.generate_shingles(words, k).tolist() == _ref(words, k)


def test_a_token_that_is_only_spaces():
    # The buffer is all separators, so the fast path must decline.
    words = [" ", " ", "x"]
    for k in (1, 2, 3):
        assert mh.generate_shingles(words, k).tolist() == _ref(words, k)


def test_large_document_matches_the_reference():
    rng = random.Random(23)
    words = [
        "".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 9)))
        for _ in range(4000)
    ]
    assert mh.generate_shingles(words, 5).tolist() == _ref(words, 5)