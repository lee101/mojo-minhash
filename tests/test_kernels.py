"""Direct tests of each exported kernel.

The rest of the suite reaches the kernels through the public API, which is the
right way to test behaviour but leaves a kernel free to be mis-wired and still
ship green, because a wrong `argtypes` or a wrong column bound shows up as the
same "the numbers disagree" failure no matter which layer is at fault. These
tests call every `@export`ed symbol by name, with buffers laid out exactly the
way the kernel documents, and assert on the values it writes:

    mh_shingle_adler32   the adler-32 window kernel
    mh_signature         the blocked universal-hash min-reduce
    mh_jaccard_rows      the upper-triangular agreement scan
    mh_score_hist        the truncated-score histogram

Any symbol added to `src/kernels.mojo` must gain a call here; the check is one
grep and it is cheap to run:

    for s in $(grep -hoE '@export\\("[A-Za-z_0-9]+"\\)' src/*.mojo \\
                | sed 's/@export("//;s/")//'); do \\
      grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"; done

Every buffer here is kept alive by a local name for the duration of the call.
Passing `arr.ctypes.data` to a ctypes call is only safe because Python holds
the array until the interpreter exits; keeping a reference is what makes that
guarantee local instead of accidental.
"""

import numpy as np
import pytest

from mojo_minhash import _lib

lib = _lib.lib
PRIME = _lib.PRIME


def test_every_export_has_a_test_in_this_file():
    # Guards the file itself: if a kernel is added to the .mojo and this
    # assertion is left alone, it fails here rather than in review.
    source = (_lib._ROOT / "src" / "kernels.mojo").read_text()
    exported = {
        line.split('"')[1]
        for line in source.splitlines()
        if line.strip().startswith("@export(")
    }
    assert exported == {
        "mh_shingle_adler32",
        "mh_signature",
        "mh_jaccard_rows",
        "mh_score_hist",
    }


# --------------------------------------------------------------------------
# mh_shingle_adler32
# --------------------------------------------------------------------------


def test_mh_shingle_adler32_writes_zlib_adler32_of_each_window():
    import zlib

    words = ["alpha", "beta", "gamma", "delta"]
    buf = np.frombuffer(" ".join(words).encode(), np.uint8)
    off = np.array([0, 6, 11, 17], dtype=np.int32)
    tlen = np.array([5, 4, 5, 5], dtype=np.int32)
    n_tok, k = 4, 2
    out = np.zeros(n_tok - k + 1, dtype=np.uint32)

    lib.mh_shingle_adler32(
        buf.ctypes.data, off.ctypes.data, tlen.ctypes.data, n_tok, k, out.ctypes.data
    )

    want = [
        zlib.adler32(" ".join(words[i : i + k]).encode())
        for i in range(n_tok - k + 1)
    ]
    assert out.tolist() == want


def test_mh_shingle_adler32_short_document_hashes_every_token():
    import zlib

    # n_tokens < k: the kernel hashes one window holding every token, which is
    # the document's own adler-32.
    buf = np.frombuffer(b"aa bb cc", np.uint8)
    off = np.array([0, 3, 6], dtype=np.int32)
    tlen = np.array([2, 2, 2], dtype=np.int32)
    out = np.zeros(1, dtype=np.uint32)

    lib.mh_shingle_adler32(
        buf.ctypes.data, off.ctypes.data, tlen.ctypes.data, 3, 7, out.ctypes.data
    )
    assert out.tolist() == [zlib.adler32(b"aa bb cc")]


def test_mh_shingle_adler32_empty_document_is_one():
    import zlib

    buf = np.frombuffer(b"", np.uint8)
    off = np.zeros(0, dtype=np.int32)
    tlen = np.zeros(0, dtype=np.int32)
    out = np.zeros(1, dtype=np.uint32)

    lib.mh_shingle_adler32(
        buf.ctypes.data, off.ctypes.data, tlen.ctypes.data, 0, 2, out.ctypes.data
    )
    assert out.tolist() == [zlib.adler32(b"")]


def test_mh_shingle_adler32_token_count_is_derived_from_the_tables():
    # The kernel takes n_tokens as a scalar. If the wrapper ever passed a count
    # that disagrees with off.size, the kernel would read past the end of the
    # span tables; the wrapper rejects that, and here the honest count is what
    # makes the window list come out right.
    import zlib

    buf = np.frombuffer(b"one two three", np.uint8)
    off = np.array([0, 4, 8], dtype=np.int32)
    tlen = np.array([3, 3, 5], dtype=np.int32)
    out = np.zeros(2, dtype=np.uint32)

    lib.mh_shingle_adler32(
        buf.ctypes.data, off.ctypes.data, tlen.ctypes.data, off.size, 2, out.ctypes.data
    )
    assert out.tolist() == [zlib.adler32(b"one two"), zlib.adler32(b"two three")]


# --------------------------------------------------------------------------
# mh_signature
# --------------------------------------------------------------------------


def _min_py(shingles, coeffs, lo, hi):
    return [
        min((int(a) * int(x) + int(b)) % PRIME for x in shingles)
        for a, b in coeffs[lo:hi]
    ]


def test_mh_signature_exact_over_the_block_and_its_tail():
    # 11 hashes crosses HASH_BLOCKS (4) twice and leaves a tail of 3, so both
    # the blocked loop and the scalar remainder have to be right.
    rng = np.random.default_rng(2026)
    shingles = rng.integers(0, 2**32, size=300, dtype=np.uint32)
    n_hash = 11
    coeffs = np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
        ],
        axis=1,
    )
    out = np.zeros(n_hash, dtype=np.uint64)

    lib.mh_signature(
        shingles.ctypes.data, coeffs.ctypes.data, shingles.size, n_hash, PRIME,
        out.ctypes.data, 0, n_hash, 1,
    )
    assert out.tolist() == _min_py(shingles, coeffs, 0, n_hash)


@pytest.mark.parametrize("n_hash", [1, 3, 4, 5, 8, 9, 16, 17])
def test_mh_signature_hash_counts_around_the_block_width(n_hash):
    rng = np.random.default_rng(n_hash)
    shingles = rng.integers(0, 2**32, size=64, dtype=np.uint32)
    coeffs = np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
        ],
        axis=1,
    )
    out = np.zeros(n_hash, dtype=np.uint64)

    lib.mh_signature(
        shingles.ctypes.data, coeffs.ctypes.data, shingles.size, n_hash, PRIME,
        out.ctypes.data, 0, n_hash, 1,
    )
    assert out.tolist() == _min_py(shingles, coeffs, 0, n_hash)


def test_mh_signature_writes_only_the_requested_column_range():
    # A kernel that ignored h0/h1 and always started at 0 would still produce
    # the right answer for a whole-range call; the sentinel catches it.
    rng = np.random.default_rng(7)
    shingles = rng.integers(0, 2**32, size=50, dtype=np.uint32)
    n_hash = 12
    coeffs = np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.int64),
        ],
        axis=1,
    )
    full = np.zeros(n_hash, dtype=np.uint64)
    lib.mh_signature(
        shingles.ctypes.data, coeffs.ctypes.data, shingles.size, n_hash, PRIME,
        full.ctypes.data, 0, n_hash, 1,
    )

    sentinel = 0xA5A5A5A5A5A5A5A5
    for lo, hi in ((0, 12), (4, 8), (4, 5), (11, 12)):
        part = np.full(n_hash, sentinel, dtype=np.uint64)
        lib.mh_signature(
            shingles.ctypes.data, coeffs.ctypes.data, shingles.size, n_hash, PRIME,
            part.ctypes.data, lo, hi, 1,
        )
        assert np.all(part[:lo] == sentinel)
        assert np.all(part[hi:] == sentinel)
        assert part[lo:hi].tolist() == full[lo:hi].tolist()


# --------------------------------------------------------------------------
# mh_jaccard_rows
# --------------------------------------------------------------------------


def test_mh_jaccard_rows_writes_row_i_at_stride_n_docs():
    # The kernel lays row i out at i * n_docs, not at a per-chunk offset, and
    # writes n_docs - i - 1 entries starting at column i + 1. Getting any of
    # those wrong shifts the triangle.
    sig = np.array(
        [[1, 1, 1, 1], [1, 1, 2, 2], [7, 7, 7, 7], [1, 2, 3, 4]], dtype=np.int64
    )
    n_docs, sig_len = sig.shape
    out = np.zeros(n_docs * n_docs, dtype=np.float64)

    lib.mh_jaccard_rows(
        sig.ctypes.data, n_docs, sig_len, 0, n_docs, out.ctypes.data, 1
    )

    rows = [out[i * n_docs : i * n_docs + (n_docs - i - 1)] for i in range(n_docs)]
    for i in range(n_docs):
        want = [
            int(np.count_nonzero(sig[i] == sig[j])) / sig_len
            for j in range(i + 1, n_docs)
        ]
        assert rows[i].tolist() == want
    # Nothing at or below the diagonal was written.
    for i in range(n_docs):
        row = out[i * n_docs : (i + 1) * n_docs]
        assert row[n_docs - i - 1 :].tolist() == [0.0] * (i + 1)


def test_mh_jaccard_rows_row_range_is_honoured():
    # Chunking splits [0, n_docs); a chunk that used a local row offset would
    # overwrite another chunk's row with its own.
    rng = np.random.default_rng(3)
    sig = rng.integers(0, 6, size=(10, 4), dtype=np.int64)
    n_docs, sig_len = sig.shape
    whole = np.zeros(n_docs * n_docs, dtype=np.float64)
    lib.mh_jaccard_rows(
        sig.ctypes.data, n_docs, sig_len, 0, n_docs, whole.ctypes.data, 1
    )

    sentinel = -12345.0
    for lo, hi in ((0, 10), (0, 1), (3, 7), (9, 10)):
        part = np.full(n_docs * n_docs, sentinel, dtype=np.float64)
        lib.mh_jaccard_rows(
            sig.ctypes.data, n_docs, sig_len, lo, hi, part.ctypes.data, 1
        )
        # Only the strict upper triangle of rows [lo, hi) is live; the
        # diagonal and lower triangle are never written, so they must still
        # hold the sentinel.
        for i in range(lo, hi):
            row = part[i * n_docs : (i + 1) * n_docs]
            n_cells = n_docs - i - 1
            assert np.array_equal(
                row[:n_cells], whole[i * n_docs : i * n_docs + n_cells]
            )
            assert np.all(row[n_cells:] == sentinel)
        # Rows outside the requested range are untouched.
        for i in range(0, lo):
            assert np.all(part[i * n_docs : (i + 1) * n_docs] == sentinel)
        for i in range(hi, n_docs):
            assert np.all(part[i * n_docs : (i + 1) * n_docs] == sentinel)


# --------------------------------------------------------------------------
# mh_score_hist
# --------------------------------------------------------------------------


def test_mh_score_hist_bins_by_truncated_tenths():
    # The expected histogram is computed, not hand-listed: a hand-written
    # expected vector is itself a place to get a bucket wrong, which is the
    # same class of bug the test exists to catch.
    scores = np.array([0.0, 0.05, 0.15, 0.29, 0.3, 0.7, 0.71, 1.0], dtype=np.float64)
    buckets = (scores * 10).astype(int)
    assert buckets.tolist() == [0, 0, 1, 2, 3, 7, 7, 10]

    n_bins = 11
    want = np.zeros(n_bins, dtype=np.int64)
    np.add.at(want, buckets, 1)

    out = np.full(n_bins, -1, dtype=np.int64)
    lib.mh_score_hist(
        scores.ctypes.data, scores.size, 10.0, n_bins, out.ctypes.data
    )
    assert out.tolist() == want.tolist()


def test_mh_score_hist_drops_what_np_histogram_drops():
    # NumPy casts NaN and +/-inf to INT_MIN, and a value that truncates below
    # the first edge to a negative bucket; none of them reach a bin. Counting
    # them into bin 0 would report a near-duplicate rate that is entirely
    # fabricated by a 0/0 signature.
    scores = np.array(
        [float("nan"), float("inf"), float("-inf"), -0.5, 0.5], dtype=np.float64
    )
    out = np.zeros(11, dtype=np.int64)
    lib.mh_score_hist(scores.ctypes.data, scores.size, 10.0, 11, out.ctypes.data)
    assert out.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert int(out.sum()) == 1


def test_mh_score_hist_zeroes_every_bin_before_counting():
    # A pre-filled output buffer must come back zeroed, or a previous call's
    # counts survive into this one.
    scores = np.array([0.5], dtype=np.float64)
    out = np.full(11, 999, dtype=np.int64)
    lib.mh_score_hist(scores.ctypes.data, scores.size, 10.0, 11, out.ctypes.data)
    assert out.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_mh_score_hist_zero_scores_is_an_empty_histogram():
    out = np.full(11, 7, dtype=np.int64)
    lib.mh_score_hist(
        np.empty(0, np.float64).ctypes.data, 0, 10.0, 11, out.ctypes.data
    )
    assert out.tolist() == [0] * 11


def test_mh_score_hist_bin_count_is_honoured():
    # n_bins sets the top bin. With 5 bins the top is index 4 and everything
    # at or above 0.4 lands there.
    scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    out = np.zeros(5, dtype=np.int64)
    lib.mh_score_hist(scores.ctypes.data, scores.size, 10.0, 5, out.ctypes.data)
    assert out.tolist() == [0, 1, 0, 0, 2]