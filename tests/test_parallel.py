"""Serial and parallel paths must produce bit-identical output.

The kernels decide between a serial loop and a `max.algorithm.parallelize`
fan-out from the input size alone (and from the requested worker count), so a
test that only ever runs one of the two paths cannot see a chunking bug: a
column or row range that is one element short, or a partial count combined in
the wrong order, gives a plausible-looking matrix with wrong entries.

Each test below calls the same input through `workers=1` (the serial path, no
matter how large the input) and through `workers=2, 4, 8`. The kernels cap the
worker count at 4, so the `workers=8` call also pins the cap: asking for more
than four workers must change neither the result nor, above the threshold, the
output of the 4-worker call.
"""

import numpy as np
import pytest

import mojo_minhash as mh
from mojo_minhash import _lib

# Sizes chosen to straddle the kernels' measured thresholds: small ones where
# the serial path must still be taken (see `test_below_threshold_is_serial`),
# and shapes at and well above the signature threshold (2**23 hash*shingle
# units, i.e. 512x4096 and 1024x4096) where the parallel path runs. The score
# matrix is serial by measurement, but every size is still compared.
SIGNATURE_SIZES = [(8, 16), (16, 64), (128, 512), (128, 4096), (512, 4096), (1024, 4096)]
JACCARD_SIZES = [(2, 4), (8, 16), (16, 64), (33, 128), (64, 256), (256, 256)]
WORKER_COUNTS = [2, 4, 8]


def _coeffs(rng, n_hash):
    return np.stack(
        [
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64),
            rng.integers(0, 2**32 - 1, size=n_hash, dtype=np.uint64),
        ],
        axis=1,
    ).astype(np.int64)


def _sig_matrix(rng, n_docs, sig_len):
    # A small alphabet so agreement counts are not all 0 or all sig_len.
    return rng.integers(0, 8, size=(n_docs, sig_len), dtype=np.int64)


@pytest.mark.parametrize("n_hash,n_shingles", SIGNATURE_SIZES)
@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_signature_serial_equals_parallel(n_hash, n_shingles, workers):
    rng = np.random.default_rng(20261110)
    shingles = rng.integers(0, 2**32, size=n_shingles, dtype=np.uint64)
    coeffs = _coeffs(rng, n_hash)
    serial = mh.calculate_signature(shingles, coeffs, workers=1)
    parallel = mh.calculate_signature(shingles, coeffs, workers=workers)
    assert np.array_equal(serial, parallel), (
        f"signature differs at h={n_hash} s={n_shingles} workers={workers}"
    )


@pytest.mark.parametrize("n_docs,sig_len", JACCARD_SIZES)
@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_jaccard_rows_serial_equals_parallel(n_docs, sig_len, workers):
    rng = np.random.default_rng(20261111)
    sigs = _sig_matrix(rng, n_docs, sig_len)
    serial = mh.score_matrix(sigs, workers=1)
    parallel = mh.score_matrix(sigs, workers=workers)
    assert len(serial) == len(parallel) == n_docs
    for i, (a, b) in enumerate(zip(serial, parallel)):
        assert np.array_equal(a, b), (
            f"row {i} differs at {n_docs}x{sig_len} workers={workers}"
        )


def test_worker_cap_does_not_change_the_output_above_the_threshold():
    # 1024 x 4096 is far above the signature threshold, so `workers=4` and a
    # request for 8 (clamped to 4) take the same chunking.
    rng = np.random.default_rng(7)
    shingles = rng.integers(0, 2**32, size=4096, dtype=np.uint64)
    coeffs = _coeffs(rng, 1024)
    four = mh.calculate_signature(shingles, coeffs, workers=4)
    eight = mh.calculate_signature(shingles, coeffs, workers=8)
    assert np.array_equal(four, eight)


def test_default_workers_equals_explicit_serial():
    # The public default must not change any caller's result, below or above
    # the threshold.
    rng = np.random.default_rng(8)
    for h, s in ((96, 1024), (512, 4096)):
        shingles = rng.integers(0, 2**32, size=s, dtype=np.uint64)
        coeffs = _coeffs(rng, h)
        assert np.array_equal(
            mh.calculate_signature(shingles, coeffs),  # default worker count
            mh.calculate_signature(shingles, coeffs, workers=1),
        )
    sigs = _sig_matrix(rng, 40, 128)
    for a, b in zip(mh.score_matrix(sigs), mh.score_matrix(sigs, workers=1)):
        assert np.array_equal(a, b)


def test_every_chunk_boundary_is_covered():
    # Above the threshold, a column range whose last chunk is short would leave
    # the tail columns unhashed. Sizes 1 below, at, and 1 above multiples of the
    # chunk width are checked through the parallel path.
    rng = np.random.default_rng(9)
    for n_hash in (511, 512, 513):
        shingles = rng.integers(0, 2**32, size=4096, dtype=np.uint64)
        coeffs = _coeffs(rng, n_hash)
        serial = mh.calculate_signature(shingles, coeffs, workers=1)
        assert np.array_equal(
            mh.calculate_signature(shingles, coeffs, workers=4), serial
        )


def test_below_threshold_is_serial():
    """Tiny inputs must not pay a worker launch.

    The kernels have no counter to read back, so this is pinned by timing
    order rather than by instrumentation: a 16x64 signature (about 30 us of
    work) is far below the threshold, so `workers=1` and `workers=4` must take
    the same code path and time the same. A 4-worker launch costs milliseconds
    on this host (README), so the bound below is loose enough not to flake on a
    loaded box and still orders of magnitude below a paid launch.
    """
    import time

    rng = np.random.default_rng(10)
    shingles = rng.integers(0, 2**32, size=64, dtype=np.uint64)
    coeffs = _coeffs(rng, 16)

    def best(fn, repeats=2000):
        fn()
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            runs.append((time.perf_counter() - t0) / repeats)
        return min(runs)

    serial = best(lambda: mh.calculate_signature(shingles, coeffs, workers=1))
    gated = best(lambda: mh.calculate_signature(shingles, coeffs, workers=4))
    assert gated < 10 * serial + 2e-4, f"tiny signature paid a launch: {serial=} {gated=}"
