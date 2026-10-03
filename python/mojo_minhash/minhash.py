"""Mojo-backed drop-in for the compute surface of the `minhash` package 0.1.1.

Upstream `minhash` is a 112-line script whose numeric core is four loops:
adler-32 over k-shingles, a minimum-reduce of the universal hash family
`(a * x + b) % prime`, an upper-triangular signature-agreement scan, and a
truncated-score histogram. Those four are compiled here; the string joining,
the random coefficient draw and the JSON/CSV plumbing stay in Python because
they are not array work.

Deliberate API deviation: `generate_hash_funcs` returns `(a, b)` coefficient
pairs rather than one Python lambda per hash. The lambdas exist only so the
reference can call them once per shingle from a Python loop, which is the work
this package removes.
"""

from __future__ import annotations

import random

import numpy as np

from mojo_minhash import _lib

__all__ = [
    "PRIME",
    "generate_shingles",
    "generate_hash_funcs",
    "calculate_signature",
    "approx_jaccard_score",
    "score_matrix",
    "score_histogram",
]

PRIME = _lib.PRIME


def generate_shingles(words, count: int = 2) -> np.ndarray:
    """adler-32 of every `count`-shingle of `words`, as uint32.

    Equivalent to `minhash.generate_shingles(words, count)`, which yields
    `" ".join(words[i : i + count])` hashed with zlib's adler-32. A document
    with fewer than `count` words still yields one shingle covering all of them,
    and an empty word list yields the single hash of the empty string.
    """
    return _lib.adler_shinges(words, count)


def generate_hash_funcs(count: int, prime: int = PRIME, seed=None):
    """Draw `count` `(a, b)` coefficient pairs for the hash family.

    Mirrors the reference's draw: `random.sample(range(2**32 - 1), 2 * count)`
    followed by popping pairs off the end of the list, so pair *i* is
    `(coeffs[2n - 1 - 2i], coeffs[2n - 2 - 2i])`. Pass `seed` for a
    reproducible draw; the reference uses the global `random` module, and it
    also writes the draw to `coeffs.json`, which this does not do.

    `prime` is accepted for signature compatibility with the reference and
    validated, but it does not reach the returned pairs: the coefficients are a
    plain `random.sample` in both implementations and are independent of the
    modulus. `calculate_signature` always reduces mod `mojo_minhash.PRIME`.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if not 0 < prime < 2**33:
        raise ValueError("prime must fit in 33 bits for the limb reduction")
    rng = random.Random(seed) if seed is not None else random
    coeffs = rng.sample(range(2**32 - 1), count * 2)
    return [(coeffs.pop(), coeffs.pop()) for _ in range(count)]


def calculate_signature(shingles, hash_funcs, workers: int = 8) -> np.ndarray:
    """`[min((a * x + b) % PRIME) for (a, b) in hash_funcs]` as int64.

    Equivalent to `minhash.calculate_signature`. Raises ValueError for an empty
    shingle list, as `min()` does.

    The modulus is always `PRIME`, matching the only value the reference's
    `__main__` uses; it is not read back off `hash_funcs`, because this API
    returns bare `(a, b)` pairs rather than the reference's closures that
    captured their own `c`.
    """
    pairs = np.asarray(hash_funcs, dtype=np.int64)
    return _lib.signature(np.asarray(shingles), pairs, PRIME, workers)


def approx_jaccard_score(a, b, axis: int = 0) -> float:
    """Fraction of signature positions where `a` and `b` agree.

    `axis=0` compares a single signature against another. This one delegates to
    NumPy rather than to `mh_jaccard_rows`: it is an O(sig_len) comparison, not
    the O(n^2 * sig_len) scan the kernel is for, and at that size a ctypes round
    trip costs more than the work. The block form, which is what the reference's
    `__main__` actually builds, is `score_matrix`.
    """
    aa = np.asarray(a, dtype=np.int64)
    if axis != 0:
        raise NotImplementedError(
            "use score_matrix() for the block comparison; it is the compiled path"
        )
    bb = np.asarray(b, dtype=np.int64)
    if aa.ndim != 1 or bb.ndim != 1 or aa.shape != bb.shape:
        raise ValueError("a and b must be 1-D signatures of equal length")
    if aa.size == 0:
        raise ValueError("a signature must have at least one position")
    return np.count_nonzero(aa == bb, axis) / len(aa)


def score_matrix(sigs, workers: int = 8) -> list[np.ndarray]:
    """Upper-triangular agreement rows for an (n_docs, sig_len) signature matrix.

    Row i holds the score of document i against every document after it, which
    is the `scores` list the reference assembles in its `__main__`. Entry
    `(i, j)` of the conceptual matrix is `row_i[j - i - 1]`.
    """
    return _lib.jaccard_scores(np.asarray(sigs), workers)


def score_histogram(rows, scale: float = 10.0) -> np.ndarray:
    """Total per-bucket counts over the score rows, bucketed to 1/10.

    Equivalent to the reference's binning loop: each score is truncated to one
    decimal and counted into one of eleven bins, the last of which absorbs a
    score of exactly 1.0.
    """
    return _lib.score_histogram(rows, scale)
