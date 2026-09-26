"""Mojo port of the numeric core of the `minhash` package (v0.1.1)."""

from mojo_minhash.minhash import (
    PRIME,
    approx_jaccard_score,
    calculate_signature,
    generate_hash_funcs,
    generate_shingles,
    score_histogram,
    score_matrix,
)
from mojo_minhash._lib import lib

__all__ = [
    "PRIME",
    "approx_jaccard_score",
    "calculate_signature",
    "generate_hash_funcs",
    "generate_shingles",
    "score_histogram",
    "score_matrix",
    "lib",
]
__version__ = "0.1.0"
