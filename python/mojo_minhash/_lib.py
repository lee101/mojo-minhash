"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory: every buffer crosses the C ABI as a 64-bit
address, so each `argtypes` entry must be `c_int64` where a buffer is passed.
`c_int` truncates the address and segfaults on anything above 4 GiB, and an
`argtypes` list that is one entry short of the real signature is silent --
ctypes zero-fills the missing slots. `_bind` therefore checks the declared
arity of every entry point at load time.
"""

import ctypes
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-minhash.so"

PRIME = 4294969733

_I64 = ctypes.c_int64
_F64 = ctypes.c_double

EXPECTED_ARITY = {
    "mh_shingle_adler32": 6,
    "mh_signature": 8,
    "mh_jaccard_rows": 6,
    "mh_score_hist": 5,
}


def _bind(lib, name, argtypes):
    """Attach argtypes/restype, refusing an arity that does not match the export.

    ctypes cannot see the callee's arity, so this is the only place a mismatch
    can be caught; every other symptom would be a silently truncated address.
    """
    want = EXPECTED_ARITY[name]
    if len(argtypes) != want:
        raise RuntimeError(
            f"{name}: argtypes has {len(argtypes)} entries, expected {want}"
        )
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = list(argtypes)


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    _bind(lib, "mh_shingle_adler32", [_I64, _I64, _I64, _I64, _I64, _I64])
    _bind(lib, "mh_signature", [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64])
    _bind(lib, "mh_jaccard_rows", [_I64, _I64, _I64, _I64, _I64, _I64])
    _bind(lib, "mh_score_hist", [_I64, _I64, _F64, _I64, _I64])
    return lib


lib = _load()

_POOL = ThreadPoolExecutor(max_workers=8)


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _chunks(n: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, n))
    step = -(-n // workers)
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def adler_shinges_from_spans(buf, off, tlen, k: int) -> np.ndarray:
    """Adler-32 every k-token window of a pre-laid-out token buffer.

    `buf` is the concatenation of the tokens, `off` gives each token's start
    and `tlen` its byte length; the kernel re-inserts the single joining space
    that `" ".join` would put between them.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    buf = np.ascontiguousarray(buf, dtype=np.uint8)
    off = np.ascontiguousarray(off, dtype=np.int32)
    tlen = np.ascontiguousarray(tlen, dtype=np.int32)
    n_tok = int(off.size)
    out = np.empty(max(1, n_tok - k + 1), dtype=np.uint32)
    lib.mh_shingle_adler32(
        _addr(buf), _addr(off), _addr(tlen), n_tok, k, _addr(out)
    )
    return out


def adler_shinges(words, k: int) -> np.ndarray:
    """Adler-32 of every k-shingle of a token list.

    Returns a uint32 array of length `max(1, len(words) - k + 1)`, matching
    `minhash.generate_shingles`, which hashes `" ".join(words[i : i + k])`
    with zlib's adler-32. A document shorter than `k` tokens still yields one
    shingle covering all of them; an empty list yields adler32(b"") == 1.
    """
    blobs = [w.encode("utf-8") for w in words]
    tlen = np.fromiter((len(b) for b in blobs), dtype=np.int32, count=len(blobs))
    off = np.zeros(len(blobs), dtype=np.int32)
    if len(blobs) > 1:
        np.cumsum(tlen[:-1], out=off[1:])
    buf = np.frombuffer(b"".join(blobs), np.uint8)
    return adler_shinges_from_spans(buf, off, tlen, k)


def signature(shingles, coeffs, prime: int = PRIME, workers: int = 8) -> np.ndarray:
    """Per-hash minimum of the universal hash family, as int64.

    `coeffs` is an (n_hash, 2) int64 array of (a, b) pairs. Raises ValueError
    on an empty shingle list, matching `min()` on an empty sequence.
    """
    sh = np.ascontiguousarray(shingles, dtype=np.uint32)
    co = np.ascontiguousarray(coeffs, dtype=np.int64)
    if sh.ndim != 1 or co.ndim != 2 or co.shape[1] != 2:
        raise ValueError("expected a 1-D shingle array and an (n, 2) coeff array")
    if sh.size == 0:
        raise ValueError("min() arg is an empty sequence")
    if not 0 < prime < 2**33:
        raise ValueError("prime must fit in 33 bits for the limb reduction")
    if int(sh.max()) >= 2**32:
        raise ValueError("shingle hashes must be below 2^32")
    if co.size and (int(co.min()) < 0 or int(co.max()) >= 2**32):
        raise ValueError("coefficients must lie in [0, 2^32)")
    out = np.empty(co.shape[0], dtype=np.uint64)
    bounds = _chunks(co.shape[0], workers)

    def run(b):
        lib.mh_signature(
            _addr(sh), _addr(co), sh.size, co.shape[0], prime, _addr(out), b[0], b[1]
        )

    if len(bounds) > 1:
        list(_POOL.map(run, bounds))
    else:
        run(bounds[0])
    return out.view(np.int64)


def jaccard_scores(sigs, workers: int = 8) -> list[np.ndarray]:
    """Upper-triangular `approx_jaccard_score` rows for a signature matrix.

    Returns `n_docs` float64 rows, where row i covers the documents after i,
    exactly as the reference builds its `scores` list.
    """
    sig = np.ascontiguousarray(sigs, dtype=np.int64)
    if sig.ndim != 2:
        raise ValueError("sigs must be 2-D (n_docs, sig_len)")
    n_docs, sig_len = sig.shape
    out = np.zeros(n_docs * n_docs, dtype=np.float64)
    bounds = _chunks(n_docs, workers)

    def run(b):
        lib.mh_jaccard_rows(_addr(sig), n_docs, sig_len, b[0], b[1], _addr(out))

    if len(bounds) > 1:
        list(_POOL.map(run, bounds))
    else:
        run(bounds[0])
    return [out[i * n_docs : i * n_docs + (n_docs - i - 1)] for i in range(n_docs)]


def score_histogram(scores, scale: float = 10.0, n_bins: int = 11) -> np.ndarray:
    """Counts per truncated-score bucket, as int64.

    `scores` is a flat array or a sequence of per-document score rows, which is
    what `jaccard_scores` returns.
    """
    if isinstance(scores, np.ndarray):
        flat = np.ascontiguousarray(scores, dtype=np.float64).ravel()
    else:
        rows = [np.asarray(r, dtype=np.float64).ravel() for r in scores]
        flat = np.ascontiguousarray(
            np.concatenate(rows) if rows else np.empty(0, dtype=np.float64)
        )
    out = np.zeros(n_bins, dtype=np.int64)
    lib.mh_score_hist(
        _addr(flat), flat.size, ctypes.c_double(scale), n_bins, _addr(out)
    )
    return out
