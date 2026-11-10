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
import zlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-minhash.so"

PRIME = 4294969733

_I64 = ctypes.c_int64
_F64 = ctypes.c_double

EXPECTED_ARITY = {
    "mh_shingle_adler32": 6,
    "mh_signature": 9,
    "mh_jaccard_rows": 7,
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
    _bind(lib, "mh_signature", [_I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64, _I64])
    _bind(lib, "mh_jaccard_rows", [_I64, _I64, _I64, _I64, _I64, _I64, _I64])
    _bind(lib, "mh_score_hist", [_I64, _I64, _F64, _I64, _I64])
    return lib


lib = _load()

# Worker count the kernel is asked for when the caller asks for more than one.
# The kernel caps it at its own WORKER_CAP (4) -- this host is shared, and more
# workers measured slower per worker -- and chooses the serial path itself
# below its measured size threshold, so a small call never pays a thread launch.
DEFAULT_WORKERS = 4


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _workers(workers) -> int:
    """Pass a caller's worker request to the kernel, as at least one worker.

    The kernel clamps it to its own WORKER_CAP; a value below 1 here would only
    mean "you choose", which the default already says explicitly.
    """
    return max(1, int(workers))


def _check_spans(buf, off, tlen) -> None:
    """Reject span tables the kernel would read past the end of `buf`.

    The kernel trusts `off` and `tlen` completely and indexes `buf` with them,
    so an out-of-range entry is an out-of-bounds read rather than an error.
    """
    if off.ndim != 1 or tlen.ndim != 1:
        raise ValueError("off and tlen must be 1-D")
    if off.size != tlen.size:
        raise ValueError(
            f"off has {off.size} entries but tlen has {tlen.size}"
        )
    if off.size == 0:
        return
    if int(off.min()) < 0 or int(tlen.min()) < 0:
        raise ValueError("span offsets and lengths must be non-negative")
    if int(off.max()) > 2**31 - 1 or int(tlen.max()) > 2**31 - 1:
        raise ValueError("span offsets and lengths must fit in int32")
    end = int(off.max()) + int(tlen[off.argmax()])
    if end > buf.size:
        raise ValueError(
            f"span at offset {int(off.max())} runs to byte {end}, "
            f"past the end of the {buf.size}-byte buffer"
        )


def adler_shinges_from_spans(buf, off, tlen, k: int) -> np.ndarray:
    """Adler-32 every k-token window of a pre-laid-out token buffer.

    `buf` is the concatenation of the tokens, `off` gives each token's start
    and `tlen` its byte length; the kernel re-inserts the single joining space
    that `" ".join` would put between them.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    buf = np.asarray(buf)
    off = np.asarray(off)
    tlen = np.asarray(tlen)
    _check_spans(buf, off, tlen)
    buf = np.ascontiguousarray(buf, dtype=np.uint8).reshape(-1)
    off = np.ascontiguousarray(off, dtype=np.int32).reshape(-1)
    tlen = np.ascontiguousarray(tlen, dtype=np.int32).reshape(-1)
    n_tok = int(off.size)
    out = np.empty(max(1, n_tok - k + 1), dtype=np.uint32)
    lib.mh_shingle_adler32(
        _addr(buf), _addr(off), _addr(tlen), n_tok, k, _addr(out)
    )
    return out


def adler_shinges(words, k: int) -> np.ndarray:
    """Adler-32 of every k-shingle of a token list.

    Matches `minhash.generate_shingles` (upstream 0.1.1, lines 12-17): that
    function builds `deque(words[:count], maxlen=count)` and hashes
    `" ".join(memory)`, so it raises ValueError for a negative `count`
    (maxlen) and TypeError when a token is not a string; `k == 0` is the port's
    documented deviation (upstream yields adler32(b"") per token).

    Returns a uint32 array of length `max(1, len(words) - k + 1)`. A document
    shorter than `k` tokens still yields one shingle covering all of them; an
    empty list yields adler32(b"") == 1.

    The token table comes from one join and one vectorised scan for the
    separators rather than one `encode()` per token. `" ".join` is exactly the
    string the reference hashes, so a token that itself contains a space is
    indistinguishable from two tokens there too; the separator count is checked
    anyway and the per-token path is kept as the fallback. That removes a
    per-token Python loop, which is invisible in a kernel profile but is the
    dominant cost of `generate_shingles` at document sizes.
    """
    if words is None:
        words = []
    if not isinstance(k, int) or k < 1:
        # minhash.generate_shingles builds `deque(words[:count], maxlen=count)`,
        # whose negative maxlen raises ValueError ("maxlen must be
        # non-negative"); k == 0 is the port's documented deviation.
        raise ValueError("k must be >= 1")
    if isinstance(words, str):
        # The reference slices and `" ".join`s it, so "abc" is three tokens.
        words = list(words)
    if not isinstance(words, (list, tuple)):
        words = list(words)
    if not words:
        return np.array([zlib.adler32(b"")], dtype=np.uint32)
    joined = " ".join(words)
    buf = np.frombuffer(joined.encode("utf-8"), np.uint8)
    sep = np.flatnonzero(buf == 0x20)
    if sep.size != len(words) - 1:
        blobs = [w.encode("utf-8") for w in words]
        tlen = np.fromiter((len(b) for b in blobs), dtype=np.int32, count=len(blobs))
        off = np.zeros(len(blobs), dtype=np.int32)
        if len(blobs) > 1:
            np.cumsum(tlen[:-1], out=off[1:])
        buf = np.frombuffer(b"".join(blobs), np.uint8)
        return adler_shinges_from_spans(buf, off, tlen, k)
    ends = np.empty(len(words), dtype=np.int64)
    ends[:-1] = sep
    ends[-1] = buf.size
    off = np.zeros(len(words), dtype=np.int32)
    if len(words) > 1:
        off[1:] = sep + 1
    return adler_shinges_from_spans(buf, off, (ends - off).astype(np.int32), k)


def signature(shingles, coeffs, prime: int = PRIME, workers: int = DEFAULT_WORKERS) -> np.ndarray:
    """Per-hash minimum of the universal hash family, as int64.

    Matches `minhash.calculate_signature` (upstream line 28,
    `np.array([min(map(hash, shingles)) for hash in hash_funcs])`): an empty
    shingle list raises ValueError from `min()`, and an empty `hash_funcs`
    returns an empty array without reading the shingles.

    `coeffs` is an (n_hash, 2) int64 array of (a, b) pairs. A 2-D shingle
    table is flattened to 1-D, the token sequence upstream's `" ".join` would
    consume; an empty `coeffs` returns early because upstream maps over zero
    hash functions.

    `workers` is a request, not a promise: the kernel splits the hash columns
    across at most `min(workers, 4)` workers only above its measured size
    threshold, and runs serially below it (or at `workers=1`). The result is
    identical either way -- the test suite compares serial and parallel on the
    same input.
    """
    sh = np.asarray(shingles)
    sh = sh.reshape(-1) if sh.ndim > 1 else sh
    co = np.asarray(coeffs)
    if co.size == 0:
        # minhash.calculate_signature maps over zero hash functions and returns
        # an empty array without touching the shingles.
        return np.zeros(0, dtype=np.int64)
    co = np.ascontiguousarray(co, dtype=np.int64)
    if sh.ndim != 1 or co.ndim != 2 or co.shape[1] != 2:
        raise ValueError("expected a 1-D shingle array and an (n, 2) coeff array")
    if sh.size == 0:
        raise ValueError("min() arg is an empty sequence")
    if not 0 < prime < 2**33:
        raise ValueError("prime must fit in 33 bits for the limb reduction")
    # The kernel reads the shingle table as UInt32. Narrowing a wider input
    # would silently fold 2**32 back onto 0 and change the residue, so an
    # out-of-range value is an error rather than a truncation.
    if int(sh.min()) < 0 or int(sh.max()) >= 2**32:
        raise ValueError("shingle hashes must lie in [0, 2**32)")
    sh = np.ascontiguousarray(sh, dtype=np.uint32)
    if co.size and (int(co.min()) < 0 or int(co.max()) >= 2**32):
        raise ValueError("coefficients must lie in [0, 2**32)")
    out = np.empty(co.shape[0], dtype=np.uint64)
    lib.mh_signature(
        _addr(sh), _addr(co), sh.size, co.shape[0], prime, _addr(out),
        0, co.shape[0], _workers(workers),
    )
    return out.view(np.int64)


def jaccard_scores(sigs, workers: int = DEFAULT_WORKERS) -> list[np.ndarray]:
    """Upper-triangular `approx_jaccard_score` rows for a signature matrix.

    Returns `n_docs` float64 rows, where row i covers the documents after i,
    exactly as the reference builds its `scores` list. `workers` is passed to
    the kernel, which splits the rows only above its size threshold; the
    output is identical either way.
    """
    sig = np.ascontiguousarray(sigs, dtype=np.int64)
    if sig.ndim != 2:
        raise ValueError("sigs must be 2-D (n_docs, sig_len)")
    n_docs, sig_len = sig.shape
    if sig_len == 0:
        # The reference computes `count_nonzero(...) / len(a)`, so an empty
        # signature scores NaN. Passing 0 through would divide by zero inside
        # the kernel and write an undefined double into the output.
        raise ValueError("sigs must have at least one signature position")
    out = np.zeros(n_docs * n_docs, dtype=np.float64)
    lib.mh_jaccard_rows(
        _addr(sig), n_docs, sig_len, 0, n_docs, _addr(out), _workers(workers)
    )
    return [out[i * n_docs : i * n_docs + (n_docs - i - 1)] for i in range(n_docs)]


def score_histogram(scores, scale: float = 10.0, n_bins: int = 11) -> np.ndarray:
    """Counts per truncated-score bucket, as int64.

    `scores` is a flat array or a sequence of per-document score rows, which is
    what `jaccard_scores` returns. Scores that the reference's `np.histogram`
    would drop -- non-finite values, and values truncating below the first bin
    edge -- are dropped here too, so the total can be less than `len(scores)`.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
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
