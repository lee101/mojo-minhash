"""MinHash primitives for the `minhash` package (k-shingle hashing, universal
hash signatures, pairwise signature agreement, score histogramming).

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The signature kernel is exact against CPython's arbitrary-precision integers:
`a`, `b` and the shingle hashes are all < 2^32, so `a * x + b` is strictly
below 2^64 and never wraps. The reduction mod `c` is done on 16-bit limbs so
that no intermediate exceeds 2^51, which keeps the whole thing in 64-bit
arithmetic and matches CPython exactly.
"""

from std.math import floor

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]

comptime ADLER_MOD = UInt64(65521)


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def u32p(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def f64p(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


@export("mh_shingle_adler32")
def mh_shingle_adler32(buf_addr: Int, off_addr: Int, len_addr: Int,
                        n_tokens: Int, k: Int, out_addr: Int) abi("C"):
    """Adler-32 of every contiguous window of `k` tokens in a joined buffer.

    `buf` holds the document's tokens concatenated; `off` gives each token's
    start and `len` its byte length. The kernel re-inserts the single space
    that `" ".join(words[i : i + k])` puts between tokens, so the window is
    hashed exactly as the reference hashes it.

    The number of windows is `max(1, n_tokens - k + 1)`: a document shorter
    than `k` tokens still yields one window holding every token, and an empty
    document yields the hash of the empty string, adler32(b"") == 1.
    """
    var buf = u8p(buf_addr)
    var off = i32p(off_addr)
    var tlen = i32p(len_addr)
    var res = u32p(out_addr)

    var width = k
    if width > n_tokens:
        width = n_tokens
    var n_sh = n_tokens - k + 1
    if n_sh < 1:
        n_sh = 1

    for i in range(n_sh):
        var a = UInt64(1)
        var b = UInt64(0)
        for j in range(width):
            var lo = Int(off[unsafe_offset=i + j])
            var hi = lo + Int(tlen[unsafe_offset=i + j])
            for t in range(hi - lo):
                a += UInt64(buf[unsafe_offset=lo + t])
                b += a
            if j + 1 < width:
                a += UInt64(32)  # the " " that " ".join inserts
                b += a
            a = a % ADLER_MOD
            b = b % ADLER_MOD
        res[unsafe_offset=i] = UInt32((b << 16) | a)


@export("mh_signature")
def mh_signature(shingles_addr: Int, coeffs_addr: Int, n_shingles: Int,
                 n_hash: Int, prime: Int, out_addr: Int, h0: Int,
                 h1: Int) abi("C"):
    """Per-hash minimum: out[h] = min_s (a[h] * x[s] + b[h]) % prime, h in [h0, h1).

    `coeffs` is interleaved [a0, b0, a1, b1, ...] as Int64. Every operand is
    below 2^32, so `a * x + b` is below 2^64 and the 16-bit limb reduction
    reproduces CPython's arbitrary-precision result exactly.
    """
    var x = u32p(shingles_addr)
    var co = i64p(coeffs_addr)
    var res = u64p(out_addr)

    var c = UInt64(prime)
    var p16 = UInt64(65536) % c
    var p32 = (p16 * p16) % c
    var p48 = (p32 * p16) % c

    for h in range(h0, h1):
        var a = UInt64(co[unsafe_offset=2 * h])
        var b = UInt64(co[unsafe_offset=2 * h + 1])
        var best = UInt64(0xFFFFFFFFFFFFFFFF)
        for s in range(n_shingles):
            var v = a * UInt64(x[unsafe_offset=s]) + b
            var l0 = v & 0xFFFF
            var l1 = (v >> 16) & 0xFFFF
            var l2 = (v >> 32) & 0xFFFF
            var l3 = (v >> 48) & 0xFFFF
            var acc = l3 * p48 + l2 * p32 + l1 * p16 + l0
            var hv = acc % c
            if hv < best:
                best = hv
        res[unsafe_offset=h] = best


@export("mh_jaccard_rows")
def mh_jaccard_rows(sig_addr: Int, n_docs: Int, sig_len: Int, i0: Int, i1: Int,
                    out_addr: Int) abi("C"):
    """Upper-triangular signature agreement for rows [i0, i1).

    Row i holds `count(sig[i] == sig[j]) / sig_len` for j in (i, n_docs), laid
    out with a fixed stride of `n_docs` so the rows stay at constant offsets.
    The count is an exact integer comparison; only the final division is float.
    """
    var sig = i64p(sig_addr)
    var res = f64p(out_addr)
    var denom = Float64(sig_len)

    for i in range(i0, i1):
        var row = i * n_docs
        var base_i = i * sig_len
        for jj in range(n_docs - i - 1):
            var base_j = (i + 1 + jj) * sig_len
            var hits = 0
            for k in range(sig_len):
                if sig[unsafe_offset=base_i + k] == sig[unsafe_offset=base_j + k]:
                    hits += 1
            res[unsafe_offset=row + jj] = Float64(hits) / denom


@export("mh_score_hist")
def mh_score_hist(scores_addr: Int, n: Int, scale: Float64, n_bins: Int,
                  out_addr: Int) abi("C"):
    """Histogram the truncated scores: bucket = clamp(trunc(score * scale)).

    This reproduces `(row * 10).astype(int) / 10` fed to
    `np.histogram(row, (0, .1, ..., 1, 42))`: truncating to one decimal makes
    every value exactly k/scale, so the bin is `k` with the top bin absorbing
    the score of exactly 1.0 (np.histogram's last bin is closed on the right).

    The truncation is explicit because Mojo's `Int64(Float64)` cast is
    round-half-to-even, while `ndarray.astype(int)` truncates toward zero; a
    score of exactly 0.15 lands in bin 1 under truncation and bin 2 under
    round-half-to-even.
    """
    var sc = f64p(scores_addr)
    var out = i64p(out_addr)
    var top = Int64(n_bins - 1)

    for b in range(n_bins):
        out[unsafe_offset=b] = 0
    for t in range(n):
        var v = sc[unsafe_offset=t] * scale
        var k = Int64(floor(v))
        if v < 0.0 and Float64(k) != v:
            k += 1
        if k < 0:
            k = 0
        if k > top:
            k = top
        out[unsafe_offset=k] += 1
