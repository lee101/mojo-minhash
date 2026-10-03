"""MinHash primitives for the `minhash` package: k-shingle hashing (adler-32),
universal-hash signatures, pairwise signature agreement, and score binning.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The signature kernel is exact against CPython's arbitrary-precision integers:
`a`, `b` and the shingle hashes are all < 2^32, so `a * x + b` is strictly
below 2^64 and never wraps. The reduction mod `c` is done on 16-bit limbs, so
no intermediate exceeds 2^51.

Two notes for anyone editing this file, both found by measurement on
mojo 1.2.0.dev2026092905:

* `HASH_BLOCKS` hash columns run per shingle so the 64-bit `div` has several
  independent chains to interleave with. It is the only speedup measured here.
* `Int64 >> 32` on a negative value yields 0 here rather than -1, so the
  division-free `2**32 + delta` fold is not available; the limb reduction is.
  Likewise the unsigned comparison `r >= cs` is not a signed test.

`Int64(Float64)` on this toolchain is round-half-to-even, so the histogram's
truncation is written out; see `_bucket`.
"""

from std.math import floor

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]

comptime ADLER_MOD = UInt64(65521)
comptime PREC16 = UInt64(0xFFFF)

# Hash columns the signature kernel keeps in flight. Each column owns its own
# minimum, so several independent divide chains hide each other's latency.
comptime HASH_BLOCKS = 4


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


def _limbs(v: UInt64, p16: UInt64, p32: UInt64, p48: UInt64) -> UInt64:
    """`v mod prime` for a prime near 2**32, without any intermediate over 2**51.

    Writing v as four 16-bit limbs and weighting them by the matching powers of
    2**16, taken modulo the prime, keeps every product below 2**51 and every
    partial sum below 2**52, so nothing wraps. CPython's arbitrary-precision
    result is reproduced exactly.
    """
    return (
        ((v >> 48) & PREC16) * p48
        + ((v >> 32) & PREC16) * p32
        + ((v >> 16) & PREC16) * p16
        + (v & PREC16)
    )


def _signature_one(x: U32Ptr, a: UInt64, b: UInt64, c: UInt64, p16: UInt64,
                   p32: UInt64, p48: UInt64, n: Int) -> UInt64:
    var best = UInt64(0xFFFFFFFFFFFFFFFF)
    for s in range(n):
        best = min(best, _limbs(a * UInt64(x[unsafe_offset=s]) + b, p16, p32, p48) % c)
    return best


def _signature_block(x: U32Ptr, co: I64Ptr, res: U64Ptr, h0: Int, h1: Int,
                     n: Int, c: UInt64, p16: UInt64, p32: UInt64, p48: UInt64):
    """Hashes [h0, h1) of `mh_signature`, HASH_BLOCKS columns at a time."""
    var h = h0
    while h + HASH_BLOCKS <= h1:
        var a0 = UInt64(co[unsafe_offset=2 * h])
        var b0 = UInt64(co[unsafe_offset=2 * h + 1])
        var a1 = UInt64(co[unsafe_offset=2 * h + 2])
        var b1 = UInt64(co[unsafe_offset=2 * h + 3])
        var a2 = UInt64(co[unsafe_offset=2 * h + 4])
        var b2 = UInt64(co[unsafe_offset=2 * h + 5])
        var a3 = UInt64(co[unsafe_offset=2 * h + 6])
        var b3 = UInt64(co[unsafe_offset=2 * h + 7])
        var best0 = UInt64(0xFFFFFFFFFFFFFFFF)
        var best1 = best0
        var best2 = best0
        var best3 = best0
        for s in range(n):
            var xs = UInt64(x[unsafe_offset=s])
            best0 = min(best0, _limbs(a0 * xs + b0, p16, p32, p48) % c)
            best1 = min(best1, _limbs(a1 * xs + b1, p16, p32, p48) % c)
            best2 = min(best2, _limbs(a2 * xs + b2, p16, p32, p48) % c)
            best3 = min(best3, _limbs(a3 * xs + b3, p16, p32, p48) % c)
        res[unsafe_offset=h] = best0
        res[unsafe_offset=h + 1] = best1
        res[unsafe_offset=h + 2] = best2
        res[unsafe_offset=h + 3] = best3
        h += HASH_BLOCKS
    while h < h1:
        res[unsafe_offset=h] = _signature_one(
            x,
            UInt64(co[unsafe_offset=2 * h]),
            UInt64(co[unsafe_offset=2 * h + 1]),
            c, p16, p32, p48, n
        )
        h += 1


def _agree(sig: I64Ptr, base_i: Int, base_j: Int, sig_len: Int) -> Int64:
    """How many positions two signatures agree on."""
    var hits = Int64(0)
    for k in range(sig_len):
        if sig[unsafe_offset=base_i + k] == sig[unsafe_offset=base_j + k]:
            hits += Int64(1)
    return hits


def _bucket(v: Float64, scale: Float64, top: Int64) -> Int64:
    """`clamp(trunc(v * scale), 0, top)`, or -1 if `np.histogram` drops `v`.

    The reference bins `(row * 10).astype(int) / 10` against
    `(0, .1, ..., 1, 42)`, so the value that is placed in a bin is the *truncated*
    one, not `v`. Three consequences are easy to get wrong:

    * The floor plus the negative-side correction is exactly truncation toward
      zero. `Int64(Float64)` on this toolchain is round-half-to-even, which
      would put a score of exactly 0.15 in bin 2 instead of bin 1.
    * The "below the first edge means drop" test applies to the truncated
      bucket, not to `v * scale`. A score of -0.05 truncates to bucket 0 and the
      reference counts it in bin 0; only a score that truncates below 0 is
      dropped.
    * Non-finite scores are dropped too. NumPy casts NaN and +/-inf to INT_MIN,
      which is to the left of the first bin edge, so `np.histogram` ignores
      them. Folding them into bin 0 would hide the upstream failure that
      produced them -- an empty signature scores 0/0, i.e. NaN.

    A finite score that truncates past the top bin is clamped into it rather
    than dropped, matching the documented "last bin absorbs the overflow"
    behaviour. `np.histogram` would drop a value above its final edge, but that
    needs the edge itself, which this kernel deliberately does not take: a score
    is `count / sig_len` and so cannot exceed 1.
    """
    var p = v * scale
    if p != p:
        return Int64(-1)  # NaN
    if p > 1.0e300 or p < -1.0e300:
        return Int64(-1)  # +/-inf: the reference casts these to INT_MIN
    var k = Int64(floor(p))
    if v < 0.0 and Float64(k) != p:
        k += 1
    if k < 0:
        return Int64(-1)
    if k > top:
        k = top
    return k


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
    reproduces CPython's arbitrary-precision residue exactly.
    """
    var x = u32p(shingles_addr)
    var co = i64p(coeffs_addr)
    var res = u64p(out_addr)

    var c = UInt64(prime)
    var p16 = UInt64(65536) % c
    var p32 = (p16 * p16) % c
    var p48 = (p32 * p16) % c

    if h0 <= 0:
        _signature_block(x, co, res, 0, h1, n_shingles, c, p16, p32, p48)
    else:
        _signature_block(x, co, res, h0, h1, n_shingles, c, p16, p32, p48)


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
        for j in range(i + 1, n_docs):
            res[unsafe_offset=row + j - i - 1] = (
                Float64(_agree(sig, base_i, j * sig_len, sig_len)) / denom
            )


@export("mh_score_hist")
def mh_score_hist(scores_addr: Int, n: Int, scale: Float64, n_bins: Int,
                  out_addr: Int) abi("C"):
    """Histogram the truncated scores: bucket = clamp(trunc(score * scale)).

    This reproduces `(row * 10).astype(int) / 10` fed to
    `np.histogram(row, (0, .1, ..., 1, 42))`: truncating to one decimal makes
    every value exactly k/scale, so the bin is `k` with the top bin absorbing
    the score of exactly 1.0 (np.histogram's last bin is closed on the right).

    Scores that `np.histogram` would drop -- non-finite scores, and scores that
    truncate below the first bin edge -- are skipped rather than folded into
    bin 0; see `_bucket`.
    """
    var sc = f64p(scores_addr)
    var out = i64p(out_addr)
    var top = Int64(n_bins - 1)

    for b in range(n_bins):
        out[unsafe_offset=b] = Int64(0)
    for t in range(n):
        var b = _bucket(sc[unsafe_offset=t], scale, top)
        if b >= 0:
            out[unsafe_offset=b] += Int64(1)