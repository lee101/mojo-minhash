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
truncation is written out; see `mh_score_hist`.
"""

from std.collections import List
from std.math import floor

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]

comptime ADLER_MOD = UInt64(65521)
comptime PREC16 = UInt64(0xFFFF)

# A token of at most this many bytes runs the shingles' adler-32 loop under one
# deferred reduction; a longer one reduces every byte. Both are exact for the
# same reason: `a` and `b` are only ever meaningful modulo ADLER_MOD, and a
# reduction taken early yields the same residue as one taken after every byte.
comptime MAX_FAST_TOKEN = 4096

# With MAX_FAST_TOKEN = 4096 and a bound R on both running sums at the start of
# a token, the next token adds at most 4096*R + 255*4096*4097/2 < 4097*R + 2**31
# to `b` (and 255*4096 + 33 to `a`, which is far below R). R = 2**50 therefore
# keeps every intermediate below 2**63 -- 4097*2**50 + 2**31 < 2**63 -- with
# room to spare, so the sums never wrap and the final residue is CPython's.
comptime FAST_REDUCE_ABOVE = UInt64(0x4000000000000)  # 2**50

# Hash columns the signature kernel keeps in flight. Each column owns its own
# minimum, so several independent divide chains hide each other's latency.
comptime HASH_BLOCKS = 4

# Partial agreement counters `_agree` keeps in flight, and the block of
# positions each of them walks per iteration. A position costs one load from
# each signature plus a compare-and-add, so eight independent counters keep the
# compare/add pipeline full instead of waiting on a single dependency chain.
comptime AGREE_ACCS = 8


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
    """Hashes [h0, h1) of `mh_signature`, HASH_BLOCKS columns at a time.

    The coefficients and the running minimum of each column are hoisted out of
    the shingle loop, and each shingle is loaded once and reused by all
    HASH_BLOCKS chains. (A `comptime`-unrolled column loop was tried here and
    measured slower at h=128 and h=256: the emitted body was identical, so it
    only cost the constant array loads.)
    """
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
    """How many positions two signatures agree on.

    `sig_len` positions are walked `AGREE_ACCS` at a time, one partial counter
    per accumulator, and the counters are combined once at the end. The block's
    trip count is a `comptime` constant, which this toolchain unrolls (a
    literal range is left as a loop), and that is also what lets it vectorise
    the compare-and-count. Positions past the last full block are counted into
    one of the same counters; a signature shorter than a block uses the scalar
    loop.

    Every position is a 0/1 add, so the reassociation changes no value: the
    count is exact for any summation order.
    """
    var hits = Int64(0)
    if sig_len >= AGREE_ACCS:
        var accs = List[Int64](capacity=AGREE_ACCS)
        comptime for c in range(AGREE_ACCS):
            accs.append(Int64(0))
        var k = 0
        while k + AGREE_ACCS <= sig_len:
            comptime for c in range(AGREE_ACCS):
                accs[c] += Int64(
                    sig[unsafe_offset=base_i + k + c]
                    == sig[unsafe_offset=base_j + k + c]
                )
            k += AGREE_ACCS
        while k < sig_len:
            accs[0] += Int64(
                sig[unsafe_offset=base_i + k] == sig[unsafe_offset=base_j + k]
            )
            k += 1
        comptime for c in range(AGREE_ACCS):
            hits += accs[c]
        return hits
    for k in range(sig_len):
        hits += Int64(sig[unsafe_offset=base_i + k] == sig[unsafe_offset=base_j + k])
    return hits


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

    Like the rest of the port this hashes the bytes of the joined buffer, so
    the result is the reference's when the joining space is the string's only
    delimiter. A token whose own bytes contain 0x20 -- a multi-byte UTF-8
    sequence such as "café" when the whole window is taken as one encode() --
    is hashed as the separator the caller never wrote; that predates this
    kernel and is not changed here.
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
            var m = Int(tlen[unsafe_offset=i + j])
            if m <= MAX_FAST_TOKEN:
                # Short token: one deferred reduction per token pays for itself
                # after a few bytes. The reduction is skipped only while `a`
                # and `b` stay under FAST_REDUCE_ABOVE, which bounds the next
                # token's growth (`b` gains at most m*a + 255*m*(m+1)/2) below
                # 2**63.
                for t in range(m):
                    a += UInt64(buf[unsafe_offset=lo + t])
                    b += a
                if j + 1 < width:
                    a += UInt64(32)  # the " " that " ".join inserts
                    b += a
                if j + 1 == width or a > FAST_REDUCE_ABOVE or b > FAST_REDUCE_ABOVE:
                    a = a % ADLER_MOD
                    b = b % ADLER_MOD
            else:
                # A token longer than MAX_FAST_TOKEN reduces per byte, so the
                # window's running sum cannot pass 2**63 before the next check.
                for t in range(m):
                    a += UInt64(buf[unsafe_offset=lo + t])
                    b += a
                    a = a % ADLER_MOD
                    b = b % ADLER_MOD
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

    The reference bins `(row * 10).astype(int) / 10` against
    `(0, .1, ..., 1, 42)`, so the value that is placed in a bin is the *truncated*
    one, not `score`, and a score of exactly 1.0 lands in the last bin
    (`np.histogram`'s last bin is closed on the right). Three consequences are
    easy to get wrong:

    * The floor plus the negative-side correction is exactly truncation toward
      zero. `Int64(Float64)` on this toolchain is round-half-to-even, which
      would put a score of exactly 0.15 in bin 2 instead of bin 1.
    * The "below the first edge means drop" test applies to the truncated
      bucket, not to `score * scale`. A score of -0.05 truncates to bucket 0 and
      the reference counts it in bin 0; only a score that truncates below 0 is
      dropped.
    * Non-finite scores are dropped too. NumPy casts NaN and +/-inf to INT_MIN,
      which is to the left of the first bin edge, so `np.histogram` ignores
      them. Folding them into bin 0 would hide the upstream failure that
      produced them -- an empty signature scores 0/0, i.e. NaN. Here they are
      caught by a finiteness test before the cast, because a non-finite
      `Int64(Float64)` is unspecified on this toolchain: measured, it yields a
      large positive value rather than INT_MIN, so it cannot be recognised
      after the cast.

    A finite score that truncates past the top bin is clamped into it rather
    than dropped, matching the documented "last bin absorbs the overflow"
    behaviour. `np.histogram` would drop a value above its final edge, but that
    needs the edge itself, which this kernel deliberately does not take: a score
    is `count / sig_len` and so cannot exceed 1.

    The common value -- a finite score in range -- costs one multiply, one
    floor, one cast and one comparison. The negative and the overflow sides are
    one level down and are reached only by scores that are already out of range.
    """
    var sc = f64p(scores_addr)
    var out = i64p(out_addr)
    var top = Int64(n_bins - 1)

    for b in range(n_bins):
        out[unsafe_offset=b] = Int64(0)
    for t in range(n):
        var p = sc[unsafe_offset=t] * scale
        if p > 1.0e300 or p < -1.0e300:
            continue  # NaN and +/-inf: `np.histogram` casts these out of range
        var b = Int64(floor(p))
        if b == top:
            # exactly `top` is in range (the last bin is closed on the right);
            # this arm keeps the common in-range value at one comparison.
            out[unsafe_offset=b] += Int64(1)
        elif b > top:
            out[unsafe_offset=top] += Int64(1)  # clamp into the top bin
        elif b >= 0:
            out[unsafe_offset=b] += Int64(1)
        elif p < 0.0:
            # A negative floor: truncation toward zero is `floor` itself when
            # the value sits exactly on it, and one above it otherwise. Only a
            # score this far out (-1 <= p < 0) ever needs the fix-up.
            if Float64(b) != p:
                b += Int64(1)
            if b >= 0:
                out[unsafe_offset=b] += Int64(1)
