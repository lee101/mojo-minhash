"""Kernel-only (Mojo CLI) probe: 1/2/4/8 workers, no ctypes, no host split.

Runs the column-chunked signature kernel on fixed stack buffers and reports
the minimum wall time over 20 calls per configuration, so sub-millisecond
shapes are timed honestly.
"""

from max.algorithm import parallelize
from std.time import perf_counter_ns
from std.memory import alloc

comptime PREC16 = UInt64(0xFFFF)
comptime S = 4096
comptime H = 512


def _limbs(v: UInt64, p16: UInt64, p32: UInt64, p48: UInt64) -> UInt64:
    return (
        ((v >> 48) & PREC16) * p48
        + ((v >> 32) & PREC16) * p32
        + ((v >> 16) & PREC16) * p16
        + (v & PREC16)
    )


def _sig_block(x: Pointer[UInt32, AnyOrigin[mut=True]],
               co: Pointer[Int64, AnyOrigin[mut=True]],
               res: Pointer[UInt64, AnyOrigin[mut=True]],
               h0: Int, h1: Int, n: Int, c: UInt64,
               p16: UInt64, p32: UInt64, p48: UInt64):
    for h in range(h0, h1):
        var a = UInt64(co[unsafe_offset=2 * h])
        var b = UInt64(co[unsafe_offset=2 * h + 1])
        var best = UInt64(0xFFFFFFFFFFFFFFFF)
        for s in range(n):
            best = min(best, _limbs(a * UInt64(x[unsafe_offset=s]) + b, p16, p32, p48) % c)
        res[unsafe_offset=h] = best


def run(sh: Pointer[UInt32, AnyOrigin[mut=True]],
        co: Pointer[Int64, AnyOrigin[mut=True]],
        res: Pointer[UInt64, AnyOrigin[mut=True]],
        per: Int, workers: Int, iters: Int) -> Float64:
    var c = UInt64(4294969733)
    var p16 = UInt64(65536) % c
    var p32 = (p16 * p16) % c
    var p48 = (p32 * p16) % c
    var n_chunks = (H + per - 1) // per
    var best = Int(0x7FFFFFFFFFFFFFFF)

    def chunk(k: Int) {imm}:
        var lo = k * per
        var hi = min(lo + per, H)
        _sig_block(sh, co, res, lo, hi, S, c, p16, p32, p48)

    for it in range(iters):
        var t0 = perf_counter_ns()
        if workers <= 1:
            for k in range(n_chunks):
                chunk(k)
        else:
            parallelize(chunk, n_chunks, workers)
        var dt = perf_counter_ns() - t0
        if dt < best:
            best = dt
    return Float64(best) / 1e6


def main():
    var sh_mem = alloc[UInt32](S)
    var sh = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=Int(sh_mem))
    var co_mem = alloc[Int64](2 * H)
    var co = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=Int(co_mem))
    var out_mem = alloc[UInt64](H)
    var out = Pointer[UInt64, AnyOrigin[mut=True]](unsafe_from_address=Int(out_mem))
    var seed = UInt64(12345)
    for i in range(S):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        sh[i] = UInt32(seed >> 32)
    for i in range(2 * H):
        seed = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        co[i] = Int64((seed >> 32) % UInt64(4294967291))

    print("per   w1      w2      w4      w8")
    comptime for per in [512, 256, 128, 64, 32, 16]:
        var t1 = run(sh, co, out, per, 1, 20)
        var t2 = run(sh, co, out, per, 2, 20)
        var t4 = run(sh, co, out, per, 4, 20)
        var t8 = run(sh, co, out, per, 8, 20)
        print(per, t1, t2, t4, t8)
