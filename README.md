# mojo-minhash

A Mojo port of the numeric core of the [`minhash`](https://pypi.org/project/minhash/)
package (version 0.1.1).

## What upstream `minhash` actually is

`minhash` is a 112-line single-file script, not a library with a numeric core
in the usual sense. Reading the installed source (`minhash.py` in the pixi
environment) it is four loops plus a `__main__`:

| Upstream symbol | What it does | Has a loop worth compiling? |
| --- | --- | --- |
| `generate_shingles` | adler-32 over every k-shingle of a token list | yes — byte-level inner loop |
| `generate_hash_funcs` | `random.sample` of `2n` coefficients; returns `lambda x: (a*x+b) % c` | the lambda application is the loop |
| `calculate_signature` | `min` over the shingle hashes, once per hash function | yes — the core kernel |
| `approx_jaccard_score` | `np.count_nonzero(a == b, axis) / len(a)` | yes — the O(n^2) scan |
| `__main__` | reads `docs.json`, writes `signatures.json` / `coeffs.json` / `discovered_dups` | no — IO |

So the numeric core is: **adler-32 over bytes, a minimum-reduce of a
universal hash family in 64-bit modular arithmetic, an upper-triangular
signature-agreement scan, and a truncated-score histogram.** All four are
ported. The random draw, the string joining and the JSON IO are not array work
and stay in Python.

## Coverage

| Upstream | Ported | Mojo kernel | Tests |
| --- | --- | --- | --- |
| `generate_shingles` | yes | `mh_shingle_adler32` | `test_shingles.py`, `test_shingle_spans.py`, `test_kernels.py` |
| `generate_hash_funcs` | partly | — | `test_signature.py::test_coefficient_draw_matches_the_reference_draw` |
| `calculate_signature` | yes | `mh_signature` | `test_signature.py`, `test_signature_blocks.py`, `test_kernels.py` |
| `approx_jaccard_score(axis=0)` | yes | — (NumPy) | `test_scores.py` |
| `approx_jaccard_score(axis=1)` | yes | `mh_jaccard_rows` | `test_scores.py`, `test_kernels.py` |
| binning loop in `__main__` | yes | `mh_score_hist` | `test_scores.py`, `test_kernels.py` |
| `__main__` file IO | no | — | not ported |
| `generate_shingles(mapper=...)` | no | — | not ported |
| `gensim.simple_preprocess` | no | — | not ported |

Every kernel is also called **by name**, with buffers laid out exactly as the
kernel documents them, in `tests/test_kernels.py`. That file is deliberately
redundant with the parity tests: a kernel reached only through the public API
can be mis-wired and still ship green, because a wrong `argtypes` and a wrong
kernel produce the same "numbers disagree" failure. To re-check coverage:

```bash
for s in $(grep -hoE '@export\("[A-Za-z_0-9]+"\)' src/*.mojo | sed 's/@export("//;s/")//'); do
  grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"
done
```

### Deliberate API deviations

* `generate_shingles` returns an array, not a generator. The reference is
  consumed with `list(...)` everywhere in its own `__main__`.
* `generate_hash_funcs` returns `(a, b)` pairs rather than one Python lambda
  per hash. The lambdas exist only so the reference can call them once per
  shingle from a Python loop, which is the work this port removes. The
  coefficient *draw* is identical: same `random.sample(range(2**32 - 1),
  2 * count)`, same pop-from-the-end pairing, so with the same seed the pairs
  match the reference exactly (there is a test for this).
* `generate_hash_funcs` does not write `coeffs.json`. Pass `seed=` for a
  reproducible draw; the reference uses the global `random` module.
* The unused `max=2**32-1` parameter of `generate_hash_funcs` is dropped.
* `approx_jaccard_score` accepts only `axis=0`; the block form is
  `score_matrix`.
* `generate_hash_funcs(prime=...)` is accepted and validated but does not reach
  the returned pairs: the draw is a plain `random.sample` in both
  implementations and is independent of the modulus. `calculate_signature`
  always reduces mod `mojo_minhash.PRIME`. Passing a `prime` here will not
  silently change the hash family.

### Not implemented, and why

* The whole `__main__` CLI. It is file IO and `print`; there is nothing in it
  to compile.
* A custom `mapper` for `generate_shingles`. Only adler-32 is compiled.
* Any other MinHash implementation. This port targets `minhash` 0.1.1 as
  installed. (`datasketch` and `nilsimsa` are separately installed in the same
  environment; neither is this package.)

## Exactness

The reference computes `(a * x + b) % prime` in CPython's arbitrary-precision
integers, where `a, b, x < 2**32` and `prime = 4294969733`. Two consequences:

* `a * x + b` is strictly below `2**64`, so the kernel computes it in `UInt64`
  with no wraparound. An `int64` intermediate *would* wrap and there is a test
  (`test_product_above_2_pow_63_does_not_wrap`) that pins this.
* The reduction mod `c` is done on 16-bit limbs, so no intermediate exceeds
  `2**51` and the result matches CPython's exactly.

All signature, shingle and score assertions therefore use exact equality
(`rtol=0, atol=0`), which is legitimate here because the operations are exact
integer arithmetic. The score matrix is the one float case: both sides divide an
exact integer count by the same integer, so the IEEE double result is
bit-identical and the tests assert that.

The histogram is the one place where the reference's own semantics are subtle,
and the kernel follows them exactly rather than approximately. The reference
bins `(row * 10).astype(int) / 10` against `(0, .1, ..., 1, 42)`:

* Truncation is toward zero, not round-half-to-even. A score of exactly
  `3/20 == 0.15` lands in bin 1 under truncation and bin 2 under
  round-half-to-even. The kernel uses an explicit `floor` plus a negative-side
  correction; `test_truncation_is_toward_zero_not_round_half_to_even` fails if
  the cast is used directly.
* Scores that NumPy would *drop* are dropped here too. NumPy casts NaN and
  `+/-inf` to `INT_MIN`, and a score that truncates below the first bin edge
  to a negative bucket; none of them reach a bin. Folding them into bin 0
  would report a near-duplicate rate fabricated entirely by an upstream 0/0 —
  which is exactly what an empty signature matrix produces, since the reference
  computes `count_nonzero(...) / len(a)`. `score_histogram` can therefore sum
  to less than `len(scores)`, and `test_mh_score_hist_drops_what_np_histogram_drops`
  pins that.

## Install and run

Requires Linux x86-64 and [`pixi`](https://pixi.sh). The Mojo toolchain is
pinned in `pixi.toml`; nothing else needs installing.

```bash
pixi install
pixi run build      # src/kernels.mojo -> dist/libmojo-minhash.so
pixi run test
pixi run bench
```

`pixi install` also fetches the real upstream `minhash` from PyPI, because the
tests assert numerical parity against it rather than against a restatement of
its formulas.

To use the package without pixi, put `python/` on `PYTHONPATH` and build the
shared library first. A bare `mojo` on `PATH` also needs `MODULAR_HOME` pointed
at the environment's `share/max`, or every import fails with
`unable to locate module 'std'`:

```bash
export MODULAR_HOME="$(dirname "$(dirname "$(which mojo)")")/share/max"
bash build/build.sh          # mojo build --emit shared-lib
export PYTHONPATH=python
python -c "import mojo_minhash as mh; print(mh.generate_shingles('a b c'.split(), 2))"
```

### Usage

```python
import mojo_minhash as mh

shingles = mh.generate_shingles("the quick brown fox".split(), 2)  # uint32[3]
pairs = mh.generate_hash_funcs(64, seed=0)                        # 64 (a, b) pairs
sig = mh.calculate_signature(shingles, pairs)                     # int64[64]

rows = mh.score_matrix([sig, sig], workers=8)      # row i scores doc i vs doc j > i
```

A complete, runnable example:

```python
import mojo_minhash as mh

docs = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
    "a completely different document about nothing in particular",
]

pairs = mh.generate_hash_funcs(64, seed=0)
sigs = [mh.calculate_signature(mh.generate_shingles(d.split(), 2), pairs) for d in docs]
rows = mh.score_matrix(sigs, workers=8)          # rows[i][j-i-1] scores doc i vs doc j
hist = mh.score_histogram(rows)                  # 11 bins of truncated tenths

print(rows[0][0])                               # 1.0  (docs 0 and 1 are identical)
print(rows[0][1])                               # 0.0  (docs 0 and 2 share no signature positions)
print(hist)                                     # counts per bin
```

## Benchmarks

Correctness is checked against the reference before every case is timed. Re-run
`pixi run bench` to get fresh numbers.

* Toolchain: Mojo 1.2.0.dev2026092905, NumPy 2.5.1
* Host: Intel Xeon E5-2697 v4 @ 2.30GHz, 72 cores
* This host is shared with other benchmark ports. Every figure below is one
  `pixi run bench` run and carries the load average it ran under; repeat runs
  of the single-threaded rows agree within about 10%, the thread-scaling rows
  move much more between runs.
* **Results can differ in the last bits.** The reduction kernels reassociate:
  `_agree` sums eight partial counters and the adler-32 kernel defers its
  modulo, so a float or integer summary is exact but its accumulation order is
  not the scalar one's. All parity tests still compare exactly (`atol=rtol=0`)
  and pass.

### Before and after the unroll phase

`before` is the tree at the start of the unroll phase (bench and raw-kernel
harness both run at load 14 in that session); `after` is the same two runs at
the end of it (both at load 13). `bench` rows are the correctness-gated
`pixi run bench` cases. `kernel` rows are `bench/kernels_micro.py`, which times
the exported kernels through the same ctypes bridge with the thread fan-out off,
so a kernel change shows up without the wrapper.

| case | before | after | change |
| --- | ---: | ---: | ---: |
| bench: shingles, n=20000 k=5 | 1.58 ms | 1.43 ms | 1.10x |
| bench: shingles end to end | 2.24 ms | 1.97 ms | 1.14x |
| bench: signature, h=128 s=4096 | 5.75 ms | 5.98 ms | unchanged |
| bench: 64 docs x 64 hashes | 26.48 ms | 26.10 ms | unchanged |
| bench: score matrix 400x64 | 4.76 ms | 4.15 ms | 1.15x |
| bench: histogram, n=7998000 | 36.67 ms | 20.04 ms | **1.83x** |
| bench: histogram from rows | 92.20 ms | 76.51 ms | 1.21x |
| bench: jaccard_rows 700x64 | 14.29 ms | 15.26 ms | unchanged |
| kernel: signature h=128 s=4096 | 5.91 ms | 5.68 ms | ~4% |
| kernel: signature h=256 s=4096 | 11.79 ms | 11.39 ms | ~3% |
| kernel: jaccard 400x64 | 4.79 ms | 4.76 ms | unchanged |
| kernel: jaccard 700x64 | 14.57 ms | 12.57 ms | 1.16x |
| kernel: jaccard 900x128 | 43.72 ms | 39.63 ms | 1.10x |
| kernel: histogram n=7998000 | 47.14 ms | 27.13 ms | **1.74x** |
| kernel: histogram n=126000 | 0.71 ms | 0.42 ms | 1.71x |
| kernel: shingles n=20000 | 1.63 ms | 1.33 ms | 1.23x |
| kernel: shingles n=200 | 0.027 ms | 0.030 ms | unchanged |

"unchanged" means the before/after difference is inside this machine's run-to-run
noise for that case; no number was chosen from a favourable repeat.

### Current `pixi run bench` (after, load 13)

| case | reference | mojo-minhash | ratio |
| --- | ---: | ---: | ---: |
| shingles, n=20000 k=5, pre-laid-out spans | 3.42 ms | 1.43 ms | **2.39x** |
| shingles end to end from a token list | 3.30 ms | 1.97 ms | **1.68x** |
| signature, h=128 s=4096, vs vectorised NumPy uint64 | 9.04 ms | 5.98 ms | 1.51x |
| signature, h=256 s=4096, 8 threads vs 1 | 13.98 ms | 2.76 ms | 5.06x |
| 64 docs x 64 hashes, vs `minhash.calculate_signature` | 658.50 ms | 26.10 ms | **25.23x** |
| score matrix 400x64, vs `approx_jaccard_score` | 16.32 ms | 4.15 ms | **3.94x** |
| `mh_jaccard_rows` 700x64, vs vectorised NumPy triangle | 69.40 ms | 15.26 ms | **4.55x** |
| score matrix 900x128, 8 threads vs 1 | 47.96 ms | 23.67 ms | 2.03x |
| histogram, n=7998000, vs `np.histogram` | 32.66 ms | 20.04 ms | **1.63x** |
| histogram from the row list `score_matrix` returns | 143.73 ms | 76.51 ms | 1.88x |

Reading the numbers honestly:

* **The flat-array histogram used to be the one loss and is now a 1.63x win.**
  `np.histogram` over 8M values is a tight two-pass NumPy primitive; this kernel
  was a single pass whose per-score branch chain was the cost. Most of the gain
  came from computing the bucket in one branch-free path and checking range once
  (five comparisons became one, on the common value), the rest from deferring
  the adler-32 modulo in the shingle kernel and from `_agree`'s block. A pooled
  multi-counter variant was tried for the increment itself and reverted: it
  measured **slower** (27.1 → 42.4 ms), because a second indexed update per score
  costs more than the single cache line it was meant to relieve.
* **The headline 25x is against CPython, not against NumPy.** The reference's
  `calculate_signature` applies a Python lambda once per shingle per hash, so
  that row measures the interpreter, not the arithmetic. The honest comparison
  for the kernel itself is the `signature, h=128 s=4096` row at 1.51x against
  the fastest vectorised uint64 NumPy formulation the same arithmetic allows.
* **The signature kernel did not unroll better.** A `comptime for` over the hash
  columns measured slower than the four hand-written chains with an identical
  emitted body, so it was reverted; the column chains and hoisted coefficients
  were already what it needed. This toolchain leaves a `for i in range(4)` as a
  runtime loop, so a fixed trip count alone buys no unrolling.
* **Threading scales worse than the core count.** 5.06x on 8 workers for a
  kernel that is genuinely compute-bound on a 64-bit `div`, on a machine shared
  with other benchmark ports. Treat the thread-scaling rows as a lower bound.


## How it works

`src/kernels.mojo` is a single compilation unit with four `abi("C")` exports.
Buffers cross the C ABI as 64-bit `Int` addresses and are rebuilt into
pointers inside each body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

1. **`mh_shingle_adler32`** walks every contiguous window of `k` tokens in a
   pre-joined buffer, re-inserting the single space that `" ".join` puts
   between tokens, and runs adler-32 over the resulting bytes. The number of
   windows is `max(1, n_tokens - k + 1)`, matching the reference's `maxlen`
   deque. The modulo is deferred: a token of up to 4096 bytes runs under one
   running modulus, which is taken at the end of the token or earlier if the
   sums approach `2**50`; a longer token reduces every byte. A reduction at any
   point gives the same residue, so both paths are exact.
2. **`mh_signature`** computes `out[h] = min_s (a[h] * x[s] + b[h]) % prime`. It
   runs `HASH_BLOCKS = 4` hash columns per shingle so the 64-bit divide has
   independent chains to interleave with, with the coefficients hoisted out of
   the shingle loop, then finishes any remainder with a scalar loop. The
   reduction mod `c` is done on 16-bit limbs to keep every intermediate below
   `2**51`.
3. **`mh_jaccard_rows`** fills the upper-triangular agreement matrix. Row `i`
   lives at a fixed stride of `n_docs` so rows stay at constant offsets under
   chunking, and the count is an exact integer comparison — only the final
   division is floating point. Each pairwise count (`_agree`) walks
   `AGREE_ACCS = 8` positions at a time into eight independent counters with a
   `comptime` trip count, which this toolchain unrolls and vectorises; the
   counters are summed at the end. The count is a sum of 0/1 terms, so the
   reassociation changes no value.
4. **`mh_score_hist`** counts each score into `clamp(trunc(score * scale))`,
   skipping the scores `np.histogram` drops. The common value (a finite score in
   range) takes one branch: multiply, floor, cast, one range comparison. Only
   the negative side (the floor-versus-truncation fix-up) and the overflow side
   (clamp versus drop an infinite) are one level down. Non-finite scores are
   rejected before the cast, because `Int64(Float64)` of NaN/+inf is
   unspecified on this toolchain.

Two techniques were tried and reverted because `pixi run bench` showed no gain:
a `comptime for` over the signature columns (identical emitted code, plus the
constant-array loads) and pooled count tables in the histogram (27.1 → 42.4 ms
on the raw kernel).

`python/mojo_minhash/_lib.py` is the ctypes bridge. It owns no memory: the
library borrows the caller's buffers for the duration of a call and writes only
into arrays the wrapper allocated. Every `argtypes` entry is `c_int64` where a
buffer is passed, because `c_int` would truncate the address and segfault above
4 GiB. The bridge validates lengths, dtypes and span bounds *before* calling,
since a mismatched `argtypes` is silent (ctypes zero-fills missing slots) and an
out-of-range length is an out-of-bounds read rather than an error.

Parallelism is a `ThreadPoolExecutor` over row or column chunks, not a Mojo
`parallelize`: the kernels are called through ctypes, where the worker-launch
path is the fragile one, and the GIL is released for the duration of each
ctypes call.

## Layout

```
build/build.sh              src/kernels.mojo -> dist/libmojo-minhash.so
src/kernels.mojo            one compilation unit, four @export kernels
python/mojo_minhash/_lib.py ctypes bridge; arity-checked argtypes, input validation
python/mojo_minhash/minhash.py  the ported public API
tests/                      342 tests, including parity against the real minhash
bench/bench.py              correctness-gated timing
bench/kernels_micro.py      raw-kernel timing, thread fan-out off
bench/analysis.md           ANALYZE phase: where the time goes
```

`MOJO_NOTES.md` records the Mojo dialect behaviour this port depends on
(verified by compiling against the pinned toolchain, not from documentation).