# mojo-minhash

A Mojo port of the numeric core of the [`minhash`](https://pypi.org/project/minhash/)
package (version 0.1.1).

## What upstream `minhash` actually is

`minhash` is a 112-line single-file script, not a library with a numeric core
in the usual sense. Reading the installed source (`minhash.py` in the test venv)
it is four loops plus a `__main__`:

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

| Upstream | Ported | Mojo kernel | Notes |
| --- | --- | --- | --- |
| `generate_shingles` | yes | `mh_shingle_adler32` | returns a uint32 array instead of a generator; see deviations |
| `generate_hash_funcs` | partly | — | the `random.sample` draw is Python; the `lambda` is replaced by coefficient pairs |
| `calculate_signature` | yes | `mh_signature` | exact, not approximate — see "Exactness" below |
| `approx_jaccard_score(axis=0)` | yes | — | single-pair compare, delegated to NumPy |
| `approx_jaccard_score(axis=1)` | yes | `mh_jaccard_rows` | via `score_matrix`, which is the block form the reference uses |
| binning loop in `__main__` | yes | `mh_score_hist` | via `score_histogram` |
| `__main__` file IO | no | — | `docs.json` / `signatures.json` / `coeffs.json` / `discovered_dups` |
| `generate_shingles(mapper=...)` | no | — | only adler-32 is compiled; a custom mapper stays in Python |
| `gensim.simple_preprocess` | no | — | third-party tokenisation, not this package's work |

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

### Not implemented, and why

* The whole `__main__` CLI. It is file IO and `print`; there is nothing in it
  to compile.
* A custom `mapper` for `generate_shingles`. Only adler-32 is compiled.
* Any other MinHash implementation. This port targets `minhash` 0.1.1 as
  installed. (`datasketch` and `nilsimsa` are separately installed in the same
  venv; neither is this package.)

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

## The Mojo cast trap

`Int64(Float64)` in Mojo 1.2.0 is **round-half-to-even**, while
`ndarray.astype(int)` in NumPy truncates toward zero. The reference's binning
computes `(row * 10).astype(int) / 10`, so a score of exactly `3/20 == 0.15`
— whose product with 10 is exactly `1.5` — lands in bin 1 under truncation and
bin 2 under round-half-to-even. The kernel uses an explicit `floor` and a
negative-side correction; `test_truncation_is_toward_zero_not_round_half_to_even`
fails if the cast is used directly.

## Install and run

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh   # sets MODULAR_HOME
cd /nvme0n1-disk/code/mojo-minhash
bash build/build.sh                              # -> dist/libmojo-minhash.so
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python bench/bench.py
```

`mojo_minhash` imports alongside the real `minhash`; the tests import both.

```python
import mojo_minhash as mh

shingles = mh.generate_shingles("the quick brown fox".split(), 2)  # uint32
pairs = mh.generate_hash_funcs(64, seed=0)                        # (a, b) pairs
sig = mh.calculate_signature(shingles, pairs)                     # int64[64]
```

## Benchmarks

Correctness is checked against the reference before every case is timed.
Measured on this box (`mojo build` 1.2.0.dev2026092605, Xeon E5-2697 v4) with
`python bench/bench.py`. **This machine is heavily shared — load average was
around 290 on 72 cores during these runs — so the absolute times and the
thread-scaling rows are noisy; the ratios move by 20-40% between runs.**

| case | reference | mojo-minhash | ratio |
| --- | ---: | ---: | ---: |
| shingles, n=20000 k=5, pre-laid-out spans | 16.05 ms | 2.13 ms | **7.5x** |
| shingles end to end from a token list | 6.78 ms | 7.82 ms | **0.87x (loss)** |
| signature, h=128 s=4096, vs vectorised NumPy uint64 | 13.01 ms | 8.98 ms | 1.45x |
| signature, h=256 s=4096, 8 threads vs 1 | 20.04 ms | 15.99 ms | 1.25x |
| 64 docs x 64 hashes, vs `minhash.calculate_signature` | 1329.09 ms | 41.52 ms | **32.0x** |
| score matrix 400x64, vs `approx_jaccard_score` | 54.19 ms | 16.39 ms | **3.3x** |
| score matrix 900x128, 8 threads vs 1 | 92.22 ms | 71.33 ms | 1.29x |
| histogram, n=7998000, vs `np.histogram` | 37.95 ms | 47.49 ms | **0.80x (loss)** |
| histogram from the row list `score_matrix` returns | 205.70 ms | 84.33 ms | 2.44x |

Reading the losses honestly:

* **shingles end to end, 0.87x.** The compiled kernel is 7.5x faster than
  `zlib.adler32` in a Python loop, but `generate_shingles` first has to build
  the token offset table, and for tokens of 3-8 bytes that Python-level
  bookkeeping costs about as much as hashing 30 bytes. Use
  `mojo_minhash._lib.adler_shinges_from_spans` when the token table is already
  available, which is the `shingles, pre-laid-out spans` row.
* **histogram on a flat array, 0.80x.** `np.histogram` over 8M values is
  already a tight two-pass NumPy primitive, and this kernel is a single pass
  with a scattered increment into 11 bins. Calling `score_histogram` with the
  ragged row list instead — the way a user reaches it — is 2.44x, because the
  reference then pays a NumPy concatenate too.
* **Threading is under-expressed** (1.25x on 8 workers, not the ~3x the
  toolchain reference measures on an idle box). The kernel body is a 64-bit
  `div` per element, which is genuinely compute-bound, but with load average
  near 290 on 72 cores the workers are competing with everything else on the
  machine. Treat these two rows as a lower bound, not as a measurement of the
  kernel.

## Layout

```
build/build.sh              src/kernels.mojo -> dist/libmojo-minhash.so
src/kernels.mojo            one compilation unit, four @export kernels
python/mojo_minhash/_lib.py ctypes bridge; arity-checked argtypes
python/mojo_minhash/minhash.py  the ported public API
tests/                      74 parity tests against the real minhash package
bench/bench.py              correctness-gated timing
```

`src/kernels.mojo` keeps the exported bodies `abi("C")`, takes every buffer as a
64-bit `Int` address and rebuilds the pointer inside, and is a single
compilation unit so the build stays at a few seconds.
