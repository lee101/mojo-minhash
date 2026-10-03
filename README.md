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

Correctness is checked against the reference before every case is timed. The
table below is one `pixi run bench` run on this machine; re-run it to get
fresh numbers.

* Toolchain: Mojo 1.2.0.dev2026092905, NumPy 2.5.1
* Host: Intel Xeon E5-2697 v4 @ 2.30GHz, 72 cores
* This host is shared: load average sat around 70-76 on 72 cores during the
  runs, so the absolute times are noisy and the thread-scaling rows in
  particular move by 30-60% between runs. The single-threaded rows were stable
  to within about 10% over three consecutive runs.

| case | reference | mojo-minhash | ratio |
| --- | ---: | ---: | ---: |
| shingles, n=20000 k=5, pre-laid-out spans | 3.08 ms | 1.55 ms | **1.99x** |
| shingles end to end from a token list | 3.00 ms | 2.09 ms | **1.44x** |
| signature, h=128 s=4096, vs vectorised NumPy uint64 | 8.15 ms | 5.84 ms | 1.40x |
| signature, h=256 s=4096, 8 threads vs 1 | 11.77 ms | 3.21 ms | 3.67x |
| 64 docs x 64 hashes, vs `minhash.calculate_signature` | 686.74 ms | 26.27 ms | **26.15x** |
| score matrix 400x64, vs `approx_jaccard_score` | 16.13 ms | 4.64 ms | **3.48x** |
| `mh_jaccard_rows` 700x64, vs vectorised NumPy triangle | 68.05 ms | 15.00 ms | **4.54x** |
| score matrix 900x128, 8 threads vs 1 | 43.07 ms | 19.56 ms | 2.20x |
| histogram, n=7998000, vs `np.histogram` | 31.26 ms | 35.69 ms | **0.88x (loss)** |
| histogram from the row list `score_matrix` returns | 140.49 ms | 69.99 ms | 2.01x |

Reading the numbers honestly:

* **The one loss is the flat-array histogram, 0.88x.** `np.histogram` over 8M
  values is already a tight two-pass NumPy primitive, while this kernel is a
  single pass with a scattered increment into 11 bins. Calling `score_histogram`
  with the ragged row list instead — the way a user actually reaches it, since
  that is what `score_matrix` returns — is 2.01x, because the reference then
  pays a NumPy concatenate too.
* **The headline 26x is against CPython, not against NumPy.** The reference's
  `calculate_signature` applies a Python lambda once per shingle per hash, so
  that row measures the interpreter, not the arithmetic. The honest comparison
  for the kernel itself is the `signature, h=128 s=4096` row at 1.40x against
  the fastest vectorised uint64 NumPy formulation the same arithmetic allows.
* **Threading scales worse than the core count.** 3.67x on 8 workers for a
  kernel that is genuinely compute-bound on a 64-bit `div`, with load average
  near the core count. Treat the thread-scaling rows as a lower bound.

## How it works

`src/kernels.mojo` is a single compilation unit with four `abi("C")` exports.
Buffers cross the C ABI as 64-bit `Int` addresses and are rebuilt into
pointers inside each body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

1. **`mh_shingle_adler32`** walks every contiguous window of `k` tokens in a
   pre-joined buffer, re-inserting the single space that `" ".join` puts
   between tokens, and runs adler-32 over the resulting bytes. The number of
   windows is `max(1, n_tokens - k + 1)`, matching the reference's `maxlen`
   deque.
2. **`mh_signature`** computes `out[h] = min_s (a[h] * x[s] + b[h]) % prime`. It
   runs `HASH_BLOCKS = 4` hash columns per shingle so the 64-bit divide has
   independent chains to interleave with, then finishes any remainder with a
   scalar loop. The reduction mod `c` is done on 16-bit limbs to keep every
   intermediate below `2**51`.
3. **`mh_jaccard_rows`** fills the upper-triangular agreement matrix. Row `i`
   lives at a fixed stride of `n_docs` so rows stay at constant offsets under
   chunking, and the count is an exact integer comparison — only the final
   division is floating point.
4. **`mh_score_hist`** zeroes its output, then counts each score into
   `clamp(trunc(score * scale))`, skipping the scores `np.histogram` drops.

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
tests/                      168 tests, including parity against the real minhash
bench/bench.py              correctness-gated timing
```

`MOJO_NOTES.md` records the Mojo dialect behaviour this port depends on
(verified by compiling against the pinned toolchain, not from documentation).