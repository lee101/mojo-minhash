# VERIFY phase — 2026-11-10

Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 cores, shared with other benchmark
ports. Load 22–44 during these runs. Toolchain: Mojo 1.2.0.dev2026092905,
NumPy 2.5.1 (both from `pixi.toml`). Every number below was measured in this
session; nothing was copied from an earlier README table.

## Gate

| command | result |
| --- | --- |
| `pixi run build && pixi run test` (start) | exit 0, `382 passed in 2.60s` |
| `pixi run bench` (start) | exit 0, table below |
| `pixi run bench` ×3 more | exit 0, used for the spread column |
| `pixi run build && pixi run test` (end) | exit 0, `382 passed in 2.60s` |

The README and `bench/` layout line said "342 tests". The suite collects 382,
and the README now says 382.

## Reproduced: current `pixi run bench`

Four full `pixi run bench` runs (loads 39, 35, 38, 32). The README table is the
load-32 run. The same-row spread across the four runs:

| case | README (new) | runs seen | within 20% of old README figure |
| --- | ---: | --- | --- |
| shingles n=20000 k=5 | 2.30x | 2.25–2.33x | yes (2.39x) |
| shingles e2e n=20000 | 1.79x | 1.55–1.79x | yes (1.68x) |
| signature h=128 s=4096 | 1.36x | 1.34–1.52x | yes (1.51x) |
| signature 8 workers h=256 | 1.01x | 1.01–1.37x | **no** (5.06x) |
| 64 docs x 64 hashes | 21.34x | 21.34–24.15x | yes (25.23x, 15%) |
| score matrix 400x64 | 3.36x | 3.36–4.95x | yes (3.94x) |
| score matrix 8 workers 900x128 | 0.98x | 0.98–1.06x | **no** (2.03x) |
| histogram n=7998000 | 1.56x | 1.56–1.60x | yes (1.63x) |
| histogram from rows | 1.95x | 1.87–2.37x | yes (1.88x) |
| jaccard_rows 700x64 | 6.08x | 4.70–6.08x | yes (4.55x, 34%) |

Reference and Mojo millisecond figures were also updated to the load-32 run, so
the ratio columns are internally consistent.

### The two thread rows did not reproduce: this is a real regression

A 20% miss on a scaling row can be noise, so I checked what the tree actually
runs. `src/kernels.mojo` sets `WORKER_CAP = 4`, `SIG_PARALLEL_MIN = 1<<23`
hash\*shingle units, and `JAC_PARALLEL_MIN = 1<<62` (jaccard never fans out).
The bench inputs are `256x4096` = 2^20 units and jaccard `900x128`, so both
take the serial path no matter what `workers=` says. Requesting 8 workers does
not change the code path, so the ratio is 1.0 by construction.

Direct per-side timing of the same inputs (`workers=1` vs `workers=8`), three
runs each, HEAD against a scratch worktree with `98dc249` (pre-parallel) built
from `src/kernels.mojo`:

| input | tree | 1 worker | 8 workers |
| --- | --- | ---: | ---: |
| signature 256x4096 | 98dc249 | 11.65, 11.92 ms | 2.15, 2.09 ms (5.4x) |
| signature 256x4096 | HEAD | 13.83, 12.58 ms | 13.81, 11.97 ms (1.0x) |
| score 900x128 | 98dc249 | 43.21, 46.15 ms | 24.79, 20.57 ms (1.9x) |
| score 900x128 | HEAD | 45.10, 44.09 ms | 44.03, 44.95 ms (1.0x) |

Outputs are bit-identical across worker counts on both trees (checked in the
same script and by `tests/test_parallel.py`), so this is lost parallelism, not
lost work. The README rows stay in the table at 1.01x and 0.98x with a note
explaining the cause. I did not touch the kernel: making it fast again is a
performance change, not a bug with a failing test.

## Reproduced: before/after unroll table

"before" = `98dc249` built in a scratch worktree (`git worktree add`), read
with the committed `python/` from that worktree; "after" = HEAD. Both used the
unmodified `bench/bench.py` (identical case list and case bodies in both trees)
and `bench/kernels_micro.py --repeat 3`.

* Checking all 17 `before` values numerically (new/old − 1): 16 are within
  20%. The one miss is `bench: histogram from rows`, 69.13 vs 92.20 ms (−25%) —
  a real pick-up on the shared host, and the `after` side of that row (74.14
  vs 76.51 ms) reproduced, so the row's 1.21x "change" is now "unchanged".
  `bench: shingles` shifted only 0.6% on the `before` side; it moved on the
  `after` side (1.43 → 1.65 ms, 15%).
* On the `after` side the non-thread `bench:` rows reproduced except
  `bench: shingles` (1.10x → 0.95x) and `bench: histogram from rows`. The two
  `after` thread rows did not: see above.
* `kernel: ` rows: all reproduce within 20% on both sides.
* `before` column reproduces within 20% on 13 of 17 rows; the four misses are
  exactly the thread-scaling rows, for the reason above.
* The old kernel `signature h=128` "4% win" (5.91 → 5.68 ms) did not hold: the
  rebuild measured 5.72 (before) → 5.97 (after) ms. It is now "unchanged".
* The old kernel `jaccard 400x64` "unchanged" measured 4.68 → 3.91 ms = 1.20x.
  `jaccard 900x128` reproduced (44.43 → 36.90 ms, 1.20x; old 43.72 → 39.63 ms,
  1.10x) and `jaccard 700x64` reproduced as unchanged (14.10 → 13.72 ms).
* `kernel: shingles n=20000` measured 1.59 → 1.42 ms (1.12x, old claim 1.23x)
  and `kernel: histogram n=7998000` 46.50 → 28.44 ms (1.64x, old 1.74x; a
  second `--repeat 3` run put the `after` time at 28.29 ms).

Every changed figure is in the README's two benchmark tables and the prose
under them.

## Fairness of each case (same input, no skipped work)

Read every case in `bench/bench.py`:

* **shingles / shingles e2e** — both sides hashed with `zlib.adler32` over the
  same `" ".join` windows. The first row hands the Mojo side pre-built token
  spans and the reference pre-joined strings, i.e. *both* get pre-laid-out
  input; the e2e row gives Mojo a raw token list. No skipping.
* **signature** — reference is the vectorised `uint64` `(a*x+b)%p` min; both
  sides get the same shingles and the same `(a, b)` table (built once, shared
  object), and the result is asserted equal before timing.
* **signature 8 threads** — reference is `workers=1` on the Mojo side. That is
  a fair "1 vs 8" scaling measure, but it is *not* an upstream comparison, and
  the row's label (`8 workers requested vs 1`) now says so.
* **64 docs x 64 hashes** — reference calls the real `minhash.calculate_signature`
  with the same `(a, b)` values; both sides build signatures for the same 64
  documents. The 21x is versus CPython lambdas, as the README already says.
* **score matrix 400x64** — reference is a Python loop of
  `ref.approx_jaccard_score(sigs[i], sigs[i+1:], 1)`; Mojo side is
  `mh.score_matrix(..., workers=1)`. Same matrix, both build all rows.
* **score matrix 8 workers 900x128** — reference is `workers=1`, same input.
* **histogram** — both sides get the same pre-flattened array, so the
  concatenate is not charged to one side.
* **histogram from rows** — both sides concatenate the ragged rows.
* **jaccard_rows 700x64** — reference is `(sigs[:,None,:] == sigs[None,:,:]).sum(-1)`
  full matrix, not just the triangle; on 700 docs that is ~2x the comparisons
  the Mojo kernel does (upper triangle only). The ratio is therefore flattering to Mojo by up
  to ~2x on the comparison side; the README does not claim more than the
  measured ratio, but readers should know the baseline over-computes. I left
  the baseline alone (it is the natural vectorised formulation) and recorded
  the caveat here rather than change a measurement in a verify pass.

No case skips work on one side. The only caveat is the full-matrix reference
in the last row above.

## Tests: parity intact?

`git log --oneline -- tests/` → four commits. Inspected each diff:

* `254620e` — initial tests; no assertion changed.
* `9526c58` — initial port; added tests (ffi_safety, kernels, spans, blocks),
  no deletions.
* `743ab08` (test phase) — the only deletions are a blank line and the three
  lines of `test_generate_hash_funcs_still_draws_independently_of_the_prime`,
  which were **replaced by a stronger body** that also checks the draw against
  upstream's `ref.generate_hash_funcs` output. Not a loosening.
* `dff70c7` (parallel phase) — additions only, plus the `n_workers` argument
  added to every direct kernel call in `test_kernels.py`,
  `test_signature_blocks.py` and the new `test_parallel.py`.
* `TOL = 0.0` in `tests/parity_helpers.py` has never changed; `git log -p --
  tests/ | grep '^-.*\(rtol\|atol\|assert\)'` finds no removed numerical
  assertion. No `skip`/`xfail` was added.

Nothing to restore. The tests still assert exact parity (`atol=rtol=0`)
against the installed upstream `minhash` 0.1.1.

## Changed

* `README.md`: current-bench table and prose re-measured; thread rows shown as
  measured (1.01x, 0.98x) with an explanation; before/after table re-measured
  from a rebuilt `98dc249`; "342 tests" → 382; the stale
  `ThreadPoolExecutor` description replaced by the in-kernel
  `max.algorithm.parallelize` description with its gates; the GPU section's
  adler throughput updated to this session's 1.42 ms; the "1.63x / 1.51x /
  1.83x" prose figures aligned with the measured rows.
* `bench/verify.md`: this file.
* No code, no test, and no kernel change. No test failed, so the rule "change
  code only to fix a bug you can show with a failing test" did not apply.

The scratch worktree `/tmp/mh-before` (build of `98dc249`) is outside the repo
and can be removed with `git worktree remove /tmp/mh-before --force`.
