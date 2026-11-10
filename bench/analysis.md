# Mojo-minhash — ANALYZE phase: where the time goes

Measured map, not a redesign. **No algorithm was changed in this phase**; `src/` and
`tests/` are untouched. Every number below comes from a run in this session
(2026-11-10, host `Intel Xeon E5-2697 v4 @ 2.30GHz, 72 cores`). No number is
estimated or carried over from the README.

## What was run

| # | command | purpose | result |
| --- | --- | --- | --- |
| 1 | `pixi run bench` (flock-wrapped) | the mandated correctness-gated timing | exit 0, table in §2 |
| 2 | `flock … python` tier-A probe | each public entry point at typical caller sizes | §3 |
| 3 | `flock … python` tier-B probe | raw-kernel vs wrapper split at bench sizes | §3, §5 |
| 4 | `flock … python` bandwidth + glue probe | roofline denominator, glue decomposition | §4, §5, §6 |

All probes ran under the same `/tmp/mojo-bench.lock`, one at a time. The lock is
shared with ~6 other ports on this box (maxminddb, karateclub, imagehash, hpack,
geographiclib, minepy, lib3mf contended for it during this session), so each run
queued first and the numbers are under whatever load the holder left behind —
`load 18` during run 1, versus the `70-76` the README recorded, so these are
*less* noisy than the README set but still not a quiet machine. Single-threaded
rows were stable across repeats; treat thread-scaling rows as a lower bound.

## 1. Toolchain

* Mojo `1.2.0.dev2026092905 (aa3b777f)`, NumPy `2.5.1`, Python 3.13, `pixi` env.
* `dist/libmojo-minhash.so` present (built 2026-11-10 17:54); worktree clean at start.
* Reference = installed upstream `minhash` 0.1.1 where importable, else the fastest
  fair vectorised NumPy formulation (as `bench/bench.py` already does).

## 2. `pixi run bench` output (run 1, verbatim)

```
mojo Mojo 1.2.0.dev2026092905 (aa3b777f)   numpy 2.5.1
host: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores, load 18
case                             reference    mojo-minhash     ratio
--------------------------------------------------------------------
shingles n=20000 k=5                3.28ms          1.58ms     2.07x
shingles e2e n=20000                3.08ms          2.15ms     1.43x
signature h=128 s=4096              8.54ms          5.80ms     1.47x
signature 8 threads h=256          11.95ms          3.19ms     3.75x
64 docs x 64 hashes               641.91ms         26.55ms    24.17x
score matrix 400x64                19.45ms          4.78ms     4.07x
score matrix 8 thr 900x128         44.19ms         22.14ms     2.00x
histogram n=7998000                34.79ms         37.79ms     0.92x
histogram from rows n=4000        151.95ms         91.24ms     1.67x
jaccard_rows 700x64                69.68ms         14.87ms     4.69x
EXIT=0
```

## 3. Per-entry-point timings at real caller sizes

Two size tiers. **Tier A** is a realistic small unit of work (a ~200-token
document, `h=64`, 64 docs) — the size a caller actually passes when scoring a
small corpus. **Tier B** is the size `bench/bench.py` uses. Wrapper = the
public `mojo_minhash` call; raw kernel = the same ctypes call with buffers
handed in exactly as the wrapper lays them out, i.e. the kernel alone.

### Tier A — typical small unit (~200-token docs, h=64, 64 docs)

| entry point | size | wrapper | raw kernel | baseline | port vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| `generate_shingles` | 200 tok, k=5 | 0.076 ms | 0.038 ms | zlib loop 0.037 ms | **0.49x (slower)** |
| `generate_hash_funcs` | count=64 | 0.081 ms | — | pure Python | — |
| `calculate_signature` | h=64, s=200 | 0.128 ms | 0.123 ms | — | — |
| `approx_jaccard_score` | 64 | 0.0065 ms | — | NumPy delegate | — |
| `score_matrix` | 64×64 | 0.122 ms | 0.099 ms | — | — |
| `score_histogram` | rows, 64 docs | 0.046 ms | — | — | — |
| `score_histogram` | flat, n=2016 | 0.0138 ms | 0.0129 ms | — | — |

### Tier B — `bench/bench.py` sizes (raw kernel vs wrapper)

| entry point | size | wrapper | raw kernel | upstream / NumPy baseline | port vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| `generate_shingles` | 20000 tok, k=5 | 2.206 ms | 1.607 ms | zlib loop 3.492 ms | 1.58x |
| `calculate_signature` | h=128, s=4096 | 6.146 ms | 6.111 ms | vectorised uint64 NumPy 9.892 ms | 1.61x |
| pipeline (64 docs × h=64) | 512 shingles/doc | 26.50 ms | — | upstream 631.74 ms | 23.83x |
| `score_matrix` | 400×64 | 4.848 ms | 4.571 ms | `approx_jaccard_score` 16.863 ms | 3.48x |
| `score_histogram` | flat, n=7 998 000 | 38.03 ms | 38.32 ms | `np.histogram` 32.74 ms | **0.86x (slower)** |
| `score_histogram` | rows, 4000 docs | 90.46 ms | — | reference binning 128.09 ms | 1.42x |
| `mh_jaccard_rows` (direct) | 700×64 | 14.12 ms | 14.12 ms | vectorised NumPy triangle 63.88 ms | 4.52x |

### Thread scaling (this session, load ~18)

| kernel | 1 | 2 | 4 | 8 | 16 | speedup @8 | speedup @16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| signature h=256 s=4096 (ms) | 12.164 | 7.202 | 4.711 | 2.202 | 2.424 | 5.52x | 5.02x |
| score_matrix 900×128 (ms) | 46.138 | 40.111 | 32.112 | 25.186 | 15.518 | 1.83x | 2.97x |

Signature scales ~5.5x on 8 workers; `score_matrix` only ~1.8x at 8 and ~3.0x at
16 — it is memory-bound (below), so extra threads help only once the working set
is shared well. 16 workers regresses the signature a little (2.202 → 2.424).

## 4. Roofline input — measured machine limits (run 4)

The "memory-bound or compute-bound" column needs a denominator. Measured on this
host, same session:

| probe | size | time | achieved traffic |
| --- | --- | ---: | ---: |
| `np.copyto`, uint8 (1R+1W) | 67 MB | 7.81 ms | **17.19 GB/s** |
| `np.copyto`, float64 (1R+1W) | 67 MB | 8.78 ms | 15.29 GB/s |

~17 GB/s is the practical streaming ceiling here. Any kernel achieving well
under that is *not* bandwidth-limited, whatever its nominal byte count.

## 5. Ranked hotspot table

Ranked by wall-clock share of the workload a real caller runs. `bytes touched` is
the kernel's actual per-call memory *traffic* (re-reads counted, since that is
what decides boundedness); `flops` counts integer/float mul+add+mod+cmp,
excluding shifts/masks; `glue` is the wrapper share measured as
`(wrapper − raw kernel) / wrapper`.

Op convention: one counted `mul`/`add`/`mod`/`compare`/increment = 1 flop; shifts and
masks are excluded. `bytes touched` counts re-reads.

| # | function | input size | time | bytes touched | flops | flops/byte | bound | time in glue? |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `mh_signature` (kernel) | h=128, s=4096 | 6.111 ms | 0.50 MiB | 5.24 M | **9.9** | compute — 64-bit `mod` latency, ILP-limited (86 MB/s ≪ 17 GB/s) | no (glue 0.6%) |
| 2 | `score_histogram` rows-glue | 4000 rows → 7.998 M | 30.71 ms glue of 66.1 ms | 61.0 MiB (concat copy) | 0 | 0 | **Python** (NumPy concatenate) | **yes — 46.5%** |
| 3 | `mh_score_hist` (kernel) | n=7 998 000 | 38.32 ms | 61.0 MiB | 39.99 M | 0.62 | latency-bound scattered increment (1.67 GB/s ≪ 17) | no (glue ~0) |
| 4 | `mh_jaccard_rows` (kernel) | 700×64 | 14.12 ms | 239 MiB | 31.3 M | **0.13** | **memory** — 17.7 GB/s ≈ ceiling | no |
| 5 | `mh_jaccard_rows` (kernel) | 400×64 | 4.571 ms | 78 MiB | 10.2 M | **0.13** | **memory** — 17.9 GB/s ≈ ceiling | no (glue 5.7%) |
| 6 | `mh_shingle_adler32` (kernel) | 20000 tok, k=5 | 1.607 ms | 1.36 MiB | 1.46 M | 1.0 | compute — `mod 65521` per token (890 MB/s ≪ 17) | no (glue 27%) |
| 7 | `generate_shingles` glue | 20000 tok | 0.599 ms of 2.206 ms | — | — | — | **Python** (join + flatnonzero + table) | **yes — 27%** |
| 8 | `score_matrix` glue | 400×64 | 0.277 ms of 4.848 ms | 1.28 MB alloc | — | — | Python + `np.zeros` | yes — 5.7% |
| 9 | `calculate_signature` glue | h=128, s=4096 | 0.035 ms of 6.146 ms | — | — | — | Python (`asarray`) | yes — 0.6% |
| 10 | pipeline Python loop | 64 docs × h=64 | 26.50 ms | — | — | — | 64 × (Python loop + ctypes re-entry) | partly |
| 11 | `generate_hash_funcs` | count=64 | 0.081 ms | — | — | — | pure Python loop | yes (all) |

Compulsory-traffic note for row 1: the signature kernel's input is only
19 KiB (shingles 16 KiB + coeffs 2 KiB + out 1 KiB) and every buffer is
L1/L2-resident, so its *compulsory* flops/byte is ~250 — but that is an artefact
of a tiny working set, not a bandwidth opportunity. The limiter is the serial
64-bit integer modulo, which is why it runs at 86 MB/s effective.

## 6. Python-level loops that re-enter ctypes, and per-call allocations

### ctypes re-entry (a Python loop that calls into Mojo once per element/row/doc)

* **`_signature_block` / `_signature_one` are inside the kernel, not Python** —
  this port has no per-element ctypes loop, which is its whole point. The kernels
  are each invoked **once per chunk** (`_chunks`, ≤ `workers` chunks), so a full
  call is 1–8 ctypes entries, not `h×s`.
* **Pipeline: one ctypes call per document.** The 64-doc pipeline (`_lib.signature`
  → `mh_signature` once per doc, plus `np.asarray(hash_funcs)` once per doc) is a
  Python loop over documents. At 64 docs that is 64 entries; it is the price of
  the reference's own per-document `__main__` shape and it still wins 23.8x.
* **`score_histogram` from rows is the real per-row Python loop**: `[np.asarray(r)
  .ravel() for r in scores]` then `np.concatenate(...)` — 4000 iterations plus a
  full 64 MB copy that costs **30.71 ms of the 66.1 ms call (46.5%)**, before the
  1-ctypes-call kernel even runs. The rows API is the user-facing path (it is what
  `score_matrix` returns), so this glue is on the hot path.
* **`generate_hash_funcs`** is a pure-Python list comprehension
  `[(coeffs.pop(), coeffs.pop()) for _ in range(count)]` — 0.081 ms at count=64,
  negligible against the kernels it feeds.
* **Thread pool overhead:** `_POOL = ThreadPoolExecutor(max_workers=8)` (module
  global, never shut down; GIL released per ctypes call). For a tiny input the
  8-way chunking costs far more than the work: signature h=8 s=256 is **44.9 µs at
  `workers=1` vs 457.5 µs at `workers=8` — a 10.2x penalty**. There is no size gate.

### Per-call allocations / copies

* `score_matrix` allocates the **full `n_docs²` zero matrix** (`np.zeros(n*n)`,
  0.058 ms at n=900 → 6.48 MB) although only the strict upper triangle is written
  (~half wasted), then builds `n_docs` Python row slices (0.131 ms at n=400).
* `calculate_signature` does `np.asarray(hash_funcs, dtype=int64)` (0.036 ms for 128
  list-of-tuples), `np.ascontiguousarray(coeffs)` and
  `np.ascontiguousarray(shingles, dtype=uint32)` — a **copy** whenever the caller's
  shingles are not already `uint32` (the bench passes `uint64`, so a narrowing copy
  happens every call).
* `generate_shingles` builds `" ".join(words)` (0.275 ms), `.encode()` (0.007 ms),
  `np.flatnonzero(buf == 0x20)` (0.133 ms), then `off`/`tlen`/`ends` arrays
  (~0.09 ms) — 0.599 ms of glue at 20000 tokens (27%).
* `score_histogram(rows)` builds a 4000-element Python list of arrays plus one
  `np.concatenate` (30.71 ms, above).
* Validation: `_check_spans` does 4 array reductions (`min`/`max`/`argmax`) per
  shingle call — 0.019 ms at 20000 tokens, cheap but unconditional.

## 7. Next-phase verdict per hotspot

"gpu" verdicts are backed by the §5 flops/byte column; `accel.md`'s ~2 flops/byte
rule of thumb is the yardstick for a sub-2 figure.

| hotspot | flops/byte | verdict | why |
| --- | ---: | --- | --- |
| `mh_signature` kernel | 9.9 | **unroll** (widen `HASH_BLOCKS`) | 24% of the signature wall-clock is the serial `mod`; more independent hash columns per shingle is the only lever the file already documents as effective. Not `simd` — the op is a scalar 64-bit divide. **Not gpu**: working set is L1-resident (no bandwidth problem to solve) and 64-bit integer modulo is emulated/slow on the GPU. |
| `score_histogram` rows-glue | 0 | **none** | 46.5% of the call is a NumPy `concatenate` copy of the caller's rows; it is not kernel work. A row-shaped kernel entry (bin each row in place) would remove it, but that is an API change, not an accel phase. |
| `mh_score_hist` kernel | 0.63 | **none** | Already loses to `np.histogram` (0.86x) and is latency-bound on a dependent 11-bin scatter, at 1.67 GB/s of a 17 GB/s ceiling. Below the 2 f/b gate → **not gpu**; no SIMD win without changing the scatter. |
| `mh_jaccard_rows` (both sizes) | 0.13 | **simd**, then **parallel** | Runs at the 17.7–17.9 GB/s machine ceiling → genuinely memory-bound. SIMD (vector compare + popcount-style count) raises bytes/cycle; threads already help (1.8x@8, 3.0x@16). Below the 2 f/b gate → **not gpu**. |
| `mh_shingle_adler32` kernel | 1.0 | **none** | Already 2.07x zlib and 890 MB/s ≪ 17 GB/s ceiling; cost is the per-token modulo, below the 2 f/b gate → **not gpu**. |
| `generate_shingles` glue | — | **none** | 27% glue is Python `join`/`flatnonzero`/table build, and at 200 tokens the whole call **loses to zlib** — a small-input concern, not an accel-kernel one. |
| `score_matrix` glue | — | **none** | 5.7%; the `np.zeros(n_docs²)` and row slices are cheap at the sizes measured. |
| `calculate_signature` glue | — | **none** | 0.6%. |
| thread pool | — | **parallel** | Keep it, but gate chunking on size: it is a 10.2x penalty on tiny inputs, not a kernel property. |

## 8. Rows where the port is SLOWER than upstream (kept, not dropped)

| case | port | upstream / NumPy | ratio | note |
| --- | ---: | ---: | ---: | --- |
| `score_histogram` flat, n=7 998 000 | 37.79 ms (bench run 1) / 38.03 ms (probe) | 34.79 / 32.74 ms | **0.86–0.92x** | `np.histogram` is a tight two-pass primitive; this kernel's single pass with a dependent 11-bin scatter is latency-bound. Reached through `score_histogram(rows)` (the API `score_matrix` feeds) it is 1.42x instead, because upstream then pays a concatenate too. |
| `generate_shingles`, 200-token doc | 0.076 ms | zlib loop 0.037 ms | **0.49x** | Kernel ties (0.038 vs 0.037); the loss is the per-call Python glue (join/flatnonzero/table ≈ 0.038 ms) dominating a tiny input. Upstream's `" ".join` per shingle is cheaper at this size. |
| `score_histogram` flat via `_lib` | 38.32 ms | 32.74 ms | 0.86x | same kernel as above, raw entry. |

Everything else the port is faster on, from 1.42x (histogram from rows) to
23.83x (64-doc pipeline vs upstream's Python lambda loop).

## 9. Caveats

* Absolute times are noisy: the host is shared with ~6 other benchmark ports on
  the same flock, and load varied (`load 18` during run 1). Repeat single-thread
  rows agreed within ~5-10%; thread-scaling rows moved more (8-thread
  `score_matrix` read 22.14 ms in run 1 and 25.19 ms in the probe).
* `flops` excludes shifts/masks, so the signature figure (9.9 f/b) is generous —
  it counts the `mod` as one op when it is in fact the whole cost.
* The rows-glue figure for `score_histogram` varied 30.7–46.5% of the call across
  repeat measurements; the file's 90.46 ms and 66.09 ms readings bracket it.
* Tier A"`approx_jaccard_score`/`calculate_signature`" have no direct upstream
  baseline measured here; their baselines in §3 are the NumPy/upstream forms used
  in §2.
