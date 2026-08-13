# Part 2 — Embed it (Software Prefetching + SIMD)

Optimizes a sum-pooled embedding lookup — the gather-and-reduce pattern behind
recommendation-model embedding bags. For each bag of input indices, the rows named by
those indices are gathered from a large embedding table and summed into one output
vector. The table is far larger than any cache level, so each gather is effectively a
random memory access. The goal is to hide that latency with software prefetching and
cut per-element work with SIMD.

Built with `-O0 -mavx2 -mfma -msse2 -mavx512f` — ISA flags only, no compiler
auto-vectorization.

| Task | Covered by |
|---|---|
| 2A Software prefetching | prefetch distance + cache fill level sweeps, CPU metrics, `sw_prefetch_access` |
| 2B SIMD | 128 / 256 / 512-bit widths × embedding dimensions, instruction counts |
| 2C Prefetching + SIMD | both applied together, all 2A and 2B analyses repeated |

## Measurement methodology

Performance counters are **gated to the kernel region** using perf's control FIFO
(`perf stat --delay=-1 --control fifo:ctl,ack`); `emb.cpp` writes `enable` / `disable`
around the timed loop. This matters: the embedding table is up to 1 GB and is filled
with `mt19937`, which costs ~86 billion instructions. Measuring the whole process makes
every variant look identical (all within 0.1%) because initialization swamps the ~8
million instructions of the kernel. With gating, a 10× change in table size leaves the
instruction count flat, as it should, since the kernel does the same work either way.

The table is evicted with `clflush` + `mfence` between runs so each variant starts cold.
Execution times are the median of 3 runs, measured inside the kernel.

## Build & run

```bash
make                    # builds bin/emb
make verify             # builds bin/verify_emb

./bin/emb               # prints "run,us" CSV for all four variants
./bin/verify_emb        # checks SIMD sums match the scalar reference

./run_experiments.sh    # all sweeps -> results/*.csv  (~10 min)
python3 plot_results.py # -> plots/*.png
python3 make_tables.py  # -> results/tables.md (Tables 2.1, 2.2, 2.3)
```

Everything is configurable by environment variable (defaults shown):

| Variable | Default | Meaning |
|---|---:|---|
| `EMB_TABLE_SIZE` | 1000000 | Rows in the embedding table |
| `EMB_DIM` | 128 | Embedding dimension per row |
| `EMB_INPUT_SIZE` | 720 | Total lookups across all bags |
| `EMB_NUM_BAGS` | 20 | Bags the lookups are split into |
| `EMB_PREFETCH_DIST` | 4 | Lookups to prefetch ahead |
| `EMB_HINT` | 0 | Cache fill level: 0=T0, 1=T1, 2=T2, 3=NTA |
| `EMB_SIMD_WIDTH` | 256 | SIMD width in bits: 128 / 256 / 512 |
| `EMB_ONLY` | 0 | Run one variant: 1=naive, 2=prefetch, 3=simd, 4=prefetch+simd |
| `EMB_REPS` | 1 | Repeat the kernel N times |

## Results (Intel i9-11900H, single-threaded, `-O0`)

Execution time in µs, dim=128, 2048 lookups, pd=4, hint=T0, AVX2:

| Table rows | Naive | Prefetch | SIMD | Prefetch+SIMD | Best speedup |
|---|---:|---:|---:|---:|---:|
| 200K | 715 | 559 | 399 | 245 | 2.92× |
| 500K | 747 | 577 | 410 | 250 | 2.99× |
| 1M | 770 | 573 | 427 | 260 | 2.96× |
| 2M | 823 | 591 | 436 | 279 | 2.95× |
| 4M | 871 | 667 | 513 | 332 | 2.62× |

**Instruction count halves with each doubling of SIMD width** (dim=512, kernel only):

| No SIMD | 128-bit | 256-bit | 512-bit |
|---:|---:|---:|---:|
| 32.85 M | 16.14 M | 8.31 M | 4.38 M |

**Prefetching mainly removes LLC misses.** At a 200K-row table the LLC misses drop from
2,289 (naive) to 432 with prefetching, for ~16,300 prefetch requests issued — confirmed
by the `sw_prefetch_access` counter, which reads 0 for the non-prefetch variants.

Best overall configuration is **prefetch + 512-bit SIMD**, reaching **3.16×** over naive
at dim=128 and **2.86×** at dim=512.

### Answers to the assignment questions

- **Trend with embedding table size** — prefetching helps *more* as the table grows
  (1.28× at 200K to 1.39× at 2M): a bigger table means a lower hit rate, so there is
  more memory latency available to hide. At 4M the gain falls back to 1.31× as the
  prefetches themselves start missing and TLB pressure rises.
- **Best prefetch distance** — 1–4 lookups ahead (flat optimum, 567–582 µs). Beyond 8 it
  degrades steadily to 703 µs at pd=64: prefetching too far ahead evicts lines before
  use, and the end-of-bag guard issues fewer useful prefetches (16.8K at pd=1 vs 6.2K at
  pd=64).
- **Best cache fill level** — T2 and NTA are marginally best (564/563 µs vs 587 µs for
  T0), and T2 shows the lowest LLC misses (456). Each row is streamed once and never
  reused, so pulling it all the way into L1 buys nothing; the spread is only ~4% and
  close to run-to-run noise.
- **Best SIMD width** — 512-bit at every embedding dimension, up to 3.34× over the
  scalar kernel at dim=512.

Full metric tables (L1D / L2 / LLC misses, prefetch requests, instructions, execution
time, speedup) for every swept configuration are in
[`results/tables.md`](results/tables.md).

## Layout

```
part2/
├── emb.cpp, verify_emb.cpp
├── Makefile, run_experiments.sh, plot_results.py, make_tables.py
├── results/   size_sweep.csv, prefetch_distance.csv, hint.csv,
│              simd_width.csv, perf.csv, tables.md
└── plots/     speedup_vs_table_size, prefetch_distance, hint_sweep,
               simd_width_vs_dim, cache_misses, instruction_count,
               combined_comparison
```

## Not covered

Task 2A deliverable 5 (enabling/disabling the **hardware prefetchers** via MSR `0x1a4`
and analysing the impact) is not included — it needs root access on the test machine.
