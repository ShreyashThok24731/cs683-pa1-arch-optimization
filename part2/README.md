# Part 2 — Embedding Table Lookup (Software Prefetch + SIMD)

Optimizes a sum-pooled embedding lookup — the gather-and-reduce pattern used by
recommendation-model embedding bags: for each "bag" of input indices, gather the
corresponding rows from a large embedding table and sum them into one output vector.
The table is far larger than any cache level, so each gather is close to a random
memory access; the goal is hiding that latency with software prefetching and cutting
per-element work with SIMD.

Built with `-O0 -mavx2 -mfma -msse2 -mavx512f` — ISA flags only, no compiler
auto-vectorization.

## Optimization variants

| Variant | Technique |
|---|---|
| Naive | Scalar gather + sum, one table row at a time |
| SW prefetch | `_mm_prefetch` issued `PREFETCH_DISTANCE` lookups ahead, one instruction per 64B cache line of the row, hint configurable (T0/T1/T2/NTA) |
| SIMD | Vectorized accumulate over the embedding dimension, width configurable (SSE 128 / AVX2 256 / AVX-512 512) |
| Prefetch + SIMD | Both combined |

Between every timed run the table is evicted with `clflush` + `mfence` so each variant
starts from a cold cache.

## Build & run

```bash
make                                        # builds bin/emb

./bin/emb                                   # prints "run,us" CSV for all four variants

g++ -O0 -mavx2 -mfma -msse2 -mavx512f verify_emb.cpp -o bin/verify_emb && ./bin/verify_emb
                                             # checks SIMD sums match scalar reference

./run_experiments.sh                        # sweeps table size / prefetch distance /
                                             # hint / SIMD width, writes results/*.csv
                                             # (~15 min at the largest table sizes)
python3 plot_results.py                     # renders plots/*.png
```

All parameters are overridable via environment variables (defaults shown):

| Variable | Default | Meaning |
|---|---:|---|
| `EMB_TABLE_SIZE` | 1000000 | Rows in the embedding table |
| `EMB_DIM` | 128 | Embedding dimension per row |
| `EMB_INPUT_SIZE` | 720 | Total lookups across all bags |
| `EMB_NUM_BAGS` | 20 | Number of bags the lookups are split into |
| `EMB_PREFETCH_DIST` | 4 | Lookups to prefetch ahead |
| `EMB_HINT` | 0 | Prefetch hint: 0=T0, 1=T1, 2=T2, 3=NTA |
| `EMB_SIMD_WIDTH` | 256 | SIMD width in bits: 128 / 256 / 512 |
| `EMB_ONLY` | 0 | Restrict to one variant: 1=naive, 2=prefetch, 3=simd, 4=prefetch+simd |

## Results (Intel i9-11900H, single-threaded, `-O0`)

Table = 1M rows × 128 dim, input = 2048 lookups:

| Variant | µs | Speedup |
|---|---:|---:|
| Naive | ~850 | 1.0× |
| Software prefetch (T0, pd=4) | ~600 | 1.4× |
| SIMD (AVX2 8-wide) | ~500 | 1.7× |
| **Prefetch + SIMD (AVX2)** | **~320** | **2.7×** |
| **Prefetch + SIMD (AVX-512)** | **~230** | **3.7×** |

`results/` also holds four parameter sweeps (`size_sweep.csv`, `prefetch_distance.csv`,
`hint.csv`, `simd_width.csv`) plotted in `plots/`: speedup vs. table size, execution
time vs. prefetch distance, cache-fill hint comparison, and speedup vs. SIMD width
across embedding dimensions.

## Layout

```
part2/
├── emb.cpp, verify_emb.cpp
├── Makefile, run_experiments.sh, plot_results.py
├── results/  (size_sweep.csv, prefetch_distance.csv, hint.csv, simd_width.csv, perf.csv)
└── plots/
```
