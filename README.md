# Architecture-Aware Optimization: Matmul, NN Training & Embedding Lookup

Hand-tuned C/C++ kernels showing how much wall-clock performance a naive implementation
leaves on the table, and how loop unrolling/reordering, cache tiling, SIMD
(SSE/AVX2/AVX-512), and software prefetching each recover it — measured on real hardware
with `perf`, not simulated.

Every variant is compiled with `-O0` and ISA feature flags only, so all reported speedups
come from the algorithm and memory-access pattern rather than the compiler.

Assignment spec: [`CS683 PA1 2025.pdf`](CS683%20PA1%202025.pdf)

## Parts

- **[`part1/`](part1/README.md)** — *The Matrix.* Matrix multiplication, standalone and
  inside a neural network's training loop. Loop reordering, loop unrolling, tiling across
  four tile sizes, SIMD at three register widths, and a combined tiling + register-blocked
  SIMD variant.
- **[`part2/`](part2/README.md)** — *Embed it.* Sum-pooled embedding-table lookup (the
  gather-and-reduce pattern behind recommendation-model embedding bags). Software
  prefetching with tunable distance and cache fill level, SIMD at three widths, and both
  combined.

## Headline results — Intel i9-11900H (Tiger Lake, AVX-512), single-threaded

| Workload | Naive | Best variant | Speedup |
|---|---:|---:|---:|
| Matrix multiplication (N=2048) | 62 871 ms | 4 091 ms — tiling + SIMD 512-bit | **15.4×** |
| Neural-net training epoch (512², batch 512) | 8 823 ms | 1 305 ms — tiling + SIMD | **6.8×** |
| Embedding lookup (200K×128, 2048 lookups) | 721 µs | 228 µs — prefetch + SIMD 512-bit | **3.2×** |

Two results worth calling out:

- **Register width dominates for matmul.** Retired instructions at N=2048 fall
  396.8 B → 237.5 B → 119.3 B → 63.5 B going scalar → 128 → 256 → 512-bit, and wall-clock
  tracks it. Combining tiling with SIMD at matched width beats either alone (15.4× vs
  9.95× for SIMD-512 by itself).
- **MPKI can move the wrong way.** Tiling cuts absolute L1-D misses 13% but cuts
  instructions 39%, so misses-per-kilo-instruction *rises*. See
  [part1](part1/README.md#1b--what-mpki-actually-shows-n2048) for why absolute misses are
  the honest metric here.

## Measurement notes

- Execution times are the median of 3 runs.
- Part 1 counters cover the whole process, which is sound because the kernel dominates:
  naive instruction counts scale as N³ across the whole range (7.88× / 7.95× / 7.98× for
  each doubling of N, against a theoretical 8×), so matrix setup is negligible.
- Part 2 counters are **gated to the kernel region** via perf's control FIFO. The
  embedding table is up to 1 GB and its `mt19937` fill costs ~86 billion instructions,
  which would otherwise swamp the ~8 million instructions of the kernel and make all four
  variants look identical.
- Correctness: every optimized variant is checked against the naive reference
  (`make verify` in each part).

## Reproducing

```bash
# Part 1 - matmul
cd part1/mat_mul && make && make verify && ./bin/verify
./run_experiments.sh && python3 plot_results.py     # ~25 min (N up to 2048)

# Part 1 - neural net
cd ../neural_net && make
./run_experiments.sh && python3 plot_results.py

# Part 2 - embedding
cd ../../part2 && make && make verify && ./bin/verify_emb
./run_experiments.sh && python3 plot_results.py && python3 make_tables.py   # ~10 min
```

## Layout

```
.
├── part1/
│   ├── mat_mul/       mat_mul.c (naive, reorder, unroll, tiling, SIMD x3, tiling+SIMD x3)
│   └── neural_net/    same techniques inside Layer::forward / backward
└── part2/
    ├── emb.cpp        naive / prefetch / SIMD / prefetch+SIMD embedding lookup
    └── results/       CSV sweeps + tables.md (Tables 2.1, 2.2, 2.3)
```

## Known gap

Task 2A deliverable 5 — toggling the **hardware prefetchers** via MSR `0x1a4` and
analysing the impact — is not covered; it requires root on the test machine.
