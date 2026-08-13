# Architecture-Aware Optimization: Matmul, NN Training & Embedding Lookup

Hand-tuned C/C++ kernels showing how much wall-clock performance is left on the table
by a naive implementation, and how loop unrolling/reordering, cache tiling, SIMD
(SSE/AVX2/AVX-512), and software prefetching each recover it — measured on real
hardware with `perf`, not simulated.

All variants are compiled with `-O0` and only ISA feature flags, so every speedup
reported comes from the algorithm and memory-access pattern, not the compiler.

Assignment spec: [`CS683 PA1 2025.pdf`](CS683%20PA1%202025.pdf)

## Parts

- **[`part1/`](part1/README.md)** — Matrix multiplication, both standalone and inside
  a small neural network's training loop. Loop reorder/unroll, tiling, AVX2+FMA SIMD,
  and a combined tiling+SIMD+register-blocking variant.
- **[`part2/`](part2/README.md)** — Sum-pooled embedding table lookup (the
  gather-and-reduce pattern behind recommendation-model embedding bags). Software
  prefetching (configurable distance/hint) and SIMD (SSE/AVX2/AVX-512), standalone
  and combined.

## Headline results (Intel i9-11900H, Tiger Lake, AVX-512, single-threaded)

| Workload | Naive | Best combined | Speedup |
|---|---:|---:|---:|
| Matrix multiplication (N=1024) | 5543 ms | 790 ms (tiling + SIMD) | **7.0×** |
| Neural-net training epoch (512×512, batch 512) | 9781 ms | 1411 ms (tiling + SIMD) | **6.9×** |
| Embedding lookup (1M×128 table, 2048 lookups) | ~850 µs | ~230 µs (prefetch + AVX-512) | **3.7×** |

See each part's README for the full per-size tables, build/run instructions, and the
parameter sweeps behind `plots/`.

## Reproducing

```bash
# Part 1 — matmul
cd part1/mat_mul
make naive loop simd
for T in 16 32 64 128; do make tiling TILE_SIZE=$T; make combination TILE_SIZE=$T; done
g++ -O0 -mavx2 -mfma -msse2 verify.cpp -o bin/verify && ./bin/verify   # correctness
./run_experiments.sh                                                    # ~2 min
python3 plot_results.py

# Part 1 — neural net
cd ../neural_net
make
./run_experiments.sh
python3 plot_results.py

# Part 2 — embedding
cd ../../part2
make
g++ -O0 -mavx2 -mfma -msse2 -mavx512f verify_emb.cpp -o bin/verify_emb && ./bin/verify_emb
./run_experiments.sh                                                    # ~15 min at largest sizes
python3 plot_results.py
```

## Layout

```
.
├── part1/
│   ├── mat_mul/       matmul.c + 5 variants, verify.cpp, Makefile, results/, plots/
│   └── neural_net/    Matrix/Layer/NeuralNetwork, same 5 variants per layer
└── part2/
    ├── emb.cpp         naive / prefetch / SIMD / prefetch+SIMD embedding lookup
    ├── verify_emb.cpp
    └── results/, plots/
```
