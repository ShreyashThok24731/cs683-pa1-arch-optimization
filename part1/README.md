# Part 1 — Matrix Multiplication & Neural Network Training

Cache- and SIMD-aware optimization of dense matrix multiplication, applied both as a
standalone kernel (`mat_mul/`) and inside a small feed-forward neural network's
forward/backward passes (`neural_net/`).

Every variant below is built with `-O0` and only ISA feature flags (`-mavx2 -mfma
-msse2`) — no compiler auto-vectorization or loop optimization, so the speedups come
entirely from the hand-written code.

## Optimization variants

| Variant | Technique |
|---|---|
| Naive | Textbook `i-j-k` triple loop |
| Loop reorder + unroll | `i-j(4)-k(4)` order, C accumulators hoisted into registers for the whole `k` loop |
| Tiling | Blocked `i-j-k` over cache-sized tiles, tile-boundary clamped for non-multiple sizes |
| SIMD | AVX2 + FMA (`_mm256_fmadd_pd`), `i-k-j` order for contiguous B/C access |
| Combined | Tiling + SIMD + 8-wide register blocking (2× `__m256d` accumulators, 4-way unrolled `k`) |

## `mat_mul/` — standalone kernel

```bash
cd mat_mul
make all                                   # builds bin/{naive,loop,simd}
for T in 16 32 64 128; do make tiling TILE_SIZE=$T; make combination TILE_SIZE=$T; done

./bin/naive 1024                           # run any variant: <binary> <matrix_dim>
./bin/combination_32 1024

g++ -O0 -mavx2 -mfma -msse2 verify.cpp -o bin/verify && ./bin/verify   # correctness vs naive

./run_experiments.sh                       # sweeps sizes/tiles, writes results/*.csv (~2 min)
python3 plot_results.py                    # renders plots/*.png
```

## `neural_net/` — matmul inside a training loop

A minimal 2-layer sigmoid network (`Matrix`, `Layer`, `NeuralNetwork`) whose forward
and backward passes route every matrix multiply through the same five variants above,
selected per-layer via `MatrixOptimization`.

```bash
cd neural_net
make
./bin/main                                 # prints matmul + NN training benchmarks

./run_experiments.sh                       # parses bin/main output into results/*.csv
python3 plot_results.py                    # renders plots/*.png
```

## Results (Intel i9-11900H, single-threaded, `-O0`)

Matrix multiplication, wall-clock:

| N    | Naive (ms) | Loop | SIMD | Tiling (T=32) | Combined (T=32) | Combined speedup |
|------|-----------:|-----:|-----:|--------------:|-----------------:|------------------:|
| 128  |   6        | 2    | 2    | 4              |  1               | 6.0×               |
| 256  |  56        | 21   | 19   | 37             | 11               | 5.1×               |
| 512  | 557        | 185  | 144  | 315            | 93               | 6.0×               |
| 1024 | 5543       | 1802 | 1218 | 2316           | **790**          | **7.0×**           |

Neural network training (512×512 hidden layer, batch 512, 1 epoch):

| Variant | Time (ms) | Speedup |
|---|---:|---:|
| Basic | 9781 | 1.0× |
| Reordered | 6143 | 1.6× |
| Unrolled | 3885 | 2.5× |
| Tiled | 6485 | 1.5× |
| **Vectorized (tile + SIMD)** | **1411** | **6.9×** |

Combining tiling with register-blocked SIMD consistently outperforms either technique
alone: tiling improves data reuse but the scalar inner loop is instruction-bound,
while pure SIMD without tiling still thrashes the cache at larger `N`.

## Layout

```
part1/
├── mat_mul/
│   ├── mat_mul.c, helper.h, verify.cpp
│   ├── Makefile, run_experiments.sh, plot_results.py
│   ├── results/  (exec_time.csv, perf.csv, run.log)
│   └── plots/
└── neural_net/
    ├── src/{main,nn,matrix_operation}.cpp
    ├── include/*.h
    ├── Makefile, run_experiments.sh, plot_results.py
    ├── results/
    └── plots/
```
