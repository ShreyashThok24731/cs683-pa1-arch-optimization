# Part 1 — The Matrix (Matrix Multiplication & Neural Network)

Cache- and SIMD-aware optimization of dense matrix multiplication, applied both as a
standalone kernel (`mat_mul/`) and inside a small feed-forward neural network's
forward/backward passes (`neural_net/`).

Everything is compiled with `-O0` and ISA feature flags only (`-mavx2 -mfma -msse2`,
plus `-mavx512f` for the 512-bit variants), so every speedup comes from the hand-written
code rather than the compiler.

| Task | Covered by |
|---|---|
| 1A Unroll Baba Unroll | `loop_reorder`, `loop_unroll`, and combined `loop_opt` variants |
| 1B Divide Karo, Rule Karo | tiling over 4 tile sizes, L1-D MPKI per tile size, MPKI/speedup vs matrix size |
| 1C Data Ko Line Mein Lagao | SIMD at 128 / 256 / 512-bit, instruction counts, speedup vs width |
| 1D Rancho's Final Year Project | all five techniques compared as separate bars, matched-width tiling+SIMD |
| 1E Confusion hi confusion hai | `neural_net/` with the same techniques in `MatrixOperation` |

## Test machine

Intel Core i9-11900H (Tiger Lake, 8 cores / 16 threads), AVX-512 capable.

| Level | Size | Ways | Line | Sets |
|---|---|---:|---:|---:|
| **L1-D (per core)** | **48 KiB** | 12 | 64 B | 64 |
| L1-I (per core) | 32 KiB | 8 | 64 B | 64 |
| L2 (per core) | 1.25 MiB | 20 | 64 B | 1024 |
| L3 (shared) | 24 MiB | 12 | 64 B | 32768 |

The 48 KiB / 12-way / 64-set L1-D geometry is what drives the tiling results below.

## `mat_mul/` — standalone kernel

### Variants

| Variant | Binary | Technique |
|---|---|---|
| Naive | `naive` | Textbook `i-j-k` triple loop |
| Loop reordering | `reorder` | `i-k-j` order so the inner loop walks B and C contiguously |
| Loop unrolling | `unroll` | Base `i-j-k` order, `k` unrolled 4× with the accumulator in a register |
| Both (1A) | `loop` | `i-j(4)-k(4)`, C block held in 4 registers across the whole `k` loop |
| SIMD | `simd_128/256/512` | `_mm_/_mm256_/_mm512_fmadd_pd`, `i-k-j` order |
| Tiling | `tiling_<T>` | Cache blocking, T ∈ {16, 32, 64, 128}, tile edges clamped |
| Tiling + SIMD | `combination<W>_<T>` | Tiling + register-blocked SIMD at width W, `k` unrolled 4× |

### Build & run

```bash
make                      # builds every variant (all widths x all tile sizes)
make simd512              # or build one: bin/simd_512
make combination TILE_SIZE=32 SIMD_WIDTH=512

./bin/naive 1024          # <binary> <matrix_dimension>
./bin/combination512_32 1024

make verify && ./bin/verify    # every variant vs naive, all sizes and tile sizes

./run_experiments.sh      # sweeps sizes/tiles/widths -> results/*.csv
python3 plot_results.py   # -> plots/*.png
```

`run_experiments.sh` uses N ∈ {256, 512, 1024, 2048}. Smaller sizes are excluded on
purpose: the base code times in whole milliseconds, and at N=128 the fastest variants
finish in 0 ms, which makes the speedup undefined.

### Results — speedup over naive

Naive baseline: 59 ms / 552 ms / 4625 ms / 62871 ms at N = 256 / 512 / 1024 / 2048.

| Technique | N=256 | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|---:|
| Loop unrolling | 1.74× | 1.76× | 1.79× | 1.77× |
| Loop reordering | 1.74× | 2.08× | 2.14× | 2.90× |
| Unroll + reorder (1A) | 2.95× | 3.00× | 2.73× | 3.19× |
| Tiling (best T) | 1.90× | 2.05× | 2.16× | 3.75× |
| SIMD 128-bit | 1.55× | 1.84× | 1.91× | 3.25× |
| SIMD 256-bit | 2.95× | 3.83× | 4.02× | 6.54× |
| SIMD 512-bit | 5.90× | 6.73× | 7.01× | 9.95× |
| Tiling + SIMD 128 | 2.36× | 2.60× | 2.63× | 4.55× |
| Tiling + SIMD 256 | 5.90× | 6.20× | 6.12× | 10.08× |
| **Tiling + SIMD 512** | **8.43×** | **8.62×** | **9.00×** | **15.37×** |

### 1C — instruction count vs SIMD width (N=2048)

| Naive | 128-bit | 256-bit | 512-bit |
|---:|---:|---:|---:|
| 396.8 B | 237.5 B | 119.3 B | 63.5 B |

Each doubling of the register width roughly halves the retired instruction count, and
the wall-clock speedup tracks it. `_mm*_fmadd_pd` was chosen so the multiply and add
issue as one instruction, and `i-k-j` order so the innermost loop loads B and C
contiguously and only `A[i][k]` needs a broadcast.

### 1B — what MPKI actually shows (N=2048)

| Variant | Instructions | L1-D misses | Miss rate | MPKI |
|---|---:|---:|---:|---:|
| Naive | 396.8 B | 10.01 B | 5.28% | 25.23 |
| Loop reordering | 310.5 B | **1.09 B** | 0.70% | **3.51** |
| Loop unrolling | 229.1 B | 9.55 B | 9.42% | 41.70 |
| Tiling T=128 | 243.9 B | 8.71 B | 7.15% | 35.71 |

Tiling lowers absolute L1-D misses (10.01 B → 8.71 B, −13%) but lowers the instruction
count much further (396.8 B → 243.9 B, −39%), so **MPKI — a ratio — goes up even though
the cache is doing less work**. Judging tiling by MPKI alone would be misleading here;
absolute misses and miss rate tell the real story.

Loop reordering is the dominant cache optimization for this kernel, cutting misses by
89%. The reason is the access pattern: this tiled version blocks the base `i-j-k` order,
so the innermost loop still walks `B[kk*size+jj]` down a column with stride `size`.
Tiling bounds the working set but does not make that access contiguous; reordering does.
That is also why tiling pays off most when combined with SIMD, which requires contiguous
loads — Tiling+SIMD-512 is the best variant at every matrix size.

Across tile sizes at N=2048 the MPKI is nearly flat (34.19 / 35.94 / 35.76 / 35.71 for
T = 16 / 32 / 64 / 128) for the same reason. At smaller N the tile size matters much
more: with a 48 KiB, 12-way, 64-set L1-D, a stride of `size` doubles maps successive B
rows onto few sets, so some (N, T) pairs thrash on conflict misses while others do not.

## `neural_net/` — matmul inside a training loop

A 2-layer sigmoid network whose forward and backward passes route every matrix multiply
through the same techniques, selected per layer via `MatrixOptimization`. Only
`MatrixOperation`'s multiply and transpose functions are modified, as the task requires.

```bash
cd neural_net
make && ./bin/main
./run_experiments.sh && python3 plot_results.py
```

### Results — 512×512 hidden layer, batch 512, 1 epoch

| Variant | Single 512² matmul (ms) | NN training (ms) | Training speedup |
|---|---:|---:|---:|
| Basic | 1463 | 8823 | 1.0× |
| Reordered | 922 | 5613 | 1.57× |
| Unrolled | 562 | 3486 | 2.53× |
| Tiled | 967 | 5860 | 1.51× |
| **Vectorized (tiling + SIMD)** | **210** | **1305** | **6.76×** |

All variants converge to the same final loss as the basic implementation.
