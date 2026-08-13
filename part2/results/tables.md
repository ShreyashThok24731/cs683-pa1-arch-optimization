# Task 2 result tables

Counters are gated to the kernel region with `perf --control fifo`, so embedding-table
initialisation is excluded. Execution time is measured inside the kernel.

## Table 2.1 - Software prefetching

### Varying embedding table size (dim=128, pd=4, hint=T0)

| Metric | 200K rows | 1000K rows | 2000K rows |
|---|---|---|---|
| **No software prefetching** |  |  |  |
| &nbsp;&nbsp;L1D misses | 27,893 | 28,464 | 28,932 |
| &nbsp;&nbsp;L2 misses | 3,353 | 5,160 | 5,206 |
| &nbsp;&nbsp;LLC misses | 2,289 | 3,288 | 3,130 |
| &nbsp;&nbsp;SW prefetch requests | 0 | 0 | 0 |
| &nbsp;&nbsp;Execution time (us) | 709 | 801 | 805 |
| **Software prefetching** |  |  |  |
| &nbsp;&nbsp;L1D misses | 25,979 | 26,459 | 26,972 |
| &nbsp;&nbsp;L2 misses | 2,890 | 3,371 | 3,442 |
| &nbsp;&nbsp;LLC misses | 432 | 649 | 1,600 |
| &nbsp;&nbsp;SW prefetch requests | 16,339 | 16,020 | 16,100 |
| &nbsp;&nbsp;Execution time (us) | 582 | 591 | 663 |
| **Speedup** (norm. to no SW prefetch) | 1.22x | 1.36x | 1.21x |

### Varying prefetch distance (table=1M, dim=128, hint=T0)

| Metric (software prefetching) | pd=1 | pd=4 | pd=16 | pd=64 |
|---|---|---|---|---|
| L1D misses | 26,438 | 25,747 | 27,392 | 32,759 |
| L2 misses | 2,658 | 2,411 | 3,406 | 4,464 |
| LLC misses | 1,071 | 1,104 | 1,161 | 1,814 |
| SW prefetch requests | 16,806 | 16,063 | 14,099 | 6,196 |
| Execution time (us) | 633 | 624 | 638 | 710 |
| **Speedup** vs naive | 1.27x | 1.28x | 1.26x | 1.13x |

### Varying cache fill level (table=1M, dim=128, pd=4)

| Metric (software prefetching) | T0 (L1) | T1 (L2) | T2 (LLC) | NTA |
|---|---|---|---|---|
| L1D misses | 27,098 | 28,113 | 28,029 | 26,175 |
| L2 misses | 3,610 | 3,221 | 3,001 | 4,491 |
| LLC misses | 823 | 607 | 456 | 1,087 |
| SW prefetch requests | 16,129 | 16,431 | 16,164 | 16,175 |
| Execution time (us) | 666 | 600 | 573 | 612 |
| **Speedup** vs naive | 1.20x | 1.33x | 1.40x | 1.31x |

## Table 2.2 - SIMD

Table = 200K rows, input = 2048 lookups.

| Metric | dim=64 | dim=128 | dim=256 | dim=512 |
|---|---|---|---|---|
| **No SIMD** |  |  |  |  |
| &nbsp;&nbsp;Instructions | 4,283,723 | 8,360,872 | 16,509,800 | 32,851,548 |
| &nbsp;&nbsp;Execution time (us) | 470 | 721 | 1,170 | 2,088 |
| **SIMD 128-bit** |  |  |  |  |
| &nbsp;&nbsp;Instructions | 2,301,475 | 4,280,889 | 8,220,184 | 16,141,376 |
| &nbsp;&nbsp;Execution time (us) | 359 | 615 | 820 | 1,394 |
| &nbsp;&nbsp;**Speedup** (norm. to no SIMD) | 1.31x | 1.17x | 1.43x | 1.50x |
| **SIMD 256-bit** |  |  |  |  |
| &nbsp;&nbsp;Instructions | 1,329,611 | 2,320,224 | 4,315,466 | 8,307,918 |
| &nbsp;&nbsp;Execution time (us) | 382 | 419 | 704 | 867 |
| &nbsp;&nbsp;**Speedup** (norm. to no SIMD) | 1.23x | 1.72x | 1.66x | 2.41x |
| **SIMD 512-bit** |  |  |  |  |
| &nbsp;&nbsp;Instructions | 825,465 | 1,333,085 | 2,364,188 | 4,377,979 |
| &nbsp;&nbsp;Execution time (us) | 195 | 344 | 444 | 625 |
| &nbsp;&nbsp;**Speedup** (norm. to no SIMD) | 2.41x | 2.10x | 2.64x | 3.34x |

## Table 2.3 - Software prefetching + SIMD (Task 2C)

| Metric | dim=64 | dim=128 | dim=256 | dim=512 |
|---|---|---|---|---|
| **Prefetch + SIMD 128-bit** |  |  |  |  |
| &nbsp;&nbsp;L1D misses | 18,979 | 27,306 | 41,989 | 82,715 |
| &nbsp;&nbsp;L2 misses | 2,372 | 1,550 | 1,838 | 3,447 |
| &nbsp;&nbsp;LLC misses | 157 | 522 | 477 | 727 |
| &nbsp;&nbsp;SW prefetch requests | 7,883 | 15,876 | 32,697 | 66,675 |
| &nbsp;&nbsp;Execution time (us) | 204 | 351 | 786 | 1,348 |
| &nbsp;&nbsp;Instructions | 2,570,930 | 4,802,058 | 9,300,715 | 18,282,602 |
| &nbsp;&nbsp;**Speedup** (norm. to no optimization) | 2.30x | 2.05x | 1.49x | 1.55x |
| **Prefetch + SIMD 256-bit** |  |  |  |  |
| &nbsp;&nbsp;L1D misses | 16,734 | 26,830 | 43,903 | 82,697 |
| &nbsp;&nbsp;L2 misses | 2,308 | 2,946 | 3,095 | 3,496 |
| &nbsp;&nbsp;LLC misses | 299 | 402 | 373 | 790 |
| &nbsp;&nbsp;SW prefetch requests | 7,898 | 15,805 | 31,564 | 65,444 |
| &nbsp;&nbsp;Execution time (us) | 185 | 251 | 537 | 884 |
| &nbsp;&nbsp;Instructions | 1,586,528 | 2,869,537 | 5,386,851 | 10,449,314 |
| &nbsp;&nbsp;**Speedup** (norm. to no optimization) | 2.54x | 2.87x | 2.18x | 2.36x |
| **Prefetch + SIMD 512-bit** |  |  |  |  |
| &nbsp;&nbsp;L1D misses | 15,930 | 26,170 | 43,755 | 82,965 |
| &nbsp;&nbsp;L2 misses | 729 | 2,992 | 1,811 | 3,506 |
| &nbsp;&nbsp;LLC misses | 633 | 723 | 902 | 1,162 |
| &nbsp;&nbsp;SW prefetch requests | 7,945 | 15,854 | 32,004 | 65,385 |
| &nbsp;&nbsp;Execution time (us) | 180 | 228 | 452 | 730 |
| &nbsp;&nbsp;Instructions | 1,092,495 | 1,871,138 | 3,420,250 | 6,528,620 |
| &nbsp;&nbsp;**Speedup** (norm. to no optimization) | 2.61x | 3.16x | 2.59x | 2.86x |

