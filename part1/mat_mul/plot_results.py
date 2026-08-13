#!/usr/bin/env python3
import csv, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "plots")
os.makedirs(P, exist_ok=True)

exec_rows = list(csv.DictReader(open(os.path.join(R, "exec_time.csv"))))
perf_rows = list(csv.DictReader(open(os.path.join(R, "perf.csv"))))

def _lookup(rows, field, variant, size, tile=""):
    for r in rows:
        if r["variant"] == variant and int(r["size"]) == size and r.get("tile", "") == str(tile):
            try:
                return float(r[field])
            except (ValueError, KeyError):
                return None
    return None

def ms(variant, size, tile=""):      return _lookup(exec_rows, "ms", variant, size, tile)
def mpki(variant, size, tile=""):    return _lookup(perf_rows, "MPKI", variant, size, tile)
def instr(variant, size, tile=""):
    v = _lookup(perf_rows, "instructions", variant, size, tile)
    return v / 1e9 if v is not None else None
def misses(variant, size, tile=""):
    v = _lookup(perf_rows, "L1D_load_misses", variant, size, tile)
    return v / 1e9 if v is not None else None
def miss_rate(variant, size, tile=""):
    m = _lookup(perf_rows, "L1D_load_misses", variant, size, tile)
    l = _lookup(perf_rows, "L1D_loads", variant, size, tile)
    return 100 * m / l if (m is not None and l) else None

sizes = sorted({int(r["size"]) for r in exec_rows})
tiles = sorted({int(r["tile"]) for r in exec_rows if r["tile"]})
SIMD = [("simd128", "SIMD 128-bit"), ("simd256", "SIMD 256-bit"), ("simd512", "SIMD 512-bit")]
COMBO = [("combination128", "Tiling+SIMD 128-bit"), ("combination256", "Tiling+SIMD 256-bit"),
         ("combination512", "Tiling+SIMD 512-bit")]
WIDTHS = [128, 256, 512]

def best_over_tiles(prefix, n):
    vals = [speedup(prefix, n, T) for T in tiles]
    vals = [v for v in vals if v]
    return max(vals) if vals else 0.0

def speedup(variant, size, tile=""):
    base, opt = ms("naive", size), ms(variant, size, tile)
    return base / opt if (base and opt) else None

# Task 1B: speedup vs matrix size for different tile sizes
plt.figure(figsize=(9, 5))
plt.title("Speedup over naive vs matrix size")
for T in tiles:
    plt.plot(sizes, [speedup("tiling", n, T) for n in sizes], marker="s", linestyle="--", label=f"Tiling T={T}")
    plt.plot(sizes, [speedup("combination512", n, T) for n in sizes], marker="^", label=f"Tiling+SIMD(512) T={T}")
plt.xlabel("Matrix size N"); plt.ylabel("Speedup over naive (x)")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(P, "speedup_vs_size.png"), dpi=140); plt.close()

# Task 1B: L1-D MPKI vs matrix size for different tile sizes
plt.figure(figsize=(9, 5))
plt.title("L1-D MPKI vs matrix size")
plt.plot(sizes, [mpki("naive", n) for n in sizes], marker="o", color="k", label="Naive")
for T in tiles:
    plt.plot(sizes, [mpki("tiling", n, T) for n in sizes], marker="s", linestyle="--", label=f"Tiling T={T}")
plt.xlabel("Matrix size N"); plt.ylabel("L1-D misses per 1000 instructions")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(P, "mpki_vs_size.png"), dpi=140); plt.close()

# Task 1B: tile-size sweep
plt.figure(figsize=(8, 5))
plt.title(f"Effect of tile size on execution time (N={sizes[-1]})")
for var, label in [("tiling", "Tiling"), ("combination512", "Tiling+SIMD (512-bit)")]:
    plt.plot(tiles, [ms(var, sizes[-1], T) for T in tiles], marker="o", label=label)
plt.xlabel("Tile size"); plt.ylabel("Execution time (ms)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "tile_size_sweep.png"), dpi=140); plt.close()

# Task 1C: speedup vs matrix size for each SIMD register width
plt.figure(figsize=(9, 5))
plt.title("SIMD speedup over naive vs matrix size, by register width")
for var, label in SIMD:
    plt.plot(sizes, [speedup(var, n) for n in sizes], marker="o", label=label)
plt.axhline(1.0, color="gray", linestyle=":")
plt.xlabel("Matrix size N"); plt.ylabel("Speedup over naive (x)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "simd_width_speedup.png"), dpi=140); plt.close()

# Task 1C: instruction count, naive vs each SIMD width
plt.figure(figsize=(9, 5))
plt.title("Retired instructions vs matrix size (naive vs SIMD widths)")
plt.plot(sizes, [instr("naive", n) for n in sizes], marker="o", color="k", label="Naive")
for var, label in SIMD:
    plt.plot(sizes, [instr(var, n) for n in sizes], marker="s", label=label)
plt.xlabel("Matrix size N"); plt.ylabel("Instructions (billions)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "instruction_count.png"), dpi=140); plt.close()

# Task 1D (Figure 1.1): each technique as its own bar
plt.figure(figsize=(10, 5))
plt.title("Normalized speedup by optimization technique")
xs = list(range(len(sizes)))
best_simd = max(SIMD, key=lambda vl: sum(speedup(vl[0], n) or 0 for n in sizes))[0]
techniques = [
    ("Loop unrolling",  lambda n: speedup("unroll", n) or 0.0),
    ("Loop reordering", lambda n: speedup("reorder", n) or 0.0),
    ("Tiling",          lambda n: best_over_tiles("tiling", n)),
    ("SIMD",            lambda n: speedup(best_simd, n) or 0.0),
    ("Tiling + SIMD",   lambda n: max(best_over_tiles(c, n) or 0 for c, _ in COMBO)),
]
width = 0.16
for i, (label, fn) in enumerate(techniques):
    plt.bar([x + i * width for x in xs], [fn(n) for n in sizes], width, label=label)
plt.axhline(1.0, color="gray", linestyle=":")
plt.xticks([x + 2 * width for x in xs], [str(n) for n in sizes])
plt.xlabel("Matrix size N"); plt.ylabel("Normalized speedup (no optimization = 1)")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "task1d_bar.png"), dpi=140); plt.close()

# Supporting cache plots
plt.figure(figsize=(9, 5))
plt.title("Absolute L1-D load misses vs matrix size")
for variant, label in [("naive", "Naive"), ("reorder", "Reorder"), ("unroll", "Unroll"), ("simd256", "SIMD 256-bit")]:
    plt.plot(sizes, [misses(variant, n) for n in sizes], marker="o", label=label)
for T in tiles:
    plt.plot(sizes, [misses("tiling", n, T) for n in sizes], marker="s", linestyle="--", label=f"Tiling T={T}")
plt.xlabel("Matrix size N"); plt.ylabel("L1-D load misses (billions)")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(P, "absolute_misses.png"), dpi=140); plt.close()

plt.figure(figsize=(9, 5))
plt.title("L1-D miss rate vs matrix size")
for variant, label in [("naive", "Naive"), ("reorder", "Reorder"), ("unroll", "Unroll"), ("simd256", "SIMD 256-bit")]:
    plt.plot(sizes, [miss_rate(variant, n) for n in sizes], marker="o", label=label)
for T in tiles:
    plt.plot(sizes, [miss_rate("tiling", n, T) for n in sizes], marker="s", linestyle="--", label=f"Tiling T={T}")
plt.xlabel("Matrix size N"); plt.ylabel("L1-D miss rate (%)")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(P, "miss_rate.png"), dpi=140); plt.close()

# Task 1D: SIMD alone vs tiling+SIMD at matched register widths
plt.figure(figsize=(10, 5))
plt.title("SIMD alone vs Tiling+SIMD at matched register width")
xs = list(range(len(sizes)))
w = 0.13
for i, W in enumerate(WIDTHS):
    plt.bar([x + i * w for x in xs], [speedup(f"simd{W}", n) or 0.0 for n in sizes], w, label=f"SIMD {W}-bit")
for i, W in enumerate(WIDTHS):
    plt.bar([x + (i + 3.3) * w for x in xs], [best_over_tiles(f"combination{W}", n) for n in sizes], w,
            label=f"Tiling+SIMD {W}-bit")
plt.xticks([x + 3 * w for x in xs], [str(n) for n in sizes])
plt.axhline(1.0, color="gray", linestyle=":")
plt.xlabel("Matrix size N"); plt.ylabel("Speedup over naive (x)")
plt.grid(alpha=0.3, axis="y"); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(P, "simd_vs_tiled_simd.png"), dpi=140); plt.close()

print("wrote plots to", P)
