#!/usr/bin/env python3
import csv, os
from collections import defaultdict
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "plots")
os.makedirs(P, exist_ok=True)

exec_rows = list(csv.DictReader(open(os.path.join(R, "exec_time.csv"))))
def ms(variant, size, tile=""):
    for r in exec_rows:
        if r["variant"] == variant and int(r["size"]) == size and r.get("tile","") == str(tile):
            return float(r["ms"])
    return None

sizes = sorted({int(r["size"]) for r in exec_rows})
tiles = sorted({int(r["tile"]) for r in exec_rows if r["tile"]})

perf_rows = list(csv.DictReader(open(os.path.join(R, "perf.csv"))))
def mpki(variant, size, tile=""):
    for r in perf_rows:
        if r["variant"] == variant and int(r["size"]) == size and r.get("tile","") == str(tile):
            try: return float(r["MPKI"])
            except: return None
    return None

plt.figure(figsize=(9, 5))
plt.title("Speedup over naive vs matrix size")
plt.xlabel("Matrix size N")
plt.ylabel("Speedup (x)")
for variant, label in [("loop","Loop reorder+unroll"), ("simd","SIMD (AVX2+FMA)")]:
    ys = [ms("naive", n) / ms(variant, n) for n in sizes]
    plt.plot(sizes, ys, marker="o", label=label)
for T in tiles:
    ys = [ms("naive", n) / ms("tiling", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="s", linestyle="--", label=f"Tiling T={T}")
    ys = [ms("naive", n) / ms("combination", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="^", label=f"Combined T={T}")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(P, "speedup_vs_size.png"), dpi=140)
plt.close()

plt.figure(figsize=(9, 5))
plt.title("L1-D MPKI vs matrix size")
plt.xlabel("Matrix size N")
plt.ylabel("L1-D misses per 1000 instructions")
for variant, label in [("naive","Naive"), ("loop","Loop"), ("simd","SIMD")]:
    ys = [mpki(variant, n) for n in sizes]
    plt.plot(sizes, ys, marker="o", label=label)
for T in tiles:
    ys = [mpki("tiling", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="s", linestyle="--", label=f"Tiling T={T}")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(P, "mpki_vs_size.png"), dpi=140)
plt.close()

plt.figure(figsize=(10, 5))
plt.title("Normalized speedup by optimization at different matrix sizes")
xs = list(range(len(sizes)))
variants = [
    ("Loop unroll+reorder", lambda n: ms("naive", n)/ms("loop", n)),
    ("SIMD",                lambda n: ms("naive", n)/ms("simd", n)),
    ("Tiling (best T)",     lambda n: ms("naive", n)/min(ms("tiling", n, T)      for T in tiles)),
    ("Combined (best T)",   lambda n: ms("naive", n)/min(ms("combination", n, T) for T in tiles)),
]
width = 0.2
for i, (label, fn) in enumerate(variants):
    ys = [fn(n) for n in sizes]
    plt.bar([x + i*width for x in xs], ys, width, label=label)
plt.xticks([x + 1.5*width for x in xs], [str(n) for n in sizes])
plt.xlabel("Matrix size N"); plt.ylabel("Speedup over naive (x)")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(P, "task1d_bar.png"), dpi=140)
plt.close()

plt.figure(figsize=(8, 5))
plt.title("Effect of tile size on execution time (N=1024)")
plt.xlabel("Tile size")
plt.ylabel("Execution time (ms)")
for var in ["tiling", "combination"]:
    ys = [ms(var, 1024, T) for T in tiles]
    plt.plot(tiles, ys, marker="o", label=var)
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(P, "tile_size_sweep.png"), dpi=140)
plt.close()

def misses(variant, size, tile=""):
    for r in perf_rows:
        if r["variant"] == variant and int(r["size"]) == size and r.get("tile","") == str(tile):
            try: return float(r["L1D_load_misses"]) / 1e9
            except: return None
    return None
plt.figure(figsize=(9, 5))
plt.title("Absolute L1-D load misses (billions) vs matrix size")
plt.xlabel("Matrix size N"); plt.ylabel("L1-D load misses (billions)")
for variant, label in [("naive","Naive"), ("loop","Loop"), ("simd","SIMD")]:
    ys = [misses(variant, n) for n in sizes]
    plt.plot(sizes, ys, marker="o", label=label)
for T in tiles:
    ys = [misses("tiling", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="s", linestyle="--", label=f"Tiling T={T}")
    ys = [misses("combination", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="^", label=f"Combined T={T}")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(P, "absolute_misses.png"), dpi=140)
plt.close()

def miss_rate(variant, size, tile=""):
    for r in perf_rows:
        if r["variant"] == variant and int(r["size"]) == size and r.get("tile","") == str(tile):
            try:
                loads = float(r["L1D_loads"])
                m = float(r["L1D_load_misses"])
                return 100 * m / loads if loads > 0 else None
            except: return None
    return None
plt.figure(figsize=(9, 5))
plt.title("L1-D miss rate (%) vs matrix size")
plt.xlabel("Matrix size N"); plt.ylabel("L1-D miss rate (%)")
for variant, label in [("naive","Naive"), ("loop","Loop"), ("simd","SIMD")]:
    ys = [miss_rate(variant, n) for n in sizes]
    plt.plot(sizes, ys, marker="o", label=label)
for T in tiles:
    ys = [miss_rate("tiling", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="s", linestyle="--", label=f"Tiling T={T}")
plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(P, "miss_rate.png"), dpi=140)
plt.close()

def instr(variant, size, tile=""):
    for r in perf_rows:
        if r["variant"] == variant and int(r["size"]) == size and r.get("tile","") == str(tile):
            try: return float(r["instructions"]) / 1e9
            except: return None
    return None
plt.figure(figsize=(9, 5))
plt.title("Retired instructions (billions) vs matrix size")
plt.xlabel("Matrix size N"); plt.ylabel("Instructions (billions)")
for variant, label in [("naive","Naive"), ("loop","Loop"), ("simd","SIMD")]:
    ys = [instr(variant, n) for n in sizes]
    plt.plot(sizes, ys, marker="o", label=label)
for T in [32]:
    ys = [instr("tiling", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="s", linestyle="--", label=f"Tiling T={T}")
    ys = [instr("combination", n, T) for n in sizes]
    plt.plot(sizes, ys, marker="^", label=f"Combined T={T}")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(P, "instruction_count.png"), dpi=140)
plt.close()

print("wrote plots to", P)
