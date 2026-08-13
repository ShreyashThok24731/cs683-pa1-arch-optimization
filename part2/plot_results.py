#!/usr/bin/env python3
import csv, os
from collections import defaultdict
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "plots")
os.makedirs(P, exist_ok=True)

def load(name):
    return list(csv.DictReader(open(os.path.join(R, name))))

rows = load("size_sweep.csv")
by_size = defaultdict(dict)
for r in rows:
    by_size[int(r["table_size"])][r["run"]] = float(r["us"])
sizes = sorted(by_size.keys())

plt.figure(figsize=(9, 5))
plt.title("Speedup over naive vs embedding table size (dim=128, input=2048)")
for run, label in [("prefetch","SW prefetch"), ("simd","SIMD (AVX2 8x)"), ("prefetch_simd","Prefetch+SIMD")]:
    ys = [by_size[s]["naive"] / by_size[s][run] for s in sizes]
    plt.plot(sizes, ys, marker="o", label=label)
plt.axhline(1.0, color="gray", linestyle=":", label="Naive")
plt.xscale("log")
plt.xlabel("Embedding table rows"); plt.ylabel("Speedup (x)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "speedup_vs_table_size.png"), dpi=140)
plt.close()

rows = load("prefetch_distance.csv")
by_run = defaultdict(dict)
for r in rows: by_run[r["run"]][int(r["pd"])] = float(r["us"])
pds = sorted(next(iter(by_run.values())).keys())
plt.figure(figsize=(8, 5))
plt.title("Execution time vs prefetch distance (table=2M, dim=128)")
for run in ["prefetch", "prefetch_simd"]:
    ys = [by_run[run][pd] for pd in pds]
    plt.plot(pds, ys, marker="o", label=run)
plt.xlabel("Prefetch distance (bags of embedding lookups ahead)")
plt.ylabel("Execution time (µs)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "prefetch_distance.png"), dpi=140)
plt.close()

rows = load("hint.csv")
labels = {0:"T0", 1:"T1", 2:"T2", 3:"NTA"}
by = defaultdict(dict)
for r in rows: by[r["run"]][int(r["hint"])] = float(r["us"])
hints = sorted(next(iter(by.values())).keys())
plt.figure(figsize=(7, 5))
plt.title("Cache-fill hint vs execution time (table=2M, dim=128, pd=4)")
xs = list(range(len(hints)))
w = 0.35
for i, run in enumerate(["prefetch", "prefetch_simd"]):
    ys = [by[run][h] for h in hints]
    plt.bar([x + i*w for x in xs], ys, w, label=run)
plt.xticks([x + w/2 for x in xs], [labels[h] for h in hints])
plt.ylabel("µs"); plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "hint_sweep.png"), dpi=140)
plt.close()

rows = load("simd_width.csv")
by = defaultdict(lambda: defaultdict(dict))
for r in rows:
    by[int(r["dim"])][int(r["simd_width"])][r["run"]] = float(r["us"])
dims = sorted(by.keys())
widths = sorted(next(iter(by.values())).keys())

plt.figure(figsize=(9, 5))
plt.title("Speedup vs SIMD width for different embedding dimensions")
xs = list(range(len(dims)))
w = 0.25
for i, W in enumerate(widths):
    ys = [by[d][W]["naive"]/by[d][W]["prefetch_simd"] for d in dims]
    plt.bar([x + i*w for x in xs], ys, w, label=f"prefetch+SIMD {W}b")
plt.xticks([x + w for x in xs], [str(d) for d in dims])
plt.xlabel("Embedding dimension"); plt.ylabel("Speedup over naive (x)")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "simd_width_vs_dim.png"), dpi=140)
plt.close()

plt.figure(figsize=(9, 5))
plt.title("Optimization comparison across (table size, dim)")
combos = [(200000,64), (1000000,128), (2000000,256), (4000000,128)]
labels_x = [f"({s//1000}K,{d})" for (s,d) in combos]
xs = list(range(len(combos)))
w = 0.28
prefetch = []
simd     = []
combined = []
sz_rows = load("size_sweep.csv")
sw_rows = load("simd_width.csv")
def get_size(sz, run):
    for r in sz_rows:
        if int(r["table_size"])==sz and int(r["dim"])==128 and r["run"]==run:
            return float(r["us"])
    return None
def get_sw(dim, w, run):
    for r in sw_rows:
        if int(r["dim"])==dim and int(r["simd_width"])==w and r["run"]==run:
            return float(r["us"])
    return None
for (s, d) in combos:
    if d == 128:
        n  = get_size(s, "naive")
        p  = get_size(s, "prefetch")
        v  = get_size(s, "simd")
        pv = get_size(s, "prefetch_simd")
    else:
        n  = get_sw(d, 256, "naive")
        p  = None
        v  = get_sw(d, 256, "simd")
        pv = get_sw(d, 256, "prefetch_simd")
    prefetch.append(n/p if (n and p) else 0)
    simd.append(n/v if (n and v) else 0)
    combined.append(n/pv if (n and pv) else 0)
plt.bar([x - w for x in xs], prefetch, w, label="Prefetch")
plt.bar(xs,                  simd,     w, label="SIMD")
plt.bar([x + w for x in xs], combined, w, label="Prefetch+SIMD")
plt.xticks(xs, labels_x)
plt.xlabel("(table rows, embedding dim)"); plt.ylabel("Normalized speedup")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "combined_comparison.png"), dpi=140)
plt.close()

print("wrote plots to", P)
