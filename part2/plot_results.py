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

HINT_LABELS = {0: "T0 (L1)", 1: "T1 (L2)", 2: "T2 (LLC)", 3: "NTA"}

# ---------- 1) speedup vs embedding table size ----------
rows = load("size_sweep.csv")
by_size = defaultdict(dict)
for r in rows:
    by_size[int(r["table_size"])][r["run"]] = float(r["us"])
sizes = sorted(by_size)

plt.figure(figsize=(9, 5))
plt.title("Speedup over naive vs embedding table size (dim=128, input=2048)")
for run, label in [("prefetch", "SW prefetch"), ("simd", "SIMD (AVX2 8x)"), ("prefetch_simd", "Prefetch + SIMD")]:
    plt.plot(sizes, [by_size[s]["naive"] / by_size[s][run] for s in sizes], marker="o", label=label)
plt.axhline(1.0, color="gray", linestyle=":", label="Naive")
plt.xscale("log")
plt.xlabel("Embedding table rows"); plt.ylabel("Speedup (x)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "speedup_vs_table_size.png"), dpi=140); plt.close()

# ---------- 2) prefetch distance ----------
rows = load("prefetch_distance.csv")
by_run = defaultdict(dict)
for r in rows:
    by_run[r["run"]][int(r["pd"])] = float(r["us"])
pds = sorted(next(iter(by_run.values())))
plt.figure(figsize=(8, 5))
plt.title("Execution time vs prefetch distance (table=2M, dim=128)")
for run in ["prefetch", "prefetch_simd"]:
    plt.plot(pds, [by_run[run][pd] for pd in pds], marker="o", label=run)
plt.xscale("log", base=2)
plt.xlabel("Prefetch distance (lookups ahead)"); plt.ylabel("Execution time (us)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "prefetch_distance.png"), dpi=140); plt.close()

# ---------- 3) cache fill level ----------
rows = load("hint.csv")
by = defaultdict(dict)
for r in rows:
    by[r["run"]][int(r["hint"])] = float(r["us"])
hints = sorted(next(iter(by.values())))
plt.figure(figsize=(7, 5))
plt.title("Cache fill level vs execution time (table=2M, dim=128, pd=4)")
xs = list(range(len(hints)))
w = 0.35
for i, run in enumerate(["prefetch", "prefetch_simd"]):
    plt.bar([x + i * w for x in xs], [by[run][h] for h in hints], w, label=run)
plt.xticks([x + w / 2 for x in xs], [HINT_LABELS[h] for h in hints])
plt.ylabel("Execution time (us)"); plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "hint_sweep.png"), dpi=140); plt.close()

# ---------- 4) SIMD width x embedding dimension ----------
rows = load("simd_width.csv")
by = defaultdict(lambda: defaultdict(dict))
for r in rows:
    by[int(r["dim"])][int(r["simd_width"])][r["run"]] = float(r["us"])
dims = sorted(by)
widths = sorted(next(iter(by.values())))

plt.figure(figsize=(9, 5))
plt.title("Speedup vs SIMD width for different embedding dimensions")
xs = list(range(len(dims)))
w = 0.25
for i, W in enumerate(widths):
    plt.bar([x + i * w for x in xs], [by[d][W]["naive"] / by[d][W]["prefetch_simd"] for d in dims], w,
            label=f"Prefetch+SIMD {W}-bit")
plt.xticks([x + w for x in xs], [str(d) for d in dims])
plt.axhline(1.0, color="gray", linestyle=":")
plt.xlabel("Embedding dimension"); plt.ylabel("Speedup over naive (x)")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "simd_width_vs_dim.png"), dpi=140); plt.close()

# ---------- 5) Figure 2.1: technique comparison ----------
plt.figure(figsize=(9, 5))
plt.title("Optimization comparison across embedding table sizes (dim=128)")
xs = list(range(len(sizes)))
w = 0.28
for i, (run, label) in enumerate([("prefetch", "Software prefetching"), ("simd", "SIMD"),
                                  ("prefetch_simd", "Software prefetching + SIMD")]):
    plt.bar([x + (i - 1) * w for x in xs], [by_size[s]["naive"] / by_size[s][run] for s in sizes], w, label=label)
plt.xticks(xs, [f"{s // 1000}K" for s in sizes])
plt.axhline(1.0, color="gray", linestyle=":")
plt.xlabel("Embedding table rows"); plt.ylabel("Normalized speedup (no optimization = 1)")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "combined_comparison.png"), dpi=140); plt.close()

# ---------- 6) cache misses across the hierarchy (kernel-gated counters) ----------
perf = load("perf.csv")

def prow(sweep, run, **kw):
    for r in perf:
        if r["sweep"] == sweep and r["run"] == run and all(r[k] == str(v) for k, v in kw.items()):
            return r
    return None

def as_num(r, field):
    try:
        return float(r[field])
    except (TypeError, ValueError):
        return 0.0

ts_perf = sorted({int(r["table_size"]) for r in perf if r["sweep"] == "table_size"})
runs = [("naive", "Naive"), ("prefetch", "Prefetch"), ("simd", "SIMD"), ("prefetch_simd", "Prefetch+SIMD")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (field, title) in zip(axes, [("L1D_misses", "L1-D misses"),
                                     ("L2_misses", "L2 misses"),
                                     ("LLC_misses", "LLC misses")]):
    xs = list(range(len(ts_perf)))
    w = 0.2
    for i, (run, label) in enumerate(runs):
        vals = [as_num(prow("table_size", run, table_size=s), field) for s in ts_perf]
        ax.bar([x + i * w for x in xs], vals, w, label=label)
    ax.set_xticks([x + 1.5 * w for x in xs]); ax.set_xticklabels([f"{s // 1000}K" for s in ts_perf])
    ax.set_title(title); ax.set_xlabel("Embedding table rows"); ax.set_ylabel("Misses")
    ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(P, "cache_misses.png"), dpi=140); plt.close()

# ---------- 7) instruction count vs SIMD width and embedding dimension ----------
dims_perf = sorted({int(r["dim"]) for r in perf if r["sweep"] == "dim_width"})
plt.figure(figsize=(9, 5))
plt.title("Retired instructions vs embedding dimension, by SIMD width")
base = [as_num(prow("dim_width", "naive", dim=d), "instructions") for d in dims_perf]
plt.plot(dims_perf, base, marker="o", color="k", label="No SIMD (naive)")
for W in [128, 256, 512]:
    vals = [as_num(prow("dim_width", "simd", dim=d, simd_width=W), "instructions") for d in dims_perf]
    plt.plot(dims_perf, vals, marker="s", label=f"SIMD {W}-bit")
plt.xscale("log", base=2); plt.yscale("log")
plt.xlabel("Embedding dimension"); plt.ylabel("Instructions (kernel only)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "instruction_count.png"), dpi=140); plt.close()

print("wrote plots to", P)
