#!/usr/bin/env python3
"""Emit Table 2.1 and Table 2.2 (assignment spec) from results/ as markdown."""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = os.path.join(R, "tables.md")

perf = list(csv.DictReader(open(os.path.join(R, "perf.csv"))))
HINT_LABELS = {"0": "T0 (L1)", "1": "T1 (L2)", "2": "T2 (LLC)", "3": "NTA"}

def prow(sweep, run, **kw):
    for r in perf:
        if r["sweep"] == sweep and r["run"] == run and all(r[k] == str(v) for k, v in kw.items()):
            return r
    return None

def num(r, field):
    if r is None:
        return None
    try:
        return float(r[field])
    except (TypeError, ValueError):
        return None

def fmt(v):
    if v is None:
        return "n/a"
    if v >= 1000:
        return f"{int(v):,}"
    return f"{v:g}"

def emit(f, header, rows):
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join("---" for _ in header) + "|\n")
    for row in rows:
        f.write("| " + " | ".join(row) + " |\n")
    f.write("\n")

METRICS = [("L1D_misses", "L1D misses"), ("L2_misses", "L2 misses"), ("LLC_misses", "LLC misses"),
           ("sw_prefetch", "SW prefetch requests"), ("us", "Execution time (us)")]

def block(f, columns, baseline_run, opt_run, col_label):
    """columns: list of (label, selector-kwargs, sweep)."""
    header = [col_label] + [c[0] for c in columns]
    rows = []
    for tag, runner in [("No software prefetching", baseline_run), ("Software prefetching", opt_run)]:
        rows.append([f"**{tag}**"] + [""] * len(columns))
        for field, mlabel in METRICS:
            cells = []
            for _, kw, sweep in columns:
                cells.append(fmt(num(prow(sweep, runner, **kw), field)))
            rows.append([f"&nbsp;&nbsp;{mlabel}"] + cells)
    spd = []
    for _, kw, sweep in columns:
        b = num(prow(sweep, baseline_run, **kw), "us")
        o = num(prow(sweep, opt_run, **kw), "us")
        spd.append(f"{b / o:.2f}x" if (b and o) else "n/a")
    rows.append(["**Speedup** (norm. to no SW prefetch)"] + spd)
    emit(f, header, rows)

with open(OUT, "w") as f:
    f.write("# Task 2 result tables\n\n")
    f.write("Counters are gated to the kernel region with `perf --control fifo`, so embedding-table\n"
            "initialisation is excluded. Execution time is measured inside the kernel.\n\n")

    # ---- Table 2.1: software prefetching ----
    f.write("## Table 2.1 - Software prefetching\n\n")

    f.write("### Varying embedding table size (dim=128, pd=4, hint=T0)\n\n")
    ts = sorted({int(r["table_size"]) for r in perf if r["sweep"] == "table_size"})
    block(f, [(f"{s // 1000}K rows", {"table_size": s}, "table_size") for s in ts], "naive", "prefetch", "Metric")

    f.write("### Varying prefetch distance (table=1M, dim=128, hint=T0)\n\n")
    pds = sorted({int(r["pd"]) for r in perf if r["sweep"] == "pd"})
    cols = [(f"pd={p}", {"pd": p}, "pd") for p in pds]
    header = ["Metric (software prefetching)"] + [c[0] for c in cols]
    rows = []
    for field, mlabel in METRICS:
        rows.append([mlabel] + [fmt(num(prow("pd", "prefetch", **kw), field)) for _, kw, _ in cols])
    base_us = num(prow("table_size", "naive", table_size=1000000), "us")
    rows.append(["**Speedup** vs naive"] +
                [f"{base_us / num(prow('pd', 'prefetch', **kw), 'us'):.2f}x"
                 if (base_us and num(prow('pd', 'prefetch', **kw), 'us')) else "n/a" for _, kw, _ in cols])
    emit(f, header, rows)

    f.write("### Varying cache fill level (table=1M, dim=128, pd=4)\n\n")
    hints = sorted({int(r["hint"]) for r in perf if r["sweep"] == "hint"})
    cols = [(HINT_LABELS[str(h)], {"hint": h}, "hint") for h in hints]
    header = ["Metric (software prefetching)"] + [c[0] for c in cols]
    rows = []
    for field, mlabel in METRICS:
        rows.append([mlabel] + [fmt(num(prow("hint", "prefetch", **kw), field)) for _, kw, _ in cols])
    rows.append(["**Speedup** vs naive"] +
                [f"{base_us / num(prow('hint', 'prefetch', **kw), 'us'):.2f}x"
                 if (base_us and num(prow('hint', 'prefetch', **kw), 'us')) else "n/a" for _, kw, _ in cols])
    emit(f, header, rows)

    # ---- Table 2.2: SIMD ----
    f.write("## Table 2.2 - SIMD\n\n")
    f.write("Table = 200K rows, input = 2048 lookups.\n\n")
    dims = sorted({int(r["dim"]) for r in perf if r["sweep"] == "dim_width"})
    widths = [128, 256, 512]
    header = ["Metric"] + [f"dim={d}" for d in dims]
    rows = [["**No SIMD**"] + [""] * len(dims)]
    for field, mlabel in [("instructions", "Instructions"), ("us", "Execution time (us)")]:
        rows.append([f"&nbsp;&nbsp;{mlabel}"] + [fmt(num(prow("dim_width", "naive", dim=d), field)) for d in dims])
    for W in widths:
        rows.append([f"**SIMD {W}-bit**"] + [""] * len(dims))
        for field, mlabel in [("instructions", "Instructions"), ("us", "Execution time (us)")]:
            rows.append([f"&nbsp;&nbsp;{mlabel}"] +
                        [fmt(num(prow("dim_width", "simd", dim=d, simd_width=W), field)) for d in dims])
        spd = []
        for d in dims:
            b = num(prow("dim_width", "naive", dim=d), "us")
            o = num(prow("dim_width", "simd", dim=d, simd_width=W), "us")
            spd.append(f"{b / o:.2f}x" if (b and o) else "n/a")
        rows.append([f"&nbsp;&nbsp;**Speedup** (norm. to no SIMD)"] + spd)
    emit(f, header, rows)

    # ---- Table 2.3: combined (Task 2C asks to combine 2.1 and 2.2) ----
    f.write("## Table 2.3 - Software prefetching + SIMD (Task 2C)\n\n")
    header = ["Metric"] + [f"dim={d}" for d in dims]
    rows = []
    for W in widths:
        rows.append([f"**Prefetch + SIMD {W}-bit**"] + [""] * len(dims))
        for field, mlabel in METRICS + [("instructions", "Instructions")]:
            rows.append([f"&nbsp;&nbsp;{mlabel}"] +
                        [fmt(num(prow("dim_width", "prefetch_simd", dim=d, simd_width=W), field)) for d in dims])
        spd = []
        for d in dims:
            b = num(prow("dim_width", "naive", dim=d), "us")
            o = num(prow("dim_width", "prefetch_simd", dim=d, simd_width=W), "us")
            spd.append(f"{b / o:.2f}x" if (b and o) else "n/a")
        rows.append([f"&nbsp;&nbsp;**Speedup** (norm. to no optimization)"] + spd)
    emit(f, header, rows)

print("wrote", OUT)
