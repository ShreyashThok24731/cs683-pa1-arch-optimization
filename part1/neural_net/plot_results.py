#!/usr/bin/env python3
import csv, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "plots")
os.makedirs(P, exist_ok=True)

def load_map(path):
    m = {}
    for row in csv.DictReader(open(path)):
        m[row["variant"]] = float(row["ms"])
    return m

mm = load_map(os.path.join(R, "matmul.csv"))
nn = load_map(os.path.join(R, "nn_training.csv"))

variants = ["Basic","Reordered","Unrolled","Tiled","Vectorized"]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

xs = list(range(len(variants)))
mm_vals = [mm[v] for v in variants]
nn_vals = [nn[v] for v in variants]

ax[0].bar(xs, mm_vals, color="steelblue")
ax[0].set_xticks(xs); ax[0].set_xticklabels(variants, rotation=20)
ax[0].set_title("Single 512x512 MatMul")
ax[0].set_ylabel("Execution time (ms)")
for x, v in zip(xs, mm_vals):
    ax[0].text(x, v, f"{v:.0f}", ha="center", va="bottom")

ax[1].bar(xs, nn_vals, color="darkorange")
ax[1].set_xticks(xs); ax[1].set_xticklabels(variants, rotation=20)
ax[1].set_title("Full NN training epoch (512 batch, 2 hidden layers)")
ax[1].set_ylabel("Execution time (ms)")
for x, v in zip(xs, nn_vals):
    ax[1].text(x, v, f"{v:.0f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(P, "nn_matmul_and_training.png"), dpi=140)

plt.figure(figsize=(8, 5))
plt.title("Speedup over Basic")
mm_spd = [mm["Basic"] / mm[v] for v in variants]
nn_spd = [nn["Basic"] / nn[v] for v in variants]
w = 0.35
plt.bar([x - w/2 for x in xs], mm_spd, w, label="MatMul")
plt.bar([x + w/2 for x in xs], nn_spd, w, label="NN training")
plt.xticks(xs, variants, rotation=20); plt.ylabel("Speedup (x)")
plt.grid(alpha=0.3, axis="y"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(P, "nn_speedup.png"), dpi=140)
print("wrote plots to", P)
