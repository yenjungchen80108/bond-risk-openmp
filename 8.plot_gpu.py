"""Plot the end-to-end time breakdown of the three implementations (report section 6).

Reads cuda_run.out (written by cuda_run.sl) and writes fig/gpu_breakdown.png:
a stacked bar per implementation (OpenMP 1 thread, OpenMP 8 threads, GPU) split into
read / init + transfer / compute / write. It shows why a 53x faster computation gives
only a 1.6x faster program: the serial CSV read and write dominate once compute is cheap.

Usage: python3 8.plot_gpu.py [cuda_run.out] [threads_per_block]   (default block 256)
"""
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_file = sys.argv[1] if len(sys.argv) > 1 else "cuda_run.out"
block = sys.argv[2] if len(sys.argv) > 2 else "256"

# --------------------------------------------------------------- parse cuda_run.out
text = open(out_file).read()
# each block is "##### title #####" followed by its output, up to the next "#####" line
sections = {title.strip(): body for title, body in
            re.findall(r"^##### (.*?) #####\n(.*?)(?=^##### |\Z)", text, re.S | re.M)}


def e2e_parts(body):
    """Return (total, {stage: seconds}) from the 'e2e total: ... (stage t, ...)' line."""
    m = re.search(r"e2e total: ([\d.]+) s \((.*?)\)", body)
    return float(m.group(1)), {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", m.group(2))}


def find(prefix):
    for name, body in sections.items():
        if name.startswith(prefix):
            return body
    sys.exit(f"section starting with '{prefix}' not found in {out_file}")


runs = []  # (label, total, read, init+transfer, compute, write)
for nthr in (1, 8):
    total, p = e2e_parts(find(f"CPU OpenMP, {nthr} thread"))
    runs.append((f"OpenMP\n{nthr} thread" + ("s" if nthr > 1 else ""), total,
                 p["read"], 0.0, p["compute"], p["write"]))
total, p = e2e_parts(find(f"GPU CUDA, {block} threads per block"))
runs.append((f"GPU (CUDA)\nblock {block}", total, p["read"],
             p["init"] + p["h2d"] + p["d2h"], p["kernel"], p["write"]))

# --------------------------------------------------------------- figure
# categorical slots 1-4 in fixed order (validated for adjacent-pair colour-blind separation)
STAGES = ["Read CSV", "GPU init + transfer", "Compute", "Write CSV"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

os.makedirs("fig", exist_ok=True)
fig, ax = plt.subplots(figsize=(6.6, 4.6))
x = range(len(runs))
bottoms = [0.0] * len(runs)
for si, (stage, color) in enumerate(zip(STAGES, COLORS)):
    vals = [r[2 + si] for r in runs]
    ax.bar(x, vals, bottom=bottoms, width=0.55, color=color, edgecolor="white",
           linewidth=1.5, label=stage)
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v >= 0.7:  # label only the segments that are tall enough to hold text
            ax.text(xi, b + v / 2, f"{v:.2f} s", ha="center", va="center", fontsize=8,
                    color="#0b0b0b")
    bottoms = [b + v for b, v in zip(bottoms, vals)]

base = runs[0][1]
for xi, r in enumerate(runs):
    ax.text(xi, r[1] + 0.25, f"{r[1]:.1f} s" + ("" if xi == 0 else f"  ({base / r[1]:.1f}x)"),
            ha="center", va="bottom", fontsize=9, fontweight="bold")

# the GPU init and compute segments are too thin to hold text: label them beside the bar
g = runs[-1]
gx = len(runs) - 1
init_y = g[2] + g[3] / 2
comp_y = g[2] + g[3] + g[4] / 2
for text_y, seg_y, label in ((0.3, init_y, f"init + transfer: {g[3]:.2f} s"),
                             (1.75, comp_y, f"compute: {g[4] * 1000:.0f} ms")):
    ax.annotate(label, xy=(gx + 0.275, seg_y), xytext=(gx + 0.36, text_y), fontsize=8,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))

ax.set_xticks(list(x))
ax.set_xticklabels([r[0] for r in runs])
ax.set_ylabel("End-to-end time (s)")
ax.set_ylim(0, base * 1.2)
ax.set_xlim(-0.5, len(runs) - 1 + 1.15)
ax.set_title("End-to-end time for $10^6$ bonds: compute is no longer the bottleneck", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("fig/gpu_breakdown.png", dpi=200)
print("wrote fig/gpu_breakdown.png")
for r in runs:
    print(f"{r[0].replace(chr(10), ' '):22s} total {r[1]:7.3f}  read {r[2]:.3f}  init+xfer {r[3]:.3f}"
          f"  compute {r[4]:.3f}  write {r[5]:.3f}")

# --------------------------------------------------------------- results table (section 6)
# Compute time of the GPU row is the median of 5 kernel launches (the e2e run above includes
# the first launch, which also pays one-time warm-up costs).
m = re.search(r"kernel \(median of 5 launches\): ([\d.]+) s", find(f"GPU CUDA, {block} threads per block"))
rows = [("OpenMP, 1 thread", runs[0][4], runs[0][1]),
        ("OpenMP, 8 threads", runs[1][4], runs[1][1]),
        (f"GPU (CUDA), block {block}", float(m.group(1)), runs[2][1])]
c1, c8 = rows[0][1], rows[1][1]


def fmt_t(t):
    return f"{t:.2f}" if t >= 1 else (f"{t:.3f}" if t >= 0.1 else f"{t:.4f}")


print("\nMarkdown table:")
print("| Version | Compute (s) | vs 1 thread | vs 8 threads | End-to-end (s) |")
print("|---|---|---|---|---|")
for i, (name, c, e) in enumerate(rows):
    v8 = "--" if i == 0 else f"{c8 / c:.1f}x"
    print(f"| {name} | {fmt_t(c)} | {c1 / c:.1f}x | {v8} | {e:.2f} |")

print("\nLaTeX rows:")
for i, (name, c, e) in enumerate(rows):
    v8 = "--" if i == 0 else f"${c8 / c:.1f}\\times$"
    print(f"{name} & {fmt_t(c)} & ${c1 / c:.1f}\\times$ & {v8} & {e:.2f} \\\\")