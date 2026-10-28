"""Plot wall-clock time, speedup and efficiency versus thread count.

Reads scaling.csv (written by run_scaling.sh: one row per run), takes the median of
the repeated runs for each (schedule, threads), and draws one curve per series.
Each curve is normalised by its own 1-thread time.

Usage: python3 6.plot_scalability.py [scaling.csv]   -> fig/scaling.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "scaling.csv"
df = pd.read_csv(csv)
med = df.groupby(["schedule", "threads"]).median(numeric_only=True).reset_index()

# (label, schedule, column). Edit this list to add or remove curves.
SERIES = [
    ("compute loop only, dynamic", "dynamic", "compute_s"),
    ("end-to-end (read+compute+write), dynamic chunk 1000", "dynamic_1000", "total_s"),
    ("compute loop only, static", "static", "compute_s"),
    ("compute loop only, dynamic chunk 1000", "dynamic_1000", "compute_s"),
]
# A series whose schedule is not in the CSV is skipped automatically.
COLORS = ["#1f6feb", "#d1481f", "#2e7d32", "#8250df"]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
all_p = set()
for (label, sched, col), color in zip(SERIES, COLORS):
    d = med[med["schedule"] == sched].sort_values("threads")
    if d.empty:
        continue
    p = d["threads"].tolist()
    t = d[col].tolist()
    s = [t[0] / x for x in t]
    e = [s[i] / p[i] for i in range(len(p))]
    all_p.update(p)
    ax[0].plot(p, t, "o-", color=color, label=label)
    ax[1].plot(p, s, "o-", color=color, label=label)
    ax[2].plot(p, e, "o-", color=color, label=label)

pmax = max(all_p)
ax[1].plot([0, pmax], [0, pmax], "k-", lw=1, label="ideal (S = p)")
ax[2].axhline(1.0, color="k", lw=1, label="ideal (E = 1)")
ax[0].set_ylabel("Wall-clock time (s)")
ax[1].set_ylabel("Speedup  S(p) = T(1)/T(p)")
ax[2].set_ylabel("Efficiency  E(p) = S(p)/p")
ax[0].set_ylim(bottom=0)
ax[1].set_ylim(0, pmax + 0.5)
ax[2].set_ylim(0, 1.2)
for a in ax:
    a.set_xlabel("Number of threads")
    a.set_xticks(sorted(all_p))
    a.grid(alpha=0.3)
ax[0].legend(fontsize=7, frameon=False)
fig.suptitle("Scalability of the portfolio program (median of repeated runs)", fontsize=10)
fig.tight_layout()
fig.savefig("fig/scaling.png", dpi=200)
print("wrote fig/scaling.png")
print(med.to_string(index=False))