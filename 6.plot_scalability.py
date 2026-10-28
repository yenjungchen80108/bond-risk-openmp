"""Plot wall-clock time, speedup and efficiency versus thread count.

Data: median of five runs, N = 1,000,000 bonds, Apple M4 Pro (compute loop only).
To add more curves (e.g. static vs dynamic scheduling), add entries to RUNS.
Usage: python3 6.plot_scalability.py   -> fig/scaling.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "OpenMP, schedule(dynamic)": {
        1: 2.229205, 2: 1.113167, 4: 0.556026, 8: 0.283055,
    },
}
COLORS = ["#1f6feb", "#d1481f", "#2e7d32"]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for (label, data), color in zip(RUNS.items(), COLORS):
    p = sorted(data)
    t = [data[k] for k in p]
    s = [t[0] / x for x in t]
    e = [s[i] / p[i] for i in range(len(p))]
    ax[0].plot(p, t, "o-", color=color, label=label)
    ax[1].plot(p, s, "o-", color=color, label=label)
    ax[2].plot(p, e, "o-", color=color, label=label)

pmax = max(max(d) for d in RUNS.values())
ax[1].plot([0, pmax], [0, pmax], "k-", lw=1, label="ideal (S = p)")
ax[2].axhline(1.0, color="k", lw=1, label="ideal (E = 1)")

ax[0].set_ylabel("Wall-clock time (s)")
ax[1].set_ylabel("Speedup  S(p) = T(1)/T(p)")
ax[2].set_ylabel("Efficiency  E(p) = S(p)/p")
ax[2].set_ylim(0, 1.2)
ax[0].set_ylim(bottom=0)
ax[1].set_ylim(0, pmax + 0.5)
for a in ax:
    a.set_xlabel("Number of threads")
    a.set_xticks(sorted(set(k for d in RUNS.values() for k in d)))
    a.grid(alpha=0.3)
    a.legend(fontsize=8, frameon=False)
fig.suptitle("Scalability of the per-bond YTM + duration + convexity loop "
             "(N = 10$^6$, Apple M4 Pro)", fontsize=10)
fig.tight_layout()
fig.savefig("fig/scaling.png", dpi=200)
print("wrote fig/scaling.png")