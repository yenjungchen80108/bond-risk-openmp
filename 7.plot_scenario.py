"""Plot the two figures of the scenario analysis (report section 6).

Reads the CSV files written by ./scenario:
  scenario_accuracy.csv  -> fig/scenario_error.png  (approximation error vs shock size)
  scenario_timing.csv    -> fig/scenario_time.png   (running time vs number of scenarios S)

Usage: python3 8.plot_scenario.py [accuracy.csv] [timing.csv]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

acc_csv = sys.argv[1] if len(sys.argv) > 1 else "scenario_accuracy.csv"
tim_csv = sys.argv[2] if len(sys.argv) > 2 else "scenario_timing.csv"
COLORS = ["#1f6feb", "#2e7d32", "#8250df", "#d1481f"]

# ---------------------------------------------------------------- Figure 1: error vs shock
acc = pd.read_csv(acc_csv)
buckets = [b for b in acc["bucket"].unique() if b != "all"]

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for b, color in zip(buckets, COLORS):
    d = acc[acc["bucket"] == b]
    neg = d[d["shock_pct"] < 0].sort_values("shock_pct", ascending=False)
    pos = d[d["shock_pct"] > 0].sort_values("shock_pct")
    ax.plot(-neg["shock_pct"], neg["mean_abs_err_pct"], "o-", color=color, label=f"{b}")
    ax.plot(pos["shock_pct"], pos["mean_abs_err_pct"], "s--", color=color, mfc="none")

# reference line of slope 3 (error ~ shock^3), drawn through the longest-maturity curve at 1%
ref = acc[(acc["bucket"] == buckets[-1]) & (acc["shock_pct"] == -1.0)]
if not ref.empty:
    y0 = float(ref["mean_abs_err_pct"].iloc[0])
    xs = np.array([0.25, 2.0])
    ax.plot(xs, y0 * (xs / 1.0) ** 3, ":", color="black", lw=1.2, label="slope 3 (error $\\propto \\Delta y^3$)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.25, 0.5, 1, 2])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticklabels(["0.25%", "0.5%", "1%", "2%"])
ax.set_xlabel("Size of the yield shock |$\\Delta y$|")
ax.set_ylabel("Mean error (percentage points of price)")
ax.set_title("Duration + convexity approximation error\n(solid: yields fall, dashed: yields rise)", fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(title="Maturity", fontsize=8, title_fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("fig/scenario_error.png", dpi=200)
print("wrote fig/scenario_error.png")

# ---------------------------------------------------------------- Figure 2: time vs S
tim = pd.read_csv(tim_csv)
nthr = int(tim.loc[tim["method"] == "A_exact", "threads"].max())
N = int(tim["N"].iloc[0])

A1 = tim[(tim["method"] == "A_exact") & (tim["threads"] == 1)].sort_values("S")
Ap = tim[(tim["method"] == "A_exact") & (tim["threads"] == nthr)].sort_values("S")
dc_p = float(tim[(tim["method"] == "DC_cost") & (tim["threads"] == nthr)]["time_s"].iloc[0])
agg = tim[tim["method"] == "B_agg"].sort_values("S")
b_total = agg["time_s"].to_numpy() + dc_p          # approximation incl. one-time cost of D and C

a_p = float(np.median(Ap["time_s"] / Ap["S"]))     # exact cost per scenario, all threads
s_star = dc_p / a_p                                # break-even number of scenarios

fig, ax = plt.subplots(figsize=(6.2, 4.4))
S_line = np.array([1, A1["S"].max() * 2])
a_1 = float(np.median(A1["time_s"] / A1["S"]))
ax.plot(S_line, a_1 * S_line, ":", color=COLORS[0], lw=1)
ax.plot(S_line, a_p * S_line, ":", color=COLORS[1], lw=1)
ax.plot(A1["S"], A1["time_s"], "o-", color=COLORS[0], label="exact repricing, 1 thread")
ax.plot(Ap["S"], Ap["time_s"], "o-", color=COLORS[1], label=f"exact repricing, {nthr} threads")
ax.plot(agg["S"], b_total, "s-", color=COLORS[3], label=f"approximation incl. one-time $D$, $C$ ({nthr} threads)")
ax.axhline(dc_p, color=COLORS[3], lw=0.8, ls=":")
ax.axvline(s_star, color="black", lw=0.8, ls="--")
ax.annotate(f"break-even\nS* $\\approx$ {s_star:.1f}", xy=(s_star, dc_p), xytext=(s_star * 1.25, dc_p * 0.22),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(sorted(set(A1["S"]) | {1}))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticklabels([str(s) for s in sorted(set(A1["S"]) | {1})])
ax.set_xlabel("Number of scenarios S")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title(f"Exact repricing vs approximation ({N:,} bonds, median of 3 runs)", fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("fig/scenario_time.png", dpi=200)
print("wrote fig/scenario_time.png")
print(f"break-even S* = {s_star:.2f}  (D/C {dc_p:.4f} s / exact {a_p:.6f} s per scenario, {nthr} threads)")