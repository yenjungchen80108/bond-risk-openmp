"""Print the scaling tables used in the report, from scaling.csv (written by run_scaling.sh).

Table 1: compute loop, speedup S and efficiency E for each OpenMP schedule.
Table 2: end-to-end (read + compute + write) with the Amdahl's-law prediction.

Usage: python3 summarize_scaling.py [scaling.csv] [--md]
Default: pretty boxed tables in the terminal (needs `pip install tabulate`).
--md   : plain Markdown tables instead (also used if tabulate is not installed).
"""
import sys
import pandas as pd

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

args = [a for a in sys.argv[1:] if not a.startswith("--")]
use_md = "--md" in sys.argv or tabulate is None
csv = args[0] if args else "scaling.csv"


def show(headers, rows):
    if use_md:
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid",
                       stralign="right", disable_numparse=True))


df = pd.read_csv(csv)
med = df.groupby(["schedule", "threads"]).median(numeric_only=True)

SCHEDULES = [("dynamic", "dynamic (chunk 1)"),
             ("static", "static"),
             ("dynamic_1000", "dynamic (chunk 1000)")]
SCHEDULES = [(k, name) for k, name in SCHEDULES if k in med.index.get_level_values(0)]
threads = sorted(med.index.get_level_values(1).unique())


def speedup_eff(sched, col, p):
    t1 = med.loc[(sched, threads[0]), col]
    tp = med.loc[(sched, p), col]
    return t1 / tp, t1 / tp / p * 100


print("Table 1. Compute loop: speedup S and efficiency E (median of repeated runs)\n")
head = ["threads"] + [f"{name} S / E" for _, name in SCHEDULES]
rows = []
for p in threads:
    row = [str(p)]
    for sched, _ in SCHEDULES:
        s, e = speedup_eff(sched, "compute_s", p)
        row.append(f"{s:.2f} / {e:.1f}%")
    rows.append(row)
show(head, rows)

# End-to-end uses the schedule used in the report (dynamic_1000 if present).
e2e = "dynamic_1000" if "dynamic_1000" in dict(SCHEDULES) else SCHEDULES[0][0]
t1 = med.loc[(e2e, threads[0])]
s_frac = (t1["read_s"] + t1["write_s"]) / t1["total_s"]
print(f"\nTable 2. End-to-end, schedule = {e2e}")
print(f"(read + write = {t1['read_s'] + t1['write_s']:.3f} s = {s_frac * 100:.1f}% of the "
      f"1-thread run; Amdahl limit 1/s = {1 / s_frac:.2f}x)\n")
head = ["threads", "total (s)", "read (s)", "compute (s)", "write (s)",
        "speedup", "Amdahl predicted"]
rows = []
for p in threads:
    r = med.loc[(e2e, p)]
    s = t1["total_s"] / r["total_s"]
    pred = 1 / (s_frac + (1 - s_frac) / p)
    rows.append([str(p), f"{r['total_s']:.3f}", f"{r['read_s']:.3f}",
                 f"{r['compute_s']:.3f}", f"{r['write_s']:.3f}", f"{s:.2f}", f"{pred:.2f}"])
show(head, rows)