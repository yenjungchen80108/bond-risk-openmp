""" file: 4.plot_DC_curve.py
Purpose: Plot duration / convexity of the bond portfolio (from results.csv),
and the price-yield curve of one bond with its duration / convexity approximations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bonds = pd.read_csv("bonds.csv")
res = pd.read_csv("results.csv")
df = bonds.merge(res, on="id")

print("max |ytm - true_ytm| =", (df["ytm"] - df["true_ytm"]).abs().max())
print(df[["ytm", "duration", "convexity"]].describe())

# ---- Figure 1: portfolio-level view ----
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].scatter(df["years"], df["duration"], s=2, alpha=0.3)
ax[0, 0].set(xlabel="Maturity (years)", ylabel="Modified duration",
             title="Duration vs. maturity")
ax[0, 1].scatter(df["years"], df["convexity"], s=2, alpha=0.3, color="C1")
ax[0, 1].set(xlabel="Maturity (years)", ylabel="Convexity",
             title="Convexity vs. maturity")
ax[1, 0].hist(df["duration"], bins=60)
ax[1, 0].set(xlabel="Modified duration", ylabel="Count", title="Duration distribution")
ax[1, 1].hist(df["convexity"], bins=60, color="C1")
ax[1, 1].set(xlabel="Convexity", ylabel="Count", title="Convexity distribution")
fig.tight_layout()
fig.savefig("portfolio_duration_convexity.png", dpi=200)

# ---- Figure 2: price-yield curve of one bond ----
b = df.iloc[0]                                   # change the row to look at another bond
def price(y):
    t = np.arange(1, int(b["years"]) + 1)
    return (b["coupon"] / (1 + y) ** t).sum() + b["face"] / (1 + b["years"] * 0 + y) ** b["years"]

y0 = b["ytm"]
y = np.linspace(max(y0 - 0.04, 0.001), y0 + 0.04, 200)
dy = y - y0
P0 = price(y0)
actual = np.array([price(v) for v in y])
lin = P0 * (1 - b["duration"] * dy)
quad = P0 * (1 - b["duration"] * dy + 0.5 * b["convexity"] * dy ** 2)

plt.figure(figsize=(7, 5))
plt.plot(y, actual, label="Actual price", lw=2)
plt.plot(y, lin, "--", label="Duration only")
plt.plot(y, quad, label="Duration + convexity")
plt.scatter([y0], [P0], color="k", zorder=3, label="Current YTM")
plt.xlabel("Yield"); plt.ylabel("Price")
plt.title(f"Bond id={int(b['id'])}: {int(b['years'])}-year, coupon {b['coupon']:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("price_yield_curve.png", dpi=200)
plt.show()