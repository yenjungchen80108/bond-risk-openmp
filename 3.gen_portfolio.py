""" file: 3.gen_portfilo.py
Purpose: Generate a synthetic portfolio of bonds for the C program. (bonds.csv)
Each bond is priced from a known "true" yield, so the YTM always exists
and the C program's result can be checked against the true_ytm column.
"""

import numpy as np
import pandas as pd
import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

rng = np.random.default_rng(0)

coupon = rng.uniform(1, 8, N)            # annual coupon payment (face = 100)
face = 100.0
years = rng.integers(2, 31, N)

# Pick a true yield near the coupon rate, then price the bond from it.
true_ytm = np.clip(coupon / 100 + rng.uniform(-0.02, 0.02, N), 0.005, None)
price = (coupon * (1 - (1 + true_ytm) ** -years) / true_ytm
         + face * (1 + true_ytm) ** -years)

pd.DataFrame({
    "id": range(N),
    "coupon": coupon,
    "face": face,
    "years": years,
    "price": price,
    "true_ytm": true_ytm,
}).to_csv("bonds.csv", index=False)