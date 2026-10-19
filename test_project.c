/* file: test_project.c
Purpose: Unit tests for YTM solvers, duration, convexity.
Build and run:  make test

Every expected value is computed independently of the code under test
(closed-form annuity price, or a direct sum over cash flows), so a bug in
bond_price() cannot hide itself. Exit code is 0 only if all checks pass.
*/

#include <stdio.h>
#include <math.h>
#include "1.ytm.h"
#include "2.duration_convexity.h"

static int total = 0, failures = 0;

static void check(const char *name, double got, double want, double tol) {
    int ok = fabs(got - want) <= tol;
    total++;
    if (!ok) failures++;
    printf("[%s] %-44s got %.10g, want %.10g (tol %g)\n",
           ok ? "PASS" : "FAIL", name, got, want, tol);
}

static void check_true(const char *name, int cond) {
    total++;
    if (!cond) failures++;
    printf("[%s] %s\n", cond ? "PASS" : "FAIL", name);
}

/* Closed-form (annuity) price: independent of bond_price(). */
static double closed_form_price(double y, double coupon, double face, int n) {
    return coupon * (1.0 - pow(1.0 + y, -n)) / y + face * pow(1.0 + y, -n);
}

/* Modified duration by direct summation: (1/P) * sum t*CF_t / (1+y)^(t+1). */
static double exact_mod_duration(double y, double coupon, double face, int n) {
    double s = 0.0;
    for (int t = 1; t <= n; t++) {
        double cf = coupon + (t == n ? face : 0.0);
        s += t * cf / pow(1.0 + y, t + 1);
    }
    return s / closed_form_price(y, coupon, face, n);
}

/* Convexity by direct summation: (1/P) * sum t(t+1)*CF_t / (1+y)^(t+2). */
static double exact_convexity(double y, double coupon, double face, int n) {
    double s = 0.0;
    for (int t = 1; t <= n; t++) {
        double cf = coupon + (t == n ? face : 0.0);
        s += t * (t + 1.0) * cf / pow(1.0 + y, t + 2);
    }
    return s / closed_form_price(y, coupon, face, n);
}

int main(void) {
    const double coupons[] = {1.0, 3.0, 5.0, 8.0};
    const int    years_[]  = {2, 10, 30};
    const double yields[]  = {0.01, 0.05, 0.10};

    printf("== Pricing ==\n");
    check("bond_price matches closed form (5%, 7.5%, 20y)",
          bond_price(0.075, 5, 100, 20), closed_form_price(0.075, 5, 100, 20), 1e-9);

    printf("\n== YTM solvers ==\n");
    check("par bond (coupon 5, price 100, 10y): bisection",
          compute_irr_binary_search(5, 100, 100, 10), 0.05, 1e-7);
    check("par bond (coupon 5, price 100, 10y): newton",
          compute_irr_newton(5, 100, 100, 10), 0.05, 1e-7);
    /* Example used in the report: price 74.51 is rounded, so allow 1e-4 in yield. */
    check("report example (coupon 5, price 74.51, 20y): newton",
          compute_irr_newton(5, 74.51, 100, 20), 0.075, 1e-4);

    /* Round trip: price a bond from a known yield, then solve for it. */
    int bad_b = 0, bad_n = 0, bad_iter = 0, max_newton_iter = 0, bad_dur = 0, bad_cvx = 0, cases = 0;
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 3; b++)
            for (int c = 0; c < 3; c++) {
                double cp = coupons[a], y = yields[c];
                int n = years_[b];
                double p = closed_form_price(y, cp, 100.0, n);
                int ib, in;
                double rb = compute_irr_binary_search_iter(cp, p, 100.0, n, &ib);
                double rn = compute_irr_newton_iter(cp, p, 100.0, n, &in);
                cases++;
                if (fabs(rb - y) > 1e-7) bad_b++;
                if (fabs(rn - y) > 1e-7) bad_n++;
                if (in > max_newton_iter) max_newton_iter = in;
                if (ib != 27) bad_iter++;

                double d  = compute_duration(y, cp, 100.0, n);
                double dx = exact_mod_duration(y, cp, 100.0, n);
                double v  = compute_convexity(y, cp, 100.0, n);
                double vx = exact_convexity(y, cp, 100.0, n);
                /* Central differences with h = 1e-4 have O(h^2) truncation error
                   (measured max relative error is about 1.5e-6), so allow 1e-5. */
                if (fabs(d - dx) > 1e-5 * dx) bad_dur++;
                if (fabs(v - vx) > 1e-5 * vx) bad_cvx++;
            }
    printf("(%d round-trip cases: coupons x maturities x yields)\n", cases);
    check("bisection recovers the true yield (failures)", bad_b, 0, 0);
    check("newton recovers the true yield (failures)", bad_n, 0, 0);
    check("bisection uses exactly 27 iterations (failures)", bad_iter, 0, 0);
    check_true("newton needs at most 10 iterations", max_newton_iter <= 10);

    printf("\n== Duration and convexity vs direct summation ==\n");
    check("duration within 1e-5 relative (failures)", bad_dur, 0, 0);
    check("convexity within 1e-5 relative (failures)", bad_cvx, 0, 0);
    /* Par bond closed form: D_mod = (1/y) * (1 - (1+y)^-n). */
    check("par bond duration (5%, 10y) closed form",
          compute_duration(0.05, 5, 100, 10), (1.0 / 0.05) * (1.0 - pow(1.05, -10)), 1e-4);

    printf("\n== Taylor approximation (5%% coupon, 10y, yield 5%%, shock +1%%) ==\n");
    {
        double y0 = 0.05, dy = 0.01;
        double p0 = closed_form_price(y0, 5, 100, 10);
        double pa = closed_form_price(y0 + dy, 5, 100, 10);
        double d  = compute_duration(y0, 5, 100, 10);
        double v  = compute_convexity(y0, 5, 100, 10);
        double e1 = fabs(p0 * (1.0 - d * dy) - pa);
        double e2 = fabs(p0 * (1.0 - d * dy + 0.5 * v * dy * dy) - pa);
        printf("price error: duration only %.4f, duration+convexity %.4f\n", e1, e2);
        check_true("adding convexity reduces the error", e2 < e1);
    }

    printf("\n%d checks, %d failed\n", total, failures);
    return failures ? 1 : 0;
}