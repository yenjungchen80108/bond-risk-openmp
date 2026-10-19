/* file: 1.ytm.c
Purpose: Calculate the YTM (internal rate of return) of a bond.
Bond YTM (internal rate of return):
  price = coupon/(1+r) + coupon/(1+r)^2 + ... + (coupon + face)/(1+r)^n
*/

#include <stdio.h>

/* Method1. Binary search to find the IRR.
   iters (may be NULL) returns how many iterations were used. */
   double compute_irr_binary_search_iter(double coupon, double price, double face, int years, int *iters) {
    double low = 0.0;
    double high = 1.0;
    double middle = 0.0;
    double tol = 1e-8;      /* same tolerance as Newton, so the comparison is fair */
    int k = 0;

    while (high - low > tol) {
        middle = (low + high) / 2.0;
        double value = 0.0;
        double discount = 1.0;

        for (int i = 1; i <= years; i++) {
            discount = discount / (1.0 + middle);
            double cashflow = coupon;
            if (i == years) {
                cashflow += face;
            }
            value += cashflow * discount;
        }

        value = value - price;

        if (value > 0) {
            low = middle;
        } else {
            high = middle;
        }
        k++;                /* count one iteration */
    }

    if (iters) *iters = k;
    return middle;
}

double compute_irr_binary_search(double coupon, double price, double face, int years) {
    return compute_irr_binary_search_iter(coupon, price, face, years, NULL);
}

/* Method2. Newton's method to find the IRR
A kind of mathematical method to find the result in real number and complex number.
*/
#include <math.h>

// f(r)  = sum_{t=1}^{n} coupon/(1+r)^t + face/(1+r)^n - price
// f'(r) = -sum_{t=1}^{n} t*coupon/(1+r)^(t+1) - n*face/(1+r)^(n+1)
double bond_price(double r, double coupon, double face, int years) {
    double p = 0.0;
    for (int t = 1; t <= years; t++) {
        p += coupon / pow(1 + r, t);
    }
    p += face / pow(1 + r, years);
    return p;
}

double bond_price_derivative(double r, double coupon, double face, int years) {
    double dp = 0.0;
    for (int t = 1; t <= years; t++) {
        dp -= t * coupon / pow(1 + r, t + 1);
    }
    dp -= years * face / pow(1 + r, years + 1);
    return dp;
}

double compute_irr_newton_iter(double coupon, double price, double face, int years, int *iters) {
    double r = 0.05;
    double tol = 1e-8;
    int max_iter = 100;
    int k;                              /* declared outside the loop so we can read it after */

    for (k = 0; k < max_iter; k++) {
        double f_rk  = bond_price(r, coupon, face, years) - price;
        double fp_rk = bond_price_derivative(r, coupon, face, years);

        if (fabs(fp_rk) < 1e-12) break;

        double r_next = r - f_rk / fp_rk;

        if (fabs(r_next - r) < tol) {
            if (iters) *iters = k + 1;
            return r_next;
        }
        r = r_next;
    }
    if (iters) *iters = k;
    return r;
}

double compute_irr_newton(double coupon, double price, double face, int years) {
    return compute_irr_newton_iter(coupon, price, face, years, NULL);
}

#ifndef UNIT_TEST
int main(void) {
    double coupon, price, face;
    int years;

    printf("Enter the annual coupon: ");
    scanf("%lf", &coupon);

    printf("Enter the bond price: ");
    scanf("%lf", &price);

    printf("Enter the face value (principal): ");
    scanf("%lf", &face);

    printf("Enter the number of years: ");
    scanf("%d", &years);

    double r = compute_irr_binary_search(coupon, price, face, years);
    printf("IRR using binary search = %0.2f%%\n", r * 100);

    double dr = compute_irr_newton(coupon, price, face, years);
    printf("IRR using Newton's method = %0.2f%%\n", dr * 100);

    return 0;
}
#endif
