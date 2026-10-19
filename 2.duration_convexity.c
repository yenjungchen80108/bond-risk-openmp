/* file: 2.duration_convexity.c
Purpose: Calculate the duration and convexity of a bond.
*/

#include <stdio.h>
#include <math.h>
#include "1.ytm.h"

/* Effective duration via central finite difference at the given yield. */
double compute_duration(double ytm, double coupon, double face, int years) {
    double h  = 1e-4;
    double p0 = bond_price(ytm,     coupon, face, years);
    double pm = bond_price(ytm - h, coupon, face, years);
    double pp = bond_price(ytm + h, coupon, face, years);
    return (pm - pp) / (2.0 * p0 * h);
}

/* Convexity via central finite difference at the given yield. */
double compute_convexity(double ytm, double coupon, double face, int years) {
    double h  = 1e-4;
    double p0 = bond_price(ytm,     coupon, face, years);
    double pm = bond_price(ytm - h, coupon, face, years);
    double pp = bond_price(ytm + h, coupon, face, years);
    return (pm + pp - 2.0 * p0) / (p0 * h * h);
}
