/* file: 2.duration_convexity.h
Purpose: Define the functions for the duration and convexity of a bond.
*/

#ifndef DURATION_CONVEXITY_H
#define DURATION_CONVEXITY_H

double compute_duration(double ytm, double coupon, double face, int years);
double compute_convexity(double ytm, double coupon, double face, int years);

#endif