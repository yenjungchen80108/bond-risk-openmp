/* file: 1.ytm.h
Purpose: Define the functions for the YTM (internal rate of return) of a bond.
*/

#ifndef YTM_H
#define YTM_H

double bond_price(double r, double coupon, double face, int years);
double bond_price_derivative(double r, double coupon, double face, int years);
double compute_irr_binary_search(double coupon, double price, double face, int years);
double compute_irr_newton(double coupon, double price, double face, int years);
double compute_irr_binary_search_iter(double coupon, double price, double face, int years, int *iters);
double compute_irr_newton_iter(double coupon, double price, double face, int years, int *iters);
#endif