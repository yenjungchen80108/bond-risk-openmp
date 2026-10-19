/* file: 5.convergence.c
Purpose: Compare bisection and Newton-Raphson on the whole portfolio at the
same tolerance (1e-8): iteration counts, total time, and error vs true_ytm.
Sequential on purpose (no OpenMP), so the numbers are not affected by threads. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "1.ytm.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    FILE *fp = fopen("bonds.csv", "r");
    if (!fp) { perror("bonds.csv"); return 1; }

    int cap = 1 << 20, n = 0;
    double *coupon = malloc(cap * sizeof(double));
    double *face   = malloc(cap * sizeof(double));
    int    *years  = malloc(cap * sizeof(int));
    double *price  = malloc(cap * sizeof(double));
    double *truey  = malloc(cap * sizeof(double));

    char line[256];
    fgets(line, sizeof line, fp);                 /* skip header */
    int id;
    while (n < cap && fgets(line, sizeof line, fp)) {
        if (sscanf(line, "%d,%lf,%lf,%d,%lf,%lf",
                   &id, &coupon[n], &face[n], &years[n], &price[n], &truey[n]) == 6)
            n++;
    }
    fclose(fp);

    long sum_b = 0, sum_n = 0;
    int max_b = 0, max_n = 0, it;
    double err_b = 0.0, err_n = 0.0;

    double t0 = now();
    for (int i = 0; i < n; i++) {
        double r = compute_irr_binary_search_iter(coupon[i], price[i], face[i], years[i], &it);
        sum_b += it;
        if (it > max_b) max_b = it;
        if (fabs(r - truey[i]) > err_b) err_b = fabs(r - truey[i]);
    }
    double t1 = now();
    for (int i = 0; i < n; i++) {
        double r = compute_irr_newton_iter(coupon[i], price[i], face[i], years[i], &it);
        sum_n += it;
        if (it > max_n) max_n = it;
        if (fabs(r - truey[i]) > err_n) err_n = fabs(r - truey[i]);
    }
    double t2 = now();

    printf("bonds: %d (tolerance 1e-8 for both)\n", n);
    printf("%-10s %10s %10s %12s %14s\n", "method", "avg iter", "max iter", "time (s)", "max |err|");
    printf("%-10s %10.2f %10d %12.6f %14.3e\n", "bisection", (double)sum_b / n, max_b, t1 - t0, err_b);
    printf("%-10s %10.2f %10d %12.6f %14.3e\n", "newton",    (double)sum_n / n, max_n, t2 - t1, err_n);

    free(coupon); free(face); free(years); free(price); free(truey);
    return 0;
}