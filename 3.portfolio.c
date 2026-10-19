/* file: 3.portfolio.c
Purpose: Read bonds.csv, solve YTM / duration / convexity for every bond
(OpenMP parallel loop), and write the results to results.csv.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "1.ytm.h"
#include "2.duration_convexity.h"

int main(void) {
    FILE *fp = fopen("bonds.csv", "r");
    if (!fp) { perror("bonds.csv"); return 1; }

    int cap = 1 << 20, n = 0;
    double *coupon = malloc(cap * sizeof(double));
    double *face   = malloc(cap * sizeof(double));
    int    *years  = malloc(cap * sizeof(int));
    double *price  = malloc(cap * sizeof(double));

    char line[256];
    fgets(line, sizeof line, fp);                 // skip the header line
    int id;
    while (n < cap && fgets(line, sizeof line, fp)) {
        if (sscanf(line, "%d,%lf,%lf,%d,%lf",
                   &id, &coupon[n], &face[n], &years[n], &price[n]) == 5)
            n++;
    }
    fclose(fp);

    /* allocate memory for the results */
    double *ytm = malloc(n * sizeof(double));
    double *dur = malloc(n * sizeof(double));
    double *cvx = malloc(n * sizeof(double));

    /* start timing */
    double t0 = omp_get_wtime();

    /* the thread will finish one bond and then move on to the next one.
    the cost of different bonds is different (the longer the maturity, the more iterations),
    dynamic scheduling can reduce the load imbalance. */
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        ytm[i] = compute_irr_newton(coupon[i], price[i], face[i], years[i]);
        dur[i] = compute_duration(ytm[i], coupon[i], face[i], years[i]);
        cvx[i] = compute_convexity(ytm[i], coupon[i], face[i], years[i]);
    }

    /* the timing only includes the computation, not the file reading and writing,
    so the measured speedup is the pure computation speedup. */
    double t1 = omp_get_wtime();
    printf("bonds: %d, threads: %d, time: %.6f s\n",
           n, omp_get_max_threads(), t1 - t0);

    FILE *out = fopen("results.csv", "w");
    if (!out) { perror("results.csv"); return 1; }
    fprintf(out, "id,ytm,duration,convexity\n");
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(ytm[i])) bad++;
        fprintf(out, "%d,%.8f,%.6f,%.6f\n", i, ytm[i], dur[i], cvx[i]);
    }
    fclose(out);
    if (bad) printf("warning: %d bonds did not converge (nan/inf)\n", bad);

    /* free the memory */
    free(coupon); free(face); free(years); free(price);
    free(ytm); free(dur); free(cvx);
    return 0;
}