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
    double t_start = omp_get_wtime();
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
    double t_read = omp_get_wtime();

    /* allocate memory for the results */
    double *ytm = malloc(n * sizeof(double));
    double *dur = malloc(n * sizeof(double));
    double *cvx = malloc(n * sizeof(double));

    /* start timing */
    double t0 = omp_get_wtime();

    /* Each thread takes bonds from a shared queue. The cost per bond varies with
    maturity and iteration count, so the schedule is chosen at run time with
    OMP_SCHEDULE; the best setting found was OMP_SCHEDULE="dynamic,1000".
    */
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        ytm[i] = compute_irr_newton(coupon[i], price[i], face[i], years[i]);
        dur[i] = compute_duration(ytm[i], coupon[i], face[i], years[i]);
        cvx[i] = compute_convexity(ytm[i], coupon[i], face[i], years[i]);
    }

    /* The compute timer excludes file reading and writing; the end-to-end timer below includes both. */
    double t1 = omp_get_wtime();
    printf("bonds: %d, threads: %d, time: %.6f s\n",
           n, omp_get_max_threads(), t1 - t0);

    FILE *out = fopen("results.csv", "w");
    if (!out) { perror("results.csv"); return 1; }
    fprintf(out, "id,ytm,duration,convexity\n");
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(ytm[i])) bad++;
        fprintf(out, "%d,%.12f,%.6f,%.6f\n", i, ytm[i], dur[i], cvx[i]);
    }
    fclose(out);
    double t_end = omp_get_wtime();
    printf("e2e total: %.6f s (read %.6f, compute %.6f, write %.6f)\n",
           t_end - t_start, t_read - t_start, t1 - t0, t_end - t1);
    if (bad) printf("warning: %d bonds did not converge (nan/inf)\n", bad);

    /* free the memory */
    free(coupon); free(face); free(years); free(price);
    free(ytm); free(dur); free(cvx);
    return 0;
}