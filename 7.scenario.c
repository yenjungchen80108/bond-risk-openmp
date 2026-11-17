/* 7.scenario.c  --  exact repricing vs duration-convexity approximation.
 *
 * Question: once parallel hardware makes exact repricing cheap, is the
 * approximation  dP/P ~ -D*dy + 0.5*C*dy^2  still worth using?
 *
 * Every bond's yield is shifted in parallel by dy (the same dy for all bonds):
 *   Method A (exact)        : price every bond again at ytm + dy.      cost N x S
 *   Method B (approx.)      : dP/P from D and C of each bond.          cost N x S (cheap arithmetic)
 *   Method B-agg (approx.)  : portfolio dollar-duration and dollar-convexity are summed
 *                             once (cost N); each extra scenario is O(1).   cost N + S
 *
 * Part 1 (accuracy):  error of B against A for 8 shocks, by maturity bucket.
 * Part 2 (timing):    running time of A (1 thread and all threads), B and B-agg
 *                     for growing numbers of scenarios S, plus the one-time cost of computing
 *                     D and C for every bond (needed by B and B-agg, not by A) and the
 *                     break-even S.  The YTM solve is shared by all methods and not counted.
 *                     File I/O is not timed here.
 *
 * Needs bonds.csv (make data N=...).  Writes scenario_accuracy.csv and scenario_timing.csv.
 * Usage:  ./scenario [N_time]     N_time = bonds used in Part 2 (default 100000)
 *         OMP_NUM_THREADS=8 ./scenario
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <omp.h>
 #include "1.ytm.h"
 #include "2.duration_convexity.h"
 
 double bond_price(double r, double coupon, double face, int years);
 
 #define NSHOCK 8
 static const double SHOCKS[NSHOCK] = {-0.02, -0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01, 0.02};
 
 #define NBUCKET 4
 static const char *BUCKET_NAME[NBUCKET] = {"1-5y", "6-10y", "11-20y", "21y+"};
 static int bucket_of(int years) {
     return years <= 5 ? 0 : years <= 10 ? 1 : years <= 20 ? 2 : 3;
 }
 
 /* ---- the three methods; each returns the portfolio value change for one shock dy ---- */
 
 /* A: exact repricing of every bond at ytm + dy. */
 static double scenario_exact(int n, const double *coupon, const double *face, const int *years,
                              const double *ytm, const double *p0, double dy) {
     double sum = 0.0;
     #pragma omp parallel for schedule(static) reduction(+:sum)
     for (int i = 0; i < n; i++)
         sum += bond_price(ytm[i] + dy, coupon[i], face[i], years[i]) - p0[i];
     return sum;
 }
 
 /* B: approximation applied bond by bond. */
 static double scenario_approx(int n, const double *p0, const double *dur, const double *cvx, double dy) {
     double sum = 0.0;
     #pragma omp parallel for schedule(static) reduction(+:sum)
     for (int i = 0; i < n; i++)
         sum += p0[i] * (-dur[i] * dy + 0.5 * cvx[i] * dy * dy);
     return sum;
 }
 
 /* B-agg: approximation from portfolio-level sums. dD = sum(P*D), dC = sum(P*C) are
    computed once (cost N); each scenario then costs O(1). */
 static void aggregate_DC(int n, const double *p0, const double *dur, const double *cvx,
                          double *dD, double *dC) {
     double sd = 0.0, sc = 0.0;
     for (int i = 0; i < n; i++) { sd += p0[i] * dur[i]; sc += p0[i] * cvx[i]; }
     *dD = sd; *dC = sc;
 }
 static double scenario_agg(double dD, double dC, double dy) {
     return -dD * dy + 0.5 * dC * dy * dy;
 }
 
 /* D and C for every bond (3 bond prices per bond). */
 static void compute_DC(int n, const double *coupon, const double *face, const int *years,
                        const double *ytm, double *dur, double *cvx) {
     #pragma omp parallel for schedule(dynamic, 1000)
     for (int i = 0; i < n; i++) {
         dur[i] = compute_duration(ytm[i], coupon[i], face[i], years[i]);
         cvx[i] = compute_convexity(ytm[i], coupon[i], face[i], years[i]);
     }
 }
 
 static double median3(double a, double b, double c) {
     if (a > b) { double t = a; a = b; b = t; }
     if (b > c) { double t = b; b = c; c = t; }
     if (a > b) { double t = a; a = b; b = t; }
     return b;
 }
 
 #define REPS 3
 /* Run f() REPS times and return the median wall-clock time. */
 #define TIME_MEDIAN(result, ...)                                    \
     do {                                                            \
         double tt[REPS];                                            \
         for (int r_ = 0; r_ < REPS; r_++) {                         \
             double t0_ = omp_get_wtime();                           \
             __VA_ARGS__;                                            \
             tt[r_] = omp_get_wtime() - t0_;                         \
         }                                                           \
         (result) = median3(tt[0], tt[1], tt[2]);                    \
     } while (0)
 
 int main(int argc, char **argv) {
     int n_time_req = argc > 1 ? atoi(argv[1]) : 100000;
 
     /* ---- read bonds.csv ---- */
     FILE *fp = fopen("bonds.csv", "r");
     if (!fp) { perror("bonds.csv"); return 1; }
     int cap = 1 << 20, n = 0;
     double *coupon = malloc(cap * sizeof(double));
     double *face   = malloc(cap * sizeof(double));
     int    *years  = malloc(cap * sizeof(int));
     double *price  = malloc(cap * sizeof(double));
     char line[256];
     int id;
     if (!fgets(line, sizeof line, fp)) { fprintf(stderr, "empty bonds.csv\n"); return 1; }
     while (n < cap && fgets(line, sizeof line, fp))
         if (sscanf(line, "%d,%lf,%lf,%d,%lf", &id, &coupon[n], &face[n], &years[n], &price[n]) == 5) n++;
     fclose(fp);
 
     /* ---- shared starting point: ytm, duration, convexity, current price ---- */
     double *ytm = malloc(n * sizeof(double));
     double *dur = malloc(n * sizeof(double));
     double *cvx = malloc(n * sizeof(double));
     double *p0  = malloc(n * sizeof(double));
     #pragma omp parallel for schedule(dynamic, 1000)
     for (int i = 0; i < n; i++) {
         ytm[i] = compute_irr_newton(coupon[i], price[i], face[i], years[i]);
         dur[i] = compute_duration(ytm[i], coupon[i], face[i], years[i]);
         cvx[i] = compute_convexity(ytm[i], coupon[i], face[i], years[i]);
         p0[i]  = bond_price(ytm[i], coupon[i], face[i], years[i]);
     }
     int nthreads = omp_get_max_threads();
     printf("bonds: %d, threads available: %d\n\n", n, nthreads);
 
     /* =========================== Part 1: accuracy =========================== */
     /* error of one bond = | approx dP/P - exact dP/P |, in percentage points of the price */
     FILE *fa = fopen("scenario_accuracy.csv", "w");
     if (!fa) { perror("scenario_accuracy.csv"); return 1; }
     fprintf(fa, "shock_pct,bucket,n_bonds,mean_exact_pct,mean_abs_err_pct,max_abs_err_pct\n");
 
     double agg_dD, agg_dC, v0 = 0.0, max_agg_diff = 0.0;
     aggregate_DC(n, p0, dur, cvx, &agg_dD, &agg_dC);
     for (int i = 0; i < n; i++) v0 += p0[i];
 
     double *err = malloc(n * sizeof(double));
     double *ex  = malloc(n * sizeof(double));
     printf("Part 1. Approximation error vs exact repricing (percentage points of price)\n");
     printf("  error = |approx dP/P - exact dP/P|;  mean over the bonds in each maturity bucket\n\n");
     printf("%8s |", "shock");
     for (int b = 0; b < NBUCKET; b++) printf(" %9s |", BUCKET_NAME[b]);
     printf(" %9s | %11s\n", "all", "portfolio");
     printf("---------+");
     for (int b = 0; b < NBUCKET + 1; b++) printf("-----------+");
     printf("-------------\n");
 
     for (int s = 0; s < NSHOCK; s++) {
         double dy = SHOCKS[s];
         #pragma omp parallel for schedule(static)
         for (int i = 0; i < n; i++) {
             double exact  = (bond_price(ytm[i] + dy, coupon[i], face[i], years[i]) - p0[i]) / p0[i];
             double approx = -dur[i] * dy + 0.5 * cvx[i] * dy * dy;
             ex[i]  = exact * 100.0;
             err[i] = fabs(approx - exact) * 100.0;
         }
         double sum_e[NBUCKET + 1] = {0}, sum_x[NBUCKET + 1] = {0}, max_e[NBUCKET + 1] = {0};
         long cnt[NBUCKET + 1] = {0};
         for (int i = 0; i < n; i++) {           /* cheap serial aggregation */
             int b = bucket_of(years[i]);
             for (int k = 0; k < 2; k++) {
                 int idx = k ? NBUCKET : b;       /* own bucket, then the "all" column */
                 sum_e[idx] += err[i]; sum_x[idx] += ex[i]; cnt[idx]++;
                 if (err[i] > max_e[idx]) max_e[idx] = err[i];
             }
         }
         /* portfolio level: error of the total value change (dollar-weighted, errors can cancel) */
         double dv_exact  = scenario_exact(n, coupon, face, years, ytm, p0, dy);
         double dv_approx = scenario_approx(n, p0, dur, cvx, dy);
         double dv_agg = scenario_agg(agg_dD, agg_dC, dy);
         double agg_diff = fabs(dv_agg - dv_approx) / v0;   /* B-agg must equal B */
         if (agg_diff > max_agg_diff) max_agg_diff = agg_diff;
         double port_err = fabs(dv_approx - dv_exact) / v0 * 100.0;
 
         printf("%+7.2f%% |", dy * 100.0);
         for (int b = 0; b < NBUCKET; b++)
             printf(" %9.5f |", cnt[b] ? sum_e[b] / cnt[b] : 0.0);
         printf(" %9.5f | %11.5f\n", sum_e[NBUCKET] / cnt[NBUCKET], port_err);
 
         for (int b = 0; b <= NBUCKET; b++)
             if (cnt[b])
                 fprintf(fa, "%.4f,%s,%ld,%.6f,%.8f,%.8f\n", dy * 100.0,
                         b < NBUCKET ? BUCKET_NAME[b] : "all", cnt[b],
                         sum_x[b] / cnt[b], sum_e[b] / cnt[b], max_e[b]);
     }
     fclose(fa);
     printf("\n  (portfolio column: |approx - exact| of the total value change, as %% of portfolio value)\n");
     printf("  check: B-agg vs B (per bond), largest difference over all shocks = %.2e of portfolio value %s\n",
            max_agg_diff, max_agg_diff < 1e-9 ? "(identical up to rounding)" : "(WARNING: differ)");
     printf("wrote scenario_accuracy.csv\n\n");
 
     /* ============================ Part 2: timing ============================ */
     int nt = n < n_time_req ? n : n_time_req;
     int S_list[] = {8, 64, 256};
     int nS = sizeof S_list / sizeof S_list[0];
     FILE *ft = fopen("scenario_timing.csv", "w");
     if (!ft) { perror("scenario_timing.csv"); return 1; }
     fprintf(ft, "N,S,method,threads,time_s\n");
 
     printf("Part 2. Running time, N = %d bonds (median of %d runs; file I/O not included)\n\n", nt, REPS);
 
     /* One-time cost of D and C for all bonds (3 bond prices per bond, central differences). */
     double *dur2 = malloc(nt * sizeof(double)), *cvx2 = malloc(nt * sizeof(double));
     double tDC1, tDCp;
     volatile double sink = 0.0;
     omp_set_num_threads(1);
     TIME_MEDIAN(tDC1, {
         compute_DC(nt, coupon, face, years, ytm, dur2, cvx2);
         sink += dur2[nt - 1]; });
     omp_set_num_threads(nthreads);
     TIME_MEDIAN(tDCp, {
         compute_DC(nt, coupon, face, years, ytm, dur2, cvx2);
         sink += dur2[nt - 1]; });
     printf("One-time cost of D and C for all %d bonds: %.4f s (1 thread), %.4f s (%d threads)\n\n",
            nt, tDC1, tDCp, nthreads);
     fprintf(ft, "%d,0,DC_cost,1,%.6f\n", nt, tDC1);
     fprintf(ft, "%d,0,DC_cost,%d,%.6f\n", nt, nthreads, tDCp);
 
     printf("%5s | %14s | %14s | %9s | %12s | %16s | %14s\n", "S",
            "A exact 1 thr", "A exact all", "B approx", "B-agg", "B-agg + D/C", "A(all)/(B+D/C)");
     printf("------+----------------+----------------+-----------+--------------+------------------+---------------\n");
     double a1 = 0.0, ap = 0.0;   /* exact cost per scenario, from the last row */
 
     for (int k = 0; k < nS; k++) {
         int S = S_list[k];
         double *dys = malloc(S * sizeof(double));
         for (int s = 0; s < S; s++) dys[s] = -0.02 + 0.04 * (S == 1 ? 0.5 : (double)s / (S - 1));
         double *out = malloc(S * sizeof(double));
         double tA1, tAp, tB, tBagg;
 
         omp_set_num_threads(1);
         TIME_MEDIAN(tA1, { for (int s = 0; s < S; s++)
             out[s] = scenario_exact(nt, coupon, face, years, ytm, p0, dys[s]); sink += out[S - 1]; });
         omp_set_num_threads(nthreads);
         TIME_MEDIAN(tAp, { for (int s = 0; s < S; s++)
             out[s] = scenario_exact(nt, coupon, face, years, ytm, p0, dys[s]); sink += out[S - 1]; });
         TIME_MEDIAN(tB, { for (int s = 0; s < S; s++)
             out[s] = scenario_approx(nt, p0, dur, cvx, dys[s]); sink += out[S - 1]; });
         /* B-agg: one O(N) pass for dollar duration/convexity, then O(1) per scenario.
            The O(N) pass is counted in the time. */
         TIME_MEDIAN(tBagg, {
             double dD, dC;
             aggregate_DC(nt, p0, dur, cvx, &dD, &dC);
             for (int s = 0; s < S; s++) out[s] = scenario_agg(dD, dC, dys[s]);
             sink += out[S - 1]; });
 
         double tBtot = tBagg + tDCp;          /* B-agg including the one-time D/C cost (all threads) */
         printf("%5d | %12.4f s | %12.4f s | %7.4f s | %10.6f s | %14.4f s | %13.1fx\n",
                S, tA1, tAp, tB, tBagg, tBtot, tAp / tBtot);
         a1 = tA1 / S; ap = tAp / S;
         fprintf(ft, "%d,%d,A_exact,1,%.6f\n", nt, S, tA1);
         fprintf(ft, "%d,%d,A_exact,%d,%.6f\n", nt, S, nthreads, tAp);
         fprintf(ft, "%d,%d,B_approx,%d,%.6f\n", nt, S, nthreads, tB);
         fprintf(ft, "%d,%d,B_agg,1,%.6f\n", nt, S, tBagg);
         free(dys); free(out);
     }
     fclose(ft);
     printf("\nBreak-even number of scenarios (exact repricing costs the same as computing D and C once):\n");
     printf("  1 thread : S* = %.1f   (D/C %.4f s / exact %.6f s per scenario)\n", tDC1 / a1, tDC1, a1);
     printf("  %d threads: S* = %.1f   (D/C %.4f s / exact %.6f s per scenario)\n", nthreads, tDCp / ap, tDCp, ap);
     printf("  For S below S*, exact repricing is already cheaper than preparing the approximation.\n");
     printf("\nwrote scenario_timing.csv  (checksum %.3e, ignore)\n", (double)sink);
 
     free(coupon); free(face); free(years); free(price);
     free(ytm); free(dur); free(cvx); free(p0); free(err); free(ex); free(dur2); free(cvx2);
     return 0;
 }