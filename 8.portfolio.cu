/* 8.portfolio.cu -- CUDA version of 3.portfolio.c: one GPU thread per bond.
 *
 * Reads bonds.csv, solves YTM (Newton), duration and convexity for every bond on the GPU,
 * and writes results_gpu.csv (same format as results.csv from the OpenMP program).
 * The device code is the same algorithm as 1.ytm.c / 2.duration_convexity.c, in double precision.
 *
 * Timing printed:  host->device copy, kernel, device->host copy, and end-to-end
 * (read + CUDA init + h2d + kernel + d2h + write), so that the GPU can be put into the same
 * Amdahl's-law analysis as the OpenMP program.
 * Also printed: histogram of Newton iterations per bond and the "warp efficiency", i.e. how much
 * of the work done by the 32 threads of a warp is useful when each thread needs a different
 * number of iterations (load imbalance / thread divergence).
 * If results.csv (OpenMP/CPU result) exists, the GPU result is compared with it.
 *
 * Build (Discovery):  module purge; module load usc; module load cuda
 *                     nvcc -O2 -o portfolio_gpu 8.portfolio.cu
 * Run:                ./portfolio_gpu [threads_per_block]      (default 256)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static double wtime(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---------------- device code: same formulas as the CPU version ---------------- */
__device__ double d_bond_price(double r, double coupon, double face, int years) {
    double p = 0.0;
    for (int t = 1; t <= years; t++) p += coupon / pow(1.0 + r, t);
    p += face / pow(1.0 + r, years);
    return p;
}

__device__ double d_bond_price_derivative(double r, double coupon, double face, int years) {
    double dp = 0.0;
    for (int t = 1; t <= years; t++) dp -= t * coupon / pow(1.0 + r, t + 1);
    dp -= years * face / pow(1.0 + r, years + 1);
    return dp;
}

__device__ double d_irr_newton(double coupon, double price, double face, int years, int *iters) {
    double r = 0.05, tol = 1e-8;
    int max_iter = 100, k;
    for (k = 0; k < max_iter; k++) {
        double f  = d_bond_price(r, coupon, face, years) - price;
        double fp = d_bond_price_derivative(r, coupon, face, years);
        if (fabs(fp) < 1e-12) break;
        double r_next = r - f / fp;
        if (fabs(r_next - r) < tol) { *iters = k + 1; return r_next; }
        r = r_next;
    }
    *iters = k;
    return r;
}

__device__ double d_duration(double ytm, double coupon, double face, int years) {
    double h = 1e-4;
    double p0 = d_bond_price(ytm, coupon, face, years);
    double pm = d_bond_price(ytm - h, coupon, face, years);
    double pp = d_bond_price(ytm + h, coupon, face, years);
    return (pm - pp) / (2.0 * p0 * h);
}

__device__ double d_convexity(double ytm, double coupon, double face, int years) {
    double h = 1e-4;
    double p0 = d_bond_price(ytm, coupon, face, years);
    double pm = d_bond_price(ytm - h, coupon, face, years);
    double pp = d_bond_price(ytm + h, coupon, face, years);
    return (pm + pp - 2.0 * p0) / (p0 * h * h);
}

/* One thread = one bond. */
__global__ void bond_kernel(int n, const double *coupon, const double *face, const int *years,
                            const double *price, double *ytm, double *dur, double *cvx, int *iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int it = 0;
    double y = d_irr_newton(coupon[i], price[i], face[i], years[i], &it);
    ytm[i]   = y;
    dur[i]   = d_duration(y, coupon[i], face[i], years[i]);
    cvx[i]   = d_convexity(y, coupon[i], face[i], years[i]);
    iters[i] = it;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int block = argc > 1 ? atoi(argv[1]) : 256;
    if (block <= 0) block = 256;
    const int REPS = 5;

    double t_start = wtime();
    /* ---------- read bonds.csv (host, sequential, same as the OpenMP program) ---------- */
    FILE *fp = fopen("bonds.csv", "r");
    if (!fp) { perror("bonds.csv"); return 1; }
    int cap = 1 << 20, n = 0;
    double *coupon = (double *)malloc(cap * sizeof(double));
    double *face   = (double *)malloc(cap * sizeof(double));
    int    *years  = (int *)malloc(cap * sizeof(int));
    double *price  = (double *)malloc(cap * sizeof(double));
    char line[256];
    int id;
    if (!fgets(line, sizeof line, fp)) { fprintf(stderr, "empty bonds.csv\n"); return 1; }
    while (n < cap && fgets(line, sizeof line, fp))
        if (sscanf(line, "%d,%lf,%lf,%d,%lf", &id, &coupon[n], &face[n], &years[n], &price[n]) == 5) n++;
    fclose(fp);
    double t_read = wtime();

    double *ytm = (double *)malloc(n * sizeof(double));
    double *dur = (double *)malloc(n * sizeof(double));
    double *cvx = (double *)malloc(n * sizeof(double));
    int *iters  = (int *)malloc(n * sizeof(int));

    /* ---------- CUDA initialisation (first runtime call creates the context) ---------- */
    CUDA_CHECK(cudaFree(0));
    double t_init = wtime();

    /* ---------- allocate device memory and copy the inputs (host -> device) ---------- */
    double *d_coupon, *d_face, *d_price, *d_ytm, *d_dur, *d_cvx;
    int *d_years, *d_iters;
    CUDA_CHECK(cudaMalloc((void **)&d_coupon, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_face,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_price,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_years,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_ytm,    n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_dur,    n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_cvx,    n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_iters,  n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_coupon, coupon, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_face,   face,   n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_price,  price,  n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_years,  years,  n * sizeof(int),    cudaMemcpyHostToDevice));
    double t_h2d = wtime();

    /* ---------- kernel: first launch is the one counted in the end-to-end time ---------- */
    int blocks = (n + block - 1) / block;
    bond_kernel<<<blocks, block>>>(n, d_coupon, d_face, d_years, d_price, d_ytm, d_dur, d_cvx, d_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_kernel = wtime();

    /* ---------- copy the results back (device -> host) ---------- */
    CUDA_CHECK(cudaMemcpy(ytm,   d_ytm,   n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dur,   d_dur,   n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cvx,   d_cvx,   n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(iters, d_iters, n * sizeof(int),    cudaMemcpyDeviceToHost));
    double t_d2h = wtime();

    /* ---------- write results_gpu.csv ---------- */
    FILE *out = fopen("results_gpu.csv", "w");
    if (!out) { perror("results_gpu.csv"); return 1; }
    fprintf(out, "id,ytm,duration,convexity\n");
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(ytm[i])) bad++;
        fprintf(out, "%d,%.12f,%.6f,%.6f\n", i, ytm[i], dur[i], cvx[i]);
    }
    fclose(out);
    double t_end = wtime();

    /* ---------- steady-state kernel time: REPS more launches, median (after the first one) ---------- */
    double tk[REPS];
    for (int r = 0; r < REPS; r++) {
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        bond_kernel<<<blocks, block>>>(n, d_coupon, d_face, d_years, d_price, d_ytm, d_dur, d_cvx, d_iters);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        tk[r] = ms * 1e-3;
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    }
    qsort(tk, REPS, sizeof(double), cmp_double);

    cudaDeviceProp prop;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, bonds: %d, threads per block: %d, blocks: %d\n", prop.name, n, block, blocks);
    printf("kernel (median of %d launches): %.6f s\n", REPS, tk[REPS / 2]);
    printf("e2e total: %.6f s (read %.6f, init %.6f, h2d %.6f, kernel %.6f, d2h %.6f, write %.6f)\n",
           t_end - t_start, t_read - t_start, t_init - t_read, t_h2d - t_init,
           t_kernel - t_h2d, t_d2h - t_kernel, t_end - t_d2h);
    if (bad) printf("warning: %d bonds did not converge (nan/inf)\n", bad);

    /* ---------- load imbalance: Newton iterations per bond and warp efficiency ---------- */
    int maxit = 0; long total = 0;
    for (int i = 0; i < n; i++) { if (iters[i] > maxit) maxit = iters[i]; total += iters[i]; }
    long *hist = (long *)calloc(maxit + 1, sizeof(long));
    for (int i = 0; i < n; i++) hist[iters[i]]++;
    long warp_work = 0;                       /* iterations paid by all lanes of every warp */
    for (int w = 0; w < n; w += 32) {
        int m = 0;
        for (int i = w; i < w + 32 && i < n; i++) if (iters[i] > m) m = iters[i];
        int lanes = (w + 32 <= n) ? 32 : n - w;
        warp_work += (long)m * lanes;
    }
    printf("Newton iterations per bond: mean %.2f, max %d\n", (double)total / n, maxit);
    printf("iterations histogram:");
    for (int k = 1; k <= maxit; k++) if (hist[k]) printf(" %d:%ld", k, hist[k]);
    printf("\nwarp efficiency (useful iterations / iterations paid by the warp): %.1f%%\n",
           100.0 * (double)total / (double)warp_work);

    /* ---------- correctness: compare with the CPU result, if results.csv is there ---------- */
    FILE *rc = fopen("results.csv", "r");
    if (rc) {
        double dy = 0, dd = 0, dc = 0;
        int cnt = 0;
        if (fgets(line, sizeof line, rc)) {
            while (fgets(line, sizeof line, rc)) {
                int j; double y, d, c;
                if (sscanf(line, "%d,%lf,%lf,%lf", &j, &y, &d, &c) != 4 || j < 0 || j >= n) continue;
                if (fabs(y - ytm[j]) > dy) dy = fabs(y - ytm[j]);
                if (fabs(d - dur[j]) > dd) dd = fabs(d - dur[j]);
                if (fabs(c - cvx[j]) > dc) dc = fabs(c - cvx[j]);
                cnt++;
            }
        }
        fclose(rc);
        printf("compared with results.csv (%d bonds): max |ytm diff| = %.3e, |duration diff| = %.3e, "
               "|convexity diff| = %.3e\n", cnt, dy, dd, dc);
    } else {
        printf("results.csv not found: run the OpenMP program first to check the GPU result\n");
    }

    cudaFree(d_coupon); cudaFree(d_face); cudaFree(d_price); cudaFree(d_years);
    cudaFree(d_ytm); cudaFree(d_dur); cudaFree(d_cvx); cudaFree(d_iters);
    free(coupon); free(face); free(years); free(price); free(ytm); free(dur); free(cvx); free(iters); free(hist);
    return 0;
}
