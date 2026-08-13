#define MAT_MUL_NO_MAIN 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define main mat_mul_original_main
#include "mat_mul.c"
#undef main

static double max_err(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n * n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    int sizes[] = {32, 63, 64, 65, 128, 129, 256};
    int tiles[] = {8, 16, 32};

    int ok = 1;
    for (unsigned s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        double *A = (double*)malloc(n*n*sizeof(double));
        double *B = (double*)malloc(n*n*sizeof(double));
        double *Cref = (double*)calloc(n*n, sizeof(double));
        double *C = (double*)malloc(n*n*sizeof(double));

        srand(1);
        for (int i = 0; i < n*n; i++) A[i] = (double)rand()/RAND_MAX;
        for (int i = 0; i < n*n; i++) B[i] = (double)rand()/RAND_MAX;

        naive_mat_mul(A, B, Cref, n);

        memset(C, 0, n*n*sizeof(double)); loop_reorder_mat_mul(A, B, C, n);
        double e = max_err(Cref, C, n);
        printf("n=%3d reorder     max_err=%.3g %s\n", n, e, e < 1e-8 ? "OK" : "FAIL");
        if (e >= 1e-8) ok = 0;

        memset(C, 0, n*n*sizeof(double)); loop_unroll_mat_mul(A, B, C, n);
        e = max_err(Cref, C, n);
        printf("n=%3d unroll      max_err=%.3g %s\n", n, e, e < 1e-8 ? "OK" : "FAIL");
        if (e >= 1e-8) ok = 0;

        memset(C, 0, n*n*sizeof(double)); loop_opt_mat_mul(A, B, C, n);
        e = max_err(Cref, C, n);
        printf("n=%3d loop        max_err=%.3g %s\n", n, e, e < 1e-8 ? "OK" : "FAIL");
        if (e >= 1e-8) ok = 0;

        memset(C, 0, n*n*sizeof(double)); simd_mat_mul_128(A, B, C, n);
        e = max_err(Cref, C, n);
        printf("n=%3d simd128     max_err=%.3g %s\n", n, e, e < 1e-8 ? "OK" : "FAIL");
        if (e >= 1e-8) ok = 0;

        memset(C, 0, n*n*sizeof(double)); simd_mat_mul_256(A, B, C, n);
        e = max_err(Cref, C, n);
        printf("n=%3d simd256     max_err=%.3g %s\n", n, e, e < 1e-8 ? "OK" : "FAIL");
        if (e >= 1e-8) ok = 0;

#ifdef __AVX512F__
        memset(C, 0, n*n*sizeof(double)); simd_mat_mul_512(A, B, C, n);
        e = max_err(Cref, C, n);
        printf("n=%3d simd512     max_err=%.3g %s\n", n, e, e < 1e-8 ? "OK" : "FAIL");
        if (e >= 1e-8) ok = 0;
#endif

        for (unsigned t = 0; t < sizeof(tiles)/sizeof(tiles[0]); t++) {
            int T = tiles[t];
            memset(C, 0, n*n*sizeof(double)); tile_mat_mul(A, B, C, n, T);
            e = max_err(Cref, C, n);
            printf("n=%3d tile   T=%2d max_err=%.3g %s\n", n, T, e, e < 1e-8 ? "OK" : "FAIL");
            if (e >= 1e-8) ok = 0;

            memset(C, 0, n*n*sizeof(double)); combination_mat_mul_128(A, B, C, n, T);
            e = max_err(Cref, C, n);
            printf("n=%3d combo128 T=%2d max_err=%.3g %s\n", n, T, e, e < 1e-8 ? "OK" : "FAIL");
            if (e >= 1e-8) ok = 0;

            memset(C, 0, n*n*sizeof(double)); combination_mat_mul_256(A, B, C, n, T);
            e = max_err(Cref, C, n);
            printf("n=%3d combo256 T=%2d max_err=%.3g %s\n", n, T, e, e < 1e-8 ? "OK" : "FAIL");
            if (e >= 1e-8) ok = 0;

#ifdef __AVX512F__
            memset(C, 0, n*n*sizeof(double)); combination_mat_mul_512(A, B, C, n, T);
            e = max_err(Cref, C, n);
            printf("n=%3d combo512 T=%2d max_err=%.3g %s\n", n, T, e, e < 1e-8 ? "OK" : "FAIL");
            if (e >= 1e-8) ok = 0;
#endif
        }
        free(A); free(B); free(C); free(Cref);
    }
    return ok ? 0 : 1;
}
