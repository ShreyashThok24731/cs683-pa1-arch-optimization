#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <xmmintrin.h>
#include <immintrin.h>
#pragma GCC target("avx2","sse2","fma")

#include "helper.h"

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

void naive_mat_mul(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void loop_opt_mat_mul(double *A, double *B, double *C, int size) {
    int i, j, k;
    int j_end = size - (size % 4);
    int k_end = size - (size % 4);

    for (i = 0; i < size; i++) {
        for (j = 0; j < j_end; j += 4) {
            double c0 = C[i * size + j];
            double c1 = C[i * size + j + 1];
            double c2 = C[i * size + j + 2];
            double c3 = C[i * size + j + 3];

            for (k = 0; k < k_end; k += 4) {
                double a0 = A[i * size + k];
                double a1 = A[i * size + k + 1];
                double a2 = A[i * size + k + 2];
                double a3 = A[i * size + k + 3];

                c0 += a0 * B[(k    ) * size + j    ];
                c1 += a0 * B[(k    ) * size + j + 1];
                c2 += a0 * B[(k    ) * size + j + 2];
                c3 += a0 * B[(k    ) * size + j + 3];

                c0 += a1 * B[(k + 1) * size + j    ];
                c1 += a1 * B[(k + 1) * size + j + 1];
                c2 += a1 * B[(k + 1) * size + j + 2];
                c3 += a1 * B[(k + 1) * size + j + 3];

                c0 += a2 * B[(k + 2) * size + j    ];
                c1 += a2 * B[(k + 2) * size + j + 1];
                c2 += a2 * B[(k + 2) * size + j + 2];
                c3 += a2 * B[(k + 2) * size + j + 3];

                c0 += a3 * B[(k + 3) * size + j    ];
                c1 += a3 * B[(k + 3) * size + j + 1];
                c2 += a3 * B[(k + 3) * size + j + 2];
                c3 += a3 * B[(k + 3) * size + j + 3];
            }
            for (; k < size; k++) {
                double a = A[i * size + k];
                c0 += a * B[k * size + j    ];
                c1 += a * B[k * size + j + 1];
                c2 += a * B[k * size + j + 2];
                c3 += a * B[k * size + j + 3];
            }
            C[i * size + j    ] = c0;
            C[i * size + j + 1] = c1;
            C[i * size + j + 2] = c2;
            C[i * size + j + 3] = c3;
        }
        for (; j < size; j++) {
            double c = C[i * size + j];
            for (k = 0; k < size; k++) {
                c += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = c;
        }
    }
}

void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int k = 0; k < size; k += tile_size) {

                int i_max = (i + tile_size < size) ? i + tile_size : size;
                int j_max = (j + tile_size < size) ? j + tile_size : size;
                int k_max = (k + tile_size < size) ? k + tile_size : size;

                for (int ii = i; ii < i_max; ii++) {
                    for (int jj = j; jj < j_max; jj++) {
                        double t = C[ii * size + jj];
                        for (int kk = k; kk < k_max; kk++) {
                            t += A[ii * size + kk] * B[kk * size + jj];
                        }
                        C[ii * size + jj] = t;
                    }
                }
            }
        }
    }
}

void simd_mat_mul(double *A, double *B, double *C, int size) {
    const int V = 4;
    int j_end = size - (size % V);

    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            __m256d a = _mm256_broadcast_sd(&A[i * size + k]);
            int j;
            for (j = 0; j < j_end; j += V) {
                __m256d b = _mm256_loadu_pd(&B[k * size + j]);
                __m256d c = _mm256_loadu_pd(&C[i * size + j]);
                _mm256_storeu_pd(&C[i * size + j], _mm256_fmadd_pd(a, b, c));
            }
            for (; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int k = 0; k < size; k += tile_size) {

                int i_max = (i + tile_size < size) ? i + tile_size : size;
                int j_max = (j + tile_size < size) ? j + tile_size : size;
                int k_max = (k + tile_size < size) ? k + tile_size : size;
                int j_vec_end = j + ((j_max - j) & ~7);

                for (int ii = i; ii < i_max; ii++) {
                    int jj;
                    for (jj = j; jj < j_vec_end; jj += 8) {
                        __m256d c0 = _mm256_loadu_pd(&C[ii * size + jj    ]);
                        __m256d c1 = _mm256_loadu_pd(&C[ii * size + jj + 4]);

                        int kk;
                        int kk_unroll_end = k + ((k_max - k) & ~3);
                        for (kk = k; kk < kk_unroll_end; kk += 4) {
                            __m256d a0 = _mm256_broadcast_sd(&A[ii * size + kk    ]);
                            __m256d a1 = _mm256_broadcast_sd(&A[ii * size + kk + 1]);
                            __m256d a2 = _mm256_broadcast_sd(&A[ii * size + kk + 2]);
                            __m256d a3 = _mm256_broadcast_sd(&A[ii * size + kk + 3]);

                            __m256d b0 = _mm256_loadu_pd(&B[(kk    ) * size + jj    ]);
                            __m256d b1 = _mm256_loadu_pd(&B[(kk    ) * size + jj + 4]);
                            __m256d b2 = _mm256_loadu_pd(&B[(kk + 1) * size + jj    ]);
                            __m256d b3 = _mm256_loadu_pd(&B[(kk + 1) * size + jj + 4]);
                            __m256d b4 = _mm256_loadu_pd(&B[(kk + 2) * size + jj    ]);
                            __m256d b5 = _mm256_loadu_pd(&B[(kk + 2) * size + jj + 4]);
                            __m256d b6 = _mm256_loadu_pd(&B[(kk + 3) * size + jj    ]);
                            __m256d b7 = _mm256_loadu_pd(&B[(kk + 3) * size + jj + 4]);

                            c0 = _mm256_fmadd_pd(a0, b0, c0);
                            c1 = _mm256_fmadd_pd(a0, b1, c1);
                            c0 = _mm256_fmadd_pd(a1, b2, c0);
                            c1 = _mm256_fmadd_pd(a1, b3, c1);
                            c0 = _mm256_fmadd_pd(a2, b4, c0);
                            c1 = _mm256_fmadd_pd(a2, b5, c1);
                            c0 = _mm256_fmadd_pd(a3, b6, c0);
                            c1 = _mm256_fmadd_pd(a3, b7, c1);
                        }
                        for (; kk < k_max; kk++) {
                            __m256d a = _mm256_broadcast_sd(&A[ii * size + kk]);
                            __m256d b0 = _mm256_loadu_pd(&B[kk * size + jj    ]);
                            __m256d b1 = _mm256_loadu_pd(&B[kk * size + jj + 4]);
                            c0 = _mm256_fmadd_pd(a, b0, c0);
                            c1 = _mm256_fmadd_pd(a, b1, c1);
                        }

                        _mm256_storeu_pd(&C[ii * size + jj    ], c0);
                        _mm256_storeu_pd(&C[ii * size + jj + 4], c1);
                    }
                    for (; jj < j_max; jj++) {
                        double t = C[ii * size + jj];
                        for (int kk = k; kk < k_max; kk++) {
                            t += A[ii * size + kk] * B[kk * size + jj];
                        }
                        C[ii * size + jj] = t;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {

    if (argc <= 1) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 0;
    }

    int size = atoi(argv[1]);

    double *A = (double *)malloc(size * size * sizeof(double));
    double *B = (double *)malloc(size * size * sizeof(double));
    double *C = (double *)calloc(size * size, sizeof(double));

    srand(time(NULL));
    initialize_matrix(A, size, size);
    initialize_matrix(B, size, size);

#ifdef OPTIMIZE_NAIVE
    initialize_result_matrix(C, size, size);
    {
        auto start = std::chrono::high_resolution_clock::now();
        naive_mat_mul(A, B, C, size);
        auto end = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Naive matrix multiplication took %ld ms to execute \n", t);
    }
#endif

#ifdef OPTIMIZE_LOOP_OPT
    initialize_result_matrix(C, size, size);
    {
        auto start = std::chrono::high_resolution_clock::now();
        loop_opt_mat_mul(A, B, C, size);
        auto end = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Loop optimized matrix multiplication took %ld ms to execute \n", t);
    }
#endif

#ifdef OPTIMIZE_TILING
    initialize_result_matrix(C, size, size);
    {
        auto start = std::chrono::high_resolution_clock::now();
        tile_mat_mul(A, B, C, size, TILE_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Tiling matrix multiplication took %ld ms to execute \n", t);
    }
#endif

#ifdef OPTIMIZE_SIMD
    initialize_result_matrix(C, size, size);
    {
        auto start = std::chrono::high_resolution_clock::now();
        simd_mat_mul(A, B, C, size);
        auto end = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("SIMD matrix multiplication took %ld ms to execute \n", t);
    }
#endif

#ifdef OPTIMIZE_COMBINED
    initialize_result_matrix(C, size, size);
    {
        auto start = std::chrono::high_resolution_clock::now();
        combination_mat_mul(A, B, C, size, TILE_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Combined optimization matrix multiplication took %ld ms to execute \n", t);
    }
#endif

    free(A);
    free(B);
    free(C);
    return 0;
}
