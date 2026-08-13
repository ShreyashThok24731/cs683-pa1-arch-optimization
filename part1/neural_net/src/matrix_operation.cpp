#include "matrix_operation.h"
#include <immintrin.h>
#pragma GCC target("avx2","fma","sse2")

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
    size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();
    if (k != B.getRows()) throw std::invalid_argument("shape");
    Matrix C(n, m);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            for (size_t l = 0; l < k; l++)
                C(i, j) += A(i, l) * B(l, j);
    return C;
}

Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();
    if (k != B.getRows()) throw std::invalid_argument("shape");
    Matrix C(n, m);
    for (size_t i = 0; i < n; i++)
        for (size_t q = 0; q < k; q++) {
            double a = A(i, q);
            for (size_t j = 0; j < m; j++) C(i, j) += a * B(q, j);
        }
    return C;
}

Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();
    if (k != B.getRows()) throw std::invalid_argument("shape");
    Matrix C(n, m);
    const size_t U = 4;
    size_t j_end = m - (m % U);

    for (size_t i = 0; i < n; i++) {
        size_t j;
        for (j = 0; j < j_end; j += U) {
            double c0 = C(i, j), c1 = C(i, j+1), c2 = C(i, j+2), c3 = C(i, j+3);
            size_t q;
            for (q = 0; q + 1 < k; q += 2) {
                double a0 = A(i, q), a1 = A(i, q+1);
                c0 += a0 * B(q, j    ) + a1 * B(q+1, j    );
                c1 += a0 * B(q, j + 1) + a1 * B(q+1, j + 1);
                c2 += a0 * B(q, j + 2) + a1 * B(q+1, j + 2);
                c3 += a0 * B(q, j + 3) + a1 * B(q+1, j + 3);
            }
            for (; q < k; q++) {
                double a = A(i, q);
                c0 += a * B(q, j    );
                c1 += a * B(q, j + 1);
                c2 += a * B(q, j + 2);
                c3 += a * B(q, j + 3);
            }
            C(i, j    ) = c0;
            C(i, j + 1) = c1;
            C(i, j + 2) = c2;
            C(i, j + 3) = c3;
        }
        for (; j < m; j++) {
            double c = C(i, j);
            for (size_t q = 0; q < k; q++) c += A(i, q) * B(q, j);
            C(i, j) = c;
        }
    }
    return C;
}

Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();
    if (k != B.getRows()) throw std::invalid_argument("shape");
    Matrix C(n, m);
    const size_t T = 32;
    for (size_t i0 = 0; i0 < n; i0 += T)
        for (size_t j0 = 0; j0 < m; j0 += T)
            for (size_t k0 = 0; k0 < k; k0 += T) {
                size_t i1 = std::min(i0 + T, n);
                size_t j1 = std::min(j0 + T, m);
                size_t k1 = std::min(k0 + T, k);
                for (size_t i = i0; i < i1; i++)
                    for (size_t q = k0; q < k1; q++) {
                        double a = A(i, q);
                        for (size_t j = j0; j < j1; j++)
                            C(i, j) += a * B(q, j);
                    }
            }
    return C;
}

Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();
    if (k != B.getRows()) throw std::invalid_argument("shape");
    Matrix C(n, m);

    const size_t T = 32;
    for (size_t i0 = 0; i0 < n; i0 += T) {
        for (size_t j0 = 0; j0 < m; j0 += T) {
            for (size_t k0 = 0; k0 < k; k0 += T) {
                size_t i1 = std::min(i0 + T, n);
                size_t j1 = std::min(j0 + T, m);
                size_t k1 = std::min(k0 + T, k);
                size_t j_vec_end = j0 + ((j1 - j0) & ~7ULL);

                for (size_t i = i0; i < i1; i++) {
                    size_t j;
                    for (j = j0; j < j_vec_end; j += 8) {
                        __m256d c0 = _mm256_loadu_pd(&C(i, j    ));
                        __m256d c1 = _mm256_loadu_pd(&C(i, j + 4));
                        for (size_t q = k0; q < k1; q++) {
                            __m256d a = _mm256_broadcast_sd(&A(i, q));
                            __m256d b0 = _mm256_loadu_pd(&B(q, j    ));
                            __m256d b1 = _mm256_loadu_pd(&B(q, j + 4));
                            c0 = _mm256_fmadd_pd(a, b0, c0);
                            c1 = _mm256_fmadd_pd(a, b1, c1);
                        }
                        _mm256_storeu_pd(&C(i, j    ), c0);
                        _mm256_storeu_pd(&C(i, j + 4), c1);
                    }
                    for (; j < j1; j++) {
                        double c = C(i, j);
                        for (size_t q = k0; q < k1; q++) c += A(i, q) * B(q, j);
                        C(i, j) = c;
                    }
                }
            }
        }
    }
    return C;
}

Matrix MatrixOperation::Transpose(const Matrix& A) {
    size_t rows = A.getRows();
    size_t cols = A.getCols();
    Matrix result(cols, rows);
    const size_t B = 32;
    for (size_t i = 0; i < rows; i += B)
        for (size_t j = 0; j < cols; j += B)
            for (size_t k = i; k < i + B && k < rows; ++k)
                for (size_t l = j; l < j + B && l < cols; ++l)
                    result(l, k) = A(k, l);
    return result;
}
