#ifndef MATRIX_OPERATION_H
#define MATRIX_OPERATION_H

#include "matrix.h"
#include "defines.h"

extern int counter;

class MatrixOperation {
	public:

	static Matrix NaiveMatMul(const Matrix &A, const Matrix &B);
	static Matrix ReorderedMatMul(const Matrix &A, const Matrix &B);
	static Matrix UnrolledMatMul(const Matrix &A, const Matrix &B);
	static Matrix TiledMatMul(const Matrix &A, const Matrix &B);
	static Matrix VectorizedMatMul(const Matrix &A, const Matrix &B);

	static Matrix Transpose(const Matrix &A);

	static Matrix MatMul(const Matrix &A, const Matrix &B, MatrixOptimization opt ) {
		counter++;
		switch (opt) {
			case NAIVE:
				return NaiveMatMul(A, B);
			case REORDERED:
				return ReorderedMatMul(A, B);
			case UNROLLED:
				return UnrolledMatMul(A, B);
			case TILED:
				return TiledMatMul(A, B);
			case VECTORIZED:
				return VectorizedMatMul(A, B);
			default:
				throw std::invalid_argument("Invalid matrix multiplication optimization option.");
		}
	}

};

#endif
