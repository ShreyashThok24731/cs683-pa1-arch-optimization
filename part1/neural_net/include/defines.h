#ifndef DEFINES_H
#define DEFINES_H

#define INITIAL_VALUE 0.0
#define TILE_SIZE 16

typedef double element_t;

enum MatrixOptimization {
	NAIVE,
	REORDERED,
	UNROLLED,
	TILED,
	VECTORIZED
};

#endif
