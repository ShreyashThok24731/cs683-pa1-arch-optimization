BIN=bin
CC=g++
CFLAGS=-O0 -mavx2 -mfma -msse2
AVX512FLAGS=-mavx512f

TILE_SIZE   ?= 32
SIMD_WIDTH  ?= 256
W512 = $(if $(filter 512,$(SIMD_WIDTH)),$(AVX512FLAGS),)

.PHONY: all naive reorder unroll loop tiling simd simd128 simd256 simd512 combination verify clean dir
.DEFAULT_GOAL := all

dir:
	@mkdir -p $(BIN)

naive: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_NAIVE                                 -o $(BIN)/naive

reorder: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_LOOP_REORDER                          -o $(BIN)/reorder

unroll: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_LOOP_UNROLL                           -o $(BIN)/unroll

loop: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_LOOP_OPT                              -o $(BIN)/loop

tiling: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_TILING -D TILE_SIZE=$(TILE_SIZE)      -o $(BIN)/tiling_$(TILE_SIZE)

simd128: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_SIMD -D SIMD_WIDTH=128                -o $(BIN)/simd_128

simd256: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_SIMD -D SIMD_WIDTH=256                -o $(BIN)/simd_256

simd512: dir
	$(CC) $(CFLAGS) $(AVX512FLAGS) mat_mul.c -D OPTIMIZE_SIMD -D SIMD_WIDTH=512 -o $(BIN)/simd_512

simd: simd128 simd256 simd512

combination: dir
	$(CC) $(CFLAGS) $(W512) mat_mul.c -D OPTIMIZE_COMBINED \
	    -D TILE_SIZE=$(TILE_SIZE) -D SIMD_WIDTH=$(SIMD_WIDTH) \
	    -o $(BIN)/combination$(SIMD_WIDTH)_$(TILE_SIZE)

verify: dir
	$(CC) $(CFLAGS) $(AVX512FLAGS) verify.cpp -o $(BIN)/verify

all: naive reorder unroll loop simd
	@for T in 16 32 64 128; do \
	    $(MAKE) --no-print-directory tiling TILE_SIZE=$$T; \
	    for W in 128 256 512; do \
	        $(MAKE) --no-print-directory combination TILE_SIZE=$$T SIMD_WIDTH=$$W; \
	    done; \
	done

clean:
	rm -rf $(BIN)
