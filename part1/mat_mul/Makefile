BIN=bin
CC=g++
CFLAGS=-O0 -mavx2 -mfma -msse2

.PHONY: all naive loop tiling simd combination clean dir

dir:
	@mkdir -p $(BIN)

naive: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_NAIVE                                          -o $(BIN)/naive

loop: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_LOOP_OPT                                       -o $(BIN)/loop

tiling: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_TILING -D TILE_SIZE=$(TILE_SIZE)              -o $(BIN)/tiling_$(TILE_SIZE)

simd: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_SIMD                                           -o $(BIN)/simd

combination: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_COMBINED -D TILE_SIZE=$(TILE_SIZE)            -o $(BIN)/combination_$(TILE_SIZE)

all: dir
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_NAIVE                                          -o $(BIN)/naive
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_LOOP_OPT                                       -o $(BIN)/loop
	$(CC) $(CFLAGS) mat_mul.c -D OPTIMIZE_SIMD                                           -o $(BIN)/simd

clean:
	rm -rf $(BIN)
