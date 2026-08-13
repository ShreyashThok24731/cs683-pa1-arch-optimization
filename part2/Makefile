CXX=g++
CXXFLAGS=-O0 -mavx2 -mfma -msse2 -mavx512f
BIN=bin

.PHONY: all verify clean dir
.DEFAULT_GOAL := all

dir:
	@mkdir -p $(BIN)

all: dir
	$(CXX) $(CXXFLAGS) emb.cpp -o $(BIN)/emb

verify: dir
	$(CXX) $(CXXFLAGS) verify_emb.cpp -o $(BIN)/verify_emb

clean:
	rm -rf $(BIN)
