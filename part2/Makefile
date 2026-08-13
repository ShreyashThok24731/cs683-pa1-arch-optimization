CXX=g++
CXXFLAGS=-O0 -mavx2 -mfma -msse2 -mavx512f
BIN=bin

.PHONY: all clean dir

dir:
	@mkdir -p $(BIN)

all: dir
	$(CXX) $(CXXFLAGS) emb.cpp -o $(BIN)/emb

clean:
	rm -rf $(BIN)
