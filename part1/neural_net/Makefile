CXX = g++
CXXFLAGS = -std=c++17 -O0 -mavx2 -mfma -Iinclude -MMD

SRC_DIR = src
BIN_DIR = bin

CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS     = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)
DEPS        = $(OBJECTS:.o=.d)

TARGET = $(BIN_DIR)/main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/*.d
	rm -rf $(BIN_DIR)

-include $(DEPS)
