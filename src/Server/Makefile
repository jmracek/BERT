CXX=g++-7

ROOT=../..
SRC_DIR=$(ROOT)/src/Server
BUILD_DIR=$(ROOT)/build/Server

test.o: $(BUILD_DIR)/handler.o $(BUILD_DIR)/State/state.o
	$(CXX) ./test.cpp $^ -std=c++17 -fmax-errors=5 -o ./test -pthread

$(BUILD_DIR)/State/state.o: $(SRC_DIR)/State/batchinferencerequeststate.cpp
	$(CXX) $< -std=c++17 -o $@ -c

$(BUILD_DIR)/handler.o: $(SRC_DIR)/handler.cpp build
	$(CXX) $< -std=c++17 -fmax-errors=5 -o $@ -c -latomic -pthread

build:
	mkdir -p $(BUILD_DIR)  &&  mkdir -p $(BUILD_DIR)/State
