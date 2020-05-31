CXX=g++-7
AXX=ar

ROOT=../..
SRC_DIR=$(ROOT)/src/Processing
BUILD_DIR=$(ROOT)/build/Processing

SRCFILES=$(wildcard $(SRC_DIR)/*.cpp)
OBJFILES=$(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCFILES))

main: $(BUILD_DIR)/libprocessing.a
	$(CXX) test.cpp $< -O3 -o $@

$(BUILD_DIR)/libprocessing.a: $(OBJFILES)
	$(AXX) cr $(BUILD_DIR)/libprocessing.a $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp build
	$(CXX) $< -std=c++17 -fmax-errors=5 -o $@ -c -O3

build:
	mkdir -p $(ROOT)/build && mkdir -p $(BUILD_DIR)
