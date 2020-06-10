CXX=g++-7

ROOT=../..
SRC_DIR=$(ROOT)/src/Server
BUILD_DIR=$(ROOT)/build/Server

STATE_SRC_FILES=$(wildcard $(SRC_DIR)/State/*.cpp)
STATE_OBJ_FILES=$(patsubst $(SRC_DIR)/State/%.cpp,$(BUILD_DIR)/State/%.o,$(STATE_SRC_FILES))

main: $(BUILD_DIR)/Handler.o $(STATE_OBJ_FILES)
	$(CXX) ./test.cpp $^ -lpthread -lcapnp -lkj -std=c++17 -fmax-errors=5 -O3 -o ./$@

$(BUILD_DIR)/Handler.o: $(SRC_DIR)/Handler.cpp build
	$(CXX) $< -std=c++17 -fmax-errors=5 -O3 -c -o $@  

$(BUILD_DIR)/State/BatchInferenceWorkflow.o: $(SRC_DIR)/State/BatchInferenceWorkflow.cpp build
	$(CXX) $< -std=c++17 -fmax-errors=5 -O3 -c -o $@

$(BUILD_DIR)/State/BatchInferenceWorkflowDispatcher.o: $(SRC_DIR)/State/BatchInferenceWorkflowDispatcher.cpp build
	$(CXX) $< -lcapnp -lkj -std=c++17 -fmax-errors=5 -O3 -c -o $@

build:
	mkdir -p $(BUILD_DIR)  &&  mkdir -p $(BUILD_DIR)/State

clean:
	rm -rf $(BUILD_DIR)


