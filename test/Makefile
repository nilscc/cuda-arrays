SRC_DIR   = .
BUILD_DIR = ../build

TEST_CXX_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
TEST_CXX_DEPS = $(TEST_CXX_SRCS:$(SRC_DIR)/%=$(BUILD_DIR)/%.d)

TEST_CU_SRCS  = $(wildcard $(SRC_DIR)/*.cu)
TEST_CU_DEPS  = $(TEST_CU_SRCS:$(SRC_DIR)/%=$(BUILD_DIR)/%.d)

TESTS = \
				$(TEST_CXX_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%) \
				$(TEST_CU_SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%)

#
# C++ compiler configuration
#

CXX ?= clang++

CXX_INCLUDES = -I../src

CXX_FLAGS = -std=c++11 -O3 -G -g $(CXX_INCLUDES)
CXX_LIBS  =
CXX_DEPS  = $(TEST_CXX_DEPS)

#
# CUDA compiler configuration
#

NVCC = nvcc

NVCC_FLAGS    = $(CXX_FLAGS) -arch=sm_30
NVCC_LIBS     = $(CXX_LIBS)
NVCC_DEPS     = $(TEST_CU_DEPS)

#
# General rules
#
.PHONY: all clean run-tests

all: run-tests

clean:
	@echo "Removing $(BUILD_DIR)"
	@rm -r $(BUILD_DIR) 2>/dev/null || /bin/true

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

run-tests: $(TESTS)
	@./run-tests.sh $(TESTS)

#
# C++ compilation
#

$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Building $@ from $<"
	@$(CXX) $(CXX_FLAGS) $< -o $@ $(CXX_LIBS)

# Dependencies
$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Generating $@"
	@$(CXX) $(CXX_FLAGS) -MM -MT $(@:%.d=%.o) $< > $@

#
# 
#

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "CUDA: Building $@ from $<"
	@$(NVCC) $(NVCC_FLAGS) $< -o $@ $(NVCC_LIBS)

# Dependencies
$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "Generating $@"
	@$(NVCC) $(NVCC_FLAGS) -M -MT $(@:%.d=%.o) $< > $@

ifneq ($(MAKECMDGOALS),clean)
-include $(CXX_DEPS)
-include $(NVCC_DEPS)
endif
