CC ?= cc
PYTHON ?= python3
REAL ?= float
CPPFLAGS ?= -Iinclude
CPPFLAGS += -DNN_REAL=$(REAL)
ifeq ($(REAL),float)
CPPFLAGS += -DNN_REAL_IS_FLOAT=1
else
CPPFLAGS += -DNN_REAL_IS_FLOAT=0
endif
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O3
LDFLAGS ?=
LDLIBS ?= -lm

BUILD_DIR := build/$(REAL)
NN_OBJ := $(BUILD_DIR)/nn.o
NN_TEST_OBJ := $(BUILD_DIR)/nn_test.o
NN_SANITIZE_OBJ := $(BUILD_DIR)/nn_sanitize.o
XOR_EXAMPLE := $(BUILD_DIR)/xor
PRIME_GAP_EXAMPLE := $(BUILD_DIR)/prime_gap
BENCHMARK := $(BUILD_DIR)/benchmark
TEST_RUNNER := $(BUILD_DIR)/test_nn
SANITIZE_TEST_RUNNER := $(BUILD_DIR)/test_nn_sanitize
SANITIZE_CFLAGS ?= -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
SANITIZE_LDFLAGS ?= -fsanitize=address,undefined
PRIME_GAP_REPORT_DIR ?= $(BUILD_DIR)/prime_gap_experiments
PRIME_GAP_EXPERIMENT_SEEDS ?= 20240511
PRIME_GAP_EXPERIMENT_EPOCHS ?= 1,2,4,8
PRIME_GAP_EXPERIMENT_GRID ?= 0.00:0.00,0.40:0.04,0.80:0.08,1.20:0.12
PRIME_GAP_EXPERIMENT_MAX_TRAIN ?= 15000
PRIME_GAP_EXPERIMENT_MAX_TEST ?= 15000

.PHONY: all run prime-gap prime-gap-sweep prime-gap-regularization-sweep prime-gap-experiments benchmark test sanitize clean

all: $(XOR_EXAMPLE) $(PRIME_GAP_EXAMPLE) $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(NN_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn.c -o $@

$(NN_TEST_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn.c -o $@

$(NN_SANITIZE_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn.c -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_GAP_EXAMPLE): examples/prime_gap.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_gap.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c examples/prime_gap.c $(NN_TEST_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(NN_TEST_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(SANITIZE_TEST_RUNNER): tests/test_nn.c examples/prime_gap.c $(NN_SANITIZE_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) tests/test_nn.c $(NN_SANITIZE_OBJ) $(LDFLAGS) $(SANITIZE_LDFLAGS) $(LDLIBS) -o $@

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

prime-gap: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE)

prime-gap-sweep: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 30000 30000
	./$(PRIME_GAP_EXAMPLE) 3000000 10000000 1 0 30000 30000
	./$(PRIME_GAP_EXAMPLE) 10000000 30000000 1 0 30000 30000

prime-gap-regularization-sweep: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 15000 15000 0.00 0.00
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 15000 15000 0.40 0.04
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 15000 15000 0.80 0.08
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 15000 15000 1.20 0.12

prime-gap-experiments: $(PRIME_GAP_EXAMPLE)
	$(PYTHON) scripts/prime_gap_experiments.py \
		--binary ./$(PRIME_GAP_EXAMPLE) \
		--output-dir $(PRIME_GAP_REPORT_DIR) \
		--supervised-epochs $(PRIME_GAP_EXPERIMENT_EPOCHS) \
		--regularization-grid $(PRIME_GAP_EXPERIMENT_GRID) \
		--seeds $(PRIME_GAP_EXPERIMENT_SEEDS) \
		--max-train-states $(PRIME_GAP_EXPERIMENT_MAX_TRAIN) \
		--max-test-states $(PRIME_GAP_EXPERIMENT_MAX_TEST)

benchmark: $(BENCHMARK)
	./$(BENCHMARK)

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

sanitize: $(SANITIZE_TEST_RUNNER)
	./$(SANITIZE_TEST_RUNNER)

clean:
	rm -rf $(BUILD_DIR)
