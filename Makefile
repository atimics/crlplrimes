CC ?= cc
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
XOR_EXAMPLE := $(BUILD_DIR)/xor
PRIME_GAP_EXAMPLE := $(BUILD_DIR)/prime_gap
BENCHMARK := $(BUILD_DIR)/benchmark
TEST_RUNNER := $(BUILD_DIR)/test_nn

.PHONY: all run prime-gap prime-gap-sweep benchmark test clean

all: $(XOR_EXAMPLE) $(PRIME_GAP_EXAMPLE) $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(NN_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn.c -o $@

$(NN_TEST_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn.c -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_GAP_EXAMPLE): examples/prime_gap.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_gap.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c $(NN_TEST_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(NN_TEST_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

prime-gap: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE)

prime-gap-sweep: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 30000 30000
	./$(PRIME_GAP_EXAMPLE) 3000000 10000000 1 0 30000 30000
	./$(PRIME_GAP_EXAMPLE) 10000000 30000000 1 0 30000 30000

benchmark: $(BENCHMARK)
	./$(BENCHMARK)

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

clean:
	rm -rf $(BUILD_DIR)
