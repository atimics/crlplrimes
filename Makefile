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
PRIME_GAP_OBJS := \
	$(BUILD_DIR)/prime_gap_data.o \
	$(BUILD_DIR)/prime_gap_features.o \
	$(BUILD_DIR)/prime_gap_verifier.o \
	$(BUILD_DIR)/prime_gap_eval.o \
	$(BUILD_DIR)/prime_gap_train.o
XOR_EXAMPLE := $(BUILD_DIR)/xor
ACTION_SCORER_EXAMPLE := $(BUILD_DIR)/action_scorer
PRIME_GAP_EXAMPLE := $(BUILD_DIR)/prime_gap
BENCHMARK := $(BUILD_DIR)/benchmark
TEST_RUNNER := $(BUILD_DIR)/test_nn
PRIME_GAP_CLI_TEST := tests/test_prime_gap_cli.sh

.PHONY: all run action-scorer prime-gap prime-gap-profile prime-gap-profile-verifier prime-gap-sweep benchmark test test-cli clean

all: $(XOR_EXAMPLE) $(ACTION_SCORER_EXAMPLE) $(PRIME_GAP_EXAMPLE) $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(NN_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn.c -o $@

$(NN_TEST_OBJ): src/nn.c include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn.c -o $@

$(BUILD_DIR)/prime_gap_%.o: src/prime_gap_%.c include/prime_gap.h include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(ACTION_SCORER_EXAMPLE): examples/action_scorer.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/action_scorer.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_GAP_EXAMPLE): examples/prime_gap.c $(PRIME_GAP_OBJS) $(NN_OBJ) include/prime_gap.h include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_gap.c $(PRIME_GAP_OBJS) $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJ) include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_OBJS) $(NN_TEST_OBJ) include/prime_gap.h include/nn.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(PRIME_GAP_OBJS) $(NN_TEST_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

action-scorer: $(ACTION_SCORER_EXAMPLE)
	./$(ACTION_SCORER_EXAMPLE)

prime-gap: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE)

prime-gap-profile: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 1 0 10000 10000

prime-gap-profile-verifier: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 1 0 10000 10000 1

prime-gap-sweep: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE) 1000000 3000000 2 1 30000 30000
	./$(PRIME_GAP_EXAMPLE) 3000000 10000000 1 0 30000 30000
	./$(PRIME_GAP_EXAMPLE) 10000000 30000000 1 0 30000 30000

benchmark: $(BENCHMARK)
	./$(BENCHMARK)

test: $(TEST_RUNNER) $(PRIME_GAP_EXAMPLE)
	./$(TEST_RUNNER)
	sh $(PRIME_GAP_CLI_TEST) ./$(PRIME_GAP_EXAMPLE)

test-cli: $(PRIME_GAP_EXAMPLE)
	sh $(PRIME_GAP_CLI_TEST) ./$(PRIME_GAP_EXAMPLE)

clean:
	rm -rf $(BUILD_DIR)
