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
NN_HEADERS := include/nn.h include/nn_core.h include/nn_train.h
PRIME_GAP_TASK_HEADERS := include/nn_tasks_prime_gap.h src/prime_gap/internal.h
PRIME_GAP_MODULE_SOURCES := $(wildcard src/prime_gap/*.c)
PRIME_GAP_TASK_SOURCES := src/nn_tasks_prime_gap.c $(PRIME_GAP_MODULE_SOURCES)
PRIME_GAP_TASK_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap/%.o,$(PRIME_GAP_MODULE_SOURCES))
NN_OBJ := $(BUILD_DIR)/nn.o
NN_TEST_OBJ := $(BUILD_DIR)/nn_test.o
NN_SANITIZE_OBJ := $(BUILD_DIR)/nn_sanitize.o
XOR_EXAMPLE := $(BUILD_DIR)/xor
ACTION_SCORE_EXAMPLE := $(BUILD_DIR)/action_score
PRIME_GAP_EXAMPLE := $(BUILD_DIR)/prime_gap
PRIME_REPLAY := $(BUILD_DIR)/prime_replay
PRIME_COMPARE := $(BUILD_DIR)/prime_compare
PRIME_RULE_SEARCH := $(BUILD_DIR)/prime_rule_search
DELAYED_CLAIM := $(BUILD_DIR)/delayed_claim
GRID_PATH := $(BUILD_DIR)/grid_path
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
RESULTS_DIR ?= results/latest

.PHONY: all run action-score prime-gap prime-gap-trace prime-gap-replay prime-gap-eval prime-rule-search prime-rule-search-baselines prime-rule-search-conditional prime-rule-search-smoke prime-gap-sweep prime-gap-regularization-sweep prime-gap-experiments delayed-claim delayed-claim-trace delayed-claim-replay delayed-claim-eval delayed-claim-self-test grid-path grid-path-rule-search grid-path-trace grid-path-replay grid-path-eval results benchmark test sanitize check clean

all: $(XOR_EXAMPLE) $(ACTION_SCORE_EXAMPLE) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH) $(DELAYED_CLAIM) $(GRID_PATH) $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(NN_OBJ): src/nn.c $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn.c -o $@

$(NN_TEST_OBJ): src/nn.c $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn.c -o $@

$(NN_SANITIZE_OBJ): src/nn.c $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn.c -o $@

$(BUILD_DIR)/obj/prime_gap/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJ) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(ACTION_SCORE_EXAMPLE): examples/action_score.c $(NN_OBJ) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/action_score.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_GAP_EXAMPLE): examples/prime_gap.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_gap.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_REPLAY): examples/prime_replay.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_replay.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_COMPARE): examples/prime_compare.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_compare.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_RULE_SEARCH): examples/prime_rule_search.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_rule_search.c $(NN_OBJ) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(DELAYED_CLAIM): examples/delayed_claim.c $(NN_OBJ) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/delayed_claim.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(GRID_PATH): examples/grid_path.c $(NN_OBJ) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/grid_path.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJ) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_TASK_SOURCES) $(NN_TEST_OBJ) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(NN_TEST_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(SANITIZE_TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_TASK_SOURCES) $(NN_SANITIZE_OBJ) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) tests/test_nn.c $(NN_SANITIZE_OBJ) $(LDFLAGS) $(SANITIZE_LDFLAGS) $(LDLIBS) -o $@

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

action-score: $(ACTION_SCORE_EXAMPLE)
	./$(ACTION_SCORE_EXAMPLE)

prime-gap: $(PRIME_GAP_EXAMPLE)
	./$(PRIME_GAP_EXAMPLE)

prime-gap-trace: $(PRIME_GAP_EXAMPLE)
	mkdir -p $(BUILD_DIR)/traces
	./$(PRIME_GAP_EXAMPLE) 100000 200000 1 0 4000 1000 1.20 0.12 20240511 --save $(BUILD_DIR)/prime_v1.nnckpt --save-rules $(BUILD_DIR)/prime_rules.crlprule --save-learned-model $(BUILD_DIR)/prime_learned_v1.nnckpt --trace $(BUILD_DIR)/traces/prime_gap.jsonl

prime-gap-replay: $(PRIME_REPLAY) prime-gap-trace prime-rule-search
	./$(PRIME_REPLAY) $(BUILD_DIR)/traces/prime_gap.jsonl --out $(BUILD_DIR)/prime_v2.nnckpt 2 --rules $(BUILD_DIR)/prime_rules_v2.crlprule

prime-gap-eval: $(PRIME_COMPARE) prime-gap-replay
	./$(PRIME_COMPARE) --rules $(BUILD_DIR)/prime_rules_v2.crlprule $(BUILD_DIR)/prime_v1.nnckpt $(BUILD_DIR)/prime_learned_v1.nnckpt $(BUILD_DIR)/prime_v2.nnckpt

prime-rule-search: $(PRIME_RULE_SEARCH)
	mkdir -p $(BUILD_DIR)/traces
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 64 --trace $(BUILD_DIR)/traces/prime_rule_search.jsonl --save $(BUILD_DIR)/prime_rules_v2.crlprule

prime-rule-search-baselines: $(PRIME_RULE_SEARCH)
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer neural
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer random --proposer-seed 20240511
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer grammar
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer structural
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer validation-oracle

prime-rule-search-conditional: $(PRIME_RULE_SEARCH)
	mkdir -p $(BUILD_DIR)/traces
	./$(PRIME_RULE_SEARCH) --gap-only --trace $(BUILD_DIR)/traces/prime_rule_search_conditional.jsonl --save $(BUILD_DIR)/prime_rules_conditional.crlprule

prime-rule-search-smoke: $(PRIME_RULE_SEARCH)
	mkdir -p $(BUILD_DIR)/traces
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:31 --train-limit 100000 --test-limit 200000 --max-train-states 1000 --max-test-states 1000 --max-rules 32 --trace $(BUILD_DIR)/traces/prime_rule_search_smoke.jsonl --save $(BUILD_DIR)/prime_rules_v2_smoke.crlprule

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

delayed-claim: $(DELAYED_CLAIM)
	./$(DELAYED_CLAIM)

delayed-claim-trace: $(DELAYED_CLAIM)
	./$(DELAYED_CLAIM) --save $(BUILD_DIR)/delayed_claim_v1.nnckpt --trace $(BUILD_DIR)/delayed_claim.jsonl --trace-train

delayed-claim-replay: $(DELAYED_CLAIM) delayed-claim-trace
	./$(DELAYED_CLAIM) --replay $(BUILD_DIR)/delayed_claim.jsonl --out $(BUILD_DIR)/delayed_claim_v2.nnckpt --epochs 6

delayed-claim-eval: $(DELAYED_CLAIM) delayed-claim-replay
	./$(DELAYED_CLAIM) --compare $(BUILD_DIR)/delayed_claim_v1.nnckpt $(BUILD_DIR)/delayed_claim_v2.nnckpt

delayed-claim-self-test: $(DELAYED_CLAIM)
	./$(DELAYED_CLAIM) --self-test-reordered-trace

grid-path: $(GRID_PATH)
	./$(GRID_PATH)

grid-path-rule-search: $(GRID_PATH)
	./$(GRID_PATH) --rule-search

grid-path-trace: $(GRID_PATH)
	./$(GRID_PATH) --save $(BUILD_DIR)/grid_path_v1.nnckpt --trace $(BUILD_DIR)/grid_path.jsonl --trace-train

grid-path-replay: $(GRID_PATH) grid-path-trace
	./$(GRID_PATH) --replay $(BUILD_DIR)/grid_path.jsonl --out $(BUILD_DIR)/grid_path_v2.nnckpt --epochs 5

grid-path-eval: $(GRID_PATH) grid-path-replay
	./$(GRID_PATH) --compare $(BUILD_DIR)/grid_path_v1.nnckpt $(BUILD_DIR)/grid_path_v2.nnckpt

results: $(DELAYED_CLAIM) $(GRID_PATH) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH)
	$(PYTHON) scripts/capture_results.py --output-dir $(RESULTS_DIR) --real $(REAL)

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

sanitize: $(SANITIZE_TEST_RUNNER)
	./$(SANITIZE_TEST_RUNNER)

check: test sanitize action-score delayed-claim-self-test delayed-claim-eval grid-path-eval prime-rule-search-smoke prime-gap-eval
	$(MAKE) test REAL=double

clean:
	rm -rf $(BUILD_DIR)
