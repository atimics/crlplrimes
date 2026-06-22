# Promoted evidence-spine command targets: prime gap, delayed claim, grid path,
# and trace determinism gates.

.PHONY: experiment-schemas-promoted
.PHONY: prime-gap prime-gap-trace prime-gap-replay prime-gap-eval
.PHONY: prime-rule-search prime-rule-search-baselines prime-rule-search-conditional
.PHONY: prime-rule-search-smoke prime-gap-sweep prime-gap-regularization-sweep
.PHONY: prime-gap-experiments prime-gate-seed-audit trace-determinism-audit
.PHONY: trace-determinism-compare trace-determinism-ci
.PHONY: delayed-claim delayed-claim-trace delayed-claim-replay delayed-claim-eval delayed-claim-self-test
.PHONY: grid-path grid-path-rule-search grid-path-budget-sweep grid-path-trace grid-path-replay grid-path-no-priors grid-path-eval

experiment-schemas-promoted: $(PRIME_GAP_EXAMPLE) $(GRID_PATH) $(DELAYED_CLAIM)
	./$(PRIME_GAP_EXAMPLE) --schema
	./$(GRID_PATH) --schema
	./$(DELAYED_CLAIM) --schema

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
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer closed-loop
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer closed-loop --closed-loop-hidden 64x32
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer random --proposer-seed 20240511
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer grammar
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer structural
	./$(PRIME_RULE_SEARCH) --candidate-only --unconditional-only --moduli-range 3:255 --max-rules 16 --proposer validation-oracle
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer neural
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer closed-loop
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer closed-loop --closed-loop-hidden 64x32
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer random --proposer-seed 20240511
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer grammar
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer structural
	./$(PRIME_RULE_SEARCH) --gap-only --max-rules 16 --proposer validation-oracle

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
	$(RETIRED_SCRIPT_TARGET)

prime-gate-seed-audit: $(PRIME_GAP_EXAMPLE) $(PRIME_RULE_SEARCH) $(PRIME_REPLAY) $(PRIME_COMPARE)
	$(RETIRED_SCRIPT_TARGET)

trace-determinism-audit: prime-gap-trace $(TRACE_DETERMINISM_TOOL)
	./$(TRACE_DETERMINISM_TOOL) --trace-determinism-audit \
		--trace-input $(BUILD_DIR)/traces/prime_gap.jsonl \
		--output $(TRACE_DETERMINISM_AUDIT) \
		--canonical-output $(TRACE_DETERMINISM_CANONICAL)

trace-determinism-compare: $(TRACE_DETERMINISM_TOOL)
	$(MAKE) trace-determinism-audit REAL=float
	$(MAKE) trace-determinism-audit REAL=double
	mkdir -p $(RESULTS_DIR)
	./$(TRACE_DETERMINISM_TOOL) --trace-determinism-compare \
		--trace-input build/float/traces/prime_gap.jsonl \
		--trace-input build/double/traces/prime_gap.jsonl \
		--output $(TRACE_DETERMINISM_COMPARE)
	cp build/float/traces/prime_gap.semantic.jsonl $(TRACE_DETERMINISM_RESULTS_CANONICAL)

trace-determinism-ci: $(TRACE_DETERMINISM_TOOL)
	$(MAKE) trace-determinism-compare
	./$(TRACE_DETERMINISM_TOOL) --trace-determinism-compare \
		--trace-input build/float/traces/prime_gap.jsonl \
		--trace-input build/double/traces/prime_gap.jsonl \
		--gate \
		--output $(TRACE_DETERMINISM_CI)

delayed-claim: $(DELAYED_CLAIM)
	./$(DELAYED_CLAIM)

delayed-claim-trace: $(DELAYED_CLAIM)
	./$(DELAYED_CLAIM) --save $(BUILD_DIR)/delayed_claim_v1.nnckpt --trace $(BUILD_DIR)/delayed_claim.jsonl --trace-train

delayed-claim-replay: $(DELAYED_CLAIM) delayed-claim-trace
	./$(DELAYED_CLAIM) --replay $(BUILD_DIR)/delayed_claim.jsonl --out $(BUILD_DIR)/delayed_claim_v2.nnckpt --epochs 6

delayed-claim-eval: $(DELAYED_CLAIM) delayed-claim-replay
	./$(DELAYED_CLAIM) --compare $(BUILD_DIR)/delayed_claim_v1.nnckpt $(BUILD_DIR)/delayed_claim_v2.nnckpt
	./$(DELAYED_CLAIM) --rule-search $(BUILD_DIR)/delayed_claim_v2.nnckpt --rule-family direct --max-rules 8 --proposer all --proposer-seed 20240511
	./$(DELAYED_CLAIM) --rule-search $(BUILD_DIR)/delayed_claim_v2.nnckpt --rule-family indirect --max-rules 8 --proposer all --proposer-seed 20240511

delayed-claim-self-test: $(DELAYED_CLAIM)
	./$(DELAYED_CLAIM) --self-test-reordered-trace

grid-path: $(GRID_PATH)
	./$(GRID_PATH)

grid-path-rule-search: $(GRID_PATH)
	./$(GRID_PATH) --rule-search

grid-path-budget-sweep: $(GRID_PATH)
	./$(GRID_PATH) --rule-search --rule-family-sweep --budget-sweep --random-trials 5

grid-path-trace: $(GRID_PATH)
	./$(GRID_PATH) --save $(BUILD_DIR)/grid_path_v1.nnckpt --trace $(BUILD_DIR)/grid_path.jsonl --trace-train

grid-path-replay: $(GRID_PATH) grid-path-trace
	./$(GRID_PATH) --replay $(BUILD_DIR)/grid_path.jsonl --out $(BUILD_DIR)/grid_path_v2.nnckpt --epochs 5

grid-path-no-priors: $(GRID_PATH)
	./$(GRID_PATH) --feature-set no-priors --save $(BUILD_DIR)/grid_path_no_priors.nnckpt

grid-path-eval: $(GRID_PATH) grid-path-replay grid-path-no-priors
	./$(GRID_PATH) --compare --rule-family-sweep $(BUILD_DIR)/grid_path_v1.nnckpt $(BUILD_DIR)/grid_path_v2.nnckpt $(BUILD_DIR)/grid_path_no_priors.nnckpt
