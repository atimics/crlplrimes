# Common build/test, promoted-result, and trace defaults.

SANITIZE_CFLAGS ?= -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
SANITIZE_LDFLAGS ?= -fsanitize=address,undefined
PRIME_GAP_REPORT_DIR ?= $(BUILD_DIR)/prime_gap_experiments
PRIME_GAP_EXPERIMENT_SEEDS ?= 20240511
PRIME_GAP_EXPERIMENT_EPOCHS ?= 1,2,4,8
PRIME_GAP_EXPERIMENT_GRID ?= 0.00:0.00,0.40:0.04,0.80:0.08,1.20:0.12
PRIME_GAP_EXPERIMENT_MAX_TRAIN ?= 15000
PRIME_GAP_EXPERIMENT_MAX_TEST ?= 15000
PRIME_GATE_SEED_AUDIT_DIR ?= $(BUILD_DIR)/prime_gate_seed_audit
PRIME_GATE_SEED_AUDIT_SEEDS ?= 20240511:20240515
RESULTS_DIR ?= results/latest
SUBSTRATE_EXTENSION_RESULTS_DIR ?= results/2026-05-17-substrate-extensions
MULTITASK_ROADMAP_RESULTS_DIR ?= results/2026-05-17-multitask-roadmap
SIGNAL_FLIGHT_RESULTS_DIR ?= results/2026-05-17-signal-flight
TUNING_DIR ?= build/$(REAL)/neurosymbolic_tuning
REJECTED_CANDIDATE_DIR ?= build/$(REAL)/rejected_candidate_learning
COMPOUNDING_EVIDENCE_DIR ?= build/$(REAL)/compounding_evidence
COMPOUNDING_ITERATIONS ?= 3
REJECTED_CANDIDATE_SEED_AUDIT_DIR ?= build/$(REAL)/rejected_candidate_seed_audit
REJECTED_CANDIDATE_SEEDS ?= 20240511:20240513
HYPOTHESIS_LIBRARY_DIR ?= build/$(REAL)/hypothesis_library
HYPOTHESIS_LIBRARY_PROMOTIONDIR ?= $(BUILD_DIR)/hypothesis_library_promotion
HYPOTHESIS_LIBRARY_ABLATIONDIR ?= $(BUILD_DIR)/hypothesis_library_ablation
TRACE_DETERMINISM_AUDIT ?= $(BUILD_DIR)/traces/trace_determinism_audit.jsonl
TRACE_DETERMINISM_CANONICAL ?= $(BUILD_DIR)/traces/prime_gap.semantic.jsonl
TRACE_DETERMINISM_COMPARE ?= $(RESULTS_DIR)/trace_determinism.jsonl
TRACE_DETERMINISM_CI ?= $(BUILD_DIR)/traces/trace_determinism_ci.jsonl
TRACE_DETERMINISM_RESULTS_CANONICAL ?= $(RESULTS_DIR)/prime_gap_trace.semantic.jsonl
