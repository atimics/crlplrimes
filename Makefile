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
LDLIBS ?= -lm
.DEFAULT_GOAL := help

# Portable BLAS: accelerate (macOS default), openblas (Linux default), none (scalar fallback).
NN_BLAS ?= $(shell uname -s | grep -q Darwin && echo accelerate || echo openblas)
ifeq ($(NN_BLAS),accelerate)
CPPFLAGS += -DNN_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
LDFLAGS ?= -framework Accelerate
else ifeq ($(NN_BLAS),openblas)
CPPFLAGS += -DNN_USE_CBLAS
LDFLAGS ?= -lopenblas
else ifeq ($(NN_BLAS),none)
LDFLAGS ?=
else
$(error Unsupported NN_BLAS=$(NN_BLAS); use accelerate, openblas, or none)
endif
RETIRED_SCRIPT_TARGET = @echo "$@: retired after script removal; port this workflow into the C runtime before use."; exit 2

BUILD_DIR := build/$(REAL)
BLAS_MODE_STAMP := $(BUILD_DIR)/.nn_blas_mode
NN_HEADERS := include/nn.h include/nn_core.h include/nn_train.h include/nn_experiment.h include/nn_symbolic.h include/nn_bitgrid.h include/nn_tilebot.h include/nn_layer.h include/nn_program.h include/nn_hypothesis.h include/nn_protocol.h include/nn_json.h include/nn_text.h include/nn_term_bridge.h include/nn_dataset.h include/nn_grammar_discovery.h include/nn_paper.h
PRIME_GAP_TASK_HEADERS := include/nn_tasks_prime_gap.h src/prime_gap/internal.h
PRIME_GAP_MODULE_SOURCES := $(wildcard src/prime_gap/*.c)
PRIME_GAP_TASK_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap/%.o,$(PRIME_GAP_MODULE_SOURCES))
PRIME_GAP_TEST_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap_test/%.o,$(PRIME_GAP_MODULE_SOURCES))
PRIME_GAP_SANITIZE_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap_sanitize/%.o,$(PRIME_GAP_MODULE_SOURCES))
NN_OBJ := $(BUILD_DIR)/nn.o
NN_BLAS_OBJ := $(BUILD_DIR)/nn_blas.o
NN_EXPERIMENT_OBJ := $(BUILD_DIR)/nn_experiment.o
NN_SYMBOLIC_OBJ := $(BUILD_DIR)/nn_symbolic.o
NN_BITGRID_OBJ := $(BUILD_DIR)/nn_bitgrid.o
NN_TILEBOT_OBJ := $(BUILD_DIR)/nn_tilebot.o
NN_LAYER_OBJ := $(BUILD_DIR)/nn_layer.o
NN_PROGRAM_OBJ := $(BUILD_DIR)/nn_program.o
NN_HYPOTHESIS_OBJ := $(BUILD_DIR)/nn_hypothesis.o
NN_PROTOCOL_OBJ := $(BUILD_DIR)/nn_protocol.o
NN_JSON_OBJ := $(BUILD_DIR)/nn_json.o
NN_TEXT_OBJ := $(BUILD_DIR)/nn_text.o
NN_DATASET_OBJ := $(BUILD_DIR)/nn_dataset.o
NN_GRAMMAR_DISCOVERY_OBJ := $(BUILD_DIR)/nn_grammar_discovery.o
NN_ARC_SELFPLAY_OBJ := $(BUILD_DIR)/nn_arc_selfplay.o
NN_TERM_BRIDGE_OBJ := $(BUILD_DIR)/nn_term_bridge.o
NN_TFIDF_OBJ := $(BUILD_DIR)/nn_tfidf.o
NN_PAPER_OBJ := $(BUILD_DIR)/nn_paper.o
NN_TEST_OBJ := $(BUILD_DIR)/nn_test.o
NN_EXPERIMENT_TEST_OBJ := $(BUILD_DIR)/nn_experiment_test.o
NN_SYMBOLIC_TEST_OBJ := $(BUILD_DIR)/nn_symbolic_test.o
NN_BITGRID_TEST_OBJ := $(BUILD_DIR)/nn_bitgrid_test.o
NN_TILEBOT_TEST_OBJ := $(BUILD_DIR)/nn_tilebot_test.o
NN_LAYER_TEST_OBJ := $(BUILD_DIR)/nn_layer_test.o
NN_PROGRAM_TEST_OBJ := $(BUILD_DIR)/nn_program_test.o
NN_HYPOTHESIS_TEST_OBJ := $(BUILD_DIR)/nn_hypothesis_test.o
NN_PROTOCOL_TEST_OBJ := $(BUILD_DIR)/nn_protocol_test.o
NN_JSON_TEST_OBJ := $(BUILD_DIR)/nn_json_test.o
NN_TEXT_TEST_OBJ := $(BUILD_DIR)/nn_text_test.o
NN_DATASET_TEST_OBJ := $(BUILD_DIR)/nn_dataset_test.o
NN_DATASET_TEST_STUBS_OBJ := $(BUILD_DIR)/nn_dataset_test_stubs.o
NN_PAPER_TEST_OBJ := $(BUILD_DIR)/nn_paper_test.o
NN_SANITIZE_OBJ := $(BUILD_DIR)/nn_sanitize.o
NN_BLAS_SANITIZE_OBJ := $(BUILD_DIR)/nn_blas_sanitize.o
NN_EXPERIMENT_SANITIZE_OBJ := $(BUILD_DIR)/nn_experiment_sanitize.o
NN_SYMBOLIC_SANITIZE_OBJ := $(BUILD_DIR)/nn_symbolic_sanitize.o
NN_BITGRID_SANITIZE_OBJ := $(BUILD_DIR)/nn_bitgrid_sanitize.o
NN_TILEBOT_SANITIZE_OBJ := $(BUILD_DIR)/nn_tilebot_sanitize.o
NN_LAYER_SANITIZE_OBJ := $(BUILD_DIR)/nn_layer_sanitize.o
NN_PROGRAM_SANITIZE_OBJ := $(BUILD_DIR)/nn_program_sanitize.o
NN_HYPOTHESIS_SANITIZE_OBJ := $(BUILD_DIR)/nn_hypothesis_sanitize.o
NN_PROTOCOL_SANITIZE_OBJ := $(BUILD_DIR)/nn_protocol_sanitize.o
NN_JSON_SANITIZE_OBJ := $(BUILD_DIR)/nn_json_sanitize.o
NN_TEXT_SANITIZE_OBJ := $(BUILD_DIR)/nn_text_sanitize.o
NN_DATASET_SANITIZE_OBJ := $(BUILD_DIR)/nn_dataset_sanitize.o
NN_PAPER_SANITIZE_OBJ := $(BUILD_DIR)/nn_paper_sanitize.o
NN_GO_EXPERIMENT_OBJ := $(BUILD_DIR)/go_experiment.o
NN_OBJS := $(NN_OBJ) $(NN_EXPERIMENT_OBJ) $(NN_SYMBOLIC_OBJ) $(NN_GO_EXPERIMENT_OBJ) $(NN_BITGRID_OBJ) $(NN_TILEBOT_OBJ) $(NN_LAYER_OBJ) $(NN_PROGRAM_OBJ) $(NN_HYPOTHESIS_OBJ) $(NN_PROTOCOL_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) $(NN_DATASET_OBJ) $(NN_BLAS_OBJ)
NN_TEST_OBJS := $(NN_TEST_OBJ) $(NN_EXPERIMENT_TEST_OBJ) $(NN_SYMBOLIC_TEST_OBJ) $(NN_BITGRID_TEST_OBJ) $(NN_TILEBOT_TEST_OBJ) $(NN_LAYER_TEST_OBJ) $(NN_PROGRAM_TEST_OBJ) $(NN_HYPOTHESIS_TEST_OBJ) $(NN_PROTOCOL_TEST_OBJ) $(NN_JSON_TEST_OBJ) $(NN_TEXT_TEST_OBJ) $(NN_DATASET_TEST_OBJ) $(NN_PAPER_TEST_OBJ)
NN_SANITIZE_OBJS := $(NN_SANITIZE_OBJ) $(NN_BLAS_SANITIZE_OBJ) $(NN_EXPERIMENT_SANITIZE_OBJ) $(NN_SYMBOLIC_SANITIZE_OBJ) $(NN_BITGRID_SANITIZE_OBJ) $(NN_TILEBOT_SANITIZE_OBJ) $(NN_LAYER_SANITIZE_OBJ) $(NN_PROGRAM_SANITIZE_OBJ) $(NN_HYPOTHESIS_SANITIZE_OBJ) $(NN_PROTOCOL_SANITIZE_OBJ) $(NN_JSON_SANITIZE_OBJ) $(NN_TEXT_SANITIZE_OBJ) $(NN_DATASET_SANITIZE_OBJ) $(NN_PAPER_SANITIZE_OBJ)
XOR_EXAMPLE := $(BUILD_DIR)/xor
BITWISE_RECURSIVE := $(BUILD_DIR)/bitwise_recursive
TILE_RECURSIVE := $(BUILD_DIR)/tile_recursive
TILE_FEATURE_ENTROPY := $(BUILD_DIR)/tile_feature_entropy
LIFE_CURRICULUM := $(BUILD_DIR)/life_curriculum
LAYER_DISCOVERY := $(BUILD_DIR)/layer_discovery
ACTION_SCORE_EXAMPLE := $(BUILD_DIR)/action_score
DATASET_ACTION_SCORE := $(BUILD_DIR)/dataset_action_score
GO_EXPERIMENT := $(BUILD_DIR)/go_experiment
PRIME_GAP_EXAMPLE := $(BUILD_DIR)/prime_gap
PRIME_REPLAY := $(BUILD_DIR)/prime_replay
PRIME_COMPARE := $(BUILD_DIR)/prime_compare
PRIME_RULE_SEARCH := $(BUILD_DIR)/prime_rule_search
DELAYED_CLAIM := $(BUILD_DIR)/delayed_claim
GRID_PATH := $(BUILD_DIR)/grid_path
MULTITASK_DOMAIN := $(BUILD_DIR)/multitask_domain
SYMBOLIC_DOMAINS := $(BUILD_DIR)/symbolic_domains
PROGRAM_SYNTHESIS := $(BUILD_DIR)/program_synthesis
MODULAR_GATE_SEARCH := $(BUILD_DIR)/modular_gate_search
CROSS_DOMAIN_TRANSFER := $(BUILD_DIR)/cross_domain_transfer
TRACE_DETERMINISM_TOOL := $(BUILD_DIR)/trace_determinism
NN_SRCS := src/nn.c \
	src/nn_numeric_certificate.inc \
	src/nn_checkpoint.inc \
	src/nn_trace.inc
NN_PAPER_SRCS := src/nn_paper.c \
	src/nn_paper_claim_registry.inc \
	src/nn_paper_test_api.inc \
	src/nn_paper_claim_evaluators.inc \
	src/nn_paper_experiment_matrix.inc \
	src/nn_paper_evidence_origin.inc
ARC_AGI_INCS := examples/arc_agi_hypothesis_templates.inc \
	examples/arc_agi_hypothesis_template_rules.inc \
	examples/arc_agi_hypothesis_program_eval.inc \
	examples/arc_agi_hypothesis_learned_prior.inc \
	examples/arc_agi_hypothesis_selection_manifest.inc \
	examples/arc_agi_hypothesis_selection.inc \
	examples/arc_agi_hypothesis_induction.inc \
	examples/arc_agi_hypothesis_sketch_programs.inc \
	examples/arc_agi_hypothesis_selection_core.inc \
	examples/arc_agi_hypothesis_template_impls.inc \
	examples/arc_agi_visual_program.inc \
	examples/arc_agi_sketch_probes.inc \
	examples/arc_agi_sketch_probe_metadata.inc \
	examples/arc_agi_sketch_probe_miners.inc \
	examples/arc_agi_sketch_probe_runtime_prior.inc \
	examples/arc_agi_sketch_probe_export.inc
ARC_AGI := $(BUILD_DIR)/arc_agi
SIGNAL_FLIGHT := $(BUILD_DIR)/signal_flight
SIGNAL_FLIGHT_ROLLOUT := $(BUILD_DIR)/signal_flight_rollout
SIGNAL_MINING := $(BUILD_DIR)/signal_mining
SIGNAL_NPC_WORKER := $(BUILD_DIR)/signal_npc_worker
HAIKU_LM := $(BUILD_DIR)/haiku_lm
HAIKU_UNDERSTANDING := $(BUILD_DIR)/haiku_understanding
HAIKU_RULE_SEARCH := $(BUILD_DIR)/haiku_rule_search
HAIKU_LINE_SEARCH := $(BUILD_DIR)/haiku_line_search

SUBSTRATE_EXPERIMENTS := $(BUILD_DIR)/substrate_experiments
$(SUBSTRATE_EXPERIMENTS): examples/substrate_experiments.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/substrate_experiments.c $(LDFLAGS) $(LDLIBS) -o $@


PREPROCESS_PRIME_GAP_TRACE := $(BUILD_DIR)/preprocess_prime_gap_trace
$(PREPROCESS_PRIME_GAP_TRACE): tools/preprocess_prime_gap_trace.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) tools/preprocess_prime_gap_trace.c $(LDFLAGS) $(LDLIBS) -o $@

NN_GRAMMAR_DISCOVERY_RUNTIME := $(BUILD_DIR)/neural_grammar_discovery_runtime
BENCHMARK := $(BUILD_DIR)/benchmark
TEST_RUNNER := $(BUILD_DIR)/test_nn
SANITIZE_TEST_RUNNER := $(BUILD_DIR)/test_nn_sanitize
TRAIN_CONTRACT_BRAIN := $(BUILD_DIR)/train_contract_brain
GO_SELFPLAY := $(BUILD_DIR)/go_selfplay
TRAIN_CROSS_DOMAIN := $(BUILD_DIR)/train_cross_domain
include mk/options.mk
.PHONY: FORCE

# Human-facing command surfaces live in mk/*.mk. Keep this block boring:
# variables and binary recipes stay here; orchestration targets stay there.
include mk/control.mk
include mk/promoted.mk
include mk/action_score.mk
include mk/arc.mk
include mk/program.mk
include mk/examples.mk
include mk/signal.mk
include mk/haiku.mk
include mk/grammar.mk
include mk/evidence.mk
include mk/paper.mk
include mk/eml.mk
include mk/ilxyr.mk
include mk/runtime.mk

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BLAS_MODE_STAMP): FORCE | $(BUILD_DIR)
	@mode="$(NN_BLAS)|$(CPPFLAGS)"; \
	if [ ! -f "$@" ] || [ "$$(cat "$@")" != "$$mode" ]; then \
		printf '%s\n' "$$mode" > "$@"; \
	fi

$(NN_OBJ): $(NN_SRCS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn.c -o $@

$(NN_BLAS_OBJ): src/nn_blas.c include/nn_blas.h include/nn_core.h $(BLAS_MODE_STAMP) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_blas.c -o $@

$(NN_EXPERIMENT_OBJ): src/nn_experiment.c include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_experiment.c -o $@

$(NN_SYMBOLIC_OBJ): src/nn_symbolic.c include/nn_symbolic.h include/nn_core.h include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_symbolic.c -o $@

$(NN_BITGRID_OBJ): src/nn_bitgrid.c include/nn_bitgrid.h include/nn_symbolic.h include/nn_core.h include/nn_train.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_bitgrid.c -o $@

$(NN_TILEBOT_OBJ): src/nn_tilebot.c include/nn_tilebot.h include/nn_symbolic.h include/nn_core.h include/nn_train.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_tilebot.c -o $@

$(NN_LAYER_OBJ): src/nn_layer.c include/nn_layer.h include/nn_bitgrid.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_layer.c -o $@

$(NN_PROGRAM_OBJ): src/nn_program.c include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_program.c -o $@

$(NN_HYPOTHESIS_OBJ): src/nn_hypothesis.c include/nn_hypothesis.h include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_hypothesis.c -o $@

$(NN_PROTOCOL_OBJ): src/nn_protocol.c include/nn_protocol.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_protocol.c -o $@

$(NN_JSON_OBJ): src/nn_json.c include/nn_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_json.c -o $@

$(NN_TEXT_OBJ): src/nn_text.c include/nn_text.h include/nn_term_bridge.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_text.c -o $@

$(NN_TFIDF_OBJ): src/nn_tfidf.c include/nn_tfidf.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_tfidf.c -o $@

$(NN_TERM_BRIDGE_OBJ): src/nn_term_bridge.c include/nn_term_bridge.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_term_bridge.c -o $@
$(NN_DATASET_OBJ): src/nn_dataset.c include/nn_dataset.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_dataset.c -o $@

$(NN_GRAMMAR_DISCOVERY_OBJ): src/nn_grammar_discovery.c src/nn_grammar_discovery_json.inc src/nn_grammar_discovery_haiku.inc src/nn_grammar_discovery_arc.inc src/nn_grammar_discovery_arc_failure.inc src/nn_grammar_discovery_arc_synthetic_curriculum.inc src/nn_grammar_discovery_arc_ranker.inc src/nn_grammar_discovery_report.inc src/nn_grammar_discovery_health.inc src/nn_grammar_discovery_arc_neural_discovery.inc include/nn_grammar_discovery.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_grammar_discovery.c -o $@

$(NN_ARC_SELFPLAY_OBJ): src/nn_arc_selfplay.c include/nn_grammar_discovery.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_arc_selfplay.c -o $@

$(NN_PAPER_OBJ): $(NN_PAPER_SRCS) include/nn_paper.h include/nn.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_paper.c -o $@

$(TRACE_DETERMINISM_TOOL): examples/trace_determinism.c $(NN_PAPER_OBJ) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_TEXT_OBJ) include/nn_paper.h include/nn.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/trace_determinism.c $(NN_PAPER_OBJ) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_TEXT_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(NN_TEST_OBJ): $(NN_SRCS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn.c -o $@

$(NN_EXPERIMENT_TEST_OBJ): src/nn_experiment.c include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_experiment.c -o $@

$(NN_SYMBOLIC_TEST_OBJ): src/nn_symbolic.c include/nn_symbolic.h include/nn_core.h include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_symbolic.c -o $@

$(NN_BITGRID_TEST_OBJ): src/nn_bitgrid.c include/nn_bitgrid.h include/nn_symbolic.h include/nn_core.h include/nn_train.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_bitgrid.c -o $@

$(NN_TILEBOT_TEST_OBJ): src/nn_tilebot.c include/nn_tilebot.h include/nn_symbolic.h include/nn_core.h include/nn_train.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_tilebot.c -o $@

$(NN_LAYER_TEST_OBJ): src/nn_layer.c include/nn_layer.h include/nn_bitgrid.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_layer.c -o $@

$(NN_PROGRAM_TEST_OBJ): src/nn_program.c include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_program.c -o $@

$(NN_HYPOTHESIS_TEST_OBJ): src/nn_hypothesis.c include/nn_hypothesis.h include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_hypothesis.c -o $@

$(NN_PROTOCOL_TEST_OBJ): src/nn_protocol.c include/nn_protocol.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_protocol.c -o $@

$(NN_JSON_TEST_OBJ): src/nn_json.c include/nn_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_json.c -o $@

$(NN_TEXT_TEST_OBJ): src/nn_text.c include/nn_text.h include/nn_term_bridge.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_text.c -o $@

$(NN_DATASET_TEST_OBJ): src/nn_dataset.c include/nn_dataset.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_dataset.c -o $@

$(NN_DATASET_TEST_STUBS_OBJ): src/nn_dataset_test_stubs.c include/nn_dataset.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_dataset_test_stubs.c -o $@

$(NN_PAPER_TEST_OBJ): $(NN_PAPER_SRCS) include/nn_paper.h include/nn.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_paper.c -o $@

$(NN_SANITIZE_OBJ): $(NN_SRCS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn.c -o $@

$(NN_BLAS_SANITIZE_OBJ): src/nn_blas.c include/nn_blas.h include/nn_core.h $(BLAS_MODE_STAMP) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_blas.c -o $@

$(NN_EXPERIMENT_SANITIZE_OBJ): src/nn_experiment.c include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_experiment.c -o $@

$(NN_SYMBOLIC_SANITIZE_OBJ): src/nn_symbolic.c include/nn_symbolic.h include/nn_core.h include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_symbolic.c -o $@

$(NN_BITGRID_SANITIZE_OBJ): src/nn_bitgrid.c include/nn_bitgrid.h include/nn_symbolic.h include/nn_core.h include/nn_train.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_bitgrid.c -o $@

$(NN_TILEBOT_SANITIZE_OBJ): src/nn_tilebot.c include/nn_tilebot.h include/nn_symbolic.h include/nn_core.h include/nn_train.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_tilebot.c -o $@

$(NN_LAYER_SANITIZE_OBJ): src/nn_layer.c include/nn_layer.h include/nn_bitgrid.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_layer.c -o $@

$(NN_PROGRAM_SANITIZE_OBJ): src/nn_program.c include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_program.c -o $@

$(NN_HYPOTHESIS_SANITIZE_OBJ): src/nn_hypothesis.c include/nn_hypothesis.h include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_hypothesis.c -o $@

$(NN_PROTOCOL_SANITIZE_OBJ): src/nn_protocol.c include/nn_protocol.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_protocol.c -o $@

$(NN_JSON_SANITIZE_OBJ): src/nn_json.c include/nn_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_json.c -o $@

$(NN_TEXT_SANITIZE_OBJ): src/nn_text.c include/nn_text.h include/nn_term_bridge.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_text.c -o $@

$(NN_DATASET_SANITIZE_OBJ): src/nn_dataset.c include/nn_dataset.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_dataset.c -o $@

$(NN_PAPER_SANITIZE_OBJ): $(NN_PAPER_SRCS) include/nn_paper.h include/nn.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_paper.c -o $@

$(BUILD_DIR)/obj/prime_gap/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffast-math -c $< -o $@

$(BUILD_DIR)/obj/prime_gap_test/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -ffast-math -c $< -o $@

$(BUILD_DIR)/obj/prime_gap_sanitize/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -ffast-math -c $< -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(BITWISE_RECURSIVE): examples/bitwise_recursive.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/bitwise_recursive.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TILE_RECURSIVE): examples/tile_recursive.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/tile_recursive.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TILE_FEATURE_ENTROPY): examples/tile_feature_entropy.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/tile_feature_entropy.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(LIFE_CURRICULUM): examples/life_curriculum.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/life_curriculum.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(LAYER_DISCOVERY): examples/layer_discovery.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/layer_discovery.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(ACTION_SCORE_EXAMPLE): examples/action_score.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/action_score.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(DATASET_ACTION_SCORE): examples/dataset_action_score.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(GO_EXP_FLAGS) $(CFLAGS) examples/dataset_action_score.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(HAIKU_LM): examples/haiku_lm.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_lm.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(HAIKU_UNDERSTANDING): examples/haiku_understanding.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_understanding.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(HAIKU_RULE_SEARCH): examples/haiku_rule_search.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_rule_search.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@
$(HAIKU_LINE_SEARCH): examples/haiku_line_search.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_line_search.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_GAP_EXAMPLE): examples/prime_gap.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_gap.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_REPLAY): examples/prime_replay.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_replay.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_COMPARE): examples/prime_compare.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_compare.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PRIME_RULE_SEARCH): examples/prime_rule_search.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/prime_rule_search.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(DELAYED_CLAIM): examples/delayed_claim.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/delayed_claim.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(GRID_PATH): examples/grid_path.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/grid_path.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(MULTITASK_DOMAIN): examples/multitask_domain.c examples/multitask_domain_reports.inc $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/multitask_domain.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SYMBOLIC_DOMAINS): examples/symbolic_domains.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/symbolic_domains.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PROGRAM_SYNTHESIS): examples/program_synthesis.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/program_synthesis.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(MODULAR_GATE_SEARCH): examples/modular_gate_search.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/modular_gate_search.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(CROSS_DOMAIN_TRANSFER): examples/cross_domain_transfer.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/cross_domain_transfer.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(ARC_AGI): examples/arc_agi.c $(ARC_AGI_INCS) $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/arc_agi.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(NN_GRAMMAR_DISCOVERY_RUNTIME): examples/neural_grammar_discovery.c $(NN_GRAMMAR_DISCOVERY_OBJ) $(NN_ARC_SELFPLAY_OBJ) $(NN_PAPER_OBJ) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_TEXT_OBJ) include/nn_grammar_discovery.h include/nn_paper.h include/nn.h include/nn_text.h include/nn_term_bridge.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/neural_grammar_discovery.c $(NN_GRAMMAR_DISCOVERY_OBJ) $(NN_ARC_SELFPLAY_OBJ) $(NN_PAPER_OBJ) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_TEXT_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

ifeq ($(WITH_SIGNAL),1)
$(SIGNAL_FLIGHT): examples/signal_flight.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/signal_flight.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_FLIGHT_ROLLOUT): examples/signal_flight_rollout.c $(NN_OBJS) $(NN_HEADERS) $(SIGNAL_FLIGHT_SIM_SOURCES) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SIGNAL_FLIGHT_INCLUDE_FLAGS) $(CFLAGS) examples/signal_flight_rollout.c $(NN_OBJS) $(SIGNAL_FLIGHT_SIM_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_MINING): examples/signal_mining.c $(NN_OBJS) $(NN_HEADERS) $(SIGNAL_FLIGHT_SIM_SOURCES) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SIGNAL_FLIGHT_INCLUDE_FLAGS) $(CFLAGS) examples/signal_mining.c $(NN_OBJS) $(SIGNAL_FLIGHT_SIM_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_NPC_WORKER): examples/signal_npc_worker.c $(NN_OBJS) $(NN_HEADERS) $(SIGNAL_FLIGHT_SIM_SOURCES) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SIGNAL_FLIGHT_INCLUDE_FLAGS) $(CFLAGS) examples/signal_npc_worker.c $(NN_OBJS) $(SIGNAL_FLIGHT_SIM_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@
endif

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_TEST_OBJS) $(NN_DATASET_TEST_STUBS_OBJ) $(NN_TEST_OBJS) $(NN_BLAS_OBJ) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(NN_TEST_OBJS) $(NN_BLAS_OBJ) $(PRIME_GAP_TEST_OBJS) $(NN_DATASET_TEST_STUBS_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(SANITIZE_TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_SANITIZE_OBJS) $(NN_DATASET_TEST_STUBS_OBJ) $(NN_SANITIZE_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) tests/test_nn.c $(NN_SANITIZE_OBJS) $(PRIME_GAP_SANITIZE_OBJS) $(NN_DATASET_TEST_STUBS_OBJ) $(LDFLAGS) $(LDLIBS) -fsanitize=address,undefined -o $@

$(TRAIN_CONTRACT_BRAIN): examples/train_contract_brain.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/train_contract_brain.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(GO_SELFPLAY): examples/go_selfplay.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/go_selfplay.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TRAIN_CROSS_DOMAIN): examples/train_cross_domain.c $(BUILD_DIR)/nn.o $(BUILD_DIR)/nn_blas.o $(BUILD_DIR)/nn_experiment.o $(BUILD_DIR)/nn_symbolic.o $(BUILD_DIR)/nn_bitgrid.o $(BUILD_DIR)/nn_tilebot.o $(BUILD_DIR)/nn_layer.o $(BUILD_DIR)/nn_program.o $(BUILD_DIR)/nn_hypothesis.o $(BUILD_DIR)/nn_protocol.o $(BUILD_DIR)/nn_dataset.o $(BUILD_DIR)/nn_json.o $(BUILD_DIR)/nn_text.o $(BUILD_DIR)/go_experiment.o | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGO_EXPERIMENT_ENABLED examples/train_cross_domain.c $(BUILD_DIR)/nn.o $(BUILD_DIR)/nn_blas.o $(BUILD_DIR)/nn_experiment.o $(BUILD_DIR)/nn_symbolic.o $(BUILD_DIR)/nn_bitgrid.o $(BUILD_DIR)/nn_tilebot.o $(BUILD_DIR)/nn_layer.o $(BUILD_DIR)/nn_program.o $(BUILD_DIR)/nn_hypothesis.o $(BUILD_DIR)/nn_protocol.o $(BUILD_DIR)/nn_dataset.o $(BUILD_DIR)/nn_json.o $(BUILD_DIR)/nn_text.o $(BUILD_DIR)/go_experiment.o $(LDFLAGS) $(LDLIBS) -o $@

# ── Go experiment schema ────────────────────────────────────────────────
GO_EXP_FLAGS = -DGO_EXPERIMENT_ENABLED

$(GO_EXPERIMENT): $(BUILD_DIR)/nn.o $(BUILD_DIR)/nn_blas.o $(BUILD_DIR)/nn_experiment.o $(BUILD_DIR)/nn_symbolic.o $(BUILD_DIR)/nn_bitgrid.o $(BUILD_DIR)/nn_tilebot.o $(BUILD_DIR)/nn_layer.o $(BUILD_DIR)/nn_program.o $(BUILD_DIR)/nn_hypothesis.o $(BUILD_DIR)/nn_protocol.o $(BUILD_DIR)/nn_dataset.o $(BUILD_DIR)/nn_json.o $(BUILD_DIR)/nn_text.o | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(GO_EXP_FLAGS) examples/go_experiment.c $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/go_experiment.o: examples/go_experiment.c include/nn.h include/nn_experiment.h include/nn_dataset.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGO_EXPERIMENT_LIBRARY_ONLY $(GO_EXP_FLAGS) -ffast-math -c $< -o $@
