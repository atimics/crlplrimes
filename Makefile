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
NN_HEADERS := include/nn.h include/nn_core.h include/nn_train.h include/nn_experiment.h include/nn_symbolic.h include/nn_program.h include/nn_hypothesis.h include/nn_protocol.h include/nn_json.h include/nn_text.h include/nn_grammar_discovery.h
PRIME_GAP_TASK_HEADERS := include/nn_tasks_prime_gap.h src/prime_gap/internal.h
PRIME_GAP_MODULE_SOURCES := $(wildcard src/prime_gap/*.c)
PRIME_GAP_TASK_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap/%.o,$(PRIME_GAP_MODULE_SOURCES))
PRIME_GAP_TEST_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap_test/%.o,$(PRIME_GAP_MODULE_SOURCES))
PRIME_GAP_SANITIZE_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap_sanitize/%.o,$(PRIME_GAP_MODULE_SOURCES))
NN_OBJ := $(BUILD_DIR)/nn.o
NN_EXPERIMENT_OBJ := $(BUILD_DIR)/nn_experiment.o
NN_SYMBOLIC_OBJ := $(BUILD_DIR)/nn_symbolic.o
NN_PROGRAM_OBJ := $(BUILD_DIR)/nn_program.o
NN_HYPOTHESIS_OBJ := $(BUILD_DIR)/nn_hypothesis.o
NN_PROTOCOL_OBJ := $(BUILD_DIR)/nn_protocol.o
NN_JSON_OBJ := $(BUILD_DIR)/nn_json.o
NN_TEXT_OBJ := $(BUILD_DIR)/nn_text.o
NN_GRAMMAR_DISCOVERY_OBJ := $(BUILD_DIR)/nn_grammar_discovery.o
NN_TEST_OBJ := $(BUILD_DIR)/nn_test.o
NN_EXPERIMENT_TEST_OBJ := $(BUILD_DIR)/nn_experiment_test.o
NN_SYMBOLIC_TEST_OBJ := $(BUILD_DIR)/nn_symbolic_test.o
NN_PROGRAM_TEST_OBJ := $(BUILD_DIR)/nn_program_test.o
NN_HYPOTHESIS_TEST_OBJ := $(BUILD_DIR)/nn_hypothesis_test.o
NN_PROTOCOL_TEST_OBJ := $(BUILD_DIR)/nn_protocol_test.o
NN_JSON_TEST_OBJ := $(BUILD_DIR)/nn_json_test.o
NN_TEXT_TEST_OBJ := $(BUILD_DIR)/nn_text_test.o
NN_SANITIZE_OBJ := $(BUILD_DIR)/nn_sanitize.o
NN_EXPERIMENT_SANITIZE_OBJ := $(BUILD_DIR)/nn_experiment_sanitize.o
NN_SYMBOLIC_SANITIZE_OBJ := $(BUILD_DIR)/nn_symbolic_sanitize.o
NN_PROGRAM_SANITIZE_OBJ := $(BUILD_DIR)/nn_program_sanitize.o
NN_HYPOTHESIS_SANITIZE_OBJ := $(BUILD_DIR)/nn_hypothesis_sanitize.o
NN_PROTOCOL_SANITIZE_OBJ := $(BUILD_DIR)/nn_protocol_sanitize.o
NN_JSON_SANITIZE_OBJ := $(BUILD_DIR)/nn_json_sanitize.o
NN_TEXT_SANITIZE_OBJ := $(BUILD_DIR)/nn_text_sanitize.o
NN_OBJS := $(NN_OBJ) $(NN_EXPERIMENT_OBJ) $(NN_SYMBOLIC_OBJ) $(NN_PROGRAM_OBJ) $(NN_HYPOTHESIS_OBJ) $(NN_PROTOCOL_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ)
NN_TEST_OBJS := $(NN_TEST_OBJ) $(NN_EXPERIMENT_TEST_OBJ) $(NN_SYMBOLIC_TEST_OBJ) $(NN_PROGRAM_TEST_OBJ) $(NN_HYPOTHESIS_TEST_OBJ) $(NN_PROTOCOL_TEST_OBJ) $(NN_JSON_TEST_OBJ) $(NN_TEXT_TEST_OBJ)
NN_SANITIZE_OBJS := $(NN_SANITIZE_OBJ) $(NN_EXPERIMENT_SANITIZE_OBJ) $(NN_SYMBOLIC_SANITIZE_OBJ) $(NN_PROGRAM_SANITIZE_OBJ) $(NN_HYPOTHESIS_SANITIZE_OBJ) $(NN_PROTOCOL_SANITIZE_OBJ) $(NN_JSON_SANITIZE_OBJ) $(NN_TEXT_SANITIZE_OBJ)
XOR_EXAMPLE := $(BUILD_DIR)/xor
ACTION_SCORE_EXAMPLE := $(BUILD_DIR)/action_score
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
ARC_AGI := $(BUILD_DIR)/arc_agi
SIGNAL_FLIGHT := $(BUILD_DIR)/signal_flight
SIGNAL_FLIGHT_ROLLOUT := $(BUILD_DIR)/signal_flight_rollout
SIGNAL_MINING := $(BUILD_DIR)/signal_mining
HAIKU_LM := $(BUILD_DIR)/haiku_lm
HAIKU_WORD_LM := $(BUILD_DIR)/haiku_word_lm
HAIKU_UNDERSTANDING := $(BUILD_DIR)/haiku_understanding
NN_GRAMMAR_DISCOVERY_RUNTIME := $(BUILD_DIR)/neural_grammar_discovery_runtime
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
PRIME_GATE_SEED_AUDIT_DIR ?= $(BUILD_DIR)/prime_gate_seed_audit
PRIME_GATE_SEED_AUDIT_SEEDS ?= 20240511:20240515
RESULTS_DIR ?= results/latest
SUBSTRATE_EXTENSION_RESULTS_DIR ?= results/2026-05-17-substrate-extensions
MULTITASK_ROADMAP_RESULTS_DIR ?= results/2026-05-17-multitask-roadmap
SIGNAL_FLIGHT_RESULTS_DIR ?= results/2026-05-17-signal-flight
TUNING_DIR ?= build/$(REAL)/neurosymbolic_tuning
REJECTED_CANDIDATE_DIR ?= build/$(REAL)/rejected_candidate_learning
REJECTED_CANDIDATE_SEED_AUDIT_DIR ?= build/$(REAL)/rejected_candidate_seed_audit
REJECTED_CANDIDATE_SEEDS ?= 20240511:20240513
HYPOTHESIS_LIBRARY_DIR ?= build/$(REAL)/hypothesis_library
TRACE_DETERMINISM_AUDIT ?= $(BUILD_DIR)/traces/trace_determinism_audit.jsonl
TRACE_DETERMINISM_CANONICAL ?= $(BUILD_DIR)/traces/prime_gap.semantic.jsonl
TRACE_DETERMINISM_COMPARE ?= $(RESULTS_DIR)/trace_determinism.jsonl
TRACE_DETERMINISM_CI ?= $(BUILD_DIR)/traces/trace_determinism_ci.jsonl
TRACE_DETERMINISM_RESULTS_CANONICAL ?= $(RESULTS_DIR)/prime_gap_trace.semantic.jsonl
HYPOTHESIS_LIBRARY_PROMOTION_TASK_IDS ?= move_to_anchor,move_to_anchor_x
HYPOTHESIS_LIBRARY_HELDOUT_TASK_IDS ?= bbox_panel_selection,connect_aligned,extend_aligned_sequence,fill_enclosed,move_to_anchor_both_positive,move_to_anchor_y,object_extract,panel_selection,recolor_by_reference,translate
HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS ?= --task-dir tests/fixtures/arc_hypothesis --task-ids $(HYPOTHESIS_LIBRARY_HELDOUT_TASK_IDS) --max-tasks 10 --program-budget-tight-grid --loader-summary-only --induction --min-induction-evidence 8 --hypothesis-top-k 1 --disable-hypothesis-coverage-selection --disable-hypothesis-structural-score --disable-hypothesis-train-filter
HYPOTHESIS_LIBRARY_SPLIT_SWEEP_DIR ?= $(BUILD_DIR)/hypothesis_library_split_sweep
HYPOTHESIS_LIBRARY_FEEDBACK_LOOP_DIR ?= $(BUILD_DIR)/hypothesis_library_feedback_loop
HYPOTHESIS_LIBRARY_PUBLIC_DIR ?= $(BUILD_DIR)/hypothesis_library_public
HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR ?=
HYPOTHESIS_LIBRARY_PUBLIC_TRAIN_MAX_TASKS ?= 400
HYPOTHESIS_LIBRARY_PUBLIC_EVAL_MAX_TASKS ?= 400
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR ?= $(BUILD_DIR)/hypothesis_library_public_filter
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_MAX_TASKS ?= 400
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_SOURCE_MAX_TASKS ?= $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_MAX_TASKS)
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_HELDOUT_MAX_TASKS ?= $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_MAX_TASKS)
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_SOURCE_COUNT ?= 24
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_HELDOUT_COUNT ?= 48
HYPOTHESIS_LIBRARY_PUBLIC_FILTER_EXTRA ?=
HYPOTHESIS_LIBRARY_PUBLIC_REPORT_TOP_K ?= 20
HYPOTHESIS_LIBRARY_PUBLIC_TRIAGE_DIR ?= $(BUILD_DIR)/hypothesis_library_public_triage
HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR ?= $(BUILD_DIR)/hypothesis_library_neural_discovery
HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_CLUSTERS ?= 12
HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_VERIFY ?= 0
HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATIONS_PER_CLUSTER ?= 2
HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATION_VERIFY ?= 1
HYPOTHESIS_LIBRARY_NEURAL_FAILURE_PROOF_DIR ?= $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR)/failure_proof
TINY_NET_PLATEAU_AUDIT_DIR ?= $(BUILD_DIR)/tiny_net_plateau_audit
TINY_NET_AUDIT_SEEDS ?= 20240516,20240517,20240518
TINY_NET_AUDIT_CAPACITIES ?= 8x4,16x8,32x16,64x32
TINY_NET_AUDIT_EPOCH_GRID ?= 1,2,4,8,16,32
TINY_NET_AUDIT_STEPS_PER_EPOCH ?= 12000
TINY_NET_AUDIT_EVAL_SAMPLES ?= 1000
ARC_AGI_TASK_DIR ?=
ARC_AGI_MAX_TASKS ?= 400
ARC_EVIDENCE_SURFACE_DIR ?= $(BUILD_DIR)/arc_evidence_surface
ARC_FAMILY_DIAGNOSTICS ?= results/2026-05-17-arc-induction-task-diagnostics.tasks.jsonl
ARC_FAMILY_DISCOVERY_DIR ?= $(BUILD_DIR)/arc_family_discovery
PAPER_AGENT_PROMPT ?= paper/agent_candidates/next_prompt.md
PAPER_AGENT_OUTPUT ?= paper/agent_candidates/llm_proposals.jsonl
PAPER_AGENT_MODEL ?= manual-llm
PAPER_AGENT_LLM_CMD ?=
SIGNAL_FLIGHT_SIGNAL_DIR ?= ../signal
SIGNAL_FLIGHT_TRACE_BIN ?= $(SIGNAL_FLIGHT_SIGNAL_DIR)/build/flight_trace
SIGNAL_FLIGHT_TRAIN_TRACE ?= $(BUILD_DIR)/signal_flight_train.csv
SIGNAL_FLIGHT_TEST_TRACE ?= $(BUILD_DIR)/signal_flight_test.csv
SIGNAL_FLIGHT_RESULT_DIR ?= $(BUILD_DIR)/signal_flight_experiment
SIGNAL_FLIGHT_EPISODES ?= 32
SIGNAL_FLIGHT_TEST_EPISODES ?= 12
SIGNAL_FLIGHT_TICKS ?= 160
SIGNAL_FLIGHT_SEED ?= 4242
SIGNAL_FLIGHT_TEST_SEED ?= 4243
SIGNAL_FLIGHT_COUNTERFACTUAL ?= 1
SIGNAL_FLIGHT_SAMPLE_STRIDE ?= 24
SIGNAL_FLIGHT_HORIZON_TICKS ?= 48
SIGNAL_FLIGHT_EXPLORE_RANDOM ?= 0.35
SIGNAL_FLIGHT_PAIN_PROBE ?= 0
SIGNAL_FLIGHT_WALL_PROBE ?= 1
SIGNAL_FLIGHT_EPOCHS ?= 3
SIGNAL_FLIGHT_MAX_TRAIN ?= 120000
SIGNAL_FLIGHT_MAX_TEST ?= 40000
SIGNAL_FLIGHT_LEARNING_RATE ?= 0.004
SIGNAL_FLIGHT_TRAIN_EXTRA ?=
SIGNAL_LIVE_BRAIN_SERVER ?= ws://127.0.0.1:9091/ws
SIGNAL_LIVE_BRAIN_SESSION ?= $(BUILD_DIR)/signal_live_brain/session.json
SIGNAL_LIVE_BRAIN_LOG ?= $(BUILD_DIR)/signal_live_brain/live.jsonl
SIGNAL_LIVE_BRAIN_CALLSIGN ?= BRN-001
SIGNAL_LIVE_BRAIN_MAX_MS ?= 0
SIGNAL_LIVE_BRAIN_CHECKPOINT ?=
SIGNAL_LIVE_BRAIN_MODE ?= shadow_neural
SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG = $(if $(SIGNAL_LIVE_BRAIN_CHECKPOINT),--brain-checkpoint "$(SIGNAL_LIVE_BRAIN_CHECKPOINT)",)
SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT ?=
SIGNAL_LIVE_BRAIN_OPTION_MODE ?= off
SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG = $(if $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT),--option-checkpoint "$(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT)",)
SIGNAL_LIVE_BRAIN_PROTOCOL_ROOT ?= $(SIGNAL_FLIGHT_SIGNAL_DIR)
SIGNAL_LIVE_BRAIN_PROTOCOL_CHECK ?= 1
SIGNAL_LIVE_BRAIN_PROTOCOL_ARG = $(if $(filter 0,$(SIGNAL_LIVE_BRAIN_PROTOCOL_CHECK)),--skip-protocol-check,--protocol-root "$(SIGNAL_LIVE_BRAIN_PROTOCOL_ROOT)")
SIGNAL_LIVE_BRAIN_EXTRA ?=
SIGNAL_LIVE_SWARM_COUNT ?= 4
SIGNAL_LIVE_SWARM_MIN_ACTIVE ?= $(SIGNAL_LIVE_SWARM_COUNT)
SIGNAL_LIVE_SWARM_MAX_MS ?= 30000
SIGNAL_LIVE_SWARM_OUT_DIR ?= $(BUILD_DIR)/signal_live_swarm
SIGNAL_LIVE_SWARM_CALLSIGN_PREFIX ?= BRN
SIGNAL_LIVE_SWARM_STAGGER_MS ?= 100
SIGNAL_LIVE_STATUS_LOG ?= $(SIGNAL_LIVE_BRAIN_LOG)
SIGNAL_LIVE_STATUS_INTERVAL_MS ?= 1000
SIGNAL_LIVE_STATUS_EXTRA ?=
SIGNAL_LIVE_ACTIVE_LOG ?= $(SIGNAL_LIVE_STATUS_LOG)
SIGNAL_LIVE_ACTIVE_DIR ?= $(BUILD_DIR)/signal_live_active_summary
SIGNAL_LIVE_ACTIVE_WINDOW_ROWS ?= 250
SIGNAL_LIVE_ACTIVE_STALE_MS ?= 3000
SIGNAL_SHADOW_REPLAY_DIR ?= $(BUILD_DIR)/signal_shadow_replay_audit
SIGNAL_SHADOW_REPLAY_HORIZON_MS ?= 500
SIGNAL_FLIGHT_ROLLOUT_DIR ?= $(BUILD_DIR)/signal_flight_rollout_eval
SIGNAL_FLIGHT_ROLLOUT_SUMMARY ?= $(SIGNAL_FLIGHT_ROLLOUT_DIR)/summary.jsonl
SIGNAL_FLIGHT_ROLLOUT_CHECKPOINT ?= $(SIGNAL_FLIGHT_RESULT_DIR)/signal_flight.nnckpt
SIGNAL_FLIGHT_ROLLOUT_EPISODES ?= 24
SIGNAL_FLIGHT_ROLLOUT_TICKS ?= 900
SIGNAL_FLIGHT_ROLLOUT_SEED ?= 4242
SIGNAL_MINING_RESULT_DIR ?= $(BUILD_DIR)/signal_mining_experiment
SIGNAL_MINING_TRAIN_GROUPS ?= 18
SIGNAL_MINING_TEST_GROUPS ?= 6
SIGNAL_MINING_HORIZON_TICKS ?= 1200
SIGNAL_MINING_EPOCHS ?= 4
SIGNAL_MINING_SEED ?= 5252
SIGNAL_MINING_LEARNING_RATE ?= 0.004
HAIKU_LM_CORPUS ?= data/haiku_tiny.txt
HAIKU_LM_DIR ?= $(BUILD_DIR)/haiku_lm_artifacts
HAIKU_LM_EPOCHS ?= 8
HAIKU_LM_LEARNING_RATE ?= 0.006
HAIKU_LM_SEED ?= 20240518
HAIKU_LM_WORD_LIST ?=
HAIKU_LM_LEXICON_WORD_LIST ?= data/wordlists/enable2k.txt
HAIKU_LM_LEXICON_GATE ?= 1
HAIKU_LM_WORD_LIST_ARG = $(if $(strip $(HAIKU_LM_WORD_LIST)),--word-list "$(HAIKU_LM_WORD_LIST)" --lexicon-gate $(HAIKU_LM_LEXICON_GATE),)
HAIKU_CORPUS_SYNTHETIC ?= data/haiku_synthetic_5000.txt
HAIKU_CORPUS_COUNT ?= 5000
HAIKU_CORPUS_SEED ?= 20240518
HAIKU_CORPUS_SUMMARY ?= $(BUILD_DIR)/haiku_corpus_synthetic/summary.json
HAIKU_CORPUS_VOCAB ?= data/wordlists/haiku_synthetic_vocab.txt
HAIKU_LM_SYNTHETIC_EPOCHS ?= 3
HAIKU_HF_RAW ?= data/haiku_open/raw/huggingface
HAIKU_HF_CLEAN ?= data/haiku_open/clean/hf_haiku_clean.txt
HAIKU_HF_VOCAB ?= data/haiku_open/clean/hf_vocab.txt
HAIKU_HF_MANIFEST ?= data/haiku_open/clean/hf_manifest.json
HAIKU_HF_MAX_POEMS ?= 8000
HAIKU_HF_FILTER_WORD_LIST ?= 0
HAIKU_HF_DICTIONARY_CLEAN ?= data/haiku_open/clean/hf_dictionary_clean.txt
HAIKU_HF_DICTIONARY_VOCAB ?= data/haiku_open/clean/hf_dictionary_vocab.txt
HAIKU_HF_DICTIONARY_MANIFEST ?= data/haiku_open/clean/hf_dictionary_manifest.json
HAIKU_LM_HF_EPOCHS ?= 2
HAIKU_WORD_LM_DIR ?= $(BUILD_DIR)/haiku_word_lm_artifacts
HAIKU_WORD_LM_EPOCHS ?= 4
HAIKU_WORD_LM_HF_EPOCHS ?= 2
HAIKU_WORD_LM_SYNTHETIC_EPOCHS ?= 3
HAIKU_WORD_LM_LEARNING_RATE ?= 0.006
HAIKU_WORD_LM_MAX_CANDIDATES ?= 128
HAIKU_WORD_LM_WORD_LIST ?=
HAIKU_WORD_LM_STRICT_ADJACENCY ?= 1
HAIKU_WORD_LM_WORD_LIST_ARG = $(if $(strip $(HAIKU_WORD_LM_WORD_LIST)),--word-list "$(HAIKU_WORD_LM_WORD_LIST)",)
HAIKU_UNDERSTANDING_DIR ?= $(BUILD_DIR)/haiku_understanding_artifacts
HAIKU_UNDERSTANDING_EPOCHS ?= 8
HAIKU_UNDERSTANDING_HF_EPOCHS ?= 4
HAIKU_UNDERSTANDING_SYNTHETIC_EPOCHS ?= 4
HAIKU_UNDERSTANDING_LEARNING_RATE ?= 0.006
HAIKU_UNDERSTANDING_TRACE_LIMIT ?= 2000
NEURAL_GRAMMAR_DISCOVERY_DIR ?= $(BUILD_DIR)/neural_grammar_discovery
NEURAL_GRAMMAR_DISCOVERY_DOMAINS ?= arc,haiku,program_synthesis,grid_path,signal_mining
NEURAL_GRAMMAR_DISCOVERY_ARC_PROOF ?= $(HYPOTHESIS_LIBRARY_NEURAL_FAILURE_PROOF_DIR)/task_proof_rows.jsonl
NEURAL_GRAMMAR_DISCOVERY_HAIKU_WORD_TRACE ?= $(HAIKU_WORD_LM_DIR)/smoke_trace.jsonl
NEURAL_GRAMMAR_DISCOVERY_HAIKU_UNDERSTANDING_TRACE ?= $(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl
NEURAL_GRAMMAR_DISCOVERY_PROGRAM_SYNTHESIS_TRACE ?= $(NEURAL_GRAMMAR_DISCOVERY_DIR)/program_synthesis_source.jsonl
NEURAL_GRAMMAR_DISCOVERY_GRID_TRACE ?= $(BUILD_DIR)/grid_path.jsonl
NEURAL_GRAMMAR_DISCOVERY_SIGNAL_MINING_TRACE ?= $(BUILD_DIR)/signal_mining_smoke/trace.jsonl
NEURAL_GRAMMAR_DISCOVERY_ARC_PROBE_COMPARE_LIMIT ?= 64
NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_TRIAGE ?= $(HYPOTHESIS_LIBRARY_PUBLIC_TRIAGE_DIR)/coverage_triage.jsonl
NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_LIMIT ?= 64
NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_PER_FAMILY ?= 12
NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_COMPARE_LIMIT ?= 64
NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_OP_HINTS_PER_TASK ?= 3
LEARNED_GRAMMAR_COVERAGE_DIR ?= $(BUILD_DIR)/learned_grammar_coverage
LEARNED_GRAMMAR_COVERAGE_ARC_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_induction_smoke.jsonl
LEARNED_GRAMMAR_COVERAGE_REPORT ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/coverage.jsonl
LEARNED_GRAMMAR_COVERAGE_EXTENDED_REPORT ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/coverage_extended.jsonl
LEARNED_GRAMMAR_COVERAGE_BUDGET ?= data/learned_grammar_coverage_budgets.jsonl
LEARNED_GRAMMAR_COVERAGE_BUDGET_ARG = $(if $(strip $(LEARNED_GRAMMAR_COVERAGE_BUDGET)),--budget "$(LEARNED_GRAMMAR_COVERAGE_BUDGET)",)
LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR ?= $(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)
LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS ?= 64
LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_train.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_eval.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR ?= $(NEURAL_GRAMMAR_DISCOVERY_DIR)/arc_sketch_program_heldout_prior.tsv
LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_fixture_visual_only.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_train_visual_only.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_eval_visual_only.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_COMPARE_REPORT ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_visual_compare.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_visual_curriculum
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_candidates.tsv
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_EXAMPLES ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_examples.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SUMMARY ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_summary.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SOURCES ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_sources.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PRIOR ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_prior.tsv
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_attempts.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR ?= $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)/arc_visual_curriculum_combined_prior.tsv
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_train_curriculum_visual_only.jsonl
LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_eval_curriculum_visual_only.jsonl
ARC_VISUAL_ACTIVE_CURRICULUM_DIR ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_visual_active_curriculum
ARC_VISUAL_ACTIVE_CURRICULUM_SUMMARY ?= $(ARC_VISUAL_ACTIVE_CURRICULUM_DIR)/summary.jsonl
ARC_VISUAL_ACTIVE_CURRICULUM_ITERATIONS ?= 3
ARC_VISUAL_CAPACITY_SWEEP_DIR ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_visual_capacity_sweep
ARC_VISUAL_CAPACITY_SWEEP_REPORT ?= $(ARC_VISUAL_CAPACITY_SWEEP_DIR)/capacity_sweep.jsonl
ARC_VISUAL_CAPACITY_SWEEP_MARKDOWN ?= $(ARC_VISUAL_CAPACITY_SWEEP_DIR)/capacity_sweep.md
ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS ?= $(NEURAL_GRAMMAR_DISCOVERY_DIR)/runtime/arc_visual_pipeline_neural_attempts.tsv
ARC_VISUAL_CAPACITY_SWEEP_EPOCHS ?= 16,64,128
ARC_VISUAL_CAPACITY_SWEEP_LEARNING_RATES ?= 0.08
ARC_VISUAL_CAPACITY_SWEEP_MAX_PER_TASK ?= 8
ARC_VISUAL_CAPACITY_SWEEP_HIDDEN_SIZES ?= 0,24,64
ARC_VISUAL_HOLE_PROPOSER_DIR ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_visual_hole_proposer
ARC_VISUAL_HOLE_PROPOSER_CANDIDATES ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_candidates.tsv
ARC_VISUAL_HOLE_PROPOSER_EXAMPLES ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_examples.jsonl
ARC_VISUAL_HOLE_PROPOSER_SUMMARY ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_summary.jsonl
ARC_VISUAL_HOLE_PROPOSER_SOURCES ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_sources.jsonl
ARC_VISUAL_HOLE_PROPOSER_PRIOR ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_prior.tsv
ARC_VISUAL_HOLE_PROPOSER_ATTEMPTS ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_attempts.jsonl
ARC_VISUAL_HOLE_PROPOSER_COMBINED_PRIOR ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_combined_prior.tsv
ARC_VISUAL_HOLE_PROPOSER_PUBLIC_TRAIN_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_train_hole_proposer_visual_only.jsonl
ARC_VISUAL_HOLE_PROPOSER_PUBLIC_EVAL_LOG ?= $(LEARNED_GRAMMAR_COVERAGE_DIR)/arc_agi_public_eval_hole_proposer_visual_only.jsonl
ARC_VISUAL_HOLE_PROPOSER_COMPARE_REPORT ?= $(ARC_VISUAL_HOLE_PROPOSER_DIR)/arc_visual_hole_compare.jsonl
EXPERIMENT_HEALTH_DIR ?= $(BUILD_DIR)/experiment_health
EXPERIMENT_HEALTH_REPORT ?= $(EXPERIMENT_HEALTH_DIR)/experiment_health.jsonl
EXPERIMENT_HEALTH_EXTENDED_REPORT ?= $(EXPERIMENT_HEALTH_DIR)/experiment_health_extended.jsonl
EXPERIMENT_HEALTH_SIGNAL_FLIGHT_SUMMARY ?= $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.jsonl
EXPERIMENT_HEALTH_SIGNAL_FLIGHT_ROLLOUT ?= $(SIGNAL_FLIGHT_RESULTS_DIR)/rollout_summary.jsonl
EXPERIMENT_HEALTH_SIGNAL_LIVE_ACTIVE ?= $(SIGNAL_FLIGHT_RESULTS_DIR)/live_active_summary.jsonl
EXPERIMENT_HEALTH_SIGNAL_SHADOW_REPLAY ?= $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_replay_audit.jsonl
EXPERIMENT_HEALTH_NEURAL_GRAMMAR_SUMMARY ?= $(NEURAL_GRAMMAR_DISCOVERY_DIR)/summary.jsonl
PROTOCOL_SPINE_DIR ?= $(BUILD_DIR)/protocol_spine
PROTOCOL_SPINE_REPORT ?= $(PROTOCOL_SPINE_DIR)/protocol_spine.jsonl
PROTOCOL_SPINE_SUMMARY ?= $(PROTOCOL_SPINE_DIR)/summary.jsonl
PROTOCOL_SPINE_GRAMMAR_SKETCH_EXAMPLES ?= $(NEURAL_GRAMMAR_DISCOVERY_DIR)/grammar_sketch_examples.jsonl
PROTOCOL_SPINE_ARC_HYPOTHESIS_EVIDENCE ?= $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl
PROTOCOL_OPERATOR_DIR ?= $(BUILD_DIR)/protocol_operator_promotion
PROTOCOL_OPERATOR_REPORT ?= $(PROTOCOL_OPERATOR_DIR)/operator_promotions.jsonl
PROTOCOL_OPERATOR_PRIOR ?= $(PROTOCOL_OPERATOR_DIR)/operator_prior.tsv
PROTOCOL_OPERATOR_SUMMARY ?= $(PROTOCOL_OPERATOR_DIR)/summary.jsonl
PROTOCOL_OPERATOR_MARKDOWN ?= $(PROTOCOL_OPERATOR_DIR)/summary.md
PROTOCOL_OPERATOR_MIN_DOMAINS ?= 2
PROTOCOL_OPERATOR_MIN_SUPPORT_ROWS ?= 4
NUMERIC_CERTIFICATE_DIR ?= $(BUILD_DIR)/numeric_certificates
NUMERIC_CERTIFICATE_TARGETS ?= action-score prime-gap-trace prime-rule-search-smoke delayed-claim-eval grid-path-eval multitask-domain symbolic-domains program-synthesis program-synthesis-baselines modular-gate-search-smoke arc-agi-induction-smoke haiku-lm-smoke haiku-word-lm-smoke haiku-understanding-smoke signal-mining-smoke
NUMERIC_CERTIFICATE_REPORT ?= $(NUMERIC_CERTIFICATE_DIR)/report.jsonl
NUMERIC_CERTIFICATE_COMPARE_REUSE_FLOAT ?= 0
NUMERIC_CERTIFICATE_STRICT_DIR ?= $(BUILD_DIR)/numeric_certificates_strict
NUMERIC_CERTIFICATE_STRICT_REPORT ?= $(NUMERIC_CERTIFICATE_STRICT_DIR)/report.jsonl
NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER ?= 100
NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER_DELTA ?= 10
NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT ?= 1000000
NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT_DELTA ?= 1000000
NUMERIC_CERTIFICATE_STRICT_MAX_SHRINK_STEPS ?= 4
NUMERIC_CERTIFICATE_STRICT_ALLOWED_SHRINK_STEPS ?= 0
NUMERIC_CERTIFICATE_COMPARE_DIR ?= build/numeric_certificate_compare
NUMERIC_CERTIFICATE_BUDGET ?= data/numeric_certificate_budgets.jsonl
NUMERIC_CERTIFICATE_BUDGET_ARG = $(if $(strip $(NUMERIC_CERTIFICATE_BUDGET)),--budget "$(NUMERIC_CERTIFICATE_BUDGET)",)

SIGNAL_FLIGHT_INCLUDE_FLAGS := \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/server \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl
SIGNAL_FLIGHT_SIM_SOURCES := \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/game_sim.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_ai.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/signal_brain.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/signal_contract_brain.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/gossip.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_autopilot.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_flight.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_nav.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_save.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_catalog.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/station_authority.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/chain_log.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/chain_log_verify.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_receipt_issue.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_asteroid.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_physics.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_production.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_construction.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_mining.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_anchor.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_ship.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/commodity.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/manifest.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/ship.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/economy.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/asteroid.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/rng.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/belt.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/module_schema.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/station_util.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/cargo_receipt.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/handoff_ticket.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/tractor.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/laser.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/trade_paging.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/tweetnacl.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/randombytes.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/signal_crypto_tweetnacl.c

SIGNAL_FLIGHT_TRACE_EXTRA :=
ifeq ($(SIGNAL_FLIGHT_COUNTERFACTUAL),1)
SIGNAL_FLIGHT_TRACE_EXTRA += --counterfactual --sample-stride $(SIGNAL_FLIGHT_SAMPLE_STRIDE) --horizon-ticks $(SIGNAL_FLIGHT_HORIZON_TICKS) --explore-random $(SIGNAL_FLIGHT_EXPLORE_RANDOM)
endif
ifeq ($(SIGNAL_FLIGHT_PAIN_PROBE),1)
SIGNAL_FLIGHT_TRACE_EXTRA += --pain-probe
endif
ifeq ($(SIGNAL_FLIGHT_WALL_PROBE),1)
SIGNAL_FLIGHT_TRACE_EXTRA += --wall-probe
endif

.PHONY: all run action-score experiment-schemas prime-gap prime-gap-trace prime-gap-replay prime-gap-eval prime-rule-search prime-rule-search-baselines prime-rule-search-conditional prime-rule-search-smoke prime-gap-sweep prime-gap-regularization-sweep prime-gap-experiments prime-gate-seed-audit trace-determinism-audit trace-determinism-compare trace-determinism-ci delayed-claim delayed-claim-trace delayed-claim-replay delayed-claim-eval delayed-claim-self-test grid-path grid-path-rule-search grid-path-budget-sweep grid-path-trace grid-path-replay grid-path-eval multitask-domain multitask-adapter-heads tiny-net-plateau-audit symbolic-domains program-synthesis program-synthesis-baselines modular-gate-search modular-gate-search-smoke arc-agi arc-agi-loader arc-agi-induction-smoke arc-agi-induction-public arc-agi-evidence-surface arc-family-discovery rejected-candidate-learning rejected-candidate-seed-audit hypothesis-library-promotion hypothesis-library-ablation hypothesis-library-split-sweep hypothesis-library-feedback-loop hypothesis-library-public hypothesis-library-public-filter hypothesis-library-public-report hypothesis-library-public-triage hypothesis-library-neural-discovery hypothesis-library-neural-failure-proof paper-grade-evidence paper-grade-evidence-core signal-flight-data signal-flight signal-flight-pain signal-flight-rollout-eval signal-mining signal-mining-smoke haiku-corpus-hf haiku-corpus-hf-dictionary haiku-corpus-synthetic haiku-lm haiku-lm-hf haiku-lm-lexicon haiku-lm-smoke haiku-lm-synthetic haiku-understanding haiku-understanding-hf haiku-understanding-hf-dictionary haiku-understanding-smoke haiku-understanding-synthetic haiku-word-lm haiku-word-lm-hf haiku-word-lm-hf-dictionary haiku-word-lm-smoke haiku-word-lm-synthetic signal-live-brain signal-live-brain-smoke signal-live-swarm signal-live-status signal-live-active-summary signal-flight-report signal-shadow-replay-audit learned-grammar-coverage learned-grammar-coverage-public learned-grammar-coverage-arc-visual-compare arc-visual-active-curriculum arc-visual-capacity-sweep arc-visual-hole-proposer learned-grammar-coverage-extended protocol-spine protocol-operator-promotion experiment-health experiment-health-extended experiment-matrix numeric-certificate-experiments numeric-certificate-report numeric-certificate-strict numeric-certificate-compare numeric-certificate-budget-refresh numeric-certificate-mandatory results experimental-results paper-pdf paper-data paper-claims paper-agent-propose paper-pre-commit install-hooks paper-support editorial-pr-check paper-ci neurosymbolic-tune python-check benchmark test sanitize check experimental-check clean

all: $(XOR_EXAMPLE) $(ACTION_SCORE_EXAMPLE) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH) $(DELAYED_CLAIM) $(GRID_PATH) $(MULTITASK_DOMAIN) $(SYMBOLIC_DOMAINS) $(PROGRAM_SYNTHESIS) $(MODULAR_GATE_SEARCH) $(ARC_AGI) $(SIGNAL_FLIGHT) $(SIGNAL_MINING) $(HAIKU_LM) $(HAIKU_WORD_LM) $(HAIKU_UNDERSTANDING) $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(NN_OBJ): src/nn.c $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn.c -o $@

$(NN_EXPERIMENT_OBJ): src/nn_experiment.c include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_experiment.c -o $@

$(NN_SYMBOLIC_OBJ): src/nn_symbolic.c include/nn_symbolic.h include/nn_core.h include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_symbolic.c -o $@

$(NN_PROGRAM_OBJ): src/nn_program.c include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_program.c -o $@

$(NN_HYPOTHESIS_OBJ): src/nn_hypothesis.c include/nn_hypothesis.h include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_hypothesis.c -o $@

$(NN_PROTOCOL_OBJ): src/nn_protocol.c include/nn_protocol.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_protocol.c -o $@

$(NN_JSON_OBJ): src/nn_json.c include/nn_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_json.c -o $@

$(NN_TEXT_OBJ): src/nn_text.c include/nn_text.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_text.c -o $@

$(NN_GRAMMAR_DISCOVERY_OBJ): src/nn_grammar_discovery.c include/nn_grammar_discovery.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/nn_grammar_discovery.c -o $@

$(NN_TEST_OBJ): src/nn.c $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn.c -o $@

$(NN_EXPERIMENT_TEST_OBJ): src/nn_experiment.c include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_experiment.c -o $@

$(NN_SYMBOLIC_TEST_OBJ): src/nn_symbolic.c include/nn_symbolic.h include/nn_core.h include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_symbolic.c -o $@

$(NN_PROGRAM_TEST_OBJ): src/nn_program.c include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_program.c -o $@

$(NN_HYPOTHESIS_TEST_OBJ): src/nn_hypothesis.c include/nn_hypothesis.h include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_hypothesis.c -o $@

$(NN_PROTOCOL_TEST_OBJ): src/nn_protocol.c include/nn_protocol.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_protocol.c -o $@

$(NN_JSON_TEST_OBJ): src/nn_json.c include/nn_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_json.c -o $@

$(NN_TEXT_TEST_OBJ): src/nn_text.c include/nn_text.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c src/nn_text.c -o $@

$(NN_SANITIZE_OBJ): src/nn.c $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn.c -o $@

$(NN_EXPERIMENT_SANITIZE_OBJ): src/nn_experiment.c include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_experiment.c -o $@

$(NN_SYMBOLIC_SANITIZE_OBJ): src/nn_symbolic.c include/nn_symbolic.h include/nn_core.h include/nn_experiment.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_symbolic.c -o $@

$(NN_PROGRAM_SANITIZE_OBJ): src/nn_program.c include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_program.c -o $@

$(NN_HYPOTHESIS_SANITIZE_OBJ): src/nn_hypothesis.c include/nn_hypothesis.h include/nn_program.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_hypothesis.c -o $@

$(NN_PROTOCOL_SANITIZE_OBJ): src/nn_protocol.c include/nn_protocol.h include/nn_symbolic.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_protocol.c -o $@

$(NN_JSON_SANITIZE_OBJ): src/nn_json.c include/nn_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_json.c -o $@

$(NN_TEXT_SANITIZE_OBJ): src/nn_text.c include/nn_text.h include/nn_core.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c src/nn_text.c -o $@

$(BUILD_DIR)/obj/prime_gap/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/obj/prime_gap_test/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/obj/prime_gap_sanitize/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) -c $< -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(ACTION_SCORE_EXAMPLE): examples/action_score.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/action_score.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(HAIKU_LM): examples/haiku_lm.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_lm.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(HAIKU_WORD_LM): examples/haiku_word_lm.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_word_lm.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(HAIKU_UNDERSTANDING): examples/haiku_understanding.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/haiku_understanding.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

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

$(MULTITASK_DOMAIN): examples/multitask_domain.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/multitask_domain.c $(NN_OBJS) $(PRIME_GAP_TASK_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SYMBOLIC_DOMAINS): examples/symbolic_domains.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/symbolic_domains.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(PROGRAM_SYNTHESIS): examples/program_synthesis.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/program_synthesis.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(MODULAR_GATE_SEARCH): examples/modular_gate_search.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/modular_gate_search.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(ARC_AGI): examples/arc_agi.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/arc_agi.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(NN_GRAMMAR_DISCOVERY_RUNTIME): examples/neural_grammar_discovery.c $(NN_GRAMMAR_DISCOVERY_OBJ) include/nn_grammar_discovery.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/neural_grammar_discovery.c $(NN_GRAMMAR_DISCOVERY_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_FLIGHT): examples/signal_flight.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/signal_flight.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_FLIGHT_ROLLOUT): examples/signal_flight_rollout.c $(NN_OBJS) $(NN_HEADERS) $(SIGNAL_FLIGHT_SIM_SOURCES) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SIGNAL_FLIGHT_INCLUDE_FLAGS) $(CFLAGS) examples/signal_flight_rollout.c $(NN_OBJS) $(SIGNAL_FLIGHT_SIM_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_MINING): examples/signal_mining.c $(NN_OBJS) $(NN_HEADERS) $(SIGNAL_FLIGHT_SIM_SOURCES) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SIGNAL_FLIGHT_INCLUDE_FLAGS) $(CFLAGS) examples/signal_mining.c $(NN_OBJS) $(SIGNAL_FLIGHT_SIM_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_TEST_OBJS) $(NN_TEST_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(NN_TEST_OBJS) $(PRIME_GAP_TEST_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SANITIZE_TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_SANITIZE_OBJS) $(NN_SANITIZE_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) tests/test_nn.c $(NN_SANITIZE_OBJS) $(PRIME_GAP_SANITIZE_OBJS) $(LDFLAGS) $(SANITIZE_LDFLAGS) $(LDLIBS) -o $@

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

action-score: $(ACTION_SCORE_EXAMPLE)
	./$(ACTION_SCORE_EXAMPLE)

haiku-corpus-synthetic:
	mkdir -p $(dir $(HAIKU_CORPUS_SUMMARY))
	$(PYTHON) scripts/generate_haiku_corpus.py --word-list $(HAIKU_LM_LEXICON_WORD_LIST) --output $(HAIKU_CORPUS_SYNTHETIC) --count $(HAIKU_CORPUS_COUNT) --seed $(HAIKU_CORPUS_SEED) --summary $(HAIKU_CORPUS_SUMMARY) --vocab-output $(HAIKU_CORPUS_VOCAB)

haiku-corpus-hf:
	$(PYTHON) scripts/clean_haiku_hf_corpus.py --input $(HAIKU_HF_RAW) --output $(HAIKU_HF_CLEAN) --vocab-output $(HAIKU_HF_VOCAB) --manifest $(HAIKU_HF_MANIFEST) --word-list $(HAIKU_LM_LEXICON_WORD_LIST) --filter-word-list $(HAIKU_HF_FILTER_WORD_LIST) --max-poems $(HAIKU_HF_MAX_POEMS)

haiku-corpus-hf-dictionary:
	$(PYTHON) scripts/clean_haiku_hf_corpus.py --input $(HAIKU_HF_RAW) --output $(HAIKU_HF_DICTIONARY_CLEAN) --vocab-output $(HAIKU_HF_DICTIONARY_VOCAB) --manifest $(HAIKU_HF_DICTIONARY_MANIFEST) --word-list $(HAIKU_LM_LEXICON_WORD_LIST) --filter-word-list 1 --max-poems $(HAIKU_HF_MAX_POEMS)

haiku-lm: $(HAIKU_LM)
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_LM_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.jsonl --save $(HAIKU_LM_DIR)/haiku.nnckpt $(HAIKU_LM_WORD_LIST_ARG)

haiku-lm-lexicon: $(HAIKU_LM)
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_LM_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.lexicon.jsonl --save $(HAIKU_LM_DIR)/haiku.lexicon.nnckpt --word-list "$(HAIKU_LM_LEXICON_WORD_LIST)" --lexicon-gate 1

haiku-lm-smoke: $(HAIKU_LM)
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_LM_CORPUS) --epochs 2 --max-updates 1600 --samples 1 --sample 0 --trace $(HAIKU_LM_DIR)/smoke_trace.jsonl --save $(HAIKU_LM_DIR)/smoke.nnckpt $(HAIKU_LM_WORD_LIST_ARG)

haiku-lm-synthetic: $(HAIKU_LM) haiku-corpus-synthetic
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_CORPUS_SYNTHETIC) --epochs $(HAIKU_LM_SYNTHETIC_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.synthetic.jsonl --save $(HAIKU_LM_DIR)/haiku.synthetic.nnckpt --word-list "$(HAIKU_CORPUS_VOCAB)" --lexicon-gate 1

haiku-lm-hf: $(HAIKU_LM) haiku-corpus-hf
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_HF_CLEAN) --epochs $(HAIKU_LM_HF_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.hf.jsonl --save $(HAIKU_LM_DIR)/haiku.hf.nnckpt --word-list "$(HAIKU_HF_VOCAB)" --lexicon-gate 1

haiku-word-lm: $(HAIKU_WORD_LM)
	mkdir -p $(HAIKU_WORD_LM_DIR)
	./$(HAIKU_WORD_LM) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_WORD_LM_EPOCHS) --lr $(HAIKU_WORD_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --max-candidates $(HAIKU_WORD_LM_MAX_CANDIDATES) --strict-adjacency $(HAIKU_WORD_LM_STRICT_ADJACENCY) --trace $(HAIKU_WORD_LM_DIR)/trace.jsonl --save $(HAIKU_WORD_LM_DIR)/haiku_word.nnckpt $(HAIKU_WORD_LM_WORD_LIST_ARG)

haiku-word-lm-smoke: $(HAIKU_WORD_LM)
	mkdir -p $(HAIKU_WORD_LM_DIR)
	./$(HAIKU_WORD_LM) --corpus $(HAIKU_LM_CORPUS) --epochs 2 --max-updates 1600 --samples 1 --sample 0 --max-candidates 64 --strict-adjacency $(HAIKU_WORD_LM_STRICT_ADJACENCY) --trace $(HAIKU_WORD_LM_DIR)/smoke_trace.jsonl --save $(HAIKU_WORD_LM_DIR)/smoke.nnckpt $(HAIKU_WORD_LM_WORD_LIST_ARG)

haiku-word-lm-synthetic: $(HAIKU_WORD_LM) haiku-corpus-synthetic
	mkdir -p $(HAIKU_WORD_LM_DIR)
	./$(HAIKU_WORD_LM) --corpus $(HAIKU_CORPUS_SYNTHETIC) --word-list "$(HAIKU_CORPUS_VOCAB)" --epochs $(HAIKU_WORD_LM_SYNTHETIC_EPOCHS) --lr $(HAIKU_WORD_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --max-candidates $(HAIKU_WORD_LM_MAX_CANDIDATES) --strict-adjacency $(HAIKU_WORD_LM_STRICT_ADJACENCY) --trace $(HAIKU_WORD_LM_DIR)/trace.synthetic.jsonl --save $(HAIKU_WORD_LM_DIR)/haiku_word.synthetic.nnckpt

haiku-word-lm-hf: $(HAIKU_WORD_LM) haiku-corpus-hf
	mkdir -p $(HAIKU_WORD_LM_DIR)
	./$(HAIKU_WORD_LM) --corpus $(HAIKU_HF_CLEAN) --word-list "$(HAIKU_HF_VOCAB)" --epochs $(HAIKU_WORD_LM_HF_EPOCHS) --lr $(HAIKU_WORD_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --max-candidates $(HAIKU_WORD_LM_MAX_CANDIDATES) --strict-adjacency $(HAIKU_WORD_LM_STRICT_ADJACENCY) --trace $(HAIKU_WORD_LM_DIR)/trace.hf.jsonl --save $(HAIKU_WORD_LM_DIR)/haiku_word.hf.nnckpt

haiku-word-lm-hf-dictionary: $(HAIKU_WORD_LM) haiku-corpus-hf-dictionary
	mkdir -p $(HAIKU_WORD_LM_DIR)
	./$(HAIKU_WORD_LM) --corpus $(HAIKU_HF_DICTIONARY_CLEAN) --word-list "$(HAIKU_HF_DICTIONARY_VOCAB)" --epochs $(HAIKU_WORD_LM_HF_EPOCHS) --lr $(HAIKU_WORD_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --max-candidates $(HAIKU_WORD_LM_MAX_CANDIDATES) --strict-adjacency $(HAIKU_WORD_LM_STRICT_ADJACENCY) --trace $(HAIKU_WORD_LM_DIR)/trace.hf.dictionary.jsonl --save $(HAIKU_WORD_LM_DIR)/haiku_word.hf.dictionary.nnckpt

haiku-understanding: $(HAIKU_UNDERSTANDING)
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_UNDERSTANDING_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace-limit $(HAIKU_UNDERSTANDING_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.nnckpt

haiku-understanding-smoke: $(HAIKU_UNDERSTANDING)
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_LM_CORPUS) --epochs 4 --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace-limit 200 --trace $(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/smoke.nnckpt

haiku-understanding-synthetic: $(HAIKU_UNDERSTANDING) haiku-corpus-synthetic
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_CORPUS_SYNTHETIC) --epochs $(HAIKU_UNDERSTANDING_SYNTHETIC_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace-limit $(HAIKU_UNDERSTANDING_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.synthetic.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.synthetic.nnckpt

haiku-understanding-hf: $(HAIKU_UNDERSTANDING) haiku-corpus-hf
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_HF_CLEAN) --epochs $(HAIKU_UNDERSTANDING_HF_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace-limit $(HAIKU_UNDERSTANDING_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.hf.nnckpt

haiku-understanding-hf-dictionary: $(HAIKU_UNDERSTANDING) haiku-corpus-hf-dictionary
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_HF_DICTIONARY_CLEAN) --epochs $(HAIKU_UNDERSTANDING_HF_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace-limit $(HAIKU_UNDERSTANDING_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.hf.dictionary.nnckpt

experiment-schemas: $(PRIME_GAP_EXAMPLE) $(GRID_PATH) $(DELAYED_CLAIM) $(PROGRAM_SYNTHESIS) $(ARC_AGI) $(SIGNAL_FLIGHT) $(SIGNAL_MINING)
	./$(PRIME_GAP_EXAMPLE) --schema
	./$(GRID_PATH) --schema
	./$(DELAYED_CLAIM) --schema
	./$(PROGRAM_SYNTHESIS) --schema
	./$(ARC_AGI) --schema
	./$(SIGNAL_FLIGHT) --schema
	./$(SIGNAL_MINING) --schema

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
	$(PYTHON) scripts/prime_gap_experiments.py \
		--binary ./$(PRIME_GAP_EXAMPLE) \
		--output-dir $(PRIME_GAP_REPORT_DIR) \
		--supervised-epochs $(PRIME_GAP_EXPERIMENT_EPOCHS) \
		--regularization-grid $(PRIME_GAP_EXPERIMENT_GRID) \
		--seeds $(PRIME_GAP_EXPERIMENT_SEEDS) \
		--max-train-states $(PRIME_GAP_EXPERIMENT_MAX_TRAIN) \
		--max-test-states $(PRIME_GAP_EXPERIMENT_MAX_TEST)

prime-gate-seed-audit: $(PRIME_GAP_EXAMPLE) $(PRIME_RULE_SEARCH) $(PRIME_REPLAY) $(PRIME_COMPARE)
	$(PYTHON) scripts/prime_gate_seed_audit.py \
		--prime-gap ./$(PRIME_GAP_EXAMPLE) \
		--prime-rule-search ./$(PRIME_RULE_SEARCH) \
		--prime-replay ./$(PRIME_REPLAY) \
		--prime-compare ./$(PRIME_COMPARE) \
		--output-dir $(PRIME_GATE_SEED_AUDIT_DIR) \
		--seeds $(PRIME_GATE_SEED_AUDIT_SEEDS)

trace-determinism-audit: prime-gap-trace
	$(PYTHON) scripts/trace_determinism_audit.py \
		--input $(BUILD_DIR)/traces/prime_gap.jsonl \
		--output $(TRACE_DETERMINISM_AUDIT) \
		--canonical-output $(TRACE_DETERMINISM_CANONICAL)

trace-determinism-compare:
	$(MAKE) trace-determinism-audit REAL=float
	$(MAKE) trace-determinism-audit REAL=double
	mkdir -p $(RESULTS_DIR)
	$(PYTHON) scripts/trace_determinism_audit.py \
		build/float/traces/prime_gap.jsonl \
		build/double/traces/prime_gap.jsonl \
		--output $(TRACE_DETERMINISM_COMPARE)
	cp build/float/traces/prime_gap.semantic.jsonl $(TRACE_DETERMINISM_RESULTS_CANONICAL)

trace-determinism-ci: trace-determinism-compare
	$(PYTHON) scripts/trace_determinism_audit.py \
		build/float/traces/prime_gap.jsonl \
		build/double/traces/prime_gap.jsonl \
		--gate \
		--output $(TRACE_DETERMINISM_CI)

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

multitask-domain: $(MULTITASK_DOMAIN)
	./$(MULTITASK_DOMAIN)

multitask-adapter-heads: $(MULTITASK_DOMAIN)
	$(PYTHON) scripts/multitask_adapter_heads.py

tiny-net-plateau-audit: $(MULTITASK_DOMAIN)
	$(PYTHON) scripts/tiny_net_plateau_audit.py --binary ./$(MULTITASK_DOMAIN) --output-dir $(TINY_NET_PLATEAU_AUDIT_DIR) --seeds $(TINY_NET_AUDIT_SEEDS) --capacities $(TINY_NET_AUDIT_CAPACITIES) --epoch-grid $(TINY_NET_AUDIT_EPOCH_GRID) --steps-per-epoch $(TINY_NET_AUDIT_STEPS_PER_EPOCH) --eval-samples $(TINY_NET_AUDIT_EVAL_SAMPLES)

symbolic-domains: $(SYMBOLIC_DOMAINS)
	./$(SYMBOLIC_DOMAINS)

program-synthesis: $(PROGRAM_SYNTHESIS)
	./$(PROGRAM_SYNTHESIS) --proposer all --max-rules 16

program-synthesis-baselines: $(PROGRAM_SYNTHESIS)
	./$(PROGRAM_SYNTHESIS) --proposer all --budget-grid

.PHONY: program-synthesis-sketch-source
program-synthesis-sketch-source: $(PROGRAM_SYNTHESIS)
	mkdir -p $(NEURAL_GRAMMAR_DISCOVERY_DIR)
	./$(PROGRAM_SYNTHESIS) --proposer all --max-rules 16 > $(NEURAL_GRAMMAR_DISCOVERY_PROGRAM_SYNTHESIS_TRACE)

modular-gate-search: $(MODULAR_GATE_SEARCH)
	./$(MODULAR_GATE_SEARCH)

modular-gate-search-smoke: $(MODULAR_GATE_SEARCH)
	mkdir -p $(BUILD_DIR)/traces
	./$(MODULAR_GATE_SEARCH) --max-states 128 --max-actions 24 --max-rules 8 --trace $(BUILD_DIR)/traces/modular_gate_search.jsonl

arc-agi: $(ARC_AGI)
	./$(ARC_AGI) --program-budget-grid --save $(BUILD_DIR)/arc_agi.nnckpt

arc-agi-loader: $(ARC_AGI)
	./$(ARC_AGI) --task-dir "$(ARC_AGI_TASK_DIR)" --max-tasks $(ARC_AGI_MAX_TASKS) --program-budget-grid

arc-agi-induction-smoke: $(ARC_AGI)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13

arc-agi-induction-public: $(ARC_AGI)
	$(PYTHON) scripts/arc_induction_parity.py --arc-agi ./$(ARC_AGI) --task-dir "$(ARC_AGI_TASK_DIR)" --max-tasks $(ARC_AGI_MAX_TASKS)

arc-agi-evidence-surface: $(ARC_AGI)
	$(PYTHON) scripts/arc_agi_evidence_surface.py --arc-agi ./$(ARC_AGI) --task-dir "$(ARC_AGI_TASK_DIR)" --max-tasks $(ARC_AGI_MAX_TASKS) --output-dir "$(ARC_EVIDENCE_SURFACE_DIR)"

arc-family-discovery:
	$(PYTHON) scripts/arc_family_discovery.py --task-dir "$(ARC_AGI_TASK_DIR)" --diagnostics "$(ARC_FAMILY_DIAGNOSTICS)" --output-dir "$(ARC_FAMILY_DISCOVERY_DIR)"

rejected-candidate-learning: $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH) $(DELAYED_CLAIM)
	$(PYTHON) scripts/rejected_candidate_reason_learning.py --binary ./$(PRIME_RULE_SEARCH) --modular-binary ./$(MODULAR_GATE_SEARCH) --delayed-claim-binary ./$(DELAYED_CLAIM) --output-dir $(REJECTED_CANDIDATE_DIR)

rejected-candidate-seed-audit: $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH) $(DELAYED_CLAIM)
	$(PYTHON) scripts/rejected_candidate_seed_audit.py --binary ./$(PRIME_RULE_SEARCH) --modular-binary ./$(MODULAR_GATE_SEARCH) --delayed-claim-binary ./$(DELAYED_CLAIM) --output-dir $(REJECTED_CANDIDATE_SEED_AUDIT_DIR) --seeds $(REJECTED_CANDIDATE_SEEDS)

hypothesis-library-promotion: $(ARC_AGI)
	mkdir -p $(HYPOTHESIS_LIBRARY_DIR)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13 > $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl
	$(PYTHON) scripts/certified_hypothesis_library.py --input $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl --output-dir $(HYPOTHESIS_LIBRARY_DIR) --min-support-tasks 2
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13 --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv > $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_library_prior_raw.jsonl
	$(PYTHON) scripts/certified_hypothesis_library.py --input $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl --output-dir $(HYPOTHESIS_LIBRARY_DIR) --min-support-tasks 2 --feedback-input $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_library_prior_raw.jsonl

hypothesis-library-ablation: $(ARC_AGI)
	mkdir -p $(HYPOTHESIS_LIBRARY_DIR)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --task-ids $(HYPOTHESIS_LIBRARY_PROMOTION_TASK_IDS) --max-tasks 10 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 2 > $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl
	$(PYTHON) scripts/certified_hypothesis_library.py --input $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl --output-dir $(HYPOTHESIS_LIBRARY_DIR) --min-support-tasks 2
	./$(ARC_AGI) $(HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS) --disable-hypothesis-slot-prior --disable-hypothesis-neural-prior --disable-hypothesis-cold-start > $(HYPOTHESIS_LIBRARY_DIR)/ablation_no_prior_raw.jsonl
	./$(ARC_AGI) $(HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS) --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv --disable-hypothesis-neural-prior --disable-hypothesis-cold-start > $(HYPOTHESIS_LIBRARY_DIR)/ablation_slot_only_raw.jsonl
	./$(ARC_AGI) $(HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS) --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv --hypothesis-prior-hidden 0 --disable-hypothesis-cold-start > $(HYPOTHESIS_LIBRARY_DIR)/ablation_linear_prior_raw.jsonl
	./$(ARC_AGI) $(HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS) --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv --hypothesis-prior-hidden 4 --disable-hypothesis-cold-start > $(HYPOTHESIS_LIBRARY_DIR)/ablation_tiny_prior_raw.jsonl
	./$(ARC_AGI) $(HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS) --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv --hypothesis-prior-hidden 24 --disable-hypothesis-cold-start > $(HYPOTHESIS_LIBRARY_DIR)/ablation_full_prior_raw.jsonl
	./$(ARC_AGI) $(HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS) --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv --hypothesis-prior-hidden 24 > $(HYPOTHESIS_LIBRARY_DIR)/ablation_full_prior_cold_raw.jsonl
	$(PYTHON) scripts/certified_hypothesis_library.py --input $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl --output-dir $(HYPOTHESIS_LIBRARY_DIR) --min-support-tasks 2 \
		--ablation-input no_prior=$(HYPOTHESIS_LIBRARY_DIR)/ablation_no_prior_raw.jsonl \
		--ablation-input slot_only=$(HYPOTHESIS_LIBRARY_DIR)/ablation_slot_only_raw.jsonl \
		--ablation-input linear_prior=$(HYPOTHESIS_LIBRARY_DIR)/ablation_linear_prior_raw.jsonl \
		--ablation-input tiny_mlp4=$(HYPOTHESIS_LIBRARY_DIR)/ablation_tiny_prior_raw.jsonl \
		--ablation-input full_mlp24=$(HYPOTHESIS_LIBRARY_DIR)/ablation_full_prior_raw.jsonl \
		--ablation-input full_mlp24_cold=$(HYPOTHESIS_LIBRARY_DIR)/ablation_full_prior_cold_raw.jsonl

hypothesis-library-split-sweep: $(ARC_AGI)
	$(PYTHON) scripts/hypothesis_library_split_sweep.py --arc-agi ./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --output-dir $(HYPOTHESIS_LIBRARY_SPLIT_SWEEP_DIR)

hypothesis-library-feedback-loop: $(ARC_AGI)
	$(PYTHON) scripts/hypothesis_library_feedback_loop.py --arc-agi ./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --generated-siblings-per-family 2 --max-feedback-tasks 3 --stress-no-prior --output-dir $(HYPOTHESIS_LIBRARY_FEEDBACK_LOOP_DIR)

hypothesis-library-public: $(ARC_AGI)
	$(PYTHON) scripts/hypothesis_library_split_sweep.py --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --output-dir $(HYPOTHESIS_LIBRARY_PUBLIC_DIR) --public-train-max-tasks $(HYPOTHESIS_LIBRARY_PUBLIC_TRAIN_MAX_TASKS) --public-eval-max-tasks $(HYPOTHESIS_LIBRARY_PUBLIC_EVAL_MAX_TASKS)

hypothesis-library-public-filter: $(ARC_AGI)
	$(PYTHON) scripts/arc_public_hard_subset.py --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --output-dir $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR) --source-max-tasks $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_SOURCE_MAX_TASKS) --heldout-max-tasks $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_HELDOUT_MAX_TASKS) --source-count $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_SOURCE_COUNT) --heldout-count $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_HELDOUT_COUNT) $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_EXTRA)

hypothesis-library-public-report: hypothesis-library-public-filter
	$(PYTHON) scripts/arc_public_diagnostics_report.py --diagnostics $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)/diagnostics.jsonl --output-dir $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR) --top-k $(HYPOTHESIS_LIBRARY_PUBLIC_REPORT_TOP_K)

hypothesis-library-public-triage: hypothesis-library-public-report
	$(PYTHON) scripts/arc_public_coverage_triage.py --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --diagnostics $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)/diagnostics.jsonl --buckets $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)/arc_public_diagnostics_buckets.jsonl --output-dir $(HYPOTHESIS_LIBRARY_PUBLIC_TRIAGE_DIR) --top-k $(HYPOTHESIS_LIBRARY_PUBLIC_REPORT_TOP_K)

hypothesis-library-neural-discovery: $(ARC_AGI) hypothesis-library-public-triage
	$(PYTHON) scripts/arc_neural_discovery.py --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --diagnostics $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)/diagnostics.jsonl --output-dir $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR) --clusters $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_CLUSTERS) --top-k $(HYPOTHESIS_LIBRARY_PUBLIC_REPORT_TOP_K) --verify-representatives $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_VERIFY) --mutations-per-cluster $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATIONS_PER_CLUSTER) --mutation-representatives $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATION_VERIFY)

hypothesis-library-neural-failure-proof: $(ARC_AGI) hypothesis-library-neural-discovery
	$(PYTHON) scripts/arc_neural_failure_proof.py --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --mutation-verification $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR)/mutation_verification.jsonl --candidate-sketches $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR)/candidate_sketches.jsonl --triage $(HYPOTHESIS_LIBRARY_PUBLIC_TRIAGE_DIR)/coverage_triage.jsonl --output-dir $(HYPOTHESIS_LIBRARY_NEURAL_FAILURE_PROOF_DIR)

.PHONY: neural-grammar-discovery
neural-grammar-discovery: $(NN_GRAMMAR_DISCOVERY_RUNTIME) $(ARC_AGI) hypothesis-library-neural-failure-proof haiku-word-lm-smoke haiku-understanding-smoke program-synthesis-sketch-source grid-path-trace signal-mining-smoke
	$(PYTHON) scripts/neural_grammar_discovery.py --domains $(NEURAL_GRAMMAR_DISCOVERY_DOMAINS) --arc-failure-proof $(NEURAL_GRAMMAR_DISCOVERY_ARC_PROOF) --arc-heldout-triage $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_TRIAGE) --arc-heldout-limit $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_LIMIT) --arc-heldout-per-family $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_PER_FAMILY) --arc-heldout-compare-limit $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_COMPARE_LIMIT) --arc-heldout-op-hints-per-task $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_OP_HINTS_PER_TASK) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --arc-agi ./$(ARC_AGI) --arc-probe-compare-limit $(NEURAL_GRAMMAR_DISCOVERY_ARC_PROBE_COMPARE_LIMIT) --haiku-word-trace $(NEURAL_GRAMMAR_DISCOVERY_HAIKU_WORD_TRACE) --haiku-understanding-trace $(NEURAL_GRAMMAR_DISCOVERY_HAIKU_UNDERSTANDING_TRACE) --program-synthesis-trace $(NEURAL_GRAMMAR_DISCOVERY_PROGRAM_SYNTHESIS_TRACE) --grid-path-trace $(NEURAL_GRAMMAR_DISCOVERY_GRID_TRACE) --signal-mining-trace $(NEURAL_GRAMMAR_DISCOVERY_SIGNAL_MINING_TRACE) --output-dir $(NEURAL_GRAMMAR_DISCOVERY_DIR) --runtime ./$(NN_GRAMMAR_DISCOVERY_RUNTIME)

paper-grade-evidence-core: rejected-candidate-learning rejected-candidate-seed-audit hypothesis-library-split-sweep tiny-net-plateau-audit multitask-adapter-heads experiment-matrix
	mkdir -p $(SUBSTRATE_EXTENSION_RESULTS_DIR) $(MULTITASK_ROADMAP_RESULTS_DIR)
	cp $(REJECTED_CANDIDATE_DIR)/reason_learning.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_learning.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/safety_report.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_safety_report.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/split_sweep.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_split_sweep.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/reason_family_sweep.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_family_sweep.jsonl
	cp $(REJECTED_CANDIDATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_seed_interval_summary.jsonl
	cp $(REJECTED_CANDIDATE_SEED_AUDIT_DIR)/reason_family_seed_interval.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_family_seed_interval.jsonl
	cp $(HYPOTHESIS_LIBRARY_SPLIT_SWEEP_DIR)/heldout_result.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/hypothesis_library_heldout_result.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_plateau_summary.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_plateau_summary.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_width_help.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_width_help.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_audit_verdict.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_audit_verdict.jsonl

paper-grade-evidence: rejected-candidate-learning rejected-candidate-seed-audit hypothesis-library-split-sweep tiny-net-plateau-audit multitask-adapter-heads signal-flight-rollout-eval signal-shadow-replay-audit signal-live-active-summary experiment-matrix
	mkdir -p $(SUBSTRATE_EXTENSION_RESULTS_DIR) $(MULTITASK_ROADMAP_RESULTS_DIR) $(SIGNAL_FLIGHT_RESULTS_DIR)
	cp $(REJECTED_CANDIDATE_DIR)/reason_learning.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_learning.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/safety_report.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_safety_report.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/split_sweep.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_split_sweep.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/reason_family_sweep.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_family_sweep.jsonl
	cp $(REJECTED_CANDIDATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_seed_interval_summary.jsonl
	cp $(REJECTED_CANDIDATE_SEED_AUDIT_DIR)/reason_family_seed_interval.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_family_seed_interval.jsonl
	cp $(HYPOTHESIS_LIBRARY_SPLIT_SWEEP_DIR)/heldout_result.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/hypothesis_library_heldout_result.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_plateau_summary.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_plateau_summary.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_width_help.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_width_help.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_audit_verdict.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_audit_verdict.jsonl
	cp $(SIGNAL_FLIGHT_RESULT_DIR)/summary.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.jsonl
	cp $(SIGNAL_FLIGHT_ROLLOUT_SUMMARY) $(SIGNAL_FLIGHT_RESULTS_DIR)/rollout_summary.jsonl
	cp $(SIGNAL_SHADOW_REPLAY_DIR)/summary.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_replay_audit.jsonl
	cp $(SIGNAL_SHADOW_REPLAY_DIR)/shadow_frames.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_frames.jsonl
	cp $(SIGNAL_LIVE_ACTIVE_DIR)/summary.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/live_active_summary.jsonl
	cp $(SIGNAL_LIVE_ACTIVE_DIR)/active_frames.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/live_active_frames.jsonl
	$(PYTHON) scripts/signal_flight_report.py --offline $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.jsonl --rollout $(SIGNAL_FLIGHT_RESULTS_DIR)/rollout_summary.jsonl --shadow $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_replay_audit.jsonl --live-active $(SIGNAL_FLIGHT_RESULTS_DIR)/live_active_summary.jsonl --output $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.md

signal-flight-data:
	$(MAKE) -C $(SIGNAL_FLIGHT_SIGNAL_DIR) build-flight-trace
	mkdir -p $(BUILD_DIR)
	$(SIGNAL_FLIGHT_TRACE_BIN) --episodes $(SIGNAL_FLIGHT_EPISODES) --ticks $(SIGNAL_FLIGHT_TICKS) --seed $(SIGNAL_FLIGHT_SEED) --format csv --out $(SIGNAL_FLIGHT_TRAIN_TRACE) $(SIGNAL_FLIGHT_TRACE_EXTRA)
	$(SIGNAL_FLIGHT_TRACE_BIN) --episodes $(SIGNAL_FLIGHT_TEST_EPISODES) --ticks $(SIGNAL_FLIGHT_TICKS) --seed $(SIGNAL_FLIGHT_TEST_SEED) --format csv --out $(SIGNAL_FLIGHT_TEST_TRACE) $(SIGNAL_FLIGHT_TRACE_EXTRA)

signal-flight: $(SIGNAL_FLIGHT) signal-flight-data
	mkdir -p $(SIGNAL_FLIGHT_RESULT_DIR)
	./$(SIGNAL_FLIGHT) --train $(SIGNAL_FLIGHT_TRAIN_TRACE) --test $(SIGNAL_FLIGHT_TEST_TRACE) --epochs $(SIGNAL_FLIGHT_EPOCHS) --max-train $(SIGNAL_FLIGHT_MAX_TRAIN) --max-test $(SIGNAL_FLIGHT_MAX_TEST) --learning-rate $(SIGNAL_FLIGHT_LEARNING_RATE) --seed $(SIGNAL_FLIGHT_SEED) --save $(SIGNAL_FLIGHT_RESULT_DIR)/signal_flight.nnckpt --summary $(SIGNAL_FLIGHT_RESULT_DIR)/summary.jsonl $(SIGNAL_FLIGHT_TRAIN_EXTRA)

signal-flight-pain: SIGNAL_FLIGHT_PAIN_PROBE=1
signal-flight-pain: SIGNAL_FLIGHT_WALL_PROBE=1
signal-flight-pain: signal-flight

signal-flight-rollout-eval: $(SIGNAL_FLIGHT_ROLLOUT) signal-flight
	mkdir -p $(SIGNAL_FLIGHT_ROLLOUT_DIR)
	./$(SIGNAL_FLIGHT_ROLLOUT) --checkpoint "$(SIGNAL_FLIGHT_ROLLOUT_CHECKPOINT)" --episodes $(SIGNAL_FLIGHT_ROLLOUT_EPISODES) --ticks $(SIGNAL_FLIGHT_ROLLOUT_TICKS) --seed $(SIGNAL_FLIGHT_ROLLOUT_SEED) --summary "$(SIGNAL_FLIGHT_ROLLOUT_SUMMARY)"

signal-mining: $(SIGNAL_MINING)
	mkdir -p $(SIGNAL_MINING_RESULT_DIR)
	./$(SIGNAL_MINING) --train-groups $(SIGNAL_MINING_TRAIN_GROUPS) --test-groups $(SIGNAL_MINING_TEST_GROUPS) --horizon-ticks $(SIGNAL_MINING_HORIZON_TICKS) --epochs $(SIGNAL_MINING_EPOCHS) --learning-rate $(SIGNAL_MINING_LEARNING_RATE) --seed $(SIGNAL_MINING_SEED) --trace "$(SIGNAL_MINING_RESULT_DIR)/trace.jsonl" --summary "$(SIGNAL_MINING_RESULT_DIR)/summary.jsonl" --save "$(SIGNAL_MINING_RESULT_DIR)/signal_mining.nnckpt"

signal-mining-smoke: SIGNAL_MINING_RESULT_DIR=$(BUILD_DIR)/signal_mining_smoke
signal-mining-smoke: SIGNAL_MINING_TRAIN_GROUPS=6
signal-mining-smoke: SIGNAL_MINING_TEST_GROUPS=3
signal-mining-smoke: SIGNAL_MINING_HORIZON_TICKS=420
signal-mining-smoke: SIGNAL_MINING_EPOCHS=2
signal-mining-smoke: signal-mining

signal-live-brain:
	node scripts/signal_live_brain.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --session "$(SIGNAL_LIVE_BRAIN_SESSION)" --log "$(SIGNAL_LIVE_BRAIN_LOG)" --callsign "$(SIGNAL_LIVE_BRAIN_CALLSIGN)" --max-ms "$(SIGNAL_LIVE_BRAIN_MAX_MS)" --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" --option-mode "$(SIGNAL_LIVE_BRAIN_OPTION_MODE)" $(SIGNAL_LIVE_BRAIN_PROTOCOL_ARG) $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-brain-smoke:
	node scripts/signal_live_brain.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --session "$(SIGNAL_LIVE_BRAIN_SESSION)" --log "$(SIGNAL_LIVE_BRAIN_LOG)" --callsign "$(SIGNAL_LIVE_BRAIN_CALLSIGN)" --max-ms 5000 --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" --option-mode "$(SIGNAL_LIVE_BRAIN_OPTION_MODE)" $(SIGNAL_LIVE_BRAIN_PROTOCOL_ARG) $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-swarm:
	node scripts/signal_live_swarm.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --count "$(SIGNAL_LIVE_SWARM_COUNT)" --min-active "$(SIGNAL_LIVE_SWARM_MIN_ACTIVE)" --max-ms "$(SIGNAL_LIVE_SWARM_MAX_MS)" --out-dir "$(SIGNAL_LIVE_SWARM_OUT_DIR)" --callsign-prefix "$(SIGNAL_LIVE_SWARM_CALLSIGN_PREFIX)" --stagger-ms "$(SIGNAL_LIVE_SWARM_STAGGER_MS)" --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" --option-mode "$(SIGNAL_LIVE_BRAIN_OPTION_MODE)" $(SIGNAL_LIVE_BRAIN_PROTOCOL_ARG) $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-status:
	node scripts/signal_live_status.mjs --log "$(SIGNAL_LIVE_STATUS_LOG)" --interval-ms "$(SIGNAL_LIVE_STATUS_INTERVAL_MS)" $(SIGNAL_LIVE_STATUS_EXTRA)

signal-live-active-summary:
	$(PYTHON) scripts/signal_live_active_summary.py --log "$(SIGNAL_LIVE_ACTIVE_LOG)" --output-dir "$(SIGNAL_LIVE_ACTIVE_DIR)" --window-rows "$(SIGNAL_LIVE_ACTIVE_WINDOW_ROWS)" --stale-ms "$(SIGNAL_LIVE_ACTIVE_STALE_MS)"

signal-flight-report:
	$(PYTHON) scripts/signal_flight_report.py --offline $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.jsonl --rollout $(SIGNAL_FLIGHT_RESULTS_DIR)/rollout_summary.jsonl --shadow $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_replay_audit.jsonl --live-active $(SIGNAL_FLIGHT_RESULTS_DIR)/live_active_summary.jsonl --output $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.md

signal-shadow-replay-audit:
	$(PYTHON) scripts/signal_shadow_replay_audit.py --log "$(SIGNAL_LIVE_STATUS_LOG)" --output-dir "$(SIGNAL_SHADOW_REPLAY_DIR)" --horizon-ms "$(SIGNAL_SHADOW_REPLAY_HORIZON_MS)"

experiment-matrix: paper-support
	$(PYTHON) scripts/experiment_matrix.py

learned-grammar-coverage: $(ARC_AGI) $(HAIKU_WORD_LM) $(HAIKU_UNDERSTANDING)
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_DIR)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13 > $(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)
	$(MAKE) haiku-word-lm-smoke
	$(MAKE) haiku-understanding-smoke
	$(PYTHON) scripts/learned_grammar_coverage.py \
		--arc-log "fixture_regression=$(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)" \
		--haiku-word-trace "smoke_trace=$(HAIKU_WORD_LM_DIR)/smoke_trace.jsonl" \
		--haiku-understanding-trace "smoke_trace=$(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl" \
		$(LEARNED_GRAMMAR_COVERAGE_BUDGET_ARG) \
		--output "$(LEARNED_GRAMMAR_COVERAGE_REPORT)" \
		--fail-on-regression

learned-grammar-coverage-public: $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_DIR)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)

learned-grammar-coverage-arc-visual-compare: $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR); run neural-grammar-discovery first"; exit 2; fi
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_DIR)
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13 > $(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)
	$(PYTHON) scripts/arc_visual_curriculum.py \
		--visual-only-log "fixture_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)" \
		--visual-only-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--visual-only-log "public_eval_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)" \
		--target-scope public_train_visual_only \
		--candidate-tsv "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES)" \
		--examples "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_EXAMPLES)" \
		--summary "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SUMMARY)"
	./$(ARC_AGI) --arc-sketch-probe-candidates "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES)" --public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" --arc-sketch-probe-sources "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SOURCES)" --arc-sketch-probe-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PRIOR)" --arc-sketch-probe-attempts "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" --arc-sketches-per-task 64
	$(PYTHON) scripts/arc_visual_curriculum.py \
		--visual-only-log "fixture_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)" \
		--visual-only-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--visual-only-log "public_eval_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)" \
		--target-scope public_train_visual_only \
		--attempt-log "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" \
		--candidate-tsv "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES)" \
		--examples "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_EXAMPLES)" \
		--summary "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SUMMARY)"
	awk 'NR==1 || FNR>1' "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PRIOR)" > "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)"
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)
	$(PYTHON) scripts/learned_grammar_coverage.py \
		--arc-log "fixture_baseline=$(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)" \
		--arc-log "fixture_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)" \
		--arc-log "public_train_baseline=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_train_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_eval_baseline=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)" \
		--arc-log "public_eval_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)" \
		--arc-log "public_eval_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)" \
		--output "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_COMPARE_REPORT)"

arc-visual-active-curriculum: $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR); run neural-grammar-discovery first"; exit 2; fi
	mkdir -p $(ARC_VISUAL_ACTIVE_CURRICULUM_DIR)
	$(PYTHON) scripts/arc_visual_active_curriculum.py \
		--arc-agi ./$(ARC_AGI) \
		--public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" \
		--seed-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" \
		--output-dir "$(ARC_VISUAL_ACTIVE_CURRICULUM_DIR)" \
		--summary "$(ARC_VISUAL_ACTIVE_CURRICULUM_SUMMARY)" \
		--max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) \
		--iterations $(ARC_VISUAL_ACTIVE_CURRICULUM_ITERATIONS)

arc-visual-capacity-sweep: $(NN_GRAMMAR_DISCOVERY_RUNTIME) $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS)" ]; then echo "missing ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS=$(ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS); run neural-grammar-discovery first"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR); run learned-grammar-coverage-arc-visual-compare first"; exit 2; fi
	mkdir -p $(ARC_VISUAL_CAPACITY_SWEEP_DIR)
	$(PYTHON) scripts/arc_visual_capacity_sweep.py \
		--runtime ./$(NN_GRAMMAR_DISCOVERY_RUNTIME) \
		--attempts-tsv "$(ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS)" \
		--arc-agi ./$(ARC_AGI) \
		--public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" \
		--prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" \
		--output-dir "$(ARC_VISUAL_CAPACITY_SWEEP_DIR)" \
		--epochs "$(ARC_VISUAL_CAPACITY_SWEEP_EPOCHS)" \
		--learning-rates "$(ARC_VISUAL_CAPACITY_SWEEP_LEARNING_RATES)" \
		--max-per-task "$(ARC_VISUAL_CAPACITY_SWEEP_MAX_PER_TASK)" \
		--hidden-sizes "$(ARC_VISUAL_CAPACITY_SWEEP_HIDDEN_SIZES)" \
		--max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) \
		--output "$(ARC_VISUAL_CAPACITY_SWEEP_REPORT)" \
		--markdown "$(ARC_VISUAL_CAPACITY_SWEEP_MARKDOWN)"

arc-visual-hole-proposer: $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS); run learned-grammar-coverage-arc-visual-compare first"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR); run learned-grammar-coverage-arc-visual-compare first"; exit 2; fi
	mkdir -p $(ARC_VISUAL_HOLE_PROPOSER_DIR)
	$(PYTHON) scripts/arc_visual_hole_proposer.py \
		--visual-only-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--visual-only-log "public_train_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)" \
		--visual-only-log "public_eval_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)" \
		--target-scope public_train_curriculum_visual_only \
		--attempt-log "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" \
		--candidate-tsv "$(ARC_VISUAL_HOLE_PROPOSER_CANDIDATES)" \
		--examples "$(ARC_VISUAL_HOLE_PROPOSER_EXAMPLES)" \
		--summary "$(ARC_VISUAL_HOLE_PROPOSER_SUMMARY)"
	./$(ARC_AGI) --arc-sketch-probe-candidates "$(ARC_VISUAL_HOLE_PROPOSER_CANDIDATES)" --public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" --arc-sketch-probe-sources "$(ARC_VISUAL_HOLE_PROPOSER_SOURCES)" --arc-sketch-probe-prior "$(ARC_VISUAL_HOLE_PROPOSER_PRIOR)" --arc-sketch-probe-attempts "$(ARC_VISUAL_HOLE_PROPOSER_ATTEMPTS)" --arc-sketches-per-task 64
	awk 'NR==1 || FNR>1' "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" "$(ARC_VISUAL_HOLE_PROPOSER_PRIOR)" > "$(ARC_VISUAL_HOLE_PROPOSER_COMBINED_PRIOR)"
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(ARC_VISUAL_HOLE_PROPOSER_COMBINED_PRIOR)" > $(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only --hypothesis-library-prior "$(ARC_VISUAL_HOLE_PROPOSER_COMBINED_PRIOR)" > $(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_EVAL_LOG)
	$(PYTHON) scripts/learned_grammar_coverage.py \
		--arc-log "public_train_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_train_hole_proposer_visual_only=$(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_eval_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)" \
		--arc-log "public_eval_hole_proposer_visual_only=$(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_EVAL_LOG)" \
		--output "$(ARC_VISUAL_HOLE_PROPOSER_COMPARE_REPORT)"

$(HAIKU_WORD_LM_DIR)/trace.hf.jsonl:
	$(MAKE) haiku-word-lm-hf

$(HAIKU_WORD_LM_DIR)/trace.hf.dictionary.jsonl:
	$(MAKE) haiku-word-lm-hf-dictionary

$(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl:
	$(MAKE) haiku-understanding-hf

$(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl:
	$(MAKE) haiku-understanding-hf-dictionary

learned-grammar-coverage-extended: learned-grammar-coverage learned-grammar-coverage-public $(HAIKU_WORD_LM_DIR)/trace.hf.jsonl $(HAIKU_WORD_LM_DIR)/trace.hf.dictionary.jsonl $(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl $(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl
	$(PYTHON) scripts/learned_grammar_coverage.py \
		--arc-log "fixture_regression=$(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)" \
		--arc-log "public_train=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_eval=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)" \
		--haiku-word-trace "smoke_trace=$(HAIKU_WORD_LM_DIR)/smoke_trace.jsonl" \
		--haiku-word-trace "hf=$(HAIKU_WORD_LM_DIR)/trace.hf.jsonl" \
		--haiku-word-trace "hf_dictionary=$(HAIKU_WORD_LM_DIR)/trace.hf.dictionary.jsonl" \
		--haiku-understanding-trace "smoke_trace=$(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl" \
		--haiku-understanding-trace "hf=$(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl" \
		--haiku-understanding-trace "hf_dictionary=$(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl" \
		$(LEARNED_GRAMMAR_COVERAGE_BUDGET_ARG) \
		--output "$(LEARNED_GRAMMAR_COVERAGE_EXTENDED_REPORT)" \
		--fail-on-regression

protocol-spine:
	mkdir -p $(PROTOCOL_SPINE_DIR)
	$(PYTHON) scripts/protocol_spine.py \
		--signal-mining-trace "$(NEURAL_GRAMMAR_DISCOVERY_SIGNAL_MINING_TRACE)" \
		--grammar-sketch-examples "$(PROTOCOL_SPINE_GRAMMAR_SKETCH_EXAMPLES)" \
		--arc-hypothesis-evidence "$(PROTOCOL_SPINE_ARC_HYPOTHESIS_EVIDENCE)" \
		--output "$(PROTOCOL_SPINE_REPORT)" \
		--summary "$(PROTOCOL_SPINE_SUMMARY)"

protocol-operator-promotion: protocol-spine
	mkdir -p $(PROTOCOL_OPERATOR_DIR)
	$(PYTHON) scripts/protocol_operator_promotion.py \
		--input "$(PROTOCOL_SPINE_REPORT)" \
		--output "$(PROTOCOL_OPERATOR_REPORT)" \
		--prior "$(PROTOCOL_OPERATOR_PRIOR)" \
		--summary "$(PROTOCOL_OPERATOR_SUMMARY)" \
		--markdown "$(PROTOCOL_OPERATOR_MARKDOWN)" \
		--min-domains $(PROTOCOL_OPERATOR_MIN_DOMAINS) \
		--min-support-rows $(PROTOCOL_OPERATOR_MIN_SUPPORT_ROWS)

experiment-health: learned-grammar-coverage numeric-certificate-report paper-support
	mkdir -p $(EXPERIMENT_HEALTH_DIR)
	$(PYTHON) scripts/experiment_health.py \
		--coverage "$(LEARNED_GRAMMAR_COVERAGE_REPORT)" \
		--numeric-summary "$(NUMERIC_CERTIFICATE_DIR)/summary.jsonl" \
		--numeric-report "$(NUMERIC_CERTIFICATE_REPORT)" \
		--paper-claims paper/generated/claims.json \
		--signal-flight-summary "$(EXPERIMENT_HEALTH_SIGNAL_FLIGHT_SUMMARY)" \
		--signal-flight-rollout "$(EXPERIMENT_HEALTH_SIGNAL_FLIGHT_ROLLOUT)" \
		--signal-live-active "$(EXPERIMENT_HEALTH_SIGNAL_LIVE_ACTIVE)" \
		--signal-shadow-replay "$(EXPERIMENT_HEALTH_SIGNAL_SHADOW_REPLAY)" \
		--neural-grammar-summary "$(EXPERIMENT_HEALTH_NEURAL_GRAMMAR_SUMMARY)" \
		--protocol-spine "$(PROTOCOL_SPINE_REPORT)" \
		--operator-promotions "$(PROTOCOL_OPERATOR_REPORT)" \
		--output "$(EXPERIMENT_HEALTH_REPORT)"

experiment-health-extended: learned-grammar-coverage-extended numeric-certificate-report paper-support
	mkdir -p $(EXPERIMENT_HEALTH_DIR)
	$(PYTHON) scripts/experiment_health.py \
		--coverage "$(LEARNED_GRAMMAR_COVERAGE_EXTENDED_REPORT)" \
		--numeric-summary "$(NUMERIC_CERTIFICATE_DIR)/summary.jsonl" \
		--numeric-report "$(NUMERIC_CERTIFICATE_REPORT)" \
		--paper-claims paper/generated/claims.json \
		--signal-flight-summary "$(EXPERIMENT_HEALTH_SIGNAL_FLIGHT_SUMMARY)" \
		--signal-flight-rollout "$(EXPERIMENT_HEALTH_SIGNAL_FLIGHT_ROLLOUT)" \
		--signal-live-active "$(EXPERIMENT_HEALTH_SIGNAL_LIVE_ACTIVE)" \
		--signal-shadow-replay "$(EXPERIMENT_HEALTH_SIGNAL_SHADOW_REPLAY)" \
		--neural-grammar-summary "$(EXPERIMENT_HEALTH_NEURAL_GRAMMAR_SUMMARY)" \
		--protocol-spine "$(PROTOCOL_SPINE_REPORT)" \
		--operator-promotions "$(PROTOCOL_OPERATOR_REPORT)" \
		--output "$(EXPERIMENT_HEALTH_EXTENDED_REPORT)"

numeric-certificate-experiments:
	$(PYTHON) scripts/numeric_certificate_experiments.py \
		--repo . \
		--make "$(MAKE)" \
		--real "$(REAL)" \
		--output-dir "$(NUMERIC_CERTIFICATE_DIR)" \
		$(foreach target,$(NUMERIC_CERTIFICATE_TARGETS),--target $(target))

numeric-certificate-report: numeric-certificate-experiments
	$(PYTHON) scripts/numeric_certificate_report.py \
		--repo . \
		--summary "$(NUMERIC_CERTIFICATE_DIR)/summary.jsonl" \
		--label "$(REAL)" \
		--output "$(NUMERIC_CERTIFICATE_REPORT)" \
		$(NUMERIC_CERTIFICATE_BUDGET_ARG) \
		--fail-on-reject \
		--max-shrink-steps 0

numeric-certificate-strict:
	CRLP_NUMERIC_REQUIRE_FINITE=1 \
	CRLP_NUMERIC_MAX_ABS_PARAMETER="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER)" \
	CRLP_NUMERIC_MAX_ABS_PARAMETER_DELTA="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER_DELTA)" \
	CRLP_NUMERIC_MAX_ABS_LOGIT="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT)" \
	CRLP_NUMERIC_MAX_ABS_LOGIT_DELTA="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT_DELTA)" \
	CRLP_NUMERIC_MAX_SHRINK_STEPS="$(NUMERIC_CERTIFICATE_STRICT_MAX_SHRINK_STEPS)" \
	$(PYTHON) scripts/numeric_certificate_experiments.py \
		--repo . \
		--make "$(MAKE)" \
		--real "$(REAL)" \
		--output-dir "$(NUMERIC_CERTIFICATE_STRICT_DIR)" \
		$(foreach target,$(NUMERIC_CERTIFICATE_TARGETS),--target $(target))
	$(PYTHON) scripts/numeric_certificate_report.py \
		--repo . \
		--summary "$(NUMERIC_CERTIFICATE_STRICT_DIR)/summary.jsonl" \
		--label "$(REAL)-strict" \
		--output "$(NUMERIC_CERTIFICATE_STRICT_REPORT)" \
		$(NUMERIC_CERTIFICATE_BUDGET_ARG) \
		--fail-on-reject \
		--max-shrink-steps "$(NUMERIC_CERTIFICATE_STRICT_ALLOWED_SHRINK_STEPS)" \
		--max-max max_abs_parameter_after="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER)" \
		--max-max max_abs_parameter_delta="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER_DELTA)" \
		--max-max max_abs_logit_after="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT)" \
		--max-max max_abs_logit_delta="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT_DELTA)"

numeric-certificate-compare:
ifeq ($(NUMERIC_CERTIFICATE_COMPARE_REUSE_FLOAT),1)
	@if [ -s build/float/numeric_certificates/summary.jsonl ]; then \
		echo "numeric-certificate-compare: reusing build/float/numeric_certificates/summary.jsonl"; \
	else \
		$(MAKE) numeric-certificate-experiments REAL=float NUMERIC_CERTIFICATE_DIR=build/float/numeric_certificates; \
	fi
else
	$(MAKE) numeric-certificate-experiments REAL=float NUMERIC_CERTIFICATE_DIR=build/float/numeric_certificates
endif
	$(MAKE) numeric-certificate-experiments REAL=double NUMERIC_CERTIFICATE_DIR=build/double/numeric_certificates
	$(PYTHON) scripts/numeric_certificate_report.py \
		--repo . \
		--summary build/float/numeric_certificates/summary.jsonl \
		--label float \
		--compare-summary build/double/numeric_certificates/summary.jsonl \
		--compare-label double \
		--output "$(NUMERIC_CERTIFICATE_COMPARE_DIR)/float_vs_double.jsonl" \
		$(NUMERIC_CERTIFICATE_BUDGET_ARG) \
		--fail-on-reject

numeric-certificate-budget-refresh:
	$(MAKE) numeric-certificate-report NUMERIC_CERTIFICATE_BUDGET=
	$(MAKE) numeric-certificate-compare NUMERIC_CERTIFICATE_BUDGET=
	$(PYTHON) scripts/numeric_certificate_budget.py \
		--report "$(NUMERIC_CERTIFICATE_REPORT)" \
		--compare "$(NUMERIC_CERTIFICATE_COMPARE_DIR)/float_vs_double.jsonl" \
		--output "$(NUMERIC_CERTIFICATE_BUDGET)"

numeric-certificate-mandatory:
	$(MAKE) numeric-certificate-report
	$(MAKE) numeric-certificate-strict
	$(MAKE) numeric-certificate-compare NUMERIC_CERTIFICATE_COMPARE_REUSE_FLOAT=1

results: $(DELAYED_CLAIM) $(GRID_PATH) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH)
	$(PYTHON) scripts/environment_manifest.py --output $(RESULTS_DIR)/environment.json >/dev/null
	$(PYTHON) scripts/capture_results.py --output-dir $(RESULTS_DIR) --real $(REAL)
	$(PYTHON) scripts/canonicalize_results.py --input $(RESULTS_DIR)/comparison.jsonl --output $(RESULTS_DIR)/comparison.canonical.jsonl
	$(MAKE) trace-determinism-compare RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) prime-gate-seed-audit
	cp $(PRIME_GATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl $(RESULTS_DIR)/prime_gate_delta_seed_interval.jsonl

experimental-results: $(DELAYED_CLAIM) $(GRID_PATH) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH) $(ARC_AGI) $(PROGRAM_SYNTHESIS) $(MULTITASK_DOMAIN)
	$(PYTHON) scripts/environment_manifest.py --output $(RESULTS_DIR)/environment.json >/dev/null
	$(PYTHON) scripts/capture_results.py --output-dir $(RESULTS_DIR) --real $(REAL) --include-experimental --arc-agi-task-dir "$(ARC_AGI_TASK_DIR)" --arc-agi-max-tasks $(ARC_AGI_MAX_TASKS)
	$(PYTHON) scripts/canonicalize_results.py --input $(RESULTS_DIR)/comparison.jsonl --output $(RESULTS_DIR)/comparison.canonical.jsonl
	$(MAKE) trace-determinism-compare RESULTS_DIR=$(RESULTS_DIR)
	$(MAKE) prime-gate-seed-audit
	cp $(PRIME_GATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl $(RESULTS_DIR)/prime_gate_delta_seed_interval.jsonl

paper-pdf: experiment-matrix paper/paper.typ
	cd paper && typst compile paper.typ paper.pdf

paper-data:
	$(PYTHON) paper/scripts/extract_learning_curves.py > paper/data.json

paper-claims:
	$(PYTHON) scripts/derive_paper_claims.py

paper-agent-propose: paper-claims
	PAPER_AGENT_MODEL="$(PAPER_AGENT_MODEL)" PAPER_AGENT_LLM_CMD="$(PAPER_AGENT_LLM_CMD)" $(PYTHON) scripts/paper_agent_propose.py --prompt-output "$(PAPER_AGENT_PROMPT)" --output "$(PAPER_AGENT_OUTPUT)"
	@if [ -n "$(PAPER_AGENT_LLM_CMD)" ]; then $(MAKE) paper-claims; fi

paper-pre-commit:
	$(PYTHON) scripts/pre_commit_paper_claims.py

install-hooks:
	git config core.hooksPath .githooks

paper-support: trace-determinism-compare
	$(PYTHON) scripts/derive_paper_claims.py --support-check

editorial-pr-check:
	$(PYTHON) scripts/check_editorial_pr.py

paper-ci: paper-support editorial-pr-check paper-pdf

neurosymbolic-tune: all
	$(PYTHON) scripts/neurosymbolic_tuning.py --output-dir $(TUNING_DIR) --real $(REAL) --skip-build

python-check:
	$(PYTHON) -m compileall -q scripts paper/scripts

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

sanitize: $(SANITIZE_TEST_RUNNER)
	./$(SANITIZE_TEST_RUNNER)

check: python-check trace-determinism-compare paper-support test sanitize action-score symbolic-domains modular-gate-search-smoke delayed-claim-self-test delayed-claim-eval grid-path-eval prime-rule-search-smoke prime-gap-eval
	$(MAKE) test REAL=double
	$(MAKE) numeric-certificate-mandatory

experimental-check: program-synthesis program-synthesis-baselines arc-agi arc-agi-induction-smoke multitask-domain haiku-word-lm-smoke haiku-understanding-smoke learned-grammar-coverage

clean:
	rm -rf $(BUILD_DIR)
