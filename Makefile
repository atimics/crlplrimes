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
NN_HEADERS := include/nn.h include/nn_core.h include/nn_train.h include/nn_experiment.h include/nn_symbolic.h include/nn_program.h include/nn_hypothesis.h
PRIME_GAP_TASK_HEADERS := include/nn_tasks_prime_gap.h src/prime_gap/internal.h
PRIME_GAP_MODULE_SOURCES := $(wildcard src/prime_gap/*.c)
PRIME_GAP_TASK_SOURCES := src/nn_tasks_prime_gap.c $(PRIME_GAP_MODULE_SOURCES)
PRIME_GAP_TASK_OBJS := $(patsubst src/prime_gap/%.c,$(BUILD_DIR)/obj/prime_gap/%.o,$(PRIME_GAP_MODULE_SOURCES))
NN_OBJ := $(BUILD_DIR)/nn.o
NN_EXPERIMENT_OBJ := $(BUILD_DIR)/nn_experiment.o
NN_SYMBOLIC_OBJ := $(BUILD_DIR)/nn_symbolic.o
NN_PROGRAM_OBJ := $(BUILD_DIR)/nn_program.o
NN_HYPOTHESIS_OBJ := $(BUILD_DIR)/nn_hypothesis.o
NN_TEST_OBJ := $(BUILD_DIR)/nn_test.o
NN_EXPERIMENT_TEST_OBJ := $(BUILD_DIR)/nn_experiment_test.o
NN_SYMBOLIC_TEST_OBJ := $(BUILD_DIR)/nn_symbolic_test.o
NN_PROGRAM_TEST_OBJ := $(BUILD_DIR)/nn_program_test.o
NN_HYPOTHESIS_TEST_OBJ := $(BUILD_DIR)/nn_hypothesis_test.o
NN_SANITIZE_OBJ := $(BUILD_DIR)/nn_sanitize.o
NN_EXPERIMENT_SANITIZE_OBJ := $(BUILD_DIR)/nn_experiment_sanitize.o
NN_SYMBOLIC_SANITIZE_OBJ := $(BUILD_DIR)/nn_symbolic_sanitize.o
NN_PROGRAM_SANITIZE_OBJ := $(BUILD_DIR)/nn_program_sanitize.o
NN_HYPOTHESIS_SANITIZE_OBJ := $(BUILD_DIR)/nn_hypothesis_sanitize.o
NN_OBJS := $(NN_OBJ) $(NN_EXPERIMENT_OBJ) $(NN_SYMBOLIC_OBJ) $(NN_PROGRAM_OBJ) $(NN_HYPOTHESIS_OBJ)
NN_TEST_OBJS := $(NN_TEST_OBJ) $(NN_EXPERIMENT_TEST_OBJ) $(NN_SYMBOLIC_TEST_OBJ) $(NN_PROGRAM_TEST_OBJ) $(NN_HYPOTHESIS_TEST_OBJ)
NN_SANITIZE_OBJS := $(NN_SANITIZE_OBJ) $(NN_EXPERIMENT_SANITIZE_OBJ) $(NN_SYMBOLIC_SANITIZE_OBJ) $(NN_PROGRAM_SANITIZE_OBJ) $(NN_HYPOTHESIS_SANITIZE_OBJ)
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
TRACE_DETERMINISM_RESULTS_CANONICAL ?= $(RESULTS_DIR)/prime_gap_trace.semantic.jsonl
HYPOTHESIS_LIBRARY_PROMOTION_TASK_IDS ?= move_to_anchor,move_to_anchor_x
HYPOTHESIS_LIBRARY_HELDOUT_TASK_IDS ?= connect_aligned,extend_aligned_sequence,fill_enclosed,move_to_anchor_both_positive,move_to_anchor_y,object_extract,recolor_by_reference,translate
HYPOTHESIS_LIBRARY_ABLATION_BASE_FLAGS ?= --task-dir tests/fixtures/arc_hypothesis --task-ids $(HYPOTHESIS_LIBRARY_HELDOUT_TASK_IDS) --max-tasks 10 --program-budget-tight-grid --loader-summary-only --induction --min-induction-evidence 8 --hypothesis-top-k 1 --disable-hypothesis-coverage-selection --disable-hypothesis-structural-score --disable-hypothesis-train-filter
HYPOTHESIS_LIBRARY_SPLIT_SWEEP_DIR ?= $(BUILD_DIR)/hypothesis_library_split_sweep
HYPOTHESIS_LIBRARY_PUBLIC_DIR ?= $(BUILD_DIR)/hypothesis_library_public
HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR ?=
HYPOTHESIS_LIBRARY_PUBLIC_TRAIN_MAX_TASKS ?= 400
HYPOTHESIS_LIBRARY_PUBLIC_EVAL_MAX_TASKS ?= 400
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
SIGNAL_LIVE_BRAIN_EXTRA ?=
SIGNAL_LIVE_STATUS_LOG ?= $(SIGNAL_LIVE_BRAIN_LOG)
SIGNAL_LIVE_STATUS_INTERVAL_MS ?= 1000
SIGNAL_LIVE_STATUS_EXTRA ?=
SIGNAL_SHADOW_REPLAY_DIR ?= $(BUILD_DIR)/signal_shadow_replay_audit
SIGNAL_SHADOW_REPLAY_HORIZON_MS ?= 500
SIGNAL_FLIGHT_ROLLOUT_DIR ?= $(BUILD_DIR)/signal_flight_rollout_eval
SIGNAL_FLIGHT_ROLLOUT_SUMMARY ?= $(SIGNAL_FLIGHT_ROLLOUT_DIR)/summary.jsonl
SIGNAL_FLIGHT_ROLLOUT_CHECKPOINT ?= $(SIGNAL_FLIGHT_RESULT_DIR)/signal_flight.nnckpt
SIGNAL_FLIGHT_ROLLOUT_EPISODES ?= 24
SIGNAL_FLIGHT_ROLLOUT_TICKS ?= 900
SIGNAL_FLIGHT_ROLLOUT_SEED ?= 4242

SIGNAL_FLIGHT_INCLUDE_FLAGS := \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/server \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl
SIGNAL_FLIGHT_SIM_SOURCES := \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/game_sim.c \
	$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_ai.c \
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

.PHONY: all run action-score experiment-schemas prime-gap prime-gap-trace prime-gap-replay prime-gap-eval prime-rule-search prime-rule-search-baselines prime-rule-search-conditional prime-rule-search-smoke prime-gap-sweep prime-gap-regularization-sweep prime-gap-experiments trace-determinism-audit trace-determinism-compare delayed-claim delayed-claim-trace delayed-claim-replay delayed-claim-eval delayed-claim-self-test grid-path grid-path-rule-search grid-path-budget-sweep grid-path-trace grid-path-replay grid-path-eval multitask-domain multitask-adapter-heads tiny-net-plateau-audit symbolic-domains program-synthesis program-synthesis-baselines modular-gate-search modular-gate-search-smoke arc-agi arc-agi-loader arc-agi-induction-smoke arc-agi-induction-public arc-agi-evidence-surface arc-family-discovery rejected-candidate-learning rejected-candidate-seed-audit hypothesis-library-promotion hypothesis-library-ablation hypothesis-library-split-sweep hypothesis-library-public paper-grade-evidence signal-flight-data signal-flight signal-flight-pain signal-flight-rollout-eval signal-live-brain signal-live-brain-smoke signal-live-status signal-shadow-replay-audit experiment-matrix results paper-pdf paper-data paper-claims paper-support neurosymbolic-tune benchmark test sanitize check clean

all: $(XOR_EXAMPLE) $(ACTION_SCORE_EXAMPLE) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH) $(DELAYED_CLAIM) $(GRID_PATH) $(MULTITASK_DOMAIN) $(SYMBOLIC_DOMAINS) $(PROGRAM_SYNTHESIS) $(MODULAR_GATE_SEARCH) $(ARC_AGI) $(SIGNAL_FLIGHT) $(BENCHMARK)

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

$(BUILD_DIR)/obj/prime_gap/%.o: src/prime_gap/%.c $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(XOR_EXAMPLE): examples/xor.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/xor.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(ACTION_SCORE_EXAMPLE): examples/action_score.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/action_score.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

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

$(MODULAR_GATE_SEARCH): examples/modular_gate_search.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/modular_gate_search.c $(LDFLAGS) $(LDLIBS) -o $@

$(ARC_AGI): examples/arc_agi.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/arc_agi.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_FLIGHT): examples/signal_flight.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/signal_flight.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_FLIGHT_ROLLOUT): examples/signal_flight_rollout.c $(NN_OBJS) $(NN_HEADERS) $(SIGNAL_FLIGHT_SIM_SOURCES) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SIGNAL_FLIGHT_INCLUDE_FLAGS) $(CFLAGS) examples/signal_flight_rollout.c $(NN_OBJS) $(SIGNAL_FLIGHT_SIM_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCHMARK): benchmarks/benchmark.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) benchmarks/benchmark.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_TASK_SOURCES) $(NN_TEST_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) tests/test_nn.c $(NN_TEST_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(SANITIZE_TEST_RUNNER): tests/test_nn.c $(PRIME_GAP_TASK_SOURCES) $(NN_SANITIZE_OBJS) $(NN_HEADERS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -DNN_ENABLE_TEST_API $(CFLAGS) $(SANITIZE_CFLAGS) tests/test_nn.c $(NN_SANITIZE_OBJS) $(LDFLAGS) $(SANITIZE_LDFLAGS) $(LDLIBS) -o $@

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

action-score: $(ACTION_SCORE_EXAMPLE)
	./$(ACTION_SCORE_EXAMPLE)

experiment-schemas: $(PRIME_GAP_EXAMPLE) $(GRID_PATH) $(DELAYED_CLAIM) $(PROGRAM_SYNTHESIS) $(ARC_AGI) $(SIGNAL_FLIGHT)
	./$(PRIME_GAP_EXAMPLE) --schema
	./$(GRID_PATH) --schema
	./$(DELAYED_CLAIM) --schema
	./$(PROGRAM_SYNTHESIS) --schema
	./$(ARC_AGI) --schema
	./$(SIGNAL_FLIGHT) --schema

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
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 10 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 10

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
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 10 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 10 > $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl
	$(PYTHON) scripts/certified_hypothesis_library.py --input $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_raw.jsonl --output-dir $(HYPOTHESIS_LIBRARY_DIR) --min-support-tasks 2
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 10 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 10 --hypothesis-library-prior $(HYPOTHESIS_LIBRARY_DIR)/library_prior.tsv > $(HYPOTHESIS_LIBRARY_DIR)/arc_hypothesis_library_prior_raw.jsonl
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

hypothesis-library-public: $(ARC_AGI)
	$(PYTHON) scripts/hypothesis_library_split_sweep.py --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --output-dir $(HYPOTHESIS_LIBRARY_PUBLIC_DIR) --public-train-max-tasks $(HYPOTHESIS_LIBRARY_PUBLIC_TRAIN_MAX_TASKS) --public-eval-max-tasks $(HYPOTHESIS_LIBRARY_PUBLIC_EVAL_MAX_TASKS)

paper-grade-evidence: rejected-candidate-learning rejected-candidate-seed-audit hypothesis-library-split-sweep tiny-net-plateau-audit signal-shadow-replay-audit experiment-matrix
	mkdir -p $(SUBSTRATE_EXTENSION_RESULTS_DIR) $(MULTITASK_ROADMAP_RESULTS_DIR) $(SIGNAL_FLIGHT_RESULTS_DIR)
	cp $(REJECTED_CANDIDATE_DIR)/reason_learning.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_learning.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/split_sweep.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_split_sweep.jsonl
	cp $(REJECTED_CANDIDATE_DIR)/reason_family_sweep.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_family_sweep.jsonl
	cp $(REJECTED_CANDIDATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_seed_interval_summary.jsonl
	cp $(REJECTED_CANDIDATE_SEED_AUDIT_DIR)/reason_family_seed_interval.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/rejected_candidate_reason_family_seed_interval.jsonl
	cp $(HYPOTHESIS_LIBRARY_SPLIT_SWEEP_DIR)/heldout_result.jsonl $(SUBSTRATE_EXTENSION_RESULTS_DIR)/hypothesis_library_heldout_result.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_plateau_summary.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_plateau_summary.jsonl
	cp $(TINY_NET_PLATEAU_AUDIT_DIR)/tiny_net_width_help.jsonl $(MULTITASK_ROADMAP_RESULTS_DIR)/tiny_net_width_help.jsonl
	cp $(SIGNAL_SHADOW_REPLAY_DIR)/summary.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_replay_audit.jsonl
	cp $(SIGNAL_SHADOW_REPLAY_DIR)/shadow_frames.jsonl $(SIGNAL_FLIGHT_RESULTS_DIR)/shadow_frames.jsonl

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

signal-flight-rollout-eval: $(SIGNAL_FLIGHT_ROLLOUT)
	mkdir -p $(SIGNAL_FLIGHT_ROLLOUT_DIR)
	./$(SIGNAL_FLIGHT_ROLLOUT) --checkpoint "$(SIGNAL_FLIGHT_ROLLOUT_CHECKPOINT)" --episodes $(SIGNAL_FLIGHT_ROLLOUT_EPISODES) --ticks $(SIGNAL_FLIGHT_ROLLOUT_TICKS) --seed $(SIGNAL_FLIGHT_ROLLOUT_SEED) --summary "$(SIGNAL_FLIGHT_ROLLOUT_SUMMARY)"

signal-live-brain:
	node scripts/signal_live_brain.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --session "$(SIGNAL_LIVE_BRAIN_SESSION)" --log "$(SIGNAL_LIVE_BRAIN_LOG)" --callsign "$(SIGNAL_LIVE_BRAIN_CALLSIGN)" --max-ms "$(SIGNAL_LIVE_BRAIN_MAX_MS)" --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-brain-smoke:
	node scripts/signal_live_brain.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --session "$(SIGNAL_LIVE_BRAIN_SESSION)" --log "$(SIGNAL_LIVE_BRAIN_LOG)" --callsign "$(SIGNAL_LIVE_BRAIN_CALLSIGN)" --max-ms 5000 --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-status:
	node scripts/signal_live_status.mjs --log "$(SIGNAL_LIVE_STATUS_LOG)" --interval-ms "$(SIGNAL_LIVE_STATUS_INTERVAL_MS)" $(SIGNAL_LIVE_STATUS_EXTRA)

signal-shadow-replay-audit:
	$(PYTHON) scripts/signal_shadow_replay_audit.py --log "$(SIGNAL_LIVE_STATUS_LOG)" --output-dir "$(SIGNAL_SHADOW_REPLAY_DIR)" --horizon-ms "$(SIGNAL_SHADOW_REPLAY_HORIZON_MS)"

experiment-matrix: paper-support
	$(PYTHON) scripts/experiment_matrix.py

results: $(DELAYED_CLAIM) $(GRID_PATH) $(PRIME_GAP_EXAMPLE) $(PRIME_REPLAY) $(PRIME_COMPARE) $(PRIME_RULE_SEARCH) $(ARC_AGI)
	$(PYTHON) scripts/environment_manifest.py --output $(RESULTS_DIR)/environment.json >/dev/null
	$(PYTHON) scripts/capture_results.py --output-dir $(RESULTS_DIR) --real $(REAL) --arc-agi-task-dir "$(ARC_AGI_TASK_DIR)" --arc-agi-max-tasks $(ARC_AGI_MAX_TASKS)
	$(PYTHON) scripts/canonicalize_results.py --input $(RESULTS_DIR)/comparison.jsonl --output $(RESULTS_DIR)/comparison.canonical.jsonl
	$(MAKE) trace-determinism-compare RESULTS_DIR=$(RESULTS_DIR)

paper-pdf: experiment-matrix paper/paper.typ
	cd paper && typst compile paper.typ paper.pdf

paper-data:
	$(PYTHON) paper/scripts/extract_learning_curves.py > paper/data.json

paper-claims:
	$(PYTHON) scripts/derive_paper_claims.py

paper-support: trace-determinism-compare
	$(PYTHON) scripts/derive_paper_claims.py --support-check

neurosymbolic-tune: all
	$(PYTHON) scripts/neurosymbolic_tuning.py --output-dir $(TUNING_DIR) --real $(REAL) --skip-build

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

sanitize: $(SANITIZE_TEST_RUNNER)
	./$(SANITIZE_TEST_RUNNER)

check: trace-determinism-compare paper-support test sanitize action-score symbolic-domains program-synthesis modular-gate-search-smoke arc-agi arc-agi-induction-smoke delayed-claim-self-test delayed-claim-eval grid-path-eval prime-rule-search-smoke prime-gap-eval
	$(MAKE) test REAL=double

clean:
	rm -rf $(BUILD_DIR)
