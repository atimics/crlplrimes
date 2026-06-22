# ARC-AGI, hypothesis-library, neural discovery, and verified self-play targets.

.PHONY: arc-public-data arc-agi arc-agi-loader arc-gate-invariance arc-agi-induction-smoke
.PHONY: arc-agi-induction-public arc-agi-evidence-surface arc-family-discovery
.PHONY: arc-ranker-benchmark arc-ranker-failure-report arc-synthetic-shape-curriculum
.PHONY: arc-synthetic-shape-probe arc-verified-self-play arc-verified-self-play-public arc-learned-sketch-ops
.PHONY: rejected-candidate-learning rejected-candidate-seed-audit compounding-loop
.PHONY: hypothesis-library-promotion hypothesis-library-ablation hypothesis-library-split-sweep
.PHONY: hypothesis-library-feedback-loop hypothesis-library-public hypothesis-library-public-filter
.PHONY: hypothesis-library-public-report hypothesis-library-public-triage
.PHONY: hypothesis-library-neural-discovery hypothesis-library-neural-failure-proof
.PHONY: neural-grammar-discovery

arc-agi: $(ARC_AGI)
	./$(ARC_AGI) --program-budget-grid --save $(BUILD_DIR)/arc_agi.nnckpt

arc-agi-loader: $(ARC_AGI)
	./$(ARC_AGI) --task-dir "$(ARC_AGI_TASK_DIR)" --max-tasks $(ARC_AGI_MAX_TASKS) --program-budget-grid

arc-gate-invariance: $(ARC_AGI)
	@if [ -z "$(ARC_AGI_TASK_DIR)" ]; then echo "set ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data"; exit 2
	$(RETIRED_SCRIPT_TARGET)

arc-agi-induction-smoke: $(ARC_AGI)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13

arc-agi-induction-public: $(ARC_AGI)
	$(RETIRED_SCRIPT_TARGET)

arc-agi-evidence-surface: $(ARC_AGI)
	$(RETIRED_SCRIPT_TARGET)

arc-family-discovery:
	$(RETIRED_SCRIPT_TARGET)

rejected-candidate-learning: $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH) $(DELAYED_CLAIM) $(SUBSTRATE_EXPERIMENTS)
	$(RETIRED_SCRIPT_TARGET)

rejected-candidate-seed-audit: $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH) $(DELAYED_CLAIM) $(SUBSTRATE_EXPERIMENTS)
	$(RETIRED_SCRIPT_TARGET)

compounding-loop: $(PRIME_RULE_SEARCH) $(MODULAR_GATE_SEARCH) $(DELAYED_CLAIM) $(ARC_AGI) $(GRID_PATH) $(SUBSTRATE_EXPERIMENTS) arc-public-data
	mkdir -p "$(COMPOUNDING_EVIDENCE_DIR)"
	./$(SUBSTRATE_EXPERIMENTS) --mode compounding-loop --build-dir "$(BUILD_DIR)" --output-dir "$(COMPOUNDING_EVIDENCE_DIR)" --iterations $(COMPOUNDING_ITERATIONS) --arc-task-dir "$(ARC_PUBLIC_DATA_DIR)/training" --real "$(REAL)"

hypothesis-library-promotion: $(ARC_AGI) $(SUBSTRATE_EXPERIMENTS)
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-ablation: $(ARC_AGI) $(SUBSTRATE_EXPERIMENTS)
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-split-sweep: $(ARC_AGI) $(SUBSTRATE_EXPERIMENTS)
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-feedback-loop: $(ARC_AGI)
	@mkdir -p "$(HYPOTHESIS_LIBRARY_FEEDBACK_LOOP_DIR)"
	@echo "$@: public-data workflow (requires ARC public data download; use make arc-public-data first)"
	@touch "$(HYPOTHESIS_LIBRARY_FEEDBACK_LOOP_DIR)/placeholder.jsonl"
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-public: $(ARC_AGI)
	@mkdir -p "$(HYPOTHESIS_LIBRARY_PUBLIC_DIR)"
	@echo "$@: public-data workflow (requires ARC public data download; use make arc-public-data first)"
	@touch "$(HYPOTHESIS_LIBRARY_PUBLIC_DIR)/placeholder.jsonl"
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-public-filter: $(ARC_AGI)
	@mkdir -p "$(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)"
	@echo "$@: public-data workflow (requires ARC public data download; use make arc-public-data first)"
	@touch "$(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)/placeholder.jsonl"
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-public-report: hypothesis-library-public-filter
	@echo "$@: skipped (public-data pipeline, not needed for smoke check)"
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-public-triage: hypothesis-library-public-report
	@echo "$@: skipped (public-data pipeline, not needed for smoke check)"
	$(RETIRED_SCRIPT_TARGET)

hypothesis-library-neural-discovery: $(ARC_AGI) $(NN_GRAMMAR_DISCOVERY_RUNTIME) hypothesis-library-public-triage
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-neural-discovery --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --diagnostics $(HYPOTHESIS_LIBRARY_PUBLIC_FILTER_DIR)/diagnostics.jsonl --output-dir $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR) --clusters $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_CLUSTERS) --top-k $(HYPOTHESIS_LIBRARY_PUBLIC_REPORT_TOP_K) --verify-representatives $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_VERIFY) --mutations-per-cluster $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATIONS_PER_CLUSTER) --mutation-representatives $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATION_VERIFY)

hypothesis-library-neural-failure-proof: $(ARC_AGI) $(NN_GRAMMAR_DISCOVERY_RUNTIME) hypothesis-library-neural-discovery
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-neural-failure-proof --arc-agi ./$(ARC_AGI) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --mutation-verification $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR)/mutation_verification.jsonl --candidate-sketches $(HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_DIR)/candidate_sketches.jsonl --triage $(HYPOTHESIS_LIBRARY_PUBLIC_TRIAGE_DIR)/coverage_triage.jsonl --output-dir $(HYPOTHESIS_LIBRARY_NEURAL_FAILURE_PROOF_DIR)

neural-grammar-discovery: $(NN_GRAMMAR_DISCOVERY_RUNTIME) $(ARC_AGI) hypothesis-library-neural-failure-proof haiku-understanding-smoke haiku-coherence-hard-negatives program-synthesis-sketch-source grid-path-trace signal-mining-smoke
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --neural-grammar-discovery --domains $(NEURAL_GRAMMAR_DISCOVERY_DOMAINS) --arc-failure-proof $(NEURAL_GRAMMAR_DISCOVERY_ARC_PROOF) --arc-heldout-triage $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_TRIAGE) --arc-heldout-limit $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_LIMIT) --arc-heldout-per-family $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_PER_FAMILY) --arc-heldout-compare-limit $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_COMPARE_LIMIT) --arc-heldout-op-hints-per-task $(NEURAL_GRAMMAR_DISCOVERY_ARC_HELDOUT_OP_HINTS_PER_TASK) --public-arc-dir "$(HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR)" --arc-agi ./$(ARC_AGI) --arc-probe-compare-limit $(NEURAL_GRAMMAR_DISCOVERY_ARC_PROBE_COMPARE_LIMIT) --haiku-understanding-trace $(NEURAL_GRAMMAR_DISCOVERY_HAIKU_UNDERSTANDING_TRACE) --haiku-coherence-hard-negatives $(NEURAL_GRAMMAR_DISCOVERY_HAIKU_COHERENCE_REPORT) --program-synthesis-trace $(NEURAL_GRAMMAR_DISCOVERY_PROGRAM_SYNTHESIS_TRACE) --grid-path-trace $(NEURAL_GRAMMAR_DISCOVERY_GRID_TRACE) --signal-mining-trace $(NEURAL_GRAMMAR_DISCOVERY_SIGNAL_MINING_TRACE) --output-dir $(NEURAL_GRAMMAR_DISCOVERY_DIR) --runtime ./$(NN_GRAMMAR_DISCOVERY_RUNTIME)

arc-ranker-benchmark: $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	@if [ ! -f "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" ]; then echo "missing ARC_RANKER_BENCHMARK_ATTEMPTS=$(ARC_RANKER_BENCHMARK_ATTEMPTS); run make neural-grammar-discovery first or set ARC_RANKER_BENCHMARK_ATTEMPTS=/path/to/arc_visual_pipeline_neural_attempts.tsv"; exit 2; fi
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-ranker-benchmark --attempts-tsv "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" --output-dir "$(ARC_RANKER_BENCHMARK_DIR)" --report "$(ARC_RANKER_BENCHMARK_REPORT)" --epochs "$(ARC_RANKER_BENCHMARK_EPOCHS)" --max-per-task "$(ARC_RANKER_BENCHMARK_MAX_PER_TASK)"

arc-ranker-failure-report: arc-ranker-benchmark
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-ranker-failure-report --attempts-tsv "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" --proposals "$(ARC_RANKER_FAILURE_PROPOSALS)" --summary "$(ARC_RANKER_FAILURE_SUMMARY)" --output "$(ARC_RANKER_FAILURE_REPORT)" --markdown "$(ARC_RANKER_FAILURE_MARKDOWN)"

arc-synthetic-shape-curriculum: $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	@if [ ! -f "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" ]; then echo "missing ARC_RANKER_BENCHMARK_ATTEMPTS=$(ARC_RANKER_BENCHMARK_ATTEMPTS); run make neural-grammar-discovery first or set ARC_RANKER_BENCHMARK_ATTEMPTS=/path/to/arc_visual_pipeline_neural_attempts.tsv"; exit 2; fi
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-synthetic-shape-curriculum --attempts-tsv "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" --output-dir "$(ARC_SYNTHETIC_SHAPE_CURRICULUM_DIR)" --synthetic-attempts "$(ARC_SYNTHETIC_SHAPE_ATTEMPTS)" --combined-attempts "$(ARC_SYNTHETIC_SHAPE_COMBINED_ATTEMPTS)" --summary "$(ARC_SYNTHETIC_SHAPE_SUMMARY)" --max-grid-size "$(ARC_SYNTHETIC_SHAPE_MAX_GRID_SIZE)" --ops "$(ARC_SYNTHETIC_SHAPE_OPS)" --negative-kinds "$(ARC_SYNTHETIC_SHAPE_NEGATIVE_KINDS)" --op-mode "$(ARC_SYNTHETIC_SHAPE_OP_MODE)" --max-accepted "$(ARC_SYNTHETIC_SHAPE_MAX_ACCEPTED)" --max-synthetic-rows "$(ARC_SYNTHETIC_SHAPE_MAX_SYNTHETIC_ROWS)" --max-seconds "$(ARC_SYNTHETIC_SHAPE_MAX_SECONDS)"
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-pipeline-attempts "$(ARC_SYNTHETIC_SHAPE_COMBINED_ATTEMPTS)" --arc-pipeline-proposals-output "$(ARC_SYNTHETIC_SHAPE_PROPOSALS)" --arc-pipeline-summary-output "$(ARC_SYNTHETIC_SHAPE_RANKER_SUMMARY)" --arc-pipeline-cache "$(ARC_SYNTHETIC_SHAPE_CACHE)" --arc-pipeline-max-per-task "$(ARC_SYNTHETIC_SHAPE_RANKER_MAX_PER_TASK)" --epochs "$(ARC_SYNTHETIC_SHAPE_RANKER_EPOCHS)" --learning-rate "$(ARC_SYNTHETIC_SHAPE_RANKER_LEARNING_RATE)"
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-ranker-failure-report --attempts-tsv "$(ARC_SYNTHETIC_SHAPE_COMBINED_ATTEMPTS)" --proposals "$(ARC_SYNTHETIC_SHAPE_PROPOSALS)" --summary "$(ARC_SYNTHETIC_SHAPE_RANKER_SUMMARY)" --output "$(ARC_SYNTHETIC_SHAPE_FAILURE_REPORT)" --markdown "$(ARC_SYNTHETIC_SHAPE_FAILURE_MARKDOWN)"

arc-synthetic-shape-probe: $(ARC_AGI) $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" ]; then echo "missing ARC_RANKER_BENCHMARK_ATTEMPTS=$(ARC_RANKER_BENCHMARK_ATTEMPTS); run make arc-ranker-benchmark first or set ARC_RANKER_BENCHMARK_ATTEMPTS=/path/to/arc_visual_pipeline_neural_attempts.tsv"; exit 2; fi
	@if [ ! -f "$(ARC_RANKER_FAILURE_REPORT)" ]; then echo "missing ARC_RANKER_FAILURE_REPORT=$(ARC_RANKER_FAILURE_REPORT); run make arc-ranker-failure-report first"; exit 2; fi
	mkdir -p $(ARC_SYNTHETIC_SHAPE_PROBE_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-synthetic-shape-probe-candidates --failure-report "$(ARC_RANKER_FAILURE_REPORT)" --public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" --candidate-tsv "$(ARC_SYNTHETIC_SHAPE_PROBE_CANDIDATES)" --examples "$(ARC_SYNTHETIC_SHAPE_PROBE_EXAMPLES)" --summary "$(ARC_SYNTHETIC_SHAPE_PROBE_CANDIDATE_SUMMARY)" --max-rows-per-task "$(ARC_SYNTHETIC_SHAPE_PROBE_MAX_ROWS_PER_TASK)" --max-tasks "$(ARC_SYNTHETIC_SHAPE_PROBE_MAX_TASKS)" --profile "$(ARC_SYNTHETIC_SHAPE_PROBE_PROFILE)"
	./$(ARC_AGI) --arc-sketch-probe-candidates "$(ARC_SYNTHETIC_SHAPE_PROBE_CANDIDATES)" --public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" --arc-sketch-probe-sources "$(ARC_SYNTHETIC_SHAPE_PROBE_SOURCES)" --arc-sketch-probe-prior "$(ARC_SYNTHETIC_SHAPE_PROBE_PRIOR)" --arc-sketch-probe-attempts "$(ARC_SYNTHETIC_SHAPE_PROBE_ATTEMPTS)" --arc-sketches-per-task "$(ARC_SYNTHETIC_SHAPE_PROBE_SKETCHES_PER_TASK)"
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-append-probe-attempts-tsv --base-attempts-tsv "$(ARC_RANKER_BENCHMARK_ATTEMPTS)" --probe-attempts-jsonl "$(ARC_SYNTHETIC_SHAPE_PROBE_ATTEMPTS)" --combined-attempts-tsv "$(ARC_SYNTHETIC_SHAPE_PROBE_COMBINED_ATTEMPTS)" --summary "$(ARC_SYNTHETIC_SHAPE_PROBE_APPEND_SUMMARY)" --split-name target
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-pipeline-attempts "$(ARC_SYNTHETIC_SHAPE_PROBE_COMBINED_ATTEMPTS)" --arc-pipeline-proposals-output "$(ARC_SYNTHETIC_SHAPE_PROBE_PROPOSALS)" --arc-pipeline-summary-output "$(ARC_SYNTHETIC_SHAPE_PROBE_RANKER_SUMMARY)" --arc-pipeline-cache "$(ARC_SYNTHETIC_SHAPE_PROBE_CACHE)" --arc-pipeline-max-per-task "$(ARC_SYNTHETIC_SHAPE_PROBE_RANKER_MAX_PER_TASK)" --epochs "$(ARC_SYNTHETIC_SHAPE_PROBE_RANKER_EPOCHS)" --learning-rate "$(ARC_SYNTHETIC_SHAPE_PROBE_RANKER_LEARNING_RATE)"
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-ranker-failure-report --attempts-tsv "$(ARC_SYNTHETIC_SHAPE_PROBE_COMBINED_ATTEMPTS)" --proposals "$(ARC_SYNTHETIC_SHAPE_PROBE_PROPOSALS)" --summary "$(ARC_SYNTHETIC_SHAPE_PROBE_RANKER_SUMMARY)" --output "$(ARC_SYNTHETIC_SHAPE_PROBE_FAILURE_REPORT)" --markdown "$(ARC_SYNTHETIC_SHAPE_PROBE_FAILURE_MARKDOWN)"

arc-public-data:
	@if [ -d "$(ARC_PUBLIC_DATA_DIR)/training" ] && [ $$(ls $(ARC_PUBLIC_DATA_DIR)/training/*.json 2>/dev/null | wc -l) -ge $(ARC_PUBLIC_DATA_EXPECTED_TRAINING) ]; then \
		echo "using existing ARC public data: $(ARC_PUBLIC_DATA_DIR)"; \
	else \
		echo "downloading ARC public data..."; \
		mkdir -p $(ARC_PUBLIC_DATA_DIR); \
		curl -sL -o /tmp/arc_agi_master.zip $(ARC_PUBLIC_DATA_ZIP_URL); \
		unzip -qo /tmp/arc_agi_master.zip -d /tmp/arc_extract; \
		cp -r /tmp/arc_extract/ARC-AGI-master/data/training $(ARC_PUBLIC_DATA_DIR)/; \
		cp -r /tmp/arc_extract/ARC-AGI-master/data/evaluation $(ARC_PUBLIC_DATA_DIR)/; \
		rm -rf /tmp/arc_extract /tmp/arc_agi_master.zip; \
		echo "ARC public data ready: $$(ls $(ARC_PUBLIC_DATA_DIR)/training/*.json | wc -l) training, $$(ls $(ARC_PUBLIC_DATA_DIR)/evaluation/*.json | wc -l) evaluation"; \
	fi

arc-verified-self-play: arc-synthetic-shape-curriculum
	mkdir -p $(ARC_VERIFIED_SELF_PLAY_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-verified-self-play-report --attempts-tsv "$(ARC_SYNTHETIC_SHAPE_COMBINED_ATTEMPTS)" --synthetic-summary "$(ARC_SYNTHETIC_SHAPE_SUMMARY)" --ranker-summary "$(ARC_SYNTHETIC_SHAPE_RANKER_SUMMARY)" --self-play-prior "$(ARC_VERIFIED_SELF_PLAY_PRIOR)" --self-play-prior-summary "$(ARC_VERIFIED_SELF_PLAY_PRIOR_SUMMARY)" --self-play-generative-prior "$(ARC_VERIFIED_SELF_PLAY_GENERATIVE_PRIOR)" --self-play-generative-prior-summary "$(ARC_VERIFIED_SELF_PLAY_GENERATIVE_PRIOR_SUMMARY)" --summary "$(ARC_VERIFIED_SELF_PLAY_SUMMARY)" --markdown "$(ARC_VERIFIED_SELF_PLAY_MARKDOWN)" --template-baseline "$(ARC_VERIFIED_SELF_PLAY_TEMPLATE_BASELINE)" --oracle-ceiling "$(ARC_VERIFIED_SELF_PLAY_ORACLE_CEILING)" --seed-count "$(ARC_VERIFIED_SELF_PLAY_SEED_COUNT)"

arc-verified-self-play-public: arc-verified-self-play arc-public-data $(ARC_AGI)
	./$(ARC_AGI) --task-dir "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_ARC_DIR)/training" --max-tasks $(ARC_VERIFIED_SELF_PLAY_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(ARC_VERIFIED_SELF_PLAY_PUBLIC_BASELINE_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(ARC_VERIFIED_SELF_PLAY_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(ARC_VERIFIED_SELF_PLAY_PUBLIC_BASELINE_EVAL_LOG)
	./$(ARC_AGI) --task-dir "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_ARC_DIR)/training" --max-tasks $(ARC_VERIFIED_SELF_PLAY_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) > $(ARC_VERIFIED_SELF_PLAY_PUBLIC_VISUAL_BASELINE_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(ARC_VERIFIED_SELF_PLAY_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) > $(ARC_VERIFIED_SELF_PLAY_PUBLIC_VISUAL_BASELINE_EVAL_LOG)
	./$(ARC_AGI) --task-dir "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_ARC_DIR)/training" --max-tasks $(ARC_VERIFIED_SELF_PLAY_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(ARC_VERIFIED_SELF_PLAY_PRIOR)" --arc-generative-prior "$(ARC_VERIFIED_SELF_PLAY_GENERATIVE_PRIOR)" $(ARC_GENERATIVE_PRIOR_OP_ARG) > $(ARC_VERIFIED_SELF_PLAY_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(ARC_VERIFIED_SELF_PLAY_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(ARC_VERIFIED_SELF_PLAY_PRIOR)" --arc-generative-prior "$(ARC_VERIFIED_SELF_PLAY_GENERATIVE_PRIOR)" $(ARC_GENERATIVE_PRIOR_OP_ARG) > $(ARC_VERIFIED_SELF_PLAY_PUBLIC_EVAL_LOG)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-verified-self-play-report --attempts-tsv "$(ARC_SYNTHETIC_SHAPE_COMBINED_ATTEMPTS)" --synthetic-summary "$(ARC_SYNTHETIC_SHAPE_SUMMARY)" --ranker-summary "$(ARC_SYNTHETIC_SHAPE_RANKER_SUMMARY)" --public-train-baseline-log "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_BASELINE_TRAIN_LOG)" --public-eval-baseline-log "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_BASELINE_EVAL_LOG)" --public-train-visual-baseline-log "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_VISUAL_BASELINE_TRAIN_LOG)" --public-eval-visual-baseline-log "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_VISUAL_BASELINE_EVAL_LOG)" --public-train-self-play-log "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_TRAIN_LOG)" --public-eval-self-play-log "$(ARC_VERIFIED_SELF_PLAY_PUBLIC_EVAL_LOG)" --self-play-prior "$(ARC_VERIFIED_SELF_PLAY_PRIOR)" --self-play-prior-summary "$(ARC_VERIFIED_SELF_PLAY_PRIOR_SUMMARY)" --self-play-generative-prior "$(ARC_VERIFIED_SELF_PLAY_GENERATIVE_PRIOR)" --self-play-generative-prior-summary "$(ARC_VERIFIED_SELF_PLAY_GENERATIVE_PRIOR_SUMMARY)" --summary "$(ARC_VERIFIED_SELF_PLAY_SUMMARY)" --markdown "$(ARC_VERIFIED_SELF_PLAY_MARKDOWN)" --template-baseline "$(ARC_VERIFIED_SELF_PLAY_TEMPLATE_BASELINE)" --oracle-ceiling "$(ARC_VERIFIED_SELF_PLAY_ORACLE_CEILING)" --seed-count "$(ARC_VERIFIED_SELF_PLAY_SEED_COUNT)"

arc-learned-sketch-ops: arc-verified-self-play-public
	mkdir -p $(ARC_LEARNED_SKETCH_OPS_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-learned-sketch-ops --attempts-tsv "$(ARC_LEARNED_SKETCH_OPS_ATTEMPTS)" --trace "$(ARC_LEARNED_SKETCH_OPS_TRACE)" --operators "$(ARC_LEARNED_SKETCH_OPS_OPERATORS)" --prior "$(ARC_LEARNED_SKETCH_OPS_PRIOR)" --summary "$(ARC_LEARNED_SKETCH_OPS_SUMMARY)" --markdown "$(ARC_LEARNED_SKETCH_OPS_MARKDOWN)" --min-accepted "$(ARC_LEARNED_SKETCH_OPS_MIN_ACCEPTED)" --min-tasks "$(ARC_LEARNED_SKETCH_OPS_MIN_TASKS)" --near-miss-score "$(ARC_LEARNED_SKETCH_OPS_NEAR_MISS_SCORE)" --diagnostic-score "$(ARC_LEARNED_SKETCH_OPS_DIAGNOSTIC_SCORE)"
