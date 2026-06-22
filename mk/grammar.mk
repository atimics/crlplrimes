# Learned grammar coverage and ARC visual coverage diagnostics.

.PHONY: learned-grammar-coverage learned-grammar-coverage-public
.PHONY: learned-grammar-coverage-arc-visual-compare learned-grammar-coverage-extended
.PHONY: arc-visual-active-curriculum arc-visual-capacity-sweep arc-visual-hole-proposer

learned-grammar-coverage: $(HAIKU_UNDERSTANDING) $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_DIR)
	$(MAKE) haiku-coherence-hard-negatives
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --learned-grammar-coverage \
		--haiku-understanding-trace "smoke_trace=$(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl" \
		--haiku-coherence-report "smoke_trace=$(LEARNED_GRAMMAR_COVERAGE_HAIKU_COHERENCE_REPORT)" \
		$(LEARNED_GRAMMAR_COVERAGE_BUDGET_ARG) \
		--output "$(LEARNED_GRAMMAR_COVERAGE_REPORT)" \
		--fail-on-regression

learned-grammar-coverage-public: $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_DIR)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)

learned-grammar-coverage-arc-visual-compare: $(ARC_AGI) $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR); run neural-grammar-discovery first"; exit 2; fi
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_DIR)
	mkdir -p $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_DIR)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 13 > $(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)
	./$(ARC_AGI) --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 > $(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-visual-curriculum \
		--visual-only-log "fixture_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)" \
		--visual-only-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--visual-only-log "public_eval_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)" \
		--target-scope public_train_visual_only \
		--candidate-tsv "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES)" \
		--examples "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_EXAMPLES)" \
		--summary "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SUMMARY)"
	./$(ARC_AGI) --arc-sketch-probe-candidates "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES)" --public-arc-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" --arc-sketch-probe-sources "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SOURCES)" --arc-sketch-probe-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PRIOR)" --arc-sketch-probe-attempts "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" --arc-sketches-per-task 64
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-visual-curriculum \
		--visual-only-log "fixture_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)" \
		--visual-only-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--visual-only-log "public_eval_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)" \
		--target-scope public_train_visual_only \
		--attempt-log "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" \
		--candidate-tsv "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_CANDIDATES)" \
		--examples "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_EXAMPLES)" \
		--summary "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_SUMMARY)"
	awk 'NR==1 || FNR>1' "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PRIOR)" > "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)"
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" > $(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --learned-grammar-coverage \
		--arc-log "fixture_baseline=$(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)" \
		--arc-log "fixture_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_FIXTURE_LOG)" \
		--arc-log "public_train_baseline=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_train_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_train_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_eval_baseline=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)" \
		--arc-log "public_eval_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PUBLIC_EVAL_LOG)" \
		--arc-log "public_eval_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)" \
		--output "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_COMPARE_REPORT)"

arc-visual-active-curriculum: $(ARC_AGI) $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_VISUAL_PRIOR); run neural-grammar-discovery first"; exit 2; fi
	$(RETIRED_SCRIPT_TARGET)

arc-visual-capacity-sweep: $(NN_GRAMMAR_DISCOVERY_RUNTIME) $(ARC_AGI)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS)" ]; then echo "missing ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS=$(ARC_VISUAL_CAPACITY_SWEEP_ATTEMPTS); run neural-grammar-discovery first"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR); run learned-grammar-coverage-arc-visual-compare first"; exit 2; fi
	$(RETIRED_SCRIPT_TARGET)

arc-visual-hole-proposer: $(ARC_AGI) $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	@if [ -z "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)" ]; then echo "set LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR or HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_ATTEMPTS); run learned-grammar-coverage-arc-visual-compare first"; exit 2; fi
	@if [ ! -f "$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR)" ]; then echo "missing LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_COMBINED_PRIOR); run learned-grammar-coverage-arc-visual-compare first"; exit 2; fi
	mkdir -p $(ARC_VISUAL_HOLE_PROPOSER_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --arc-visual-hole-proposer \
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
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/training" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(ARC_VISUAL_HOLE_PROPOSER_COMBINED_PRIOR)" > $(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_TRAIN_LOG)
	./$(ARC_AGI) --task-dir "$(LEARNED_GRAMMAR_COVERAGE_PUBLIC_ARC_DIR)/evaluation" --max-tasks $(LEARNED_GRAMMAR_COVERAGE_PUBLIC_MAX_TASKS) --program-budget-grid --loader-summary-only --induction --min-induction-evidence 0 --arc-visual-only $(ARC_RUNTIME_SKETCH_MINE_ARG) --hypothesis-library-prior "$(ARC_VISUAL_HOLE_PROPOSER_COMBINED_PRIOR)" > $(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_EVAL_LOG)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --learned-grammar-coverage \
		--arc-log "public_train_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_train_hole_proposer_visual_only=$(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_eval_curriculum_visual_only=$(LEARNED_GRAMMAR_COVERAGE_ARC_CURRICULUM_PUBLIC_EVAL_LOG)" \
		--arc-log "public_eval_hole_proposer_visual_only=$(ARC_VISUAL_HOLE_PROPOSER_PUBLIC_EVAL_LOG)" \
		--output "$(ARC_VISUAL_HOLE_PROPOSER_COMPARE_REPORT)"

$(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl:
	$(MAKE) haiku-understanding-hf

$(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl:
	$(MAKE) haiku-understanding-hf-dictionary

learned-grammar-coverage-extended: learned-grammar-coverage learned-grammar-coverage-public $(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl $(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl haiku-coherence-hard-negatives-hf-dictionary $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --learned-grammar-coverage \
		--arc-log "fixture_regression=$(LEARNED_GRAMMAR_COVERAGE_ARC_LOG)" \
		--arc-log "public_train=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_TRAIN_LOG)" \
		--arc-log "public_eval=$(LEARNED_GRAMMAR_COVERAGE_ARC_PUBLIC_EVAL_LOG)" \
		--haiku-understanding-trace "smoke_trace=$(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl" \
		--haiku-understanding-trace "hf=$(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl" \
		--haiku-understanding-trace "hf_dictionary=$(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl" \
		--haiku-coherence-report "smoke_trace=$(LEARNED_GRAMMAR_COVERAGE_HAIKU_COHERENCE_REPORT)" \
		--haiku-coherence-report "hf_dictionary=$(LEARNED_GRAMMAR_COVERAGE_HAIKU_COHERENCE_HF_DICTIONARY_REPORT)" \
		$(LEARNED_GRAMMAR_COVERAGE_BUDGET_ARG) \
		--output "$(LEARNED_GRAMMAR_COVERAGE_EXTENDED_REPORT)" \
		--fail-on-regression
