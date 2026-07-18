# Evidence rollups, numeric certificate gates, and result capture.

.PHONY: protocol-spine protocol-operator-promotion experiment-health experiment-health-extended
.PHONY: numeric-certificate-experiments numeric-certificate-report numeric-certificate-strict
.PHONY: numeric-certificate-compare numeric-certificate-budget-refresh numeric-certificate-mandatory
.PHONY: numeric-certificate-promoted promoted-replication results results-check report experimental-results
.PHONY: paper-grade-evidence-core paper-grade-evidence

protocol-spine:
	$(RETIRED_SCRIPT_TARGET)

protocol-operator-promotion: protocol-spine
	$(RETIRED_SCRIPT_TARGET)

experiment-health: learned-grammar-coverage numeric-certificate-report paper-support dataset-action-score-engine-evidence $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	mkdir -p $(EXPERIMENT_HEALTH_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --experiment-health \
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
		--engine-evidence "$(DATASET_ACTION_SCORE_EVIDENCE)" \
		--tiny-net-diagnostics "$(EXPERIMENT_HEALTH_TINY_NET_DIAGNOSTICS)" \
		--failure-atlas "$(EXPERIMENT_HEALTH_FAILURE_ATLAS)" \
		--failure-atlas-markdown "$(EXPERIMENT_HEALTH_FAILURE_ATLAS_MARKDOWN)" \
		--failure-work-queue "$(EXPERIMENT_HEALTH_WORK_QUEUE)" \
		--failure-work-queue-markdown "$(EXPERIMENT_HEALTH_WORK_QUEUE_MARKDOWN)" \
		--output "$(EXPERIMENT_HEALTH_REPORT)"

experiment-health-extended: learned-grammar-coverage-extended numeric-certificate-report paper-support dataset-action-score-engine-evidence $(NN_GRAMMAR_DISCOVERY_RUNTIME)
	mkdir -p $(EXPERIMENT_HEALTH_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --experiment-health \
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
		--engine-evidence "$(DATASET_ACTION_SCORE_EVIDENCE)" \
		--tiny-net-diagnostics "$(EXPERIMENT_HEALTH_TINY_NET_DIAGNOSTICS)" \
		--failure-atlas "$(EXPERIMENT_HEALTH_EXTENDED_FAILURE_ATLAS)" \
		--failure-atlas-markdown "$(EXPERIMENT_HEALTH_EXTENDED_FAILURE_ATLAS_MARKDOWN)" \
		--failure-work-queue "$(EXPERIMENT_HEALTH_EXTENDED_WORK_QUEUE)" \
		--failure-work-queue-markdown "$(EXPERIMENT_HEALTH_EXTENDED_WORK_QUEUE_MARKDOWN)" \
		--output "$(EXPERIMENT_HEALTH_EXTENDED_REPORT)"

numeric-certificate-experiments:
	mkdir -p "$(NUMERIC_CERTIFICATE_DIR)"
	@: > "$(NUMERIC_CERTIFICATE_DIR)/summary.jsonl"
	@set -e; for target in $(NUMERIC_CERTIFICATE_TARGETS); do \
		safe=$$(printf '%s' "$$target" | tr -c 'A-Za-z0-9_-' '_'); \
		cert="$(NUMERIC_CERTIFICATE_DIR)/$$safe.numeric_steps.jsonl"; \
		stdout="$(NUMERIC_CERTIFICATE_DIR)/$$safe.stdout.log"; \
		stderr="$(NUMERIC_CERTIFICATE_DIR)/$$safe.stderr.log"; \
		: > "$$cert"; \
		echo "numeric-certificates: $$target"; \
		if CRLP_NUMERIC_CERT_PATH="$$cert" CRLP_NUMERIC_CERT_TARGET="$$target" $(MAKE) "$$target" REAL="$(REAL)" > "$$stdout" 2> "$$stderr"; then status=passed; code=0; else status=failed; code=$$?; fi; \
		rows=$$(grep -c '"schema":"crlp.numeric_step_certificate.v1"' "$$cert" 2>/dev/null || true); \
		rejected=$$(grep -c '"step_committed":false' "$$cert" 2>/dev/null || true); \
		committed=$$((rows - rejected)); \
		finite=$$(grep -c '"finite_after":true' "$$cert" 2>/dev/null || true); \
		printf '{"schema":"crlp.numeric_certificate_experiment_summary.v1","target":"%s","status":"%s","exit_code":%s,"certificate_path":"%s","stdout_path":"%s","stderr_path":"%s","certificate_rows":%s,"committed_rows":%s,"rejected_rows":%s,"finite_after_rows":%s}\n' "$$target" "$$status" "$$code" "$$cert" "$$stdout" "$$stderr" "$$rows" "$$committed" "$$rejected" "$$finite" >> "$(NUMERIC_CERTIFICATE_DIR)/summary.jsonl"; \
		test "$$code" -eq 0; \
	done

numeric-certificate-report: numeric-certificate-experiments
	cp "$(NUMERIC_CERTIFICATE_DIR)/summary.jsonl" "$(NUMERIC_CERTIFICATE_REPORT)"
	@if grep -Eq '"status":"failed"|"rejected_rows":[1-9]' "$(NUMERIC_CERTIFICATE_REPORT)"; then echo "numeric certificate report failed"; exit 1; fi

numeric-certificate-strict:
	CRLP_NUMERIC_REQUIRE_FINITE=1 \
	CRLP_NUMERIC_MAX_ABS_PARAMETER="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER)" \
	CRLP_NUMERIC_MAX_ABS_PARAMETER_DELTA="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_PARAMETER_DELTA)" \
	CRLP_NUMERIC_MAX_ABS_LOGIT="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT)" \
	CRLP_NUMERIC_MAX_ABS_LOGIT_DELTA="$(NUMERIC_CERTIFICATE_STRICT_MAX_ABS_LOGIT_DELTA)" \
	CRLP_NUMERIC_MAX_SHRINK_STEPS="$(NUMERIC_CERTIFICATE_STRICT_MAX_SHRINK_STEPS)" \
	$(MAKE) numeric-certificate-experiments NUMERIC_CERTIFICATE_DIR="$(NUMERIC_CERTIFICATE_STRICT_DIR)"
	cp "$(NUMERIC_CERTIFICATE_STRICT_DIR)/summary.jsonl" "$(NUMERIC_CERTIFICATE_STRICT_REPORT)"
	@if grep -Eq '"status":"failed"|"rejected_rows":[1-9]' "$(NUMERIC_CERTIFICATE_STRICT_REPORT)"; then echo "strict numeric certificate report failed"; exit 1; fi

numeric-certificate-compare:
ifeq ($(NUMERIC_CERTIFICATE_COMPARE_REUSE_FLOAT),1)
	@if [ -s "$(NUMERIC_CERTIFICATE_COMPARE_FLOAT_DIR)/summary.jsonl" ]; then \
		echo "numeric-certificate-compare: reusing $(NUMERIC_CERTIFICATE_COMPARE_FLOAT_DIR)/summary.jsonl"; \
	else \
		$(MAKE) numeric-certificate-experiments REAL=float NUMERIC_CERTIFICATE_DIR="$(NUMERIC_CERTIFICATE_COMPARE_FLOAT_DIR)"; \
	fi
else
	$(MAKE) numeric-certificate-experiments REAL=float NUMERIC_CERTIFICATE_DIR="$(NUMERIC_CERTIFICATE_COMPARE_FLOAT_DIR)"
endif
	$(MAKE) numeric-certificate-experiments REAL=double NUMERIC_CERTIFICATE_DIR="$(NUMERIC_CERTIFICATE_COMPARE_DOUBLE_DIR)"
	mkdir -p "$(NUMERIC_CERTIFICATE_COMPARE_DIR)"
	cat "$(NUMERIC_CERTIFICATE_COMPARE_FLOAT_DIR)/summary.jsonl" "$(NUMERIC_CERTIFICATE_COMPARE_DOUBLE_DIR)/summary.jsonl" > "$(NUMERIC_CERTIFICATE_COMPARE_DIR)/float_vs_double.jsonl"
	@if grep -Eq '"status":"failed"|"rejected_rows":[1-9]' "$(NUMERIC_CERTIFICATE_COMPARE_DIR)/float_vs_double.jsonl"; then echo "numeric certificate compare failed"; exit 1; fi

numeric-certificate-budget-refresh:
	$(RETIRED_SCRIPT_TARGET)

numeric-certificate-mandatory:
	$(MAKE) numeric-certificate-report
	$(MAKE) numeric-certificate-strict
	$(MAKE) numeric-certificate-compare NUMERIC_CERTIFICATE_COMPARE_REUSE_FLOAT=1

numeric-certificate-promoted:
	$(MAKE) numeric-certificate-report \
		NUMERIC_CERTIFICATE_TARGETS="$(NUMERIC_CERTIFICATE_PROMOTED_TARGETS)" \
		NUMERIC_CERTIFICATE_DIR="$(NUMERIC_CERTIFICATE_PROMOTED_DIR)" \
		NUMERIC_CERTIFICATE_REPORT="$(NUMERIC_CERTIFICATE_PROMOTED_REPORT)"
	$(MAKE) numeric-certificate-strict \
		NUMERIC_CERTIFICATE_TARGETS="$(NUMERIC_CERTIFICATE_PROMOTED_TARGETS)" \
		NUMERIC_CERTIFICATE_STRICT_DIR="$(NUMERIC_CERTIFICATE_PROMOTED_STRICT_DIR)" \
		NUMERIC_CERTIFICATE_STRICT_REPORT="$(NUMERIC_CERTIFICATE_PROMOTED_STRICT_REPORT)"
	$(MAKE) numeric-certificate-compare NUMERIC_CERTIFICATE_COMPARE_REUSE_FLOAT=1 \
		NUMERIC_CERTIFICATE_TARGETS="$(NUMERIC_CERTIFICATE_PROMOTED_TARGETS)" \
		NUMERIC_CERTIFICATE_COMPARE_FLOAT_DIR="$(NUMERIC_CERTIFICATE_PROMOTED_DIR)" \
		NUMERIC_CERTIFICATE_COMPARE_DOUBLE_DIR="build/double/numeric_certificates_promoted"

paper-grade-evidence-core: tiny-net-plateau-audit multitask-adapter-heads experiment-matrix
	mkdir -p "$(MULTITASK_ROADMAP_RESULTS_DIR)"

paper-grade-evidence: tiny-net-plateau-audit multitask-adapter-heads signal-mining-smoke experiment-matrix
	mkdir -p "$(MULTITASK_ROADMAP_RESULTS_DIR)"

promoted-replication: $(GRID_PATH) $(DELAYED_CLAIM) grid-path-no-priors delayed-claim-replay
	REAL="$(REAL)" ./scripts/run_promoted_replication.sh "$(BUILD_DIR)/promoted_replication"

results: $(PROMOTED_BINARY_TARGETS) promoted-replication
	mkdir -p $(RESULTS_DIR)/logs
	@: > $(RESULTS_DIR)/comparison.jsonl
	@: > $(RESULTS_DIR)/commands.txt
	./scripts/capture_make_targets.sh "$(RESULTS_DIR)" "$(REAL)" experiment-schemas-promoted delayed-claim-eval grid-path-eval grid-path-budget-sweep modular-gate-search-smoke prime-rule-search prime-rule-search-baselines prime-gap-eval
	cp $(RESULTS_DIR)/comparison.jsonl $(RESULTS_DIR)/comparison.canonical.jsonl
	cp "$(BUILD_DIR)/promoted_replication/raw.jsonl" "$(RESULTS_DIR)/replication.raw.jsonl"
	cp "$(BUILD_DIR)/promoted_replication/summary.jsonl" "$(RESULTS_DIR)/replication.summary.jsonl"
	cp "$(BUILD_DIR)/promoted_replication/summary.md" "$(RESULTS_DIR)/replication.summary.md"
	cp "$(BUILD_DIR)/promoted_replication/cost_verdict.jsonl" "$(RESULTS_DIR)/cost_verdict.jsonl"
	cp "$(BUILD_DIR)/promoted_replication/robustness.jsonl" "$(RESULTS_DIR)/robustness.jsonl"
	cp "$(BUILD_DIR)/promoted_replication/robustness_verdict.jsonl" "$(RESULTS_DIR)/robustness_verdict.jsonl"
	./scripts/write_results_metadata.sh "$(RESULTS_DIR)" "$(REAL)" "$(CC)"
	./scripts/generate_results_summary.sh "$(RESULTS_DIR)"
	./scripts/check_results_bundle.sh "$(RESULTS_DIR)"
	$(MAKE) trace-determinism-compare RESULTS_DIR=$(RESULTS_DIR)
	@if [ -f "$(PRIME_GATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl" ]; then cp "$(PRIME_GATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl" "$(RESULTS_DIR)/prime_gate_delta_seed_interval.jsonl"; elif [ -f "$(RESULTS_DIR)/prime_gate_delta_seed_interval.jsonl" ]; then echo "reusing existing prime gate seed interval"; else echo "missing prime gate seed interval artifact"; fi

results-check:
	./scripts/check_results_bundle.sh "$(RESULTS_DIR)"

report: results
	sh scripts/build_pages_site.sh

experimental-results: $(PROMOTED_BINARY_TARGETS) $(ARC_AGI) $(PROGRAM_SYNTHESIS) $(MULTITASK_DOMAIN)
	$(MAKE) results
	./scripts/capture_make_targets.sh "$(RESULTS_DIR)" "$(REAL)" arc-agi arc-agi-induction-smoke program-synthesis-baselines multitask-domain
	cp $(RESULTS_DIR)/comparison.jsonl $(RESULTS_DIR)/comparison.canonical.jsonl
	$(MAKE) trace-determinism-compare RESULTS_DIR=$(RESULTS_DIR)
	@if [ -f "$(PRIME_GATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl" ]; then cp "$(PRIME_GATE_SEED_AUDIT_DIR)/seed_interval_summary.jsonl" "$(RESULTS_DIR)/prime_gate_delta_seed_interval.jsonl"; elif [ -f "$(RESULTS_DIR)/prime_gate_delta_seed_interval.jsonl" ]; then echo "reusing existing prime gate seed interval"; else echo "missing prime gate seed interval artifact"; fi
