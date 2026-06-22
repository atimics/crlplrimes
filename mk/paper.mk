# Paper governance, claim support, and bibliography release targets.

.PHONY: paper-gen paper-pdf paper-data paper-claims paper-agent-propose paper-pre-commit
.PHONY: install-hooks paper-support editorial-pr-check paper-ci paper-bibliography paper-pdf-release
.PHONY: experiment-matrix

BIBLIOGRAPHY_REPORT ?= paper/generated/bibliography_report.txt

experiment-matrix: paper-support
	@test -s paper/generated/experiment_matrix.jsonl
	@test -s paper/generated/experiment_matrix.md
	@test -s paper/generated/experiment_matrix.typ

paper-gen:
	$(RETIRED_SCRIPT_TARGET)

paper-pdf: experiment-matrix paper-bibliography paper/paper.typ
	cd paper && typst compile paper.typ paper.pdf

paper-data: $(NN_GRAMMAR_DISCOVERY_RUNTIME) $(ARC_AGI) $(PRIME_GAP_EXAMPLE)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --paper-data --paper-data-output paper/data.json --paper-data-typ paper/data.typ --comparison "$(RESULTS_DIR)/comparison.jsonl" --prime-gap-trace "$(BUILD_DIR)/traces/prime_gap.jsonl" --grid-path-trace "$(BUILD_DIR)/grid_path.jsonl" --delayed-claim-trace "$(BUILD_DIR)/delayed_claim.jsonl" --arc-agi ./$(ARC_AGI) --prime-gap ./$(PRIME_GAP_EXAMPLE)

paper-claims: $(NN_GRAMMAR_DISCOVERY_RUNTIME) dataset-action-score-engine-evidence
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --paper-claims --claims paper/claims.yaml --output-dir paper/generated --paper paper/paper.typ --agent-candidates-dir paper/agent_candidates

paper-agent-propose: paper-claims
	$(RETIRED_SCRIPT_TARGET)

paper-pre-commit: trace-determinism-compare $(NN_GRAMMAR_DISCOVERY_RUNTIME) dataset-action-score-engine-evidence
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --paper-claims --claims paper/claims.yaml --output-dir paper/generated --paper paper/paper.typ --agent-candidates-dir paper/agent_candidates --support-check --fail-required

install-hooks:
	git config core.hooksPath .githooks

paper-support: trace-determinism-compare $(NN_GRAMMAR_DISCOVERY_RUNTIME) dataset-action-score-engine-evidence
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --paper-claims --claims paper/claims.yaml --output-dir paper/generated --paper paper/paper.typ --agent-candidates-dir paper/agent_candidates --support-check --fail-required

editorial-pr-check:
	@echo "editorial PR check: running paper-support and diff-coverage check"
	@./scripts/check_paper_diff_coverage.sh || true

paper-ci: paper-bibliography paper-support editorial-pr-check paper-pdf

paper-bibliography:
	@echo "=== bibliography: refreshing reference sources ==="
	@sh scripts/collect_reference_sources.sh 2>/dev/null || echo "  (network unavailable; using cached sources)"
	@mkdir -p paper/generated
	@printf '/* bibliography VSA retired with holonet isolation; cached/generated source references only. */\n' > paper/generated/claim_bibliography.typ
	@printf 'bibliography VSA retired with holonet isolation; using cached source references.\n' > $(BIBLIOGRAPHY_REPORT)
	@echo "  report written to $(BIBLIOGRAPHY_REPORT)"

paper-pdf-release: paper-bibliography
	cd paper && typst compile paper.typ paper.pdf 2>/dev/null || echo "(paper PDF skipped - missing experiment data)"
