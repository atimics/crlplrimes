# Human-scale entry points for the build and evidence surfaces. Keep the long
# binary recipes in the main Makefile; keep tier selection and policy gates here.

.PHONY: help core promoted evidence sandbox evidence-audit all-core source-policy-check source-policy-report
.PHONY: test sanitize check experimental-check clean

CORE_EXAMPLE_TARGETS := \
	$(GO_EXPERIMENT) \
	$(XOR_EXAMPLE) \
	$(BITWISE_RECURSIVE) \
	$(TILE_RECURSIVE) \
	$(TILE_FEATURE_ENTROPY) \
	$(LIFE_CURRICULUM) \
	$(LAYER_DISCOVERY) \
	$(ACTION_SCORE_EXAMPLE) \
	$(DATASET_ACTION_SCORE) \
	$(BENCHMARK)

PROMOTED_BINARY_TARGETS := \
	$(PRIME_GAP_EXAMPLE) \
	$(PRIME_REPLAY) \
	$(PRIME_COMPARE) \
	$(PRIME_RULE_SEARCH) \
	$(DELAYED_CLAIM) \
	$(GRID_PATH) \
	$(MODULAR_GATE_SEARCH)

SANDBOX_BINARY_TARGETS := \
	$(MULTITASK_DOMAIN) \
	$(SYMBOLIC_DOMAINS) \
	$(PROGRAM_SYNTHESIS) \
	$(CROSS_DOMAIN_TRANSFER) \
	$(SIGNAL_FLIGHT) \
	$(SIGNAL_MINING) \
	$(SIGNAL_NPC_WORKER) \
	$(HAIKU_LM) \
	$(HAIKU_UNDERSTANDING) \
	$(HAIKU_RULE_SEARCH) \
	$(HAIKU_LINE_SEARCH)

ALL_BINARY_TARGETS := \
	$(CORE_EXAMPLE_TARGETS) \
	$(PROMOTED_BINARY_TARGETS) \
	$(SANDBOX_BINARY_TARGETS)

help:
	@echo "CRLPLRIMES build tiers"
	@echo ""
	@echo "  make core          Build library/core objects and the unit test runner"
	@echo "  make test          Run the unit suite"
	@echo "  make promoted      Build canonical evidence-spine binaries"
	@echo "  make all           Build core, promoted, and sandbox binaries"
	@echo "  make evidence      Refresh promoted evidence gates and results"
	@echo "  make sandbox       Run non-promoted smoke coverage"
	@echo "  make paper-support Check generated paper claim support"
	@echo "  make check         Run CI-style local checks"
	@echo ""
	@echo "Controls: REAL=float|double NN_BLAS=accelerate|openblas|none"

# core: library objects + grammar-discovery + paper + unit test runner.
core: $(NN_OBJS) $(NN_GRAMMAR_DISCOVERY_OBJ) $(NN_ARC_SELFPLAY_OBJ) $(NN_PAPER_OBJ) $(TEST_RUNNER)

# promoted: canonical evidence binaries, without Signal/sandbox/grammar-discovery deps.
promoted: $(PROMOTED_BINARY_TARGETS)

# all is the full superset including sandboxes and integrations.
all: $(ALL_BINARY_TARGETS)

# evidence: one promoted truth pass: policy, binaries, result capture, numeric gates, trace replay.
evidence: source-policy-check promoted results numeric-certificate-promoted trace-determinism-compare

# sandbox: explicit smoke surface for non-promoted experiments.
sandbox: experimental-check

# evidence-audit: list every registered claim with support status.
evidence-audit: promoted $(NN_GRAMMAR_DISCOVERY_RUNTIME) dataset-action-score-engine-evidence
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --paper-claims --claims paper/claims.yaml --output-dir paper/generated --paper paper/paper.typ --agent-candidates-dir paper/agent_candidates --support-check --fail-required --evidence-origin-audit

all-core: core

source-policy-check:
	@found=$$(find . -path ./.git -prune -o -path ./build -prune -o -path ./paper/sources -prune -o \( -name '*.py' -o -name '*.pyi' -o -name '*.pyc' -o -name '__pycache__' \) -print); \
	if [ -n "$$found" ]; then echo "$$found"; exit 1; fi

source-policy-report:
	@echo "Tracked Python source files:"
	@git ls-files '*.py' '*.pyi' | sed 's/^/  /' || true
	@echo ""
	@echo "Policy: orchestration is first-class C runtime surface; tracked Python must be"
	@echo "ported into C or retired before merge."

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

sanitize: $(SANITIZE_TEST_RUNNER)
	./$(SANITIZE_TEST_RUNNER)

check: source-policy-check ilxyr-integration-check test sanitize action-score symbolic-domains modular-gate-search-smoke delayed-claim-self-test delayed-claim-eval grid-path-eval prime-rule-search-smoke prime-gap-eval
	$(MAKE) test REAL=double
	$(MAKE) numeric-certificate-mandatory
	$(MAKE) paper-support

experimental-check: program-synthesis program-synthesis-baselines multitask-domain haiku-understanding-smoke haiku-coherence-hard-negatives haiku-rule-search-smoke

clean:
	rm -rf $(BUILD_DIR)
