.PHONY: experiment-schemas
.PHONY: multitask-domain multitask-adapter-heads multitask-failure-diagnostics
.PHONY: tiny-net-plateau-audit tiny-net-stronger-baseline symbolic-domains
.PHONY: program-synthesis program-synthesis-baselines program-synthesis-sketch-source
.PHONY: modular-gate-search modular-gate-search-smoke
.PHONY: cross-domain-transfer cross-domain-transfer-smoke

experiment-schemas: $(GO_EXPERIMENT) $(PRIME_GAP_EXAMPLE) $(GRID_PATH) $(DELAYED_CLAIM) $(PROGRAM_SYNTHESIS) $(ARC_AGI) $(SIGNAL_FLIGHT) $(SIGNAL_MINING)
	./$(PRIME_GAP_EXAMPLE) --schema
	./$(GRID_PATH) --schema
	./$(DELAYED_CLAIM) --schema
	./$(PROGRAM_SYNTHESIS) --schema
	./$(ARC_AGI) --schema
	./$(SIGNAL_FLIGHT) --schema
	./$(SIGNAL_MINING) --schema

multitask-domain: $(MULTITASK_DOMAIN)
	./$(MULTITASK_DOMAIN)

multitask-adapter-heads: $(MULTITASK_DOMAIN)
	./$(MULTITASK_DOMAIN) --adapter-heads-report \
		--output-dir "$(MULTITASK_ROADMAP_RESULTS_DIR)" \
		--seeds "$(MULTITASK_ADAPTER_HEADS_SEEDS)" \
		--epochs "$(MULTITASK_ADAPTER_HEADS_EPOCHS)" \
		--steps-per-epoch "$(MULTITASK_ADAPTER_HEADS_STEPS_PER_EPOCH)" \
		--eval-samples "$(MULTITASK_ADAPTER_HEADS_EVAL_SAMPLES)" \
		--hidden "$(MULTITASK_ADAPTER_HEADS_HIDDEN)" \
		--arc-ood-task "$(MULTITASK_ADAPTER_HEADS_ARC_OOD_TASK)" \
		--transfer-regression-margin "$(MULTITASK_ADAPTER_HEADS_TRANSFER_MARGIN)"

multitask-failure-diagnostics: $(MULTITASK_DOMAIN)
	$(RETIRED_SCRIPT_TARGET)

tiny-net-plateau-audit: $(MULTITASK_DOMAIN)
	./$(MULTITASK_DOMAIN) --tiny-net-plateau-audit \
		--output-dir "$(TINY_NET_PLATEAU_AUDIT_DIR)" \
		--seeds "$(TINY_NET_AUDIT_SEEDS)" \
		--capacities "$(TINY_NET_AUDIT_CAPACITIES)" \
		--epoch-grid "$(TINY_NET_AUDIT_EPOCH_GRID)" \
		--steps-per-epoch "$(TINY_NET_AUDIT_STEPS_PER_EPOCH)" \
		--eval-samples "$(TINY_NET_AUDIT_EVAL_SAMPLES)" \
		--plateau-threshold "$(TINY_NET_AUDIT_PLATEAU_THRESHOLD)" \
		--plateau-window "$(TINY_NET_AUDIT_PLATEAU_WINDOW)" \
		$(TINY_NET_AUDIT_RESUME_ARG) \
		$(TINY_NET_AUDIT_SHARD_ARG) \
		$(TINY_NET_AUDIT_EXTRA)

tiny-net-stronger-baseline: $(MULTITASK_DOMAIN)
	./$(MULTITASK_DOMAIN) --tiny-net-plateau-audit \
		--output-dir "$(TINY_NET_STRONGER_BASELINE_DIR)" \
		--seeds "$(TINY_NET_STRONGER_BASELINE_SEEDS)" \
		--capacities "$(TINY_NET_STRONGER_BASELINE_CAPACITIES)" \
		--epoch-grid "$(TINY_NET_STRONGER_BASELINE_EPOCH_GRID)" \
		--steps-per-epoch "$(TINY_NET_STRONGER_BASELINE_STEPS_PER_EPOCH)" \
		--eval-samples "$(TINY_NET_STRONGER_BASELINE_EVAL_SAMPLES)" \
		--plateau-threshold "$(TINY_NET_STRONGER_BASELINE_PLATEAU_THRESHOLD)" \
		--plateau-window "$(TINY_NET_STRONGER_BASELINE_PLATEAU_WINDOW)" \
		$(TINY_NET_STRONGER_BASELINE_RESUME_ARG) \
		$(TINY_NET_STRONGER_BASELINE_SHARD_ARG) \
		$(TINY_NET_STRONGER_BASELINE_EXTRA)

symbolic-domains: $(SYMBOLIC_DOMAINS)
	./$(SYMBOLIC_DOMAINS)

program-synthesis: $(PROGRAM_SYNTHESIS)
	./$(PROGRAM_SYNTHESIS) --proposer all --max-rules 16

program-synthesis-baselines: $(PROGRAM_SYNTHESIS)
	./$(PROGRAM_SYNTHESIS) --proposer all --budget-grid

program-synthesis-sketch-source: $(PROGRAM_SYNTHESIS)
	mkdir -p $(NEURAL_GRAMMAR_DISCOVERY_DIR)
	./$(PROGRAM_SYNTHESIS) --proposer all --max-rules 16 > $(NEURAL_GRAMMAR_DISCOVERY_PROGRAM_SYNTHESIS_TRACE)

modular-gate-search: $(MODULAR_GATE_SEARCH)
	./$(MODULAR_GATE_SEARCH)

modular-gate-search-smoke: $(MODULAR_GATE_SEARCH)
	mkdir -p $(BUILD_DIR)/traces
	./$(MODULAR_GATE_SEARCH) --max-states 128 --max-actions 24 --max-rules 8 --trace $(BUILD_DIR)/traces/modular_gate_search.jsonl

cross-domain-transfer: $(CROSS_DOMAIN_TRANSFER)
	mkdir -p $(BUILD_DIR)/traces
	./$(CROSS_DOMAIN_TRANSFER) --trace $(BUILD_DIR)/traces/cross_domain_transfer.jsonl --summary $(BUILD_DIR)/traces/cross_domain_transfer_summary.jsonl --leave-out $(BUILD_DIR)/traces/cross_domain_leave_out.jsonl

cross-domain-transfer-smoke: $(CROSS_DOMAIN_TRANSFER)
	./$(CROSS_DOMAIN_TRANSFER)
