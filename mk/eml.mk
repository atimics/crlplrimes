.PHONY: prime-gap-eml grid-path-eml delayed-claim-eml eml-benchmarks eml-claim-evidence eml-clean

EML_PRIME_GAP := $(BUILD_DIR)/prime_gap_eml
EML_GRID_PATH := $(BUILD_DIR)/grid_path_eml
EML_DELAYED_CLAIM := $(BUILD_DIR)/delayed_claim_eml
EML_DELAYED_CLAIM_SOURCE := examples/delayed_claim_eml.c

$(EML_PRIME_GAP): $(PRIME_GAP_EXAMPLE_OBJS) $(NN_OBJS) $(PRIME_GAP_TASK_HEADERS) | $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/obj/prime_gap_eml
	for src in block_verifier data eval features internal replay_compare rule_search run symbolic_wheel train; do \
		$(CC) -Iinclude -DPRIME_GAP_EML $(CPPFLAGS) $(CFLAGS) -c src/prime_gap/$$src.c -o $(BUILD_DIR)/obj/prime_gap_eml/$$src.o; \
	done
	$(CC) -Iinclude $(BUILD_DIR)/obj/prime_gap_eml/*.o $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(EML_GRID_PATH): examples/grid_path.c $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) -Iinclude -DGRID_PATH_EML $(CPPFLAGS) $(CFLAGS) examples/grid_path.c $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

ifneq ($(wildcard $(EML_DELAYED_CLAIM_SOURCE)),)
$(EML_DELAYED_CLAIM): $(EML_DELAYED_CLAIM_SOURCE) $(NN_OBJS) $(NN_HEADERS) | $(BUILD_DIR)
	$(CC) -Iinclude $(CPPFLAGS) $(CFLAGS) $(EML_DELAYED_CLAIM_SOURCE) $(NN_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

delayed-claim-eml: $(EML_DELAYED_CLAIM)
	./$(EML_DELAYED_CLAIM) --trace /tmp/dc_eml.jsonl --save /tmp/dc_eml.nnckpt --trace-train --trace-samples 200
else
delayed-claim-eml:
	@echo "$@: skipped ($(EML_DELAYED_CLAIM_SOURCE) not found)"
endif

prime-gap-eml: $(EML_PRIME_GAP)
	./$(EML_PRIME_GAP) 1000 2000 4 0 5000 2000

grid-path-eml: $(EML_GRID_PATH)
	./$(EML_GRID_PATH) --trace /tmp/gp_eml.jsonl --save /tmp/gp_eml.nnckpt --trace-train --trace-samples 200

eml-benchmarks: $(BUILD_DIR)/eml_benchmarks

eml-claim-evidence:
	@echo "EML claim evidence is unavailable: add persistent matched-parameter, multi-seed artifacts before promoting paper metrics."
	@exit 2

$(BUILD_DIR)/eml_benchmarks: $(EML_PRIME_GAP) $(EML_GRID_PATH)
	@echo "=== EML Benchmark Suite ==="
	@echo ""
	@echo "--- prime_gap EML ---"
	./$(EML_PRIME_GAP) 1000 2000 1 0 500 200 2>&1 | grep 'model supervised' -A1 | head -2 || true
	@echo ""
	@echo "--- grid_path EML ---"
	./$(EML_GRID_PATH) --trace /tmp/gp_eml_bench.jsonl --save /tmp/gp_eml_bench.nnckpt --trace-train --trace-samples 100 2>&1 | grep 'accuracy=' || true
ifneq ($(wildcard $(EML_DELAYED_CLAIM_SOURCE)),)
	@echo ""
	@echo "--- delayed_claim EML ---"
	./$(EML_DELAYED_CLAIM) --trace /tmp/dc_eml_bench.jsonl --save /tmp/dc_eml_bench.nnckpt --trace-train --trace-samples 100 2>&1 | grep 'accuracy=' || true
else
	@echo ""
	@echo "--- delayed_claim EML: skipped ($(EML_DELAYED_CLAIM_SOURCE) not found) ---"
endif

eml-clean:
	rm -rf $(BUILD_DIR)/obj/prime_gap_eml $(BUILD_DIR)/prime_gap_eml $(BUILD_DIR)/grid_path_eml $(BUILD_DIR)/delayed_claim_eml
