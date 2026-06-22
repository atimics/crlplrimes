.PHONY: neurosymbolic-tune train-cross-domain go-experiment
.PHONY: watch tournament hybrid gated-runner cross-big

neurosymbolic-tune: all
	$(RETIRED_SCRIPT_TARGET)

train-cross-domain: $(TRAIN_CROSS_DOMAIN)

go-experiment: $(GO_EXPERIMENT)

watch:
	./scripts/watch_tournament.sh

tournament: build/float/nn.o build/float/nn_blas.o build/float/nn_experiment.o build/float/nn_symbolic.o build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o
	$(CC) -std=c11 -O3 -Iinclude -DNN_REAL=float -DNN_REAL_IS_FLOAT=1 examples/go_tournament.c $^ -lm -o build/float/go_tournament
	while true; do ./build/float/go_tournament --agents 16 --games 1 --threads 8 --display 4; echo "[make] crashed, restarting in 1s..."; sleep 1; done

ifneq ($(strip $(HYBRID)),)
hybrid: $(HYBRID)
	./$(HYBRID)
else
hybrid:
	@echo "$@: skipped (HYBRID is not defined)"
endif

ifneq ($(strip $(GATED_DEMO)),)
gated-runner: $(GATED_DEMO)
	./$(GATED_DEMO)
else
gated-runner:
	@echo "$@: skipped (GATED_DEMO is not defined)"
endif

ifneq ($(strip $(CROSS_BIG)),)
cross-big: $(CROSS_BIG)
	./$(CROSS_BIG)
else
cross-big:
	@echo "$@: skipped (CROSS_BIG is not defined)"
endif
