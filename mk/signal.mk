# Signal integration targets. These are optional: when ../signal is missing,
# every Signal entry point reports a skipped target instead of failing.

.PHONY: signal-flight-data signal-flight signal-flight-pain signal-flight-rollout-eval
.PHONY: signal-mining signal-mining-smoke
.PHONY: signal-npc-worker signal-npc-worker-smoke signal-npc-worker-economy
.PHONY: signal-npc-worker-economy-stress signal-npc-worker-persistent
.PHONY: signal-npc-worker-persistent-smoke
.PHONY: signal-live-brain signal-live-brain-smoke signal-live-swarm signal-live-status
.PHONY: signal-live-active-summary signal-flight-report signal-shadow-replay-audit

ifeq ($(WITH_SIGNAL),1)

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

signal-npc-worker: $(SIGNAL_NPC_WORKER)
	mkdir -p $(SIGNAL_NPC_WORKER_RESULT_DIR)
	./$(SIGNAL_NPC_WORKER) --train-groups $(SIGNAL_NPC_WORKER_TRAIN_GROUPS) --test-groups $(SIGNAL_NPC_WORKER_TEST_GROUPS) --epochs $(SIGNAL_NPC_WORKER_EPOCHS) --learning-rate $(SIGNAL_NPC_WORKER_LEARNING_RATE) --seed $(SIGNAL_NPC_WORKER_SEED) --trace "$(SIGNAL_NPC_WORKER_RESULT_DIR)/trace.jsonl" --summary "$(SIGNAL_NPC_WORKER_RESULT_DIR)/summary.jsonl" --save "$(SIGNAL_NPC_WORKER_RESULT_DIR)/signal_npc_worker.nnckpt"

signal-npc-worker-smoke: SIGNAL_NPC_WORKER_RESULT_DIR=$(BUILD_DIR)/signal_npc_worker_smoke
signal-npc-worker-smoke: SIGNAL_NPC_WORKER_TRAIN_GROUPS=12
signal-npc-worker-smoke: SIGNAL_NPC_WORKER_TEST_GROUPS=6
signal-npc-worker-smoke: SIGNAL_NPC_WORKER_EPOCHS=2
signal-npc-worker-smoke: signal-npc-worker

signal-npc-worker-economy: SIGNAL_NPC_WORKER_RESULT_DIR=$(SIGNAL_NPC_WORKER_ECONOMY_RESULT_DIR)
signal-npc-worker-economy: SIGNAL_NPC_WORKER_TRAIN_GROUPS=96
signal-npc-worker-economy: SIGNAL_NPC_WORKER_TEST_GROUPS=768
signal-npc-worker-economy: SIGNAL_NPC_WORKER_EPOCHS=2
signal-npc-worker-economy: signal-npc-worker

signal-npc-worker-economy-stress: SIGNAL_NPC_WORKER_RESULT_DIR=$(SIGNAL_NPC_WORKER_ECONOMY_RESULT_DIR)/stress
signal-npc-worker-economy-stress: SIGNAL_NPC_WORKER_TRAIN_GROUPS=12
signal-npc-worker-economy-stress: SIGNAL_NPC_WORKER_TEST_GROUPS=768
signal-npc-worker-economy-stress: SIGNAL_NPC_WORKER_EPOCHS=1
signal-npc-worker-economy-stress: signal-npc-worker

signal-npc-worker-persistent: $(SIGNAL_NPC_WORKER)
	mkdir -p "$(SIGNAL_NPC_WORKER_PERSISTENT_RESULT_DIR)"
	./$(SIGNAL_NPC_WORKER) --persistent-economy --ticks $(SIGNAL_NPC_WORKER_PERSISTENT_TICKS) --npcs $(SIGNAL_NPC_WORKER_PERSISTENT_NPCS) --stations $(SIGNAL_NPC_WORKER_PERSISTENT_STATIONS) --seed $(SIGNAL_NPC_WORKER_SEED) --trace "$(SIGNAL_NPC_WORKER_PERSISTENT_RESULT_DIR)/trace.jsonl" --summary "$(SIGNAL_NPC_WORKER_PERSISTENT_RESULT_DIR)/summary.jsonl"

signal-npc-worker-persistent-smoke: SIGNAL_NPC_WORKER_PERSISTENT_RESULT_DIR=$(BUILD_DIR)/signal_npc_worker_persistent_smoke
signal-npc-worker-persistent-smoke: SIGNAL_NPC_WORKER_PERSISTENT_TICKS=48
signal-npc-worker-persistent-smoke: SIGNAL_NPC_WORKER_PERSISTENT_NPCS=12
signal-npc-worker-persistent-smoke: SIGNAL_NPC_WORKER_PERSISTENT_STATIONS=8
signal-npc-worker-persistent-smoke: signal-npc-worker-persistent

signal-live-brain:
	node scripts/signal_live_brain.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --session "$(SIGNAL_LIVE_BRAIN_SESSION)" --log "$(SIGNAL_LIVE_BRAIN_LOG)" --callsign "$(SIGNAL_LIVE_BRAIN_CALLSIGN)" --max-ms "$(SIGNAL_LIVE_BRAIN_MAX_MS)" --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" --option-mode "$(SIGNAL_LIVE_BRAIN_OPTION_MODE)" $(SIGNAL_LIVE_BRAIN_PROTOCOL_ARG) $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-brain-smoke:
	node scripts/signal_live_brain.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --session "$(SIGNAL_LIVE_BRAIN_SESSION)" --log "$(SIGNAL_LIVE_BRAIN_LOG)" --callsign "$(SIGNAL_LIVE_BRAIN_CALLSIGN)" --max-ms 5000 --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" --option-mode "$(SIGNAL_LIVE_BRAIN_OPTION_MODE)" $(SIGNAL_LIVE_BRAIN_PROTOCOL_ARG) $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-swarm:
	node scripts/signal_live_swarm.mjs --server "$(SIGNAL_LIVE_BRAIN_SERVER)" --count "$(SIGNAL_LIVE_SWARM_COUNT)" --min-active "$(SIGNAL_LIVE_SWARM_MIN_ACTIVE)" --max-ms "$(SIGNAL_LIVE_SWARM_MAX_MS)" --out-dir "$(SIGNAL_LIVE_SWARM_OUT_DIR)" --callsign-prefix "$(SIGNAL_LIVE_SWARM_CALLSIGN_PREFIX)" --stagger-ms "$(SIGNAL_LIVE_SWARM_STAGGER_MS)" --brain-mode "$(SIGNAL_LIVE_BRAIN_MODE)" --option-mode "$(SIGNAL_LIVE_BRAIN_OPTION_MODE)" $(SIGNAL_LIVE_BRAIN_PROTOCOL_ARG) $(SIGNAL_LIVE_BRAIN_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_OPTION_CHECKPOINT_ARG) $(SIGNAL_LIVE_BRAIN_EXTRA)

signal-live-status:
	node scripts/signal_live_status.mjs --log "$(SIGNAL_LIVE_STATUS_LOG)" --interval-ms "$(SIGNAL_LIVE_STATUS_INTERVAL_MS)" $(SIGNAL_LIVE_STATUS_EXTRA)

signal-live-active-summary:
	$(RETIRED_SCRIPT_TARGET)

signal-flight-report:
	@if [ -f "$(SIGNAL_FLIGHT_RESULTS_DIR)/summary.md" ]; then echo "using existing Signal flight report: $(SIGNAL_FLIGHT_RESULTS_DIR)/summary.md"; else echo "missing Signal flight report; port report generation into C before refreshing it"; exit 2; fi

signal-shadow-replay-audit:
	$(RETIRED_SCRIPT_TARGET)

else

signal-flight-data signal-flight signal-flight-pain signal-flight-rollout-eval signal-mining signal-mining-smoke signal-npc-worker signal-npc-worker-smoke signal-npc-worker-economy signal-npc-worker-economy-stress signal-npc-worker-persistent signal-npc-worker-persistent-smoke signal-live-brain signal-live-brain-smoke signal-live-swarm signal-live-status signal-live-active-summary signal-flight-report signal-shadow-replay-audit:
	@echo "$@: skipped (../signal not found)"

endif
