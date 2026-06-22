# Action scorer and dataset action-score smoke/evidence targets.

.PHONY: action-score dataset-action-score dataset-action-score-smoke dataset-action-score-replay-smoke
.PHONY: dataset-action-score-adapter-smoke dataset-action-score-multitarget-smoke
.PHONY: dataset-action-score-bitwise-smoke dataset-action-score-bitwise-macro-smoke
.PHONY: dataset-action-score-haiku-smoke dataset-action-score-grid-smoke
.PHONY: dataset-action-score-delayed-claim-smoke dataset-action-score-prime-gap-smoke
.PHONY: dataset-action-score-arc-smoke dataset-action-score-program-smoke
.PHONY: dataset-action-score-signal-smoke dataset-action-score-rest-smoke
.PHONY: dataset-action-score-engine-evidence

action-score: $(ACTION_SCORE_EXAMPLE)
	./$(ACTION_SCORE_EXAMPLE)

dataset-action-score: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --dataset $(DATASET_ACTION_SCORE_FIXTURE) --features 4 --task dataset_threshold_actions --epochs 800 --hidden 8 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/dataset_action_score.nnckpt

dataset-action-score-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --dataset $(DATASET_ACTION_SCORE_FIXTURE) --features 4 --task dataset_threshold_actions --epochs 800 --hidden 8 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/smoke.nnckpt

dataset-action-score-replay-smoke: $(DATASET_ACTION_SCORE) dataset-action-score-smoke
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --replay-trace $(DATASET_ACTION_SCORE_DIR)/smoke_trace.jsonl --task dataset_threshold_replay_actions --epochs 300 --hidden 12 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/replay_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/replay_smoke.nnckpt

dataset-action-score-adapter-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --adapter threshold --task adapter_threshold_actions --epochs 800 --hidden 8 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/adapter_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/adapter_smoke.nnckpt

dataset-action-score-multitarget-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment threshold_multi --task threshold_multi_actions --epochs 800 --hidden 8 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/multitarget_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/multitarget_smoke.nnckpt

dataset-action-score-bitwise-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --adapter bitwise_truth_table --task bitwise_truth_table_actions --epochs 800 --hidden 16 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/bitwise_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/bitwise_smoke.nnckpt

dataset-action-score-bitwise-macro-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --adapter bitwise_macro_step --task bitwise_macro_step_actions --epochs 1200 --hidden 32 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/bitwise_macro_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/bitwise_macro_smoke.nnckpt

dataset-action-score-haiku-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment haiku_adjacency --corpus $(HAIKU_LM_CORPUS) --task haiku_adjacency_actions --epochs 900 --rollout-passes 2 --hidden 16 --lr 0.04 --max-groups 64 --max-candidates 96 --trace $(DATASET_ACTION_SCORE_DIR)/haiku_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/haiku_smoke.nnckpt

dataset-action-score-grid-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment grid_path --task grid_path_actions --epochs 500 --hidden 24 --lr 0.04 --max-groups 96 --trace $(DATASET_ACTION_SCORE_DIR)/grid_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/grid_smoke.nnckpt

dataset-action-score-delayed-claim-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment delayed_claim --task delayed_claim_actions --epochs 500 --hidden 24 --lr 0.04 --max-groups 96 --trace $(DATASET_ACTION_SCORE_DIR)/delayed_claim_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/delayed_claim_smoke.nnckpt

dataset-action-score-prime-gap-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment prime_gap_candidate --task prime_gap_candidate_actions --epochs 500 --hidden 24 --lr 0.04 --max-groups 96 --trace $(DATASET_ACTION_SCORE_DIR)/prime_gap_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/prime_gap_smoke.nnckpt

dataset-action-score-arc-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment arc_cell_color --task arc_cell_color_actions --epochs 500 --hidden 24 --lr 0.04 --trace $(DATASET_ACTION_SCORE_DIR)/arc_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/arc_smoke.nnckpt

dataset-action-score-program-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment program_synthesis_ops --task program_synthesis_ops_actions --epochs 500 --hidden 24 --lr 0.04 --max-groups 96 --trace $(DATASET_ACTION_SCORE_DIR)/program_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/program_smoke.nnckpt

dataset-action-score-signal-smoke: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	./$(DATASET_ACTION_SCORE) --experiment signal_wasd --task signal_wasd_actions --epochs 500 --hidden 24 --lr 0.04 --max-groups 96 --trace $(DATASET_ACTION_SCORE_DIR)/signal_smoke_trace.jsonl --save $(DATASET_ACTION_SCORE_DIR)/signal_smoke.nnckpt

dataset-action-score-rest-smoke: dataset-action-score-grid-smoke dataset-action-score-delayed-claim-smoke dataset-action-score-prime-gap-smoke dataset-action-score-arc-smoke dataset-action-score-program-smoke dataset-action-score-signal-smoke

dataset-action-score-engine-evidence: $(DATASET_ACTION_SCORE)
	mkdir -p $(DATASET_ACTION_SCORE_DIR)
	rm -f $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment bitwise_macro_step --task bitwise_macro_step_actions --epochs 300 --hidden 32 --lr 0.04 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment haiku_adjacency --corpus $(HAIKU_LM_CORPUS) --task haiku_adjacency_actions --epochs 300 --rollout-passes 2 --hidden 16 --lr 0.04 --max-groups 64 --max-candidates 96 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment grid_path --task grid_path_actions --epochs 300 --hidden 24 --lr 0.04 --max-groups 96 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment delayed_claim --task delayed_claim_actions --epochs 300 --hidden 24 --lr 0.04 --max-groups 96 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment prime_gap_candidate --task prime_gap_candidate_actions --epochs 300 --hidden 24 --lr 0.04 --max-groups 96 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment arc_cell_color --task arc_cell_color_actions --epochs 300 --hidden 24 --lr 0.04 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment program_synthesis_ops --task program_synthesis_ops_actions --epochs 300 --hidden 24 --lr 0.04 --max-groups 96 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
	./$(DATASET_ACTION_SCORE) --experiment signal_wasd --task signal_wasd_actions --epochs 300 --hidden 24 --lr 0.04 --max-groups 96 --evidence $(DATASET_ACTION_SCORE_EVIDENCE)
