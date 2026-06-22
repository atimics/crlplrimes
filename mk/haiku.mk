# Haiku and text experiment entry points.

.PHONY: haiku-corpus-hf haiku-corpus-hf-dictionary haiku-corpus-synthetic
.PHONY: haiku-lm haiku-lm-hf haiku-lm-lexicon haiku-lm-smoke haiku-lm-synthetic
.PHONY: haiku-coherence-hard-negatives haiku-coherence-hard-negatives-hf-dictionary
.PHONY: haiku-rule-search haiku-rule-search-baselines haiku-rule-search-smoke
.PHONY: haiku-understanding haiku-understanding-hf haiku-understanding-hf-dictionary
.PHONY: haiku-understanding-smoke haiku-understanding-synthetic
.PHONY: haiku-line-search haiku-line-search-smoke

haiku-corpus-synthetic:
	@if [ -f "$(HAIKU_CORPUS_SYNTHETIC)" ] && [ -f "$(HAIKU_CORPUS_VOCAB)" ]; then echo "using existing synthetic corpus: $(HAIKU_CORPUS_SYNTHETIC)"; else echo "missing synthetic corpus artifacts; port corpus generation into C before refreshing them"; exit 2

haiku-corpus-hf:
	@if [ -f "$(HAIKU_HF_CLEAN)" ] && [ -f "$(HAIKU_HF_VOCAB)" ] && [ -f "$(HAIKU_HF_MANIFEST)" ]; then echo "using existing cleaned HF corpus: $(HAIKU_HF_CLEAN)"; else echo "missing cleaned HF corpus artifacts; port corpus cleaning into C before refreshing them"; exit 2

haiku-corpus-hf-dictionary:
	@if [ -f "$(HAIKU_HF_DICTIONARY_CLEAN)" ] && [ -f "$(HAIKU_HF_DICTIONARY_VOCAB)" ] && [ -f "$(HAIKU_HF_DICTIONARY_MANIFEST)" ]; then echo "using existing cleaned HF dictionary corpus: $(HAIKU_HF_DICTIONARY_CLEAN)"; else echo "missing cleaned HF dictionary corpus artifacts; port corpus cleaning into C before refreshing them"; exit 2

haiku-lm: $(HAIKU_LM)
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_LM_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.jsonl --save $(HAIKU_LM_DIR)/haiku.nnckpt $(HAIKU_LM_WORD_LIST_ARG)

haiku-lm-lexicon: $(HAIKU_LM)
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_LM_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.lexicon.jsonl --save $(HAIKU_LM_DIR)/haiku.lexicon.nnckpt --word-list "$(HAIKU_LM_LEXICON_WORD_LIST)" --lexicon-gate 1

haiku-lm-smoke: $(HAIKU_LM)
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_LM_CORPUS) --epochs 2 --max-updates 1600 --samples 1 --sample 0 --trace $(HAIKU_LM_DIR)/smoke_trace.jsonl --save $(HAIKU_LM_DIR)/smoke.nnckpt $(HAIKU_LM_WORD_LIST_ARG)

haiku-lm-synthetic: $(HAIKU_LM) haiku-corpus-synthetic
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_CORPUS_SYNTHETIC) --epochs $(HAIKU_LM_SYNTHETIC_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.synthetic.jsonl --save $(HAIKU_LM_DIR)/haiku.synthetic.nnckpt --word-list "$(HAIKU_CORPUS_VOCAB)" --lexicon-gate 1

haiku-lm-hf: $(HAIKU_LM) haiku-corpus-hf
	mkdir -p $(HAIKU_LM_DIR)
	./$(HAIKU_LM) --corpus $(HAIKU_HF_CLEAN) --epochs $(HAIKU_LM_HF_EPOCHS) --lr $(HAIKU_LM_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --trace $(HAIKU_LM_DIR)/trace.hf.jsonl --save $(HAIKU_LM_DIR)/haiku.hf.nnckpt --word-list "$(HAIKU_HF_VOCAB)" --lexicon-gate 1

haiku-understanding: $(HAIKU_UNDERSTANDING)
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_LM_CORPUS) --epochs $(HAIKU_UNDERSTANDING_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --splice-scan-limit $(HAIKU_UNDERSTANDING_SPLICE_SCAN_LIMIT) --hard-negative-extra-updates $(HAIKU_UNDERSTANDING_HARD_NEGATIVE_EXTRA_UPDATES) --trace-limit $(HAIKU_UNDERSTANDING_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.nnckpt

haiku-understanding-smoke: $(HAIKU_UNDERSTANDING)
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_LM_CORPUS) --epochs 4 --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --splice-scan-limit $(HAIKU_UNDERSTANDING_SPLICE_SCAN_LIMIT) --hard-negative-extra-updates $(HAIKU_UNDERSTANDING_HARD_NEGATIVE_EXTRA_UPDATES) --trace-limit 200 --trace $(HAIKU_UNDERSTANDING_DIR)/smoke_trace.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/smoke.nnckpt

haiku-understanding-synthetic: $(HAIKU_UNDERSTANDING) haiku-corpus-synthetic
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_CORPUS_SYNTHETIC) --epochs $(HAIKU_UNDERSTANDING_SYNTHETIC_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --splice-scan-limit $(HAIKU_UNDERSTANDING_SPLICE_SCAN_LIMIT) --hard-negative-extra-updates $(HAIKU_UNDERSTANDING_HARD_NEGATIVE_EXTRA_UPDATES) --trace-limit $(HAIKU_UNDERSTANDING_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.synthetic.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.synthetic.nnckpt

haiku-understanding-hf: $(HAIKU_UNDERSTANDING) haiku-corpus-hf
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_HF_CLEAN) --epochs $(HAIKU_UNDERSTANDING_HF_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --splice-scan-limit $(HAIKU_UNDERSTANDING_SPLICE_SCAN_LIMIT) --hard-negative-extra-updates $(HAIKU_UNDERSTANDING_HARD_NEGATIVE_EXTRA_UPDATES) --trace-limit $(HAIKU_UNDERSTANDING_HF_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.hf.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.hf.nnckpt

haiku-understanding-hf-dictionary: $(HAIKU_UNDERSTANDING) haiku-corpus-hf-dictionary
	mkdir -p $(HAIKU_UNDERSTANDING_DIR)
	./$(HAIKU_UNDERSTANDING) --corpus $(HAIKU_HF_DICTIONARY_CLEAN) --epochs $(HAIKU_UNDERSTANDING_HF_EPOCHS) --lr $(HAIKU_UNDERSTANDING_LEARNING_RATE) --seed $(HAIKU_LM_SEED) --splice-scan-limit $(HAIKU_UNDERSTANDING_SPLICE_SCAN_LIMIT) --hard-negative-extra-updates $(HAIKU_UNDERSTANDING_HARD_NEGATIVE_EXTRA_UPDATES) --trace-limit $(HAIKU_UNDERSTANDING_HF_TRACE_LIMIT) --trace $(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl --save $(HAIKU_UNDERSTANDING_DIR)/haiku_understanding.hf.dictionary.nnckpt

haiku-coherence-hard-negatives: $(NN_GRAMMAR_DISCOVERY_RUNTIME) haiku-understanding-smoke
	mkdir -p $(HAIKU_COHERENCE_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --haiku-coherence-trace "$(HAIKU_COHERENCE_TRACE)" --haiku-coherence-output-dir "$(HAIKU_COHERENCE_DIR)"

haiku-coherence-hard-negatives-hf-dictionary: $(NN_GRAMMAR_DISCOVERY_RUNTIME) $(HAIKU_UNDERSTANDING_DIR)/trace.hf.dictionary.jsonl
	@if [ ! -f "$(HAIKU_COHERENCE_HF_DICTIONARY_TRACE)" ]; then echo "missing HAIKU_COHERENCE_HF_DICTIONARY_TRACE=$(HAIKU_COHERENCE_HF_DICTIONARY_TRACE); run haiku-understanding-hf-dictionary or override this variable with an existing trace"; exit 2; fi
	mkdir -p $(HAIKU_COHERENCE_HF_DICTIONARY_DIR)
	./$(NN_GRAMMAR_DISCOVERY_RUNTIME) --haiku-coherence-trace "$(HAIKU_COHERENCE_HF_DICTIONARY_TRACE)" --haiku-coherence-output-dir "$(HAIKU_COHERENCE_HF_DICTIONARY_DIR)"

haiku-rule-search: $(HAIKU_RULE_SEARCH)
	mkdir -p $(HAIKU_RULE_SEARCH_DIR)
	./$(HAIKU_RULE_SEARCH) --corpus $(HAIKU_LM_CORPUS) --max-candidates $(HAIKU_RULE_SEARCH_MAX_CANDIDATES) --max-states $(HAIKU_RULE_SEARCH_MAX_STATES) --max-rules $(HAIKU_RULE_SEARCH_MAX_RULES) --proposer $(HAIKU_RULE_SEARCH_PROPOSER) --trace $(HAIKU_RULE_SEARCH_DIR)/trace.jsonl --save $(HAIKU_RULE_SEARCH_DIR)/haiku_rules.crlprule

haiku-rule-search-smoke: $(HAIKU_RULE_SEARCH)
	mkdir -p $(HAIKU_RULE_SEARCH_DIR)
	./$(HAIKU_RULE_SEARCH) --corpus $(HAIKU_LM_CORPUS) --max-candidates 48 --max-states 600 --max-rules 16 --proposer closed-loop --trace $(HAIKU_RULE_SEARCH_DIR)/smoke_trace.jsonl --save $(HAIKU_RULE_SEARCH_DIR)/haiku_rules_smoke.crlprule

haiku-line-search: $(HAIKU_LINE_SEARCH)

haiku-rule-search-baselines: $(HAIKU_RULE_SEARCH)
	./$(HAIKU_RULE_SEARCH) --corpus $(HAIKU_LM_CORPUS) --max-candidates 48 --max-states 600 --max-rules 16 --proposer grammar
	./$(HAIKU_RULE_SEARCH) --corpus $(HAIKU_LM_CORPUS) --max-candidates 48 --max-states 600 --max-rules 16 --proposer structural
	./$(HAIKU_RULE_SEARCH) --corpus $(HAIKU_LM_CORPUS) --max-candidates 48 --max-states 600 --max-rules 16 --proposer random --proposer-seed 20240518
	./$(HAIKU_RULE_SEARCH) --corpus $(HAIKU_LM_CORPUS) --max-candidates 48 --max-states 600 --max-rules 16 --proposer closed-loop
