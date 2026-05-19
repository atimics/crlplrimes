# Verifier-Gated Haiku LM Prototype

`examples/haiku_lm.c` is the first language-modeling wedge for the engine. It
is intentionally not a transformer. It uses the existing action-scorer
substrate:

1. A text prefix becomes the state.
2. Each legal next character is an action candidate.
3. A lightweight haiku verifier gates candidates by line count, line length, and a syllable proxy.
4. Corpus unigram/bigram/trigram counts become explicit structural priors.
5. The current dense MLP ranks the surviving candidates with softmax loss.
6. Generation emits replayable trace rows when `--trace` is supplied.
7. Optional `--word-list` loads a fixed lexicon. With `--lexicon-gate 1`, the
   lexicon becomes a symbolic grammar: generated word prefixes must remain in
   the trie, and word boundaries are legal only after complete dictionary words.
   The trie also stores the shortest completion distance, so a letter is legal
   only if the word can still finish inside the current line budget. With
   `--lexicon-gate 0`, the same list is used only for post-hoc scoring.

The claim this supports is narrow: the engine can train a verifier-gated character generator over a tiny corpus. It does not support an LLM-scale claim, transformer claim, or broad poetry-quality claim.

`examples/haiku_word_lm.c` is the v2 wedge. It changes the action space from
characters to whole words plus `<newline>` and `<eos>` actions. The symbolic
side supplies the dataset vocabulary, train-split adjacency graph,
syllable-proxy line budget, line-length budget, and legal boundary actions; the
neural scorer only ranks a bounded legal candidate set. This makes "real words"
and observed haiku-local transitions part of the trained grammar rather than a
post-hoc sample score.

Useful commands:

```sh
make haiku-lm-smoke
make haiku-word-lm-smoke
make haiku-corpus-synthetic
make haiku-corpus-hf
make haiku-lm
make haiku-lm-lexicon
make haiku-lm-synthetic
make haiku-lm-hf
make haiku-word-lm
make haiku-word-lm-synthetic
make haiku-word-lm-hf
make haiku-word-lm-hf-dictionary
make haiku-understanding-smoke
make haiku-understanding-synthetic
make haiku-understanding-hf
make haiku-understanding-hf-dictionary
make haiku-lm HAIKU_LM_WORD_LIST=/path/to/scrabble_or_word_list.txt
make haiku-word-lm HAIKU_WORD_LM_WORD_LIST=/path/to/scrabble_or_word_list.txt
make haiku-lm HAIKU_LM_WORD_LIST=/path/to/scrabble_or_word_list.txt HAIKU_LM_LEXICON_GATE=0
./build/float/haiku_lm --epochs 12 --sample 1 --top-k 5 --trace build/float/haiku_lm_artifacts/trace.jsonl --save build/float/haiku_lm_artifacts/haiku.nnckpt
./build/float/haiku_word_lm --corpus data/haiku_open/clean/hf_haiku_clean.txt --word-list data/haiku_open/clean/hf_vocab.txt --epochs 2 --trace build/float/haiku_word_lm_artifacts/trace.hf.jsonl
```

`make haiku-corpus-synthetic` generates `data/haiku_synthetic_5000.txt`
from `data/wordlists/enable2k.txt` with a deterministic template grammar and
the same vowel-run syllable proxy used by the C verifier. Its summary is written
to `build/float/haiku_corpus_synthetic/summary.json`, and its generated
vocabulary is written to `data/wordlists/haiku_synthetic_vocab.txt`.
`make haiku-lm-synthetic` uses that smaller generated vocabulary as the active
lexicon gate so samples stay inside the corpus grammar. This is training data
for scale testing, not a human-authored poetry corpus.

`make haiku-corpus-hf` cleans user-provided Hugging Face haiku dumps from
`data/haiku_open/raw/huggingface` into `data/haiku_open/clean/hf_haiku_clean.txt`.
It supports parquet, JSON/JSONL, CSV/TSV, and plain text inputs. The cleaner
normalizes to ASCII lowercase, requires exactly three nonempty lines, enforces
the C verifier's line length and syllable-proxy bounds, deduplicates stanzas,
writes `data/haiku_open/clean/hf_vocab.txt`, and records source hashes and
reject counts in `data/haiku_open/clean/hf_manifest.json`. Parquet input
requires Python with `pandas` and a parquet engine such as `pyarrow`.
`make haiku-corpus-hf-dictionary` runs the same cleaner with the
Scrabble-like list as a hard corpus filter and writes separate
`hf_dictionary_*` artifacts. The word-action model still derives its active
vocabulary and adjacency graph from the accepted dataset.

`make haiku-lm-lexicon` defaults to `data/wordlists/enable2k.txt`, a
public-domain Scrabble-like word-game list documented in
`data/wordlists/README.md`.

The lexicon gate is part of the candidate-set verifier when enabled, so it can
change both training rows and generated candidates. A run with
`--word-list /usr/share/dict/web2` or a Scrabble list prints per-sample
`lexicon_v1` rows and adds these fields to the summary JSON:

- `lexicon_enabled`
- `lexicon_gate_enabled`
- `lexicon_entries`
- `lexicon_trie_nodes`
- `lexicon_hash`
- `generated_words`
- `generated_dictionary_words`
- `generated_dictionary_ratio`
- `generated_reject_shape`
- `generated_reject_lexicon_prefix`
- `generated_reject_lexicon_boundary`
- `generated_reject_lexicon_completion`
- `generated_reject_dead_end`

This remains a surface-form gate. It can prevent pseudo-word tokens under a
given dictionary, but it does not measure grammar, meaning, poetic quality, or
whether a generated line is a good haiku.

The word-action model writes `crlp.haiku_word_lm.generation_trace.v1` rows with
candidate action ids, candidate tokens, scores, probabilities, selected action,
checkpoint hash, state hash, feature encoder version, adjacency hash, and
verifier reject counts. Its summary row is `crlp.haiku_word_lm.summary.v1`.
By default `--strict-adjacency 1` builds the adjacency graph from the train
split only, so validation rows skipped by the gate are visible as unsupported
held-out transitions rather than hidden leakage. The current implementation is
still a local action scorer over handcrafted state features: there are no
learned token embeddings, no recurrent state, no transformer attention, and no
broad language-model claim.

`examples/haiku_understanding.c` is the first non-generation understanding
gate. It trains the same action-scorer substrate on paired candidates:

```text
real haiku vs corrupted haiku
```

The corruption families are line shuffle, line splice from another poem,
syllable-budget break, repetition, frequent-word salad, topic splice,
same-season splice, and plausible topic splice. The plausible splice family
preserves line position and syllable form, prefers train-adjacency-plausible
replacement lines, and uses the resulting pair as a coherence negative rather
than an obvious form violation. The candidate features include syllable errors, line
balance, repetition, stopword/nature/season/abstract word ratios,
word-frequency priors, train-split adjacency coverage, line-to-line topic
similarity, poem topic tightness, per-line topic coverage, cross-line content
reuse, minimum line-pair similarity, line-pair spread, topic-core consistency,
topic drift, and a symbolic poem-plan score. Feature encoder v3 keeps this as
a small neurosymbolic verifier: the dense scorer sees explicit poem-level
coherence aggregates rather than relying on a larger hidden model to infer
them.

The trace schema is `crlp.haiku_understanding.trace.v1`; rows are extended with
additive verifier fields such as `real_plan_score`, `corrupt_plan_score`,
`plan_score_delta`, `symbolic_selected`, topic-tightness deltas, minimum
line-similarity deltas, line-spread values, and adjacency ratios. The summary
schema is `crlp.haiku_understanding.summary.v1` and includes neural and
symbolic top-1 diagnostics by corruption family, `coherence_top1` over the
topic-splice families, and a `hard_negative_*` cohort for `line_splice`,
`word_salad`, `topic_splice`, `same_season_splice`, and
`plausible_topic_splice`. Plausibility traces also expose bigram and exact
syllable ratios. The `plausible_negative_*` cohort isolates the form-valid,
adjacency-plausible coherence failures. This benchmark measures whether the
engine can reject near-miss poems, not whether generated poems are good.

Training also includes a small online hard-negative loop for the plausible
splice family. After the normal update, the scorer re-scores the plausible
splice; if the corrupt candidate is still tied with or above the real poem, the
example receives `--plausible-negative-extra-updates` additional updates. The
summary records `mined_plausible_negative_examples` and
`mined_plausible_negative_updates`, making the curriculum auditable rather than
silently changing the dataset mix.

Next architecture gates before calling this an LLM:

- token embeddings
- sequence windows as first-class training examples
- recurrent or transformer state
- learned poem-level topic or image embeddings beyond the current lexical topic
  proxy
- residual blocks and normalization
- AdamW or another adaptive optimizer
- held-out perplexity reporting across larger corpora
