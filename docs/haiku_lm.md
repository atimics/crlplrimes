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
make haiku-coherence-hard-negatives
make haiku-rule-search-smoke
make haiku-rule-search-baselines
make haiku-understanding-synthetic
make haiku-understanding-hf
make haiku-understanding-hf-dictionary
make haiku-lm HAIKU_LM_WORD_LIST=/path/to/scrabble_or_word_list.txt
make haiku-word-lm HAIKU_WORD_LM_WORD_LIST=/path/to/scrabble_or_word_list.txt
make haiku-lm HAIKU_LM_WORD_LIST=/path/to/scrabble_or_word_list.txt HAIKU_LM_LEXICON_GATE=0
./build/float/haiku_lm --epochs 12 --sample 1 --top-k 5 --trace build/float/haiku_lm_artifacts/trace.jsonl --save build/float/haiku_lm_artifacts/haiku.nnckpt
./build/float/haiku_word_lm --corpus data/haiku_open/clean/hf_haiku_clean.txt --word-list data/haiku_open/clean/hf_vocab.txt --epochs 2 --trace build/float/haiku_word_lm_artifacts/trace.hf.jsonl
./build/float/haiku_rule_search --corpus data/haiku_tiny.txt --proposer closed-loop --trace build/float/haiku_rule_search_artifacts/trace.jsonl
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
requires Python with `pandas` and a parquet engine such as `pyarrow`, but the
Makefile reuses existing `data/haiku_open/clean/*` artifacts when present so
normal C experiment runs do not depend on parquet tooling.
`make haiku-corpus-hf-dictionary` runs the same cleaner with the
Scrabble-like list as a hard corpus filter and writes separate
`hf_dictionary_*` artifacts. The word-action model still derives its active
vocabulary and adjacency graph from the accepted dataset.

The local raw dataset pool is not exhausted. The HF mirror is imported into the
current English ASCII corpora. `data/haiku_open/source_manifest.tsv` also lists
PLOS arthropod haiku workbooks plus public-domain Japanese Basho and Masaoka
Shiki sources; those need separate text extraction and, for Japanese, a
different tokenizer/feature path before they are comparable training inputs.

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
reuse, minimum line-pair similarity, line-pair spread, adjacent-line fit,
adjacency/bigram products, topic-core consistency, topic drift, and a symbolic
poem-plan score. Feature encoder v5 keeps this as a small neurosymbolic
verifier: the dense scorer sees explicit poem-level coherence aggregates rather
than relying on a larger hidden model to infer them.

The trace schema is `crlp.haiku_understanding.trace.v1`; rows are extended with
additive verifier fields such as `real_plan_score`, `corrupt_plan_score`,
`plan_score_delta`, `symbolic_selected`, topic-tightness deltas, minimum
line-similarity deltas, line-spread values, adjacency ratios, and splice
provenance fields (`corrupt_line`, `source_poem_index`, `source_line`,
replacement-line similarity, local-neighbor-fit deltas, and line word/syllable
deltas). The summary schema is `crlp.haiku_understanding.summary.v1` and
includes neural and symbolic top-1 diagnostics by corruption family,
`coherence_top1` over the topic-splice families, and a `hard_negative_*` cohort
for `line_splice`, `word_salad`, `topic_splice`, `same_season_splice`, and
`plausible_topic_splice`. Plausibility traces also expose bigram and exact
syllable ratios. The `plausible_negative_*` cohort isolates the form-valid,
adjacency-plausible coherence failures. This benchmark measures whether the
engine can reject near-miss poems, not whether generated poems are good.

`examples/haiku_rule_search.c` ports the haiku word-action surface onto the
same rule-search reporting stack used by the prime/grid experiments. It
enumerates deterministic reject-action rules over the haiku word generator
state, certifies them by replaying observed train next-actions, scores their
negative-candidate coverage on train/validation/test splits, and writes
`crlp.rule_trace.v1` plus `crlp.rule_search.summary.v1` rows. The current rule
grammar covers syllable overflow, line-character overflow, early newline, early
EOS, train-adjacency absence, and line-position vocabulary support. The
`--proposer` switch supports `grammar`, `structural`, `random`, and
`closed-loop`; the closed-loop path uses the shared neural residual rule policy
with an audited `haiku_rule_features_v1` schema.

This is a verifier/certificate layer over the haiku word-action model, not a
claim that the generator has learned poetic quality. The certificate means a
rule has zero false rejections against train positives under this corpus and
candidate construction, while held-out false rejections remain visible in the
trace and summary fields.

Training also includes a small online hard-negative loop for the plausible
splice family. After the normal update, the scorer re-scores the plausible
splice; if the corrupt candidate is still tied with or above the real poem, the
example receives `--plausible-negative-extra-updates` additional updates. The
summary records `mined_plausible_negative_examples` and
`mined_plausible_negative_updates`, making the curriculum auditable rather than
silently changing the dataset mix.

The same online loop can target the weakest non-plausible families with
`--hard-negative-extra-updates`, currently `line_splice` and `word_salad`. The
Makefile default is conservative (`0`) because the first smoke check improved
train accuracy but hurt validation hard-negative top-1. When enabled, those
updates are reported as `mined_hard_negative_examples` and
`mined_hard_negative_updates`, so family-weighted training pressure remains
visible in the generated summaries.

HF-scale understanding runs use `--splice-scan-limit` through
`HAIKU_UNDERSTANDING_SPLICE_SCAN_LIMIT` to bound C-side splice-cache
construction. The default `512` keeps the import/regeneration path in C while
avoiding the previous full-corpus quadratic scan before the first training
status line.

`make haiku-coherence-hard-negatives` is the auditable report layer over the
understanding trace. It writes
`build/float/haiku_coherence_hard_negatives/hard_negative_rows.jsonl`,
`summary.jsonl`, and `summary.md` with schema
`crlp.haiku_coherence_hard_negative.v1` and
`crlp.haiku_coherence_hard_negative_summary.v1`. These rows are explicitly
`diagnostic_only_no_verifier_certification`: neural grammar discovery may use
them as temporal-frame pressure for hard-negative ranking, but they do not
support paper claims or promote haiku grammar.

`make haiku-coherence-hard-negatives-hf-dictionary` runs the same report over
the larger HF-dictionary trace path. The C learned-grammar coverage driver ingests these
rows with `--haiku-coherence-report`, and the C `experiment-health` driver promotes weak
hard-negative families such as `line_splice` and `word_salad` into explicit
diagnostic triage items.

The current HF-dictionary regeneration uses the existing cleaned corpus in C,
not a parquet-clean dependency path. With feature encoder v4 and
`--splice-scan-limit 512`, the 8k-poem run writes 12,800 validation trace rows:
overall validation top-1 is 85.96%, hard-negative top-1 is 79.83%,
topic-coherence top-1 is 96.10%, and plausible-topic-splice top-1 is 96.31%.
The remaining weak families are `line_splice` at 40.25% and `word_salad` at
70.56%, so random line replacement remains the main coherence-label gap.

Next architecture gates before calling this an LLM:

- token embeddings
- sequence windows as first-class training examples
- recurrent or transformer state
- learned poem-level topic or image embeddings beyond the current lexical topic
  proxy
- residual blocks and normalization
- AdamW or another adaptive optimizer
- held-out perplexity reporting across larger corpora
