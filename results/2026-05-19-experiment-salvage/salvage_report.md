# Experiment Salvage Cycle

Date: 2026-05-19

Purpose: decide whether the sandbox experiments should stay active, be narrowed, or be archived after the promoted-result cleanup.

## Commands

```sh
make experimental-results RESULTS_DIR=results/2026-05-19-experiment-salvage
./build/float/haiku_word_lm --corpus data/haiku_open/clean/hf_dictionary_clean.txt --word-list data/haiku_open/clean/hf_dictionary_vocab.txt --epochs 2 --lr 0.004 --seed 20240518 --max-candidates 64 --strict-adjacency 1 --samples 3 --sample 1 --trace results/2026-05-19-experiment-salvage/haiku_word_lm.trace.jsonl --save build/float/haiku_word_lm_artifacts/salvage_hf_dictionary.nnckpt
./build/float/haiku_understanding --corpus data/haiku_open/clean/hf_dictionary_clean.txt --max-poems 4000 --epochs 6 --lr 0.004 --seed 20240518 --trace results/2026-05-19-experiment-salvage/haiku_understanding_4k.trace.jsonl --trace-limit 4000 --save build/float/haiku_understanding_artifacts/salvage_hf_dictionary_4k.nnckpt
```

## Verdict

| Experiment | Salvage verdict | Reason |
| --- | --- | --- |
| ARC-AGI fixture induction | Keep, but narrow | The fixture loader certifies 12/12 fixture tasks and 324/324 cells, and the learned/structural proposers cover 100% at budget 8. This is useful as a typed-family induction fixture, not as an ARC-generalization claim. |
| Program synthesis | Keep and scale first | At budget 16, `neural_structural` reaches 62.5% test compression and grammar reaches 75%; at budget 64 the task saturates. The domain fits the verifier/discovery story but the current corpus is too small. |
| Haiku understanding | Keep as coherence benchmark | On 4k dictionary-filtered poems, valid top-1 is 80.02% and coherence top-1 is 97.69%. The remaining hard failures are line splice at 31.87% and word salad at 65.38%. |
| Haiku generation | Do not promote | Strict adjacency is useful for a corpus-grammar trace, but samples still expose local-transition behavior rather than poem-level coherence. Keep only as support tooling for the understanding benchmark. |
| Multitask shared scorer | Keep only as negative diagnostic | Shared training handles grid/delayed-claim but does not solve ARC: shared ARC top-1 is 28.5%, single-task ARC top-1 is 42.8%, and both miss exact-grid reconstruction. |

## Patch Applied

`examples/haiku_understanding.c` now precomputes topic-splice and same-season-splice source choices once per run. This preserves the corruption semantics while avoiding repeated O(n^2) source searches during every epoch and evaluation pass.

The 4k run still spends noticeable time building the cache, but training itself completed quickly after that: 134,400 updates in 3.86 training seconds. Before the cache, 4k and full-corpus runs were not practical for this cycle.

## Opinionated Next Step

Do not delete the experiments yet. Prune the claims instead:

1. Move program synthesis to the front of the next research cycle and scale the dataset with family-level held-out splits.
2. Keep ARC only if the next step is a public/family split that can falsify fixture overfitting.
3. Keep haiku understanding, but stop treating haiku generation as evidence.
4. Keep multitask as a failure-analysis tool, not as a path to the main claim until ARC gets a geometry-aware head or separate adapter path.
