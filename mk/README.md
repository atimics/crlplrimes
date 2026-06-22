# Makefile Control Surfaces

This directory holds the human-facing Make targets that decide which evidence
surface runs. The top-level `Makefile` still owns object lists and binary
recipes; these fragments keep policy and orchestration targets findable.

| File | Surface |
| --- | --- |
| `options.mk` | Ordered manifest for all option fragments |
| `options_core.mk` | Common build/test, promoted-result, and trace defaults |
| `options_arc.mk` | ARC hypothesis-library and ARC public-data defaults |
| `options_program.mk` | Multitask, tiny-net, program, and ARC family defaults |
| `options_paper_agent.mk` | Paper agent proposal defaults |
| `options_signal.mk` | Signal experiment and live-run defaults |
| `options_haiku.mk` | Haiku, text, dataset-action-score, and coherence defaults |
| `options_grammar.mk` | Neural grammar, learned grammar, and ARC visual defaults |
| `options_evidence.mk` | Experiment-health, protocol, and numeric-certificate defaults |
| `options_signal_wiring.mk` | Optional Signal integration wiring |
| `control.mk` | Default help, build tiers, source policy, unit/check gates |
| `promoted.mk` | Prime gap, delayed claim, grid path, and trace determinism gates |
| `action_score.mk` | Action scorer and dataset action-score smoke/evidence targets |
| `arc.mk` | ARC-AGI, hypothesis-library, neural discovery, and self-play targets |
| `program.mk` | Experiment schemas, multitask audits, program synthesis, and transfer targets |
| `examples.mk` | Small demo runners and benchmark entry points |
| `signal.mk` | Optional `../signal` integrations and skipped-target fallbacks |
| `haiku.mk` | Haiku corpus, LM, understanding, and coherence targets |
| `grammar.mk` | Learned grammar coverage and ARC visual coverage diagnostics |
| `evidence.mk` | Result capture, numeric certificates, experiment-health rollups |
| `paper.mk` | Paper claims, bibliography, PDF, editorial gates |
| `eml.mk` | EML variants and benchmark helpers |
| `runtime.mk` | Go/tournament runtime helpers and legacy demo launchers |

When adding a target, prefer the narrowest matching file. Leave low-level object
recipes in the main `Makefile` until the corresponding binary family is split.
