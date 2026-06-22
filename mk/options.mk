# User-tunable defaults for Make targets. Binary names and recipes stay in the root Makefile.
# Keep these includes in dependency order: later defaults may reference earlier names.

include mk/options_core.mk
include mk/options_arc.mk
include mk/options_program.mk
include mk/options_paper_agent.mk
include mk/options_signal.mk
include mk/options_haiku.mk
include mk/options_grammar.mk
include mk/options_evidence.mk
include mk/options_signal_wiring.mk
