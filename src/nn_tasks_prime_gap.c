/*
 * Prime-gap task module.
 *
 * This file intentionally uses a unity-style split: the implementation is
 * organized into focused chunks under src/prime_gap/, but still compiled as a
 * single translation unit so existing static helper boundaries, tests, and
 * command wrappers stay unchanged.
 */

#include "prime_gap/internal.c"
#include "prime_gap/data.c"
#include "prime_gap/features.c"
#include "prime_gap/symbolic_wheel.c"
#include "prime_gap/rule_search.c"
#include "prime_gap/block_verifier.c"
#include "prime_gap/eval.c"
#include "prime_gap/train.c"
#include "prime_gap/run.c"
#include "prime_gap/replay_compare.c"
