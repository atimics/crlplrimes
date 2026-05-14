/*
 * Prime-gap task module index.
 *
 * Production builds compile the implementation files under src/prime_gap/ as
 * separate translation units. The test runner includes this file with
 * NN_ENABLE_TEST_API so private helpers remain directly testable without
 * exporting them from the public task header.
 */

#if defined(NN_ENABLE_TEST_API)
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
#endif
