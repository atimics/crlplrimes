#ifndef NN_PAPER_H
#define NN_PAPER_H

#include <stddef.h>

int nn_paper_write_data(const char *json_output_path,
                        const char *typ_output_path,
                        const char *comparison_path,
                        const char *prime_gap_trace_path,
                        const char *grid_path_trace_path,
                        const char *delayed_claim_trace_path,
                        const char *arc_agi_path,
                        const char *prime_gap_path);

int nn_paper_write_claims(const char *claims_path,
                          const char *output_dir,
                          const char *paper_path,
                          const char *agent_candidates_dir,
                          int support_check,
                          int fail_required,
                          int fail_unlinked_roadmap);

int nn_paper_write_trace_determinism(const char *const *trace_paths,
                                     size_t trace_path_count,
                                     const char *output_path,
                                     const char *canonical_output_path,
                                     int gate);

#ifdef NN_ENABLE_TEST_API
int nn_paper_test_claim_governance_contract(const char *work_dir,
                                            const char *claims_path);
#endif

#endif
