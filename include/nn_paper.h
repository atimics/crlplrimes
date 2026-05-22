#ifndef NN_PAPER_H
#define NN_PAPER_H

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

#endif
