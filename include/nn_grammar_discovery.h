#ifndef NN_GRAMMAR_DISCOVERY_H
#define NN_GRAMMAR_DISCOVERY_H

#include <stddef.h>

int nn_grammar_discovery_write_delta_outputs(const char *candidate_rows_path,
                                             const char *compiled_output_path,
                                             const char *verification_output_path);

int nn_grammar_discovery_write_neural_sketch_proposals(
    const char *sketches_path,
    const char *examples_path,
    const char *proposals_output_path,
    const char *checkpoint_input_path,
    const char *checkpoint_output_path,
    unsigned epochs,
    double learning_rate);

int nn_grammar_discovery_write_closed_loop_sketch_proposals(
    const char *sketches_path,
    const char *examples_path,
    const char *static_proposals_path,
    const char *proposals_output_path,
    const char *summary_output_path);

int nn_grammar_discovery_write_arc_visual_pipeline_ranker(
    const char *attempt_rows_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task);

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_with_cache(
    const char *attempt_rows_path,
    const char *cache_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task);

int nn_grammar_discovery_run_arc_ranker_benchmark(
    const char *attempt_rows_path,
    const char *output_dir,
    const char *report_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task);

int nn_grammar_discovery_write_arc_ranker_failure_report(
    const char *attempt_rows_path,
    const char *proposals_path,
    const char *summary_path,
    const char *output_path,
    const char *markdown_path);

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_from_jsonl(
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task);

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_from_jsonl_with_cache(
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const char *cache_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task);

int nn_grammar_discovery_dedupe_arc_visual_pipeline_attempts(
    const char *const *input_paths,
    size_t input_count,
    const char *output_path);

int nn_grammar_discovery_write_arc_visual_hole_proposer(
    const char *const *visual_only_logs,
    size_t visual_only_log_count,
    const char *const *target_scopes,
    size_t target_scope_count,
    const char *const *attempt_logs,
    size_t attempt_log_count,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *summary_path,
    unsigned nearest_signatures,
    unsigned max_attempt_seeds,
    unsigned max_ops_per_task,
    int emit_broad_repair,
    int include_selected);

int nn_grammar_discovery_write_arc_visual_curriculum(
    const char *const *visual_only_logs,
    size_t visual_only_log_count,
    const char *const *target_scopes,
    size_t target_scope_count,
    const char *attempt_log,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *summary_path,
    unsigned max_op_hints);

int nn_grammar_discovery_write_arc_synthetic_shape_probe_candidates(
    const char *failure_report_path,
    const char *public_arc_dir,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *summary_path,
    unsigned max_rows_per_task,
    unsigned max_tasks,
    const char *profile,
    int include_solved);

int nn_grammar_discovery_write_arc_synthetic_shape_curriculum(
    const char *attempt_rows_path,
    const char *output_dir,
    const char *synthetic_attempts_path,
    const char *combined_attempts_path,
    const char *summary_path,
    unsigned max_grid_size,
    const char *ops_csv,
    const char *negative_kinds_csv,
    const char *op_mode,
    unsigned max_accepted,
    unsigned max_synthetic_rows,
    unsigned max_seconds);

int nn_grammar_discovery_append_arc_probe_attempts_tsv(
    const char *base_attempts_tsv_path,
    const char *probe_attempts_jsonl_path,
    const char *combined_attempts_tsv_path,
    const char *summary_path,
    const char *split_name,
    int accepted_only,
    int dedupe);

int nn_grammar_discovery_write_haiku_coherence_hard_negatives(
    const char *trace_path,
    const char *output_dir);

int nn_grammar_discovery_write_neural_grammar_discovery_report(
    const char *domains,
    const char *output_dir,
    const char *runtime_path,
    const char *arc_agi_path,
    const char *public_arc_dir,
    const char *arc_failure_proof_path,
    const char *arc_heldout_triage_path,
    const char *haiku_word_trace_path,
    const char *haiku_understanding_trace_path,
    const char *haiku_coherence_path,
    const char *program_synthesis_trace_path,
    const char *grid_path_trace_path,
    const char *signal_mining_trace_path,
    unsigned arc_probe_compare_limit,
    unsigned arc_heldout_limit,
    unsigned arc_heldout_per_family,
    unsigned arc_heldout_compare_limit,
    unsigned arc_heldout_op_hints_per_task);

int nn_grammar_discovery_write_arc_neural_discovery(
    const char *arc_agi_path,
    const char *public_arc_dir,
    const char *diagnostics_path,
    const char *output_dir,
    unsigned clusters,
    unsigned top_k,
    unsigned verify_representatives,
    unsigned mutations_per_cluster,
    unsigned mutation_representatives);

int nn_grammar_discovery_write_arc_neural_failure_proof(
    const char *arc_agi_path,
    const char *public_arc_dir,
    const char *mutation_verification_path,
    const char *candidate_sketches_path,
    const char *triage_path,
    const char *output_dir,
    unsigned max_promotion_tasks);

int nn_grammar_discovery_write_experiment_health(
    const char *coverage_path,
    const char *numeric_summary_path,
    const char *numeric_report_path,
    const char *paper_claims_path,
    const char *signal_flight_summary_path,
    const char *signal_flight_rollout_path,
    const char *signal_live_active_path,
    const char *signal_shadow_replay_path,
    const char *neural_grammar_summary_path,
    const char *protocol_spine_path,
    const char *operator_promotions_path,
    const char *engine_evidence_path,
    const char *output_path);

int nn_grammar_discovery_write_learned_grammar_coverage(
    const char *const *arc_logs,
    size_t arc_log_count,
    const char *const *haiku_word_traces,
    size_t haiku_word_trace_count,
    const char *const *haiku_understanding_traces,
    size_t haiku_understanding_trace_count,
    const char *const *haiku_coherence_reports,
    size_t haiku_coherence_report_count,
    const char *budget_path,
    const char *output_path,
    int fail_on_regression);

#endif
