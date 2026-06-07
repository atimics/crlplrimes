#include "nn_grammar_discovery.h"
#include "nn_paper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg_value(int argc, char **argv, const char *name) {
    int i;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return NULL;
}

static int has_arg(int argc, char **argv, const char *name) {
    int i;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static size_t count_arg_values(int argc, char **argv, const char *name) {
    int i;
    size_t count = 0U;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            ++count;
            ++i;
        }
    }
    return count;
}

static size_t collect_arg_values(int argc,
                                 char **argv,
                                 const char *name,
                                 const char **values,
                                 size_t max_values) {
    int i;
    size_t count = 0U;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            if (count < max_values) {
                values[count] = argv[i + 1];
            }
            ++count;
            ++i;
        }
    }
    return count;
}

static const char **collect_arg_values_alloc(int argc,
                                             char **argv,
                                             const char *name,
                                             size_t count) {
    const char **values;
    if (count == 0U) {
        return NULL;
    }
    values = (const char **)calloc(count, sizeof(char *));
    if (values == NULL) {
        return NULL;
    }
    collect_arg_values(argc, argv, name, values, count);
    return values;
}

static void free_arc_arg_arrays(const char **dedupe_inputs,
                                const char **source_inputs,
                                const char **target_inputs,
                                const char **visual_only_logs,
                                const char **target_scopes,
                                const char **attempt_logs) {
    free(dedupe_inputs);
    free(source_inputs);
    free(target_inputs);
    free(visual_only_logs);
    free(target_scopes);
    free(attempt_logs);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--candidate-rows PATH --compiled-output PATH "
            "--verification-output PATH] [--sketches PATH --sketch-examples PATH "
            "--neural-proposals-output PATH] [--arc-pipeline-attempts PATH "
            "--arc-pipeline-proposals-output PATH --arc-pipeline-summary-output PATH] "
            "[--arc-pipeline-source-attempts-jsonl PATH ... "
            "--arc-pipeline-target-attempts-jsonl PATH ... "
            "--arc-pipeline-proposals-output PATH --arc-pipeline-summary-output PATH] "
            "[--arc-pipeline-dedupe-attempts-input PATH ... "
            "--arc-pipeline-dedupe-attempts-output PATH] "
            "[--haiku-coherence-trace PATH --haiku-coherence-output-dir PATH] "
            "[--arc-synthetic-shape-probe-candidates --failure-report PATH "
            "--public-arc-dir DIR --candidate-tsv PATH --examples PATH "
            "--summary PATH --max-rows-per-task N --max-tasks N "
            "--profile focused|broad] "
            "[--arc-synthetic-shape-curriculum --attempts-tsv PATH "
            "--output-dir DIR --synthetic-attempts PATH --combined-attempts PATH "
            "--summary PATH --max-grid-size N --ops CSV "
            "--negative-kinds CSV --op-mode cycle|all --max-accepted N "
            "--max-synthetic-rows N --max-seconds N] "
            "[--arc-append-probe-attempts-tsv --base-attempts-tsv PATH "
            "--probe-attempts-jsonl PATH --combined-attempts-tsv PATH "
            "--summary PATH --split-name NAME] "
            "[--arc-ranker-benchmark --attempts-tsv PATH --output-dir DIR "
            "--report PATH --epochs N --max-per-task N] "
            "[--arc-ranker-failure-report --attempts-tsv PATH --proposals PATH "
            "--summary PATH --output PATH --markdown PATH] ",
            argv0);
    fputs(
            "[--arc-verified-self-play-report --attempts-tsv PATH "
            "--synthetic-summary PATH --ranker-summary PATH "
            "--public-train-baseline-log PATH --public-eval-baseline-log PATH "
            "--public-train-visual-baseline-log PATH "
            "--public-eval-visual-baseline-log PATH "
            "--public-train-self-play-log PATH --public-eval-self-play-log PATH "
            "--self-play-prior PATH --self-play-prior-summary PATH "
            "--self-play-generative-prior PATH "
            "--self-play-generative-prior-summary PATH --summary PATH "
            "--markdown PATH --template-baseline N --oracle-ceiling N "
            "--seed-count N] "
            "[--arc-learned-sketch-ops --attempts-tsv PATH --trace PATH "
            "--operators PATH --prior PATH --summary PATH --markdown PATH "
            "--min-accepted N --min-tasks N --near-miss-score X "
            "--diagnostic-score X] "
            "[--paper-data --paper-data-output PATH --paper-data-typ PATH "
            "--comparison PATH --prime-gap-trace PATH --grid-path-trace PATH "
            "--delayed-claim-trace PATH --arc-agi PATH --prime-gap PATH] "
            "[--paper-claims --claims PATH --output-dir DIR --paper PATH "
            "--agent-candidates-dir DIR --support-check --fail-required "
            "--fail-unlinked-roadmap --evidence-origin-audit --fail-synthetic-required] "
            "[--trace-determinism-audit --trace-input PATH --output PATH "
            "--canonical-output PATH] "
            "[--trace-determinism-compare --trace-input PATH ... --output PATH "
            "--gate] "
            "[--neural-grammar-discovery --domains CSV --output-dir DIR "
            "--runtime PATH --arc-agi PATH --public-arc-dir DIR "
            "--arc-failure-proof PATH --arc-heldout-triage PATH "
            "--haiku-word-trace PATH --haiku-understanding-trace PATH "
            "--haiku-coherence-hard-negatives PATH --program-synthesis-trace PATH "
            "--grid-path-trace PATH --signal-mining-trace PATH] "
            "[--arc-neural-discovery --arc-agi PATH --public-arc-dir DIR "
            "--diagnostics PATH --output-dir DIR --clusters N --top-k N "
            "--verify-representatives N --mutations-per-cluster N "
            "--mutation-representatives N] "
            "[--arc-neural-failure-proof --arc-agi PATH --public-arc-dir DIR "
            "--mutation-verification PATH --candidate-sketches PATH "
            "--triage PATH --output-dir DIR --max-promotion-tasks N] "
            "[--experiment-health --coverage PATH --numeric-summary PATH "
            "--numeric-report PATH --paper-claims PATH --signal-flight-summary PATH "
            "--signal-flight-rollout PATH --signal-live-active PATH "
            "--signal-shadow-replay PATH --neural-grammar-summary PATH "
            "--protocol-spine PATH --operator-promotions PATH "
            "--engine-evidence PATH --tiny-net-diagnostics PATH "
            "--failure-atlas PATH --failure-atlas-markdown PATH "
            "--failure-work-queue PATH --failure-work-queue-markdown PATH "
            "--output PATH] "
            "[--learned-grammar-coverage --arc-log [SCOPE=]PATH ... "
            "--haiku-word-trace [SCOPE=]PATH ... "
            "--haiku-understanding-trace [SCOPE=]PATH ... "
            "--haiku-coherence-report [SCOPE=]PATH ... --budget PATH "
            "--output PATH --fail-on-regression] "
            "[--arc-visual-hole-proposer|--arc-visual-curriculum "
            "--visual-only-log [SCOPE=]PATH ... --target-scope SCOPE ... "
            "--attempt-log PATH ... --candidate-tsv PATH --examples PATH "
            "--summary PATH] "
            "[--arc-pipeline-cache PATH] [--memory-in PATH] [--memory-out PATH] [--epochs N] "
            "[--learning-rate LR] [--arc-pipeline-max-per-task N] "
            "[--nearest-signatures N] [--max-attempt-seeds N] [--max-ops-per-task N] "
            "[--max-op-hints N]\n",
            stderr);
}

int main(int argc, char **argv) {
    const char *candidate_rows = arg_value(argc, argv, "--candidate-rows");
    const char *compiled_output = arg_value(argc, argv, "--compiled-output");
    const char *verification_output = arg_value(argc, argv, "--verification-output");
    const char *sketches = arg_value(argc, argv, "--sketches");
    const char *sketch_examples = arg_value(argc, argv, "--sketch-examples");
    const char *neural_proposals_output =
        arg_value(argc, argv, "--neural-proposals-output");
    const char *arc_pipeline_attempts = arg_value(argc, argv, "--arc-pipeline-attempts");
    const char *arc_pipeline_proposals_output =
        arg_value(argc, argv, "--arc-pipeline-proposals-output");
    const char *arc_pipeline_summary_output =
        arg_value(argc, argv, "--arc-pipeline-summary-output");
    const char *arc_pipeline_cache = arg_value(argc, argv, "--arc-pipeline-cache");
    const char *arc_pipeline_dedupe_attempts_output =
        arg_value(argc, argv, "--arc-pipeline-dedupe-attempts-output");
    const char *haiku_coherence_trace = arg_value(argc, argv, "--haiku-coherence-trace");
    const char *haiku_coherence_output_dir =
        arg_value(argc, argv, "--haiku-coherence-output-dir");
    const char *arc_hole_candidate_tsv = arg_value(argc, argv, "--candidate-tsv");
    const char *arc_hole_examples = arg_value(argc, argv, "--examples");
    const char *arc_hole_summary = arg_value(argc, argv, "--summary");
    const char *arc_synthetic_failure_report =
        arg_value(argc, argv, "--failure-report");
    const char *arc_synthetic_public_arc_dir =
        arg_value(argc, argv, "--public-arc-dir");
    const char *arc_synthetic_profile = arg_value(argc, argv, "--profile");
    const char *arc_synthetic_max_rows_arg =
        arg_value(argc, argv, "--max-rows-per-task");
    const char *arc_synthetic_max_tasks_arg = arg_value(argc, argv, "--max-tasks");
    const char *arc_append_base_attempts =
        arg_value(argc, argv, "--base-attempts-tsv");
    const char *arc_append_probe_attempts =
        arg_value(argc, argv, "--probe-attempts-jsonl");
    const char *arc_append_combined_attempts =
        arg_value(argc, argv, "--combined-attempts-tsv");
    const char *arc_append_split_name = arg_value(argc, argv, "--split-name");
    const char *arc_ranker_benchmark_attempts =
        arg_value(argc, argv, "--attempts-tsv");
    const char *arc_ranker_benchmark_output_dir =
        arg_value(argc, argv, "--output-dir");
    const char *arc_synthetic_curriculum_attempts =
        arg_value(argc, argv, "--synthetic-attempts");
    const char *arc_synthetic_curriculum_combined_attempts =
        arg_value(argc, argv, "--combined-attempts");
    const char *arc_synthetic_curriculum_max_grid_size_arg =
        arg_value(argc, argv, "--max-grid-size");
    const char *arc_synthetic_curriculum_ops = arg_value(argc, argv, "--ops");
    const char *arc_synthetic_curriculum_negative_kinds =
        arg_value(argc, argv, "--negative-kinds");
    const char *arc_synthetic_curriculum_op_mode =
        arg_value(argc, argv, "--op-mode");
    const char *arc_synthetic_curriculum_max_accepted_arg =
        arg_value(argc, argv, "--max-accepted");
    const char *arc_synthetic_curriculum_max_rows_arg =
        arg_value(argc, argv, "--max-synthetic-rows");
    const char *arc_synthetic_curriculum_max_seconds_arg =
        arg_value(argc, argv, "--max-seconds");
    const char *arc_ranker_benchmark_report = arg_value(argc, argv, "--report");
    const char *arc_ranker_failure_proposals =
        arg_value(argc, argv, "--proposals");
    const char *arc_ranker_failure_output = arg_value(argc, argv, "--output");
    const char *arc_ranker_failure_markdown =
        arg_value(argc, argv, "--markdown");
    const char *arc_selfplay_synthetic_summary =
        arg_value(argc, argv, "--synthetic-summary");
    const char *arc_selfplay_ranker_summary =
        arg_value(argc, argv, "--ranker-summary");
    const char *arc_selfplay_public_train_baseline =
        arg_value(argc, argv, "--public-train-baseline-log");
    const char *arc_selfplay_public_eval_baseline =
        arg_value(argc, argv, "--public-eval-baseline-log");
    const char *arc_selfplay_public_train_visual_baseline =
        arg_value(argc, argv, "--public-train-visual-baseline-log");
    const char *arc_selfplay_public_eval_visual_baseline =
        arg_value(argc, argv, "--public-eval-visual-baseline-log");
    const char *arc_selfplay_public_train_log =
        arg_value(argc, argv, "--public-train-self-play-log");
    const char *arc_selfplay_public_eval_log =
        arg_value(argc, argv, "--public-eval-self-play-log");
    const char *arc_selfplay_prior = arg_value(argc, argv, "--self-play-prior");
    const char *arc_selfplay_prior_summary =
        arg_value(argc, argv, "--self-play-prior-summary");
    const char *arc_selfplay_generative_prior =
        arg_value(argc, argv, "--self-play-generative-prior");
    const char *arc_selfplay_generative_prior_summary =
        arg_value(argc, argv, "--self-play-generative-prior-summary");
    const char *arc_selfplay_template_baseline_arg =
        arg_value(argc, argv, "--template-baseline");
    const char *arc_selfplay_oracle_ceiling_arg =
        arg_value(argc, argv, "--oracle-ceiling");
    const char *arc_selfplay_seed_count_arg =
        arg_value(argc, argv, "--seed-count");
    const char *arc_learned_sketch_ops_trace = arg_value(argc, argv, "--trace");
    const char *arc_learned_sketch_ops_operators =
        arg_value(argc, argv, "--operators");
    const char *arc_learned_sketch_ops_prior = arg_value(argc, argv, "--prior");
    const char *arc_learned_sketch_ops_min_accepted_arg =
        arg_value(argc, argv, "--min-accepted");
    const char *arc_learned_sketch_ops_min_tasks_arg =
        arg_value(argc, argv, "--min-tasks");
    const char *arc_learned_sketch_ops_near_miss_score_arg =
        arg_value(argc, argv, "--near-miss-score");
    const char *arc_learned_sketch_ops_diagnostic_score_arg =
        arg_value(argc, argv, "--diagnostic-score");
    const char *paper_data_output = arg_value(argc, argv, "--paper-data-output");
    const char *paper_data_typ = arg_value(argc, argv, "--paper-data-typ");
    const char *paper_comparison = arg_value(argc, argv, "--comparison");
    const char *paper_prime_gap_trace = arg_value(argc, argv, "--prime-gap-trace");
    const char *paper_grid_path_trace = arg_value(argc, argv, "--grid-path-trace");
    const char *paper_delayed_claim_trace =
        arg_value(argc, argv, "--delayed-claim-trace");
    const char *paper_arc_agi = arg_value(argc, argv, "--arc-agi");
    const char *paper_prime_gap = arg_value(argc, argv, "--prime-gap");
    const char *paper_claims_path = arg_value(argc, argv, "--claims");
    const char *paper_claims_output_dir = arg_value(argc, argv, "--output-dir");
    const char *paper_claims_paper_path = arg_value(argc, argv, "--paper");
    const char *paper_agent_candidates_dir =
        arg_value(argc, argv, "--agent-candidates-dir");
    const char *trace_determinism_output = arg_value(argc, argv, "--output");
    const char *trace_determinism_canonical =
        arg_value(argc, argv, "--canonical-output");
    const char *arc_neural_diagnostics =
        arg_value(argc, argv, "--diagnostics");
    const char *arc_neural_clusters_arg =
        arg_value(argc, argv, "--clusters");
    const char *arc_neural_top_k_arg =
        arg_value(argc, argv, "--top-k");
    const char *arc_neural_verify_representatives_arg =
        arg_value(argc, argv, "--verify-representatives");
    const char *arc_neural_mutations_per_cluster_arg =
        arg_value(argc, argv, "--mutations-per-cluster");
    const char *arc_neural_mutation_representatives_arg =
        arg_value(argc, argv, "--mutation-representatives");
    const char *arc_neural_mutation_verification =
        arg_value(argc, argv, "--mutation-verification");
    const char *arc_neural_candidate_sketches =
        arg_value(argc, argv, "--candidate-sketches");
    const char *arc_neural_triage = arg_value(argc, argv, "--triage");
    const char *arc_neural_max_promotion_tasks_arg =
        arg_value(argc, argv, "--max-promotion-tasks");
    const char *experiment_health_coverage = arg_value(argc, argv, "--coverage");
    const char *experiment_health_numeric_summary =
        arg_value(argc, argv, "--numeric-summary");
    const char *experiment_health_numeric_report =
        arg_value(argc, argv, "--numeric-report");
    const char *experiment_health_paper_claims =
        arg_value(argc, argv, "--paper-claims");
    const char *experiment_health_signal_flight_summary =
        arg_value(argc, argv, "--signal-flight-summary");
    const char *experiment_health_signal_flight_rollout =
        arg_value(argc, argv, "--signal-flight-rollout");
    const char *experiment_health_signal_live_active =
        arg_value(argc, argv, "--signal-live-active");
    const char *experiment_health_signal_shadow_replay =
        arg_value(argc, argv, "--signal-shadow-replay");
    const char *experiment_health_neural_grammar_summary =
        arg_value(argc, argv, "--neural-grammar-summary");
    const char *experiment_health_protocol_spine =
        arg_value(argc, argv, "--protocol-spine");
    const char *experiment_health_operator_promotions =
        arg_value(argc, argv, "--operator-promotions");
    const char *experiment_health_engine_evidence =
        arg_value(argc, argv, "--engine-evidence");
    const char *experiment_health_tiny_net_diagnostics =
        arg_value(argc, argv, "--tiny-net-diagnostics");
    const char *experiment_health_failure_atlas =
        arg_value(argc, argv, "--failure-atlas");
    const char *experiment_health_failure_atlas_markdown =
        arg_value(argc, argv, "--failure-atlas-markdown");
    const char *experiment_health_failure_work_queue =
        arg_value(argc, argv, "--failure-work-queue");
    const char *experiment_health_failure_work_queue_markdown =
        arg_value(argc, argv, "--failure-work-queue-markdown");
    const char *experiment_health_output = arg_value(argc, argv, "--output");
    const char *ngd_domains = arg_value(argc, argv, "--domains");
    const char *ngd_runtime_path = arg_value(argc, argv, "--runtime");
    const char *ngd_arc_failure_proof =
        arg_value(argc, argv, "--arc-failure-proof");
    const char *ngd_arc_heldout_triage =
        arg_value(argc, argv, "--arc-heldout-triage");
    const char *ngd_arc_probe_compare_limit_arg =
        arg_value(argc, argv, "--arc-probe-compare-limit");
    const char *ngd_arc_heldout_limit_arg =
        arg_value(argc, argv, "--arc-heldout-limit");
    const char *ngd_arc_heldout_per_family_arg =
        arg_value(argc, argv, "--arc-heldout-per-family");
    const char *ngd_arc_heldout_compare_limit_arg =
        arg_value(argc, argv, "--arc-heldout-compare-limit");
    const char *ngd_arc_heldout_op_hints_per_task_arg =
        arg_value(argc, argv, "--arc-heldout-op-hints-per-task");
    const char *ngd_haiku_word_trace =
        arg_value(argc, argv, "--haiku-word-trace");
    const char *ngd_haiku_understanding_trace =
        arg_value(argc, argv, "--haiku-understanding-trace");
    const char *ngd_haiku_coherence =
        arg_value(argc, argv, "--haiku-coherence-hard-negatives");
    const char *ngd_program_synthesis_trace =
        arg_value(argc, argv, "--program-synthesis-trace");
    const char *ngd_grid_path_trace = arg_value(argc, argv, "--grid-path-trace");
    const char *ngd_signal_mining_trace =
        arg_value(argc, argv, "--signal-mining-trace");
    const char *arc_ranker_benchmark_max_per_task_arg =
        arg_value(argc, argv, "--max-per-task");
    size_t arc_pipeline_dedupe_attempts_input_count =
        count_arg_values(argc, argv, "--arc-pipeline-dedupe-attempts-input");
    size_t arc_pipeline_source_attempts_jsonl_count =
        count_arg_values(argc, argv, "--arc-pipeline-source-attempts-jsonl");
    size_t arc_pipeline_target_attempts_jsonl_count =
        count_arg_values(argc, argv, "--arc-pipeline-target-attempts-jsonl");
    size_t arc_hole_visual_log_count = count_arg_values(argc, argv, "--visual-only-log");
    size_t arc_hole_target_scope_count = count_arg_values(argc, argv, "--target-scope");
    size_t arc_hole_attempt_log_count = count_arg_values(argc, argv, "--attempt-log");
    size_t coverage_arc_log_count = count_arg_values(argc, argv, "--arc-log");
    size_t coverage_haiku_word_count =
        count_arg_values(argc, argv, "--haiku-word-trace");
    size_t coverage_haiku_understanding_count =
        count_arg_values(argc, argv, "--haiku-understanding-trace");
    size_t coverage_haiku_coherence_count =
        count_arg_values(argc, argv, "--haiku-coherence-report");
    size_t trace_determinism_input_count =
        count_arg_values(argc, argv, "--trace-input");
    const char **arc_pipeline_dedupe_attempts_inputs = NULL;
    const char **arc_pipeline_source_attempts_jsonl = NULL;
    const char **arc_pipeline_target_attempts_jsonl = NULL;
    const char **arc_hole_visual_logs = NULL;
    const char **arc_hole_target_scopes = NULL;
    const char **arc_hole_attempt_logs = NULL;
    const char **coverage_arc_logs = NULL;
    const char **coverage_haiku_word_traces = NULL;
    const char **coverage_haiku_understanding_traces = NULL;
    const char **coverage_haiku_coherence_reports = NULL;
    const char **trace_determinism_inputs = NULL;
    const char *arc_pipeline_max_per_task_arg =
        arg_value(argc, argv, "--arc-pipeline-max-per-task");
    const char *arc_hole_nearest_signatures_arg =
        arg_value(argc, argv, "--nearest-signatures");
    const char *arc_hole_max_attempt_seeds_arg =
        arg_value(argc, argv, "--max-attempt-seeds");
    const char *arc_hole_max_ops_per_task_arg =
        arg_value(argc, argv, "--max-ops-per-task");
    const char *arc_curriculum_max_op_hints_arg =
        arg_value(argc, argv, "--max-op-hints");
    const char *memory_in = arg_value(argc, argv, "--memory-in");
    const char *memory_out = arg_value(argc, argv, "--memory-out");
    const char *epochs_arg = arg_value(argc, argv, "--epochs");
    const char *learning_rate_arg = arg_value(argc, argv, "--learning-rate");
    unsigned epochs = epochs_arg == NULL ? 64U : (unsigned)strtoul(epochs_arg, NULL, 10);
    double learning_rate = learning_rate_arg == NULL ? 0.08 : strtod(learning_rate_arg, NULL);
    unsigned arc_pipeline_max_per_task = arc_pipeline_max_per_task_arg == NULL
                                             ? 8U
                                             : (unsigned)strtoul(arc_pipeline_max_per_task_arg,
                                                                 NULL,
                                                                 10);
    unsigned arc_hole_nearest_signatures = arc_hole_nearest_signatures_arg == NULL
                                               ? 1U
                                               : (unsigned)strtoul(
                                                     arc_hole_nearest_signatures_arg,
                                                     NULL,
                                                     10);
    unsigned arc_hole_max_attempt_seeds = arc_hole_max_attempt_seeds_arg == NULL
                                              ? 2U
                                              : (unsigned)strtoul(
                                                    arc_hole_max_attempt_seeds_arg,
                                                    NULL,
                                                    10);
    unsigned arc_hole_max_ops_per_task = arc_hole_max_ops_per_task_arg == NULL
                                             ? 2U
                                             : (unsigned)strtoul(
                                                   arc_hole_max_ops_per_task_arg,
                                                   NULL,
                                                   10);
    unsigned arc_curriculum_max_op_hints =
        arc_curriculum_max_op_hints_arg == NULL
            ? 8U
            : (unsigned)strtoul(arc_curriculum_max_op_hints_arg, NULL, 10);
    unsigned arc_synthetic_max_rows =
        arc_synthetic_max_rows_arg == NULL
            ? 6U
            : (unsigned)strtoul(arc_synthetic_max_rows_arg, NULL, 10);
    unsigned arc_synthetic_max_tasks =
        arc_synthetic_max_tasks_arg == NULL
            ? 0U
            : (unsigned)strtoul(arc_synthetic_max_tasks_arg, NULL, 10);
    unsigned arc_ranker_benchmark_max_per_task =
        arc_ranker_benchmark_max_per_task_arg == NULL
            ? arc_pipeline_max_per_task
            : (unsigned)strtoul(arc_ranker_benchmark_max_per_task_arg, NULL, 10);
    unsigned arc_synthetic_curriculum_max_grid_size =
        arc_synthetic_curriculum_max_grid_size_arg == NULL
            ? 30U
            : (unsigned)strtoul(arc_synthetic_curriculum_max_grid_size_arg,
                                NULL,
                                10);
    unsigned arc_synthetic_curriculum_max_accepted =
        arc_synthetic_curriculum_max_accepted_arg == NULL
            ? 0U
            : (unsigned)strtoul(arc_synthetic_curriculum_max_accepted_arg,
                                NULL,
                                10);
    unsigned arc_synthetic_curriculum_max_rows =
        arc_synthetic_curriculum_max_rows_arg == NULL
            ? 0U
            : (unsigned)strtoul(arc_synthetic_curriculum_max_rows_arg, NULL, 10);
    unsigned arc_synthetic_curriculum_max_seconds =
        arc_synthetic_curriculum_max_seconds_arg == NULL
            ? 0U
            : (unsigned)strtoul(arc_synthetic_curriculum_max_seconds_arg, NULL, 10);
    long arc_selfplay_template_baseline =
        arc_selfplay_template_baseline_arg == NULL
            ? 7L
            : strtol(arc_selfplay_template_baseline_arg, NULL, 10);
    long arc_selfplay_oracle_ceiling =
        arc_selfplay_oracle_ceiling_arg == NULL
            ? 14L
            : strtol(arc_selfplay_oracle_ceiling_arg, NULL, 10);
    long arc_selfplay_seed_count =
        arc_selfplay_seed_count_arg == NULL
            ? 1L
            : strtol(arc_selfplay_seed_count_arg, NULL, 10);
    long arc_learned_sketch_ops_min_accepted =
        arc_learned_sketch_ops_min_accepted_arg == NULL
            ? 1L
            : strtol(arc_learned_sketch_ops_min_accepted_arg, NULL, 10);
    long arc_learned_sketch_ops_min_tasks =
        arc_learned_sketch_ops_min_tasks_arg == NULL
            ? 1L
            : strtol(arc_learned_sketch_ops_min_tasks_arg, NULL, 10);
    double arc_learned_sketch_ops_near_miss_score =
        arc_learned_sketch_ops_near_miss_score_arg == NULL
            ? 0.75
            : strtod(arc_learned_sketch_ops_near_miss_score_arg, NULL);
    double arc_learned_sketch_ops_diagnostic_score =
        arc_learned_sketch_ops_diagnostic_score_arg == NULL
            ? 0.90
            : strtod(arc_learned_sketch_ops_diagnostic_score_arg, NULL);
    int arc_visual_hole_requested = has_arg(argc, argv, "--arc-visual-hole-proposer");
    int arc_visual_curriculum_requested =
        has_arg(argc, argv, "--arc-visual-curriculum");
    int arc_synthetic_requested =
        has_arg(argc, argv, "--arc-synthetic-shape-probe-candidates");
    int arc_synthetic_curriculum_requested =
        has_arg(argc, argv, "--arc-synthetic-shape-curriculum");
    int arc_append_requested =
        has_arg(argc, argv, "--arc-append-probe-attempts-tsv");
    int arc_ranker_benchmark_requested =
        has_arg(argc, argv, "--arc-ranker-benchmark");
    int arc_ranker_failure_report_requested =
        has_arg(argc, argv, "--arc-ranker-failure-report");
    int arc_verified_selfplay_requested =
        has_arg(argc, argv, "--arc-verified-self-play-report");
    int arc_learned_sketch_ops_requested =
        has_arg(argc, argv, "--arc-learned-sketch-ops");
    int paper_data_requested = has_arg(argc, argv, "--paper-data");
    int paper_claims_requested = has_arg(argc, argv, "--paper-claims");
    int trace_determinism_requested =
        has_arg(argc, argv, "--trace-determinism-audit") ||
        has_arg(argc, argv, "--trace-determinism-compare");
    int neural_grammar_discovery_requested =
        has_arg(argc, argv, "--neural-grammar-discovery");
    int arc_neural_discovery_requested =
        has_arg(argc, argv, "--arc-neural-discovery");
    int arc_neural_failure_proof_requested =
        has_arg(argc, argv, "--arc-neural-failure-proof");
    int experiment_health_requested = has_arg(argc, argv, "--experiment-health");
    int learned_grammar_coverage_requested =
        has_arg(argc, argv, "--learned-grammar-coverage");

    if (has_arg(argc, argv, "--help")) {
        usage(argv[0]);
        return 0;
    }
    if (trace_determinism_requested) {
        trace_determinism_inputs = collect_arg_values_alloc(argc,
                                                            argv,
                                                            "--trace-input",
                                                            trace_determinism_input_count);
        if (trace_determinism_input_count == 0U ||
            trace_determinism_inputs == NULL ||
            trace_determinism_output == NULL) {
            usage(argv[0]);
            free(trace_determinism_inputs);
            return 2;
        }
        if (has_arg(argc, argv, "--trace-determinism-audit") &&
            trace_determinism_input_count != 1U) {
            usage(argv[0]);
            free(trace_determinism_inputs);
            return 2;
        }
        if (nn_paper_write_trace_determinism(
                trace_determinism_inputs,
                trace_determinism_input_count,
                trace_determinism_output,
                trace_determinism_canonical,
                has_arg(argc, argv, "--gate")) != 0) {
            fprintf(stderr, "failed to write trace determinism report\n");
            free(trace_determinism_inputs);
            return 1;
        }
        free(trace_determinism_inputs);
        return 0;
    }
    if (experiment_health_requested) {
        if (experiment_health_coverage == NULL || experiment_health_output == NULL) {
            usage(argv[0]);
            return 2;
        }
        if (nn_grammar_discovery_write_experiment_health(
                experiment_health_coverage,
                experiment_health_numeric_summary,
                experiment_health_numeric_report,
                experiment_health_paper_claims,
                experiment_health_signal_flight_summary,
                experiment_health_signal_flight_rollout,
                experiment_health_signal_live_active,
                experiment_health_signal_shadow_replay,
                experiment_health_neural_grammar_summary,
                experiment_health_protocol_spine,
                experiment_health_operator_promotions,
                experiment_health_engine_evidence,
                experiment_health_tiny_net_diagnostics,
                experiment_health_failure_atlas,
                experiment_health_failure_atlas_markdown,
                experiment_health_failure_work_queue,
                experiment_health_failure_work_queue_markdown,
                experiment_health_output) != 0) {
            fprintf(stderr, "failed to write experiment health report\n");
            return 1;
        }
        return 0;
    }
    if (learned_grammar_coverage_requested) {
        coverage_arc_logs = collect_arg_values_alloc(argc,
                                                     argv,
                                                     "--arc-log",
                                                     coverage_arc_log_count);
        coverage_haiku_word_traces =
            collect_arg_values_alloc(argc,
                                     argv,
                                     "--haiku-word-trace",
                                     coverage_haiku_word_count);
        coverage_haiku_understanding_traces =
            collect_arg_values_alloc(argc,
                                     argv,
                                     "--haiku-understanding-trace",
                                     coverage_haiku_understanding_count);
        coverage_haiku_coherence_reports =
            collect_arg_values_alloc(argc,
                                     argv,
                                     "--haiku-coherence-report",
                                     coverage_haiku_coherence_count);
        if (experiment_health_output == NULL ||
            (coverage_arc_log_count > 0U && coverage_arc_logs == NULL) ||
            (coverage_haiku_word_count > 0U &&
             coverage_haiku_word_traces == NULL) ||
            (coverage_haiku_understanding_count > 0U &&
             coverage_haiku_understanding_traces == NULL) ||
            (coverage_haiku_coherence_count > 0U &&
             coverage_haiku_coherence_reports == NULL)) {
            usage(argv[0]);
            free(coverage_arc_logs);
            free(coverage_haiku_word_traces);
            free(coverage_haiku_understanding_traces);
            free(coverage_haiku_coherence_reports);
            return 2;
        }
        if (nn_grammar_discovery_write_learned_grammar_coverage(
                coverage_arc_logs,
                coverage_arc_log_count,
                coverage_haiku_word_traces,
                coverage_haiku_word_count,
                coverage_haiku_understanding_traces,
                coverage_haiku_understanding_count,
                coverage_haiku_coherence_reports,
                coverage_haiku_coherence_count,
                arg_value(argc, argv, "--budget"),
                experiment_health_output,
                has_arg(argc, argv, "--fail-on-regression")) != 0) {
            fprintf(stderr, "failed to write learned grammar coverage report\n");
            free(coverage_arc_logs);
            free(coverage_haiku_word_traces);
            free(coverage_haiku_understanding_traces);
            free(coverage_haiku_coherence_reports);
            return 1;
        }
        free(coverage_arc_logs);
        free(coverage_haiku_word_traces);
        free(coverage_haiku_understanding_traces);
        free(coverage_haiku_coherence_reports);
        return 0;
    }
    if (arc_neural_discovery_requested) {
        unsigned clusters = arc_neural_clusters_arg == NULL
                                ? 12U
                                : (unsigned)strtoul(arc_neural_clusters_arg,
                                                    NULL,
                                                    10);
        unsigned top_k = arc_neural_top_k_arg == NULL
                             ? 20U
                             : (unsigned)strtoul(arc_neural_top_k_arg, NULL, 10);
        unsigned verify_representatives =
            arc_neural_verify_representatives_arg == NULL
                ? 0U
                : (unsigned)strtoul(arc_neural_verify_representatives_arg,
                                    NULL,
                                    10);
        unsigned mutations_per_cluster =
            arc_neural_mutations_per_cluster_arg == NULL
                ? 2U
                : (unsigned)strtoul(arc_neural_mutations_per_cluster_arg,
                                    NULL,
                                    10);
        unsigned mutation_representatives =
            arc_neural_mutation_representatives_arg == NULL
                ? 1U
                : (unsigned)strtoul(arc_neural_mutation_representatives_arg,
                                    NULL,
                                    10);
        if (nn_grammar_discovery_write_arc_neural_discovery(
                paper_arc_agi == NULL ? "build/float/arc_agi" : paper_arc_agi,
                arc_synthetic_public_arc_dir == NULL
                    ? ""
                    : arc_synthetic_public_arc_dir,
                arc_neural_diagnostics == NULL
                    ? "build/float/hypothesis_library_public_filter/diagnostics.jsonl"
                    : arc_neural_diagnostics,
                arc_ranker_benchmark_output_dir == NULL
                    ? "build/float/hypothesis_library_neural_discovery"
                    : arc_ranker_benchmark_output_dir,
                clusters,
                top_k,
                verify_representatives,
                mutations_per_cluster,
                mutation_representatives) != 0) {
            fprintf(stderr, "failed to write ARC neural discovery artifacts\n");
            return 1;
        }
        return 0;
    }
    if (arc_neural_failure_proof_requested) {
        unsigned max_promotion_tasks =
            arc_neural_max_promotion_tasks_arg == NULL
                ? 0U
                : (unsigned)strtoul(arc_neural_max_promotion_tasks_arg,
                                    NULL,
                                    10);
        if (nn_grammar_discovery_write_arc_neural_failure_proof(
                paper_arc_agi == NULL ? "build/float/arc_agi" : paper_arc_agi,
                arc_synthetic_public_arc_dir == NULL
                    ? ""
                    : arc_synthetic_public_arc_dir,
                arc_neural_mutation_verification == NULL
                    ? "build/float/hypothesis_library_neural_discovery/mutation_verification.jsonl"
                    : arc_neural_mutation_verification,
                arc_neural_candidate_sketches == NULL
                    ? "build/float/hypothesis_library_neural_discovery/candidate_sketches.jsonl"
                    : arc_neural_candidate_sketches,
                arc_neural_triage == NULL
                    ? "build/float/hypothesis_library_public_triage/coverage_triage.jsonl"
                    : arc_neural_triage,
                arc_ranker_benchmark_output_dir == NULL
                    ? "build/float/hypothesis_library_neural_discovery/failure_proof"
                    : arc_ranker_benchmark_output_dir,
                max_promotion_tasks) != 0) {
            fprintf(stderr, "failed to write ARC neural failure proof artifacts\n");
            return 1;
        }
        return 0;
    }
    if (neural_grammar_discovery_requested) {
        unsigned ngd_arc_probe_compare_limit =
            ngd_arc_probe_compare_limit_arg == NULL
                ? 64U
                : (unsigned)strtoul(ngd_arc_probe_compare_limit_arg, NULL, 10);
        unsigned ngd_arc_heldout_limit =
            ngd_arc_heldout_limit_arg == NULL
                ? 64U
                : (unsigned)strtoul(ngd_arc_heldout_limit_arg, NULL, 10);
        unsigned ngd_arc_heldout_per_family =
            ngd_arc_heldout_per_family_arg == NULL
                ? 12U
                : (unsigned)strtoul(ngd_arc_heldout_per_family_arg, NULL, 10);
        unsigned ngd_arc_heldout_compare_limit =
            ngd_arc_heldout_compare_limit_arg == NULL
                ? 64U
                : (unsigned)strtoul(ngd_arc_heldout_compare_limit_arg, NULL, 10);
        unsigned ngd_arc_heldout_op_hints_per_task =
            ngd_arc_heldout_op_hints_per_task_arg == NULL
                ? 3U
                : (unsigned)strtoul(ngd_arc_heldout_op_hints_per_task_arg,
                                    NULL,
                                    10);
        if (nn_grammar_discovery_write_neural_grammar_discovery_report(
                ngd_domains == NULL
                    ? "arc,haiku,program_synthesis,grid_path,signal_mining"
                    : ngd_domains,
                arc_ranker_benchmark_output_dir == NULL
                    ? "build/float/neural_grammar_discovery"
                    : arc_ranker_benchmark_output_dir,
                ngd_runtime_path == NULL ? argv[0] : ngd_runtime_path,
                paper_arc_agi == NULL ? "build/float/arc_agi" : paper_arc_agi,
                arc_synthetic_public_arc_dir == NULL
                    ? ""
                    : arc_synthetic_public_arc_dir,
                ngd_arc_failure_proof == NULL
                    ? "build/float/hypothesis_library_neural_discovery/failure_proof/task_proof_rows.jsonl"
                    : ngd_arc_failure_proof,
                ngd_arc_heldout_triage == NULL
                    ? "build/float/hypothesis_library_public_triage/coverage_triage.jsonl"
                    : ngd_arc_heldout_triage,
                ngd_haiku_word_trace == NULL
                    ? "build/float/haiku_word_lm_artifacts/smoke_trace.jsonl"
                    : ngd_haiku_word_trace,
                ngd_haiku_understanding_trace == NULL
                    ? "build/float/haiku_understanding_artifacts/smoke_trace.jsonl"
                    : ngd_haiku_understanding_trace,
                ngd_haiku_coherence == NULL
                    ? "build/float/haiku_coherence_hard_negatives/hard_negative_rows.jsonl"
                    : ngd_haiku_coherence,
                ngd_program_synthesis_trace == NULL
                    ? "build/float/neural_grammar_discovery/program_synthesis_source.jsonl"
                    : ngd_program_synthesis_trace,
                ngd_grid_path_trace == NULL ? "build/float/grid_path.jsonl"
                                             : ngd_grid_path_trace,
                ngd_signal_mining_trace == NULL
                    ? "build/float/signal_mining_smoke/trace.jsonl"
                    : ngd_signal_mining_trace,
                ngd_arc_probe_compare_limit,
                ngd_arc_heldout_limit,
                ngd_arc_heldout_per_family,
                ngd_arc_heldout_compare_limit,
                ngd_arc_heldout_op_hints_per_task) != 0) {
            fprintf(stderr, "failed to write neural grammar discovery report\n");
            return 1;
        }
        return 0;
    }
    if (paper_data_requested) {
        if (paper_data_output == NULL || paper_data_typ == NULL ||
            paper_comparison == NULL || paper_prime_gap_trace == NULL ||
            paper_grid_path_trace == NULL || paper_delayed_claim_trace == NULL ||
            paper_arc_agi == NULL || paper_prime_gap == NULL) {
            usage(argv[0]);
            return 2;
        }
        if (nn_paper_write_data(paper_data_output,
                                paper_data_typ,
                                paper_comparison,
                                paper_prime_gap_trace,
                                paper_grid_path_trace,
                                paper_delayed_claim_trace,
                                paper_arc_agi,
                                paper_prime_gap) != 0) {
            fprintf(stderr, "failed to write paper data artifacts\n");
            return 1;
        }
        return 0;
    }
    if (paper_claims_requested) {
        if (nn_paper_write_claims(paper_claims_path == NULL ? "paper/claims.yaml"
                                                            : paper_claims_path,
                                  paper_claims_output_dir == NULL ? "paper/generated"
                                                                  : paper_claims_output_dir,
                                  paper_claims_paper_path == NULL ? "paper/paper.typ"
                                                                  : paper_claims_paper_path,
                                  paper_agent_candidates_dir == NULL
                                      ? "paper/agent_candidates"
                                      : paper_agent_candidates_dir,
                                  has_arg(argc, argv, "--support-check"),
                                  has_arg(argc, argv, "--fail-required"),
                                  has_arg(argc, argv, "--fail-unlinked-roadmap")) != 0) {
            fprintf(stderr, "failed to write paper claim artifacts\n");
            return 1;
        }
        if (has_arg(argc, argv, "--evidence-origin-audit")) {
            int fail_synthetic = has_arg(argc, argv, "--fail-synthetic-required");
            if (nn_paper_write_evidence_origin_audit(
                    paper_claims_output_dir == NULL ? "paper/generated"
                                                    : paper_claims_output_dir,
                    fail_synthetic) != 0) {
                fprintf(stderr, "evidence origin audit: synthetic-only required claims detected\n");
                if (fail_synthetic) {
                    return 1;
                }
            }
        }
        return 0;
    }
    if (arc_verified_selfplay_requested) {
        if (arc_ranker_benchmark_attempts == NULL ||
            arc_selfplay_prior == NULL ||
            arc_selfplay_prior_summary == NULL ||
            arc_selfplay_generative_prior == NULL ||
            arc_selfplay_generative_prior_summary == NULL ||
            arc_hole_summary == NULL ||
            arc_ranker_failure_markdown == NULL) {
            usage(argv[0]);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_verified_self_play_report(
                arc_ranker_benchmark_attempts,
                arc_selfplay_synthetic_summary,
                arc_selfplay_ranker_summary,
                arc_selfplay_public_train_baseline,
                arc_selfplay_public_eval_baseline,
                arc_selfplay_public_train_visual_baseline,
                arc_selfplay_public_eval_visual_baseline,
                arc_selfplay_public_train_log,
                arc_selfplay_public_eval_log,
                arc_selfplay_prior,
                arc_selfplay_prior_summary,
                arc_selfplay_generative_prior,
                arc_selfplay_generative_prior_summary,
                arc_hole_summary,
                arc_ranker_failure_markdown,
                arc_selfplay_template_baseline,
                arc_selfplay_oracle_ceiling,
                arc_selfplay_seed_count) != 0) {
            fprintf(stderr, "failed to write ARC verified self-play report\n");
            return 1;
        }
        return 0;
    }
    if (arc_learned_sketch_ops_requested) {
        if (arc_ranker_benchmark_attempts == NULL ||
            arc_learned_sketch_ops_operators == NULL ||
            arc_learned_sketch_ops_prior == NULL ||
            arc_hole_summary == NULL ||
            arc_ranker_failure_markdown == NULL) {
            usage(argv[0]);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_learned_sketch_ops(
                arc_ranker_benchmark_attempts,
                arc_learned_sketch_ops_trace,
                arc_learned_sketch_ops_operators,
                arc_learned_sketch_ops_prior,
                arc_hole_summary,
                arc_ranker_failure_markdown,
                arc_learned_sketch_ops_min_accepted,
                arc_learned_sketch_ops_min_tasks,
                arc_learned_sketch_ops_near_miss_score,
                arc_learned_sketch_ops_diagnostic_score) != 0) {
            fprintf(stderr, "failed to write ARC learned sketch-op report\n");
            return 1;
        }
        return 0;
    }
    arc_pipeline_dedupe_attempts_inputs = collect_arg_values_alloc(
        argc,
        argv,
        "--arc-pipeline-dedupe-attempts-input",
        arc_pipeline_dedupe_attempts_input_count);
    arc_pipeline_source_attempts_jsonl = collect_arg_values_alloc(
        argc,
        argv,
        "--arc-pipeline-source-attempts-jsonl",
        arc_pipeline_source_attempts_jsonl_count);
    arc_pipeline_target_attempts_jsonl = collect_arg_values_alloc(
        argc,
        argv,
        "--arc-pipeline-target-attempts-jsonl",
        arc_pipeline_target_attempts_jsonl_count);
    arc_hole_visual_logs = collect_arg_values_alloc(
        argc,
        argv,
        "--visual-only-log",
        arc_hole_visual_log_count);
    arc_hole_target_scopes = collect_arg_values_alloc(
        argc,
        argv,
        "--target-scope",
        arc_hole_target_scope_count);
    arc_hole_attempt_logs = collect_arg_values_alloc(
        argc,
        argv,
        "--attempt-log",
        arc_hole_attempt_log_count);
    if ((arc_pipeline_dedupe_attempts_input_count > 0U &&
         arc_pipeline_dedupe_attempts_inputs == NULL) ||
        (arc_pipeline_source_attempts_jsonl_count > 0U &&
         arc_pipeline_source_attempts_jsonl == NULL) ||
        (arc_pipeline_target_attempts_jsonl_count > 0U &&
         arc_pipeline_target_attempts_jsonl == NULL) ||
        (arc_hole_visual_log_count > 0U && arc_hole_visual_logs == NULL) ||
        (arc_hole_target_scope_count > 0U && arc_hole_target_scopes == NULL) ||
        (arc_hole_attempt_log_count > 0U && arc_hole_attempt_logs == NULL)) {
        free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                            arc_pipeline_source_attempts_jsonl,
                            arc_pipeline_target_attempts_jsonl,
                            arc_hole_visual_logs,
                            arc_hole_target_scopes,
                            arc_hole_attempt_logs);
        return 1;
    }
    if (candidate_rows != NULL || compiled_output != NULL || verification_output != NULL) {
        if (candidate_rows == NULL || compiled_output == NULL || verification_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_delta_outputs(candidate_rows,
                                                     compiled_output,
                                                     verification_output) != 0) {
            fprintf(stderr, "failed to compile or verify neural grammar delta rows\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (sketches != NULL || sketch_examples != NULL || neural_proposals_output != NULL) {
        if (sketches == NULL || sketch_examples == NULL ||
            neural_proposals_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_neural_sketch_proposals(sketches,
                                                               sketch_examples,
                                                               neural_proposals_output,
                                                               memory_in,
                                                               memory_out,
                                                               epochs,
                                                               learning_rate) != 0) {
            fprintf(stderr, "failed to train neural sketch proposer\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (haiku_coherence_trace != NULL || haiku_coherence_output_dir != NULL) {
        if (haiku_coherence_trace == NULL || haiku_coherence_output_dir == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_haiku_coherence_hard_negatives(
                haiku_coherence_trace,
                haiku_coherence_output_dir) != 0) {
            fprintf(stderr, "failed to write haiku coherence hard negatives\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_synthetic_requested) {
        if (arc_synthetic_failure_report == NULL ||
            arc_synthetic_public_arc_dir == NULL ||
            arc_hole_candidate_tsv == NULL ||
            arc_hole_examples == NULL ||
            arc_hole_summary == NULL ||
            arc_synthetic_max_rows == 0U) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_synthetic_shape_probe_candidates(
                arc_synthetic_failure_report,
                arc_synthetic_public_arc_dir,
                arc_hole_candidate_tsv,
                arc_hole_examples,
                arc_hole_summary,
                arc_synthetic_max_rows,
                arc_synthetic_max_tasks,
                arc_synthetic_profile == NULL ? "focused" : arc_synthetic_profile,
                has_arg(argc, argv, "--include-solved")) != 0) {
            fprintf(stderr, "failed to write ARC synthetic shape-probe candidates\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_synthetic_curriculum_requested) {
        if (arc_ranker_benchmark_attempts == NULL ||
            arc_ranker_benchmark_output_dir == NULL ||
            arc_synthetic_curriculum_attempts == NULL ||
            arc_synthetic_curriculum_combined_attempts == NULL ||
            arc_hole_summary == NULL ||
            arc_synthetic_curriculum_max_grid_size == 0U) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_synthetic_shape_curriculum(
                arc_ranker_benchmark_attempts,
                arc_ranker_benchmark_output_dir,
                arc_synthetic_curriculum_attempts,
                arc_synthetic_curriculum_combined_attempts,
                arc_hole_summary,
                arc_synthetic_curriculum_max_grid_size,
                arc_synthetic_curriculum_ops == NULL
                    ? "object_motion,recolor,tile_scale,fill_or_complete,symmetry,masked_or_normalized_crop,crop,recolor_crop,macrocell_substitute,component_color_summary,visual_combinator,separator_panel_map"
                    : arc_synthetic_curriculum_ops,
                arc_synthetic_curriculum_negative_kinds == NULL
                    ? "unsupported_shape,no_train_match"
                    : arc_synthetic_curriculum_negative_kinds,
                arc_synthetic_curriculum_op_mode == NULL
                    ? "cycle"
                    : arc_synthetic_curriculum_op_mode,
                arc_synthetic_curriculum_max_accepted,
                arc_synthetic_curriculum_max_rows,
                arc_synthetic_curriculum_max_seconds) != 0) {
            fprintf(stderr, "failed to write ARC synthetic shape curriculum\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_append_requested) {
        if (arc_append_base_attempts == NULL ||
            arc_append_probe_attempts == NULL ||
            arc_append_combined_attempts == NULL ||
            arc_hole_summary == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_append_arc_probe_attempts_tsv(
                arc_append_base_attempts,
                arc_append_probe_attempts,
                arc_append_combined_attempts,
                arc_hole_summary,
                arc_append_split_name == NULL ? "target" : arc_append_split_name,
                has_arg(argc, argv, "--accepted-only"),
                !has_arg(argc, argv, "--no-dedupe")) != 0) {
            fprintf(stderr, "failed to append ARC probe attempts TSV rows\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_ranker_benchmark_requested) {
        if (arc_ranker_benchmark_attempts == NULL ||
            arc_ranker_benchmark_output_dir == NULL ||
            arc_ranker_benchmark_report == NULL ||
            epochs == 0U ||
            arc_ranker_benchmark_max_per_task == 0U) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_run_arc_ranker_benchmark(
                arc_ranker_benchmark_attempts,
                arc_ranker_benchmark_output_dir,
                arc_ranker_benchmark_report,
                epochs,
                learning_rate,
                arc_ranker_benchmark_max_per_task) != 0) {
            fprintf(stderr, "failed to run ARC ranker benchmark\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_ranker_failure_report_requested) {
        if (arc_ranker_benchmark_attempts == NULL ||
            arc_ranker_failure_proposals == NULL ||
            arc_hole_summary == NULL ||
            arc_ranker_failure_output == NULL ||
            arc_ranker_failure_markdown == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_ranker_failure_report(
                arc_ranker_benchmark_attempts,
                arc_ranker_failure_proposals,
                arc_hole_summary,
                arc_ranker_failure_output,
                arc_ranker_failure_markdown) != 0) {
            fprintf(stderr, "failed to write ARC ranker failure report\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (!arc_synthetic_requested &&
        !arc_synthetic_curriculum_requested &&
        !arc_append_requested &&
        !arc_ranker_benchmark_requested &&
        !arc_ranker_failure_report_requested &&
        (arc_hole_visual_log_count > 0U || arc_hole_candidate_tsv != NULL ||
         arc_hole_examples != NULL || arc_hole_summary != NULL ||
         arc_visual_hole_requested || arc_visual_curriculum_requested)) {
        if (arc_hole_visual_log_count == 0U || arc_hole_candidate_tsv == NULL ||
            arc_hole_examples == NULL || arc_hole_summary == NULL ||
            (arc_visual_hole_requested && arc_visual_curriculum_requested) ||
            (arc_visual_curriculum_requested && arc_hole_attempt_log_count > 1U)) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (arc_visual_curriculum_requested) {
            if (nn_grammar_discovery_write_arc_visual_curriculum(
                    arc_hole_visual_logs,
                    arc_hole_visual_log_count,
                    arc_hole_target_scopes,
                    arc_hole_target_scope_count,
                    arc_hole_attempt_log_count == 0U ? NULL : arc_hole_attempt_logs[0],
                    arc_hole_candidate_tsv,
                    arc_hole_examples,
                    arc_hole_summary,
                    arc_curriculum_max_op_hints) != 0) {
                fprintf(stderr, "failed to write ARC visual curriculum rows\n");
                free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                    arc_pipeline_source_attempts_jsonl,
                                    arc_pipeline_target_attempts_jsonl,
                                    arc_hole_visual_logs,
                                    arc_hole_target_scopes,
                                    arc_hole_attempt_logs);
                return 1;
            }
        } else if (nn_grammar_discovery_write_arc_visual_hole_proposer(
                       arc_hole_visual_logs,
                       arc_hole_visual_log_count,
                       arc_hole_target_scopes,
                       arc_hole_target_scope_count,
                       arc_hole_attempt_logs,
                       arc_hole_attempt_log_count,
                       arc_hole_candidate_tsv,
                       arc_hole_examples,
                       arc_hole_summary,
                       arc_hole_nearest_signatures,
                       arc_hole_max_attempt_seeds,
                       arc_hole_max_ops_per_task,
                       has_arg(argc, argv, "--emit-broad-repair"),
                       has_arg(argc, argv, "--include-selected")) != 0) {
            fprintf(stderr, "failed to write ARC visual hole proposer rows\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_pipeline_source_attempts_jsonl_count > 0U ||
        arc_pipeline_target_attempts_jsonl_count > 0U) {
        if (arc_pipeline_source_attempts_jsonl_count == 0U ||
            arc_pipeline_target_attempts_jsonl_count == 0U ||
            arc_pipeline_proposals_output == NULL ||
            arc_pipeline_summary_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_visual_pipeline_ranker_from_jsonl_with_cache(
                arc_pipeline_source_attempts_jsonl,
                arc_pipeline_source_attempts_jsonl_count,
                arc_pipeline_target_attempts_jsonl,
                arc_pipeline_target_attempts_jsonl_count,
                arc_pipeline_cache,
                arc_pipeline_proposals_output,
                arc_pipeline_summary_output,
                epochs,
                learning_rate,
                arc_pipeline_max_per_task) != 0) {
            fprintf(stderr, "failed to train ARC visual pipeline ranker from JSONL\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_pipeline_attempts != NULL) {
        if (arc_pipeline_attempts == NULL || arc_pipeline_proposals_output == NULL ||
            arc_pipeline_summary_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_visual_pipeline_ranker_with_cache(
                arc_pipeline_attempts,
                arc_pipeline_cache,
                arc_pipeline_proposals_output,
                arc_pipeline_summary_output,
                epochs,
                learning_rate,
                arc_pipeline_max_per_task) != 0) {
            fprintf(stderr, "failed to train ARC visual pipeline ranker\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (arc_pipeline_dedupe_attempts_input_count > 0U ||
        arc_pipeline_dedupe_attempts_output != NULL) {
        if (arc_pipeline_dedupe_attempts_input_count == 0U ||
            arc_pipeline_dedupe_attempts_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 2;
        }
        if (nn_grammar_discovery_dedupe_arc_visual_pipeline_attempts(
                arc_pipeline_dedupe_attempts_inputs,
                arc_pipeline_dedupe_attempts_input_count,
                arc_pipeline_dedupe_attempts_output) != 0) {
            fprintf(stderr, "failed to dedupe ARC visual pipeline attempts\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl,
                                arc_hole_visual_logs,
                                arc_hole_target_scopes,
                                arc_hole_attempt_logs);
            return 1;
        }
    }
    if (candidate_rows == NULL && sketches == NULL && arc_pipeline_attempts == NULL &&
        arc_pipeline_source_attempts_jsonl_count == 0U &&
        arc_pipeline_target_attempts_jsonl_count == 0U &&
        arc_pipeline_dedupe_attempts_output == NULL &&
        haiku_coherence_trace == NULL && arc_hole_visual_log_count == 0U &&
        !arc_synthetic_requested && !arc_synthetic_curriculum_requested &&
        !arc_append_requested &&
        !arc_ranker_benchmark_requested && !arc_ranker_failure_report_requested &&
        !arc_verified_selfplay_requested) {
        usage(argv[0]);
        free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                            arc_pipeline_source_attempts_jsonl,
                            arc_pipeline_target_attempts_jsonl,
                            arc_hole_visual_logs,
                            arc_hole_target_scopes,
                            arc_hole_attempt_logs);
        return 2;
    }
    free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                        arc_pipeline_source_attempts_jsonl,
                        arc_pipeline_target_attempts_jsonl,
                        arc_hole_visual_logs,
                        arc_hole_target_scopes,
                        arc_hole_attempt_logs);
    return 0;
}
