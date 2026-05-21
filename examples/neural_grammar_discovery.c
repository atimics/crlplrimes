#include "nn_grammar_discovery.h"

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
                                const char **target_inputs) {
    free(dedupe_inputs);
    free(source_inputs);
    free(target_inputs);
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
            "[--arc-pipeline-cache PATH] [--memory-in PATH] [--memory-out PATH] [--epochs N] "
            "[--learning-rate LR] [--arc-pipeline-max-per-task N]\n",
            argv0);
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
    size_t arc_pipeline_dedupe_attempts_input_count =
        count_arg_values(argc, argv, "--arc-pipeline-dedupe-attempts-input");
    size_t arc_pipeline_source_attempts_jsonl_count =
        count_arg_values(argc, argv, "--arc-pipeline-source-attempts-jsonl");
    size_t arc_pipeline_target_attempts_jsonl_count =
        count_arg_values(argc, argv, "--arc-pipeline-target-attempts-jsonl");
    const char **arc_pipeline_dedupe_attempts_inputs = NULL;
    const char **arc_pipeline_source_attempts_jsonl = NULL;
    const char **arc_pipeline_target_attempts_jsonl = NULL;
    const char *arc_pipeline_max_per_task_arg =
        arg_value(argc, argv, "--arc-pipeline-max-per-task");
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

    if (has_arg(argc, argv, "--help")) {
        usage(argv[0]);
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
    if ((arc_pipeline_dedupe_attempts_input_count > 0U &&
         arc_pipeline_dedupe_attempts_inputs == NULL) ||
        (arc_pipeline_source_attempts_jsonl_count > 0U &&
         arc_pipeline_source_attempts_jsonl == NULL) ||
        (arc_pipeline_target_attempts_jsonl_count > 0U &&
         arc_pipeline_target_attempts_jsonl == NULL)) {
        free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                            arc_pipeline_source_attempts_jsonl,
                            arc_pipeline_target_attempts_jsonl);
        return 1;
    }
    if (candidate_rows != NULL || compiled_output != NULL || verification_output != NULL) {
        if (candidate_rows == NULL || compiled_output == NULL || verification_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl);
            return 2;
        }
        if (nn_grammar_discovery_write_delta_outputs(candidate_rows,
                                                     compiled_output,
                                                     verification_output) != 0) {
            fprintf(stderr, "failed to compile or verify neural grammar delta rows\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl);
            return 1;
        }
    }
    if (sketches != NULL || sketch_examples != NULL || neural_proposals_output != NULL) {
        if (sketches == NULL || sketch_examples == NULL ||
            neural_proposals_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl);
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
                                arc_pipeline_target_attempts_jsonl);
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
                                arc_pipeline_target_attempts_jsonl);
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
                                arc_pipeline_target_attempts_jsonl);
            return 1;
        }
    }
    if (arc_pipeline_attempts != NULL) {
        if (arc_pipeline_attempts == NULL || arc_pipeline_proposals_output == NULL ||
            arc_pipeline_summary_output == NULL) {
            usage(argv[0]);
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl);
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
                                arc_pipeline_target_attempts_jsonl);
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
                                arc_pipeline_target_attempts_jsonl);
            return 2;
        }
        if (nn_grammar_discovery_dedupe_arc_visual_pipeline_attempts(
                arc_pipeline_dedupe_attempts_inputs,
                arc_pipeline_dedupe_attempts_input_count,
                arc_pipeline_dedupe_attempts_output) != 0) {
            fprintf(stderr, "failed to dedupe ARC visual pipeline attempts\n");
            free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                                arc_pipeline_source_attempts_jsonl,
                                arc_pipeline_target_attempts_jsonl);
            return 1;
        }
    }
    if (candidate_rows == NULL && sketches == NULL && arc_pipeline_attempts == NULL &&
        arc_pipeline_source_attempts_jsonl_count == 0U &&
        arc_pipeline_target_attempts_jsonl_count == 0U &&
        arc_pipeline_dedupe_attempts_output == NULL) {
        usage(argv[0]);
        free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                            arc_pipeline_source_attempts_jsonl,
                            arc_pipeline_target_attempts_jsonl);
        return 2;
    }
    free_arc_arg_arrays(arc_pipeline_dedupe_attempts_inputs,
                        arc_pipeline_source_attempts_jsonl,
                        arc_pipeline_target_attempts_jsonl);
    return 0;
}
