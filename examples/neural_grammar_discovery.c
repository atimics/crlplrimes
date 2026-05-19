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

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--candidate-rows PATH --compiled-output PATH "
            "--verification-output PATH] [--sketches PATH --sketch-examples PATH "
            "--neural-proposals-output PATH] [--arc-pipeline-attempts PATH "
            "--arc-pipeline-proposals-output PATH --arc-pipeline-summary-output PATH] "
            "[--memory-in PATH] [--memory-out PATH] [--epochs N] "
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
    if (candidate_rows != NULL || compiled_output != NULL || verification_output != NULL) {
        if (candidate_rows == NULL || compiled_output == NULL || verification_output == NULL) {
            usage(argv[0]);
            return 2;
        }
        if (nn_grammar_discovery_write_delta_outputs(candidate_rows,
                                                     compiled_output,
                                                     verification_output) != 0) {
            fprintf(stderr, "failed to compile or verify neural grammar delta rows\n");
            return 1;
        }
    }
    if (sketches != NULL || sketch_examples != NULL || neural_proposals_output != NULL) {
        if (sketches == NULL || sketch_examples == NULL ||
            neural_proposals_output == NULL) {
            usage(argv[0]);
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
            return 1;
        }
    }
    if (arc_pipeline_attempts != NULL || arc_pipeline_proposals_output != NULL ||
        arc_pipeline_summary_output != NULL) {
        if (arc_pipeline_attempts == NULL || arc_pipeline_proposals_output == NULL ||
            arc_pipeline_summary_output == NULL) {
            usage(argv[0]);
            return 2;
        }
        if (nn_grammar_discovery_write_arc_visual_pipeline_ranker(
                arc_pipeline_attempts,
                arc_pipeline_proposals_output,
                arc_pipeline_summary_output,
                epochs,
                learning_rate,
                arc_pipeline_max_per_task) != 0) {
            fprintf(stderr, "failed to train ARC visual pipeline ranker\n");
            return 1;
        }
    }
    if (candidate_rows == NULL && sketches == NULL && arc_pipeline_attempts == NULL) {
        usage(argv[0]);
        return 2;
    }
    return 0;
}
