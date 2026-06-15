#include "nn_paper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int has_arg(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static const char *arg_value(int argc, char **argv, const char *name)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return NULL;
}

static size_t count_arg_values(int argc, char **argv, const char *name)
{
    size_t count = 0U;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            ++count;
            ++i;
        }
    }
    return count;
}

static const char **collect_arg_values(int argc,
                                       char **argv,
                                       const char *name,
                                       size_t count)
{
    const char **values;
    size_t out = 0U;
    if (count == 0U) {
        return NULL;
    }
    values = (const char **)calloc(count, sizeof(char *));
    if (values == NULL) {
        return NULL;
    }
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            values[out++] = argv[i + 1];
            ++i;
        }
    }
    return values;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --trace-determinism-audit --trace-input PATH --output PATH "
            "[--canonical-output PATH] [--gate]\n"
            "       %s --trace-determinism-compare --trace-input PATH ... "
            "--output PATH [--gate]\n",
            argv0,
            argv0);
}

int main(int argc, char **argv)
{
    const int audit = has_arg(argc, argv, "--trace-determinism-audit");
    const int compare = has_arg(argc, argv, "--trace-determinism-compare");
    const char *output = arg_value(argc, argv, "--output");
    const char *canonical_output = arg_value(argc, argv, "--canonical-output");
    const int gate = has_arg(argc, argv, "--gate");
    const size_t trace_count = count_arg_values(argc, argv, "--trace-input");
    const char **traces = collect_arg_values(argc, argv, "--trace-input", trace_count);
    int rc;

    if ((audit == compare) || output == NULL || trace_count == 0U || traces == NULL) {
        usage(argv[0]);
        free(traces);
        return 2;
    }
    if (audit && trace_count != 1U) {
        usage(argv[0]);
        free(traces);
        return 2;
    }
    if (compare && canonical_output != NULL) {
        usage(argv[0]);
        free(traces);
        return 2;
    }

    rc = nn_paper_write_trace_determinism(traces,
                                          trace_count,
                                          output,
                                          canonical_output,
                                          gate);
    free(traces);
    return rc == 0 ? 0 : 1;
}
