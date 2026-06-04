#include "leos_hnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char *name) {
    if (condition) {
        printf("ok   %s\n", name);
    } else {
        printf("FAIL %s\n", name);
        failures++;
    }
}

int main(void) {
    printf("=== EML Holographic Neuron Tests ===\n\n");

    /* Test 1: eml_bind produces normalized output */
    {
        HolographicNet *net = hnn_create(64);
        check(net != NULL, "hnn_create(64)");

        double *a = malloc(64 * sizeof(double));
        double *b = malloc(64 * sizeof(double));
        double *out = malloc(64 * sizeof(double));

        leos_engine_keygen(hnn_engine(net), 42, a);
        leos_engine_keygen(hnn_engine(net), 99, b);

        leos_engine_eml_bind(hnn_engine(net), a, b, out);

        /* Output should be normalized (unit length) */
        double norm = 0.0;
        for (size_t i = 0; i < 64; i++) norm += out[i] * out[i];
        norm = sqrt(norm);
        check(norm > 0.999 && norm < 1.001, "eml_bind output is unit-norm");

        /* eml_bind(a, b) should differ from bind(a, b) */
        double *bind_out = malloc(64 * sizeof(double));
        leos_engine_bind(hnn_engine(net), a, b, bind_out);
        double dot = leos_engine_dot(hnn_engine(net), out, bind_out);
        check(dot < 0.99, "eml_bind differs from bind (dot < 0.99)");

        free(a); free(b); free(out); free(bind_out);
        hnn_destroy(net);
    }

    /* Test 2: hnn_add_layer_mixed with EML neurons */
    {
        HolographicNet *net = hnn_create(64);
        check(net != NULL, "hnn_create(64) for mixed layer");

        size_t li = hnn_add_layer_mixed(net, 8, 0.25, 0.25, 0.5);
        check(li == 0, "hnn_add_layer_mixed returns index 0");
        check(net->layer_count == 1, "layer_count == 1");

        /* Verify neuron mode distribution */
        int n_bind = 0, n_unbind = 0, n_eml = 0;
        for (size_t i = 0; i < 8; i++) {
            switch (net->layers[0].neurons[i].mode) {
            case HOLOGRAPHIC_NEURON_BIND:   n_bind++; break;
            case HOLOGRAPHIC_NEURON_UNBIND: n_unbind++; break;
            case HOLOGRAPHIC_NEURON_EML:    n_eml++; break;
            }
        }
        check(n_bind == 2, "2 bind neurons (25%%)");
        check(n_unbind == 2, "2 unbind neurons (25%%)");
        check(n_eml == 4, "4 eml neurons (50%%)");

        hnn_destroy(net);
    }

    /* Test 3: Forward pass through EML layer doesn't crash */
    {
        HolographicNet *net = hnn_create(64);
        size_t li = hnn_add_layer_mixed(net, 4, 0.0, 0.0, 1.0);
        check(li == 0, "all-EML layer created");

        double *input = malloc(64 * sizeof(double));
        double *output = malloc(64 * sizeof(double));
        leos_engine_keygen(hnn_engine(net), 7, input);

        hnn_layer_forward(net, 0, input, output);

        /* Output should be normalized */
        double norm = 0.0;
        for (size_t i = 0; i < 64; i++) norm += output[i] * output[i];
        norm = sqrt(norm);
        check(norm > 0.999 && norm < 1.001, "EML layer forward pass: output is unit-norm");

        /* Should not be NaN */
        int has_nan = 0;
        for (size_t i = 0; i < 64; i++)
            if (output[i] != output[i]) has_nan = 1;
        check(!has_nan, "EML layer forward pass: no NaN in output");

        free(input); free(output);
        hnn_destroy(net);
    }

    /* Test 4: EML neuron mode through full forward pass */
    {
        HolographicNet *net = hnn_create(64);
        hnn_add_memory(net);
        hnn_add_memory(net);
        hnn_add_layer_mixed(net, 6, 0.33, 0.33, 0.34);

        double *key = malloc(64 * sizeof(double));
        double *input = malloc(64 * sizeof(double));
        double *output = malloc(64 * sizeof(double));
        leos_engine_keygen(hnn_engine(net), 1, key);
        leos_engine_keygen(hnn_engine(net), 2, input);

        hnn_forward(net, key, input, output);

        double norm = 0.0;
        for (size_t i = 0; i < 64; i++) norm += output[i] * output[i];
        norm = sqrt(norm);
        check(norm > 0.999 && norm < 1.001, "full forward pass with EML neurons: unit-norm output");

        int has_nan = 0;
        for (size_t i = 0; i < 64; i++)
            if (output[i] != output[i]) has_nan = 1;
        check(!has_nan, "full forward pass with EML neurons: no NaN");

        free(key); free(input); free(output);
        hnn_destroy(net);
    }

    printf("\n=== Results: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
