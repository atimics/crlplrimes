#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

static int print_grid(const char *label, const NNBitGrid *grid)
{
    char buffer[128];
    if (!nn_bitgrid_write_pattern(grid, buffer, sizeof(buffer))) {
        return 0;
    }
    printf("%s=%s", label, buffer);
    return 1;
}

static int show_width(size_t width,
                      const NNBitwiseGate *xor_gate,
                      const NNBitwiseGate *and_gate,
                      const NNBitwiseGate *or_gate,
                      const NNBitwiseGate *not_a_gate)
{
    NNBitGrid a = {0};
    NNBitGrid b = {0};
    NNBitGrid x = {0};
    NNBitGrid y = {0};
    NNBitGrid z = {0};
    NNBitGrid n = {0};
    int ok = nn_bitgrid_init(&a, width, 1u) &&
             nn_bitgrid_init(&b, width, 1u) &&
             nn_bitgrid_init(&x, width, 1u) &&
             nn_bitgrid_init(&y, width, 1u) &&
             nn_bitgrid_init(&z, width, 1u) &&
             nn_bitgrid_init(&n, width, 1u);
    if (!ok) {
        goto done;
    }

    ok = nn_bitgrid_fill_alternating(&a, 0u) &&
         nn_bitgrid_fill_alternating(&b, 1u) &&
         nn_bitwise_gate_apply(xor_gate, &a, &b, &x) &&
         nn_bitwise_gate_apply(and_gate, &a, &b, &y) &&
         nn_bitwise_gate_apply(or_gate, &a, &b, &z) &&
         nn_bitwise_gate_apply(not_a_gate, &a, &b, &n);
    if (ok) {
        printf("bits=%zu ", width);
        ok = print_grid("a", &a) &&
             printf(" ") >= 0 &&
             print_grid("b", &b) &&
             printf(" ") >= 0 &&
             print_grid("xor", &x) &&
             printf(" ") >= 0 &&
             print_grid("and", &y) &&
             printf(" ") >= 0 &&
             print_grid("or", &z) &&
             printf(" ") >= 0 &&
             print_grid("not_a", &n) &&
             printf("\n") >= 0;
    }

done:
    nn_bitgrid_free(&a);
    nn_bitgrid_free(&b);
    nn_bitgrid_free(&x);
    nn_bitgrid_free(&y);
    nn_bitgrid_free(&z);
    nn_bitgrid_free(&n);
    return ok;
}

static int build_all_rows_grid(NNBitGrid *x, NNBitGrid *y, NNBitGrid *target)
{
    return nn_bitgrid_set(x, 0u, 0u, 0u) &&
           nn_bitgrid_set(y, 0u, 0u, 0u) &&
           nn_bitgrid_set(target, 0u, 0u, 0u) &&
           nn_bitgrid_set(x, 1u, 0u, 0u) &&
           nn_bitgrid_set(y, 1u, 0u, 1u) &&
           nn_bitgrid_set(target, 1u, 0u, 1u) &&
           nn_bitgrid_set(x, 0u, 1u, 1u) &&
           nn_bitgrid_set(y, 0u, 1u, 0u) &&
           nn_bitgrid_set(target, 0u, 1u, 1u) &&
           nn_bitgrid_set(x, 1u, 1u, 1u) &&
           nn_bitgrid_set(y, 1u, 1u, 1u) &&
           nn_bitgrid_set(target, 1u, 1u, 0u);
}

static int show_transform_solver(const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES])
{
    NNBitwiseToolPolicyConfig policy_config;
    NNBitwiseToolPolicy policy;
    NNBitwiseMacroLibrary library;
    NNBitwiseChain chain;
    NNBitGrid x = {0};
    NNBitGrid y = {0};
    NNBitGrid target = {0};
    NNBitGrid inferred = {0};
    NNBitGrid macro_out = {0};

    nn_bitwise_tool_policy_config_init(&policy_config);
    policy_config.seed = 31337u;
    nn_bitwise_tool_policy_init(&policy);
    nn_bitwise_macro_library_init(&library);
    nn_bitwise_chain_init(&chain);

    int ok = nn_bitwise_tool_policy_train(&policy,
                                          nn_bitwise_default_tool_mask(),
                                          4u,
                                          &policy_config);
    printf("tool_policy oracle_coverage=%.2f updates=%zu final_loss=%.6f\n",
           policy.oracle_coverage,
           policy.training_updates,
           (double)policy.final_loss);
    ok = ok && nn_bitwise_macro_library_learn(&library,
                                              &policy,
                                              gates,
                                              0xffffu);
    printf("macro_library promoted=%zu/%zu fallback_promoted=%zu masks=%u\n",
           library.promoted_count,
           library.candidate_count,
           library.fallback_promoted_count,
           (unsigned)library.promoted_truth_masks);
    ok = ok &&
         nn_bitgrid_init(&x, 2u, 2u) &&
         nn_bitgrid_init(&y, 2u, 2u) &&
         nn_bitgrid_init(&target, 2u, 2u) &&
         nn_bitgrid_init(&inferred, 2u, 2u) &&
         nn_bitgrid_init(&macro_out, 2u, 2u) &&
         build_all_rows_grid(&x, &y, &target) &&
         nn_bitwise_chain_solve_grid_transform(&policy, gates, &x, &y, &target, &chain) &&
         nn_bitwise_chain_apply(&chain, gates, &x, &y, &inferred) &&
         nn_bitgrid_hamming_distance(&inferred, &target) == 0u;
    const NNBitwiseMacroTool *xor_macro =
        nn_bitwise_macro_library_find(&library, NN_BITWISE_XOR);
    ok = ok && xor_macro != NULL &&
         nn_bitwise_macro_tool_apply(xor_macro, gates, &x, &y, &macro_out) &&
         nn_bitgrid_hamming_distance(&macro_out, &target) == 0u;

    if (ok) {
        fputs("transform_chain X + Y + Z = 0 over GF(2):\n", stdout);
        nn_bitwise_chain_write(stdout, &chain);
        fputs("promoted_macro xor:\n", stdout);
        nn_bitwise_chain_write(stdout, &xor_macro->chain);
        print_grid("x", &x);
        printf(" ");
        print_grid("y", &y);
        printf(" ");
        print_grid("target_z", &target);
        printf(" ");
        print_grid("inferred_z", &inferred);
        printf(" ");
        print_grid("macro_z", &macro_out);
        printf("\n");
    }

    nn_bitgrid_free(&x);
    nn_bitgrid_free(&y);
    nn_bitgrid_free(&target);
    nn_bitgrid_free(&inferred);
    nn_bitgrid_free(&macro_out);
    nn_bitwise_tool_policy_free(&policy);
    return ok;
}

int main(void)
{
    NNBitwiseTrainConfig config;
    nn_bitwise_train_config_init(&config);
    config.seed = 424242u;
    config.min_margin = (nn_real)0.15;

    NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES];
    for (size_t i = 0; i < NN_BITWISE_BINARY_TRUTH_TABLES; ++i) {
        nn_bitwise_gate_init(&gates[i]);
    }

    size_t promoted = 0;
    for (uint8_t mask = 0; mask < NN_BITWISE_BINARY_TRUTH_TABLES; ++mask) {
        if (nn_bitwise_gate_train_truth_table(&gates[mask], mask, &config)) {
            ++promoted;
        }
        nn_bitwise_gate_write_json(stdout, &gates[mask]);
    }

    printf("certified_binary_operations: %zu/%u\n",
           promoted,
           (unsigned)NN_BITWISE_BINARY_TRUTH_TABLES);
    if (promoted != NN_BITWISE_BINARY_TRUTH_TABLES) {
        for (size_t i = 0; i < NN_BITWISE_BINARY_TRUTH_TABLES; ++i) {
            nn_bitwise_gate_free(&gates[i]);
        }
        return EXIT_FAILURE;
    }

    size_t recursive_cells = 0;
    for (size_t width = 1; width <= 6; ++width) {
        if (!show_width(width,
                        &gates[NN_BITWISE_XOR],
                        &gates[NN_BITWISE_AND],
                        &gates[NN_BITWISE_OR],
                        &gates[NN_BITWISE_NOT_A])) {
            for (size_t i = 0; i < NN_BITWISE_BINARY_TRUTH_TABLES; ++i) {
                nn_bitwise_gate_free(&gates[i]);
            }
            return EXIT_FAILURE;
        }
        recursive_cells += width;
    }
    printf("certificate_fast_path_cells: %zu\n", recursive_cells);

    if (!show_transform_solver(gates)) {
        for (size_t i = 0; i < NN_BITWISE_BINARY_TRUTH_TABLES; ++i) {
            nn_bitwise_gate_free(&gates[i]);
        }
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < NN_BITWISE_BINARY_TRUTH_TABLES; ++i) {
        nn_bitwise_gate_free(&gates[i]);
    }
    return EXIT_SUCCESS;
}
