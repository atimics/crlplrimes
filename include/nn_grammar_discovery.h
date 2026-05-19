#ifndef NN_GRAMMAR_DISCOVERY_H
#define NN_GRAMMAR_DISCOVERY_H

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

#endif
