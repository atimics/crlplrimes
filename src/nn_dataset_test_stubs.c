/* Stubs for external experiment adapters not needed in the test runner. */
#include "nn_dataset.h"

int gen_go_adapter(const NNDatasetAdapterConfig *cfg, NNDataset *ds) {
    (void)cfg;
    (void)ds;
    return 0;
}
