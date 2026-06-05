/* Stub: Signal integration holo-neural network shim.
 * The real architecture is now memory_neuron.h. These stubs keep the
 * Signal sandbox experiments compiling. */
#include "leos_hnn.h"
#include <stdlib.h>
#include <string.h>

void hnn_action_table_init(void *tbl, size_t dim) { (void)tbl; (void)dim; }
void hnn_encode_state(const void *st, size_t dim, double *out) {
    (void)st; (void)dim;
    memset(out, 0, dim * sizeof(double));
}
void hnn_memory_cleanup(void *mem) { (void)mem; }
void hnn_memory_store(void *mem, const double *st, const double *act) {
    (void)mem; (void)st; (void)act;
}
void hnn_normalize(double *v) { (void)v; }
double hnn_similarity(const double *a, const double *b, size_t dim) {
    (void)a; (void)b; (void)dim;
    return 0.0;
}
