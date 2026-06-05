#ifndef LEOS_HNN_H
#define LEOS_HNN_H
#include <stddef.h>

/* Stub: Signal holo-neural network integration shim. */
void hnn_action_table_init(void *tbl, size_t dim);
void hnn_encode_state(const void *st, size_t dim, double *out);
void hnn_memory_cleanup(void *mem);
void hnn_memory_store(void *mem, const double *st, const double *act);
void hnn_normalize(double *v);
double hnn_similarity(const double *a, const double *b, size_t dim);

#endif
