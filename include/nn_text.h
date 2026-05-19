#ifndef NN_TEXT_H
#define NN_TEXT_H

#include "nn_core.h"

#include <stdint.h>

int nn_parse_size_text(const char *text, size_t *out_value);
int nn_parse_size_nonnegative_text(const char *text, size_t *out_value);
int nn_parse_u64_text(const char *text, uint64_t *out_value);
int nn_parse_i64_text(const char *text, int64_t *out_value);
int nn_parse_double_text(const char *text, double *out_value);
int nn_parse_real_text(const char *text, nn_real *out_value);
int nn_parse_real_positive_text(const char *text, nn_real *out_value);
int nn_parse_real_nonnegative_text(const char *text, nn_real *out_value);
int nn_write_json_string(FILE *fp, const char *text);

#endif
