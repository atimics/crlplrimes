#ifndef NN_JSON_H
#define NN_JSON_H

#include <stddef.h>

typedef struct {
    const char *cursor;
    const char *end;
} NNJsonCursor;

void nn_json_skip_ws(NNJsonCursor *parser);
int nn_json_expect(NNJsonCursor *parser, char expected);
int nn_json_parse_string(NNJsonCursor *parser, char *out, size_t out_size);
int nn_json_skip_value(NNJsonCursor *parser);
int nn_json_parse_uint(NNJsonCursor *parser, unsigned long *out_value);

#endif
