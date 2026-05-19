#include "nn_json.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

void nn_json_skip_ws(NNJsonCursor *parser)
{
    if (parser == NULL) {
        return;
    }
    while (parser->cursor < parser->end &&
           isspace((unsigned char)*parser->cursor)) {
        ++parser->cursor;
    }
}

int nn_json_expect(NNJsonCursor *parser, char expected)
{
    if (parser == NULL) {
        return 0;
    }
    nn_json_skip_ws(parser);
    if (parser->cursor >= parser->end || *parser->cursor != expected) {
        return 0;
    }
    ++parser->cursor;
    return 1;
}

int nn_json_parse_string(NNJsonCursor *parser, char *out, size_t out_size)
{
    if (parser == NULL) {
        return 0;
    }
    nn_json_skip_ws(parser);
    if (parser->cursor >= parser->end || *parser->cursor != '"') {
        return 0;
    }
    ++parser->cursor;
    size_t used = 0;
    while (parser->cursor < parser->end && *parser->cursor != '"') {
        unsigned char ch = (unsigned char)*parser->cursor++;
        if (ch == '\\') {
            if (parser->cursor >= parser->end) {
                return 0;
            }
            ch = (unsigned char)*parser->cursor++;
            if (ch == 'u') {
                for (size_t i = 0; i < 4; ++i) {
                    if (parser->cursor >= parser->end ||
                        !isxdigit((unsigned char)*parser->cursor)) {
                        return 0;
                    }
                    ++parser->cursor;
                }
                ch = '?';
            }
        }
        if (used + 1u < out_size) {
            out[used++] = (char)ch;
        }
    }
    if (parser->cursor >= parser->end || *parser->cursor != '"') {
        return 0;
    }
    ++parser->cursor;
    if (out_size > 0) {
        out[used] = '\0';
    }
    return 1;
}

static int nn_json_skip_array_or_object(NNJsonCursor *parser, char open, char close)
{
    if (!nn_json_expect(parser, open)) {
        return 0;
    }
    nn_json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == close) {
        ++parser->cursor;
        return 1;
    }
    for (;;) {
        if (open == '{') {
            char key[64];
            if (!nn_json_parse_string(parser, key, sizeof(key)) ||
                !nn_json_expect(parser, ':') ||
                !nn_json_skip_value(parser)) {
                return 0;
            }
        } else if (!nn_json_skip_value(parser)) {
            return 0;
        }
        nn_json_skip_ws(parser);
        if (parser->cursor < parser->end && *parser->cursor == ',') {
            ++parser->cursor;
            continue;
        }
        if (parser->cursor < parser->end && *parser->cursor == close) {
            ++parser->cursor;
            return 1;
        }
        return 0;
    }
}

int nn_json_skip_value(NNJsonCursor *parser)
{
    if (parser == NULL) {
        return 0;
    }
    nn_json_skip_ws(parser);
    if (parser->cursor >= parser->end) {
        return 0;
    }
    if (*parser->cursor == '"') {
        char scratch[1];
        return nn_json_parse_string(parser, scratch, sizeof(scratch));
    }
    if (*parser->cursor == '[') {
        return nn_json_skip_array_or_object(parser, '[', ']');
    }
    if (*parser->cursor == '{') {
        return nn_json_skip_array_or_object(parser, '{', '}');
    }
    while (parser->cursor < parser->end &&
           strchr(",]}\r\n\t ", *parser->cursor) == NULL) {
        ++parser->cursor;
    }
    return 1;
}

int nn_json_parse_uint(NNJsonCursor *parser, unsigned long *out_value)
{
    if (parser == NULL || out_value == NULL) {
        return 0;
    }
    nn_json_skip_ws(parser);
    if (parser->cursor >= parser->end || !isdigit((unsigned char)*parser->cursor)) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(parser->cursor, &end, 10);
    if (end == parser->cursor || errno == ERANGE || end > parser->end) {
        return 0;
    }
    parser->cursor = end;
    *out_value = value;
    return 1;
}
