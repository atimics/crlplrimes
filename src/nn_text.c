#include "nn_text.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

static int nn_text_has_unsigned_decimal_form(const char *text)
{
    return text != NULL && text[0] != '\0' && text[0] != '-' && text[0] != '+';
}

int nn_parse_size_text(const char *text, size_t *out_value)
{
    if (!nn_text_has_unsigned_decimal_form(text) || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE ||
        value == 0 || value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *out_value = (size_t)value;
    return 1;
}

int nn_parse_size_nonnegative_text(const char *text, size_t *out_value)
{
    if (!nn_text_has_unsigned_decimal_form(text) || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE ||
        value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *out_value = (size_t)value;
    return 1;
}

int nn_parse_u64_text(const char *text, uint64_t *out_value)
{
    if (!nn_text_has_unsigned_decimal_form(text) || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE ||
        value > (unsigned long long)UINT64_MAX) {
        return 0;
    }
    *out_value = (uint64_t)value;
    return 1;
}

int nn_parse_i64_text(const char *text, int64_t *out_value)
{
    if (text == NULL || text[0] == '\0' || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    long long value = strtoll(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE ||
        value < (long long)INT64_MIN || value > (long long)INT64_MAX) {
        return 0;
    }
    *out_value = (int64_t)value;
    return 1;
}

int nn_parse_double_text(const char *text, double *out_value)
{
    if (text == NULL || text[0] == '\0' || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value)) {
        return 0;
    }
    *out_value = value;
    return 1;
}

int nn_parse_real_text(const char *text, nn_real *out_value)
{
    double value = 0.0;
    if (!nn_parse_double_text(text, &value) || out_value == NULL) {
        return 0;
    }
    *out_value = (nn_real)value;
    return 1;
}

int nn_parse_real_positive_text(const char *text, nn_real *out_value)
{
    nn_real value = (nn_real)0;
    if (!nn_parse_real_text(text, &value) || value <= (nn_real)0) {
        return 0;
    }
    *out_value = value;
    return 1;
}

int nn_parse_real_nonnegative_text(const char *text, nn_real *out_value)
{
    nn_real value = (nn_real)0;
    if (!nn_parse_real_text(text, &value) || value < (nn_real)0) {
        return 0;
    }
    *out_value = value;
    return 1;
}

int nn_write_json_string(FILE *fp, const char *text)
{
    if (fp == NULL || fputc('"', fp) == EOF) {
        return 0;
    }

    if (text != NULL) {
        for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
            switch (*p) {
            case '"':
                if (fputs("\\\"", fp) == EOF) return 0;
                break;
            case '\\':
                if (fputs("\\\\", fp) == EOF) return 0;
                break;
            case '\b':
                if (fputs("\\b", fp) == EOF) return 0;
                break;
            case '\f':
                if (fputs("\\f", fp) == EOF) return 0;
                break;
            case '\n':
                if (fputs("\\n", fp) == EOF) return 0;
                break;
            case '\r':
                if (fputs("\\r", fp) == EOF) return 0;
                break;
            case '\t':
                if (fputs("\\t", fp) == EOF) return 0;
                break;
            default:
                if (*p < 0x20u) {
                    if (fprintf(fp, "\\u%04x", (unsigned int)*p) < 0) return 0;
                } else if (fputc((int)*p, fp) == EOF) {
                    return 0;
                }
                break;
            }
        }
    }

    return fputc('"', fp) != EOF;
}
