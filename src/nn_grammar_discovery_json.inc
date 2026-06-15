/* Internal JSON/JSONL helpers for nn_grammar_discovery.c. */
static char *nn_gd_read_line_dynamic(FILE *fp) {
    size_t capacity = 4096U;
    size_t used = 0U;
    char *line = (char *)malloc(capacity);
    if (line == NULL) {
        return NULL;
    }
    line[0] = '\0';
    for (;;) {
        if (fgets(line + used, (int)(capacity - used), fp) == NULL) {
            if (used == 0U) {
                free(line);
                return NULL;
            }
            return line;
        }
        used += strlen(line + used);
        if (used > 0U && line[used - 1U] == '\n') {
            return line;
        }
        if (capacity - used <= 1U) {
            size_t next_capacity = capacity * 2U;
            char *next = (char *)realloc(line, next_capacity);
            if (next == NULL) {
                free(line);
                return NULL;
            }
            line = next;
            capacity = next_capacity;
        }
    }
}

static const char *nn_gd_json_find_key_value(const char *line, const char *key) {
    char pattern[128];
    int written;
    const char *start;
    if (line == NULL || key == NULL || key[0] == '\0') {
        return NULL;
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    start = strstr(line, pattern);
    return start == NULL ? NULL : start + (size_t)written;
}

static const char *nn_gd_json_skip_space(const char *cursor) {
    while (cursor != NULL && isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static int nn_gd_json_append_char(char **buffer,
                                  size_t *used,
                                  size_t *capacity,
                                  char value) {
    if (*used + 1U >= *capacity) {
        size_t next_capacity = *capacity * 2U;
        char *next = (char *)realloc(*buffer, next_capacity);
        if (next == NULL) {
            return -1;
        }
        *buffer = next;
        *capacity = next_capacity;
    }
    (*buffer)[(*used)++] = value;
    return 0;
}

static char *nn_gd_json_dup_string_value(const char *cursor) {
    size_t capacity = 64U;
    size_t used = 0U;
    char *value;
    cursor = nn_gd_json_skip_space(cursor);
    if (cursor == NULL || *cursor != '"') {
        return nn_gd_strdup("");
    }
    ++cursor;
    value = (char *)malloc(capacity);
    if (value == NULL) {
        return NULL;
    }
    while (*cursor != '\0' && *cursor != '"') {
        unsigned char ch = (unsigned char)*cursor++;
        if (ch == '\\') {
            ch = (unsigned char)*cursor++;
            switch (ch) {
                case '"':
                case '\\':
                case '/':
                    break;
                case 'b':
                    ch = '\b';
                    break;
                case 'f':
                    ch = '\f';
                    break;
                case 'n':
                    ch = '\n';
                    break;
                case 'r':
                    ch = '\r';
                    break;
                case 't':
                    ch = '\t';
                    break;
                case 'u': {
                    size_t i;
                    for (i = 0U; i < 4U; ++i) {
                        if (!isxdigit((unsigned char)cursor[i])) {
                            free(value);
                            return NULL;
                        }
                    }
                    cursor += 4U;
                    ch = '?';
                    break;
                }
                default:
                    free(value);
                    return NULL;
            }
        }
        if (nn_gd_json_append_char(&value, &used, &capacity, (char)ch) != 0) {
            free(value);
            return NULL;
        }
    }
    if (*cursor != '"') {
        free(value);
        return NULL;
    }
    value[used] = '\0';
    return value;
}

static char *nn_gd_json_dup_string_value_advance(const char **cursor_ref) {
    const char *cursor;
    char *value;
    cursor = nn_gd_json_skip_space(cursor_ref == NULL ? NULL : *cursor_ref);
    if (cursor == NULL || *cursor != '"') {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor;
        }
        return nn_gd_strdup("");
    }
    value = nn_gd_json_dup_string_value(cursor);
    if (value == NULL) {
        return NULL;
    }
    ++cursor;
    while (*cursor != '\0' && *cursor != '"') {
        if (*cursor == '\\' && cursor[1] != '\0') {
            cursor += 2;
        } else {
            ++cursor;
        }
    }
    if (*cursor == '"') {
        ++cursor;
    }
    if (cursor_ref != NULL) {
        *cursor_ref = cursor;
    }
    return value;
}

static void nn_gd_sanitize_tsv_value(char *value) {
    char *p;
    if (value == NULL) {
        return;
    }
    for (p = value; *p != '\0'; ++p) {
        if (*p == '\t' || *p == '\r' || *p == '\n') {
            *p = ' ';
        }
    }
}

static char *nn_gd_json_dup_key_string(const char *line, const char *key) {
    return nn_gd_json_dup_string_value(nn_gd_json_find_key_value(line, key));
}

static int nn_gd_json_get_bool(const char *line, const char *key) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_find_key_value(line, key));
    if (cursor == NULL) {
        return 0;
    }
    return strncmp(cursor, "true", 4U) == 0 || strncmp(cursor, "1", 1U) == 0;
}

static double nn_gd_json_get_double_or_zero(const char *line, const char *key) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_find_key_value(line, key));
    char *end = NULL;
    double value;
    if (cursor == NULL) {
        return 0.0;
    }
    value = strtod(cursor, &end);
    if (end == cursor || !isfinite(value)) {
        return 0.0;
    }
    return value;
}

static char *nn_gd_json_dup_scalar_tsv_value_at(const char **cursor_ref,
                                                int bool_as_int,
                                                int null_as_none_literal) {
    const char *cursor = nn_gd_json_skip_space(cursor_ref == NULL ? NULL : *cursor_ref);
    const char *end;
    char *value;
    size_t len;
    if (cursor == NULL) {
        return nn_gd_strdup("");
    }
    if (*cursor == '"') {
        value = nn_gd_json_dup_string_value_advance(&cursor);
        nn_gd_sanitize_tsv_value(value);
        if (cursor_ref != NULL) {
            *cursor_ref = cursor;
        }
        return value;
    }
    if (strncmp(cursor, "true", 4U) == 0) {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor + 4U;
        }
        return nn_gd_strdup(bool_as_int ? "1" : "True");
    }
    if (strncmp(cursor, "false", 5U) == 0) {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor + 5U;
        }
        return nn_gd_strdup(bool_as_int ? "0" : "False");
    }
    if (strncmp(cursor, "null", 4U) == 0) {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor + 4U;
        }
        return nn_gd_strdup(null_as_none_literal ? "None" : "");
    }
    end = cursor;
    while (*end != '\0' && *end != ',' && *end != '}' && *end != ']' &&
           !isspace((unsigned char)*end)) {
        ++end;
    }
    len = (size_t)(end - cursor);
    value = (char *)malloc(len + 1U);
    if (value == NULL) {
        return NULL;
    }
    memcpy(value, cursor, len);
    value[len] = '\0';
    if (cursor_ref != NULL) {
        *cursor_ref = end;
    }
    return value;
}

static char *nn_gd_json_dup_scalar_tsv_value_at_stable_number(
    const char **cursor_ref,
    int bool_as_int) {
    const char *cursor = nn_gd_json_skip_space(cursor_ref == NULL ? NULL : *cursor_ref);
    const char *end;
    int looks_float = 0;
    char *value;
    if (cursor == NULL || *cursor == '"' ||
        strncmp(cursor, "true", 4U) == 0 ||
        strncmp(cursor, "false", 5U) == 0 ||
        strncmp(cursor, "null", 4U) == 0) {
        return nn_gd_json_dup_scalar_tsv_value_at(cursor_ref, bool_as_int, 0);
    }
    end = cursor;
    while (*end != '\0' && *end != ',' && *end != '}' && *end != ']' &&
           !isspace((unsigned char)*end)) {
        if (*end == '.' || *end == 'e' || *end == 'E') {
            looks_float = 1;
        }
        ++end;
    }
    if (!looks_float) {
        return nn_gd_json_dup_scalar_tsv_value_at(cursor_ref, bool_as_int, 0);
    }
    {
        char *parse_end = NULL;
        double parsed = strtod(cursor, &parse_end);
        char buffer[64];
        size_t len;
        int has_decimal = 0;
        if (parse_end == cursor || !isfinite(parsed)) {
            return nn_gd_json_dup_scalar_tsv_value_at(cursor_ref, bool_as_int, 0);
        }
        snprintf(buffer, sizeof(buffer), "%.15g", parsed);
        for (len = 0U; buffer[len] != '\0'; ++len) {
            if (buffer[len] == '.' || buffer[len] == 'e' || buffer[len] == 'E') {
                has_decimal = 1;
                break;
            }
        }
        if (!has_decimal) {
            len = strlen(buffer);
            if (len + 2U < sizeof(buffer)) {
                buffer[len++] = '.';
                buffer[len++] = '0';
                buffer[len] = '\0';
            }
        }
        value = nn_gd_strdup(buffer);
        if (cursor_ref != NULL) {
            *cursor_ref = end;
        }
        return value;
    }
}

static const char *nn_gd_json_skip_string_literal(const char *cursor) {
    if (cursor == NULL || *cursor != '"') {
        return cursor;
    }
    ++cursor;
    while (*cursor != '\0') {
        if (*cursor == '\\' && cursor[1] != '\0') {
            cursor += 2;
            continue;
        }
        if (*cursor == '"') {
            return cursor + 1;
        }
        ++cursor;
    }
    return cursor;
}

static void nn_gd_json_index_line(const char *line, NNGDJsonIndex *index) {
    const char *cursor = line;
    if (index == NULL) {
        return;
    }
    index->count = 0U;
    while (cursor != NULL && *cursor != '\0') {
        const char *key_start;
        const char *key_end;
        const char *after_key;
        const char *value;
        while (*cursor != '\0' && *cursor != '"') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        key_start = cursor + 1;
        after_key = nn_gd_json_skip_string_literal(cursor);
        if (after_key == NULL || after_key == cursor || after_key[-1] != '"') {
            break;
        }
        key_end = after_key - 1;
        value = nn_gd_json_skip_space(after_key);
        if (value != NULL && *value == ':' && index->count < 192U) {
            ++value;
            index->items[index->count].key = key_start;
            index->items[index->count].key_len = (size_t)(key_end - key_start);
            index->items[index->count].value = value;
            ++index->count;
        }
        cursor = after_key;
    }
}

static const char *nn_gd_json_index_find(const NNGDJsonIndex *index,
                                         const char *key) {
    size_t i;
    size_t key_len;
    if (index == NULL || key == NULL) {
        return NULL;
    }
    key_len = strlen(key);
    for (i = 0U; i < index->count; ++i) {
        if (index->items[i].key_len == key_len &&
            strncmp(index->items[i].key, key, key_len) == 0) {
            return index->items[i].value;
        }
    }
    return NULL;
}

static char *nn_gd_json_dup_index_tsv_string(const NNGDJsonIndex *index,
                                             const char *key) {
    char *value = nn_gd_json_dup_string_value(nn_gd_json_index_find(index, key));
    nn_gd_sanitize_tsv_value(value);
    return value;
}

static char *nn_gd_json_dup_index_scalar_tsv(const NNGDJsonIndex *index,
                                             const char *key,
                                             int bool_as_int) {
    const char *cursor = nn_gd_json_index_find(index, key);
    return nn_gd_json_dup_scalar_tsv_value_at(&cursor, bool_as_int, 0);
}

static char *nn_gd_json_dup_index_scalar_tsv_stable_number(
    const NNGDJsonIndex *index,
    const char *key,
    int bool_as_int) {
    const char *cursor = nn_gd_json_index_find(index, key);
    return nn_gd_json_dup_scalar_tsv_value_at_stable_number(&cursor, bool_as_int);
}

static int nn_gd_append_sanitized_value(char **buffer,
                                        size_t *used,
                                        size_t *capacity,
                                        const char *value) {
    const char *p;
    if (value == NULL) {
        return 0;
    }
    for (p = value; *p != '\0'; ++p) {
        char ch = *p;
        if (ch == '|') {
            ch = '/';
        } else if (ch == '\t' || ch == '\r' || ch == '\n') {
            ch = ' ';
        }
        if (nn_gd_json_append_char(buffer, used, capacity, ch) != 0) {
            return -1;
        }
    }
    return 0;
}

static char *nn_gd_make_slot_pair(const char *key, const char *value) {
    size_t capacity = 64U;
    size_t used = 0U;
    char *pair = (char *)malloc(capacity);
    if (pair == NULL) {
        return NULL;
    }
    if (nn_gd_append_sanitized_value(&pair, &used, &capacity, key) != 0 ||
        nn_gd_json_append_char(&pair, &used, &capacity, '=') != 0 ||
        nn_gd_append_sanitized_value(&pair, &used, &capacity, value) != 0) {
        free(pair);
        return NULL;
    }
    pair[used] = '\0';
    return pair;
}

static char *nn_gd_json_dup_slot_tokens_at_ex(const char *cursor,
                                              int bool_as_int,
                                              int null_as_none_literal) {
    NNGDStringList pairs = {0};
    size_t capacity = 64U;
    size_t used = 0U;
    char *joined = NULL;
    size_t i;
    cursor = nn_gd_json_skip_space(cursor);
    if (cursor == NULL || *cursor != '{') {
        return nn_gd_strdup("");
    }
    ++cursor;
    for (;;) {
        char *key;
        char *value;
        char *pair;
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (*cursor == '}') {
            break;
        }
        key = nn_gd_json_dup_string_value_advance(&cursor);
        if (key == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL || *cursor != ':') {
            free(key);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        ++cursor;
        value = nn_gd_json_dup_scalar_tsv_value_at(&cursor,
                                                   bool_as_int,
                                                   null_as_none_literal);
        if (value == NULL) {
            free(key);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        pair = nn_gd_make_slot_pair(key, value);
        free(key);
        free(value);
        if (pair == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (nn_gd_string_list_add(&pairs, pair) != 0) {
            free(pair);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        free(pair);
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor == '}') {
            break;
        }
        nn_gd_string_list_free(&pairs);
        return NULL;
    }
    nn_gd_string_list_sort(&pairs);
    joined = (char *)malloc(capacity);
    if (joined == NULL) {
        nn_gd_string_list_free(&pairs);
        return NULL;
    }
    for (i = 0U; i < pairs.count; ++i) {
        if (i > 0U && nn_gd_json_append_char(&joined, &used, &capacity, '|') != 0) {
            free(joined);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (nn_gd_append_sanitized_value(&joined,
                                         &used,
                                         &capacity,
                                         pairs.items[i]) != 0) {
            free(joined);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
    }
    joined[used] = '\0';
    nn_gd_string_list_free(&pairs);
    return joined;
}

static char *nn_gd_json_dup_slot_tokens_at(const char *cursor) {
    return nn_gd_json_dup_slot_tokens_at_ex(cursor, 0, 1);
}

static char *nn_gd_json_dup_index_slot_tokens(const NNGDJsonIndex *index) {
    return nn_gd_json_dup_slot_tokens_at(nn_gd_json_index_find(index, "slot_values"));
}
