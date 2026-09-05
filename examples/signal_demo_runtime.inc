/* Small POSIX runtime for the connected study. All commands use argv arrays. */
#include "nn.h"
#include "nn_json.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct DemoAllocation {
    void *pointer;
    struct DemoAllocation *next;
} DemoAllocation;
static DemoAllocation *demo_allocations;
static jmp_buf demo_failure;
static char demo_error[2048];
static double demo_phase_deadline;
static volatile sig_atomic_t demo_child;

static void stop_child(int signal_number)
{
    pid_t child = (pid_t)demo_child;
    if (child > 0) {
        kill(-child, SIGKILL);
        while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {}
    }
    _exit(128 + signal_number);
}
static void install_signal_cleanup(void)
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_child;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, NULL) || sigaction(SIGINT, &action, NULL)) {
        fputs("Cannot install child cleanup\n", stderr);
        exit(1);
    }
}

static void fail(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(demo_error, sizeof(demo_error), format, args);
    va_end(args);
    longjmp(demo_failure, 1);
}

static void *allocate(size_t size)
{
    DemoAllocation *item = malloc(sizeof(*item));
    void *pointer = calloc(size ? size : 1, 1);
    if (!item || !pointer) {
        free(item);
        free(pointer);
        fail("Allocation failed");
    }
    item->pointer = pointer;
    item->next = demo_allocations;
    demo_allocations = item;
    return pointer;
}

static void release_allocations(void)
{
    while (demo_allocations) {
        DemoAllocation *next = demo_allocations->next;
        free(demo_allocations->pointer);
        free(demo_allocations);
        demo_allocations = next;
    }
}

static char *text_copy(const char *text, size_t size)
{
    char *copy = allocate(size + 1);
    memcpy(copy, text, size);
    return copy;
}

static char *format(const char *pattern, ...)
{
    va_list args, copy;
    va_start(args, pattern);
    va_copy(copy, args);
    int size = vsnprintf(NULL, 0, pattern, copy);
    va_end(copy);
    if (size < 0) fail("Text formatting failed");
    char *result = allocate((size_t)size + 1);
    vsnprintf(result, (size_t)size + 1, pattern, args);
    va_end(args);
    return result;
}

typedef enum { J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT } JsonKind;
typedef struct Json Json;
typedef struct JsonItem {
    char *key;
    Json *value;
    struct JsonItem *next;
} JsonItem;
struct Json {
    JsonKind kind;
    double number;
    char *string;
    JsonItem *items;
    JsonItem *tail;
    size_t count;
};

static Json *json_new(JsonKind kind)
{
    Json *value = allocate(sizeof(*value));
    value->kind = kind;
    return value;
}
static Json *object(void) { return json_new(J_OBJECT); }
static Json *array(void) { return json_new(J_ARRAY); }
static Json *number(double value) { Json *j = json_new(J_NUMBER); j->number = value; return j; }
static Json *boolean(int value) { Json *j = json_new(J_BOOL); j->number = !!value; return j; }
static Json *string(const char *value)
{
    Json *j = json_new(J_STRING);
    j->string = text_copy(value, strlen(value));
    return j;
}
static Json *find(Json *value, const char *key)
{
    if (!value || value->kind != J_OBJECT) fail("Expected a JSON object for %s", key);
    for (JsonItem *i = value->items; i; i = i->next)
        if (!strcmp(i->key, key)) return i->value;
    return NULL;
}
static Json *get(Json *value, const char *key)
{
    Json *result = find(value, key);
    if (!result) fail("Missing JSON field: %s", key);
    return result;
}
static void put(Json *value, const char *key, Json *child)
{
    if (value->kind != J_OBJECT && value->kind != J_ARRAY) fail("Expected JSON container");
    if (value->kind == J_OBJECT) {
        for (JsonItem *i = value->items; i; i = i->next)
            if (!strcmp(i->key, key)) { i->value = child; return; }
    }
    JsonItem *item = allocate(sizeof(*item));
    item->key = key ? text_copy(key, strlen(key)) : NULL;
    item->value = child;
    if (value->tail) value->tail->next = item; else value->items = item;
    value->tail = item;
    value->count++;
}
static void append(Json *a, Json *value) { if (a->kind != J_ARRAY) fail("Expected JSON array"); put(a, NULL, value); }
static const char *str(Json *j) { if (!j || j->kind != J_STRING) fail("Expected JSON string"); return j->string; }
static double num(Json *j) { if (!j || j->kind != J_NUMBER) fail("Expected JSON number"); return j->number; }
static int truth(Json *j) { if (!j || j->kind != J_BOOL) fail("Expected JSON boolean"); return j->number != 0; }
static const char *field(Json *j, const char *key) { return str(get(j, key)); }
static double value(Json *j, const char *key) { return num(get(j, key)); }
static int equal(Json *a, Json *b)
{
    if (!a || !b || a->kind != b->kind) return 0;
    if (a->kind == J_STRING) return !strcmp(a->string, b->string);
    if (a->kind == J_NUMBER || a->kind == J_BOOL) return a->number == b->number;
    if (a->count != b->count) return 0;
    JsonItem *other = b->items;
    for (JsonItem *i = a->items; i; i = i->next) {
        if (!equal(i->value, a->kind == J_OBJECT ? find(b, i->key) : other->value)) return 0;
        other = other->next;
    }
    return 1;
}

static void json_ws(NNJsonCursor *p)
{
    while (p->cursor < p->end && strchr(" \t\r\n", *p->cursor)) ++p->cursor;
}
static int json_expect(NNJsonCursor *p, char expected)
{
    json_ws(p);
    if (p->cursor == p->end || *p->cursor != expected) return 0;
    ++p->cursor;
    return 1;
}
static void validate_utf8(const unsigned char *p, size_t size)
{
    for (size_t n = 0; n < size;) {
        unsigned ch = p[n++], code = ch, length = 0, minimum = 0;
        if (ch < 128) continue;
        if (ch >= 0xc2 && ch <= 0xdf) { length = 1; code = ch & 31; minimum = 0x80; }
        else if (ch >= 0xe0 && ch <= 0xef) { length = 2; code = ch & 15; minimum = 0x800; }
        else if (ch >= 0xf0 && ch <= 0xf4) { length = 3; code = ch & 7; minimum = 0x10000; }
        else fail("Invalid UTF-8 input");
        if (n + length > size) fail("Incomplete UTF-8 input");
        for (unsigned i = 0; i < length; ++i) {
            unsigned next = p[n++];
            if ((next & 0xc0) != 0x80) fail("Invalid UTF-8 continuation");
            code = (code << 6) | (next & 63);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) fail("Invalid UTF-8 code point");
    }
}

static unsigned unicode_hex(NNJsonCursor *p)
{
    unsigned code = 0;
    for (int n = 0; n < 4; ++n) {
        if (p->cursor == p->end || !isxdigit((unsigned char)*p->cursor)) fail("Invalid JSON Unicode escape");
        unsigned char ch = (unsigned char)*p->cursor++;
        code = code * 16 + (isdigit(ch) ? ch - '0' : (unsigned)tolower(ch) - 'a' + 10);
    }
    return code;
}
/* Braid text contains escaped CSV newlines. Preserve the decoded UTF-8 bytes. */
static char *parse_string(NNJsonCursor *p)
{
    if (!json_expect(p, '"')) fail("Expected JSON string");
    char *out = allocate((size_t)(p->end - p->cursor) + 1);
    size_t n = 0;
    while (p->cursor < p->end && *p->cursor != '"') {
        unsigned char ch = (unsigned char)*p->cursor++;
        if (ch < 32) fail("Invalid JSON control character");
        if (ch == '\\') {
            if (p->cursor == p->end) fail("Incomplete JSON escape");
            ch = (unsigned char)*p->cursor++;
            switch (ch) {
            case 'n': ch = '\n'; break;
            case 'r': ch = '\r'; break;
            case 't': ch = '\t'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case '"': case '\\': case '/': break;
            case 'u': {
                unsigned code = unicode_hex(p);
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (p->end - p->cursor < 2 || p->cursor[0] != '\\' || p->cursor[1] != 'u') fail("Incomplete JSON surrogate pair");
                    p->cursor += 2;
                    unsigned low = unicode_hex(p);
                    if (low < 0xdc00 || low > 0xdfff) fail("Invalid JSON surrogate pair");
                    code = 0x10000 + (code - 0xd800) * 1024 + low - 0xdc00;
                } else if (code >= 0xdc00 && code <= 0xdfff) fail("Invalid JSON surrogate");
                if (!code) fail("JSON strings require nonzero bytes");
                if (code < 0x80) out[n++] = (char)code;
                else if (code < 0x800) { out[n++] = (char)(0xc0 | (code >> 6)); out[n++] = (char)(0x80 | (code & 63)); }
                else if (code < 0x10000) { out[n++] = (char)(0xe0 | (code >> 12)); out[n++] = (char)(0x80 | ((code >> 6) & 63)); out[n++] = (char)(0x80 | (code & 63)); }
                else { out[n++] = (char)(0xf0 | (code >> 18)); out[n++] = (char)(0x80 | ((code >> 12) & 63)); out[n++] = (char)(0x80 | ((code >> 6) & 63)); out[n++] = (char)(0x80 | (code & 63)); }
                continue;
            }
            default: fail("Invalid JSON escape");
            }
        }
        out[n++] = (char)ch;
    }
    if (p->cursor == p->end) fail("Unterminated JSON string");
    ++p->cursor;
    return out;
}

static Json *parse_value(NNJsonCursor *p, unsigned depth)
{
    if (depth > 128) fail("JSON nesting exceeds 128 levels");
    json_ws(p);
    if (p->cursor == p->end) fail("Missing JSON value");
    char ch = *p->cursor;
    if (ch == '"') { Json *j = json_new(J_STRING); j->string = parse_string(p); return j; }
    if (ch == '[' || ch == '{') {
        Json *j = ch == '[' ? array() : object();
        char close = ch == '[' ? ']' : '}';
        ++p->cursor;
        json_ws(p);
        if (p->cursor < p->end && *p->cursor == close) { ++p->cursor; return j; }
        for (;;) {
            char *key = NULL;
            if (ch == '{') {
                key = parse_string(p);
                if (find(j, key)) fail("Repeated JSON key: %s", key);
                if (!json_expect(p, ':')) fail("Missing JSON colon");
            }
            put(j, key, parse_value(p, depth + 1));
            json_ws(p);
            if (p->cursor < p->end && *p->cursor == close) { ++p->cursor; return j; }
            if (!json_expect(p, ',')) fail("Missing JSON comma");
        }
    }
    const char *start = p->cursor;
    if (ch == '-' || isdigit((unsigned char)ch)) {
        if (*p->cursor == '-') ++p->cursor;
        if (p->cursor == p->end) fail("Invalid JSON number");
        if (*p->cursor == '0') ++p->cursor;
        else {
            if (*p->cursor < '1' || *p->cursor > '9') fail("Invalid JSON number");
            while (p->cursor < p->end && isdigit((unsigned char)*p->cursor)) ++p->cursor;
        }
        if (p->cursor < p->end && *p->cursor == '.') {
            ++p->cursor;
            if (p->cursor == p->end || !isdigit((unsigned char)*p->cursor)) fail("Invalid JSON fraction");
            while (p->cursor < p->end && isdigit((unsigned char)*p->cursor)) ++p->cursor;
        }
        if (p->cursor < p->end && (*p->cursor == 'e' || *p->cursor == 'E')) {
            ++p->cursor;
            if (p->cursor < p->end && (*p->cursor == '+' || *p->cursor == '-')) ++p->cursor;
            if (p->cursor == p->end || !isdigit((unsigned char)*p->cursor)) fail("Invalid JSON exponent");
            while (p->cursor < p->end && isdigit((unsigned char)*p->cursor)) ++p->cursor;
        }
        char *literal = text_copy(start, (size_t)(p->cursor - start));
        errno = 0;
        double n = strtod(literal, NULL);
        if (!isfinite(n) || errno == ERANGE) fail("Non-finite JSON number");
        return number(n);
    }
    const char *words[] = {"true", "false", "null"};
    for (int n = 0; n < 3; ++n) {
        size_t length = strlen(words[n]);
        if ((size_t)(p->end - p->cursor) >= length && !memcmp(p->cursor, words[n], length)) {
            p->cursor += length;
            return n == 2 ? json_new(J_NULL) : boolean(n == 0);
        }
    }
    fail("Invalid JSON value");
    return NULL;
}
static Json *parse_json(const char *text)
{
    validate_utf8((const unsigned char *)text, strlen(text));
    NNJsonCursor p = {text, text + strlen(text)};
    Json *j = parse_value(&p, 0);
    json_ws(&p);
    if (p.cursor != p.end) fail("Trailing JSON text");
    return j;
}
static char *read_stream(FILE *file)
{
    if (fseek(file, 0, SEEK_END)) fail("Cannot seek input");
    long size = ftell(file);
    if (size < 0 || size > 256 * 1024 * 1024) fail("Input exceeds 256 MiB");
    rewind(file);
    char *data = allocate((size_t)size + 1);
    if (fread(data, 1, (size_t)size, file) != (size_t)size) fail("Cannot read input");
    if (memchr(data, 0, (size_t)size)) fail("Embedded NUL in text input");
    return data;
}
static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) fail("Cannot read %s: %s", path, strerror(errno));
    char *data = read_stream(f);
    if (fclose(f)) fail("Cannot close %s", path);
    return data;
}
static Json *read_json(const char *path) { return parse_json(read_text(path)); }
static void emit_string(FILE *f, const char *text)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') fprintf(f, "\\%c", *p);
        else if (*p < 32) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}
static void emit_json(FILE *f, Json *j)
{
    switch (j->kind) {
    case J_NULL: fputs("null", f); break;
    case J_BOOL: fputs(j->number ? "true" : "false", f); break;
    case J_NUMBER:
        if (!isfinite(j->number)) fail("Non-finite JSON output");
        fprintf(f, "%.17g", j->number); break;
    case J_STRING: emit_string(f, j->string); break;
    default:
        fputc(j->kind == J_OBJECT ? '{' : '[', f);
        for (JsonItem *i = j->items; i; i = i->next) {
            if (i != j->items) fputc(',', f);
            if (j->kind == J_OBJECT) { emit_string(f, i->key); fputc(':', f); }
            emit_json(f, i->value);
        }
        fputc(j->kind == J_OBJECT ? '}' : ']', f);
    }
}
static void write_json(const char *path, Json *j)
{
    FILE *f = fopen(path, "wb");
    if (!f) fail("Cannot write %s: %s", path, strerror(errno));
    emit_json(f, j);
    fputc('\n', f);
    int bad = ferror(f);
    if (fclose(f) || bad) fail("Cannot finish %s", path);
}
static void write_text(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");
    if (!f) fail("Cannot write %s", path);
    int bad = fputs(text, f) == EOF;
    if (fclose(f) || bad) fail("Cannot finish %s", path);
}
static int exists(const char *path) { struct stat s; return lstat(path, &s) == 0; }
static char *join(const char *a, const char *b) { return format("%s/%s", a, b); }
static char *parent(const char *path)
{
    char *result = text_copy(path, strlen(path));
    char *slash = strrchr(result, '/');
    if (!slash) return text_copy(".", 1);
    if (slash == result) slash[1] = 0; else *slash = 0;
    return result;
}
/* Resolve existing parents as well as the final component for fresh outputs. */
static char *absolute(const char *path)
{
    char resolved[PATH_MAX];
    if (realpath(path, resolved)) return text_copy(resolved, strlen(resolved));
    if (errno != ENOENT) fail("Cannot resolve %s: %s", path, strerror(errno));
    char *trimmed = text_copy(path, strlen(path));
    size_t length = strlen(trimmed);
    while (length > 1 && trimmed[length - 1] == '/') trimmed[--length] = 0;
    char *base = strrchr(trimmed, '/');
    const char *name = base ? base + 1 : trimmed;
    char *up = absolute(parent(trimmed));
    if (!strcmp(name, ".")) return up;
    if (!strcmp(name, "..")) return parent(up);
    return join(up, name);
}
static void mkdirs(const char *path)
{
    if (exists(path)) {
        struct stat s;
        if (stat(path, &s) || !S_ISDIR(s.st_mode)) fail("Expected directory: %s", path);
        return;
    }
    mkdirs(parent(path));
    if (mkdir(path, 0777)) fail("Cannot create %s: %s", path, strerror(errno));
}
static char *digest(const char *path)
{
    char *hash = allocate(65);
    if (!nn_hash_checkpoint_hex(path, hash)) fail("Cannot hash %s", path);
    return hash;
}
static char *which(const char *name)
{
    const char *env = getenv("PATH");
    if (!env) fail("PATH is required to locate %s", name);
    char *paths = text_copy(env, strlen(env)), *saved = NULL;
    for (char *part = strtok_r(paths, ":", &saved); part; part = strtok_r(NULL, ":", &saved)) {
        char *path = join(part, name);
        if (!access(path, X_OK)) return absolute(path);
    }
    fail("Required executable is missing: %s", name);
    return NULL;
}
static double seconds_now(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("Cannot read monotonic clock");
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
static char *command(const char *cwd, const char *log, int capture, int timeout,
                     int simulation, const char *const *argv)
{
    FILE *output = log ? fopen(log, "wb") : capture ? tmpfile() : NULL;
    if ((log || capture) && !output) fail("Cannot open command output");
    const char *launch[128];
    if (simulation) {
        struct rlimit limit;
        if (getrlimit(RLIMIT_STACK, &limit)) fail("Cannot inspect stack limit");
#ifdef __APPLE__
        rlim_t requested = 65520u * 1024u;
#else
        rlim_t requested = 65536u * 1024u;
#endif
        if (limit.rlim_max != RLIM_INFINITY && requested > limit.rlim_max) requested = limit.rlim_max;
        if (requested < 32u * 1024u * 1024u) fail("Signal trace generation needs a 32 MiB or larger stack limit");
        launch[0] = "/bin/sh"; launch[1] = "-c";
        launch[2] = "ulimit -s \"$1\" || exit; shift; exec \"$@\"";
        launch[3] = "signal-demo-stack"; launch[4] = format("%llu", (unsigned long long)(requested / 1024));
        size_t n = 0;
        for (; argv[n]; ++n) { if (n + 6 >= 128) fail("Command has too many arguments"); launch[n + 5] = argv[n]; }
        launch[n + 5] = NULL;
        argv = launch;
    }
    fflush(NULL);
    pid_t child = fork();
    if (child < 0) fail("Cannot start %s", argv[0]);
    if (!child) {
        if (setpgid(0, 0) || chdir(cwd)) _exit(126);
        if (dup2(output ? fileno(output) : STDERR_FILENO, STDOUT_FILENO) < 0) _exit(126);
        if (log && dup2(fileno(output), STDERR_FILENO) < 0) _exit(126);
        execvp(argv[0], (char *const *)argv);
        fprintf(stderr, "Cannot execute %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    (void)setpgid(child, child);
    demo_child = (sig_atomic_t)child;
    double deadline = seconds_now() + timeout;
    if (demo_phase_deadline && deadline > demo_phase_deadline) deadline = demo_phase_deadline;
    int status = 0, expired = 0;
    for (;;) {
        pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR) fail("Cannot wait for %s", argv[0]);
        if (seconds_now() >= deadline) {
            kill(-child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            expired = 1;
            break;
        }
        struct timespec pause = {0, 100000000};
        nanosleep(&pause, NULL);
    }
    demo_child = 0;
    char *result = capture && !log ? read_stream(output) : text_copy("", 0);
    if (output) fclose(output);
    if (expired) fail("Command timed out: %s", argv[0]);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) fail("Command failed: %s (status %d)%s%s", argv[0], WIFEXITED(status) ? WEXITSTATUS(status) : -1, log ? "; log: " : "", log ? log : "");
    return result;
}
#define CMD(cwd, log, capture, timeout, simulation, ...) command(cwd, log, capture, timeout, simulation, (const char *const[]){__VA_ARGS__, NULL})
static char *trim(char *text)
{
    size_t n = strlen(text);
    while (n && isspace((unsigned char)text[n - 1])) text[--n] = 0;
    while (isspace((unsigned char)*text)) ++text;
    return text;
}
