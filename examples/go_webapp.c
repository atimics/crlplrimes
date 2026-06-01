/**
 * go_webapp.c — Self-contained Go web app. Play 9x9 against the MCTS engine.
 *
 * Usage: ./go_webapp [--load checkpoint.nnckpt] [--port 8086]
 *
 * Serves an interactive Go board at http://127.0.0.1:8086
 */

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define BOARD_SIZE 9
#define CELLS (BOARD_SIZE * BOARD_SIZE)
#define MAX_MOVES (CELLS + 1)
#define BOARD_FEATURES (CELLS * 3)
#define SLOT_MOVE_ID    (BOARD_FEATURES)
#define SLOT_REWARD     (BOARD_FEATURES + 1)
#define FEATURES        (BOARD_FEATURES + 2)
#define HIDDEN0 128
#define HIDDEN1 64

#define MCTS_SIMULATIONS 25
#define MCTS_C_PUCT 1.4f
#define MCTS_MAX_KIDS 82
#define MCTS_MIN_EXPAND 2
#define TT_SIZE 8192

static const int DIRS[4] = {-BOARD_SIZE, -1, 1, BOARD_SIZE};

/* ── Go engine ───────────────────────────────────────────────────────── */

static uint64_t zobrist_side;

typedef struct {
    int8_t board[CELLS];
    int to_play, moves, passes;
    int8_t prev[CELLS];
    int last_move;
    uint64_t zhash;
} GoState;

static uint64_t zobrist[CELLS][3];
static bool z_init_done = false;

static void z_init(void) {
    if (z_init_done) return;
    unsigned int z_seed = 20260531u;
    for (int i = 0; i < CELLS; i++) {
        for (int c = 0; c < 3; c++) {
            z_seed = z_seed * 1103515245u + 12345u;
            uint64_t lo = ((uint64_t)z_seed) & 0xFFFFFFFFu;
            z_seed = z_seed * 1103515245u + 12345u;
            uint64_t hi = ((uint64_t)z_seed) & 0xFFFFFFFFu;
            zobrist[i][c] = (hi << 32) | lo;
        }
    }
    z_seed = z_seed * 1103515245u + 12345u;
    uint64_t lo = ((uint64_t)z_seed) & 0xFFFFFFFFu;
    z_seed = z_seed * 1103515245u + 12345u;
    uint64_t hi = ((uint64_t)z_seed) & 0xFFFFFFFFu;
    zobrist_side = (hi << 32) | lo;
    z_init_done = true;
}

static void go_init(GoState *s) {
    z_init();
    memset(s, 0, sizeof(*s)); s->to_play = 1; s->last_move = -1;
    s->zhash = zobrist_side;
}

static int xy(int x, int y) { return y * BOARD_SIZE + x; }
static int in_bounds(int i) { return i >= 0 && i < CELLS; }
static int on_left(int i)  { return i % BOARD_SIZE == 0; }
static int on_right(int i) { return i % BOARD_SIZE == BOARD_SIZE - 1; }

static void go_hash_update(GoState *s, int idx, int from, int to) {
    s->zhash ^= zobrist[idx][from]; s->zhash ^= zobrist[idx][to];
}

static void remove_group(GoState *s, int i, int8_t *vis) {
    int8_t c = s->board[i];
    s->board[i] = 0; vis[i] = 1;
    go_hash_update(s, i, c == 1 ? 1 : 2, 0);
    for (int d = 0; d < 4; d++) {
        int ni = i + DIRS[d];
        if ((d == 1 && on_left(i)) || (d == 2 && on_right(i))) continue;
        if (in_bounds(ni) && !vis[ni] && s->board[ni] == c)
            remove_group(s, ni, vis);
    }
}

static int has_liberty(GoState *s, int i, int8_t *vis, int8_t color) {
    if (!in_bounds(i) || vis[i]) return 0;
    if (s->board[i] == 0) return 1;
    if (s->board[i] != color) return 0;  /* opponent stone blocks search */
    vis[i] = 1;
    for (int d = 0; d < 4; d++) {
        int ni = i + DIRS[d];
        if ((d == 1 && on_left(i)) || (d == 2 && on_right(i))) continue;
        if (has_liberty(s, ni, vis, color)) return 1;
    }
    return 0;
}

static int group_dead(GoState *s, int i, int8_t color) {
    if (!in_bounds(i) || s->board[i] != color) return 0;
    int8_t vis[CELLS] = {0};
    return !has_liberty(s, i, vis, color);
}

static int count_liberties(GoState *s, int i) {
    if (!in_bounds(i) || s->board[i] == 0) return 0;
    int8_t vis[CELLS] = {0}, libs[CELLS] = {0};
    int8_t c = s->board[i];
    int q[CELLS], h = 0, t = 0; q[t++] = i; vis[i] = 1;
    int count = 0;
    while (h < t) {
        int idx = q[h++];
        for (int d = 0; d < 4; d++) {
            int ni = idx + DIRS[d];
            if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
            if (!in_bounds(ni)) continue;
            if (s->board[ni] == 0 && !libs[ni]) { libs[ni] = 1; count++; }
            else if (s->board[ni] == c && !vis[ni]) { vis[ni] = 1; q[t++] = ni; }
        }
    }
    return count;
}

static int is_atari(GoState *s, int i, int8_t color) { (void)color; return count_liberties(s, i) == 1; }

static int is_suicide(GoState *s, int idx) {
    GoState cp = *s; cp.board[idx] = cp.to_play;
    int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) return 0;
    }
    return group_dead(&cp, idx, cp.to_play);
}

static int is_self_atari(GoState *s, int idx) {
    GoState cp = *s; cp.board[idx] = cp.to_play;
    int opp = -cp.to_play, captured = 0;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) {
            captured = 1; int8_t vis[CELLS] = {0}; remove_group(&cp, ni, vis);
        }
    }
    if (captured) return 0;
    return count_liberties(&cp, idx) == 1;
}

static int move_captures(GoState *s, int idx) {
    int opp = -s->to_play, caps = 0;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == opp && is_atari(s, ni, opp)) caps++;
    }
    return caps;
}

static int move_saves_own(GoState *s, int idx) {
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == s->to_play && is_atari(s, ni, s->to_play)) return 1;
    }
    return 0;
}

static int is_tactical(GoState *s, int idx) {
    if (idx == CELLS) return 0;
    return move_captures(s, idx) > 0 || move_saves_own(s, idx);
}

static int is_legal(GoState *s, int idx) {
    if (idx == CELLS) return 1;
    if (!in_bounds(idx) || s->board[idx] != 0) return 0;
    if (is_suicide(s, idx)) return 0;
    GoState cp = *s; cp.board[idx] = cp.to_play;
    int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) {
            int8_t vis[CELLS] = {0}; remove_group(&cp, ni, vis);
        }
    }
    return memcmp(cp.board, s->prev, sizeof(s->prev)) != 0;
}

static void go_move(GoState *s, int idx) {
    s->last_move = idx;
    if (idx == CELLS) { s->passes++; s->to_play = -s->to_play; s->zhash ^= zobrist_side; return; }
    s->passes = 0;
    memcpy(s->prev, s->board, sizeof(s->board));
    go_hash_update(s, idx, 0, s->to_play == 1 ? 1 : 2);
    s->zhash ^= zobrist_side;
    s->board[idx] = s->to_play;
    int opp = -s->to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == opp && group_dead(s, ni, opp)) {
            int8_t vis[CELLS] = {0}; remove_group(s, ni, vis);
        }
    }
    s->to_play = -s->to_play; s->moves++;
}

static int go_over(GoState *s) { return s->passes >= 2 || s->moves >= 200; }

static float go_score(GoState *s) {
    int black = 0, white = 0;
    int8_t vis[CELLS] = {0};
    for (int i = 0; i < CELLS; i++) {
        if (s->board[i] == 1) black++;
        else if (s->board[i] == -1) white++;
        else if (!vis[i]) {
            int q[CELLS], h = 0, t = 0, hb = 0, hw = 0, cnt = 0;
            q[t++] = i; vis[i] = 1;
            while (h < t) {
                int idx = q[h++]; cnt++;
                for (int d = 0; d < 4; d++) {
                    int ni = idx + DIRS[d];
                    if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
                    if (!in_bounds(ni)) continue;
                    if (s->board[ni] == 1) hb = 1;
                    else if (s->board[ni] == -1) hw = 1;
                    else if (!vis[ni]) { vis[ni] = 1; q[t++] = ni; }
                }
            }
            if (hb && !hw) black += cnt;
            if (hw && !hb) white += cnt;
        }
    }
    return (float)(black - white);
}

static float go_stone_ratio(GoState *s) {
    int n = 0; for (int i = 0; i < CELLS; i++) if (s->board[i] != 0) n++;
    return (float)n / CELLS;
}

/* ── features ────────────────────────────────────────────────────────── */

static void go_features(GoState *s, nn_real row[FEATURES]) {
    memset(row, 0, BOARD_FEATURES * sizeof(nn_real));
    for (int i = 0; i < CELLS; i++) {
        int off = i * 3;
        if (s->board[i] == 1) row[off] = 1.0f;
        else if (s->board[i] == -1) row[off + 1] = 1.0f;
        else row[off + 2] = 1.0f;
    }
}

static void go_evaluate(NeuralNet *net, GoState *s, float policy[MAX_MOVES], float *value) {
    nn_real feats[FEATURES]; go_features(s, feats);
    float total = 0.0f;
    for (int m = 0; m < MAX_MOVES; m++) {
        if (!is_legal(s, m)) { policy[m] = 0.0f; continue; }
        nn_real mf[FEATURES]; memcpy(mf, feats, FEATURES * sizeof(nn_real));
        mf[SLOT_MOVE_ID] = (nn_real)m / MAX_MOVES;
        nn_real sc; nn_forward(net, mf, &sc);
        float sv = (float)sc; if (sv < 0.005f) sv = 0.005f;
        if (m < CELLS && is_self_atari(s, m)) sv *= 0.3f;
        policy[m] = sv; total += sv;
    }
    if (total > 0.0f) for (int m = 0; m < MAX_MOVES; m++) policy[m] /= total;
    nn_real vs; nn_forward(net, feats, &vs);
    *value = (float)vs * 2.0f - 1.0f;
}

static float policy_entropy(float policy[MAX_MOVES]) {
    float h = 0.0f; int n = 0;
    for (int m = 0; m < MAX_MOVES; m++)
        if (policy[m] > 0.0f) { h -= policy[m] * logf(policy[m] + 1e-12f); n++; }
    if (n <= 1) return 0.0f;
    return h / logf((float)n);
}

static bool in_local_mask(GoState *s, int idx) {
    if (s->last_move < 0 || s->last_move >= CELLS) return false;
    int lx = s->last_move % BOARD_SIZE, ly = s->last_move / BOARD_SIZE;
    int mx = idx % BOARD_SIZE, my = idx / BOARD_SIZE;
    return abs(lx - mx) <= 2 && abs(ly - my) <= 2;
}

/* ── RAVE ────────────────────────────────────────────────────────────── */

static float rave_wins[CELLS + 1], rave_visits[CELLS + 1];
static void rave_reset(void) { memset(rave_wins, 0, sizeof(rave_wins)); memset(rave_visits, 0, sizeof(rave_visits)); }
static float rave_q(int move) { return rave_visits[move] < 1.0f ? 0.0f : rave_wins[move] / rave_visits[move]; }

/* ── TT cache ────────────────────────────────────────────────────────── */

typedef struct { uint64_t hash; float policy[12]; int8_t moves[12]; int n; bool valid; } TTEntry;
static TTEntry tt[TT_SIZE];
static TTEntry *tt_lookup(uint64_t hash) { int idx = hash % TT_SIZE; return (tt[idx].valid && tt[idx].hash == hash) ? &tt[idx] : NULL; }
static void tt_store(uint64_t hash, float policy[MAX_MOVES]) {
    int idx = hash % TT_SIZE;
    if (tt[idx].valid && tt[idx].hash == hash) return;
    tt[idx].hash = hash; tt[idx].valid = true; tt[idx].n = 0;
    for (int m = 0; m < MAX_MOVES && tt[idx].n < 12; m++)
        if (policy[m] > 0.001f) { tt[idx].moves[tt[idx].n] = (int8_t)m; tt[idx].policy[tt[idx].n] = policy[m]; tt[idx].n++; }
}

/* ── MCTS ────────────────────────────────────────────────────────────── */

typedef struct GoNode {
    int move, to_play; struct GoNode *parent, *kids[MCTS_MAX_KIDS];
    int nkids; float prior[MCTS_MAX_KIDS], visits, total_val; bool expanded;
} GoNode;

static GoNode *gn_alloc(int move, int to_play, GoNode *parent) {
    GoNode *n = calloc(1, sizeof(GoNode)); if (!n) return NULL;
    n->move = move; n->to_play = to_play; n->parent = parent; return n;
}
static void gn_free(GoNode *n) { if (!n) return; for (int i = 0; i < n->nkids; i++) gn_free(n->kids[i]); free(n); }

static int gn_select(GoNode *n) {
    int best = 0; float best_score = -1e30f, sqrt_n = sqrtf(n->visits + 1e-8f);
    for (int i = 0; i < n->nkids; i++) {
        GoNode *c = n->kids[i];
        float q = c->visits > 0 ? c->total_val / c->visits : 0.0f;
        float r = rave_q(c->move);
        float rave_beta = rave_visits[c->move] / (rave_visits[c->move] + c->visits + 50.0f);
        float q_blend = (1.0f - rave_beta) * q + rave_beta * r;
        float u = MCTS_C_PUCT * n->prior[i] * sqrt_n / (1.0f + c->visits);
        if (q_blend + u > best_score) { best_score = q_blend + u; best = i; }
    }
    return best;
}

static uint64_t go_hash(GoState *s) { return s->zhash; }

static void go_mcts_search(NeuralNet *net, GoState *root_st, float probs[MAX_MOVES]) {
    GoNode *root = gn_alloc(-1, root_st->to_play, NULL);
    if (!root) return;
    rave_reset();
    for (int sim = 0; sim < MCTS_SIMULATIONS; sim++) {
        GoState s = *root_st; GoNode *node = root;
        while (node->expanded && node->nkids > 0 && !go_over(&s)) { node = node->kids[gn_select(node)]; go_move(&s, node->move); }
        float v;
        if (go_over(&s)) { float sc = go_score(&s); v = (sc > 0) ? 1.0f : (sc < 0) ? -1.0f : 0.0f; v *= (float)(-node->to_play); }
        else {
            float policy[MAX_MOVES]; uint64_t h = go_hash(&s); TTEntry *te = tt_lookup(h);
            if (te) { memset(policy, 0, sizeof(policy)); for (int i = 0; i < te->n; i++) policy[te->moves[i]] = te->policy[i];
                nn_real vs; nn_real feats[FEATURES]; go_features(&s, feats); nn_forward(net, feats, &vs); v = (float)vs * 2.0f - 1.0f; }
            else { go_evaluate(net, &s, policy, &v); tt_store(h, policy); }
            float phase = go_stone_ratio(&s), ent = policy_entropy(policy);
            int base_k = (phase < 0.25f) ? 12 : (phase > 0.75f) ? 4 : 8;
            int top_k = base_k + (int)(ent * 16.0f); if (top_k < MCTS_MIN_EXPAND) top_k = MCTS_MIN_EXPAND; if (top_k > 30) top_k = 30;
            typedef struct { int move; float prior; int tactical; } candidate_t;
            candidate_t cands[MAX_MOVES]; int nc = 0; float max_prior = 0.001f;
            for (int m = 0; m < MAX_MOVES; m++) {
                if (!is_legal(&s, m)) continue;
                int tactical = is_tactical(&s, m); float prior = policy[m];
                if (tactical && prior < 0.01f) prior = 0.01f;
                if (in_local_mask(&s, m)) prior *= 1.3f;
                if (prior > max_prior) max_prior = prior;
                cands[nc].move = m; cands[nc].prior = prior; cands[nc].tactical = tactical; nc++;
            }
            for (int i = 0; i < nc; i++) for (int j = i+1; j < nc; j++) if (cands[j].prior > cands[i].prior) { candidate_t t = cands[i]; cands[i] = cands[j]; cands[j] = t; }
            bool expanded_move[MAX_MOVES] = {false}; node->nkids = 0; float prior_threshold = max_prior * 0.15f;
            for (int i = 0; i < nc && node->nkids < top_k; i++)
                if (cands[i].tactical || cands[i].prior >= prior_threshold) {
                    int m = cands[i].move; GoNode *kid = gn_alloc(m, -node->to_play, node);
                    if (!kid) break; node->kids[node->nkids]=kid; node->prior[node->nkids]=cands[i].prior; node->nkids++; expanded_move[m]=true;
                }
            for (int i = 0; i < nc; i++)
                if (!expanded_move[cands[i].move] && cands[i].tactical && node->nkids < MCTS_MAX_KIDS) {
                    int m = cands[i].move; GoNode *kid = gn_alloc(m, -node->to_play, node);
                    if (!kid) break; node->kids[node->nkids]=kid; node->prior[node->nkids]=cands[i].prior; node->nkids++;
                }
            node->expanded = true;
        }
        while (node) {
            node->visits += 1.0f; node->total_val += v;
            if (v < 0.0f && node->parent && node->move >= 0) { rave_wins[node->move] += 1.0f; rave_visits[node->move] += 1.0f; }
            else if (node->parent && node->move >= 0) rave_visits[node->move] += 1.0f;
            v = -v; node = node->parent;
        }
    }
    float total = 0.0f; for (int i = 0; i < root->nkids; i++) { probs[root->kids[i]->move] = root->kids[i]->visits; total += root->kids[i]->visits; }
    if (total > 0.0f) for (int m = 0; m < MAX_MOVES; m++) probs[m] /= total;
    gn_free(root);
}

static int go_mcts_move(NeuralNet *net, GoState *s) {
    float probs[MAX_MOVES] = {0}; go_mcts_search(net, s, probs);
    int best = CELLS; for (int m = 0; m < CELLS; m++) if (probs[m] > probs[best]) best = m;
    return best;
}



/* ── board JSON helper ───────────────────────────────────────────────── */

static int build_board_json(GoState *s, char *buf, int bufsz) {
    int off = snprintf(buf, bufsz, "\"board\":[");
    for (int i = 0; i < CELLS; i++) {
        off += snprintf(buf + off, bufsz - off, "%s%d", i ? "," : "", s->board[i]);
    }
    off += snprintf(buf + off, bufsz - off, "],\"last\":%d", s->last_move);
    return off;
}

/* ── legal moves helper ──────────────────────────────────────────────── */

static int build_legal_json(GoState *s, char *buf, int bufsz) {
    int off = 0;
    off += snprintf(buf + off, bufsz - off, "\"legal\":[");
    int first = 1;
    for (int m = 0; m < CELLS; m++) {
        if (is_legal(s, m)) {
            off += snprintf(buf + off, bufsz - off, "%s%d", first ? "" : ",", m);
            first = 0;
        }
    }
    if (is_legal(s, CELLS)) {
        off += snprintf(buf + off, bufsz - off, "%s%d", first ? "" : ",", CELLS);
    }
    off += snprintf(buf + off, bufsz - off, "]");
    return off;
}

/* ── HTTP ────────────────────────────────────────────────────────────── */

static char *http_200 = "HTTP/1.1 200 OK\r\n";
static char *http_json = "Content-Type: application/json\r\n";
static char *http_html = "Content-Type: text/html; charset=utf-8\r\n";

static void send_response(int fd, const char *ctype, const char *body, int len) {
    char hdr[256];
    int hl = snprintf(hdr, sizeof(hdr), "%s%sContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                      http_200, ctype, len);
    write(fd, hdr, hl);
    write(fd, body, len);
}

static int read_request(int fd, char *method, char *path, int max_path __attribute__((unused))) {
    char buf[2048];
    int n = (int)read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) return -1;
    buf[n] = '\0';
    sscanf(buf, "%s %s", method, path);
    return 0;
}

/* ── HTML page ───────────────────────────────────────────────────────── */

static const char *html =
"<!DOCTYPE html><html><head><meta charset=utf-8><title>Go</title><style>"
"*{margin:0;padding:0;box-sizing:border-box}"
"body{font-family:system-ui;background:#0f0f1a;color:#c8d6e5;display:flex;flex-direction:column;align-items:center;padding:16px}"
"h1{font-size:.9rem;margin-bottom:8px}"
"#g{display:grid;grid-template-columns:repeat(9,38px);grid-template-rows:repeat(9,38px);gap:1px;background:#1a1a33;border:2px solid #2a2a44;border-radius:3px}"
".c{width:38px;height:38px;background:#141428;display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative}"
".c:hover{background:#1e1e3e}"
".c.illegal{cursor:default}"
".s{width:28px;height:28px;border-radius:50%;position:absolute}"
".b{background:radial-gradient(circle at 36% 36%,#555,#111);box-shadow:0 1px 3px rgba(0,0,0,.5)}"
".w{background:radial-gradient(circle at 36% 36%,#fff,#bbb);box-shadow:0 1px 3px rgba(0,0,0,.3)}"
".hl{width:28px;height:28px;border-radius:50%;background:#5a4a1a;position:absolute;opacity:.5}"
"#t{margin:8px 0;font-size:.8rem;color:#fc3;min-height:20px}"
"button{background:#1e1e3a;color:#c8d6e5;border:1px solid #334;padding:4px 14px;cursor:pointer;border-radius:3px;font:inherit;font-size:.75rem;margin-top:4px}"
"button:hover{background:#2a2a55}"
"</style></head><body>"
"<h1>MCTS Go 9x9</h1>"
"<div id=t>Your turn</div>"
"<div id=g></div>"
"<button onclick=\"fetch('/reset',{method:'POST'}).then(()=>location.reload())\">New Game</button>"
"<script>"
"let S=9,b=Array(S*S).fill(0),t=1,go=false;"
"function r(){"
" let h='';"
" for(let y=0;y<S;y++)for(let x=0;x<S;x++){"
"  let i=y*S+x,v=b[i];"
"  let inner=v?('<div class=\"s '+(v==1?'b':'w')+'\"></div>'):'';"
"  h+='<div class=c data-i='+i+' id=\"c'+i+'\">'+inner+'</div>';"
" }"
" document.getElementById('g').innerHTML=h;"
"}"
"document.getElementById('g').addEventListener('mouseover',function(e){"
" let el=e.target;while(el&&!el.dataset.i)el=el.parentElement;"
" if(!el||go||t!=1)return;let i=parseInt(el.dataset.i);"
" if(b[i]!=0)return;"
" if(!el.querySelector('.hl')){let d=document.createElement('div');d.className='hl';el.appendChild(d);}"
"});"
"document.getElementById('g').addEventListener('mouseout',function(e){"
" let el=e.target;while(el&&!el.dataset.i)el=el.parentElement;"
" if(!el)return;let hl=el.querySelector('.hl');if(hl)hl.remove();"
"});"
"document.getElementById('g').addEventListener('click',function(e){"
" let el=e.target;while(el&&!el.dataset.i)el=el.parentElement;"
" if(!el||go||t!=1)return;let i=parseInt(el.dataset.i);"
" if(b[i]!=0)return;"
" b[i]=1;r();t=-1;"
" document.getElementById('t').textContent='Engine thinking...';"
" fetch('/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x:i%S,y:Math.floor(i/S)})})"
"  .then(r=>r.json()).then(d=>{"
"   if(d.board){b=d.board;r()}"
"   if(d.game_over){go=true;document.getElementById('t').textContent=d.result}"
"   else if(d.move==-1){go=true;document.getElementById('t').textContent='Engine passes'}"
"   else{t=1;document.getElementById('t').textContent='Your turn'}"
"  }).catch(e=>{t=1;document.getElementById('t').textContent='Error'})"
"});"
"r();"
"</script></body></html>";

/* ── JSON helpers ────────────────────────────────────────────────────── */

static int json_int(const char *json, const char *key) {
    char pattern[64]; snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    return p ? atoi(p + strlen(pattern)) : 0;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *load_path = NULL;
    int port = 8086;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0 && i+1 < argc) load_path = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i+1 < argc) port = atoi(argv[++i]);
    }

    const size_t layers[] = {FEATURES, HIDDEN0, HIDDEN1, 1};
    NNConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.layer_sizes = layers; cfg.layer_count = 4; cfg.seed = (unsigned)time(NULL);
    cfg.hidden_activation = NN_ACTIVATION_RELU; cfg.output_activation = NN_ACTIVATION_SIGMOID; cfg.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&cfg);
    if (!net) { fprintf(stderr, "create failed\n"); return 1; }
    if (load_path) {
        NNCheckpointMeta meta; NeuralNet *loaded = nn_load_checkpoint(load_path, &meta);
        if (loaded && nn_input_size(loaded) == FEATURES && nn_output_size(loaded) == 1) { nn_free(net); net = loaded; fprintf(stderr, "loaded %s\n", load_path); }
        else if (loaded) { nn_free(loaded); fprintf(stderr, "dim mismatch, fresh net\n"); }
        else fprintf(stderr, "load failed, fresh net\n");
    }

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port), .sin_addr.s_addr = INADDR_ANY};
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    listen(srv, 8);
    fprintf(stderr, "http://127.0.0.1:%d\n", port);

    GoState state; go_init(&state);

    for (;;) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) continue;
        char method[16] = {0}, path[256] = {0};
        if (read_request(fd, method, path, sizeof(path)) < 0) { close(fd); continue; }

        if (strcmp(method, "OPTIONS") == 0) {
            char *hdr = "HTTP/1.1 204 No Content\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Content-Type\r\n"
                        "Content-Length: 0\r\n"
                        "Connection: close\r\n\r\n";
            write(fd, hdr, (int)strlen(hdr));
        } else if (strcmp(path, "/") == 0) {
            send_response(fd, http_html, html, (int)strlen(html));
        } else if (strcmp(path, "/reset") == 0 && strcmp(method, "POST") == 0) {
            go_init(&state);
            send_response(fd, http_json, "{\"ok\":1}", 8);
        } else if (strcmp(path, "/move") == 0 && strcmp(method, "POST") == 0) {
            char body[4096] = {0};
            read(fd, body, sizeof(body) - 1);
            char resp[4096];
            int off = 0;

            if (go_over(&state)) {
                float sc = go_score(&state);
                off = snprintf(resp, sizeof(resp), "{\"game_over\":true,\"result\":\"%s\"}", sc > 0 ? "Black wins" : sc < 0 ? "White wins" : "Draw");
            } else if (state.to_play != 1) {
                off = snprintf(resp, sizeof(resp), "{\"error\":\"not your turn\"}");
            } else {
                int x = json_int(body, "x");
                int y = json_int(body, "y");
                int move = (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) ? xy(x, y) : -1;
                if (move < 0 || !is_legal(&state, move)) {
                    off = snprintf(resp, sizeof(resp), "{\"error\":\"illegal\",");
                    off += build_legal_json(&state, resp + off, sizeof(resp) - off);
                    off += snprintf(resp + off, sizeof(resp) - off, "}");
                } else {
                    go_move(&state, move);
                    if (go_over(&state)) {
                        float sc = go_score(&state);
                        off = snprintf(resp, sizeof(resp), "{\"game_over\":true,\"move\":-1,\"result\":\"%s\"}", sc > 0 ? "Black wins" : "Draw");
                    } else {
                        int eng = go_mcts_move(net, &state);
                        go_move(&state, eng);
                        if (go_over(&state)) {
                            float sc = go_score(&state);
                            if (eng == CELLS)
                                off = snprintf(resp, sizeof(resp), "{\"move\":-1,\"result\":\"%s\"}", sc > 0 ? "Black wins" : sc < 0 ? "White wins" : "Draw");
                            else
                                off = snprintf(resp, sizeof(resp), "{\"move\":%d,\"x\":%d,\"y\":%d,\"result\":\"%s\"}",
                                         eng, eng % BOARD_SIZE, eng / BOARD_SIZE, sc > 0 ? "Black wins" : sc < 0 ? "White wins" : "Draw");
                        } else {
                            if (eng == CELLS)
                                off = snprintf(resp, sizeof(resp), "{\"move\":-1,");
                            else
                                off = snprintf(resp, sizeof(resp), "{\"move\":%d,\"x\":%d,\"y\":%d,", eng, eng % BOARD_SIZE, eng / BOARD_SIZE);
                            off += build_board_json(&state, resp + off, sizeof(resp) - off);
                            off += snprintf(resp + off, sizeof(resp) - off, ",");
                            off += build_legal_json(&state, resp + off, sizeof(resp) - off);
                            off += snprintf(resp + off, sizeof(resp) - off, "}");
                        }
                    }
                }
            }
            if (off == 0) off = snprintf(resp, sizeof(resp), "{\"error\":\"unknown\"}");
            send_response(fd, http_json, resp, off);
        } else if (strcmp(path, "/legal") == 0) {
            char resp[4096];
            int off = snprintf(resp, sizeof(resp), "{");
            off += build_legal_json(&state, resp + off, sizeof(resp) - off);
            off += snprintf(resp + off, sizeof(resp) - off, "}");
            send_response(fd, http_json, resp, off);
        } else {
            char *nf = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            write(fd, nf, (int)strlen(nf));
        }
        close(fd);
    }
}

