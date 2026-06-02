#ifndef GO_RULES_H
#define GO_RULES_H
/* Shared Go rules engine for crlplrimes experiments.
 * Board: 9x9, 82 moves (81 intersections + pass), territory scoring.
 * Include this header in any experiment that needs Go position generation. */

#include <stdint.h>
#include <string.h>

#define GO_BS 9
#define GO_CL (GO_BS * GO_BS)
#define GO_MM (GO_CL + 1)
#define GO_MG 150
static const int GO_DR[4] = {-GO_BS, -1, 1, GO_BS};

typedef struct {
    int8_t board[GO_CL];
    int to_play, moves, passes;
    int8_t prev[GO_CL];
} GoBoard;

static inline int go_ol(int i)  { return i % GO_BS == 0; }
static inline int go_or(int i)  { return i % GO_BS == GO_BS - 1; }
static inline int go_ib(int i)  { return i >= 0 && i < GO_CL; }

static inline void go_init(GoBoard *b) {
    memset(b, 0, sizeof(*b)); b->to_play = 1;
}

static int go_lib(const GoBoard *b, int i) {
    int8_t c = b->board[i]; if (!c) return 0;
    int8_t lb[GO_CL] = {0}, vs[GO_CL] = {0};
    int q[GO_CL], h = 0, t = 0, cn = 0;
    q[t++] = i; vs[i] = 1;
    while (h < t) { int x = q[h++];
        for (int d = 0; d < 4; d++) { int n = x + GO_DR[d];
            if ((d == 1 && go_ol(x)) || (d == 2 && go_or(x))) continue;
            if (!go_ib(n)) continue;
            if (!b->board[n] && !lb[n]) { lb[n] = 1; cn++; }
            else if (b->board[n] == c && !vs[n]) { vs[n] = 1; q[t++] = n; }}}
    return cn;
}

static void go_rm(GoBoard *b, int i) {
    int8_t c = b->board[i], vs[GO_CL] = {0};
    int q[GO_CL], h = 0, t = 0;
    q[t++] = i; vs[i] = 1;
    while (h < t) { int x = q[h++]; b->board[x] = 0;
        for (int d = 0; d < 4; d++) { int n = x + GO_DR[d];
            if ((d == 1 && go_ol(x)) || (d == 2 && go_or(x))) continue;
            if (go_ib(n) && !vs[n] && b->board[n] == c)
                { vs[n] = 1; q[t++] = n; }}}
}

static inline int go_dead(const GoBoard *b, int i) {
    return b->board[i] && go_lib(b, i) == 0;
}

static int go_legal(const GoBoard *b, int idx) {
    if (idx == GO_CL) return 1;
    if (!go_ib(idx) || b->board[idx]) return 0;
    GoBoard cp = *b; cp.board[idx] = cp.to_play;
    int o = -cp.to_play;
    for (int d = 0; d < 4; d++) { int n = idx + GO_DR[d];
        if ((d == 1 && go_ol(idx)) || (d == 2 && go_or(idx))) continue;
        if (go_ib(n) && cp.board[n] == o && go_dead(&cp, n)) go_rm(&cp, n); }
    if (go_dead(&cp, idx)) return 0;
    if (!memcmp(cp.board, b->prev, sizeof(b->prev))) return 0;
    return 1;
}

static void go_move(GoBoard *b, int idx) {
    if (idx == GO_CL) { b->passes++; b->to_play = -b->to_play; return; }
    memcpy(b->prev, b->board, sizeof(b->board)); b->passes = 0;
    b->board[idx] = b->to_play; int o = -b->to_play;
    for (int d = 0; d < 4; d++) { int n = idx + GO_DR[d];
        if ((d == 1 && go_ol(idx)) || (d == 2 && go_or(idx))) continue;
        if (go_ib(n) && b->board[n] == o && go_dead(b, n)) go_rm(b, n); }
    b->to_play = -b->to_play; b->moves++;
}

static inline int go_over(const GoBoard *b) {
    return b->passes >= 2 || b->moves >= GO_MG;
}

static float go_score(const GoBoard *b) {
    int8_t ow[GO_CL] = {0}, vs[GO_CL] = {0};
    for (int i = 0; i < GO_CL; i++) {
        if (b->board[i]) { ow[i] = b->board[i]; continue; }
        if (vs[i]) continue;
        int q[GO_CL], h = 0, t = 0, hb = 0, hw = 0;
        q[t++] = i; vs[i] = 1;
        while (h < t) { int x = q[h++];
            for (int d = 0; d < 4; d++) { int n = x + GO_DR[d];
                if ((d == 1 && go_ol(x)) || (d == 2 && go_or(x))) continue;
                if (!go_ib(n)) continue;
                if (b->board[n] == 1) hb = 1;
                else if (b->board[n] == -1) hw = 1;
                else if (!vs[n]) { vs[n] = 1; q[t++] = n; }}}
        int8_t o = (hb && !hw) ? 1 : (!hb && hw) ? -1 : 0;
        memset(vs, 0, sizeof(vs)); h = 0; t = 0; q[t++] = i; vs[i] = 1;
        while (h < t) { int x = q[h++];
            if (!b->board[x]) ow[x] = o;
            for (int d = 0; d < 4; d++) { int n = x + GO_DR[d];
                if ((d == 1 && go_ol(x)) || (d == 2 && go_or(x))) continue;
                if (go_ib(n) && !b->board[n] && !vs[n]) { vs[n] = 1; q[t++] = n; }}}
    }
    int bk = 0, wh = 0;
    for (int i = 0; i < GO_CL; i++) { if (ow[i] == 1) bk++; if (ow[i] == -1) wh++; }
    return (float)(bk - wh);
}

/* Deterministic replay: apply move, random rollout to terminal.
   Returns score from move-maker's perspective. */
static unsigned go_rng;
static inline void go_sd(unsigned s) { go_rng = s; }
static inline unsigned go_nx(void) { go_rng = go_rng * 1103515245u + 12345u; return go_rng; }

static float go_replay(const GoBoard *st, int mv, unsigned seed) {
    GoBoard b = *st; go_move(&b, mv); int mover = -b.to_play;
    go_sd(seed);
    while (!go_over(&b)) {
        int lg[GO_MM], nl = 0;
        for (int m = 0; m < GO_MM; m++) if (go_legal(&b, m)) lg[nl++] = m;
        if (!nl) break;
        go_move(&b, lg[go_nx() % nl]);
    }
    float sc = go_score(&b);
    return mover == 1 ? sc : -sc;
}

/* Self-atari, capture, save helpers (used by feature extraction) */
static int go_sa(const GoBoard *b, int idx) {
    GoBoard cp = *b; cp.board[idx] = cp.to_play; int o = -cp.to_play;
    for (int d = 0; d < 4; d++) { int n = idx + GO_DR[d];
        if ((d == 1 && go_ol(idx)) || (d == 2 && go_or(idx))) continue;
        if (go_ib(n) && cp.board[n] == o && go_dead(&cp, n)) go_rm(&cp, n); }
    return go_lib(&cp, idx) == 1;
}
static int go_cap(const GoBoard *b, int idx) {
    int o = -b->to_play;
    for (int d = 0; d < 4; d++) { int n = idx + GO_DR[d];
        if ((d == 1 && go_ol(idx)) || (d == 2 && go_or(idx))) continue;
        if (go_ib(n) && b->board[n] == o && go_dead(b, n)) return 1; }
    return 0;
}
static int go_save(const GoBoard *b, int idx) {
    for (int d = 0; d < 4; d++) { int n = idx + GO_DR[d];
        if ((d == 1 && go_ol(idx)) || (d == 2 && go_or(idx))) continue;
        if (go_ib(n) && b->board[n] == b->to_play && go_lib(b, n) == 1) return 1; }
    return 0;
}

#endif
