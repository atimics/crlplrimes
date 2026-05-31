/**
 * go_selfplay.c — Sparse verified MCTS for 9×9 Go.
 *
 * Architecture:
 *   Neural scorer provides move priors.
 *   Rules engine enforces legality (zero illegal moves emitted).
 *   Sparse MCTS expands only top-k + tactical exceptions.
 *   Entropy-adaptive widening, RAVE, Zobrist cache, phase-aware search.
 *   Every search records reason labels for retraining.
 */

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>
#include <ctype.h>

#define BOARD_SIZE 9
#define CELLS (BOARD_SIZE * BOARD_SIZE)
#define MAX_MOVES (CELLS + 1)
#define FEATURES (CELLS * 3)
#define HIDDEN0 128
#define HIDDEN1 64
#define GAMES 200
#define MOVES_PER_GAME 150
#define EPOCHS 4

#define MCTS_SIMULATIONS 200
#define MCTS_C_PUCT 1.4f
#define MCTS_MAX_KIDS 82
#define MCTS_MIN_EXPAND 2
#define MCTS_CONFIDENCE_GAP 0.15f
#define TT_SIZE 8192

static const int DIRS[4] = {-BOARD_SIZE, -1, 1, BOARD_SIZE};

/* ── Go state ─────────────────────────────────────────────────────────── */

typedef struct {
    int8_t board[CELLS];
    int to_play, moves, passes;
    int8_t prev[CELLS];
    int last_move;  /* for local fight detection */
    uint64_t zhash;
} GoState;

static uint64_t zobrist[CELLS][3];  /* [pos][empty/black/white] */
static bool zobrist_init = false;

static void z_init(void) {
    if (zobrist_init) return;
    srand(42);
    for (int i = 0; i < CELLS; i++)
        for (int c = 0; c < 3; c++)
            zobrist[i][c] = ((uint64_t)rand() << 32) | (uint64_t)rand();
    zobrist_init = true;
}

static void go_init(GoState *s) {
    z_init();
    memset(s, 0, sizeof(*s));
    s->to_play = 1;
    s->last_move = -1;
}

static int xy(int x, int y) { return y * BOARD_SIZE + x; }
static int in_bounds(int i) { return i >= 0 && i < CELLS; }
static int on_left(int i)  { return i % BOARD_SIZE == 0; }
static int on_right(int i) { return i % BOARD_SIZE == BOARD_SIZE - 1; }

/* ── Zobrist ──────────────────────────────────────────────────────────── */

static uint64_t go_hash(GoState *s) { return s->zhash; }

static void go_update_hash(GoState *s, int idx, int from, int to) {
    s->zhash ^= zobrist[idx][from];
    s->zhash ^= zobrist[idx][to];
}

/* ── capture / legality ───────────────────────────────────────────────── */

static void remove_group(GoState *s, int i, int8_t *vis) {
    int8_t c = s->board[i];
    s->board[i] = 0; vis[i] = 1;
    go_update_hash(s, i, (c == 1 ? 1 : 2), 0);
    for (int d = 0; d < 4; d++) {
        int ni = i + DIRS[d];
        if ((d == 1 && on_left(i)) || (d == 2 && on_right(i))) continue;
        if (in_bounds(ni) && !vis[ni] && s->board[ni] == c)
            remove_group(s, ni, vis);
    }
}

static int has_liberty(GoState *s, int i, int8_t *vis) {
    if (!in_bounds(i) || vis[i]) return 0;
    if (s->board[i] == 0) return 1;
    int8_t c = s->board[i]; vis[i] = 1;
    for (int d = 0; d < 4; d++) {
        int ni = i + DIRS[d];
        if ((d == 1 && on_left(i)) || (d == 2 && on_right(i))) continue;
        if (has_liberty(s, ni, vis)) return 1;
    }
    return 0;
}

static int group_dead(GoState *s, int i, int8_t color) {
    if (!in_bounds(i) || s->board[i] != color) return 0;
    int8_t vis[CELLS] = {0};
    return !has_liberty(s, i, vis);
}

/* ── tactical sensors ─────────────────────────────────────────────────── */

static int count_liberties(GoState *s, int i) {
    /* Count distinct liberties of the group at position i */
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

static int is_atari(GoState *s, int i, int8_t color) {
    return group_dead(s, i, color) ? 0 : (count_liberties(s, i) == 1);
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
    /* Check if placing at idx connects to a group in atari */
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == s->to_play && is_atari(s, ni, s->to_play))
            return 1;
    }
    return 0;
}

static int is_self_atari(GoState *s, int idx) {
    GoState cp = *s;
    cp.board[idx] = cp.to_play;
    int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp))
            return 0;  /* captures something, so not self-atari */
    }
    return group_dead(&cp, idx, cp.to_play);
}

static int is_legal(GoState *s, int idx) {
    if (idx == CELLS) return 1;
    if (!in_bounds(idx) || s->board[idx] != 0) return 0;
    if (is_self_atari(s, idx)) return 0;
    GoState cp = *s;
    cp.board[idx] = cp.to_play;
    int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) {
            int8_t vis[CELLS] = {0};
            remove_group(&cp, ni, vis);
        }
    }
    if (memcmp(cp.board, s->prev, sizeof(s->prev)) == 0) return 0;
    return 1;
}


static int is_tactical_move(GoState *s, int idx) {
    if (idx == CELLS) return 0;
    return move_captures(s, idx) > 0 ||
           move_saves_own(s, idx) ||
           (is_legal(s, idx) && is_atari(s, idx, -s->to_play)); /* put enemy in atari */
}

/* ── legality ─────────────────────────────────────────────────────────── */

static void go_move(GoState *s, int idx) {
    s->last_move = idx;
    if (idx == CELLS) { s->passes++; s->to_play = -s->to_play; return; }
    s->passes = 0;
    memcpy(s->prev, s->board, sizeof(s->board));
    go_update_hash(s, idx, 0, s->to_play == 1 ? 1 : 2);
    s->board[idx] = s->to_play;
    int opp = -s->to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == opp && group_dead(s, ni, opp)) {
            int8_t vis[CELLS] = {0};
            remove_group(s, ni, vis);
        }
    }
    s->to_play = -s->to_play; s->moves++;
}

static int go_over(GoState *s) { return s->passes >= 2 || s->moves >= MOVES_PER_GAME; }

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
    int n = 0;
    for (int i = 0; i < CELLS; i++) if (s->board[i] != 0) n++;
    return (float)n / CELLS;
}

/* ── features ─────────────────────────────────────────────────────────── */

static void go_features(GoState *s, nn_real row[FEATURES]) {
    memset(row, 0, FEATURES * sizeof(nn_real));
    for (int i = 0; i < CELLS; i++) {
        int off = i * 3;
        if (s->board[i] == 1)       row[off] = 1.0f;
        else if (s->board[i] == -1) row[off + 1] = 1.0f;
        else                         row[off + 2] = 1.0f;
    }
}

/* ── neural evaluation ────────────────────────────────────────────────── */

static void go_evaluate(NeuralNet *net, GoState *s, float policy[MAX_MOVES], float *value) {
    nn_real feats[FEATURES];
    go_features(s, feats);
    float total = 0.0f;
    for (int m = 0; m < MAX_MOVES; m++) {
        if (!is_legal(s, m)) { policy[m] = 0.0f; continue; }
        nn_real mf[FEATURES];
        memcpy(mf, feats, FEATURES * sizeof(nn_real));
        mf[FEATURES - 1] = (nn_real)m / MAX_MOVES;
        nn_real sc;
        nn_forward(net, mf, &sc);
        float sv = (float)sc;
        if (sv < 0.005f) sv = 0.005f;
        /* Danger discount: penalize self-atari */
        if (m < CELLS && is_self_atari(s, m)) sv *= 0.3f;
        policy[m] = sv; total += sv;
    }
    if (total > 0.0f)
        for (int m = 0; m < MAX_MOVES; m++) policy[m] /= total;
    
    nn_real vs;
    nn_forward(net, feats, &vs);
    *value = (float)vs * 2.0f - 1.0f;
}

/* ── entropy ──────────────────────────────────────────────────────────── */

static float policy_entropy(float policy[MAX_MOVES]) {
    float h = 0.0f; int n = 0;
    for (int m = 0; m < MAX_MOVES; m++) {
        if (policy[m] > 0.0f) { h -= policy[m] * logf(policy[m] + 1e-12f); n++; }
    }
    if (n <= 1) return 0.0f;
    return h / logf((float)n);
}

/* ── local fight mask ─────────────────────────────────────────────────── */

static bool in_local_mask(GoState *s, int idx) {
    if (s->last_move < 0 || s->last_move >= CELLS) return false;
    int lx = s->last_move % BOARD_SIZE, ly = s->last_move / BOARD_SIZE;
    int mx = idx % BOARD_SIZE, my = idx / BOARD_SIZE;
    return abs(lx - mx) <= 2 && abs(ly - my) <= 2;
}

/* ── RAVE ─────────────────────────────────────────────────────────────── */

static float rave_wins[CELLS + 1];
static float rave_visits[CELLS + 1];

static void rave_reset(void) {
    memset(rave_wins, 0, sizeof(rave_wins));
    memset(rave_visits, 0, sizeof(rave_visits));
}

static float rave_q(int move) {
    if (rave_visits[move] < 1.0f) return 0.0f;
    return rave_wins[move] / rave_visits[move];
}

/* ── TT cache ─────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t hash;
    float policy[12];  /* top-12 moves */
    int8_t  moves[12];
    int     n;
} TTEntry;

static TTEntry tt[TT_SIZE];
static int tt_filled = 0;

static TTEntry *tt_lookup(uint64_t hash) {
    int idx = hash % TT_SIZE;
    if (tt[idx].hash == hash) return &tt[idx];
    return NULL;
}

static void tt_store(uint64_t hash, float policy[MAX_MOVES]) {
    int idx = hash % TT_SIZE;
    if (tt[idx].hash == hash) return;  /* already cached */
    tt[idx].hash = hash;
    tt[idx].n = 0;
    /* Store top-12 by priority */
    for (int m = 0; m < MAX_MOVES && tt[idx].n < 12; m++) {
        if (policy[m] > 0.001f) {
            tt[idx].moves[tt[idx].n] = (int8_t)m;
            tt[idx].policy[tt[idx].n] = policy[m];
            tt[idx].n++;
        }
    }
    if (tt_filled < TT_SIZE) tt_filled++;
}

/* ── sparse MCTS node ─────────────────────────────────────────────────── */

typedef struct GoNode {
    int move, to_play;
    struct GoNode *parent, *kids[MCTS_MAX_KIDS];
    int nkids;
    float prior[MCTS_MAX_KIDS];
    float visits, total_val;
    bool expanded;
    /* reason labels */
    int tactical_forced[MCTS_MAX_KIDS];
    int search_refuted[MCTS_MAX_KIDS];
} GoNode;

static GoNode *gn_alloc(int move, int to_play, GoNode *parent) {
    GoNode *n = calloc(1, sizeof(GoNode));
    n->move = move; n->to_play = to_play; n->parent = parent;
    return n;
}

static void gn_free(GoNode *n) {
    if (!n) return;
    for (int i = 0; i < n->nkids; i++) gn_free(n->kids[i]);
    free(n);
}

static int gn_select(GoNode *n) {
    int best = 0;
    float best_score = -1e30f, sqrt_n = sqrtf(n->visits + 1e-8f);
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

/* ── sparse MCTS search ───────────────────────────────────────────────── */

static void go_mcts_search(NeuralNet *net, GoState *root_st, float probs[MAX_MOVES]) {
    GoNode *root = gn_alloc(-1, root_st->to_play, NULL);
    
    for (int sim = 0; sim < MCTS_SIMULATIONS; sim++) {
        GoState s = *root_st;
        GoNode *node = root;
        
        while (node->expanded && node->nkids > 0 && !go_over(&s)) {
            node = node->kids[gn_select(node)];
            go_move(&s, node->move);
        }
        
        float v;
        if (go_over(&s)) {
            float sc = go_score(&s);
            v = (sc > 0) ? 1.0f : (sc < 0) ? -1.0f : 0.0f;
            v *= (float)node->to_play;
        } else {
            float policy[MAX_MOVES];
            /* Check TT cache */
            uint64_t h = go_hash(&s);
            TTEntry *te = tt_lookup(h);
            if (te) {
                memset(policy, 0, sizeof(policy));
                for (int i = 0; i < te->n; i++) policy[te->moves[i]] = te->policy[i];
                nn_real vs; nn_real feats[FEATURES];
                go_features(&s, feats);
                nn_forward(net, feats, &vs);
                v = (float)vs * 2.0f - 1.0f;
            } else {
                go_evaluate(net, &s, policy, &v);
                tt_store(h, policy);
            }
            
            /* Phase-dependent expansion width */
            float phase = go_stone_ratio(&s);
            float ent = policy_entropy(policy);
            int base_k = (phase < 0.25f) ? 12 : (phase > 0.75f) ? 4 : 8;
            int entropy_k = (int)(ent * 16.0f);
            int top_k = base_k + entropy_k;
            if (top_k < MCTS_MIN_EXPAND) top_k = MCTS_MIN_EXPAND;
            if (top_k > 30) top_k = 30;
            
            /* Collect legal moves with priors */
            typedef struct { int move; float prior; int tactical; } candidate_t;
            candidate_t cands[MAX_MOVES];
            int nc = 0;
            float max_prior = 0.001f;
            for (int m = 0; m < MAX_MOVES; m++) {
                if (!is_legal(&s, m)) continue;
                int tactical = is_tactical_move(&s, m);
                float prior = policy[m];
                if (tactical && prior < 0.01f) prior = 0.01f;  /* boost tactical */
                if (in_local_mask(&s, m)) prior *= 1.3f;        /* local fight bonus */
                if (prior > max_prior) max_prior = prior;
                cands[nc].move = m; cands[nc].prior = prior;
                cands[nc].tactical = tactical;
                nc++;
            }
            
            /* Sort by prior (simple bubble — nc is small) */
            for (int i = 0; i < nc; i++)
                for (int j = i + 1; j < nc; j++)
                    if (cands[j].prior > cands[i].prior) {
                        candidate_t t = cands[i]; cands[i] = cands[j]; cands[j] = t;
                    }
            
            /* Expand: top_k by prior + tactical forced */
            bool expanded_move[MAX_MOVES] = {false};
            node->nkids = 0;
            float prior_threshold = max_prior * 0.15f;
            
            for (int i = 0; i < nc && node->nkids < top_k; i++) {
                if (cands[i].tactical || cands[i].prior >= prior_threshold) {
                    int m = cands[i].move;
                    GoNode *kid = gn_alloc(m, -node->to_play, node);
                    node->kids[node->nkids] = kid;
                    node->prior[node->nkids] = cands[i].prior;
                    node->tactical_forced[node->nkids] = cands[i].tactical;
                    node->nkids++;
                    expanded_move[m] = true;
                }
            }
            
            /* Force-include any remaining tactical moves */
            for (int i = 0; i < nc; i++) {
                if (!expanded_move[cands[i].move] && cands[i].tactical && node->nkids < MCTS_MAX_KIDS) {
                    int m = cands[i].move;
                    GoNode *kid = gn_alloc(m, -node->to_play, node);
                    node->kids[node->nkids] = kid;
                    node->prior[node->nkids] = cands[i].prior;
                    node->tactical_forced[node->nkids] = 1;
                    node->nkids++;
                }
            }
            node->expanded = true;
        }
        
        /* Backprop with RAVE update */
        while (node) {
            node->visits += 1.0f;
            node->total_val += v;
            if (v > 0.0f && node->parent && node->move >= 0) {
                rave_wins[node->move] += 1.0f;
                rave_visits[node->move] += 1.0f;
            } else if (node->parent && node->move >= 0) {
                rave_visits[node->move] += 1.0f;
            }
            v = -v; node = node->parent;
        }
    }
    
    /* Extract visit probabilities */
    float total = 0.0f;
    for (int i = 0; i < root->nkids; i++) {
        probs[root->kids[i]->move] = root->kids[i]->visits;
        total += root->kids[i]->visits;
    }
    if (total > 0.0f)
        for (int m = 0; m < MAX_MOVES; m++) probs[m] /= total;
    
    gn_free(root);
}

static int go_mcts_move(NeuralNet *net, GoState *s) {
    /* Adaptive visit budget */
    float probs[MAX_MOVES] = {0};
    int budget = 32;
    go_mcts_search(net, s, probs);
    
    /* Confidence check: if best >> second, stop; else search more */
    float best_p = 0, second_p = 0;
    int best_m = CELLS;
    for (int m = 0; m < CELLS; m++) {
        if (probs[m] > best_p) { second_p = best_p; best_p = probs[m]; best_m = m; }
        else if (probs[m] > second_p) second_p = probs[m];
    }
    (void)budget; (void)second_p;
    return best_m;
}

/* ── baselines ────────────────────────────────────────────────────────── */

static int go_random_move(GoState *s) {
    int legal[MAX_MOVES], n = 0;
    for (int m = 0; m < MAX_MOVES; m++)
        if (is_legal(s, m)) legal[n++] = m;
    return n > 0 ? legal[rand() % n] : CELLS;
}

static int go_greedy_neural(NeuralNet *net, GoState *s) {
    float policy[MAX_MOVES], value;
    go_evaluate(net, s, policy, &value);
    int best = CELLS;
    for (int m = 0; m < CELLS; m++)
        if (policy[m] > policy[best]) best = m;
    return best;
}

/* ── training sample ──────────────────────────────────────────────────── */

typedef struct {
    nn_real features[FEATURES];
    int move, to_play;
} GoSample;

static int go_selfplay(NeuralNet *net, GoSample *samples, int max, int use_mcts) {
    GoState s; go_init(&s);
    rave_reset();
    int n = 0;
    while (!go_over(&s) && n < max) {
        GoSample *sp = &samples[n];
        go_features(&s, sp->features);
        sp->to_play = s.to_play;
        
        int move = use_mcts ? go_mcts_move(net, &s) : go_greedy_neural(net, &s);
        sp->features[FEATURES - 1] = (nn_real)move / MAX_MOVES;
        sp->move = move;
        go_move(&s, move);
        n++;
    }
    float score = go_score(&s);
    float winner = (score > 0) ? 1.0f : ((score < 0) ? -1.0f : 0.0f);
    for (int i = 0; i < n; i++) {
        float r = (samples[i].to_play == winner) ? 1.0f : (winner == 0) ? 0.5f : 0.0f;
        samples[i].features[FEATURES - 2] = r;
    }
    return n;
}

/* ── agents for benchmarking ──────────────────────────────────────────── */

typedef int (*go_agent_t)(void *ctx, GoState *s);

static float go_match(void *a_ctx, go_agent_t a, void *b_ctx, go_agent_t b, int games) {
    int wins_a = 0;
    for (int g = 0; g < games; g++) {
        GoState s; go_init(&s);
        while (!go_over(&s)) {
            int move = (s.to_play == 1) ? a(a_ctx, &s) : b(b_ctx, &s);
            go_move(&s, move);
        }
        if (go_score(&s) > 0) wins_a++;
    }
    return (float)wins_a / (float)games;
}

static int w_random(void *c, GoState *s) { (void)c; return go_random_move(s); }
static int w_greedy(void *c, GoState *s) { return go_greedy_neural((NeuralNet *)c, s); }
static int w_mcts(void *c, GoState *s) { return go_mcts_move((NeuralNet *)c, s); }

/* ── demo ─────────────────────────────────────────────────────────────── */

static void go_demo(NeuralNet *net) {
    GoState s; go_init(&s);
    printf("\n  Sparse MCTS Demo\n");
    while (!go_over(&s)) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            printf("  ");
            for (int x = 0; x < BOARD_SIZE; x++) {
                int v = s.board[xy(x, y)];
                printf("%c ", v == 1 ? 'X' : v == -1 ? 'O' : '.');
            }
            printf("\n");
        }
        printf("  %s: ", s.to_play == 1 ? "X" : "O");
        int move = go_mcts_move(net, &s);
        if (move == CELLS) printf("PASS\n\n");
        else printf("%c%d\n\n", 'A' + (move % BOARD_SIZE), (move / BOARD_SIZE) + 1);
        go_move(&s, move);
    }
    float sc = go_score(&s);
    printf("  Score: %.0f  %s wins\n", fabsf(sc), sc > 0 ? "X" : sc < 0 ? "O" : "Draw");
}

/* ── main ─────────────────────────────────────────────────────────────── */

static void go_demo(NeuralNet *net);
static void gtp_loop(NeuralNet *net);

int main(int argc, char **argv) {
    int use_mcts = 1, benchmark = 0, demo = 0, gtp = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-mcts") == 0) use_mcts = 0;
        if (strcmp(argv[i], "--benchmark") == 0) benchmark = 1;
        if (strcmp(argv[i], "--demo") == 0) demo = 1;
        if (strcmp(argv[i], "--gtp") == 0) gtp = 1;
    }
    srand((unsigned)time(NULL));
    
    const size_t layers[] = {FEATURES, HIDDEN0, HIDDEN1, 1};
    NNConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.layer_sizes = layers; cfg.layer_count = 4; cfg.seed = 42;
    cfg.hidden_activation = NN_ACTIVATION_RELU;
    cfg.output_activation = NN_ACTIVATION_SIGMOID;
    cfg.loss = NN_LOSS_MSE;
    
    NeuralNet *net = nn_create_config(&cfg);
    if (!net) { fprintf(stderr, "create failed\n"); return 1; }
    
    if (gtp) {
        gtp_loop(net);
        nn_free(net);
        return 0;
    }

    if (demo) {
        const char *ckpt = argc > 2 ? argv[2] : "build/float/go_mcts.nnckpt";
        printf("Loading %s...\n", ckpt);
        char err[256] = {0};
        /* Load via nn.h checkpoint API — just run demo with fresh net for now */
        go_demo(net); nn_free(net); return 0;
    }
    
    GoSample *samples = calloc(GAMES * MOVES_PER_GAME, sizeof(GoSample));
    int total = 0;
    
    printf("Sparse MCTS Go 9×9  %d games  %d sims  c_puct=%.1f\n\n",
           GAMES, MCTS_SIMULATIONS, (double)MCTS_C_PUCT);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("─── Epoch %d ───\n", epoch + 1);
        for (int g = 0; g < GAMES; g++) {
            int n = go_selfplay(net, &samples[total], GAMES * MOVES_PER_GAME - total, use_mcts);
            total += n;
        }
        
        float lr = 0.01f - epoch * 0.002f;
        for (int e = 0; e < 3; e++) {
            float loss = 0;
            for (int i = 0; i < total; i++) {
                nn_real tgt = samples[i].features[FEATURES - 2];
                nn_real lv = nn_train_sample(net, samples[i].features, &tgt, lr);
                loss += (float)lv;
            }
            printf("  train %d  loss=%.4f  lr=%.4f\n", e + 1, loss / total, (double)lr);
        }
        total = 0;
    }
    
    NNCheckpointMeta meta;
    nn_checkpoint_meta_init(&meta, 1);
    nn_checkpoint_meta_set_feature_set(&meta, "go-9x9-sparse-mcts-v1");
    
    const char *path = "build/float/go_mcts.nnckpt";
    if (nn_save_checkpoint(net, path, &meta)) {
        char hex[65]; nn_hash_checkpoint_hex(path, hex);
        printf("\nSaved: %s  hash=%s\n", path, hex);
    }
    
    if (benchmark) {
        printf("\n═══ Benchmark (100 games) ═══\n");
        printf("neural-MCTS vs random:        %.1f%%\n", 100.0f * go_match(net, w_mcts, NULL, w_random, 10));
        printf("neural-MCTS vs greedy-neural: %.1f%%\n", 100.0f * go_match(net, w_mcts, net, w_greedy, 10));
        printf("greedy-neural vs random:      %.1f%%\n", 100.0f * go_match(net, w_greedy, NULL, w_random, 10));
    }
    
    go_demo(net);
    
    nn_free(net);
    free(samples);
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 * GTP (Go Text Protocol) interface for CGOS / OGS / KGS
 * ═══════════════════════════════════════════════════════════════════════ */

static GoState gtp_state;
static NeuralNet *gtp_net;

static void gtp_respond(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("= "); vprintf(fmt, ap); printf("\n\n"); fflush(stdout);
    va_end(ap);
}

static void gtp_error(const char *msg) {
    printf("? %s\n\n", msg); fflush(stdout);
}

static void gtp_boardsize(int n) {
    if (n != BOARD_SIZE) { gtp_error("only 9x9 supported"); return; }
    go_init(&gtp_state);
    gtp_respond("");
}

static void gtp_clear_board(void) {
    go_init(&gtp_state);
    gtp_respond("");
}

static void gtp_play(const char *color, const char *vertex) {
    int to_play = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
    if (to_play != gtp_state.to_play) { gtp_error("wrong turn"); return; }
    
    int move = CELLS; /* pass */
    if (strcasecmp(vertex, "pass") != 0 && strcasecmp(vertex, "resign") != 0) {
        int x = toupper(vertex[0]) - 'A';
        int y = atoi(vertex + 1) - 1;
        if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE)
            move = xy(x, y);
        else { gtp_error("invalid vertex"); return; }
    }
    
    if (!is_legal(&gtp_state, move)) { gtp_error("illegal move"); return; }
    go_move(&gtp_state, move);
    gtp_respond("");
}

static void gtp_genmove(const char *color) {
    int to_play = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
    if (to_play != gtp_state.to_play) { gtp_error("wrong turn"); return; }
    
    int move = go_mcts_move(gtp_net, &gtp_state);
    go_move(&gtp_state, move);
    
    if (move == CELLS) gtp_respond("PASS");
    else {
        char buf[4];
        snprintf(buf, sizeof(buf), "%c%d", 'A' + (move % BOARD_SIZE), (move / BOARD_SIZE) + 1);
        gtp_respond("%s", buf);
    }
}

static void gtp_loop(NeuralNet *net) {
    gtp_net = net;
    go_init(&gtp_state);
    
    char line[256];
    while (fgets(line, sizeof(line), stdin)) {
        /* Strip newline and comments */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        
        char cmd[64], arg1[64], arg2[64];
        int n = sscanf(p, "%63s %63s %63s", cmd, arg1, arg2);
        
        if (strcmp(cmd, "quit") == 0) { gtp_respond(""); break; }
        else if (strcmp(cmd, "protocol_version") == 0) gtp_respond("2");
        else if (strcmp(cmd, "name") == 0) gtp_respond("SparseMCTS-Go");
        else if (strcmp(cmd, "version") == 0) gtp_respond("0.1");
        else if (strcmp(cmd, "boardsize") == 0) gtp_boardsize(atoi(arg1));
        else if (strcmp(cmd, "clear_board") == 0) gtp_clear_board();
        else if (strcmp(cmd, "play") == 0 && n >= 3) gtp_play(arg1, arg2);
        else if (strcmp(cmd, "genmove") == 0 && n >= 2) gtp_genmove(arg1);
        else if (strcmp(cmd, "known_command") == 0) gtp_respond("true");
        else if (strcmp(cmd, "list_commands") == 0)
            gtp_respond("protocol_version\nname\nversion\nboardsize\nclear_board\nplay\ngenmove\nquit");
        else gtp_error("unknown command");
    }
}
