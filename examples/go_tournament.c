/**
 * go_tournament.c — Adversarial Go tournament with continuous learning.
 *
 * N evolving networks play round-robin matches. Winners train on game data.
 * A frozen baseline network measures absolute progress.
 * Serves a live status page at :8087 showing win rates and ELO.
 *
 * Usage: ./go_tournament [--players N] [--games-per-pair K] [--baseline ckpt]
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
#include <signal.h>
#include <pthread.h>

#define BOARD_SIZE 9
#define CELLS (BOARD_SIZE * BOARD_SIZE)
#define MAX_MOVES (CELLS + 1)
#define CONV_FILTERS 16
#define CONV_KERNEL  3
#define CONV_IN_C    3
#define BOARD_ONEHOT   (CELLS * CONV_IN_C)
#define CONV_FEATURES  (CELLS * CONV_FILTERS)
#define BOARD_FEATURES CONV_FEATURES
#define SLOT_MOVE_ID   (BOARD_FEATURES)
/* Reward / training target is stored OUTSIDE the input vector (GameSample.reward)
 * to prevent label leakage: a slot containing the answer in features[] would let
 * the network minimize loss by copying its input. */
#define FEATURES       (BOARD_FEATURES + 1)
#define HIDDEN0 128
#define HIDDEN1 64

#define MCTS_SIMULATIONS 4
#define MCTS_C_PUCT 1.4f
#define MCTS_MAX_KIDS 82
#define MCTS_MIN_EXPAND 2
#define TT_SIZE 16384
#define MAX_GAME_MOVES 150
#define REPLAY_CAPACITY 2000

static const int DIRS[4] = {-BOARD_SIZE, -1, 1, BOARD_SIZE};
static uint64_t zobrist_side;

/* ── Go engine (condensed) ───────────────────────────────────────────── */

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
    for (int i = 0; i < CELLS; i++)
        for (int c = 0; c < 3; c++) {
            z_seed = z_seed * 1103515245u + 12345u;
            uint64_t lo = ((uint64_t)z_seed) & 0xFFFFFFFFu;
            z_seed = z_seed * 1103515245u + 12345u;
            uint64_t hi = ((uint64_t)z_seed) & 0xFFFFFFFFu;
            zobrist[i][c] = (hi << 32) | lo;
        }
    z_seed = z_seed * 1103515245u + 12345u;
    uint64_t lo = ((uint64_t)z_seed) & 0xFFFFFFFFu;
    z_seed = z_seed * 1103515245u + 12345u;
    uint64_t hi = ((uint64_t)z_seed) & 0xFFFFFFFFu;
    zobrist_side = (hi << 32) | lo;
    z_init_done = true;
}

static void go_init(GoState *s) {
    z_init(); memset(s, 0, sizeof(*s)); s->to_play = 1; s->last_move = -1; s->zhash = zobrist_side;
}
static int xy(int x, int y) { return y * BOARD_SIZE + x; }
static inline int in_bounds(int i) { return i >= 0 && i < CELLS; }
static inline int on_left(int i)  { return i % BOARD_SIZE == 0; }
static inline int on_right(int i) { return i % BOARD_SIZE == BOARD_SIZE - 1; }

static void go_hash_update(GoState *s, int idx, int from, int to) { s->zhash ^= zobrist[idx][from]; s->zhash ^= zobrist[idx][to]; }

static void remove_group(GoState *s, int i, int8_t *vis) {
    int8_t c = s->board[i]; s->board[i] = 0; vis[i] = 1;
    go_hash_update(s, i, c == 1 ? 1 : 2, 0);
    for (int d = 0; d < 4; d++) {
        int ni = i + DIRS[d]; if ((d == 1 && on_left(i)) || (d == 2 && on_right(i))) continue;
        if (in_bounds(ni) && !vis[ni] && s->board[ni] == c) remove_group(s, ni, vis);
    }
}

static int has_liberty(GoState *s, int i, int8_t *vis, int8_t color) {
    if (!in_bounds(i) || vis[i]) return 0;
    if (s->board[i] == 0) return 1;
    if (s->board[i] != color) return 0;
    vis[i] = 1;
    for (int d = 0; d < 4; d++) {
        int ni = i + DIRS[d]; if ((d == 1 && on_left(i)) || (d == 2 && on_right(i))) continue;
        if (has_liberty(s, ni, vis, color)) return 1;
    }
    return 0;
}

static int group_dead(GoState *s, int i, int8_t color) {
    if (!in_bounds(i) || s->board[i] != color) return 0;
    int8_t vis[CELLS] = {0}; return !has_liberty(s, i, vis, color);
}

static int count_liberties(GoState *s, int i) {
    if (!in_bounds(i) || s->board[i] == 0) return 0;
    int8_t vis[CELLS] = {0}, libs[CELLS] = {0}; int8_t c = s->board[i];
    int q[CELLS], h = 0, t = 0; q[t++] = i; vis[i] = 1; int count = 0;
    while (h < t) {
        int idx = q[h++];
        for (int d = 0; d < 4; d++) {
            int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
            if (!in_bounds(ni)) continue;
            if (s->board[ni] == 0 && !libs[ni]) { libs[ni] = 1; count++; }
            else if (s->board[ni] == c && !vis[ni]) { vis[ni] = 1; q[t++] = ni; }
        }
    }
    return count;
}

static int is_suicide(GoState *s, int idx) {
    GoState cp = *s; cp.board[idx] = cp.to_play; int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) return 0;
    }
    return group_dead(&cp, idx, cp.to_play);
}

static int is_self_atari(GoState *s, int idx) {
    GoState cp = *s; cp.board[idx] = cp.to_play; int opp = -cp.to_play, captured = 0;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) {
            captured = 1; int8_t vis[CELLS] = {0}; remove_group(&cp, ni, vis);
        }
    }
    if (captured) return 0;
    return count_liberties(&cp, idx) == 1;
}

static int is_atari(GoState *s, int i, int8_t color) { (void)color; return count_liberties(s, i) == 1; }
static int move_captures(GoState *s, int idx) {
    int opp = -s->to_play, caps = 0;
    for (int d = 0; d < 4; d++) { int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == opp && is_atari(s, ni, opp)) caps++; }
    return caps;
}
static int move_saves_own(GoState *s, int idx) {
    for (int d = 0; d < 4; d++) { int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == s->to_play && is_atari(s, ni, s->to_play)) return 1; }
    return 0;
}
static int is_tactical(GoState *s, int idx) { if (idx == CELLS) return 0; return move_captures(s, idx) > 0 || move_saves_own(s, idx); }

static int is_legal(GoState *s, int idx) {
    if (idx == CELLS) return 1;
    if (!in_bounds(idx) || s->board[idx] != 0) return 0;
    if (is_suicide(s, idx)) return 0;
    GoState cp = *s; cp.board[idx] = cp.to_play; int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) { int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) { int8_t vis[CELLS] = {0}; remove_group(&cp, ni, vis); } }
    return memcmp(cp.board, s->prev, sizeof(s->prev)) != 0;
}

static void go_move(GoState *s, int idx) {
    s->last_move = idx;
    if (idx == CELLS) { s->passes++; s->to_play = -s->to_play; s->zhash ^= zobrist_side; return; }
    s->passes = 0; memcpy(s->prev, s->board, sizeof(s->board));
    go_hash_update(s, idx, 0, s->to_play == 1 ? 1 : 2); s->zhash ^= zobrist_side;
    s->board[idx] = s->to_play; int opp = -s->to_play;
    for (int d = 0; d < 4; d++) { int ni = idx + DIRS[d]; if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && s->board[ni] == opp && group_dead(s, ni, opp)) { int8_t vis[CELLS] = {0}; remove_group(s, ni, vis); } }
    s->to_play = -s->to_play; s->moves++;
}
static inline int go_over(GoState *s) { return s->passes >= 2 || s->moves >= MAX_GAME_MOVES; }
static float go_score(GoState *s) {
    /* Komi: White gets 6.5 points compensation for going second */
    int black = 0, white = 0; int8_t vis[CELLS] = {0};
    for (int i = 0; i < CELLS; i++) { if (s->board[i] == 1) black++; else if (s->board[i] == -1) white++; else if (!vis[i]) {
        int q[CELLS], h = 0, t = 0, hb = 0, hw = 0, cnt = 0; q[t++] = i; vis[i] = 1;
        while (h < t) { int idx = q[h++]; cnt++; for (int d = 0; d < 4; d++) { int ni = idx + DIRS[d];
            if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue; if (!in_bounds(ni)) continue;
            if (s->board[ni] == 1) hb = 1; else if (s->board[ni] == -1) hw = 1; else if (!vis[ni]) { vis[ni] = 1; q[t++] = ni; } } }
        if (hb && !hw) black += cnt; if (hw && !hb) white += cnt; } }
    return (float)(black - white) - 6.5f;
}
static inline float go_stone_ratio(GoState *s) { int n = 0; for (int i = 0; i < CELLS; i++) if (s->board[i] != 0) n++; return (float)n / CELLS; }

/* ── Conv2D feature extractor (32 filters, 3x3) ────────────────────── */

static nn_real conv_weights[CONV_FILTERS * CONV_IN_C * CONV_KERNEL * CONV_KERNEL];
static nn_real conv_biases[CONV_FILTERS];
static int conv_initialized = 0;

static void conv_init(unsigned seed) {
    if (conv_initialized) return;
    unsigned s = seed;
    nn_real scale = (nn_real)(2.0 / sqrt((double)(CONV_IN_C * CONV_KERNEL * CONV_KERNEL)));
    for (int i = 0; i < CONV_FILTERS * CONV_IN_C * CONV_KERNEL * CONV_KERNEL; i++) {
        s = s * 1103515245u + 12345u;
        conv_weights[i] = ((nn_real)(s & 0xFFFF) / 65535.0f * 2.0f - 1.0f) * scale;
    }
    memset(conv_biases, 0, sizeof(conv_biases));

    conv_initialized = 1;
}

static void conv_forward(const nn_real *input, nn_real *output) {
    int pad = CONV_KERNEL / 2;
    for (int k = 0; k < CONV_FILTERS; k++)
        for (int y = 0; y < BOARD_SIZE; y++)
            for (int x = 0; x < BOARD_SIZE; x++) {
                nn_real sum = conv_biases[k];
                for (int c = 0; c < CONV_IN_C; c++)
                    for (int ky = 0; ky < CONV_KERNEL; ky++)
                        for (int kx = 0; kx < CONV_KERNEL; kx++) {
                            int iy = y + ky - pad, ix = x + kx - pad;
                            if (iy >= 0 && iy < BOARD_SIZE && ix >= 0 && ix < BOARD_SIZE)
                                sum += input[(c * BOARD_SIZE + iy) * BOARD_SIZE + ix] *
                                       conv_weights[((k * CONV_IN_C + c) * CONV_KERNEL + ky) * CONV_KERNEL + kx];
                        }
                output[(k * BOARD_SIZE + y) * BOARD_SIZE + x] = sum > 0 ? sum : 0;
            }
}

static void go_features(GoState *s, nn_real row[FEATURES]) {
    memset(row, 0, FEATURES * sizeof(nn_real));
    nn_real onehot[CELLS * 3];
    for (int i = 0; i < CELLS; i++) {
        int off = i * 3;
        if (s->board[i] == 1)      onehot[off]   = 1.0f;
        else if (s->board[i] == -1) onehot[off+1] = 1.0f;
        else                        onehot[off+2] = 1.0f;
    }
    conv_forward(onehot, row);
}

static void go_evaluate(NeuralNet *net, GoState *s, float policy[MAX_MOVES], float *value) {
    nn_real feats[FEATURES];
    go_features(s, feats); float total = 0.0f;
    for (int m = 0; m < MAX_MOVES; m++) { if (!is_legal(s, m)) { policy[m] = 0.0f; continue; }
        nn_real mf[FEATURES]; memcpy(mf, feats, FEATURES*sizeof(nn_real)); mf[SLOT_MOVE_ID] = (nn_real)m / MAX_MOVES;
        nn_real sc; nn_forward(net, mf, &sc); float sv = (float)sc; if (sv < 0.005f) sv = 0.005f;
        /* self-atari discount skipped for tournament speed */ policy[m] = sv; total += sv; }
    if (total > 0.0f) for (int m = 0; m < MAX_MOVES; m++) policy[m] /= total;
    nn_real vs; nn_forward(net, feats, &vs); *value = (float)vs * 2.0f - 1.0f;
}
static float policy_entropy(float policy[MAX_MOVES]) { float h = 0.0f; int n = 0;
    for (int m = 0; m < MAX_MOVES; m++) if (policy[m] > 0.0f) { h -= policy[m]*logf(policy[m]+1e-12f); n++; }
    return n <= 1 ? 0.0f : h / logf((float)n); }
static bool in_local_mask(GoState *s, int idx) { if (s->last_move < 0 || s->last_move >= CELLS) return false;
    return abs(s->last_move%BOARD_SIZE - idx%BOARD_SIZE) <= 2 && abs(s->last_move/BOARD_SIZE - idx/BOARD_SIZE) <= 2; }

/* ── MCTS (per-network) ──────────────────────────────────────────────── */

typedef struct GoNode { int move, to_play; struct GoNode *parent, *kids[MCTS_MAX_KIDS]; int nkids; float prior[MCTS_MAX_KIDS], visits, total_val; bool expanded; } GoNode;

static GoNode *gn_alloc(int move, int to_play, GoNode *parent) { GoNode *n = calloc(1, sizeof(GoNode)); if (!n) return NULL; n->move=move; n->to_play=to_play; n->parent=parent; return n; }
static void gn_free(GoNode *n) { if (!n) return; for (int i = 0; i < n->nkids; i++) gn_free(n->kids[i]); free(n); }

/* RAVE arrays are now per-search (stack-allocated in go_mcts_move_with_probs) */

typedef struct { uint64_t hash; float policy[12]; int8_t moves[12]; int n; bool valid; } TTEntry;
static TTEntry tt[TT_SIZE];
static TTEntry *tt_lookup(uint64_t hash) { int idx = hash % TT_SIZE; return (tt[idx].valid && tt[idx].hash == hash) ? &tt[idx] : NULL; }
static void tt_store(uint64_t hash, float policy[MAX_MOVES]) { int idx = hash % TT_SIZE; if (tt[idx].valid && tt[idx].hash == hash) return;
    tt[idx].hash=hash; tt[idx].valid=true; tt[idx].n=0; for (int m=0; m<MAX_MOVES && tt[idx].n<12; m++) if (policy[m]>0.001f) { tt[idx].moves[tt[idx].n]=(int8_t)m; tt[idx].policy[tt[idx].n]=policy[m]; tt[idx].n++; } }


static inline uint64_t go_hash(GoState *s) { return s->zhash; }

static int go_mcts_move_with_probs(NeuralNet *net, GoState *root_st, float *out_probs) {
    if (out_probs) memset(out_probs, 0, MAX_MOVES * sizeof(float));
    float probs[MAX_MOVES] = {0};
    GoNode *root = gn_alloc(-1, root_st->to_play, NULL); if (!root) return CELLS;
    /* Per-search RAVE arrays on stack for thread safety */
    float rave_w[CELLS+1] = {0}, rave_v[CELLS+1] = {0};
    /* Dirichlet noise for root exploration (AlphaZero-style) */
    float dirichlet_alpha = 0.03f * (float)MCTS_SIMULATIONS;
    float noise[MAX_MOVES] = {0};
    if (dirichlet_alpha > 0.01f) {
        /* Generate Dirichlet noise using Gamma approximation */
        float noise_sum = 0;
        for (int m = 0; m < MAX_MOVES; m++) {
            if (is_legal(root_st, m)) {
                /* Simple exponential approximation of Gamma(alpha, 1) */
                float u = (float)(rand() ^ (unsigned long)pthread_self()) / RAND_MAX;
                noise[m] = -logf(u + 1e-12f) * dirichlet_alpha / 82.0f;
                noise_sum += noise[m];
            }
        }
        if (noise_sum > 0) for (int m = 0; m < MAX_MOVES; m++) noise[m] /= noise_sum;
    }
    for (int sim = 0; sim < MCTS_SIMULATIONS; sim++) {
        /* Early stop: if best move overwhelming after 8 sims, skip rest */
        if (sim >= 4 && root->nkids > 0) {
            float best_v = 0, second_v = 0;
            for (int i = 0; i < root->nkids; i++) {
                if (root->kids[i]->visits > best_v) { second_v = best_v; best_v = root->kids[i]->visits; }
                else if (root->kids[i]->visits > second_v) second_v = root->kids[i]->visits;
            }
            if (best_v > 0 && second_v / (best_v + 0.001f) < 0.6f) break;
        }
        GoState s = *root_st; GoNode *node = root;
        while (node->expanded && node->nkids > 0 && !go_over(&s)) { {
                GoNode *n = node;
                int best = 0; float best_score = -1e30f, sqrt_n = sqrtf(n->visits + 1e-8f);
                for (int i = 0; i < n->nkids; i++) { GoNode *c = n->kids[i]; float q = c->visits > 0 ? c->total_val/c->visits : 0.0f;
                    float r = (rave_v[c->move] < 1.0f ? 0.0f : rave_w[c->move] / rave_v[c->move]);
                    float beta = rave_v[c->move]/(rave_v[c->move]+c->visits+50.0f);
                    float u = MCTS_C_PUCT * n->prior[i] * sqrt_n / (1.0f + c->visits);
                    if ((1.0f-beta)*q + beta*r + u > best_score) { best_score = (1.0f-beta)*q + beta*r + u; best = i; } }
                node = node->kids[best]; } go_move(&s, node->move); }
        float v;
        if (go_over(&s)) { float sc = go_score(&s); v = (sc>0)?1.0f:(sc<0)?-1.0f:0.0f; v *= (float)(-node->to_play); }
        else {
            /* Virtual loss: penalize this path during evaluation to encourage exploration */
            node->visits += 3.0f; node->total_val -= 3.0f;
            float policy[MAX_MOVES]; uint64_t h = go_hash(&s); TTEntry *te = tt_lookup(h);
            if (te) { memset(policy,0,sizeof(policy)); for (int i=0;i<te->n;i++) policy[te->moves[i]]=te->policy[i]; nn_real vs; nn_real f[FEATURES]; go_features(&s,f); nn_forward(net,f,&vs); v=(float)vs*2.0f-1.0f; }
            else { go_evaluate(net, &s, policy, &v); tt_store(h, policy); }
            /* Remove virtual loss */
            node->visits -= 3.0f; node->total_val += 3.0f;
            float phase = go_stone_ratio(&s), ent = policy_entropy(policy);
            int top_k = ((phase<0.25f)?12:(phase>0.75f)?4:8) + (int)(ent*16.0f); if (top_k<MCTS_MIN_EXPAND) top_k=MCTS_MIN_EXPAND; if (top_k>30) top_k=30;
            typedef struct { int move; float prior; int tactical; } cand_t;
            cand_t cands[MAX_MOVES]; int nc=0; float maxp=0.001f;
            /* Reuse legality from go_evaluate if available, otherwise compute */
            for (int m=0;m<MAX_MOVES;m++) { if(!is_legal(&s,m))continue; int tac=is_tactical(&s,m); float pr=policy[m]; if(tac&&pr<0.01f)pr=0.01f; if(in_local_mask(&s,m))pr*=1.3f; if(pr>maxp)maxp=pr; cands[nc].move=m;cands[nc].prior=pr;cands[nc].tactical=tac;nc++; }
            for (int i=0;i<nc;i++) for (int j=i+1;j<nc;j++) if(cands[j].prior>cands[i].prior){cand_t t=cands[i];cands[i]=cands[j];cands[j]=t;}
            bool em_[MAX_MOVES]={false}; node->nkids=0; float pt=maxp*0.15f;
            for (int i=0;i<nc&&node->nkids<top_k;i++) if(cands[i].tactical||cands[i].prior>=pt){ int m=cands[i].move; GoNode*k=gn_alloc(m,-node->to_play,node); if(!k)break; node->kids[node->nkids]=k;node->prior[node->nkids]=cands[i].prior;node->nkids++;em_[m]=true; }
            for (int i=0;i<nc;i++) if(!em_[cands[i].move]&&cands[i].tactical&&node->nkids<MCTS_MAX_KIDS){ int m=cands[i].move; GoNode*k=gn_alloc(m,-node->to_play,node); if(!k)break; node->kids[node->nkids]=k;node->prior[node->nkids]=cands[i].prior;node->nkids++; }
            node->expanded=true; }
        while (node) { node->visits+=1.0f; node->total_val+=v; if(v<0.0f&&node->parent&&node->move>=0){rave_w[node->move]+=1.0f;rave_v[node->move]+=1.0f;} else if(node->parent&&node->move>=0)rave_v[node->move]+=1.0f; v=-v; node=node->parent; }
    }
    for (int i=0;i<root->nkids;i++) probs[root->kids[i]->move]=root->kids[i]->visits;
    float total=0; for (int m=0;m<MAX_MOVES;m++) total+=probs[m]; if(total>0) for(int m=0;m<MAX_MOVES;m++)probs[m]/=total;
    if (out_probs) memcpy(out_probs, probs, MAX_MOVES * sizeof(float));
    /* Temperature-based selection: first 10 moves use temp=1 for exploration */
    float temp = root_st->moves < 10 ? 1.0f : 0.1f;
    if (temp > 0.5f) {
        /* Sample from visit distribution using temperature */
        float weights[MAX_MOVES];
        float total_weight = 0;
        for (int m = 0; m < CELLS; m++) {
            weights[m] = powf(probs[m] + 1e-9f, 1.0f / temp);
            total_weight += weights[m];
        }
        if (total_weight > 0) {
            float r = (float)rand() / RAND_MAX * total_weight;
            float cum = 0;
            int best = CELLS;
            for (int m = 0; m < CELLS; m++) {
                cum += weights[m];
                if (r < cum) { best = m; break; }
            }
            if (best == CELLS) { for(int m=0;m<CELLS;m++) if(probs[m]>probs[best])best=m; }
            gn_free(root); return best;
        }
    }
    int best=CELLS; for(int m=0;m<CELLS;m++) if(probs[m]>probs[best])best=m;
    gn_free(root); return best;
}

static int go_mcts_move(NeuralNet *net, GoState *root_st) {
    return go_mcts_move_with_probs(net, root_st, NULL);
}


/* ── Data augmentation: D4 board symmetry (8x) ────────────────────────── */

/* Map cell index under D4 transformation t (0-7):
   0=identity, 1=rot90, 2=rot180, 3=rot270,
   4=reflect, 5=reflect+rot90, 6=reflect+rot180, 7=reflect+rot270 */
static int transform_cell(int i, int t) {
    int x = i % BOARD_SIZE, y = i / BOARD_SIZE;
    int nx, ny;
    if (t >= 4) { x = BOARD_SIZE - 1 - x; t -= 4; }  /* reflect */
    switch (t) {
        case 0: nx = x;     ny = y;     break;
        case 1: nx = y;     ny = BOARD_SIZE - 1 - x; break;
        case 2: nx = BOARD_SIZE - 1 - x; ny = BOARD_SIZE - 1 - y; break;
        case 3: nx = BOARD_SIZE - 1 - y; ny = x;     break;
        default: nx = x; ny = y; break;
    }
    return ny * BOARD_SIZE + nx;
}

/* Transform board features under D4 symmetry t.
   Features: 81 cells × 3 one-hot planes. Just permute the cell order. */
static void augment_features(const nn_real *src, nn_real *dst, int t) {
    memset(dst, 0, BOARD_FEATURES * sizeof(nn_real));
    for (int i = 0; i < CELLS; i++) {
        int j = transform_cell(i, t);
        for (int f = 0; f < CONV_FILTERS; f++)
            dst[j * CONV_FILTERS + f] = src[i * CONV_FILTERS + f];
    }
}

/* Transform move index under D4 symmetry.
   Pass (CELLS) is invariant. Regular moves rotate/reflect. */
static int augment_move(int move, int t) {
    if (move == CELLS) return CELLS;
    return transform_cell(move, t);
}


/* ── Training sample ─────────────────────────────────────────────────── */

typedef struct { nn_real features[FEATURES]; int move, to_play; float visit_prob; float mcts_entropy; float reward; } GameSample;

static float go_score_global(GameSample *samples, int n);

static int play_game(NeuralNet *black_net, NeuralNet *white_net, GameSample *samples, int max_samples, const char *b_name, const char *w_name) {
    GoState s; go_init(&s); int n = 0;
    if (!black_net || !white_net) { if (n == 0) n = -1; return n; }
    while (!go_over(&s) && n < max_samples) {
        GameSample *sp = &samples[n];
        go_features(&s, sp->features); sp->to_play = s.to_play;
        NeuralNet *net = (s.to_play == 1) ? black_net : white_net;
        float vprobs[MAX_MOVES];
        int move = go_mcts_move_with_probs(net, &s, vprobs);
        sp->features[SLOT_MOVE_ID] = (nn_real)move / MAX_MOVES; sp->move = move;
        sp->visit_prob = vprobs[move];
        /* Compute MCTS entropy for interest weighting */
        float ent = 0; int n_legal = 0;
        for (int m = 0; m < MAX_MOVES; m++) if (vprobs[m] > 0.001f) { ent -= vprobs[m] * logf(vprobs[m] + 1e-12f); n_legal++; }
        sp->mcts_entropy = n_legal > 1 ? ent / logf((float)n_legal) : 0.0f;
        go_move(&s, move);
        n++;
    }
    float score = go_score(&s);
    float winner = (score > 0) ? 1.0f : ((score < 0) ? -1.0f : 0.0f);
    for (int i = 0; i < n; i++) {
        float r = (samples[i].to_play == winner) ? 1.0f : (winner == 0) ? 0.5f : 0.0f;
        samples[i].reward = r;
    }
    return n;
}


#define NAME_LEN 32

static void ts_record_moves(int moves);
static void ts_snapshot(void);

/* ── Game history ────────────────────────────────────────────────────── */

#define HISTORY_SIZE 200
typedef struct {
    char black[NAME_LEN], white[NAME_LEN];
    float score;
    int move_count;
} GameRecord;

static GameRecord history[HISTORY_SIZE];
static int history_count = 0;


/* ── Game replay storage ─────────────────────────────────────────────── */

#define REPLAY_QUEUE 50
#define MAX_REPLAY_MOVES 250

typedef struct {
    char black[NAME_LEN], white[NAME_LEN];
    int moves[MAX_REPLAY_MOVES];
    int move_count;
    float final_score;
    bool ready;
} StoredReplay;

static StoredReplay replay_queue[REPLAY_QUEUE];
static int replay_write = 0;
static int replay_read = 0;
static int replay_pending = 0;

/* Current replay being viewed */
static StoredReplay current_replay;
static int current_replay_pos = -1;  /* -1 = no replay active */
static pthread_mutex_t replay_lock = PTHREAD_MUTEX_INITIALIZER;

static void replay_store(const char *b, const char *w, GameSample *samples, int n, float score) {
    pthread_mutex_lock(&replay_lock);
    int idx = replay_write % REPLAY_QUEUE;
    StoredReplay *r = &replay_queue[idx];
    snprintf(r->black, NAME_LEN, "%s", b);
    snprintf(r->white, NAME_LEN, "%s", w);
    r->move_count = n < MAX_REPLAY_MOVES ? n : MAX_REPLAY_MOVES;
    for (int i = 0; i < r->move_count; i++) r->moves[i] = samples[i].move;
    r->final_score = score;
    r->ready = true;
    replay_write++;
    replay_pending++;
    pthread_mutex_unlock(&replay_lock);
}

static void replay_pick_best(void) {
    pthread_mutex_lock(&replay_lock);
    if (replay_pending == 0) { pthread_mutex_unlock(&replay_lock); return; }
    /* Pick the longest game from pending */
    int best_idx = -1, best_len = 0;
    int start = replay_read % REPLAY_QUEUE;
    for (int i = 0; i < replay_pending && i < REPLAY_QUEUE; i++) {
        int idx = (start + i) % REPLAY_QUEUE;
        if (replay_queue[idx].ready && replay_queue[idx].move_count > best_len) {
            best_len = replay_queue[idx].move_count;
            best_idx = idx;
        }
    }
    if (best_idx >= 0) {
        current_replay = replay_queue[best_idx];
        current_replay_pos = 0;
        replay_queue[best_idx].ready = false;
    }
    /* Advance read pointer past consumed entries */
    while (replay_pending > 0 && !replay_queue[replay_read % REPLAY_QUEUE].ready) {
        replay_read++;
        replay_pending--;
    }
    pthread_mutex_unlock(&replay_lock);
}

static void record_game(const char *b, const char *w, float score, int moves) {
    GameRecord *r = &history[history_count % HISTORY_SIZE];
    snprintf(r->black, NAME_LEN, "%s", b);
    snprintf(r->white, NAME_LEN, "%s", w);
    r->score = score;
    r->move_count = moves;
    history_count++;
    ts_record_moves(moves);
}

/* ── ELO ─────────────────────────────────────────────────────────────── */

static void update_elo(double *elo_a, double *elo_b, int a_wins, int b_wins, int draws) {
    int total = a_wins + b_wins + draws;
    if (total == 0) return;
    double ea = 1.0 / (1.0 + pow(10.0, (*elo_b - *elo_a) / 400.0));
    double sa = (a_wins + 0.5 * draws) / total;
    double k = 32.0;
    *elo_a += k * (sa - ea);
    *elo_b += k * ((1.0 - sa) - (1.0 - ea));
}

/* ── Time series ──────────────────────────────────────────────────── */


static int num_threads = 1;
#define MAX_THREADS 16

/* ── Parallel game execution ─────────────────────────────────────────── */

typedef struct {
    NeuralNet *black_net, *white_net;
    const char *b_name, *w_name;
    GameSample *samples;
    int max_samples;
    int n_moves;
    float score;
    int done;
} GameJob;

static void *play_game_thread(void *arg) {
    GameJob *job = (GameJob *)arg;
    job->n_moves = play_game(job->black_net, job->white_net, job->samples, job->max_samples, job->b_name, job->w_name);
    job->score = go_score_global(job->samples, job->n_moves);
    job->done = 1;
    return NULL;
}

/* Play all games in a list using up to num_threads threads */
static void play_parallel(GameJob *jobs, int n_jobs) {
    int started = 0, completed = 0;
    pthread_t *threads = calloc(n_jobs, sizeof(pthread_t));
    
    while (completed < n_jobs) {
        /* Start new jobs if threads available */
        while (started < n_jobs && (started - completed) < num_threads) {
            jobs[started].done = 0;
            pthread_create(&threads[started], NULL, play_game_thread, &jobs[started]);
            started++;
        }
        /* Check for completed jobs */
        for (int i = completed; i < started; i++) {
            if (jobs[i].done) {
                pthread_join(threads[i], NULL);
                completed++;
            }
        }
        usleep(10000);  /* 10ms */
    }
    free(threads);
}

/* ── Tournament state ────────────────────────────────────────────────── */

#define MAX_PLAYERS 64
typedef struct {
    NeuralNet *net;
    NeuralNet *target_net;   /* frozen copy for stable training targets */
    char name[NAME_LEN];
    double elo;
    int wins, losses, draws, games;
    int replay_count;
    GameSample *replay;
    int games_since_target_update;
} Player;

static Player players[MAX_PLAYERS];
static int display_count = 4;  /* show top N on dashboard */
static int player_count = 0;
static Player *baseline = NULL;
static int games_per_pair = 1;
static int total_games = 0;

/* ── Training loss log ───────────────────────────────────────────────── */

#define LOSS_LOG_SIZE 200
typedef struct { char agent[NAME_LEN]; float loss; int samples; int game_id; } LossEntry;
static LossEntry loss_log[LOSS_LOG_SIZE];
static int loss_count = 0;

static void loss_record(const char *agent, float loss, int samples) {
    LossEntry *e = &loss_log[loss_count % LOSS_LOG_SIZE];
    snprintf(e->agent, NAME_LEN, "%s", agent);
    e->loss = loss; e->samples = samples; e->game_id = total_games;
    loss_count++;
}



/* ── Time series for charts ──────────────────────────────────────────── */

#define TS_MAX 500
typedef struct {
    int game_id;
    double elo[MAX_PLAYERS + 1];  /* one per player + baseline */
    int avg_moves;                 /* average moves over last N games */
} TSPoint;

static TSPoint ts[TS_MAX];
static int ts_count = 0;
static int ts_moves_sum = 0;
static int ts_moves_n = 0;

static void ts_snapshot(void) {
    if (ts_count >= TS_MAX) return;
    TSPoint *p = &ts[ts_count++];
    p->game_id = total_games;
    for (int i = 0; i < player_count; i++) p->elo[i] = players[i].elo;
    if (baseline) p->elo[player_count] = baseline->elo;
    p->avg_moves = ts_moves_n > 0 ? ts_moves_sum / ts_moves_n : 0;
    ts_moves_sum = 0; ts_moves_n = 0;
}

static void ts_record_moves(int moves) {
    ts_moves_sum += moves;
    ts_moves_n++;
}



static bool running = true;

static NeuralNet *create_net(unsigned seed) {
    size_t layers[] = {FEATURES, HIDDEN0, HIDDEN1, 1};
    NNConfig cfg = {0}; cfg.layer_sizes = layers; cfg.layer_count = 4; cfg.seed = seed;
    cfg.hidden_activation = NN_ACTIVATION_RELU; cfg.output_activation = NN_ACTIVATION_SIGMOID; cfg.loss = NN_LOSS_MSE;
    return nn_create_config(&cfg);
}

static void train_on_replay(Player *p) {
    if (p->replay_count < 32) return;
    /* Learning rate decay: starts at 0.005, decays to 0.001 over time */
    float lr = 0.005f / (1.0f + p->games * 0.0001f);
    if (lr < 0.001f) lr = 0.001f;
    int batch = p->replay_count < 128 ? p->replay_count : 128;
    float loss = 0;
    int trained = 0;

    NeuralNet *value_net = p->target_net;

    for (int i = 0; i < batch; i++) {
        int ri = rand() % p->replay_count;
        float r = p->replay[ri].reward;
        float visit_p = p->replay[ri].visit_prob;
        float ent = p->replay[ri].mcts_entropy;

        /* Interest weighting: prefer high-entropy (uncertain) and decisive positions */
        float interest = 0.3f + 0.4f * fabsf(r - 0.5f) + 0.3f * ent;
        if ((float)rand() / RAND_MAX > interest) continue;

        /* 8x data augmentation */
        for (int t = 0; t < 8; t++) {
            nn_real aug[FEATURES];
            augment_features(p->replay[ri].features, aug, t);
            int orig_move = (int)(p->replay[ri].features[SLOT_MOVE_ID] * MAX_MOVES + 0.5f);
            aug[SLOT_MOVE_ID] = (nn_real)augment_move(orig_move, t) / MAX_MOVES;
            /* DO NOT write the reward into aug — that was the old leakage path.
             * The reward is the target only; never the input. */

            /* MCTS policy distillation: blend win reward with visit probability */
            nn_real tgt = 0.7f * r + 0.3f * visit_p;

            /* For non-terminal draws, bootstrap from target network */
            if (r > 0.3f && r < 0.7f && fabsf(visit_p) < 0.1f) {
                nn_real vs;
                nn_forward(value_net, aug, &vs);
                tgt = 0.4f * r + 0.3f * vs + 0.3f * visit_p;
            }

            loss += (float)nn_train_sample(p->net, aug, &tgt, lr);
            trained++;
        }
    }

    if (trained > 0) {
        fprintf(stderr, "[%s] train loss=%.4f samples=%d (augmented %d)\n",
                p->name, loss/trained, batch, trained);
        loss_record(p->name, loss/trained, trained);
    }

    /* Update target network periodically */
    p->games_since_target_update++;
    if (p->games_since_target_update >= 20) {
        /* Copy weights from online net to target net */
        if (p->target_net) nn_free(p->target_net);
        p->target_net = nn_clone(p->net);
        if (!p->target_net) p->target_net = nn_clone(p->net);  /* retry once */
        p->games_since_target_update = 0;
        fprintf(stderr, "[%s] target network updated\n", p->name);
    }
}

/* ── HTTP status page ────────────────────────────────────────────────── */

static int http_port = 8087;

static void send_http(int fd, const char *ctype, const char *body) {
    char hdr[512];
    int len = (int)strlen(body);
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nContent-Type: %s\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        ctype, len);
    write(fd, hdr, hl); write(fd, body, len);
}

static void build_status_json(char *buf, int bufsz) {
    if (bufsz < 16384) return;
    int off = snprintf(buf, bufsz, "{\"games\":%d,\"players\":[", total_games);
    /* Collect all players including baseline, sort by ELO descending */
    typedef struct { Player *p; int is_baseline; } Ranked;
    Ranked ranked[MAX_PLAYERS + 1];
    int n_ranked = 0;
    for (int i = 0; i < player_count; i++) { ranked[n_ranked].p = &players[i]; ranked[n_ranked].is_baseline = 0; n_ranked++; }
    if (baseline) { ranked[n_ranked].p = baseline; ranked[n_ranked].is_baseline = 1; n_ranked++; }
    for (int i = 0; i < n_ranked; i++)
        for (int j = i+1; j < n_ranked; j++)
            if (ranked[j].p->elo > ranked[i].p->elo) { Ranked t = ranked[i]; ranked[i] = ranked[j]; ranked[j] = t; }
    /* Only show top display_count + baseline on dashboard */
    int show_count = 0;
    for (int i = 0; i < n_ranked; i++) {
        if (!ranked[i].is_baseline && show_count >= display_count) continue;
        show_count++;
        Player *p = ranked[i].p;
        int rank = ranked[i].is_baseline ? 99 : (show_count <= display_count ? show_count : 99);
        off += snprintf(buf + off, bufsz - off, "%s{\"name\":\"%s\",\"elo\":%.0f,\"w\":%d,\"l\":%d,\"d\":%d,\"games\":%d,\"ci\":%d,\"rank\":%d%s}",
            i ? "," : "", p->name, p->elo, p->wins, p->losses, p->draws, p->games,
            ranked[i].is_baseline ? 6 : (int)(p - players), rank,
            ranked[i].is_baseline ? ",\"baseline\":true" : "");
    }
    off += snprintf(buf + off, bufsz - off, "],\"history\":[");
    int h_start = history_count > HISTORY_SIZE ? history_count - HISTORY_SIZE : 0;
    int h_end = history_count;
    if (h_end - h_start > 50) h_start = h_end - 50;
    for (int i = h_start; i < h_end; i++) {
        GameRecord *r = &history[i % HISTORY_SIZE];
        const char *result = r->score > 0 ? r->black : r->score < 0 ? r->white : "draw";
        off += snprintf(buf + off, bufsz - off, "%s{\"b\":\"%s\",\"w\":\"%s\",\"s\":%.0f,\"m\":%d,\"r\":\"%s\"}",
            i > h_start ? "," : "", r->black, r->white, r->score, r->move_count, result);
    }
    off += snprintf(buf + off, bufsz - off, "],\"chart\":{\"labels\":[");
    for (int i = 0; i < ts_count; i++) {
        off += snprintf(buf + off, bufsz - off, "%s%d", i ? "," : "", ts[i].game_id);
    }
    off += snprintf(buf + off, bufsz - off, "],\"series\":[");
    /* One series per player + baseline */
    for (int p_idx = 0; p_idx < player_count + (baseline ? 1 : 0); p_idx++) {
        const char *pname = p_idx < player_count ? players[p_idx].name : baseline->name;
        int is_bl = p_idx >= player_count;
        int crank = is_bl ? 99 : (p_idx < display_count ? p_idx + 1 : 99);
        off += snprintf(buf + off, bufsz - off, "%s{\"name\":\"%s\",\"ci\":%d,\"rank\":%d,\"baseline\":%s,\"data\":[",
            p_idx ? "," : "", pname, is_bl ? 6 : p_idx, crank, is_bl ? "true" : "false");
        for (int i = 0; i < ts_count; i++) {
            off += snprintf(buf + off, bufsz - off, "%s%.0f", i ? "," : "", ts[i].elo[p_idx]);
        }
        off += snprintf(buf + off, bufsz - off, "]}");
    }
    /* Moves series */
    off += snprintf(buf + off, bufsz - off, ",{\"name\":\"avg moves\",\"moves\":true,\"data\":[");
    for (int i = 0; i < ts_count; i++) {
        off += snprintf(buf + off, bufsz - off, "%s%d", i ? "," : "", ts[i].avg_moves);
    }
    off += snprintf(buf + off, bufsz - off, "]}]},\"loss_log\":[");
    int l_start = loss_count > LOSS_LOG_SIZE ? loss_count - LOSS_LOG_SIZE : 0;
    int l_end = loss_count;
    if (l_end - l_start > 80) l_start = l_end - 80;
    for (int i = l_start; i < l_end; i++) {
        LossEntry *e = &loss_log[i % LOSS_LOG_SIZE];
        off += snprintf(buf + off, bufsz - off, "%s{\"agent\":\"%s\",\"loss\":%.4f,\"samples\":%d,\"game\":%d}",
            i > l_start ? "," : "", e->agent, e->loss, e->samples, e->game_id);
    }
    off += snprintf(buf + off, bufsz - off, "]");
    off += snprintf(buf + off, bufsz - off, "}");
}

static const char *status_html =
"<!DOCTYPE html><html><head><meta charset=utf-8><title>Tournament</title><style>"
"*{margin:0;padding:0;box-sizing:border-box}"
"body{font-family:Menlo,monospace;background:#0f0f1a;color:#c8d6e5;padding:12px}"
"h1{font-size:.85rem;margin-bottom:2px}.sub{color:#556;font-size:.65rem;margin-bottom:8px}"
"#tb{font-size:.7rem;margin-bottom:10px}"
"#tb div{display:flex;gap:10px;padding:1px 0;border-bottom:1px solid #1a1a33}"
"#tb div span{min-width:36px;text-align:right}"
"#tb .name{min-width:54px;text-align:left}"
".elo{color:#fc3}.w{color:#5a5}.l{color:#a55}.d{color:#556}"
"#main{display:flex;gap:14px;align-items:flex-start}"
"#replay{border:1px solid #2a2a44;border-radius:3px;padding:6px}"
"#replay .hdr{color:#fc3;font-size:.65rem;margin-bottom:3px}"
"#replay .board{font-size:10px;line-height:1.12;white-space:pre;font-family:Menlo,monospace}"
"canvas{display:block;margin:3px 0;border:1px solid #1a1a33;border-radius:3px}"
"</style></head><body>"
"<h1>Go Tournament</h1><div class=sub id=info></div>"
"<div id=tb></div>"
"<div id=main>"
"<div id=replay><div class=hdr>Replay</div><div class=board id=rb></div></div>"
"<div><canvas id=elo width=580 height=180></canvas><canvas id=mov width=580 height=80></canvas></div>"
"</div>"
"<script>"
"let C=['#fc3','#5af','#f5a','#5fa','#a5f','#fa5','#ccc'],lastData=null,rows=[];"
"function drawChart(canvas,labels,series,ylabel){"
" let c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;"
" c.clearRect(0,0,W,H);if(!labels.length)return;"
" let pad=40,right=16,top=12,bot=20,w=W-pad-right,h=H-top-bot;"
" let allData=series.flatMap(s=>s.data);"
" let yMin=Math.min(...allData),yMax=Math.max(...allData);"
" if(yMin==yMax){yMin-=50;yMax+=50}"
" c.strokeStyle='#1a1a33';c.lineWidth=.5;"
" for(let i=0;i<=4;i++){let y=top+h*(1-i/4);c.beginPath();c.moveTo(pad,y);c.lineTo(W-right,y);c.stroke();"
"  c.fillStyle='#556';c.font='9px Menlo';c.fillText(Math.round(yMin+(yMax-yMin)*i/4),2,y+4);}"
" c.fillStyle='#556';c.textAlign='right';c.fillText(ylabel,W-2,top-6);c.textAlign='start';"
" for(let si=0;si<series.length;si++){"
"  let s=series[si],col=C[s.ci!=null?s.ci:si];"
"  let alpha=s.rank>4&&!s.baseline?0.15:1.0;"
"  c.globalAlpha=alpha;c.strokeStyle=col;c.lineWidth=s.moves?1:1.5;c.beginPath();"
"  for(let i=0;i<s.data.length;i++){"
"   let x=pad+(w/(labels.length-1||1))*i;"
"   let y=top+h-(s.data[i]-yMin)/(yMax-yMin)*h;"
"   i==0?c.moveTo(x,y):c.lineTo(x,y);"
"  }"
"  c.stroke();c.globalAlpha=1.0;"
"  if(!s.moves){c.fillStyle=col;c.font=\'9px Menlo\';c.globalAlpha=alpha;c.fillText(s.name,pad+w-5+3,top+h-((s.data[s.data.length-1]-yMin)/(yMax-yMin))*h+3);c.globalAlpha=1.0;}"
" }}"
"function updateTable(players){"
" let tb=document.getElementById('tb');"
" if(!rows.length)for(let i=0;i<players.length;i++){"
"  let d=document.createElement('div');"
"  d.innerHTML='<span style=color:#556;min-width:16px></span><span class=name></span><span class=elo></span><span class=w></span><span class=l></span><span class=d></span><span></span>';"
"  tb.appendChild(d);rows.push(d);}"
" for(let i=0;i<players.length;i++){"
"  let p=players[i],sp=rows[i].children;"
"  sp[0].textContent='#'+(i+1);"
"  let alpha2=p.rank>4&&!p.baseline?0.25:1.0;"
"  sp[1].innerHTML=(p.baseline?'* ':'')+'<span style=color:'+(C[p.ci]||'#ccc')+';opacity:'+alpha2+'>'+p.name+'</span>';"
"  sp[2].textContent=Math.round(p.elo);sp[3].textContent=p.w+'W';"
"  sp[4].textContent=p.l+'L';sp[5].textContent=p.d+'D';sp[6].textContent=p.games+'g';"
" }}"
"function jsonEq(a,b){return JSON.stringify(a)==JSON.stringify(b)}"
"async function poll(){"
" let r=await fetch('/status');let d=await r.json();"
" document.getElementById('info').textContent=d.games+' games, '+d.players.length+' shown of total';"
" if(!jsonEq(d.players,lastData)){lastData=d.players;updateTable(d.players);}"
" if(d.chart){"
"  let eloSeries=d.chart.series.filter(s=>!s.moves),movSeries=d.chart.series.filter(s=>s.moves);"
"  drawChart(document.getElementById('elo'),d.chart.labels,eloSeries,'ELO');"
"  drawChart(document.getElementById('mov'),d.chart.labels,movSeries,'moves/game');"
" }"
"}"
"async function replayPoll(){"
" let r=await fetch('/replay');let d=await r.json();"
" if(d.wait)return;"
" let bcol=C[d.bci]||'#ccc',wcol=C[d.wci]||'#ddd',cols='ABCDEFGHJ';"
" let h='<b style=color:'+bcol+'>'+d.black+'</b> vs <b style=color:'+wcol+'>'+d.white+'</b>  m'+d.move+'/'+d.total+'<br>';"
" h+=' ';for(let x=0;x<9;x++)h+=cols[x];h+='<br>';"
" for(let y=0;y<9;y++){h+=(y+1)+(y<8?' ':'')+' ';"
"  for(let x=0;x<9;x++){let i=y*9+x,v=d.board[i];"
"   h+=v==1?'<b style=color:'+bcol+'>X</b>':v==-1?'<b style=color:'+wcol+'>O</b>':'.';}"
"  h+=' '+(y+1)+'<br>';}"
" h+=' ';for(let x=0;x<9;x++)h+=cols[x];"
" if(d.move>=d.total)h+='<br>Final: '+(d.score>0?d.black:d.score<0?d.white:'draw')+' wins by '+Math.abs(d.score);"
" document.getElementById('rb').innerHTML=h;"
"}"
"setInterval(replayPoll,100);replayPoll();"
"setInterval(poll,1000);poll();"
"</script></body></html>";
static void *http_loop(void *arg) {
    (void)arg;
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return NULL;
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)http_port), .sin_addr.s_addr = INADDR_ANY};
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("[status] bind"); close(srv); return NULL; }
    listen(srv, 8);
    fprintf(stderr, "[status] listening on http://127.0.0.1:%d\n", http_port); fflush(stderr);

/* ── ELO persistence ────────────────────────────────────────────────── */




    while (running) {
        int fd = accept(srv, NULL, NULL); if (fd < 0) continue;
        char buf[2048] = {0}; read(fd, buf, sizeof(buf)-1);
        if (strstr(buf, "/live")) {
            send_http(fd, "application/json", "{}");
        } else if (strstr(buf, "/replay")) {
            pthread_mutex_lock(&replay_lock);
            /* If no replay active, pick the best pending one */
            if (current_replay_pos < 0) {
                pthread_mutex_unlock(&replay_lock);
                replay_pick_best();
                pthread_mutex_lock(&replay_lock);
            }
            char json[16384];
            int off = 0;
            if (current_replay_pos >= 0 && current_replay_pos < current_replay.move_count) {
                /* Reconstruct board up to current position */
                GoState rs; go_init(&rs);
                for (int i = 0; i <= current_replay_pos && i < current_replay.move_count; i++) {
                    go_move(&rs, current_replay.moves[i]);
                }
                /* Find color indices */
                int rbci = 0, rwci = 0;
                for (int pi = 0; pi < player_count; pi++) {
                    if (strcmp(players[pi].name, current_replay.black) == 0) rbci = pi;
                    if (strcmp(players[pi].name, current_replay.white) == 0) rwci = pi;
                }
                if (baseline && strcmp(baseline->name, current_replay.black) == 0) rbci = 6;
                if (baseline && strcmp(baseline->name, current_replay.white) == 0) rwci = 6;
                off = snprintf(json, sizeof(json), "{\"black\":\"%s\",\"white\":\"%s\",\"move\":%d,\"total\":%d,\"score\":%.0f,\"bci\":%d,\"wci\":%d,\"board\":[",
                    current_replay.black, current_replay.white, current_replay_pos + 1, current_replay.move_count, current_replay.final_score, rbci, rwci);
                for (int i = 0; i < CELLS; i++) off += snprintf(json + off, sizeof(json) - off, "%s%d", i ? "," : "", rs.board[i]);
                off += snprintf(json + off, sizeof(json) - off, "],\"last\":%d,\"to_play\":%d}", rs.last_move, rs.to_play);
                current_replay_pos++;
                if (current_replay_pos >= current_replay.move_count) current_replay_pos = -1; /* done */
            } else {
                off = snprintf(json, sizeof(json), "{\"wait\":true}");
            }
            pthread_mutex_unlock(&replay_lock);
            send_http(fd, "application/json", json);
        } else if (strstr(buf, "/status")) {
            char json[16384]; build_status_json(json, sizeof(json));
            send_http(fd, "application/json", json);
        } else {
            send_http(fd, "text/html", status_html);
        }
        close(fd);
    }
    close(srv);
    return NULL;
}

static void handle_signal(int sig) {
    fprintf(stderr, "\n[tournament] signal %d, saving...\n", sig);
    for (int i = 0; i < player_count; i++) {
        char path[64];
        snprintf(path, sizeof(path), "build/float/tournament_%s.nnckpt", players[i].name);
        NNCheckpointMeta meta; nn_checkpoint_meta_init(&meta, 1);
        nn_save_checkpoint(players[i].net, path, &meta);
    }
    running = false;
}

/* ── ELO persistence ────────────────────────────────────────────────── */



static void save_elo(void) {
    FILE *f = fopen("build/float/tournament_elo.txt", "w");
    if (!f) return;
    for (int i = 0; i < player_count; i++)
        fprintf(f, "%s %.1f %d %d %d %d\n", players[i].name, players[i].elo,
                players[i].wins, players[i].losses, players[i].draws, players[i].games);
    if (baseline) fprintf(f, "%s %.1f %d %d %d %d\n", baseline->name, baseline->elo,
                baseline->wins, baseline->losses, baseline->draws, baseline->games);
    fclose(f);
}
static void load_elo(void) {
    FILE *f = fopen("build/float/tournament_elo.txt", "r");
    if (!f) return;
    char name[NAME_LEN]; double elo; int w, l, d, g;
    while (fscanf(f, "%s %lf %d %d %d %d", name, &elo, &w, &l, &d, &g) == 6) {
        for (int i = 0; i < player_count; i++)
            if (strcmp(players[i].name, name) == 0) {
                players[i].elo=elo; players[i].wins=w; players[i].losses=l; players[i].draws=d; players[i].games=g;
            }
        if (baseline && strcmp(baseline->name, name) == 0) {
            baseline->elo=elo; baseline->wins=w; baseline->losses=l; baseline->draws=d; baseline->games=g;
        }
    }
    fclose(f);
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int num_players = 4;
    const char *baseline_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--players") == 0 && i+1 < argc) num_players = atoi(argv[++i]);
        else if (strcmp(argv[i], "--agents") == 0 && i+1 < argc) num_players = atoi(argv[++i]);
        else if (strcmp(argv[i], "--display") == 0 && i+1 < argc) display_count = atoi(argv[++i]);
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc) games_per_pair = atoi(argv[++i]);
        else if (strcmp(argv[i], "--baseline") == 0 && i+1 < argc) baseline_path = argv[++i];
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) { num_threads = atoi(argv[++i]); if (num_threads > MAX_THREADS) num_threads = MAX_THREADS; if (num_threads < 1) num_threads = 1; }
        else if (strcmp(argv[i], "--port") == 0 && i+1 < argc) http_port = atoi(argv[++i]);
    }
    if (num_players > MAX_PLAYERS) num_players = MAX_PLAYERS;
    if (num_players < 2) num_players = 2;

    srand((unsigned)time(NULL));
    conv_init((unsigned)time(NULL));
    conv_init((unsigned)time(NULL));
    load_elo();

    /* Create evolving players */
    for (int i = 0; i < num_players; i++) {
        Player *p = &players[player_count++];
        snprintf(p->name, NAME_LEN, "Alpha%d", i+1);
        /* Try loading existing checkpoint */
        char ckpt_path[64];
        snprintf(ckpt_path, sizeof(ckpt_path), "build/float/tournament_%s.nnckpt", p->name);
        NNCheckpointMeta meta;
        NeuralNet *loaded = nn_load_checkpoint(ckpt_path, &meta);
        if (loaded && nn_input_size(loaded) == FEATURES && nn_output_size(loaded) == 1) {
            p->net = loaded;
            fprintf(stderr, "[%s] loaded %s\n", p->name, ckpt_path);
        } else {
            p->net = create_net((unsigned)(time(NULL) + i * 1000));
            if (loaded) nn_free(loaded);
            fprintf(stderr, "[%s] fresh network\n", p->name);
        }
        p->target_net = nn_clone(p->net);
        if (!p->target_net) { fprintf(stderr, "[%s] FATAL: target network clone failed\n", p->name); }
        p->replay = calloc(REPLAY_CAPACITY, sizeof(GameSample));
        p->elo = 1500.0;
    }

    /* Load or create baseline */
    Player bl;
    memset(&bl, 0, sizeof(bl));
    bl.replay = calloc(REPLAY_CAPACITY, sizeof(GameSample));
    baseline = &bl;
    snprintf(baseline->name, NAME_LEN, "Baseline");
    baseline->elo = 1500.0;
    if (baseline_path) {
        NNCheckpointMeta meta;
        baseline->net = nn_load_checkpoint(baseline_path, &meta);
    }
    if (!baseline->net) {
        baseline->net = create_net(42);
        fprintf(stderr, "[baseline] using fresh frozen network\n");
    } else {
        fprintf(stderr, "[baseline] loaded %s\n", baseline_path);
    }

    /* Start HTTP status server */
    pthread_t http_thread;
    pthread_create(&http_thread, NULL, http_loop, NULL);

    /* Tournament loop */
fprintf(stderr, "[tournament] %d players, %d games/pair, %d threads\n", player_count, games_per_pair, num_threads);


/* ── ELO persistence ────────────────────────────────────────────────── */




    while (running) {
        /* Collect all game jobs for this round */
        int max_jobs = player_count * (player_count - 1) * games_per_pair + player_count;
        GameJob *jobs = calloc(max_jobs, sizeof(GameJob));
        GameSample **job_samples = calloc(max_jobs, sizeof(GameSample*));
        int n_jobs = 0;

        /* ELO-proximity pairing: each agent plays K nearby opponents */
        int pairs_per_agent = 3;  /* play 3 opponents per round */
        if (pairs_per_agent > player_count - 1) pairs_per_agent = player_count - 1;
        
        for (int a = 0; a < player_count && running; a++) {
            /* Find nearest ELO opponents that haven't been paired with 'a' this round */
            int paired = 0;
            double best_dist[3] = {1e9, 1e9, 1e9};
            int best_opp[3] = {-1, -1, -1};
            
            for (int b = 0; b < player_count; b++) {
                if (a == b) continue;
                double dist = fabs(players[a].elo - players[b].elo);
                /* Insert into sorted best_dist */
                for (int k = 0; k < pairs_per_agent; k++) {
                    if (dist < best_dist[k]) {
                        for (int m = pairs_per_agent-1; m > k; m--) {
                            best_dist[m] = best_dist[m-1];
                            best_opp[m] = best_opp[m-1];
                        }
                        best_dist[k] = dist; best_opp[k] = b;
                        break;
                    }
                }
            }
            
            for (int k = 0; k < pairs_per_agent && best_opp[k] >= 0; k++) {
                int b = best_opp[k];
                /* Only play each pair once per round (a < b to avoid duplicates) */
                if (a >= b) continue;
                for (int g = 0; g < games_per_pair; g++) {
                    job_samples[n_jobs] = calloc(MAX_GAME_MOVES, sizeof(GameSample));
                    jobs[n_jobs] = (GameJob){players[a].target_net, players[b].target_net, players[a].name, players[b].name,
                        job_samples[n_jobs], MAX_GAME_MOVES, 0, 0, 0};
                    n_jobs++;
                    job_samples[n_jobs] = calloc(MAX_GAME_MOVES, sizeof(GameSample));
                    jobs[n_jobs] = (GameJob){players[b].target_net, players[a].target_net, players[b].name, players[a].name,
                        job_samples[n_jobs], MAX_GAME_MOVES, 0, 0, 0};
                    n_jobs++;
                }
            }
        }
        for (int i = 0; i < player_count; i++) {
            job_samples[n_jobs] = calloc(MAX_GAME_MOVES, sizeof(GameSample));
            jobs[n_jobs] = (GameJob){players[i].target_net, baseline->net, players[i].name, baseline->name,
                job_samples[n_jobs], MAX_GAME_MOVES, 0, 0, 0};
            n_jobs++;
        }

        play_parallel(jobs, n_jobs);

        for (int j = 0; j < n_jobs; j++) {
            GameJob *job = &jobs[j];
            int n = job->n_moves;
            float score = job->score;
            replay_store(job->b_name, job->w_name, job_samples[j], n, score);
            record_game(job->b_name, job->w_name, score, n);

            int a_idx = -1, b_idx = -1;
            for (int pi = 0; pi < player_count; pi++) {
                if (strcmp(players[pi].name, job->b_name) == 0) a_idx = pi;
                if (strcmp(players[pi].name, job->w_name) == 0) b_idx = pi;
            }
            if (a_idx >= 0 && b_idx >= 0) {
                if (score > 0) { players[a_idx].wins++; players[b_idx].losses++; }
                else if (score < 0) { players[b_idx].wins++; players[a_idx].losses++; }
                else { players[a_idx].draws++; players[b_idx].draws++; }
                players[a_idx].games++; players[b_idx].games++;
                total_games++;
                update_elo(&players[a_idx].elo, &players[b_idx].elo, score>0?1:0, score<0?1:0, score==0?1:0);
            } else {
                int pi = -1;
                for (int pk = 0; pk < player_count; pk++) if (strcmp(players[pk].name, job->b_name) == 0) pi = pk;
                if (pi >= 0) {
                    if (score > 0) { players[pi].wins++; baseline->losses++; }
                    else if (score < 0) { baseline->wins++; players[pi].losses++; }
                    else { players[pi].draws++; baseline->draws++; }
                    players[pi].games++; baseline->games++;
                    total_games++;
                }
            }

            if (score != 0 && a_idx >= 0 && b_idx >= 0) {
                Player *winner = score > 0 ? &players[a_idx] : &players[b_idx];
                for (int i = 0; i < n && winner->replay_count < REPLAY_CAPACITY; i++) {
                    memcpy(winner->replay[winner->replay_count].features, job_samples[j][i].features, FEATURES*sizeof(nn_real));
                    winner->replay[winner->replay_count].move = job_samples[j][i].move;
                    winner->replay[winner->replay_count].to_play = job_samples[j][i].to_play;
                    winner->replay[winner->replay_count].visit_prob = job_samples[j][i].visit_prob;
                    winner->replay[winner->replay_count].mcts_entropy = job_samples[j][i].mcts_entropy;
                    float r = (job_samples[j][i].to_play == (score > 0 ? 1.0f : -1.0f)) ? 1.0f : 0.0f;
                    winner->replay[winner->replay_count].reward = r;
                    winner->replay_count++;
                }
            }
            free(job_samples[j]);
        }
        free(jobs);
        free(job_samples);

        /* Synchronous training between rounds */
        static int train_round = 0;
        train_round++;
        if (train_round % 3 == 0) {
            for (int i = 0; i < player_count; i++) train_on_replay(&players[i]);
        }

        save_elo();
        ts_snapshot();
        fprintf(stderr, "[tournament] round complete, %d total games, %d threads\n", total_games, num_threads);

        /* Save checkpoints */
        for (int i = 0; i < player_count; i++) {
            char path[64];
            snprintf(path, sizeof(path), "build/float/tournament_%s.nnckpt", players[i].name);
            NNCheckpointMeta meta; nn_checkpoint_meta_init(&meta, 1);
            nn_save_checkpoint(players[i].net, path, &meta);
        }
    }
    return 0;
}

/* Helper for score from samples — uses the per-sample reward field (NOT the
 * input vector) and treats r == 0.5 explicitly as a draw so a drawn game is
 * no longer misclassified as a decisive ±1 win/loss. */
static float go_score_global(GameSample *samples, int n) {
    if (n == 0) return 0.0f;
    float r = samples[n-1].reward;
    int last_player = samples[n-1].to_play;
    if (fabsf(r - 0.5f) < 1e-3f) return 0.0f;          /* draw */
    if (r > 0.5f) return last_player == 1 ? 1.0f : -1.0f;
    return last_player == 1 ? -1.0f : 1.0f;
}
