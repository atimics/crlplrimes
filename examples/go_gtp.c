/**
 * go_gtp.c — Continuous-learning GTP engine for CGOS.
 *
 * Speaks GTPv2 on stdin/stdout. Accumulates self-play samples from
 * every CGOS game. Trains between games. Network improves over time.
 *
 * Usage: ./go_gtp [--load checkpoint.nnckpt] [--save checkpoint.nnckpt]
 */

/* pthread_rwlock_* and strcasecmp are XSI / BSD extensions; the project builds
 * with -std=c11 strict, so enable the GNU feature set explicitly here. */
#define _GNU_SOURCE

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>
#include <ctype.h>
#include <pthread.h>
#include <unistd.h>

#define BOARD_SIZE 9
#define CELLS (BOARD_SIZE * BOARD_SIZE)
#define MAX_MOVES (CELLS + 1)
/* Board encoding: CELLS * 3 one-hot planes + 1 scratch slot for per-move query id.
 * The reward / training target is tracked separately (ReplayEntry.reward); it is
 * never stored in the input vector, so the network cannot learn the trivial
 * identity-from-input shortcut. */
#define BOARD_FEATURES (CELLS * 3)
#define SLOT_MOVE_ID    (BOARD_FEATURES)
#define FEATURES        (BOARD_FEATURES + 1)
#define HIDDEN0 128
#define HIDDEN1 64
#define MAX_GAME_MOVES 200
#define REPLAY_BUFFER 50000  /* max stored positions */
#define TRAIN_BATCH 256
#define TRAIN_EPOCHS 2
#define TRAIN_LR 0.005f

#define MCTS_SIMULATIONS 200
#define MCTS_C_PUCT 1.4f
#define MCTS_MAX_KIDS 82
#define MCTS_MIN_EXPAND 2
#define TT_SIZE 8192

/* ── Go engine (same as go_selfplay.c, condensed) ─────────────────────── */

static const int DIRS[4] = {-BOARD_SIZE, -1, 1, BOARD_SIZE};

/* Zobrist side-to-move constant */
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
    /* Private LCG */
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

static int is_atari(GoState *s, int i, int8_t color) { (void)color;
    return count_liberties(s, i) == 1;
}

/* is_suicide: placing at idx would leave own group with 0 liberties. */
static int is_suicide(GoState *s, int idx) {
    GoState cp = *s;
    cp.board[idx] = cp.to_play;
    int opp = -cp.to_play;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) return 0;
    }
    return group_dead(&cp, idx, cp.to_play);
}

/* True self-atari: placing at idx does not capture but leaves own group
   with exactly 1 liberty. */
static int is_self_atari(GoState *s, int idx) {
    GoState cp = *s;
    cp.board[idx] = cp.to_play;
    int opp = -cp.to_play;
    int captured = 0;
    for (int d = 0; d < 4; d++) {
        int ni = idx + DIRS[d];
        if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue;
        if (in_bounds(ni) && cp.board[ni] == opp && group_dead(&cp, ni, opp)) {
            captured = 1;
            int8_t vis[CELLS] = {0};
            remove_group(&cp, ni, vis);
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
    return memcmp(cp.board, s->prev, sizeof(s->prev)) != 0;
}

static void go_move(GoState *s, int idx) {
    s->last_move = idx;
    if (idx == CELLS) {
        s->passes++; s->to_play = -s->to_play;
        s->zhash ^= zobrist_side;
        return;
    }
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
            int8_t vis[CELLS] = {0};
            remove_group(s, ni, vis);
        }
    }
    s->to_play = -s->to_play; s->moves++;
}

static int go_over(GoState *s) { return s->passes >= 2 || s->moves >= MAX_GAME_MOVES; }

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

/* ── features ─────────────────────────────────────────────────────────── */

static void go_features(GoState *s, nn_real row[FEATURES]) {
    /* Zero ALL feature bytes so SLOT_MOVE_ID is initialized — callers that
     * don't set it (e.g., the value-head forward) would otherwise read stack
     * garbage from that slot. */
    memset(row, 0, FEATURES * sizeof(nn_real));
    for (int i = 0; i < CELLS; i++) {
        int off = i * 3;
        if (s->board[i] == 1) row[off] = 1.0f;
        else if (s->board[i] == -1) row[off + 1] = 1.0f;
        else row[off + 2] = 1.0f;
    }
    /* SLOT_MOVE_ID left zero — filled by caller for per-move policy queries. */
}

/* ── neural evaluation ────────────────────────────────────────────────── */

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

/* ── MCTS (condensed from go_selfplay.c) ──────────────────────────────── */

typedef struct GoNode {
    int move, to_play;
    struct GoNode *parent, *kids[MCTS_MAX_KIDS];
    int nkids;
    float prior[MCTS_MAX_KIDS], visits, total_val;
    bool expanded;
} GoNode;

static GoNode *gn_new(int move, int to_play, GoNode *parent) {
    GoNode *n = calloc(1, sizeof(GoNode));
    if (n == NULL) return NULL;
    n->move = move; n->to_play = to_play; n->parent = parent; return n;
}
static void gn_free(GoNode *n) {
    if (!n) return;
    for (int i = 0; i < n->nkids; i++) gn_free(n->kids[i]);
    free(n);
}

static float rave_w[CELLS+1], rave_v[CELLS+1];

static void go_mcts(NeuralNet *net, GoState *rs, float probs[MAX_MOVES]) {
    memset(rave_w, 0, sizeof(rave_w)); memset(rave_v, 0, sizeof(rave_v));
    GoNode *root = gn_new(-1, rs->to_play, NULL);
    if (root == NULL) return;

    for (int sim = 0; sim < MCTS_SIMULATIONS; sim++) {
        GoState s = *rs; GoNode *node = root;
        while (node->expanded && node->nkids > 0 && !go_over(&s)) {
            int best = 0; float best_sc = -1e30f, sq = sqrtf(node->visits + 1e-8f);
            for (int i = 0; i < node->nkids; i++) {
                GoNode *c = node->kids[i];
                float q = c->visits > 0 ? c->total_val / c->visits : 0.0f;
                float u = MCTS_C_PUCT * node->prior[i] * sq / (1.0f + c->visits);
                if (q + u > best_sc) { best_sc = q + u; best = i; }
            }
            node = node->kids[best]; go_move(&s, node->move);
        }
        
        float v;
        if (go_over(&s)) {
            /* terminal — use final board state as proxy */
            float sc = go_score(&s);
            v = (sc > 0) ? 1.0f : (sc < 0) ? -1.0f : 0.0f;
            v *= (float)(-node->to_play);
        } else {
            float policy[MAX_MOVES], val;
            go_evaluate(net, &s, policy, &val);
            v = val;
            node->nkids = 0;
            float maxp = 0.001f;
            for (int m = 0; m < MAX_MOVES; m++)
                if (policy[m] > maxp) maxp = policy[m];
            for (int m = 0; m < MAX_MOVES; m++) {
                if (policy[m] < maxp * 0.15f && !is_tactical(&s, m)) continue;
                if (!is_legal(&s, m)) continue;
                if (node->nkids >= MCTS_MAX_KIDS) break;
                GoNode *kid = gn_new(m, -node->to_play, node);
                if (kid == NULL) break;
                node->kids[node->nkids] = kid;
                node->prior[node->nkids] = policy[m];
                node->nkids++;
            }
            node->expanded = true;
        }
        while (node) {
            node->visits += 1.0f; node->total_val += v; v = -v;
            if (node->move >= 0) { rave_w[node->move] += (v < 0 ? 1.0f : 0.0f); rave_v[node->move] += 1.0f; }
            node = node->parent;
        }
    }
    for (int i = 0; i < root->nkids; i++) probs[root->kids[i]->move] = root->kids[i]->visits;
    float total = 0;
    for (int m = 0; m < MAX_MOVES; m++) total += probs[m];
    if (total > 0) for (int m = 0; m < MAX_MOVES; m++) probs[m] /= total;
    gn_free(root);
}

static int go_mcts_move(NeuralNet *net, GoState *s) {
    float probs[MAX_MOVES] = {0};
    go_mcts(net, s, probs);
    int best = CELLS;
    for (int m = 0; m < CELLS; m++)
        if (probs[m] > probs[best]) best = m;
    return best;
}

/* ── replay buffer ────────────────────────────────────────────────────── */

typedef struct {
    nn_real features[FEATURES];
    float reward;
} ReplayEntry;

static ReplayEntry replay[REPLAY_BUFFER];
static int replay_count = 0;
static pthread_mutex_t replay_lock = PTHREAD_MUTEX_INITIALIZER;

/* Ring-buffer add: wraps once full, never drops */
static void replay_add(nn_real *feats, float reward) {
    pthread_mutex_lock(&replay_lock);
    int idx = replay_count % REPLAY_BUFFER;
    memcpy(replay[idx].features, feats, FEATURES * sizeof(nn_real));
    replay[idx].reward = reward;
    replay_count++;
    pthread_mutex_unlock(&replay_lock);
}

/* ── game state (for tracking CGOS games) ─────────────────────────────── */

static GoState game_state;
static nn_real game_moves[MAX_GAME_MOVES][FEATURES];
static int game_move_count = 0;
static int game_our_color = 0;  /* which color we're playing */
static bool game_active = false;
static float game_outcome = 0.0f;

/* ── training thread ──────────────────────────────────────────────────── */

static bool train_running = false;
static pthread_t train_thread;
static NeuralNet *train_net;
static const char *save_path = NULL;
static pthread_rwlock_t net_lock = PTHREAD_RWLOCK_INITIALIZER;

static int validate_net_dims(NeuralNet *net) {
    return nn_input_size(net) == FEATURES && nn_output_size(net) == 1;
}

static void *train_loop(void *arg) {
    (void)arg;
    while (train_running) {
        pthread_mutex_lock(&replay_lock);
        int n = replay_count;
        pthread_mutex_unlock(&replay_lock);
        
        if (n >= TRAIN_BATCH) {
            /* Snapshot replay under lock, then train unlocked so replay_add
             * in the GTP path never stalls on the mutex. */
            int snap_n = n < REPLAY_BUFFER ? n : REPLAY_BUFFER;
            static ReplayEntry snap[REPLAY_BUFFER];
            pthread_mutex_lock(&replay_lock);
            int start = replay_count > REPLAY_BUFFER ? replay_count % REPLAY_BUFFER : 0;
            for (int i = 0; i < snap_n; i++)
                snap[i] = replay[(start + i) % REPLAY_BUFFER];
            pthread_mutex_unlock(&replay_lock);

            pthread_rwlock_wrlock(&net_lock);
            float lr = TRAIN_LR;
            for (int e = 0; e < TRAIN_EPOCHS; e++) {
                float loss = 0;
                for (int i = 0; i < snap_n; i++) {
                    nn_real tgt = snap[i].reward;
                    loss += (float)nn_train_sample(train_net, snap[i].features, &tgt, lr);
                }
                if (snap_n > 0) fprintf(stderr, "[train] epoch %d loss=%.4f samples=%d\n", e+1, loss/snap_n, snap_n);
            }
            pthread_rwlock_unlock(&net_lock);
            
            if (save_path) {
                NNCheckpointMeta meta;
                nn_checkpoint_meta_init(&meta, 1);
                nn_save_checkpoint(train_net, save_path, &meta);
                fprintf(stderr, "[train] saved %s\n", save_path);
            }
        }
        sleep(5);  /* check every 5 seconds */
    }
    return NULL;
}


/* ── GTP with game tracking ───────────────────────────────────────────── */

static NeuralNet *gtp_net;
static void gtp_out(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("= "); vprintf(fmt, ap); printf("\n\n"); fflush(stdout);
    va_end(ap);
}
static void gtp_err(const char *msg) { printf("? %s\n\n", msg); fflush(stdout); }

static void gtp_boardsize(int n) {
    if (n != BOARD_SIZE) { gtp_err("only 9x9"); return; }
    go_init(&game_state); game_move_count = 0; game_active = true; game_outcome = 0.0f;
    gtp_out("");
}

static void gtp_clear_board(void) {
    /* Train on the completed game before clearing */
    if (game_move_count > 0 && game_active && game_our_color != 0) {
        float outcome = game_outcome;
        for (int i = 0; i < game_move_count; i++) {
            float r = (game_our_color == 1) ? outcome : -outcome;
            replay_add(game_moves[i], r);
        }
        fprintf(stderr, "[gtp] game ended outcome=%.0f, %d moves added to replay\n",
                (double)outcome, game_move_count);
    }
    go_init(&game_state); game_move_count = 0; game_active = true; game_outcome = 0.0f;
    gtp_out("");
}

static void gtp_play(const char *color, const char *vertex) {
    int to_play = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
    if (to_play != game_state.to_play) { gtp_err("wrong turn"); return; }
    
    int move = CELLS;
    if (strcasecmp(vertex, "pass") && strcasecmp(vertex, "resign")) {
        int x = toupper(vertex[0]) - 'A';
        if (x > 8) x--;
        int y = atoi(vertex + 1) - 1;
        if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) move = xy(x, y);
        else { gtp_err("invalid vertex"); return; }
    }
    if (!is_legal(&game_state, move)) { gtp_err("illegal move"); return; }
    
    /* Record opponent's move as a training sample */
    if (game_our_color != 0 && to_play != game_our_color && game_move_count < MAX_GAME_MOVES) {
        go_features(&game_state, game_moves[game_move_count]);
        game_moves[game_move_count][SLOT_MOVE_ID] = (nn_real)move / MAX_MOVES;
        game_move_count++;
    }
    
    go_move(&game_state, move);

    if (go_over(&game_state)) {
        float sc = go_score(&game_state);
        game_outcome = (sc > 0) ? 1.0f : (sc < 0) ? -1.0f : 0.0f;
        if (game_our_color != 0 && game_our_color == -1) game_outcome = -game_outcome;
    }

    gtp_out("");
}

static void gtp_genmove(const char *color) {
    int to_play = (color[0] == 'b' || color[0] == 'B') ? 1 : -1;
    if (to_play != game_state.to_play) { gtp_err("wrong turn"); return; }
    
    /* Remember which color we are */
    game_our_color = to_play;
    
    /* Record our position before the move */
    if (game_move_count < MAX_GAME_MOVES) {
        go_features(&game_state, game_moves[game_move_count]);
    }
    
    pthread_rwlock_rdlock(&net_lock);
    int move = go_mcts_move(gtp_net, &game_state);
    pthread_rwlock_unlock(&net_lock);

    /* Store our move as training data */
    if (game_move_count < MAX_GAME_MOVES) {
        game_moves[game_move_count][SLOT_MOVE_ID] = (nn_real)move / MAX_MOVES;
        game_move_count++;
    }
    
    go_move(&game_state, move);

    if (go_over(&game_state)) {
        float sc = go_score(&game_state);
        game_outcome = (sc > 0) ? 1.0f : (sc < 0) ? -1.0f : 0.0f;
        if (game_our_color != 0 && game_our_color == -1) game_outcome = -game_outcome;
    }

    if (move == CELLS) gtp_out("PASS");
    else {
        char buf[4];
        int col = move % BOARD_SIZE;
        if (col >= 8) col++;
        snprintf(buf, sizeof(buf), "%c%d", 'A' + col, (move / BOARD_SIZE) + 1);
        gtp_out("%s", buf);
    }
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *load_path = NULL;
    save_path = "build/float/go_gtp.nnckpt";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) load_path = argv[++i];
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];
    }
    
    const size_t layers[] = {FEATURES, HIDDEN0, HIDDEN1, 1};
    NNConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.layer_sizes = layers; cfg.layer_count = 4; cfg.seed = (unsigned)time(NULL);
    cfg.hidden_activation = NN_ACTIVATION_RELU;
    cfg.output_activation = NN_ACTIVATION_SIGMOID;
    cfg.loss = NN_LOSS_MSE;
    
    gtp_net = nn_create_config(&cfg);
    train_net = gtp_net;  /* same net, trained in background */

    if (load_path) {
        fprintf(stderr, "[gtp] loading checkpoint %s...\n", load_path);
        NNCheckpointMeta meta;
        NeuralNet *loaded = nn_load_checkpoint(load_path, &meta);
        if (loaded) {
            if (validate_net_dims(loaded)) {
                nn_free(gtp_net);
                gtp_net = loaded;
                train_net = loaded;
                fprintf(stderr, "[gtp] checkpoint loaded (input=%zu output=%zu)\n",
                        nn_input_size(loaded), nn_output_size(loaded));
            } else {
                fprintf(stderr, "[gtp] checkpoint dimension mismatch (got input=%zu output=%zu, expected input=%d output=1), using fresh net\n",
                        nn_input_size(loaded), nn_output_size(loaded), FEATURES);
                nn_free(loaded);
            }
        } else {
            fprintf(stderr, "[gtp] load failed, using untrained net\n");
        }
    }

    go_init(&game_state);
    game_active = true;
    
    /* Start background training thread */
    train_running = true;
    pthread_create(&train_thread, NULL, train_loop, NULL);
    
    fprintf(stderr, "[gtp] Continuous-learning GTP engine ready\n");
    fprintf(stderr, "[gtp] Save: %s  Replay buffer: %d\n", save_path, REPLAY_BUFFER);
    
    /* GTP main loop */
    char line[256];
    while (fgets(line, sizeof(line), stdin)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        
        char cmd[64], a1[64], a2[64];
        cmd[0] = a1[0] = a2[0] = '\0';
        int n = sscanf(p, "%63s %63s %63s", cmd, a1, a2);

        if (n < 1) continue;
        if (!strcmp(cmd, "quit")) { gtp_out(""); break; }
        else if (!strcmp(cmd, "protocol_version")) gtp_out("2");
        else if (!strcmp(cmd, "name")) gtp_out("SparseMCTS-Go-Continuous");
        else if (!strcmp(cmd, "version")) gtp_out("0.2");
        else if (!strcmp(cmd, "boardsize")) gtp_boardsize(atoi(a1));
        else if (!strcmp(cmd, "clear_board")) gtp_clear_board();
        else if (!strcmp(cmd, "play") && n >= 3) gtp_play(a1, a2);
        else if (!strcmp(cmd, "genmove") && n >= 2) gtp_genmove(a1);
        else if (!strcmp(cmd, "known_command")) gtp_out("true");
        else if (!strcmp(cmd, "list_commands")) gtp_out("protocol_version\nname\nversion\nboardsize\nclear_board\nplay\ngenmove\nquit");
        else if (!strcmp(cmd, "kgs-genmove_cleanup") && n >= 2) gtp_genmove(a1);
        else gtp_err("unknown command");
    }
    
    train_running = false;
    pthread_join(train_thread, NULL);
    nn_free(gtp_net);
    return 0;
}
