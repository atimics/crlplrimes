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
#include <pthread.h>

#define BOARD_SIZE 9
#define CELLS (BOARD_SIZE * BOARD_SIZE)
#define MAX_MOVES (CELLS + 1)
#define BOARD_FEATURES (CELLS * 3)
#define SLOT_MOVE_ID    (BOARD_FEATURES)
#define SLOT_REWARD     (BOARD_FEATURES + 1)
#define FEATURES        (BOARD_FEATURES + 2)
#define HIDDEN0 128
#define HIDDEN1 64

#define MCTS_SIMULATIONS 15
#define MCTS_C_PUCT 1.4f
#define MCTS_MAX_KIDS 82
#define MCTS_MIN_EXPAND 2
#define TT_SIZE 8192
#define MAX_GAME_MOVES 200
#define REPLAY_CAPACITY 20000

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
static int in_bounds(int i) { return i >= 0 && i < CELLS; }
static int on_left(int i)  { return i % BOARD_SIZE == 0; }
static int on_right(int i) { return i % BOARD_SIZE == BOARD_SIZE - 1; }

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
static int go_over(GoState *s) { return s->passes >= 2 || s->moves >= MAX_GAME_MOVES; }
static float go_score(GoState *s) {
    int black = 0, white = 0; int8_t vis[CELLS] = {0};
    for (int i = 0; i < CELLS; i++) { if (s->board[i] == 1) black++; else if (s->board[i] == -1) white++; else if (!vis[i]) {
        int q[CELLS], h = 0, t = 0, hb = 0, hw = 0, cnt = 0; q[t++] = i; vis[i] = 1;
        while (h < t) { int idx = q[h++]; cnt++; for (int d = 0; d < 4; d++) { int ni = idx + DIRS[d];
            if ((d == 1 && on_left(idx)) || (d == 2 && on_right(idx))) continue; if (!in_bounds(ni)) continue;
            if (s->board[ni] == 1) hb = 1; else if (s->board[ni] == -1) hw = 1; else if (!vis[ni]) { vis[ni] = 1; q[t++] = ni; } } }
        if (hb && !hw) black += cnt; if (hw && !hb) white += cnt; } }
    return (float)(black - white);
}
static float go_stone_ratio(GoState *s) { int n = 0; for (int i = 0; i < CELLS; i++) if (s->board[i] != 0) n++; return (float)n / CELLS; }

static void go_features(GoState *s, nn_real row[FEATURES]) {
    memset(row, 0, BOARD_FEATURES * sizeof(nn_real));
    for (int i = 0; i < CELLS; i++) { int off = i * 3; if (s->board[i] == 1) row[off] = 1.0f; else if (s->board[i] == -1) row[off+1]=1.0f; else row[off+2]=1.0f; }
}

static void go_evaluate(NeuralNet *net, GoState *s, float policy[MAX_MOVES], float *value) {
    nn_real feats[FEATURES]; go_features(s, feats); float total = 0.0f;
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

static float rave_w[CELLS+1], rave_v[CELLS+1];
static void rave_reset(void) { memset(rave_w, 0, sizeof(rave_w)); memset(rave_v, 0, sizeof(rave_v)); }
static float rave_q(int move) { return rave_v[move] < 1.0f ? 0.0f : rave_w[move] / rave_v[move]; }

typedef struct { uint64_t hash; float policy[12]; int8_t moves[12]; int n; bool valid; } TTEntry;
static TTEntry tt[TT_SIZE];
static TTEntry *tt_lookup(uint64_t hash) { int idx = hash % TT_SIZE; return (tt[idx].valid && tt[idx].hash == hash) ? &tt[idx] : NULL; }
static void tt_store(uint64_t hash, float policy[MAX_MOVES]) { int idx = hash % TT_SIZE; if (tt[idx].valid && tt[idx].hash == hash) return;
    tt[idx].hash=hash; tt[idx].valid=true; tt[idx].n=0; for (int m=0; m<MAX_MOVES && tt[idx].n<12; m++) if (policy[m]>0.001f) { tt[idx].moves[tt[idx].n]=(int8_t)m; tt[idx].policy[tt[idx].n]=policy[m]; tt[idx].n++; } }

static int gn_select(GoNode *n) { int best = 0; float best_score = -1e30f, sqrt_n = sqrtf(n->visits + 1e-8f);
    for (int i = 0; i < n->nkids; i++) { GoNode *c = n->kids[i]; float q = c->visits > 0 ? c->total_val/c->visits : 0.0f;
        float r = rave_q(c->move); float beta = rave_v[c->move]/(rave_v[c->move]+c->visits+50.0f);
        float u = MCTS_C_PUCT * n->prior[i] * sqrt_n / (1.0f + c->visits);
        if ((1.0f-beta)*q + beta*r + u > best_score) { best_score = (1.0f-beta)*q + beta*r + u; best = i; } }
    return best; }

static uint64_t go_hash(GoState *s) { return s->zhash; }

static int go_mcts_move(NeuralNet *net, GoState *root_st) {
    float probs[MAX_MOVES] = {0};
    GoNode *root = gn_alloc(-1, root_st->to_play, NULL); if (!root) return CELLS;
    rave_reset();
    for (int sim = 0; sim < MCTS_SIMULATIONS; sim++) {
        GoState s = *root_st; GoNode *node = root;
        while (node->expanded && node->nkids > 0 && !go_over(&s)) { node = node->kids[gn_select(node)]; go_move(&s, node->move); }
        float v;
        if (go_over(&s)) { float sc = go_score(&s); v = (sc>0)?1.0f:(sc<0)?-1.0f:0.0f; v *= (float)(-node->to_play); }
        else { float policy[MAX_MOVES]; uint64_t h = go_hash(&s); TTEntry *te = tt_lookup(h);
            if (te) { memset(policy,0,sizeof(policy)); for (int i=0;i<te->n;i++) policy[te->moves[i]]=te->policy[i]; nn_real vs; nn_real f[FEATURES]; go_features(&s,f); nn_forward(net,f,&vs); v=(float)vs*2.0f-1.0f; }
            else { go_evaluate(net, &s, policy, &v); tt_store(h, policy); }
            float phase = go_stone_ratio(&s), ent = policy_entropy(policy);
            int top_k = ((phase<0.25f)?12:(phase>0.75f)?4:8) + (int)(ent*16.0f); if (top_k<MCTS_MIN_EXPAND) top_k=MCTS_MIN_EXPAND; if (top_k>30) top_k=30;
            typedef struct { int move; float prior; int tactical; } cand_t;
            cand_t cands[MAX_MOVES]; int nc=0; float maxp=0.001f;
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
    int best=CELLS; for(int m=0;m<CELLS;m++) if(probs[m]>probs[best])best=m;
    gn_free(root); return best;
}

/* ── Training sample ─────────────────────────────────────────────────── */

typedef struct { nn_real features[FEATURES]; int move, to_play; } GameSample;

static float go_score_global(GameSample *samples, int n);

static int play_game(NeuralNet *black_net, NeuralNet *white_net, GameSample *samples, int max_samples) {
    GoState s; go_init(&s); int n = 0;
    while (!go_over(&s) && n < max_samples) {
        GameSample *sp = &samples[n];
        go_features(&s, sp->features); sp->to_play = s.to_play;
        NeuralNet *net = (s.to_play == 1) ? black_net : white_net;
        int move = go_mcts_move(net, &s);
        sp->features[SLOT_MOVE_ID] = (nn_real)move / MAX_MOVES; sp->move = move;
        go_move(&s, move); n++;
    }
    float score = go_score(&s);
    float winner = (score > 0) ? 1.0f : ((score < 0) ? -1.0f : 0.0f);
    for (int i = 0; i < n; i++) {
        float r = (samples[i].to_play == winner) ? 1.0f : (winner == 0) ? 0.5f : 0.0f;
        samples[i].features[SLOT_REWARD] = r;
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

/* ── Tournament state ────────────────────────────────────────────────── */

#define MAX_PLAYERS 8
typedef struct {
    NeuralNet *net;
    char name[NAME_LEN];
    double elo;
    int wins, losses, draws, games;
    int replay_count;
    GameSample *replay;
} Player;

static Player players[MAX_PLAYERS];
static int player_count = 0;
static Player *baseline = NULL;
static int games_per_pair = 1;
static int total_games = 0;

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
    float lr = 0.005f;
    int batch = p->replay_count < 128 ? p->replay_count : 128;
    float loss = 0;
    for (int i = 0; i < batch; i++) {
        int ri = rand() % p->replay_count;
        nn_real tgt = p->replay[ri].features[SLOT_REWARD];
        loss += (float)nn_train_sample(p->net, p->replay[ri].features, &tgt, lr);
    }
    if (batch > 0) fprintf(stderr, "[%s] train loss=%.4f samples=%d\n", p->name, loss/batch, batch);
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
    int off = snprintf(buf, bufsz, "{\"games\":%d,\"players\":[", total_games);
    /* Collect all players including baseline, sort by ELO descending */
    typedef struct { Player *p; int is_baseline; } Ranked;
    Ranked ranked[MAX_PLAYERS + 1];
    int n_ranked = 0;
    for (int i = 0; i < player_count; i++) { ranked[n_ranked].p = &players[i]; ranked[n_ranked].is_baseline = 0; n_ranked++; }
    if (baseline) { ranked[n_ranked].p = baseline; ranked[n_ranked].is_baseline = 1; n_ranked++; }
    /* Simple sort by ELO descending */
    for (int i = 0; i < n_ranked; i++)
        for (int j = i+1; j < n_ranked; j++)
            if (ranked[j].p->elo > ranked[i].p->elo) { Ranked t = ranked[i]; ranked[i] = ranked[j]; ranked[j] = t; }
    for (int i = 0; i < n_ranked; i++) {
        Player *p = ranked[i].p;
        off += snprintf(buf + off, bufsz - off, "%s{\"name\":\"%s\",\"elo\":%.0f,\"w\":%d,\"l\":%d,\"d\":%d,\"games\":%d%s}",
            i ? "," : "", p->name, p->elo, p->wins, p->losses, p->draws, p->games,
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
        off += snprintf(buf + off, bufsz - off, "%s{\"name\":\"%s\",\"baseline\":%s,\"data\":[",
            p_idx ? "," : "", pname, is_bl ? "true" : "false");
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
    off += snprintf(buf + off, bufsz - off, "]}]}");
    off += snprintf(buf + off, bufsz - off, "}");
}

static const char *status_html =
"<!DOCTYPE html><html><head><meta charset=utf-8><title>Tournament</title><style>"
"*{margin:0;padding:0;box-sizing:border-box}"
"body{font-family:Menlo,monospace;background:#0f0f1a;color:#c8d6e5;padding:16px}"
"h1{font-size:.9rem;margin-bottom:2px}.sub{color:#556;font-size:.7rem;margin-bottom:6px}"
"#tb{font-size:.73rem;margin-bottom:12px}"
"#tb div{display:flex;gap:12px;padding:2px 0;border-bottom:1px solid #1a1a33}"
"#tb div span{min-width:42px;text-align:right}"
"#tb .name{min-width:62px;text-align:left}"
".elo{color:#fc3}.w{color:#5a5}.l{color:#a55}.d{color:#556}"
"canvas{display:block;margin:8px 0;border:1px solid #1a1a33;border-radius:3px}"
"#log{font-size:.68rem;max-height:160px;overflow-y:auto}"
"#log div{padding:1px 0;border-bottom:1px solid #111}"
"#log .win{color:#5a5}#log .lose{color:#a55}#log .draw{color:#556}"
"</style></head><body>"
"<h1>Go Tournament</h1><div class=sub id=info></div>"
"<div id=tb></div>"
"<canvas id=elo width=720 height=220></canvas>"
"<canvas id=mov width=720 height=100></canvas>"
"<div id=log></div>"
"<script>"
"let C=['#fc3','#5af','#f5a','#5fa','#a5f','#fa5','#ccc'],lastH=0,lastData=null,rows=[];"
"function drawChart(canvas,labels,series,ylabel){"
" let c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;"
" c.clearRect(0,0,W,H);if(!labels.length)return;"
" let pad=40,right=20,top=12,bot=20,w=W-pad-right,h=H-top-bot;"
" let allData=series.flatMap(s=>s.data);"
" let yMin=Math.min(...allData),yMax=Math.max(...allData);"
" if(yMin==yMax){yMin-=50;yMax+=50}"
" c.strokeStyle='#1a1a33';c.lineWidth=.5;"
" for(let i=0;i<=4;i++){let y=top+h*(1-i/4);c.beginPath();c.moveTo(pad,y);c.lineTo(W-right,y);c.stroke();"
"  c.fillStyle='#556';c.font='9px Menlo';c.fillText(Math.round(yMin+(yMax-yMin)*i/4),2,y+4);}"
" c.fillStyle='#556';c.textAlign='right';c.fillText(ylabel,W-2,top-6);c.textAlign='start';"
" for(let si=0;si<series.length;si++){"
"  let s=series[si],col=C[si%C.length];"
"  c.strokeStyle=col;c.lineWidth=s.moves?1:1.5;c.beginPath();"
"  for(let i=0;i<s.data.length;i++){"
"   let x=pad+(w/(labels.length-1||1))*i;"
"   let y=top+h-(s.data[i]-yMin)/(yMax-yMin)*h;"
"   i==0?c.moveTo(x,y):c.lineTo(x,y);"
"  }"
"  c.stroke();"
"  if(!s.moves){c.fillStyle=col;c.font='9px Menlo';c.fillText(s.name,pad+w-5+3,top+h-((s.data[s.data.length-1]-yMin)/(yMax-yMin))*h+3);}"
" }"
"}"
"function updateTable(players){"
" let tb=document.getElementById('tb');"
" if(!rows.length)for(let i=0;i<players.length;i++){"
"  let d=document.createElement('div');d.innerHTML='<span style=color:#556;min-width:18px></span><span class=name></span><span class=elo></span><span class=w></span><span class=l></span><span class=d></span><span></span>';"
"  tb.appendChild(d);rows.push(d);}"
" for(let i=0;i<players.length;i++){"
"  let p=players[i],sp=rows[i].children;"
"  sp[0].textContent='#'+(i+1);sp[1].textContent=(p.baseline?'* ':'')+p.name;sp[2].textContent=Math.round(p.elo);"
"  sp[3].textContent=p.w+'W';sp[4].textContent=p.l+'L';sp[5].textContent=p.d+'D';sp[6].textContent=p.games+'g';"
" }"
"}"
"function jsonEq(a,b){return JSON.stringify(a)==JSON.stringify(b)}"
"async function poll(){"
" let r=await fetch('/status');let d=await r.json();"
" document.getElementById('info').textContent=d.games+' games';"
" if(!jsonEq(d.players,lastData)){lastData=d.players;updateTable(d.players);}"
" if(d.chart){"
"  let eloSeries=d.chart.series.filter(s=>!s.moves),movSeries=d.chart.series.filter(s=>s.moves);"
"  drawChart(document.getElementById('elo'),d.chart.labels,eloSeries,'ELO');"
"  drawChart(document.getElementById('mov'),d.chart.labels,movSeries,'moves/game');"
" }"
" if(d.history){"
"  let log=document.getElementById('log');"
"  for(let i=lastH;i<d.history.length;i++){"
"   let g=d.history[i],cls=g.r==g.b?'win':g.r==g.w?'lose':'draw';"
"   let div=document.createElement('div');div.className=cls;div.textContent=g.b+' vs '+g.w+' \u2192 '+g.r+' ('+g.m+'m)';"
"   log.appendChild(div);"
"  }"
"  while(log.children.length>100)log.removeChild(log.firstChild);"
"  log.scrollTop=log.scrollHeight;"
" }"
" lastH=d.history?d.history.length:0;"
"}"
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
    while (running) {
        int fd = accept(srv, NULL, NULL); if (fd < 0) continue;
        char buf[2048] = {0}; read(fd, buf, sizeof(buf)-1);
        if (strstr(buf, "/status")) {
            char json[4096]; build_status_json(json, sizeof(json));
            send_http(fd, "application/json", json);
        } else {
            send_http(fd, "text/html", status_html);
        }
        close(fd);
    }
    close(srv);
    return NULL;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int num_players = 4;
    const char *baseline_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--players") == 0 && i+1 < argc) num_players = atoi(argv[++i]);
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc) games_per_pair = atoi(argv[++i]);
        else if (strcmp(argv[i], "--baseline") == 0 && i+1 < argc) baseline_path = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i+1 < argc) http_port = atoi(argv[++i]);
    }
    if (num_players > MAX_PLAYERS) num_players = MAX_PLAYERS;
    if (num_players < 2) num_players = 2;

    srand((unsigned)time(NULL));

    /* Create evolving players */
    for (int i = 0; i < num_players; i++) {
        Player *p = &players[player_count++];
        snprintf(p->name, NAME_LEN, "Alpha%d", i+1);
        p->net = create_net((unsigned)(time(NULL) + i * 1000));
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
    GameSample *samples = calloc(MAX_GAME_MOVES, sizeof(GameSample));
    fprintf(stderr, "[tournament] %d players, %d games/pair\n", player_count, games_per_pair);

    while (running) {
        /* Round-robin: each pair plays */
        for (int a = 0; a < player_count && running; a++) {
            for (int b = a + 1; b < player_count && running; b++) {
                for (int g = 0; g < games_per_pair; g++) {
                    /* a=Black, b=White */
                    int n = play_game(players[a].net, players[b].net, samples, MAX_GAME_MOVES);
                    float score = go_score_global(samples, n);
                    record_game(players[a].name, players[b].name, score, n);
                    if (score > 0) { players[a].wins++; players[b].losses++; }
                    else if (score < 0) { players[b].wins++; players[a].losses++; }
                    else { players[a].draws++; players[b].draws++; }
                    players[a].games++; players[b].games++;
                    total_games++;
                    update_elo(&players[a].elo, &players[b].elo, score>0?1:0, score<0?1:0, score==0?1:0);

                    /* Store game data for training */
                    if (score != 0) {
                        Player *winner = score > 0 ? &players[a] : &players[b];
                        for (int i = 0; i < n && winner->replay_count < REPLAY_CAPACITY; i++) {
                            memcpy(winner->replay[winner->replay_count].features, samples[i].features, FEATURES*sizeof(nn_real));
                            winner->replay[winner->replay_count].move = samples[i].move;
                            winner->replay[winner->replay_count].to_play = samples[i].to_play;
                            /* Assign reward: 1.0 for winner's moves, 0.0 for loser's */
                            float r = (samples[i].to_play == (score > 0 ? 1.0f : -1.0f)) ? 1.0f : 0.0f;
                            winner->replay[winner->replay_count].features[SLOT_REWARD] = r;
                            winner->replay_count++;
                        }
                    }

                    /* b=Black, a=White (swap colors) */
                    n = play_game(players[b].net, players[a].net, samples, MAX_GAME_MOVES);
                    score = go_score_global(samples, n);
                    record_game(players[b].name, players[a].name, score, n);
                    if (score > 0) { players[b].wins++; players[a].losses++; }
                    else if (score < 0) { players[a].wins++; players[b].losses++; }
                    else { players[a].draws++; players[b].draws++; }
                    players[a].games++; players[b].games++;
                    total_games++;
                    update_elo(&players[a].elo, &players[b].elo, score<0?1:0, score>0?1:0, score==0?1:0);

                    if (score != 0) {
                        Player *winner = score > 0 ? &players[b] : &players[a];
                        for (int i = 0; i < n && winner->replay_count < REPLAY_CAPACITY; i++) {
                            memcpy(winner->replay[winner->replay_count].features, samples[i].features, FEATURES*sizeof(nn_real));
                            float r = (samples[i].to_play == (score > 0 ? 1.0f : -1.0f)) ? 1.0f : 0.0f;
                            winner->replay[winner->replay_count].features[SLOT_REWARD] = r;
                            winner->replay_count++;
                        }
                    }
                }
            }
        }

        /* Train each player on their replay buffer */
        for (int i = 0; i < player_count; i++) {
            train_on_replay(&players[i]);
        }

        /* Baseline evaluation: each player vs baseline */
        for (int i = 0; i < player_count; i++) {
            int n = play_game(players[i].net, baseline->net, samples, MAX_GAME_MOVES);
            float score = go_score_global(samples, n);
            record_game(players[i].name, baseline->name, score, n);
            if (score > 0) { players[i].wins++; baseline->losses++; }
            else if (score < 0) { baseline->wins++; players[i].losses++; }
            else { players[i].draws++; baseline->draws++; }
            players[i].games++; baseline->games++;
            total_games++;
        }

        ts_snapshot();
        fprintf(stderr, "[tournament] round complete, %d total games\n", total_games);

        /* Save checkpoints */
        for (int i = 0; i < player_count; i++) {
            char path[64];
            snprintf(path, sizeof(path), "build/float/tournament_%s.nnckpt", players[i].name);
            NNCheckpointMeta meta; nn_checkpoint_meta_init(&meta, 1);
            nn_save_checkpoint(players[i].net, path, &meta);
        }
    }

    running = false;
    pthread_join(http_thread, NULL);
    free(samples);
    return 0;
}

/* Helper for score from samples */
static float go_score_global(GameSample *samples, int n) {
    if (n == 0) return 0.0f;
    float r = samples[n-1].features[SLOT_REWARD];
    int last_player = samples[n-1].to_play;
    if (r > 0.5f) return last_player == 1 ? 1.0f : -1.0f;
    if (r < 0.5f && r > 0.0f) return 0.0f;
    return last_player == 1 ? -1.0f : 1.0f;
}
