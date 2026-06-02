/**
 * go_experiment.c — Go position experiment under crlp.experiment_schema.v1.
 */
#include "nn.h"
#include "nn_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#define GO_BOARD_SIZE 9
#define GO_CELLS (GO_BOARD_SIZE * GO_BOARD_SIZE)
#define GO_MAX_MOVES (GO_CELLS + 1)
#define GO_MAX_GAME_MOVES 150
static const int GO_DIRS[4] = {-GO_BOARD_SIZE, -1, 1, GO_BOARD_SIZE};

typedef struct {
    int8_t board[GO_CELLS]; int to_play, moves, passes; int8_t prev[GO_CELLS];
} GoBoard;

static int go_on_left(int i)  { return i % GO_BOARD_SIZE == 0; }
static int go_on_right(int i) { return i % GO_BOARD_SIZE == GO_BOARD_SIZE - 1; }
static int go_in_bounds(int i) { return i >= 0 && i < GO_CELLS; }
static void go_init_board(GoBoard *b) { memset(b, 0, sizeof(*b)); b->to_play = 1; }

static int go_count_liberties(const GoBoard *b, int i) {
    int8_t c = b->board[i]; if (!c) return 0;
    int8_t lb[GO_CELLS]={0}, vs[GO_CELLS]={0}; int q[GO_CELLS], h=0, t=0, cnt=0;
    q[t++]=i; vs[i]=1;
    while (h<t) { int x=q[h++];
        for (int d=0;d<4;d++) { int n=x+GO_DIRS[d];
            if ((d==1&&go_on_left(x))||(d==2&&go_on_right(x))) continue;
            if (!go_in_bounds(n)) continue;
            if (!b->board[n]&&!lb[n]) { lb[n]=1; cnt++; }
            else if (b->board[n]==c&&!vs[n]) { vs[n]=1; q[t++]=n; }}}
    return cnt;
}

static void go_remove_group(GoBoard *b, int idx) {
    int8_t c=b->board[idx], vs[GO_CELLS]={0}; int q[GO_CELLS], h=0, t=0;
    q[t++]=idx; vs[idx]=1;
    while (h<t) { int x=q[h++]; b->board[x]=0;
        for (int d=0;d<4;d++) { int n=x+GO_DIRS[d];
            if ((d==1&&go_on_left(x))||(d==2&&go_on_right(x))) continue;
            if (go_in_bounds(n)&&!vs[n]&&b->board[n]==c) { vs[n]=1; q[t++]=n; }}}
}

static int go_group_dead(const GoBoard *b, int i) {
    return b->board[i]!=0 && go_count_liberties(b,i)==0;
}

static int go_is_legal(const GoBoard *b, int idx) {
    if (idx==GO_CELLS) return 1;
    if (!go_in_bounds(idx)||b->board[idx]) return 0;
    GoBoard cp=*b; cp.board[idx]=cp.to_play;
    int opp=-cp.to_play;
    for (int d=0;d<4;d++) { int n=idx+GO_DIRS[d];
        if ((d==1&&go_on_left(idx))||(d==2&&go_on_right(idx))) continue;
        if (go_in_bounds(n)&&cp.board[n]==opp&&go_group_dead(&cp,n))
            go_remove_group(&cp,n); }
    if (go_group_dead(&cp,idx)) return 0;
    if (!memcmp(cp.board,b->prev,sizeof(b->prev))) return 0;
    return 1;
}

static void go_apply_move(GoBoard *b, int idx) {
    if (idx==GO_CELLS) { b->passes++; b->to_play=-b->to_play; return; }
    memcpy(b->prev,b->board,sizeof(b->board)); b->passes=0;
    b->board[idx]=b->to_play; int opp=-b->to_play;
    for (int d=0;d<4;d++) { int n=idx+GO_DIRS[d];
        if ((d==1&&go_on_left(idx))||(d==2&&go_on_right(idx))) continue;
        if (go_in_bounds(n)&&b->board[n]==opp&&go_group_dead(b,n))
            go_remove_group(b,n); }
    b->to_play=-b->to_play; b->moves++;
}

static int go_game_over(const GoBoard *b) {
    return b->passes>=2 || b->moves>=GO_MAX_GAME_MOVES;
}

static float go_territory_score(const GoBoard *b) {
    int8_t ow[GO_CELLS]={0}, vs[GO_CELLS]={0};
    for (int i=0;i<GO_CELLS;i++) {
        if (b->board[i]) { ow[i]=b->board[i]; continue; }
        if (vs[i]) continue;
        int q[GO_CELLS],h=0,t=0,hb=0,hw=0;
        q[t++]=i; vs[i]=1;
        while (h<t) { int x=q[h++];
            for (int d=0;d<4;d++) { int n=x+GO_DIRS[d];
                if ((d==1&&go_on_left(x))||(d==2&&go_on_right(x))) continue;
                if (!go_in_bounds(n)) continue;
                if (b->board[n]==1) hb=1; else if (b->board[n]==-1) hw=1;
                else if (!vs[n]) { vs[n]=1; q[t++]=n; }}}
        int8_t o=(hb&&!hw)?1:(!hb&&hw)?-1:0;
        memset(vs,0,sizeof(vs)); h=0;t=0;q[t++]=i;vs[i]=1;
        while (h<t) { int x=q[h++];
            if (!b->board[x]) ow[x]=o;
            for (int d=0;d<4;d++) { int n=x+GO_DIRS[d];
                if ((d==1&&go_on_left(x))||(d==2&&go_on_right(x))) continue;
                if (go_in_bounds(n)&&!b->board[n]&&!vs[n]) { vs[n]=1;q[t++]=n; }}}
    }
    int bk=0,wh=0;
    for (int i=0;i<GO_CELLS;i++) { if (ow[i]==1) bk++; if (ow[i]==-1) wh++; }
    return (float)(bk-wh);
}

/* Deterministic replay */
static unsigned go_rng_st;
static void go_rng_seed(unsigned s) { go_rng_st=s; }
static unsigned go_rng_next(void) { go_rng_st=go_rng_st*1103515245u+12345u; return go_rng_st; }

static float go_replay(const GoBoard *st, int mv, unsigned seed) {
    GoBoard b=*st; go_apply_move(&b,mv); int mr=-b.to_play;
    go_rng_seed(seed);
    while (!go_game_over(&b)) {
        int lg[GO_MAX_MOVES],nl=0;
        for (int m=0;m<GO_MAX_MOVES;m++) if (go_is_legal(&b,m)) lg[nl++]=m;
        if (!nl) break;
        go_apply_move(&b,lg[go_rng_next()%nl]);
    }
    float sc=go_territory_score(&b);
    return mr==1?sc:-sc;
}

/* Features */
#define GO_EXP_FEATURES 96
enum { GF_BIAS=0,GF_R0=1,GF_R1=2,GF_CI=3,GF_CC=4,GF_CI2=5,GF_DOM=6,
    GF_BK=10,GF_WH=11,GF_PR=12,GF_TP=13,GF_PS=14,
    GF_IP=20,GF_MX=21,GF_MY=22,GF_ON=23,GF_OP=24,GF_EM=25,
    GF_AT=27,GF_CP=28,GF_SV=29,GF_CC2=30,GF_LI=31 };

static const NNFeatureSchema GO_EXPERIMENT_FEATURES[] = {
    {"bias",NN_FEATURE_STATE_OBSERVATION,1},
    {"role_0",NN_FEATURE_STATE_OBSERVATION,1},
    {"role_1",NN_FEATURE_STATE_OBSERVATION,1},
    {"cand_idx",NN_FEATURE_STATE_OBSERVATION,1},
    {"cand_cnt",NN_FEATURE_STATE_OBSERVATION,1},
    {"cand_idx2",NN_FEATURE_STATE_OBSERVATION,1},
    {"domain",NN_FEATURE_STATE_OBSERVATION,1},
    {"_p7",NN_FEATURE_STATE_OBSERVATION,0},{"_p8",NN_FEATURE_STATE_OBSERVATION,0},{"_p9",NN_FEATURE_STATE_OBSERVATION,0},
    {"black_r",NN_FEATURE_STATE_OBSERVATION,1},
    {"white_r",NN_FEATURE_STATE_OBSERVATION,1},
    {"progress",NN_FEATURE_STATE_OBSERVATION,1},
    {"to_play",NN_FEATURE_STATE_OBSERVATION,1},
    {"passed",NN_FEATURE_STATE_OBSERVATION,1},
    {"_p15",NN_FEATURE_STATE_OBSERVATION,0},{"_p16",NN_FEATURE_STATE_OBSERVATION,0},
    {"_p17",NN_FEATURE_STATE_OBSERVATION,0},{"_p18",NN_FEATURE_STATE_OBSERVATION,0},
    {"_p19",NN_FEATURE_STATE_OBSERVATION,0},
    {"is_pass",NN_FEATURE_STATE_OBSERVATION,1},
    {"move_x",NN_FEATURE_STATE_OBSERVATION,1},
    {"move_y",NN_FEATURE_STATE_OBSERVATION,1},
    {"own_nb",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"opp_nb",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"empty_nb",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"_p26",NN_FEATURE_STATE_OBSERVATION,0},
    {"atari",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"capture",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"save",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"cap_cnt",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"lib_inv",NN_FEATURE_STRUCTURAL_PRIOR,1},
    {"_p32",NN_FEATURE_STATE_OBSERVATION,0},{"_p33",NN_FEATURE_STATE_OBSERVATION,0},
    {"_p34",NN_FEATURE_STATE_OBSERVATION,0},{"_p35",NN_FEATURE_STATE_OBSERVATION,0},
    {"_p36",NN_FEATURE_STATE_OBSERVATION,0},{"_p37",NN_FEATURE_STATE_OBSERVATION,0},
    {"_p38",NN_FEATURE_STATE_OBSERVATION,0},{"_p39",NN_FEATURE_STATE_OBSERVATION,0},
    {"gr_40_55",NN_FEATURE_STATE_OBSERVATION,0},
    {"rsv_56_63",NN_FEATURE_STATE_OBSERVATION,0},
    {"rsv_64_95",NN_FEATURE_STATE_OBSERVATION,0},
};

static int go_self_atari(const GoBoard *b, int idx) {
    GoBoard cp=*b; cp.board[idx]=cp.to_play; int opp=-cp.to_play;
    for (int d=0;d<4;d++) { int n=idx+GO_DIRS[d];
        if ((d==1&&go_on_left(idx))||(d==2&&go_on_right(idx))) continue;
        if (go_in_bounds(n)&&cp.board[n]==opp&&go_group_dead(&cp,n))
            go_remove_group(&cp,n); }
    return go_count_liberties(&cp,idx)==1;
}
static int go_move_caps(const GoBoard *b, int idx) {
    int opp=-b->to_play;
    for (int d=0;d<4;d++) { int n=idx+GO_DIRS[d];
        if ((d==1&&go_on_left(idx))||(d==2&&go_on_right(idx))) continue;
        if (go_in_bounds(n)&&b->board[n]==opp&&go_group_dead(b,n)) return 1; }
    return 0;
}
static int go_move_saves(const GoBoard *b, int idx) {
    for (int d=0;d<4;d++) { int n=idx+GO_DIRS[d];
        if ((d==1&&go_on_left(idx))||(d==2&&go_on_right(idx))) continue;
        if (go_in_bounds(n)&&b->board[n]==b->to_play&&go_count_liberties(b,n)==1)
            return 1; }
    return 0;
}

static void go_fill_features(const GoBoard *b, int mv, nn_real r[GO_EXP_FEATURES]) {
    memset(r,0,GO_EXP_FEATURES*sizeof(nn_real));
    r[GF_BIAS]=1;r[GF_R0]=1;r[GF_R1]=1;r[GF_DOM]=1;
    int lc=0,mr=0;
    for (int m=0;m<GO_MAX_MOVES;m++) if (go_is_legal(b,m)) lc++;
    for (int m=0;m<mv;m++) if (go_is_legal(b,m)) mr++;
    r[GF_CI]=lc<=1?0:(nn_real)mr/(nn_real)(lc-1);
    r[GF_CC]=(nn_real)lc/(nn_real)GO_MAX_MOVES;
    int bk=0,wh=0;
    for (int i=0;i<GO_CELLS;i++) {
        if (b->board[i]==1) bk++; else if (b->board[i]==-1) wh++; }
    r[GF_BK]=(nn_real)bk/GO_CELLS; r[GF_WH]=(nn_real)wh/GO_CELLS;
    r[GF_PR]=(nn_real)b->moves/GO_MAX_GAME_MOVES;
    r[GF_TP]=b->to_play==1?1:0; r[GF_PS]=b->passes>0?1:0;
    if (mv==GO_CELLS) { r[GF_IP]=1; }
    else {
        int x=mv%GO_BOARD_SIZE,y=mv/GO_BOARD_SIZE;
        r[GF_MX]=(nn_real)x/(GO_BOARD_SIZE-1); r[GF_MY]=(nn_real)y/(GO_BOARD_SIZE-1);
        int no=0,np=0,ne=0;
        for (int d=0;d<4;d++) { int n=mv+GO_DIRS[d];
            if ((d==1&&go_on_left(mv))||(d==2&&go_on_right(mv))) continue;
            if (!go_in_bounds(n)) continue;
            if (b->board[n]==b->to_play) no++;
            else if (b->board[n]==-b->to_play) np++; else ne++; }
        r[GF_ON]=(nn_real)no/4;r[GF_OP]=(nn_real)np/4;r[GF_EM]=(nn_real)ne/4;
        r[GF_AT]=go_self_atari(b,mv)?1:0;
        r[GF_CP]=go_move_caps(b,mv)?1:0;
        r[GF_SV]=go_move_saves(b,mv)?1:0;
        int caps=0;
        for (int d=0;d<4;d++) { int n=mv+GO_DIRS[d];
            if ((d==1&&go_on_left(mv))||(d==2&&go_on_right(mv))) continue;
            if (go_in_bounds(n)&&b->board[n]==-b->to_play&&go_group_dead(b,n)) caps++; }
        r[GF_CC2]=(nn_real)caps/4;
        GoBoard cp=*b; cp.board[mv]=cp.to_play;
        int libs=go_count_liberties(&cp,mv);
        r[GF_LI]=libs>0?1.0f/(nn_real)libs:0;
    }
}

/* Experiment spec */
static const NNExperimentSpec GO_POSITION_EXPERIMENT_SPEC = {
    .task_name="go_position",
    .role="9x9 Go action-ranking under deterministic replay grounding",
    .state_space="9x9 Go board with side-to-move, pass count, move count, ko",
    .state_distribution="random self-play positions from empty-board rollouts",
    .action_space="82 moves: 81 intersections + pass",
    .candidate_enumerator="enumerate legal moves under suicide and ko rules",
    .observable_features="stone counts, progress, side-to-move, pass, coords, neighbors, self-atari, capture, save, liberty",
    .leakage_policy=NN_LEAKAGE_PRIORS_ALLOWED,
    .neural_scorer="shared MLP, 96-wide board/action rows, softmax ranks survivors",
    .grounding_authority=NN_GROUNDING_DETERMINISTIC_REPLAY,
    .grounding_operator="deterministic random-rollout score replay with fixed LCG seed",
    .exact_verifier="terminal territory scoring after deterministic rollout",
    .verifier_feedback="target, correctness, score diff, candidate count, calibration",
    .candidate_grammar="82 moves with suicide, ko, repetition legality gates",
    .outcome_row_schema="crlp.trace.v1 with verifier score",
    .claim_scope="action ranking, certified pruning on generated 9x9 Go positions",
    .known_non_claims="no competitive play strength, optimal move, or engine comparison claims",
    .rule_grammar="legality, self-atari, capture, save, liberty-threshold predicates",
    .certificate_checker="rollout-based: rule is safe if rollout oracle never picks rejected",
    .gate_executor="accepted predicates remove moves before neural scoring",
    .proposer_inputs="rule kind, direction, neighbor, liberty, capture descriptors",
    .proposer_targets="train/val residual marginal utility under certified budget",
    .proposer_mode=NN_PROPOSER_SELECTION,
    .baseline_orderings="neural, random, grammar-order, structural, validation oracle",
    .generalization_axis="train on random positions, test held-out seed ranges",
    .trace_schema="crlp.trace.v1",
    .replay_protocol="stable action-id JSONL traces with parent checkpoint hashes",
    .rl_protocol="supervised scorer, strict verifier PG, KL/entropy-regularized verifier PG",
    .features=GO_EXPERIMENT_FEATURES,
    .feature_count=sizeof(GO_EXPERIMENT_FEATURES)/sizeof(GO_EXPERIMENT_FEATURES[0]),
};

/* Dataset adapter */
int gen_go_adapter(const NNDatasetAdapterConfig *cfg, NNDataset *ds) {
    nn_dataset_init(ds,GO_EXP_FEATURES);
    size_t mg=cfg->max_groups==0?128:cfg->max_groups;
    size_t tc=(mg*3)/5, vc=mg/5;
    unsigned bs=cfg->seed==0?20260601u:cfg->seed;
    for (size_t gi=0;gi<mg;gi++) {
        GoBoard b; go_init_board(&b);
        unsigned ps=bs+(unsigned)(gi*137u); srand(ps);
        int np=(rand()%18)+3;
        for (int p=0;p<np&&!go_game_over(&b);p++) {
            int lg[GO_MAX_MOVES],nl=0;
            for (int m=0;m<GO_MAX_MOVES;m++) if (go_is_legal(&b,m)) lg[nl++]=m;
            if (!nl) break; go_apply_move(&b,lg[rand()%nl]);
        }
        if (go_game_over(&b)) continue;
        int nl=0; for (int m=0;m<GO_MAX_MOVES;m++) if (go_is_legal(&b,m)) nl++;
        if (!nl) continue;
        float best=-9999,sc[GO_MAX_MOVES];
        for (int m=0;m<GO_MAX_MOVES;m++) {
            if (go_is_legal(&b,m)) {
                sc[m]=go_replay(&b,m,ps+(unsigned)m);
                if (sc[m]>best) best=sc[m];
            } else sc[m]=-9999;
        }
        NNDatasetSplit sp=(gi<tc)?NN_DATASET_SPLIT_TRAIN:
                          (gi<tc+vc)?NN_DATASET_SPLIT_VALID:NN_DATASET_SPLIT_TEST;
        size_t begin=ds->row_count;
        /* Two-pass insertion: targets first so the candidate cap never
         * excludes them, then non-targets up to the cap. */
        int included[GO_MAX_MOVES]; memset(included,0,sizeof(included));
        for (int pass=0;pass<2;pass++) {
            for (int m=0;m<GO_MAX_MOVES;m++) {
                if (!go_is_legal(&b,m) || included[m]) continue;
                int tgt=(sc[m]>=best-0.001f)?1:0;
                if ((pass==0 && !tgt) || (pass==1 && tgt)) continue;
                if (cfg->max_candidates_per_group>0 &&
                    ds->row_count-begin>=cfg->max_candidates_per_group) break;
                nn_real feats[GO_EXP_FEATURES]; go_fill_features(&b,m,feats);
                if (!nn_dataset_append_candidate(ds,sp,(uint64_t)gi,
                                                 (uint64_t)ps,(uint32_t)m,tgt,feats))
                    return 0;
                included[m]=1;
            }}
        if (ds->row_count>begin &&
            !nn_dataset_finalize_group(ds,begin,ds->row_count))
            return 0;
    }
    return 1;
}

#ifndef GO_EXPERIMENT_LIBRARY_ONLY
int main(int argc, char **argv) {
    if (argc==2&&!strcmp(argv[1],"--schema")) {
        nn_experiment_write_schema_json(stdout,&GO_POSITION_EXPERIMENT_SPEC);
        return 0;
    }
    NNDatasetExperimentConfig dc; memset(&dc,0,sizeof(dc));
    dc.feature_count=GO_EXP_FEATURES; dc.max_groups=128;
    dc.max_candidates_per_group=32; dc.seed=20260601u;
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--groups")&&i+1<argc) dc.max_groups=(size_t)atoi(argv[++i]);
        if (!strcmp(argv[i],"--seed")&&i+1<argc) dc.seed=(unsigned)atoi(argv[++i]);
    }
    NNDataset ds;
    if (!gen_go_adapter(&dc,&ds)) {
        fprintf(stderr,"[go_exp] dataset gen failed\n"); return 1; }
    fprintf(stderr,"[go_exp] %zu groups, %zu rows\n",ds.group_count,ds.row_count);
    const size_t ly[]={GO_EXP_FEATURES,64,32,1};
    NNConfig nc; memset(&nc,0,sizeof(nc));
    nc.layer_sizes=ly; nc.layer_count=4; nc.seed=dc.seed;
    nc.hidden_activation=NN_ACTIVATION_RELU;
    nc.output_activation=NN_ACTIVATION_SIGMOID; nc.loss=NN_LOSS_MSE;
    NeuralNet *net=nn_create_config(&nc);
    if (!net) { nn_dataset_free(&ds); return 1; }
    float lr=0.004f;
    for (int e=0;e<4;e++) {
        float loss=0; size_t n=0;
        for (size_t ri=0;ri<ds.row_count;ri++) {
            if (ds.rows[ri].split!=NN_DATASET_SPLIT_TRAIN) continue;
            nn_real tgt=(nn_real)ds.rows[ri].target;
            loss+=(float)nn_train_sample(net,ds.rows[ri].features,&tgt,lr); n++; }
        if (n) fprintf(stderr,"[go_exp] epoch %d loss=%.4f n=%zu\n",e+1,loss/(float)n,n);
    }
    size_t corr=0,tot=0;
    for (size_t gi=0;gi<ds.group_count;gi++) {
        NNDatasetGroup *g=&ds.groups[gi];
        if (!g->count) continue;
        if (ds.rows[g->begin].split!=NN_DATASET_SPLIT_TEST) continue;
        float best=-1e9f; size_t bri=g->begin;
        for (size_t ri=g->begin;ri<g->begin+g->count;ri++) {
            nn_real sc; nn_forward(net,ds.rows[ri].features,&sc);
            if ((float)sc>best) { best=(float)sc; bri=ri; } }
        if (ds.rows[bri].target) corr++; tot++;
    }
    if (tot) fprintf(stderr,"[go_exp] test top-1: %zu/%zu (%.1f%%)\n",
                      corr,tot,100.0*(double)corr/(double)tot);
    nn_free(net); nn_dataset_free(&ds);
    return 0;
}
#endif
