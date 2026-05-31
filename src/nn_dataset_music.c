#include <string.h>

typedef struct { int pitch; int duration; } NNDatasetMusicNote;

#define NN_DATASET_MUSIC_CORPUS_NOTE_COUNT 36
#include "nn_dataset_music_corpus.inc"

static const NNDatasetMusicNote MUSIC_CORPUS_NOTES[NN_DATASET_MUSIC_CORPUS_NOTE_COUNT] = {
    { 0,1 },{ 0,2 },{ 0,4 },{ 2,1 },{ 2,2 },{ 3,1 },{ 4,1 },{ 4,2 },{ 4,4 },
    { 5,1 },{ 5,2 },{ 5,4 },{ 7,1 },{ 7,2 },{ 7,4 },{ 8,1 },{ 9,1 },{ 9,2 },
    { 9,4 },{10,1 },{11,1 },{11,2 },{11,4 },{12,1 },{12,2 },{12,4 },{14,1 },
    {14,2 },{16,1 },{16,2 },{17,1 },{19,1 },{19,2 },{21,1 },{21,2 },{24,1 },
};

static int music_phrase_for_line(int line_index) { return line_index < 128 ? 0 : 1; }

static int music_transition_exists(int beat, int prev, int next) {
    int li; for(li=0;li<200;++li){if(beat<1)continue;
    if(MUSIC_CORPUS_LINES[li][beat-1]==prev&&MUSIC_CORPUS_LINES[li][beat]==next)return 1;}return 0;
}

static void encode_music_features(nn_real f[NN_DATASET_MUSIC_FEATURE_COUNT],
    int phrase,int beat,int prev,int cand,int *prefix,int plen,int adj,int viable_unused) {
    memset(f,0,NN_DATASET_MUSIC_FEATURE_COUNT*sizeof(f[0]));
    const NNDatasetMusicNote *c=&MUSIC_CORPUS_NOTES[cand];
    int sr=(prev>=0&&prev==cand),ou=0,ib=0,i;
    {int ct=0;for(i=0;i<plen;++i)if(prefix[i]==cand)++ct;ou=ct>=2;}
    if(prev>=0){int j=c->pitch-MUSIC_CORPUS_NOTES[prev].pitch;if(j<0)j=-j;ib=j>=7;}
    f[0]=(nn_real)1.0; f[1]=(nn_real)((double)phrase); f[2]=(nn_real)((double)beat/7.0);
    f[3]=(nn_real)((double)c->pitch/24.0); f[4]=(nn_real)((double)c->duration/4.0);
    f[5]=adj?(nn_real)1:(nn_real)0; f[6]=(nn_real)0;
    f[7]=prev<0?(nn_real)1:(nn_real)0; f[8]=beat>=6?(nn_real)1:(nn_real)0;
    f[9]=(nn_real)((double)plen/7.0); f[10]=sr?(nn_real)1:(nn_real)0;
    f[11]=ou?(nn_real)1:(nn_real)0; f[12]=(phrase==1&&beat<2)?(nn_real)1:(nn_real)0;
    f[13]=ib?(nn_real)1:(nn_real)0;
    f[14]=(nn_real)((double)(beat>0&&prev>=0?c->pitch-MUSIC_CORPUS_NOTES[prev].pitch+12:0)/24.0);
    f[15]=(nn_real)((double)cand/35.0); f[16]=beat<4?(nn_real)1:(nn_real)0;
    f[17]=beat>=4?(nn_real)1:(nn_real)0;
}

static uint64_t music_state_hash(int line,int beat,int prev) {
    uint64_t h=1469598103934665603ull; h^=(uint64_t)line+0x9e3779b97f4a7c15ull;
    h*=1099511628211ull; h^=(uint64_t)beat; h*=1099511628211ull;
    h^=(uint64_t)(uint32_t)(prev+1); return h;
}

static int generate_music_sequence_adapter(const NNDatasetAdapterConfig *cfg, NNDataset *ds) {
    int nc=NN_DATASET_MUSIC_CORPUS_NOTE_COUNT,lc=NN_DATASET_MUSIC_CORPUS_LINE_COUNT;
    size_t mg=cfg->max_groups==0u?SIZE_MAX:cfg->max_groups;
    size_t mc=cfg->max_candidates_per_group==0u?(size_t)nc:cfg->max_candidates_per_group;
    nn_dataset_init(ds,NN_DATASET_MUSIC_FEATURE_COUNT);
    uint64_t gid=0; int line,beat,cand;
    for(line=0;line<lc&&ds->group_count<mg;++line){
        int ph=music_phrase_for_line(line),pv=-1,px[NN_DATASET_MUSIC_MAX_BEAT],pl=0;
        for(beat=0;beat<NN_DATASET_MUSIC_MAX_BEAT&&ds->group_count<mg;++beat){
            size_t beg=ds->row_count;
            for(cand=0;(size_t)cand<mc&&cand<nc;++cand){
                int adj=music_transition_exists(beat,pv,cand);
                int tgt=(cand==MUSIC_CORPUS_LINES[line][beat]);
                if(pv<0)tgt=(cand==MUSIC_CORPUS_LINES[line][0]);
                nn_real f[NN_DATASET_MUSIC_FEATURE_COUNT];
                encode_music_features(f,ph,beat,pv,cand,px,pl,adj,0);
                if(!nn_dataset_append_candidate(ds,MUSIC_LINE_SPLITS[line],gid,
                    music_state_hash(line,beat,pv),(uint32_t)cand,tgt,f))return 0;
            }
            if(!nn_dataset_finalize_group(ds,beg,ds->row_count))return 0;
            px[pl++]=MUSIC_CORPUS_LINES[line][beat]; pv=MUSIC_CORPUS_LINES[line][beat]; ++gid;
        }
    }
    return ds->group_count>0u;
}
