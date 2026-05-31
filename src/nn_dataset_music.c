/* Automatically included into nn_dataset.c — toy music sequence adapter for sensor-transfer experiments */

#include <string.h>

typedef struct {
    int pitch;
    int duration;
} NNDatasetMusicNote;

#define NN_DATASET_MUSIC_CORPUS_NOTE_COUNT 36
#define NN_DATASET_MUSIC_CORPUS_LINE_COUNT 24
#define NN_DATASET_MUSIC_MAX_BEAT 8

static const NNDatasetMusicNote MUSIC_CORPUS_NOTES[NN_DATASET_MUSIC_CORPUS_NOTE_COUNT] = {
    { 0, 1 }, { 0, 2 }, { 0, 4 }, { 2, 1 }, { 2, 2 },
    { 3, 1 }, { 4, 1 }, { 4, 2 }, { 4, 4 }, { 5, 1 },
    { 5, 2 }, { 5, 4 }, { 7, 1 }, { 7, 2 }, { 7, 4 },
    { 8, 1 }, { 9, 1 }, { 9, 2 }, { 9, 4 }, {10, 1 },
    {11, 1 }, {11, 2 }, {11, 4 }, {12, 1 }, {12, 2 },
    {12, 4 }, {14, 1 }, {14, 2 }, {16, 1 }, {16, 2 },
    {17, 1 }, {19, 1 }, {19, 2 }, {21, 1 }, {21, 2 },
    {24, 1 },
};

static const int MUSIC_CORPUS_LINES[NN_DATASET_MUSIC_CORPUS_LINE_COUNT][NN_DATASET_MUSIC_MAX_BEAT] = {
    { 0, 3, 7, 9,13, 9, 7, 3 }, { 0, 7,12,16,19,16,12, 7 },
    { 3, 9,13,17,21,17,13, 9 }, { 5, 9,13,17,13, 9, 5, 3 },
    { 1, 5, 9,13,17,13, 9, 5 }, { 2, 6,10,14,18,14,10, 6 },
    { 4, 8,12,16,20,16,12, 8 }, { 0, 4, 8,12,16,12, 8, 4 },
    { 7,11,15,19,23,19,15,11 }, { 9,13,17,21,25,21,17,13 },
    {10,14,18,22,26,22,18,14 }, {12,16,20,24,28,24,20,16 },
    {15,19,23,27,31,27,23,19 }, {17,21,25,29,33,29,25,21 },
    {20,24,28,32,35,32,28,24 }, {22,26,30,34,35,34,30,26 },
    { 1, 4, 8,12,16,12, 8, 4 }, { 6,10,14,18,22,18,14,10 },
    {11,15,19,23,27,23,19,15 }, {18,22,26,30,34,30,26,22 },
    { 0, 5,10,15,20,15,10, 5 }, { 8,13,18,23,28,23,18,13 },
    { 3, 8,13,18,23,18,13, 8 }, {14,19,24,29,34,29,24,19 },
};

static const NNDatasetSplit MUSIC_LINE_SPLITS[NN_DATASET_MUSIC_CORPUS_LINE_COUNT] = {
    NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN, NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_VALID,  NN_DATASET_SPLIT_VALID,  NN_DATASET_SPLIT_VALID,  NN_DATASET_SPLIT_VALID,
    NN_DATASET_SPLIT_TEST,   NN_DATASET_SPLIT_TEST,   NN_DATASET_SPLIT_TEST,   NN_DATASET_SPLIT_TEST,
};

static int music_phrase_for_line(int line_index) {
    return line_index < 12 ? 0 : 1;
}

/* Adjacency: check training split only. Test melodies use held-out patterns. */
static int music_transition_exists(int beat, int prev_note, int next_note) {
    int li;
    for (li = 0; li < 16; ++li) { /* train lines only */
        if (beat < 1) continue;
        if (MUSIC_CORPUS_LINES[li][beat - 1] == prev_note &&
            MUSIC_CORPUS_LINES[li][beat]     == next_note) return 1;
    }
    return 0;
}

/* Viable: note appears at this beat position in ANY corpus line */
static int music_note_in_beat(int beat, int note) {
    int li;
    for (li = 0; li < NN_DATASET_MUSIC_CORPUS_LINE_COUNT; ++li) {
        if (MUSIC_CORPUS_LINES[li][beat] == note) return 1;
    }
    return 0;
}

static void encode_music_features(nn_real features[NN_DATASET_MUSIC_FEATURE_COUNT],
                                   int phrase, int beat, int prev_note,
                                   int candidate_note, const int *prefix,
                                   int prefix_len, int adjacent, int viable)
{
    memset(features, 0, NN_DATASET_MUSIC_FEATURE_COUNT * sizeof(features[0]));
    const NNDatasetMusicNote *cand = &MUSIC_CORPUS_NOTES[candidate_note];
    int same_repeat = (prev_note >= 0 && prev_note == candidate_note);
    int overuse = 0, interval_big = 0, i;
    { int c=0; for(i=0;i<prefix_len;++i) if(prefix[i]==candidate_note)++c; overuse=c>=2; }
    if(prev_note>=0){int j=cand->pitch-MUSIC_CORPUS_NOTES[prev_note].pitch;if(j<0)j=-j;interval_big=j>=7;}
    int phrase_b_dist = (phrase == 1 && beat < 2) ? 1 : 0;

    features[0]  = (nn_real)1.0;
    features[1]  = (nn_real)((double)phrase / 1.0);
    features[2]  = (nn_real)((double)beat / (NN_DATASET_MUSIC_MAX_BEAT - 1.0));
    features[3]  = (nn_real)((double)cand->pitch / 24.0);
    features[4]  = (nn_real)((double)cand->duration / 4.0);
    features[5]  = adjacent  ? (nn_real)1.0 : (nn_real)0.0;
    features[6]  = viable    ? (nn_real)1.0 : (nn_real)0.0;
    features[7]  = prev_note < 0 ? (nn_real)1.0 : (nn_real)0.0;
    features[8]  = beat >= NN_DATASET_MUSIC_MAX_BEAT - 2 ? (nn_real)1.0 : (nn_real)0.0;
    features[9]  = (nn_real)((double)prefix_len / (NN_DATASET_MUSIC_MAX_BEAT - 1.0));
    features[10] = same_repeat ? (nn_real)1.0 : (nn_real)0.0;
    features[11] = overuse ? (nn_real)1.0 : (nn_real)0.0;
    features[12] = phrase_b_dist ? (nn_real)1.0 : (nn_real)0.0;
    features[13] = interval_big ? (nn_real)1.0 : (nn_real)0.0;
    features[14] = (nn_real)((double)(beat>0&&prev_note>=0?cand->pitch-MUSIC_CORPUS_NOTES[prev_note].pitch+12:0)/24.0);
    features[15] = (nn_real)((double)candidate_note / (NN_DATASET_MUSIC_CORPUS_NOTE_COUNT - 1.0));
    features[16] = (nn_real)(beat < NN_DATASET_MUSIC_MAX_BEAT / 2 ? 1.0 : 0.0);
    features[17] = (nn_real)(beat >= NN_DATASET_MUSIC_MAX_BEAT / 2 ? 1.0 : 0.0);
}

static uint64_t music_state_hash(int line, int beat, int prev_note) {
    uint64_t hash = 1469598103934665603ull;
    hash ^= (uint64_t)line + 0x9e3779b97f4a7c15ull;
    hash *= 1099511628211ull;
    hash ^= (uint64_t)beat;
    hash *= 1099511628211ull;
    hash ^= (uint64_t)(uint32_t)(prev_note + 1);
    return hash;
}

static int generate_music_sequence_adapter(const NNDatasetAdapterConfig *config,
                                            NNDataset *dataset)
{
    const int note_count = NN_DATASET_MUSIC_CORPUS_NOTE_COUNT;
    const int line_count = NN_DATASET_MUSIC_CORPUS_LINE_COUNT;
    size_t max_groups = config->max_groups == 0u ? SIZE_MAX : config->max_groups;
    size_t max_candidates = config->max_candidates_per_group == 0u
                                ? (size_t)note_count : config->max_candidates_per_group;
    nn_dataset_init(dataset, NN_DATASET_MUSIC_FEATURE_COUNT);
    uint64_t group_id = 0;
    int line;
    for (line = 0; line < line_count && dataset->group_count < max_groups; ++line) {
        int phrase = music_phrase_for_line(line);
        int prev_note = -1;
        int prefix[NN_DATASET_MUSIC_MAX_BEAT];
        int prefix_len = 0;
        int beat;
        for (beat = 0; beat < NN_DATASET_MUSIC_MAX_BEAT && dataset->group_count < max_groups; ++beat) {
            size_t begin = dataset->row_count;
            int cand;
            for (cand = 0; (size_t)cand < max_candidates && cand < note_count; ++cand) {
                int adjacent = music_transition_exists(beat, prev_note, cand);
                /* viable = note exists at this beat position in some corpus line */
                int viable = (cand == MUSIC_CORPUS_LINES[line][beat]);
                if (prev_note < 0) viable = (cand == MUSIC_CORPUS_LINES[line][0]);
                nn_real features[NN_DATASET_MUSIC_FEATURE_COUNT];
                encode_music_features(features, phrase, beat, prev_note, cand,
                                      prefix, prefix_len, adjacent, viable);
                if (!nn_dataset_append_candidate(dataset, MUSIC_LINE_SPLITS[line],
                                                 group_id, music_state_hash(line, beat, prev_note),
                                                 (uint32_t)cand, viable, features)) return 0;
            }
            if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) return 0;
            prefix[prefix_len++] = MUSIC_CORPUS_LINES[line][beat];
            prev_note = MUSIC_CORPUS_LINES[line][beat];
            ++group_id;
        }
    }
    return dataset->group_count > 0u;
}
