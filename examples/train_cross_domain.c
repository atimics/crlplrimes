/**
 * train_cross_domain.c — Unified cross-domain training pipeline.
 *
 * Loads tournament game samples and trains a shared scorer across
 * all domains plus tournament replay data.  Saves checkpoints for
 * warm-starting the tournament.
 *
 * Build:  make train-cross-domain
 */
#include "nn.h"
#include "nn_dataset.h"
#include "go_rules.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>

#define CD_FEATURES 96
#define CD_MAX_CANDIDATES 82

typedef struct {
    nn_real features[CD_FEATURES];
    int move, to_play;
    float reward;
} TourSample;

static size_t load_tournament_samples(const char *dir,
                                       TourSample *samples, size_t max_samples) {
    if (!dir) return 0;
    DIR *d = opendir(dir);
    if (!d) return 0;
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL && count < max_samples) {
        if (!strstr(entry->d_name, ".jsonl")) continue;
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char line[16384];
        while (fgets(line, sizeof(line), f) && count < max_samples) {
            TourSample *s = &samples[count];
            memset(s, 0, sizeof(*s));
            char *mp = strstr(line, "\"move\":");
            char *rp = strstr(line, "\"reward\":");
            char *tp = strstr(line, "\"to_play\":");
            char *fp = strstr(line, "\"features\":[");
            if (mp) s->move = atoi(mp + 7);
            if (rp) s->reward = (float)atof(rp + 9);
            if (tp) s->to_play = atoi(tp + 10);
            if (fp) {
                fp += 12;
                for (int i = 0; i < CD_FEATURES && fp; i++) {
                    s->features[i] = (nn_real)strtod(fp, &fp);
                    if (fp && *fp == ',') fp++;
                }
            }
            count++;
        }
        fclose(f);
    }
    closedir(d);
    return count;
}

int main(int argc, char **argv) {
    const char *samples_dir = NULL;
    const char *save_path = "build/float/unified.nnckpt";
    const char *load_path = NULL;
    int epochs = 4, steps_per_epoch = 3000, hidden0 = 64, hidden1 = 32;
    unsigned seed = 42;
    float lr = 0.004f;
    int tournament_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--samples-dir") && i+1 < argc) samples_dir = argv[++i];
        else if (!strcmp(argv[i], "--save") && i+1 < argc) save_path = argv[++i];
        else if (!strcmp(argv[i], "--load") && i+1 < argc) load_path = argv[++i];
        else if (!strcmp(argv[i], "--epochs") && i+1 < argc) epochs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) steps_per_epoch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc) {
            sscanf(argv[++i], "%dx%d", &hidden0, &hidden1);
        }
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tournament-only")) tournament_only = 1;
    }

    TourSample *tour_samples = NULL;
    size_t tour_count = 0;
    if (samples_dir) {
        tour_samples = calloc(200000, sizeof(TourSample));
        if (tour_samples) {
            tour_count = load_tournament_samples(samples_dir, tour_samples, 200000);
            fprintf(stderr, "[cd] loaded %zu tournament samples from %s\n",
                    tour_count, samples_dir);
        }
    }

    const size_t layers[] = {CD_FEATURES, (size_t)hidden0, (size_t)hidden1, 1};
    NNConfig nc; memset(&nc, 0, sizeof(nc));
    nc.layer_sizes = layers; nc.layer_count = 4; nc.seed = seed;
    nc.hidden_activation = NN_ACTIVATION_RELU;
    nc.output_activation = NN_ACTIVATION_SIGMOID;
    nc.loss = NN_LOSS_MSE;

    NeuralNet *net;
    if (load_path) {
        NNCheckpointMeta meta;
        net = nn_load_checkpoint(load_path, &meta);
        if (!net || nn_input_size(net) != CD_FEATURES || nn_output_size(net) != 1) {
            if (net) nn_free(net);
            net = nn_create_config(&nc);
            fprintf(stderr, "[cd] load failed, fresh network\n");
        } else {
            fprintf(stderr, "[cd] loaded %s\n", load_path);
        }
    } else {
        net = nn_create_config(&nc);
        fprintf(stderr, "[cd] fresh network %dx%d\n", hidden0, hidden1);
    }
    if (!net) { free(tour_samples); return 1; }

    const char *domains[] = {"prime_gap_candidate", "grid_path", "delayed_claim", "go_position", "signal_wasd", "signal_mining"};
    int n_domains = tournament_only ? 0 : 6;

    for (int e = 0; e < epochs; e++) {
        float total_loss = 0;
        size_t total_steps = 0;
        srand(seed + (unsigned)e * 1000);

        for (int step = 0; step < steps_per_epoch; step++) {
            int use_tournament = tour_count > 0 && (rand() % 100 < 60);
            nn_real rows[CD_MAX_CANDIDATES * CD_FEATURES];
            size_t cand_count = 0, target = 0;

            if (use_tournament) {
                size_t si = (size_t)rand() % tour_count;
                TourSample *ts = &tour_samples[si];
                cand_count = 0;
                for (size_t j = 0; j < CD_MAX_CANDIDATES && j < tour_count; j++) {
                    size_t sj = (si + j) % tour_count;
                    if (cand_count >= CD_MAX_CANDIDATES) break;
                    memcpy(&rows[cand_count * CD_FEATURES],
                           tour_samples[sj].features,
                           CD_FEATURES * sizeof(nn_real));
                    if (sj == si) target = cand_count;
                    cand_count++;
                }
                if (ts->reward <= 0.0f && target < cand_count) {
                    for (size_t j = 0; j < cand_count; j++) {
                        size_t sj = (si + j) % tour_count;
                        if (tour_samples[sj].reward > 0.5f) { target = j; break; }
                    }
                }
            } else if (n_domains > 0) {
                int di = rand() % n_domains;
                const NNDatasetExperiment *exp = nn_dataset_find_experiment(domains[di]);
                if (exp) {
                    NNDatasetExperimentConfig dcfg;
                    memset(&dcfg, 0, sizeof(dcfg));
                    dcfg.feature_count = exp->feature_count;  /* use adapter's real width */
                    dcfg.max_groups = 1;
                    dcfg.max_candidates_per_group = CD_MAX_CANDIDATES;
                    dcfg.seed = seed + (unsigned)step * 137;

                    NNDataset ds;
                    if (nn_dataset_generate_experiment(exp, &dcfg, &ds) && ds.group_count > 0) {
                        NNDatasetGroup *g = &ds.groups[0];
                        cand_count = g->count;
                        if (cand_count > CD_MAX_CANDIDATES) cand_count = CD_MAX_CANDIDATES;
                        size_t src_fc = exp->feature_count;
                        for (size_t j = 0; j < cand_count; j++) {
                            /* Zero-pad to 96 features for the shared scorer */
                            memset(&rows[j * CD_FEATURES], 0,
                                   CD_FEATURES * sizeof(nn_real));
                            memcpy(&rows[j * CD_FEATURES],
                                   ds.rows[g->begin + j].features,
                                   src_fc * sizeof(nn_real));
                            if (ds.rows[g->begin + j].target) target = j;
                        }
                        nn_dataset_free(&ds);
                    }
                }
            }


            if (cand_count > 1 && target < cand_count) {
                nn_real loss = nn_train_action_scorer(net, rows, cand_count, target, lr);
                if (!isnan((double)loss)) { total_loss += (double)loss; total_steps++; }
            }
        }
        if (total_steps > 0)
            fprintf(stderr, "[cd] epoch %d loss=%.4f steps=%zu\n",
                    e+1, total_loss / (double)total_steps, total_steps);
    }

    NNCheckpointMeta meta; nn_checkpoint_meta_init(&meta, 1);
    nn_save_checkpoint(net, save_path, &meta);
    fprintf(stderr, "[cd] saved %s\n", save_path);

    if (!tournament_only) {
        size_t correct = 0, total = 0;
        NNDatasetExperimentConfig dcfg;
        memset(&dcfg, 0, sizeof(dcfg));
        dcfg.feature_count = CD_FEATURES;
        dcfg.max_groups = 32;
        dcfg.max_candidates_per_group = CD_MAX_CANDIDATES;
        dcfg.seed = 999;
        const NNDatasetExperiment *exp = nn_dataset_find_experiment("go_position");
        if (exp) {
            NNDataset ds;
            if (nn_dataset_generate_experiment(exp, &dcfg, &ds)) {
                for (size_t gi = 0; gi < ds.group_count; gi++) {
                    NNDatasetGroup *g = &ds.groups[gi];
                    float best_sc = -1e9f; size_t best_j = 0;
                    for (size_t j = 0; j < g->count; j++) {
                        nn_real sc;
                        nn_forward(net, ds.rows[g->begin + j].features, &sc);
                        if ((float)sc > best_sc) { best_sc = (float)sc; best_j = j; }
                    }
                    if (ds.rows[g->begin + best_j].target) correct++;
                    total++;
                }
                nn_dataset_free(&ds);
            }
        }
        if (total > 0)
            fprintf(stderr, "[cd] go eval top-1: %zu/%zu (%.1f%%)\n",
                    correct, total, 100.0 * (double)correct / (double)total);
    }

    nn_free(net);
    free(tour_samples);
    return 0;
}
