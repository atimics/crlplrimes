#include "nn.h"

#include "commodity.h"
#include "game_sim.h"
#include "manifest.h"
#include "sim_ai.h"
#include "station_util.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SNW_TRACE_SCHEMA "signal.npc_worker_counterfactual.v1"
#define SNW_SUMMARY_SCHEMA "crlp.signal_npc_worker.summary.v1"
#define SNW_TASK_NAME "signal_npc_worker"
#define SNW_OPTION_COUNT 8u
#define SNW_FEATURE_COUNT 56u

typedef enum {
    SNW_OPT_WAIT = 0,
    SNW_OPT_MINE_HOME = 1,
    SNW_OPT_HAUL_CONTRACT = 2,
    SNW_OPT_GOSSIP_COURIER = 3,
    SNW_OPT_SELF_REFIT_HOME = 4,
    SNW_OPT_IMPORT_FRAME = 5,
    SNW_OPT_IMPORT_LASER = 6,
    SNW_OPT_IMPORT_TRACTOR = 7,
} snw_option_t;

typedef struct {
    snw_option_t option;
    const char *name;
} snw_hypothesis_t;

typedef struct {
    double utility;
    double credit_delta;
    double refit_progress;
    double cargo_moved;
    double contract_value;
    int legal;
    int travel;
    int self_upgrade;
    int import_module;
} snw_outcome_t;

typedef struct {
    uint64_t group_id;
    uint32_t seed;
    int scenario_kind;
    int home_station;
    int role;
    ship_upgrade_t desired_upgrade;
    commodity_t desired_commodity;
    int desired_units;
    double home_balance;
    double persona_risk;
    double persona_growth;
    double persona_patience;
    nn_real features[SNW_OPTION_COUNT][SNW_FEATURE_COUNT];
    snw_outcome_t outcomes[SNW_OPTION_COUNT];
    size_t target;
    size_t baseline;
    double second_best_utility;
} snw_group_t;

typedef struct {
    snw_group_t *groups;
    size_t count;
} snw_dataset_t;

typedef struct {
    const char *summary_path;
    const char *trace_path;
    const char *save_path;
    uint32_t seed;
    size_t train_groups;
    size_t test_groups;
    size_t epochs;
    nn_real learning_rate;
} snw_config_t;

typedef struct {
    size_t groups;
    size_t top1;
    size_t top3;
    double selected_utility;
    double oracle_utility;
    double regret;
    double baseline_utility;
    double baseline_regret;
    size_t selected_legal;
    size_t selected_self_upgrade;
    size_t selected_import;
    size_t selected_travel;
    size_t option_selected[SNW_OPTION_COUNT];
    size_t option_target[SNW_OPTION_COUNT];
} snw_metrics_t;

static const snw_hypothesis_t SNW_HYPOTHESES[SNW_OPTION_COUNT] = {
    {SNW_OPT_WAIT, "wait"},
    {SNW_OPT_MINE_HOME, "mine_home_ore"},
    {SNW_OPT_HAUL_CONTRACT, "haul_best_contract"},
    {SNW_OPT_GOSSIP_COURIER, "gossip_courier"},
    {SNW_OPT_SELF_REFIT_HOME, "self_refit_home"},
    {SNW_OPT_IMPORT_FRAME, "import_frame_modules"},
    {SNW_OPT_IMPORT_LASER, "import_laser_modules"},
    {SNW_OPT_IMPORT_TRACTOR, "import_tractor_modules"},
};

static uint32_t snw_rng_next(uint32_t *state)
{
    *state = (*state * UINT32_C(1664525)) + UINT32_C(1013904223);
    return *state;
}

static double snw_rand01(uint32_t *state)
{
    return (double)(snw_rng_next(state) >> 8) / 16777215.0;
}

static double snw_clip(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static const snw_hypothesis_t *snw_hypothesis(size_t idx)
{
    return idx < SNW_OPTION_COUNT ? &SNW_HYPOTHESES[idx] : &SNW_HYPOTHESES[0];
}

static commodity_t snw_upgrade_commodity(ship_upgrade_t upgrade)
{
    return (commodity_t)(COMMODITY_FRAME + upgrade_required_product(upgrade));
}

static ship_upgrade_t snw_preferred_upgrade(const npc_ship_t *npc, const ship_t *ship)
{
    const ship_t *view = ship != NULL ? ship : (npc != NULL ? &npc->ship : NULL);
    if (npc == NULL || view == NULL) return SHIP_UPGRADE_COUNT;
    if (npc->role == NPC_ROLE_HAULER) {
        if (!ship_upgrade_maxed(view, SHIP_UPGRADE_HOLD)) return SHIP_UPGRADE_HOLD;
        if (!ship_upgrade_maxed(view, SHIP_UPGRADE_TRACTOR)) return SHIP_UPGRADE_TRACTOR;
        if (!ship_upgrade_maxed(view, SHIP_UPGRADE_MINING)) return SHIP_UPGRADE_MINING;
    } else {
        if (!ship_upgrade_maxed(view, SHIP_UPGRADE_MINING)) return SHIP_UPGRADE_MINING;
        if (!ship_upgrade_maxed(view, SHIP_UPGRADE_TRACTOR)) return SHIP_UPGRADE_TRACTOR;
        if (!ship_upgrade_maxed(view, SHIP_UPGRADE_HOLD)) return SHIP_UPGRADE_HOLD;
    }
    return SHIP_UPGRADE_COUNT;
}

static int snw_best_remote_source(const world_t *w,
                                  int home_station,
                                  commodity_t commodity,
                                  int *out_stock)
{
    int best = -1;
    int best_stock = 0;
    if (out_stock) *out_stock = 0;
    if (w == NULL || commodity >= COMMODITY_COUNT) return -1;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        if (s == home_station) continue;
        if (!station_is_active(&w->stations[s])) continue;
        int stock = station_finished_count(&w->stations[s], commodity);
        if (stock > best_stock) {
            best_stock = stock;
            best = s;
        }
    }
    if (out_stock) *out_stock = best_stock;
    return best;
}

static double snw_best_contract_value(const world_t *w,
                                      int home_station,
                                      commodity_t *out_commodity,
                                      int *out_dest,
                                      int *out_stock)
{
    double best = 0.0;
    commodity_t best_c = COMMODITY_COUNT;
    int best_dest = -1;
    int best_stock = 0;
    if (w == NULL || home_station < 0 || home_station >= MAX_STATIONS) return 0.0;
    for (int i = 0; i < MAX_CONTRACTS; ++i) {
        const contract_t *ct = &w->contracts[i];
        if (!ct->active || ct->commodity >= COMMODITY_COUNT) continue;
        if (ct->action != CONTRACT_TRACTOR && ct->action != CONTRACT_DELIVERY) continue;
        int stock = station_finished_count(&w->stations[home_station], ct->commodity);
        if (stock <= 0) continue;
        double units = snw_clip(ceil(ct->quantity_needed), 1.0, (double)stock);
        double value = contract_price(ct) * units;
        if (ct->station_index == home_station) value *= 0.15;
        if (value > best) {
            best = value;
            best_c = ct->commodity;
            best_dest = ct->station_index;
            best_stock = stock;
        }
    }
    if (out_commodity) *out_commodity = best_c;
    if (out_dest) *out_dest = best_dest;
    if (out_stock) *out_stock = best_stock;
    return best;
}

static void snw_clear_npcs(world_t *w)
{
    if (w == NULL) return;
    for (int i = 0; i < MAX_NPC_SHIPS; ++i) {
        w->npc_ships[i].active = false;
    }
    rebuild_characters_from_npcs(w);
}

static void snw_clear_contracts(world_t *w)
{
    if (w != NULL) memset(w->contracts, 0, sizeof(w->contracts));
}

static void snw_seed_finished(station_t *st, commodity_t c, int units)
{
    if (st == NULL || c >= COMMODITY_COUNT) return;
    int have = station_finished_count(st, c);
    if (have < units) {
        (void)station_finished_mint(st, c, units - have, NULL);
    }
}

static void snw_make_contract(contract_t *ct,
                              contract_action_t action,
                              int station,
                              int source,
                              commodity_t c,
                              float qty,
                              float price)
{
    if (ct == NULL) return;
    *ct = (contract_t){
        .active = true,
        .action = action,
        .station_index = (uint8_t)station,
        .target_index = source,
        .commodity = c,
        .quantity_needed = qty,
        .base_price = price,
        .claimed_by = -1,
    };
}

static int snw_setup_group(snw_group_t *group, uint32_t seed)
{
    if (group == NULL) return 0;
    memset(group, 0, sizeof(*group));
    group->seed = seed;
    group->group_id = seed;
    uint32_t rng = seed;

    world_t w;
    memset(&w, 0, sizeof(w));
    world_reset(&w);
    snw_clear_contracts(&w);
    snw_clear_npcs(&w);

    group->scenario_kind = (int)(snw_rng_next(&rng) % 5u);
    group->home_station = (int)(snw_rng_next(&rng) % 3u);
    group->role = (snw_rng_next(&rng) & 1u) ? NPC_ROLE_HAULER : NPC_ROLE_MINER;
    group->persona_risk = snw_rand01(&rng);
    group->persona_growth = snw_rand01(&rng);
    group->persona_patience = snw_rand01(&rng);

    int slot = spawn_npc(&w, group->home_station, (npc_role_t)group->role);
    if (slot < 0) return 0;
    npc_ship_t *npc = &w.npc_ships[slot];
    ship_t *ship = world_npc_ship_for(&w, slot);
    if (ship == NULL) return 0;
    npc->state = NPC_STATE_DOCKED;
    npc->state_timer = 0.0f;
    npc->known_contract_count = 0;
    memset(npc->known_contracts, 0, sizeof(npc->known_contracts));
    memset(&npc->knowledge, 0, sizeof(npc->knowledge));

    int mining_level = (int)(snw_rng_next(&rng) % 3u);
    int hold_level = (int)(snw_rng_next(&rng) % 3u);
    int tractor_level = (int)(snw_rng_next(&rng) % 3u);
    npc->ship.mining_level = ship->mining_level = mining_level;
    npc->ship.hold_level = ship->hold_level = hold_level;
    npc->ship.tractor_level = ship->tractor_level = tractor_level;

    group->desired_upgrade = snw_preferred_upgrade(npc, ship);
    group->desired_commodity = snw_upgrade_commodity(group->desired_upgrade);
    group->desired_units = group->desired_upgrade < SHIP_UPGRADE_COUNT
        ? (int)ceilf(upgrade_product_cost(ship, group->desired_upgrade))
        : 0;

    station_t *home = &w.stations[group->home_station];
    double balance = 80.0 + 940.0 * snw_rand01(&rng);
    if (group->scenario_kind == 1 || group->scenario_kind == 2) balance += 650.0;
    ledger_earn(home, npc->session_token, (float)balance);
    group->home_balance = ledger_balance(home, npc->session_token);

    if (group->desired_commodity < COMMODITY_COUNT) {
        int home_stock = 0;
        int remote_stock = 0;
        if (group->scenario_kind == 1) {
            home_stock = group->desired_units + (int)(snw_rng_next(&rng) % 6u);
        } else if (group->scenario_kind == 2) {
            home_stock = (int)(snw_rng_next(&rng) % (uint32_t)(group->desired_units > 1 ? group->desired_units : 1));
            remote_stock = group->desired_units + 4 + (int)(snw_rng_next(&rng) % 10u);
        } else {
            home_stock = (int)(snw_rng_next(&rng) % 5u);
            remote_stock = (int)(snw_rng_next(&rng) % 8u);
        }
        snw_seed_finished(home, group->desired_commodity, home_stock);
        int remote = (group->home_station + 1 + (int)(snw_rng_next(&rng) % 2u)) % 3;
        snw_seed_finished(&w.stations[remote], group->desired_commodity, remote_stock);
    }

    if (group->scenario_kind == 0 || group->scenario_kind == 3) {
        snw_seed_finished(home, COMMODITY_FERRITE_INGOT, 20);
        snw_make_contract(&w.contracts[0], CONTRACT_TRACTOR,
                          (group->home_station + 1) % 3,
                          -1,
                          COMMODITY_FERRITE_INGOT,
                          8.0f,
                          group->scenario_kind == 3 ? 70.0f : 25.0f);
    }
    if (group->scenario_kind == 3) {
        snw_seed_finished(home, COMMODITY_CUPRITE_INGOT, 12);
        snw_make_contract(&w.contracts[1], CONTRACT_TRACTOR,
                          (group->home_station + 2) % 3,
                          -1,
                          COMMODITY_CUPRITE_INGOT,
                          5.0f,
                          90.0f);
    }

    int mine_pressure = group->scenario_kind == 4 ? 1 : 0;
    if (mine_pressure) {
        for (int c = 0; c < COMMODITY_RAW_ORE_COUNT; ++c) {
            home->_inventory_cache[c] = 0.0f;
        }
    }

    commodity_t best_contract_commodity = COMMODITY_COUNT;
    int best_contract_dest = -1;
    int best_contract_stock = 0;
    double best_contract_value = snw_best_contract_value(&w,
                                                         group->home_station,
                                                         &best_contract_commodity,
                                                         &best_contract_dest,
                                                         &best_contract_stock);
    int remote_refit_stock = 0;
    int remote_refit_station = snw_best_remote_source(&w,
                                                      group->home_station,
                                                      group->desired_commodity,
                                                      &remote_refit_stock);
    int home_refit_stock = group->desired_commodity < COMMODITY_COUNT
        ? station_finished_count(home, group->desired_commodity)
        : 0;
    double refit_unit_price = group->desired_commodity < COMMODITY_COUNT
        ? station_sell_price(home, group->desired_commodity)
        : 0.0;
    if (refit_unit_price < 0.01) refit_unit_price = 1.0;
    double refit_cost = refit_unit_price * (double)group->desired_units;
    double can_pay_refit = group->home_balance + 1e-6 >= refit_cost ? 1.0 : 0.0;
    double route_km = 0.0;
    if (remote_refit_station >= 0) {
        route_km = v2_len(v2_sub(w.stations[remote_refit_station].pos, home->pos)) / 1000.0;
    }

    for (size_t oi = 0; oi < SNW_OPTION_COUNT; ++oi) {
        snw_outcome_t *out = &group->outcomes[oi];
        const snw_hypothesis_t *h = snw_hypothesis(oi);
        memset(out, 0, sizeof(*out));
        out->legal = 1;

        switch (h->option) {
        case SNW_OPT_WAIT:
            out->utility = (best_contract_value <= 0.0 && !mine_pressure && home_refit_stock <= 0) ? 0.08 : -0.35;
            break;
        case SNW_OPT_MINE_HOME:
            out->legal = npc->role == NPC_ROLE_MINER || group->persona_growth > 0.45;
            out->utility = out->legal ? (mine_pressure ? 3.8 : 0.25) : -1.0;
            out->credit_delta = mine_pressure ? 30.0 : 5.0;
            break;
        case SNW_OPT_HAUL_CONTRACT:
            out->legal = best_contract_value > 0.0;
            out->contract_value = best_contract_value;
            out->cargo_moved = best_contract_stock > 0 ? snw_clip(best_contract_value / 50.0, 1.0, 10.0) : 0.0;
            out->travel = out->legal && best_contract_dest != group->home_station;
            out->credit_delta = best_contract_value;
            out->utility = out->legal
                ? (best_contract_value / 85.0) + (npc->role == NPC_ROLE_HAULER ? 0.45 : 0.0)
                : -0.85;
            break;
        case SNW_OPT_GOSSIP_COURIER:
            out->travel = 1;
            out->utility = (best_contract_value <= 0.0 && !mine_pressure)
                ? 0.45 + group->persona_patience * 0.25
                : 0.05;
            break;
        case SNW_OPT_SELF_REFIT_HOME:
            out->legal = group->desired_upgrade < SHIP_UPGRADE_COUNT &&
                         home_refit_stock >= group->desired_units &&
                         can_pay_refit > 0.5;
            out->self_upgrade = out->legal;
            out->refit_progress = out->legal ? 1.0 : 0.0;
            out->credit_delta = out->legal ? -refit_cost : 0.0;
            out->utility = out->legal
                ? 6.0 + group->persona_growth * 2.0 - refit_cost / 900.0
                : -0.75;
            break;
        case SNW_OPT_IMPORT_FRAME:
        case SNW_OPT_IMPORT_LASER:
        case SNW_OPT_IMPORT_TRACTOR: {
            commodity_t candidate = h->option == SNW_OPT_IMPORT_FRAME ? COMMODITY_FRAME :
                                    h->option == SNW_OPT_IMPORT_LASER ? COMMODITY_LASER_MODULE :
                                    COMMODITY_TRACTOR_MODULE;
            out->legal = group->desired_commodity == candidate &&
                         home_refit_stock < group->desired_units &&
                         remote_refit_stock >= group->desired_units &&
                         can_pay_refit > 0.5;
            out->travel = out->legal;
            out->import_module = out->legal;
            out->cargo_moved = out->legal ? (double)group->desired_units : 0.0;
            out->refit_progress = out->legal ? 0.65 : 0.0;
            out->credit_delta = out->legal ? -refit_cost : 0.0;
            out->utility = out->legal
                ? 4.8 + group->persona_growth * 1.6 - route_km * 0.08
                : -0.65;
            break;
        }
        default:
            out->legal = 0;
            out->utility = -1.0;
            break;
        }
        if (!out->legal) out->utility -= 1.0;
    }

    double best = -1e300;
    double second = -1e300;
    group->target = 0;
    group->baseline = best_contract_value > 0.0 ? SNW_OPT_HAUL_CONTRACT :
                      mine_pressure ? SNW_OPT_MINE_HOME :
                      SNW_OPT_GOSSIP_COURIER;
    for (size_t i = 0; i < SNW_OPTION_COUNT; ++i) {
        double u = group->outcomes[i].utility;
        if (u > best) {
            second = best;
            best = u;
            group->target = i;
        } else if (u > second) {
            second = u;
        }
    }
    group->second_best_utility = second;

    for (size_t oi = 0; oi < SNW_OPTION_COUNT; ++oi) {
        nn_real *f = group->features[oi];
        memset(f, 0, sizeof(group->features[oi]));
        const snw_hypothesis_t *h = snw_hypothesis(oi);
        f[0] = 1.0f;
        f[1] = (nn_real)oi / (nn_real)(SNW_OPTION_COUNT - 1u);
        for (size_t k = 0; k < SNW_OPTION_COUNT; ++k) f[2 + k] = oi == k ? 1.0f : 0.0f;
        f[10] = npc->role == NPC_ROLE_MINER ? 1.0f : 0.0f;
        f[11] = npc->role == NPC_ROLE_HAULER ? 1.0f : 0.0f;
        f[12] = (nn_real)ship->mining_level / 5.0f;
        f[13] = (nn_real)ship->hold_level / 5.0f;
        f[14] = (nn_real)ship->tractor_level / 5.0f;
        f[15] = (nn_real)group->home_station / 2.0f;
        f[16] = (nn_real)snw_clip(group->home_balance / 1500.0, 0.0, 1.5);
        f[17] = (nn_real)home_refit_stock / 24.0f;
        f[18] = (nn_real)remote_refit_stock / 24.0f;
        f[19] = (nn_real)group->desired_units / 20.0f;
        f[20] = (nn_real)snw_clip(refit_cost / 1500.0, 0.0, 2.0);
        f[21] = (nn_real)can_pay_refit;
        f[22] = group->desired_upgrade == SHIP_UPGRADE_MINING ? 1.0f : 0.0f;
        f[23] = group->desired_upgrade == SHIP_UPGRADE_HOLD ? 1.0f : 0.0f;
        f[24] = group->desired_upgrade == SHIP_UPGRADE_TRACTOR ? 1.0f : 0.0f;
        f[25] = group->desired_commodity == COMMODITY_FRAME ? 1.0f : 0.0f;
        f[26] = group->desired_commodity == COMMODITY_LASER_MODULE ? 1.0f : 0.0f;
        f[27] = group->desired_commodity == COMMODITY_TRACTOR_MODULE ? 1.0f : 0.0f;
        f[28] = (nn_real)snw_clip(best_contract_value / 600.0, 0.0, 2.0);
        f[29] = (nn_real)best_contract_stock / 24.0f;
        f[30] = best_contract_dest >= 0 && best_contract_dest != group->home_station ? 1.0f : 0.0f;
        f[31] = mine_pressure ? 1.0f : 0.0f;
        f[32] = (nn_real)group->persona_risk;
        f[33] = (nn_real)group->persona_growth;
        f[34] = (nn_real)group->persona_patience;
        f[35] = (nn_real)snw_clip(route_km / 20.0, 0.0, 1.0);
        f[36] = station_has_module(home, MODULE_DOCK) ? 1.0f : 0.0f;
        f[37] = station_has_module(home, MODULE_SHIPYARD) ? 1.0f : 0.0f;
        f[38] = station_has_module(home, MODULE_FURNACE) ? 1.0f : 0.0f;
        f[39] = station_has_module(home, MODULE_FRAME_PRESS) ? 1.0f : 0.0f;
        f[40] = station_has_module(home, MODULE_LASER_FAB) ? 1.0f : 0.0f;
        f[41] = station_has_module(home, MODULE_TRACTOR_FAB) ? 1.0f : 0.0f;
        f[42] = h->option == SNW_OPT_WAIT ? 1.0f : 0.0f;
        f[43] = h->option == SNW_OPT_SELF_REFIT_HOME ? (nn_real)(home_refit_stock >= group->desired_units) : 0.0f;
        f[44] = (h->option == SNW_OPT_IMPORT_FRAME ||
                 h->option == SNW_OPT_IMPORT_LASER ||
                 h->option == SNW_OPT_IMPORT_TRACTOR) ? 1.0f : 0.0f;
        f[45] = group->outcomes[oi].legal ? 1.0f : 0.0f;
        f[46] = group->outcomes[oi].travel ? 1.0f : 0.0f;
        f[47] = group->outcomes[oi].self_upgrade ? 1.0f : 0.0f;
        f[48] = group->outcomes[oi].import_module ? 1.0f : 0.0f;
        f[49] = (nn_real)snw_clip(group->outcomes[oi].credit_delta / 1000.0, -1.5, 2.0);
        f[50] = (nn_real)group->outcomes[oi].refit_progress;
        f[51] = (nn_real)snw_clip(group->outcomes[oi].contract_value / 600.0, 0.0, 2.0);
        f[52] = (nn_real)snw_clip(group->outcomes[oi].cargo_moved / 16.0, 0.0, 1.0);
        f[53] = best_contract_commodity == COMMODITY_FERRITE_INGOT ? 1.0f : 0.0f;
        f[54] = best_contract_commodity == COMMODITY_CUPRITE_INGOT ? 1.0f : 0.0f;
        f[55] = best_contract_commodity == COMMODITY_CRYSTAL_INGOT ? 1.0f : 0.0f;
    }
    return 1;
}

static int snw_dataset_build(snw_dataset_t *dataset, size_t count, uint32_t seed)
{
    if (dataset == NULL) return 0;
    memset(dataset, 0, sizeof(*dataset));
    dataset->groups = calloc(count, sizeof(*dataset->groups));
    if (dataset->groups == NULL) return 0;
    dataset->count = count;
    for (size_t i = 0; i < count; ++i) {
        if (!snw_setup_group(&dataset->groups[i], seed + (uint32_t)i * UINT32_C(7919))) {
            return 0;
        }
    }
    return 1;
}

static void snw_dataset_free(snw_dataset_t *dataset)
{
    if (dataset == NULL) return;
    free(dataset->groups);
    memset(dataset, 0, sizeof(*dataset));
}

static NeuralNet *snw_create_model(uint32_t seed)
{
    static const size_t layers[] = {SNW_FEATURE_COUNT, 32u, 16u, 1u};
    NNConfig config;
    memset(&config, 0, sizeof(config));
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int snw_train_model(NeuralNet *net, const snw_dataset_t *train, const snw_config_t *config)
{
    if (net == NULL || train == NULL || config == NULL || train->count == 0u) return 0;
    uint32_t rng = config->seed ^ UINT32_C(0x776f726b);
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss_sum = 0.0;
        for (size_t step = 0; step < train->count; ++step) {
            size_t idx = (size_t)(snw_rng_next(&rng) % (uint32_t)train->count);
            const snw_group_t *group = &train->groups[idx];
            double gap = group->outcomes[group->target].utility - group->second_best_utility;
            nn_real weight = (nn_real)(1.0 + snw_clip(gap, 0.0, 8.0));
            nn_real loss = nn_train_softmax_scorer_weighted(net,
                                                            &group->features[0][0],
                                                            SNW_OPTION_COUNT,
                                                            group->target,
                                                            weight,
                                                            config->learning_rate);
            if (!isfinite((double)loss)) return 0;
            loss_sum += (double)loss;
        }
        printf("signal_npc_worker epoch %zu loss=%.4f groups=%zu\n",
               epoch + 1u, loss_sum / (double)train->count, train->count);
    }
    return 1;
}

static snw_metrics_t snw_evaluate_model(NeuralNet *net, const snw_dataset_t *test)
{
    snw_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    if (net == NULL || test == NULL || test->count == 0u) return metrics;

    nn_real scores[SNW_OPTION_COUNT];
    nn_real probabilities[SNW_OPTION_COUNT];
    size_t ranked[SNW_OPTION_COUNT];
    for (size_t i = 0; i < test->count; ++i) {
        const snw_group_t *group = &test->groups[i];
        if (!nn_score_action_rows(net,
                                  &group->features[0][0],
                                  SNW_OPTION_COUNT,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, SNW_OPTION_COUNT, ranked)) {
            continue;
        }
        size_t selected = ranked[0];
        size_t rank = SNW_OPTION_COUNT;
        for (size_t j = 0; j < SNW_OPTION_COUNT; ++j) {
            if (ranked[j] == group->target) {
                rank = j;
                break;
            }
        }
        if (rank >= SNW_OPTION_COUNT) continue;
        const snw_outcome_t *sel = &group->outcomes[selected];
        const snw_outcome_t *oracle = &group->outcomes[group->target];
        const snw_outcome_t *base = &group->outcomes[group->baseline];
        ++metrics.groups;
        metrics.top1 += rank == 0 ? 1u : 0u;
        metrics.top3 += rank < 3u ? 1u : 0u;
        metrics.selected_utility += sel->utility;
        metrics.oracle_utility += oracle->utility;
        metrics.regret += oracle->utility - sel->utility;
        metrics.baseline_utility += base->utility;
        metrics.baseline_regret += oracle->utility - base->utility;
        metrics.selected_legal += sel->legal ? 1u : 0u;
        metrics.selected_self_upgrade += sel->self_upgrade ? 1u : 0u;
        metrics.selected_import += sel->import_module ? 1u : 0u;
        metrics.selected_travel += sel->travel ? 1u : 0u;
        metrics.option_selected[selected]++;
        metrics.option_target[group->target]++;
    }
    double denom = metrics.groups == 0u ? 1.0 : (double)metrics.groups;
    metrics.selected_utility /= denom;
    metrics.oracle_utility /= denom;
    metrics.regret /= denom;
    metrics.baseline_utility /= denom;
    metrics.baseline_regret /= denom;
    return metrics;
}

static void snw_json_string(FILE *fp, const char *s)
{
    fputc('"', fp);
    if (s != NULL) {
        for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
            if (*p == '"' || *p == '\\') {
                fputc('\\', fp);
                fputc((int)*p, fp);
            } else if (*p == '\n') {
                fputs("\\n", fp);
            } else {
                fputc((int)*p, fp);
            }
        }
    }
    fputc('"', fp);
}

static int snw_write_trace(const char *path, const snw_dataset_t *train, const snw_dataset_t *test)
{
    if (path == NULL) return 1;
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open Signal NPC worker trace: %s\n", path);
        return 0;
    }
    const snw_dataset_t *sets[2] = {train, test};
    const char *names[2] = {"train", "test"};
    for (size_t si = 0; si < 2; ++si) {
        const snw_dataset_t *set = sets[si];
        if (set == NULL) continue;
        for (size_t gi = 0; gi < set->count; ++gi) {
            const snw_group_t *g = &set->groups[gi];
            for (size_t oi = 0; oi < SNW_OPTION_COUNT; ++oi) {
                fprintf(fp, "{\"schema\":");
                snw_json_string(fp, SNW_TRACE_SCHEMA);
                fprintf(fp, ",\"split\":");
                snw_json_string(fp, names[si]);
                fprintf(fp, ",\"group_id\":%" PRIu64, g->group_id);
                fprintf(fp, ",\"candidate_index\":%zu", oi);
                fprintf(fp, ",\"candidate_name\":");
                snw_json_string(fp, snw_hypothesis(oi)->name);
                fprintf(fp, ",\"target_index\":%zu", g->target);
                fprintf(fp, ",\"baseline_index\":%zu", g->baseline);
                fprintf(fp, ",\"utility\":%.9g", g->outcomes[oi].utility);
                fprintf(fp, ",\"legal\":%s", g->outcomes[oi].legal ? "true" : "false");
                fprintf(fp, ",\"self_upgrade\":%s", g->outcomes[oi].self_upgrade ? "true" : "false");
                fprintf(fp, ",\"import_module\":%s", g->outcomes[oi].import_module ? "true" : "false");
                fprintf(fp, ",\"home_station\":%d", g->home_station);
                fprintf(fp, ",\"role\":%d", g->role);
                fprintf(fp, ",\"desired_upgrade\":%d", (int)g->desired_upgrade);
                fprintf(fp, ",\"features\":[");
                for (size_t fi = 0; fi < SNW_FEATURE_COUNT; ++fi) {
                    fprintf(fp, "%s%.9g", fi == 0 ? "" : ",", (double)g->features[oi][fi]);
                }
                fputs("]}\n", fp);
            }
        }
    }
    fclose(fp);
    return 1;
}

static int snw_save_checkpoint(NeuralNet *net, const char *path, const char *trace_path, char out_hash[65])
{
    if (net == NULL || path == NULL) return 0;
    NNCheckpointMeta meta;
    nn_checkpoint_meta_init(&meta, 1u);
    if (!nn_checkpoint_meta_set_parent_model_hash(&meta, "none") ||
        !nn_checkpoint_meta_set_benchmark_report_hash(&meta, "pending") ||
        !nn_checkpoint_meta_set_training_trace_root(&meta, trace_path != NULL ? trace_path : "generated_in_memory") ||
        !nn_checkpoint_meta_set_notes(&meta, "Signal NPC worker strategic option scorer trained from Signal simulator state and certified utility rows") ||
        !nn_checkpoint_meta_set_feature_set(&meta, "signal-npc-worker-v1")) {
        return 0;
    }
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static double snw_rate(size_t numerator, size_t denominator)
{
    return denominator == 0u ? 0.0 : (double)numerator / (double)denominator;
}

static void snw_write_option_counts(FILE *fp, const size_t counts[SNW_OPTION_COUNT])
{
    fputc('{', fp);
    for (size_t i = 0; i < SNW_OPTION_COUNT; ++i) {
        if (i > 0) fputc(',', fp);
        snw_json_string(fp, snw_hypothesis(i)->name);
        fprintf(fp, ":%zu", counts[i]);
    }
    fputc('}', fp);
}

static void snw_write_summary(FILE *fp,
                              const snw_config_t *config,
                              const char *model_hash,
                              const NeuralNet *net,
                              const snw_dataset_t *train,
                              const snw_dataset_t *test,
                              const snw_metrics_t *metrics)
{
    fprintf(fp, "{\"schema\":");
    snw_json_string(fp, SNW_SUMMARY_SCHEMA);
    fprintf(fp, ",\"task\":");
    snw_json_string(fp, SNW_TASK_NAME);
    fprintf(fp, ",\"trace_schema\":");
    snw_json_string(fp, SNW_TRACE_SCHEMA);
    fprintf(fp, ",\"training_mode\":\"counterfactual_worker_option_utility\"");
    fprintf(fp, ",\"candidate_layer\":\"npc_worker_strategy_v1\"");
    fprintf(fp, ",\"model_hash\":");
    snw_json_string(fp, model_hash != NULL ? model_hash : "unsaved");
    fprintf(fp, ",\"trace_path\":");
    snw_json_string(fp, config->trace_path);
    fprintf(fp, ",\"train_groups\":%zu", train != NULL ? train->count : 0u);
    fprintf(fp, ",\"test_groups\":%zu", test != NULL ? test->count : 0u);
    fprintf(fp, ",\"candidate_count\":%u", SNW_OPTION_COUNT);
    fprintf(fp, ",\"feature_count\":%u", SNW_FEATURE_COUNT);
    fprintf(fp, ",\"epochs\":%zu", config->epochs);
    fprintf(fp, ",\"learning_rate\":%.9g", (double)config->learning_rate);
    fprintf(fp, ",\"model_parameters\":%zu", net != NULL ? nn_parameter_count(net) : 0u);
    fprintf(fp, ",\"model_forward_flops\":%zu", net != NULL ? nn_forward_flop_count(net) : 0u);
    fprintf(fp, ",\"top1\":%.9g", snw_rate(metrics->top1, metrics->groups));
    fprintf(fp, ",\"top3\":%.9g", snw_rate(metrics->top3, metrics->groups));
    fprintf(fp, ",\"selected_utility\":%.9g", metrics->selected_utility);
    fprintf(fp, ",\"oracle_utility\":%.9g", metrics->oracle_utility);
    fprintf(fp, ",\"regret\":%.9g", metrics->regret);
    fprintf(fp, ",\"baseline_utility\":%.9g", metrics->baseline_utility);
    fprintf(fp, ",\"baseline_regret\":%.9g", metrics->baseline_regret);
    fprintf(fp, ",\"selected_legal_rate\":%.9g", snw_rate(metrics->selected_legal, metrics->groups));
    fprintf(fp, ",\"selected_self_upgrade_rate\":%.9g", snw_rate(metrics->selected_self_upgrade, metrics->groups));
    fprintf(fp, ",\"selected_import_rate\":%.9g", snw_rate(metrics->selected_import, metrics->groups));
    fprintf(fp, ",\"selected_travel_rate\":%.9g", snw_rate(metrics->selected_travel, metrics->groups));
    fprintf(fp, ",\"option_selected\":");
    snw_write_option_counts(fp, metrics->option_selected);
    fprintf(fp, ",\"option_target\":");
    snw_write_option_counts(fp, metrics->option_target);
    fputs("}\n", fp);
}

static int snw_write_summary_file(const char *path,
                                  const snw_config_t *config,
                                  const char *model_hash,
                                  const NeuralNet *net,
                                  const snw_dataset_t *train,
                                  const snw_dataset_t *test,
                                  const snw_metrics_t *metrics)
{
    if (path == NULL) return 1;
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open Signal NPC worker summary: %s\n", path);
        return 0;
    }
    snw_write_summary(fp, config, model_hash, net, train, test, metrics);
    fclose(fp);
    return 1;
}

static void snw_usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--schema] [--train-groups n] [--test-groups n] "
            "[--epochs n] [--learning-rate r] [--seed n] "
            "[--trace path] [--summary path] [--save path]\n",
            argv0);
}

static int snw_parse_size(const char *s, size_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return 0;
    *out = (size_t)v;
    return 1;
}

static int snw_parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int snw_parse_real(const char *s, nn_real *out)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) return 0;
    *out = (nn_real)v;
    return 1;
}

static int snw_parse_config(int argc, char **argv, snw_config_t *config, int *schema_only)
{
    if (config == NULL || schema_only == NULL) return 0;
    config->summary_path = "build/float/signal_npc_worker/summary.jsonl";
    config->trace_path = "build/float/signal_npc_worker/trace.jsonl";
    config->save_path = "build/float/signal_npc_worker/signal_npc_worker.nnckpt";
    config->seed = 6262u;
    config->train_groups = 64u;
    config->test_groups = 24u;
    config->epochs = 4u;
    config->learning_rate = (nn_real)0.004;
    *schema_only = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schema") == 0) {
            *schema_only = 1;
        } else if (strcmp(argv[i], "--train-groups") == 0 && i + 1 < argc) {
            if (!snw_parse_size(argv[++i], &config->train_groups)) return 0;
        } else if (strcmp(argv[i], "--test-groups") == 0 && i + 1 < argc) {
            if (!snw_parse_size(argv[++i], &config->test_groups)) return 0;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!snw_parse_size(argv[++i], &config->epochs)) return 0;
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            if (!snw_parse_real(argv[++i], &config->learning_rate)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!snw_parse_u32(argv[++i], &config->seed)) return 0;
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--summary") == 0 && i + 1 < argc) {
            config->summary_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else {
            return 0;
        }
    }
    return *schema_only || (config->train_groups > 0u && config->test_groups > 0u && config->epochs > 0u);
}

static void snw_print_schema(void)
{
    printf("{\"task\":\"%s\",\"trace_schema\":\"%s\",\"summary_schema\":\"%s\","
           "\"candidate_count\":%u,\"feature_count\":%u,"
           "\"feature_set\":\"signal-npc-worker-v1\",\"options\":[",
           SNW_TASK_NAME, SNW_TRACE_SCHEMA, SNW_SUMMARY_SCHEMA,
           SNW_OPTION_COUNT, SNW_FEATURE_COUNT);
    for (size_t i = 0; i < SNW_OPTION_COUNT; ++i) {
        if (i > 0) fputc(',', stdout);
        snw_json_string(stdout, snw_hypothesis(i)->name);
    }
    puts("]}");
}

int main(int argc, char **argv)
{
    snw_config_t config;
    int schema_only = 0;
    if (!snw_parse_config(argc, argv, &config, &schema_only)) {
        snw_usage(argv[0]);
        return 2;
    }
    if (schema_only) {
        snw_print_schema();
        return 0;
    }

    snw_dataset_t train;
    snw_dataset_t test;
    if (!snw_dataset_build(&train, config.train_groups, config.seed) ||
        !snw_dataset_build(&test, config.test_groups, config.seed ^ UINT32_C(0x9e3779b9))) {
        fprintf(stderr, "failed to build Signal NPC worker datasets\n");
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }

    if (!snw_write_trace(config.trace_path, &train, &test)) {
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }

    NeuralNet *net = snw_create_model(config.seed ^ UINT32_C(0x4e504357));
    if (net == NULL) {
        fprintf(stderr, "failed to create Signal NPC worker model\n");
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }

    if (!snw_train_model(net, &train, &config)) {
        nn_free(net);
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }

    char model_hash[65] = {0};
    if (!snw_save_checkpoint(net, config.save_path, config.trace_path, model_hash)) {
        fprintf(stderr, "failed to save Signal NPC worker checkpoint: %s\n", config.save_path);
        nn_free(net);
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }

    snw_metrics_t metrics = snw_evaluate_model(net, &test);
    snw_write_summary(stdout, &config, model_hash, net, &train, &test, &metrics);
    if (!snw_write_summary_file(config.summary_path,
                                &config,
                                model_hash,
                                net,
                                &train,
                                &test,
                                &metrics)) {
        nn_free(net);
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }

    printf("signal_npc_worker top1=%.2f%% top3=%.2f%% regret=%.4f baseline_regret=%.4f groups=%zu\n",
           snw_rate(metrics.top1, metrics.groups) * 100.0,
           snw_rate(metrics.top3, metrics.groups) * 100.0,
           metrics.regret,
           metrics.baseline_regret,
           metrics.groups);

    nn_free(net);
    snw_dataset_free(&train);
    snw_dataset_free(&test);
    return 0;
}
