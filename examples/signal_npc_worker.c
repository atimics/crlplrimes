#include "nn.h"

#include "commodity.h"
#include "game_sim.h"
#include "manifest.h"
#include "sim_ai.h"
#include "station_authority.h"
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
#define SNW_SOCIAL_SLICE_SCHEMA "crlp.signal_npc_worker.social_slice.v1"
#define SNW_PERSISTENT_SCHEMA "crlp.signal_npc_worker.persistent_world.v1"
#define SNW_PERSISTENT_SUMMARY_SCHEMA "crlp.signal_npc_worker.persistent_summary.v1"
#define SNW_TASK_NAME "signal_npc_worker"
#define SNW_OPTION_COUNT 12u
#define SNW_FEATURE_COUNT 78u
#define SNW_PERSONA_COUNT 3u
#define SNW_PERSISTENT_MAX_STATIONS 128u
#define SNW_PERSISTENT_FACTIONS 3u

typedef enum {
    SNW_OPT_WAIT = 0,
    SNW_OPT_MINE_HOME = 1,
    SNW_OPT_HAUL_CONTRACT = 2,
    SNW_OPT_GOSSIP_COURIER = 3,
    SNW_OPT_SELF_REFIT_HOME = 4,
    SNW_OPT_IMPORT_FRAME = 5,
    SNW_OPT_IMPORT_LASER = 6,
    SNW_OPT_IMPORT_TRACTOR = 7,
    SNW_OPT_SUPPLY_FRONTIER = 8,
    SNW_OPT_ESCORT_CONVOY = 9,
    SNW_OPT_PATROL_ROUTE = 10,
    SNW_OPT_RISKY_PROFIT = 11,
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
    double faction_delta;
    double frontier_delta;
    double pirate_delta;
    int legal;
    int travel;
    int self_upgrade;
    int import_module;
    int faction_work;
    int pirate_work;
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
    int faction_id;
    int faction_goal;
    double faction_reputation;
    double frontier_need;
    double expansion_pressure;
    double route_pirate_risk;
    double route_recent_attack;
    double escort_available;
    double convoy_bonus;
    double pirate_bounty;
    int contract_at_own_station;
    int contract_at_faction_station;
    double station_credit_weight;
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
    int persistent_economy;
    size_t persistent_ticks;
    size_t persistent_npcs;
    size_t persistent_stations;
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
    size_t social_groups;
    size_t social_intent_rows;
    size_t social_memory_updates;
    size_t social_gossip_facts;
    size_t social_selected_legal;
    size_t social_memory_influenced;
    size_t social_gossip_influenced;
    size_t social_persona_divergent;
    double social_selected_utility;
    double social_oracle_utility;
    double social_regret;
} snw_metrics_t;

typedef struct {
    const char *name;
    double risk_weight;
    double growth_weight;
    double logistics_weight;
    double patience_weight;
} snw_persona_t;

typedef struct {
    double route_risk;
    double job_profit;
    double refit_need;
    double gossip_contract;
    double gossip_route_risk;
    int has_memory;
    int heard_gossip;
} snw_worker_memory_t;

typedef struct {
    size_t worker;
    size_t base_option;
    size_t selected_option;
    const char *intent;
    const char *memory;
    const char *gossip;
    int memory_influenced;
    int gossip_influenced;
    int persona_divergent;
    double selected_utility;
    double oracle_utility;
    double regret;
} snw_social_decision_t;

static const snw_hypothesis_t SNW_HYPOTHESES[SNW_OPTION_COUNT] = {
    {SNW_OPT_WAIT, "wait"},
    {SNW_OPT_MINE_HOME, "mine_home_ore"},
    {SNW_OPT_HAUL_CONTRACT, "haul_best_contract"},
    {SNW_OPT_GOSSIP_COURIER, "gossip_courier"},
    {SNW_OPT_SELF_REFIT_HOME, "self_refit_home"},
    {SNW_OPT_IMPORT_FRAME, "import_frame_modules"},
    {SNW_OPT_IMPORT_LASER, "import_laser_modules"},
    {SNW_OPT_IMPORT_TRACTOR, "import_tractor_modules"},
    {SNW_OPT_SUPPLY_FRONTIER, "supply_frontier"},
    {SNW_OPT_ESCORT_CONVOY, "escort_convoy"},
    {SNW_OPT_PATROL_ROUTE, "patrol_route"},
    {SNW_OPT_RISKY_PROFIT, "take_risky_profit_contract"},
};

static const snw_persona_t SNW_PERSONAS[SNW_PERSONA_COUNT] = {
    {"mira_cautious_hauler", 1.15, 0.25, 0.65, 0.80},
    {"voss_greedy_miner", -0.25, 0.95, 0.20, 0.15},
    {"iko_logistics_nerd", 0.35, 0.65, 1.10, 0.45},
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

static const snw_persona_t *snw_persona(size_t idx)
{
    return idx < SNW_PERSONA_COUNT ? &SNW_PERSONAS[idx] : &SNW_PERSONAS[0];
}

static size_t snw_group_worker(const snw_group_t *group)
{
    if (group == NULL) return 0u;
    return (size_t)((group->seed / UINT32_C(7919)) % SNW_PERSONA_COUNT);
}

static const char *snw_intent_for(size_t option, const snw_group_t *group)
{
    switch (snw_hypothesis(option)->option) {
    case SNW_OPT_WAIT:
        return "waiting: no safe profitable work beats patience";
    case SNW_OPT_MINE_HOME:
        return "mining home ore: low-risk resource loop";
    case SNW_OPT_HAUL_CONTRACT:
        return "hauling contract: best payout and route looks acceptable";
    case SNW_OPT_GOSSIP_COURIER:
        return "running gossip courier: scouting work for the crew";
    case SNW_OPT_SELF_REFIT_HOME:
        return "refitting at home: module stock can improve future jobs";
    case SNW_OPT_IMPORT_FRAME:
        return "importing frame modules: remote stock unlocks refit";
    case SNW_OPT_IMPORT_LASER:
        return "importing laser modules: remote stock unlocks combat/mining refit";
    case SNW_OPT_IMPORT_TRACTOR:
        return "importing tractor modules: remote stock improves hauling reach";
    case SNW_OPT_SUPPLY_FRONTIER:
        return "supplying frontier: faction expansion needs materials more than cash";
    case SNW_OPT_ESCORT_CONVOY:
        return "escorting convoy: pirate pressure makes shared travel valuable";
    case SNW_OPT_PATROL_ROUTE:
        return "patrolling route: reduce pirate pressure before trade resumes";
    case SNW_OPT_RISKY_PROFIT:
        return "taking risky profit contract: payout beats political caution";
    default:
        break;
    }
    (void)group;
    return "choosing worker option: highest certified utility score";
}

static const char *snw_memory_for(size_t option, const snw_outcome_t *out)
{
    if (out != NULL && out->import_module) return "module import helped future refit planning";
    if (out != NULL && out->pirate_work && out->pirate_delta < 0.0) return "pirate pressure fell after defensive work";
    if (out != NULL && out->frontier_delta > 0.0) return "frontier supply improved faction expansion";
    if (out != NULL && out->pirate_work) return "pirate route stayed dangerous and needs caution";
    if (out != NULL && out->faction_delta > 0.0) return "faction reputation improved after aligned work";
    if (out != NULL && out->self_upgrade) return "self refit completed and raises growth confidence";
    if (out != NULL && out->contract_value > 0.0) return "contract route paid and becomes preferred work";
    if (snw_hypothesis(option)->option == SNW_OPT_MINE_HOME) return "home ore loop was productive";
    if (snw_hypothesis(option)->option == SNW_OPT_GOSSIP_COURIER) return "scouting run refreshed shared route facts";
    return "idle outcome recorded for patience baseline";
}

static const char *snw_gossip_for(size_t option, const snw_outcome_t *out)
{
    if (out != NULL && out->contract_value > 0.0) return "good contract value seen near home station";
    if (out != NULL && out->pirate_work && out->pirate_delta < 0.0) return "pirate route pressure reduced after patrol or escort";
    if (out != NULL && out->pirate_work) return "pirate threat seen on profitable route";
    if (out != NULL && out->frontier_delta > 0.0) return "frontier station needs supply for expansion";
    if (out != NULL && out->faction_delta > 0.0) return "faction-aligned work is earning trust";
    if (out != NULL && out->import_module) return "remote module stock exists for crew refits";
    if (out != NULL && out->self_upgrade) return "home station has enough stock for a refit";
    if (snw_hypothesis(option)->option == SNW_OPT_GOSSIP_COURIER) return "route scout reports low immediate work pressure";
    return "no high-value gossip published";
}

static double snw_social_bonus(const snw_group_t *group,
                               const snw_worker_memory_t *memory,
                               const snw_persona_t *persona,
                               size_t option,
                               int *memory_used,
                               int *gossip_used)
{
    if (memory_used) *memory_used = 0;
    if (gossip_used) *gossip_used = 0;
    if (group == NULL || memory == NULL || persona == NULL) return 0.0;

    const snw_outcome_t *out = &group->outcomes[option];
    double bonus = 0.0;
    if (out->travel && memory->route_risk > 0.25) {
        bonus -= persona->risk_weight * memory->route_risk * 0.30;
        if (memory_used) *memory_used = 1;
    }
    if (out->pirate_work && memory->route_risk > 0.20) {
        bonus += memory->route_risk * (0.20 + persona->risk_weight * 0.16);
        if (memory_used) *memory_used = 1;
    }
    if (out->contract_value > 0.0 && memory->job_profit > 0.10) {
        bonus += memory->job_profit * (0.15 + persona->logistics_weight * 0.10);
        if (memory_used) *memory_used = 1;
    }
    if (out->frontier_delta > 0.0 && memory->refit_need > 0.10) {
        bonus += memory->refit_need * (0.12 + persona->growth_weight * 0.10);
        if (memory_used) *memory_used = 1;
    }
    if ((out->self_upgrade || out->import_module) && memory->refit_need > 0.10) {
        bonus += memory->refit_need * (0.20 + persona->growth_weight * 0.10);
        if (memory_used) *memory_used = 1;
    }
    if (out->contract_value > 0.0 && memory->gossip_contract > 0.10) {
        bonus += memory->gossip_contract * (0.12 + persona->logistics_weight * 0.08);
        if (gossip_used) *gossip_used = 1;
    }
    if (out->travel && memory->gossip_route_risk > 0.20) {
        bonus -= memory->gossip_route_risk * (0.10 + persona->risk_weight * 0.08);
        if (gossip_used) *gossip_used = 1;
    }
    if (out->pirate_work && memory->gossip_route_risk > 0.20) {
        bonus += memory->gossip_route_risk * (0.22 + persona->risk_weight * 0.14);
        if (gossip_used) *gossip_used = 1;
    }
    if (snw_hypothesis(option)->option == SNW_OPT_WAIT) {
        bonus += persona->patience_weight * 0.03;
    }
    if (snw_hypothesis(option)->option == SNW_OPT_HAUL_CONTRACT) {
        bonus += persona->logistics_weight * 0.04;
    }
    if (snw_hypothesis(option)->option == SNW_OPT_MINE_HOME) {
        bonus += persona->growth_weight * 0.04;
    }
    if (snw_hypothesis(option)->option == SNW_OPT_SUPPLY_FRONTIER) {
        bonus += persona->growth_weight * 0.05 + persona->logistics_weight * 0.03;
    }
    if (snw_hypothesis(option)->option == SNW_OPT_RISKY_PROFIT) {
        bonus += (1.0 - persona->risk_weight) * 0.06;
    }
    return bonus;
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

    world_t *w = calloc(1u, sizeof(*w));
    if (w == NULL) return 0;
    world_reset(w);
    snw_clear_contracts(w);
    snw_clear_npcs(w);

    group->scenario_kind = (int)(snw_rng_next(&rng) % 9u);
    group->home_station = (int)(snw_rng_next(&rng) % 3u);
    group->role = (snw_rng_next(&rng) & 1u) ? NPC_ROLE_HAULER : NPC_ROLE_MINER;
    group->persona_risk = snw_rand01(&rng);
    group->persona_growth = snw_rand01(&rng);
    group->persona_patience = snw_rand01(&rng);
    group->faction_id = (int)(snw_rng_next(&rng) % 3u);
    group->faction_goal = (int)(snw_rng_next(&rng) % 4u);
    group->faction_reputation = snw_rand01(&rng);
    group->frontier_need = snw_clip(0.20 + snw_rand01(&rng) * 0.95 +
                                    (group->scenario_kind == 5 ? 0.55 : 0.0),
                                    0.0,
                                    1.5);
    group->expansion_pressure = snw_clip(snw_rand01(&rng) +
                                         (group->faction_goal == 0 ? 0.35 : 0.0) +
                                         (group->scenario_kind == 5 ? 0.35 : 0.0),
                                         0.0,
                                         1.5);
    group->route_pirate_risk = snw_clip(snw_rand01(&rng) +
                                        (group->scenario_kind == 6 ? 0.65 : 0.0) +
                                        (group->scenario_kind == 8 ? 0.45 : 0.0),
                                        0.0,
                                        1.6);
    group->route_recent_attack = snw_clip(snw_rand01(&rng) +
                                          (group->scenario_kind == 6 ? 0.60 : 0.0),
                                          0.0,
                                          1.5);
    group->escort_available = snw_rand01(&rng) > 0.35 ? 1.0 : 0.0;
    if (group->scenario_kind == 6 || group->scenario_kind == 8) group->escort_available = 1.0;
    group->convoy_bonus = snw_clip(snw_rand01(&rng) +
                                   (group->faction_goal == 1 ? 0.30 : 0.0),
                                   0.0,
                                   1.4);
    group->pirate_bounty = snw_clip(0.10 + snw_rand01(&rng) * 1.15 +
                                    (group->scenario_kind == 7 ? 0.55 : 0.0),
                                    0.0,
                                    1.7);

    int slot = spawn_npc(w, group->home_station, (npc_role_t)group->role);
    if (slot < 0) {
        free(w);
        return 0;
    }
    npc_ship_t *npc = &w->npc_ships[slot];
    ship_t *ship = world_npc_ship_for(w, slot);
    if (ship == NULL) {
        free(w);
        return 0;
    }
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

    station_t *home = &w->stations[group->home_station];
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
        snw_seed_finished(&w->stations[remote], group->desired_commodity, remote_stock);
    }

    if (group->scenario_kind == 0 || group->scenario_kind == 3 ||
        group->scenario_kind == 5 || group->scenario_kind == 8) {
        snw_seed_finished(home, COMMODITY_FERRITE_INGOT, 20);
        snw_make_contract(&w->contracts[0], CONTRACT_TRACTOR,
                          (group->home_station + 1) % 3,
                          -1,
                          COMMODITY_FERRITE_INGOT,
                          8.0f,
                          group->scenario_kind == 3 ? 70.0f :
                          group->scenario_kind == 8 ? 95.0f : 25.0f);
    }
    if (group->scenario_kind == 3 || group->scenario_kind == 5 || group->scenario_kind == 8) {
        snw_seed_finished(home, COMMODITY_CUPRITE_INGOT, 12);
        snw_make_contract(&w->contracts[1], CONTRACT_TRACTOR,
                          (group->home_station + 2) % 3,
                          -1,
                          COMMODITY_CUPRITE_INGOT,
                          5.0f,
                          group->scenario_kind == 8 ? 135.0f : 90.0f);
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
    double best_contract_value = snw_best_contract_value(w,
                                                         group->home_station,
                                                         &best_contract_commodity,
                                                         &best_contract_dest,
                                                         &best_contract_stock);
    int remote_refit_stock = 0;
    int remote_refit_station = snw_best_remote_source(w,
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
        route_km = v2_len(v2_sub(w->stations[remote_refit_station].pos, home->pos)) / 1000.0;
    }
    int contract_at_own_station = best_contract_dest == group->home_station;
    int contract_at_faction_station = best_contract_dest >= 0 &&
                                      (best_contract_dest % 3) == group->faction_id;
    double station_credit_weight = contract_at_own_station ? 1.25 :
                                   contract_at_faction_station ? 1.0 : 0.45;
    group->contract_at_own_station = contract_at_own_station;
    group->contract_at_faction_station = contract_at_faction_station;
    group->station_credit_weight = station_credit_weight;
    double pirate_pressure = snw_clip(group->route_pirate_risk * 0.65 +
                                      group->route_recent_attack * 0.35,
                                      0.0,
                                      1.7);
    double faction_alignment = 0.45 + group->faction_reputation * 0.35 +
                               group->expansion_pressure * 0.25;
    double frontier_value = group->frontier_need * (1.4 + group->expansion_pressure) +
                            (group->faction_goal == 0 ? 0.85 : 0.0);
    double pirate_defense_value = pirate_pressure * (1.3 + group->pirate_bounty) +
                                  (group->faction_goal == 1 ? 0.65 : 0.0);
    double station_aligned_contract_value = best_contract_value * station_credit_weight;
    double risky_profit_value = best_contract_value / 80.0 +
                                group->convoy_bonus * 0.70 -
                                pirate_pressure * (1.05 + group->persona_risk * 0.75);

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
            out->faction_delta = out->legal && contract_at_faction_station ? 0.18 : 0.0;
            out->utility = out->legal
                ? (station_aligned_contract_value / 90.0) +
                  (npc->role == NPC_ROLE_HAULER ? 0.45 : 0.0) -
                  pirate_pressure * (0.65 + group->persona_risk * 0.55)
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
                ? 4.8 + group->persona_growth * 1.6 - route_km * 0.08 -
                  pirate_pressure * (0.30 + group->persona_risk * 0.35)
                : -0.65;
            break;
        }
        case SNW_OPT_SUPPLY_FRONTIER:
            out->legal = best_contract_stock > 0 || mine_pressure || home_refit_stock > 0;
            out->travel = out->legal;
            out->faction_work = out->legal;
            out->cargo_moved = out->legal ? snw_clip(4.0 + group->frontier_need * 8.0, 1.0, 14.0) : 0.0;
            out->frontier_delta = out->legal ? group->frontier_need * 0.70 : 0.0;
            out->faction_delta = out->legal ? 0.20 + group->expansion_pressure * 0.35 : 0.0;
            out->credit_delta = out->legal ? 65.0 + group->frontier_need * 80.0 : 0.0;
            out->utility = out->legal
                ? frontier_value + faction_alignment - pirate_pressure * 0.45
                : -0.80;
            break;
        case SNW_OPT_ESCORT_CONVOY:
            out->legal = group->escort_available > 0.5 &&
                         (pirate_pressure > 0.35 || best_contract_value > 0.0);
            out->travel = out->legal;
            out->faction_work = out->legal;
            out->pirate_work = out->legal;
            out->pirate_delta = out->legal ? -0.35 * pirate_pressure : 0.0;
            out->faction_delta = out->legal ? 0.15 + group->convoy_bonus * 0.25 : 0.0;
            out->credit_delta = out->legal ? 80.0 + group->convoy_bonus * 95.0 : 0.0;
            out->utility = out->legal
                ? pirate_defense_value + group->convoy_bonus * 0.80 +
                  (npc->role == NPC_ROLE_HAULER ? 0.25 : 0.0) -
                  group->persona_risk * 0.25
                : -0.70;
            break;
        case SNW_OPT_PATROL_ROUTE:
            out->legal = pirate_pressure > 0.45;
            out->travel = out->legal;
            out->faction_work = out->legal;
            out->pirate_work = out->legal;
            out->pirate_delta = out->legal ? -0.55 * pirate_pressure : 0.0;
            out->faction_delta = out->legal ? 0.18 + group->pirate_bounty * 0.20 : 0.0;
            out->credit_delta = out->legal ? 45.0 + group->pirate_bounty * 130.0 : 0.0;
            out->utility = out->legal
                ? pirate_defense_value + group->pirate_bounty * 0.65 +
                  (npc->role == NPC_ROLE_MINER ? 0.20 : 0.0) -
                  group->persona_patience * 0.15
                : -0.75;
            break;
        case SNW_OPT_RISKY_PROFIT:
            out->legal = best_contract_value > 0.0 && pirate_pressure > 0.20;
            out->travel = out->legal;
            out->contract_value = out->legal ? best_contract_value * (1.0 + group->convoy_bonus * 0.25) : 0.0;
            out->credit_delta = out->contract_value;
            out->cargo_moved = out->legal ? snw_clip(best_contract_value / 45.0, 1.0, 14.0) : 0.0;
            out->faction_delta = out->legal
                ? (contract_at_own_station || contract_at_faction_station ? 0.05 : -0.18) -
                  pirate_pressure * 0.08
                : 0.0;
            out->pirate_delta = out->legal ? pirate_pressure * 0.12 : 0.0;
            out->utility = out->legal
                ? risky_profit_value +
                  (contract_at_own_station || contract_at_faction_station ? 0.35 : -0.35) +
                  (1.0 - group->persona_risk) * 0.70
                : -0.85;
            break;
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
                      pirate_pressure > 0.65 ? SNW_OPT_PATROL_ROUTE :
                      group->frontier_need > 1.0 ? SNW_OPT_SUPPLY_FRONTIER :
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
        f[14] = npc->role == NPC_ROLE_MINER ? 1.0f : 0.0f;
        f[15] = npc->role == NPC_ROLE_HAULER ? 1.0f : 0.0f;
        f[16] = (nn_real)ship->mining_level / 5.0f;
        f[17] = (nn_real)ship->hold_level / 5.0f;
        f[18] = (nn_real)ship->tractor_level / 5.0f;
        f[19] = (nn_real)group->home_station / 2.0f;
        f[20] = (nn_real)snw_clip(group->home_balance / 1500.0, 0.0, 1.5);
        f[21] = (nn_real)home_refit_stock / 24.0f;
        f[22] = (nn_real)remote_refit_stock / 24.0f;
        f[23] = (nn_real)group->desired_units / 20.0f;
        f[24] = (nn_real)snw_clip(refit_cost / 1500.0, 0.0, 2.0);
        f[25] = (nn_real)can_pay_refit;
        f[26] = group->desired_upgrade == SHIP_UPGRADE_MINING ? 1.0f : 0.0f;
        f[27] = group->desired_upgrade == SHIP_UPGRADE_HOLD ? 1.0f : 0.0f;
        f[28] = group->desired_upgrade == SHIP_UPGRADE_TRACTOR ? 1.0f : 0.0f;
        f[29] = group->desired_commodity == COMMODITY_FRAME ? 1.0f : 0.0f;
        f[30] = group->desired_commodity == COMMODITY_LASER_MODULE ? 1.0f : 0.0f;
        f[31] = group->desired_commodity == COMMODITY_TRACTOR_MODULE ? 1.0f : 0.0f;
        f[32] = (nn_real)snw_clip(best_contract_value / 600.0, 0.0, 2.0);
        f[33] = (nn_real)best_contract_stock / 24.0f;
        f[34] = best_contract_dest >= 0 && best_contract_dest != group->home_station ? 1.0f : 0.0f;
        f[35] = mine_pressure ? 1.0f : 0.0f;
        f[36] = (nn_real)group->persona_risk;
        f[37] = (nn_real)group->persona_growth;
        f[38] = (nn_real)group->persona_patience;
        f[39] = (nn_real)snw_clip(route_km / 20.0, 0.0, 1.0);
        f[40] = station_has_module(home, MODULE_DOCK) ? 1.0f : 0.0f;
        f[41] = station_has_module(home, MODULE_SHIPYARD) ? 1.0f : 0.0f;
        f[42] = station_has_module(home, MODULE_FURNACE) ? 1.0f : 0.0f;
        f[43] = station_has_module(home, MODULE_FRAME_PRESS) ? 1.0f : 0.0f;
        f[44] = station_has_module(home, MODULE_LASER_FAB) ? 1.0f : 0.0f;
        f[45] = station_has_module(home, MODULE_TRACTOR_FAB) ? 1.0f : 0.0f;
        f[46] = h->option == SNW_OPT_WAIT ? 1.0f : 0.0f;
        f[47] = h->option == SNW_OPT_SELF_REFIT_HOME ? (nn_real)(home_refit_stock >= group->desired_units) : 0.0f;
        f[48] = (h->option == SNW_OPT_IMPORT_FRAME ||
                 h->option == SNW_OPT_IMPORT_LASER ||
                 h->option == SNW_OPT_IMPORT_TRACTOR) ? 1.0f : 0.0f;
        f[49] = group->outcomes[oi].legal ? 1.0f : 0.0f;
        f[50] = group->outcomes[oi].travel ? 1.0f : 0.0f;
        f[51] = group->outcomes[oi].self_upgrade ? 1.0f : 0.0f;
        f[52] = group->outcomes[oi].import_module ? 1.0f : 0.0f;
        f[53] = (nn_real)snw_clip(group->outcomes[oi].credit_delta / 1000.0, -1.5, 2.0);
        f[54] = (nn_real)group->outcomes[oi].refit_progress;
        f[55] = (nn_real)snw_clip(group->outcomes[oi].contract_value / 600.0, 0.0, 2.0);
        f[56] = (nn_real)snw_clip(group->outcomes[oi].cargo_moved / 16.0, 0.0, 1.0);
        f[57] = (nn_real)group->faction_id / 2.0f;
        f[58] = (nn_real)group->faction_goal / 3.0f;
        f[59] = (nn_real)group->faction_reputation;
        f[60] = (nn_real)snw_clip(group->frontier_need / 1.5, 0.0, 1.0);
        f[61] = (nn_real)snw_clip(group->expansion_pressure / 1.5, 0.0, 1.0);
        f[62] = (nn_real)snw_clip(group->route_pirate_risk / 1.6, 0.0, 1.0);
        f[63] = (nn_real)snw_clip(group->route_recent_attack / 1.5, 0.0, 1.0);
        f[64] = (nn_real)group->escort_available;
        f[65] = (nn_real)snw_clip(group->convoy_bonus / 1.4, 0.0, 1.0);
        f[66] = (nn_real)snw_clip(group->pirate_bounty / 1.7, 0.0, 1.0);
        f[67] = (nn_real)snw_clip(pirate_pressure / 1.7, 0.0, 1.0);
        f[68] = group->outcomes[oi].faction_work ? 1.0f : 0.0f;
        f[69] = group->outcomes[oi].pirate_work ? 1.0f : 0.0f;
        f[70] = (nn_real)snw_clip(group->outcomes[oi].faction_delta, -1.0, 1.0);
        f[71] = (nn_real)snw_clip(group->outcomes[oi].frontier_delta, 0.0, 1.0);
        f[72] = (nn_real)snw_clip(group->outcomes[oi].pirate_delta, -1.0, 1.0);
        f[73] = best_contract_commodity == COMMODITY_FERRITE_INGOT ? 1.0f : 0.0f;
        f[74] = best_contract_commodity == COMMODITY_CUPRITE_INGOT ? 1.0f : 0.0f;
        f[75] = best_contract_commodity == COMMODITY_CRYSTAL_INGOT ? 1.0f : 0.0f;
        f[76] = contract_at_own_station ? 1.0f : 0.0f;
        f[77] = contract_at_faction_station ? 1.0f : 0.0f;
    }
    free(w);
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

static void snw_json_string(FILE *fp, const char *s);

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

static void snw_update_worker_memory(snw_worker_memory_t *memory,
                                     const snw_group_t *group,
                                     size_t selected,
                                     double regret)
{
    if (memory == NULL || group == NULL || selected >= SNW_OPTION_COUNT) return;
    const snw_outcome_t *out = &group->outcomes[selected];
    memory->route_risk *= 0.88;
    memory->job_profit *= 0.86;
    memory->refit_need *= 0.82;
    if (out->travel && (regret > 0.25 || out->utility < 0.5)) {
        memory->route_risk = snw_clip(memory->route_risk + 0.35 + regret * 0.08, 0.0, 1.5);
    }
    if (out->pirate_work && out->pirate_delta < 0.0) {
        memory->route_risk = snw_clip(memory->route_risk * 0.70, 0.0, 1.5);
    }
    if (out->pirate_work && out->pirate_delta > 0.0) {
        memory->route_risk = snw_clip(memory->route_risk + out->pirate_delta * 0.80, 0.0, 1.5);
    }
    if (out->contract_value > 0.0) {
        memory->job_profit = snw_clip(memory->job_profit + out->contract_value / 900.0, 0.0, 1.5);
    }
    if (out->frontier_delta > 0.0 || out->faction_delta > 0.0) {
        memory->refit_need = snw_clip(memory->refit_need + 0.15 + out->frontier_delta * 0.30,
                                      0.0,
                                      1.5);
    }
    if (group->desired_upgrade < SHIP_UPGRADE_COUNT &&
        (selected == SNW_OPT_SELF_REFIT_HOME ||
         selected == SNW_OPT_IMPORT_FRAME ||
         selected == SNW_OPT_IMPORT_LASER ||
         selected == SNW_OPT_IMPORT_TRACTOR)) {
        memory->refit_need = snw_clip(memory->refit_need + 0.35, 0.0, 1.5);
    }
    memory->has_memory = 1;
}

static void snw_publish_gossip(snw_worker_memory_t memories[SNW_PERSONA_COUNT],
                               size_t source,
                               const snw_group_t *group,
                               size_t selected,
                               double regret)
{
    if (memories == NULL || group == NULL || selected >= SNW_OPTION_COUNT) return;
    const snw_outcome_t *out = &group->outcomes[selected];
    for (size_t i = 0; i < SNW_PERSONA_COUNT; ++i) {
        if (i == source) continue;
        if (out->contract_value > 0.0) {
            memories[i].gossip_contract =
                snw_clip(memories[i].gossip_contract * 0.90 + out->contract_value / 1000.0,
                         0.0,
                         1.5);
            memories[i].heard_gossip = 1;
        }
        if (out->travel && regret > 0.25) {
            memories[i].gossip_route_risk =
                snw_clip(memories[i].gossip_route_risk * 0.90 + 0.25 + regret * 0.05,
                         0.0,
                         1.5);
            memories[i].heard_gossip = 1;
        }
        if (out->pirate_work) {
            double delta = out->pirate_delta < 0.0 ? 0.18 : 0.35;
            memories[i].gossip_route_risk =
                snw_clip(memories[i].gossip_route_risk * 0.85 + delta,
                         0.0,
                         1.5);
            memories[i].heard_gossip = 1;
        }
        if (out->frontier_delta > 0.0 || out->faction_delta > 0.0) {
            memories[i].refit_need = snw_clip(memories[i].refit_need + 0.12, 0.0, 1.5);
            memories[i].heard_gossip = 1;
        }
        if (out->import_module || out->self_upgrade) {
            memories[i].refit_need = snw_clip(memories[i].refit_need + 0.10, 0.0, 1.5);
            memories[i].heard_gossip = 1;
        }
    }
}

static int snw_append_social_slice_trace(const char *path,
                                         NeuralNet *net,
                                         const snw_dataset_t *test,
                                         snw_metrics_t *metrics)
{
    if (path == NULL || net == NULL || test == NULL || metrics == NULL) return 1;
    FILE *fp = fopen(path, "a");
    if (fp == NULL) {
        fprintf(stderr, "failed to append Signal NPC worker social slice trace: %s\n", path);
        return 0;
    }

    snw_worker_memory_t memories[SNW_PERSONA_COUNT];
    memset(memories, 0, sizeof(memories));
    nn_real scores[SNW_OPTION_COUNT];
    nn_real probabilities[SNW_OPTION_COUNT];
    size_t ranked[SNW_OPTION_COUNT];

    for (size_t gi = 0; gi < test->count; ++gi) {
        const snw_group_t *group = &test->groups[gi];
        if (!nn_score_action_rows(net,
                                  &group->features[0][0],
                                  SNW_OPTION_COUNT,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, SNW_OPTION_COUNT, ranked)) {
            continue;
        }

        size_t worker = snw_group_worker(group);
        const snw_persona_t *persona = snw_persona(worker);
        size_t base = ranked[0];
        size_t selected = base;
        double best_adjusted = -1e300;
        int selected_memory_used = 0;
        int selected_gossip_used = 0;
        for (size_t oi = 0; oi < SNW_OPTION_COUNT; ++oi) {
            int memory_used = 0;
            int gossip_used = 0;
            double bonus = snw_social_bonus(group,
                                            &memories[worker],
                                            persona,
                                            oi,
                                            &memory_used,
                                            &gossip_used);
            double adjusted = (double)scores[oi] + bonus;
            if (!group->outcomes[oi].legal) adjusted -= 1000.0;
            if (adjusted > best_adjusted) {
                best_adjusted = adjusted;
                selected = oi;
                selected_memory_used = memory_used;
                selected_gossip_used = gossip_used;
            }
        }

        const snw_outcome_t *sel = &group->outcomes[selected];
        const snw_outcome_t *oracle = &group->outcomes[group->target];
        double regret = oracle->utility - sel->utility;
        int memory_influenced = selected != base && selected_memory_used;
        int gossip_influenced = selected != base && selected_gossip_used;
        int persona_divergent = selected != base;
        const char *intent = snw_intent_for(selected, group);
        const char *memory = snw_memory_for(selected, sel);
        const char *gossip = snw_gossip_for(selected, sel);

        ++metrics->social_groups;
        ++metrics->social_intent_rows;
        ++metrics->social_memory_updates;
        if (strcmp(gossip, "no high-value gossip published") != 0) ++metrics->social_gossip_facts;
        if (sel->legal) ++metrics->social_selected_legal;
        if (memory_influenced) ++metrics->social_memory_influenced;
        if (gossip_influenced) ++metrics->social_gossip_influenced;
        if (persona_divergent) ++metrics->social_persona_divergent;
        metrics->social_selected_utility += sel->utility;
        metrics->social_oracle_utility += oracle->utility;
        metrics->social_regret += regret;

        fprintf(fp, "{\"schema\":");
        snw_json_string(fp, SNW_SOCIAL_SLICE_SCHEMA);
        fprintf(fp, ",\"task\":");
        snw_json_string(fp, SNW_TASK_NAME);
        fprintf(fp, ",\"group_id\":%" PRIu64, group->group_id);
        fprintf(fp, ",\"worker_index\":%zu", worker);
        fprintf(fp, ",\"persona\":");
        snw_json_string(fp, persona->name);
        fprintf(fp, ",\"base_option\":");
        snw_json_string(fp, snw_hypothesis(base)->name);
        fprintf(fp, ",\"selected_option\":");
        snw_json_string(fp, snw_hypothesis(selected)->name);
        fprintf(fp, ",\"target_option\":");
        snw_json_string(fp, snw_hypothesis(group->target)->name);
        fprintf(fp, ",\"faction_id\":%d", group->faction_id);
        fprintf(fp, ",\"faction_goal\":%d", group->faction_goal);
        fprintf(fp, ",\"frontier_need\":%.9g", group->frontier_need);
        fprintf(fp, ",\"route_pirate_risk\":%.9g", group->route_pirate_risk);
        fprintf(fp, ",\"route_recent_attack\":%.9g", group->route_recent_attack);
        fprintf(fp, ",\"escort_available\":%.9g", group->escort_available);
        fprintf(fp, ",\"intent\":");
        snw_json_string(fp, intent);
        fprintf(fp, ",\"memory_written\":");
        snw_json_string(fp, memory);
        fprintf(fp, ",\"gossip_published\":");
        snw_json_string(fp, gossip);
        fprintf(fp, ",\"memory_influenced\":%s", memory_influenced ? "true" : "false");
        fprintf(fp, ",\"gossip_influenced\":%s", gossip_influenced ? "true" : "false");
        fprintf(fp, ",\"persona_divergent\":%s", persona_divergent ? "true" : "false");
        fprintf(fp, ",\"selected_utility\":%.9g", sel->utility);
        fprintf(fp, ",\"oracle_utility\":%.9g", oracle->utility);
        fprintf(fp, ",\"regret\":%.9g", regret);
        fprintf(fp, ",\"worker_memory\":{\"route_risk\":%.9g,\"job_profit\":%.9g,"
                    "\"refit_need\":%.9g,\"gossip_contract\":%.9g,"
                    "\"gossip_route_risk\":%.9g}}\n",
                memories[worker].route_risk,
                memories[worker].job_profit,
                memories[worker].refit_need,
                memories[worker].gossip_contract,
                memories[worker].gossip_route_risk);

        snw_update_worker_memory(&memories[worker], group, selected, regret);
        snw_publish_gossip(memories, worker, group, selected, regret);
    }

    double denom = metrics->social_groups == 0u ? 1.0 : (double)metrics->social_groups;
    metrics->social_selected_utility /= denom;
    metrics->social_oracle_utility /= denom;
    metrics->social_regret /= denom;
    fclose(fp);
    return 1;
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
                fprintf(fp, ",\"faction_work\":%s", g->outcomes[oi].faction_work ? "true" : "false");
                fprintf(fp, ",\"pirate_work\":%s", g->outcomes[oi].pirate_work ? "true" : "false");
                fprintf(fp, ",\"faction_delta\":%.9g", g->outcomes[oi].faction_delta);
                fprintf(fp, ",\"frontier_delta\":%.9g", g->outcomes[oi].frontier_delta);
                fprintf(fp, ",\"pirate_delta\":%.9g", g->outcomes[oi].pirate_delta);
                fprintf(fp, ",\"home_station\":%d", g->home_station);
                fprintf(fp, ",\"role\":%d", g->role);
                fprintf(fp, ",\"desired_upgrade\":%d", (int)g->desired_upgrade);
                fprintf(fp, ",\"scenario_kind\":%d", g->scenario_kind);
                fprintf(fp, ",\"faction_id\":%d", g->faction_id);
                fprintf(fp, ",\"faction_goal\":%d", g->faction_goal);
                fprintf(fp, ",\"faction_reputation\":%.9g", g->faction_reputation);
                fprintf(fp, ",\"frontier_need\":%.9g", g->frontier_need);
                fprintf(fp, ",\"expansion_pressure\":%.9g", g->expansion_pressure);
                fprintf(fp, ",\"route_pirate_risk\":%.9g", g->route_pirate_risk);
                fprintf(fp, ",\"route_recent_attack\":%.9g", g->route_recent_attack);
                fprintf(fp, ",\"escort_available\":%.9g", g->escort_available);
                fprintf(fp, ",\"convoy_bonus\":%.9g", g->convoy_bonus);
                fprintf(fp, ",\"pirate_bounty\":%.9g", g->pirate_bounty);
                fprintf(fp, ",\"contract_at_own_station\":%s",
                        g->contract_at_own_station ? "true" : "false");
                fprintf(fp, ",\"contract_at_faction_station\":%s",
                        g->contract_at_faction_station ? "true" : "false");
                fprintf(fp, ",\"station_credit_weight\":%.9g", g->station_credit_weight);
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
        !nn_checkpoint_meta_set_notes(&meta, "Signal NPC worker strategic option scorer trained from Signal simulator state, frontier pressure, faction goals, pirate risk, and certified utility rows") ||
        !nn_checkpoint_meta_set_feature_set(&meta, "signal-npc-worker-v2")) {
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
    fprintf(fp, ",\"candidate_layer\":\"npc_worker_strategy_v2_frontier_factions_pirates\"");
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
    fprintf(fp, ",\"social_slice_schema\":");
    snw_json_string(fp, SNW_SOCIAL_SLICE_SCHEMA);
    fprintf(fp, ",\"social_groups\":%zu", metrics->social_groups);
    fprintf(fp, ",\"social_intent_rows\":%zu", metrics->social_intent_rows);
    fprintf(fp, ",\"social_memory_updates\":%zu", metrics->social_memory_updates);
    fprintf(fp, ",\"social_gossip_facts\":%zu", metrics->social_gossip_facts);
    fprintf(fp, ",\"social_selected_legal_rate\":%.9g",
            snw_rate(metrics->social_selected_legal, metrics->social_groups));
    fprintf(fp, ",\"social_memory_influenced\":%zu", metrics->social_memory_influenced);
    fprintf(fp, ",\"social_gossip_influenced\":%zu", metrics->social_gossip_influenced);
    fprintf(fp, ",\"social_persona_divergent\":%zu", metrics->social_persona_divergent);
    fprintf(fp, ",\"social_selected_utility\":%.9g", metrics->social_selected_utility);
    fprintf(fp, ",\"social_oracle_utility\":%.9g", metrics->social_oracle_utility);
    fprintf(fp, ",\"social_regret\":%.9g", metrics->social_regret);
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

typedef struct {
    double faction_credit[SNW_PERSISTENT_FACTIONS];
    double frontier_need[SNW_PERSISTENT_MAX_STATIONS];
    double pirate_risk[SNW_PERSISTENT_MAX_STATIONS];
    double contraband_stock[SNW_PERSISTENT_MAX_STATIONS];
    double contraband_demand[SNW_PERSISTENT_MAX_STATIONS];
    double route_profit_memory[SNW_PERSISTENT_MAX_STATIONS][SNW_PERSISTENT_MAX_STATIONS];
    double route_patrol_heat[SNW_PERSISTENT_MAX_STATIONS][SNW_PERSISTENT_MAX_STATIONS];
    double faction_control[SNW_PERSISTENT_MAX_STATIONS][SNW_PERSISTENT_FACTIONS];
    size_t action_count[SNW_OPTION_COUNT];
    size_t ticks;
    size_t decisions;
    size_t legal_decisions;
    size_t faction_aligned_decisions;
    size_t pirate_events;
    size_t frontier_supply_events;
    size_t pirate_damage_events;
    size_t npc_losses;
    size_t pirate_market_count;
    size_t pirate_npcs;
    size_t lawful_npcs;
    size_t smuggle_runs;
    size_t lawful_risky_profit_runs;
    size_t autonomous_route_choices;
    size_t memory_route_reuses;
    size_t patrol_deterrence_events;
    double utility_sum;
    double social_utility_sum;
    double own_station_credit;
    double faction_station_credit;
    double off_faction_credit;
    double pirate_station_credit;
    double lawful_station_credit;
    double contraband_credit;
    double contraband_volume;
    double lawful_supply_leakage;
    double pirate_damage;
    double pirate_pressure_sum;
    double frontier_need_sum;
    double contraband_demand_sum;
    double route_profit_memory_sum;
    double route_patrol_heat_sum;
} snw_persistent_metrics_t;

static void snw_persistent_seed_station(world_t *w, size_t idx, uint32_t seed)
{
    if (w == NULL || idx >= MAX_STATIONS) return;
    station_t *st = &w->stations[idx];
    if (!station_exists(st)) {
        st->id = w->next_station_id++;
        snprintf(st->name, sizeof(st->name), "Frontier %zu", idx);
        snprintf(st->station_slug, sizeof(st->station_slug), "frontier-%zu", idx);
        st->pos = v2(cosf((float)idx * 0.73f) * (2600.0f + (float)(idx % 5u) * 520.0f),
                    sinf((float)idx * 0.73f) * (2600.0f + (float)(idx % 7u) * 430.0f));
        st->radius = 36.0f;
        st->dock_radius = 220.0f;
        st->signal_range = 6500.0f;
        st->signal_connected = true;
        st->base_price[COMMODITY_FERRITE_INGOT] = 12.0f + (float)(idx % 3u) * 3.0f;
        st->base_price[COMMODITY_CUPRITE_INGOT] = 18.0f + (float)(idx % 4u) * 4.0f;
        st->base_price[COMMODITY_FRAME] = 8.0f + (float)(idx % 5u);
        st->base_price[COMMODITY_LASER_MODULE] = 18.0f + (float)(idx % 3u) * 2.0f;
        st->base_price[COMMODITY_TRACTOR_MODULE] = 20.0f + (float)(idx % 3u) * 2.0f;
        rebuild_station_services(st);
        station_authority_init_outpost(st, NULL, (uint64_t)seed + idx);
    }
    snw_seed_finished(st, COMMODITY_FERRITE_INGOT, 6 + (int)(idx % 9u));
    snw_seed_finished(st, COMMODITY_CUPRITE_INGOT, 3 + (int)(idx % 7u));
    snw_seed_finished(st, COMMODITY_FRAME, 2 + (int)(idx % 5u));
    snw_seed_finished(st, COMMODITY_LASER_MODULE, 2 + (int)(idx % 4u));
    snw_seed_finished(st, COMMODITY_TRACTOR_MODULE, 2 + (int)(idx % 4u));
}

static double snw_station_credit_weight(int home_station, int dest_station, int faction)
{
    if (dest_station == home_station) return 1.25;
    if (dest_station >= 0 && (dest_station % (int)SNW_PERSISTENT_FACTIONS) == faction) return 1.0;
    return 0.35;
}

static int snw_persistent_is_pirate_market(size_t station_idx, size_t station_count)
{
    if (station_count < 8u) return station_idx + 1u == station_count;
    return station_idx % 7u == 6u;
}

static int snw_persistent_choose_destination(const snw_persistent_metrics_t *metrics,
                                             size_t station_count,
                                             int home,
                                             int home_is_pirate,
                                             int default_dest,
                                             uint32_t *rng,
                                             int *used_memory,
                                             int *patrol_deterrence)
{
    if (used_memory != NULL) *used_memory = 0;
    if (patrol_deterrence != NULL) *patrol_deterrence = 0;
    if (metrics == NULL || station_count <= 1u || home < 0 || home >= (int)station_count)
        return default_dest;

    double best_score = -1e300;
    int best_dest = default_dest;
    for (size_t tries = 0; tries < station_count; ++tries) {
        int dest = (int)tries;
        if (dest == home) continue;
        double jitter = (double)(snw_rng_next(rng) % 1000u) / 10000.0;
        double patrol_heat = metrics->route_patrol_heat[home][dest];
        double profit_memory = metrics->route_profit_memory[home][dest];
        double score = metrics->frontier_need[dest] * 0.35 +
                       metrics->pirate_risk[dest] * 0.20 +
                       jitter;
        if (home_is_pirate) {
            score += metrics->contraband_stock[dest] * 0.55 +
                     metrics->contraband_demand[home] * 0.75 +
                     profit_memory * 0.65 -
                     patrol_heat * 1.15;
            if (snw_persistent_is_pirate_market((size_t)dest, station_count)) score += 0.15;
        } else {
            score += metrics->pirate_risk[dest] * 0.65 +
                     metrics->contraband_stock[dest] * 0.18 -
                     profit_memory * 0.10 +
                     patrol_heat * 0.20;
            if (snw_persistent_is_pirate_market((size_t)dest, station_count)) score += 0.35;
        }
        if (score > best_score) {
            best_score = score;
            best_dest = dest;
        }
    }
    if (used_memory != NULL && metrics->route_profit_memory[home][best_dest] > 0.35) {
        *used_memory = 1;
    }
    if (patrol_deterrence != NULL &&
        home_is_pirate &&
        default_dest >= 0 &&
        default_dest < (int)station_count &&
        metrics->route_patrol_heat[home][default_dest] >
            metrics->route_patrol_heat[home][best_dest] + 0.20) {
        *patrol_deterrence = 1;
    }
    return best_dest;
}

static void snw_persistent_write_decision(FILE *fp,
                                          uint64_t tick,
                                          int npc_slot,
                                          int home_station,
                                          int faction,
                                          const snw_group_t *group,
                                          size_t selected,
                                          double utility,
                                          double social_utility,
                                          double pirate_damage,
                                          double hull_after,
                                          int home_is_pirate,
                                          int dest_is_pirate,
                                          double contraband_delta,
                                          double contraband_demand,
                                          double route_profit_memory,
                                          double route_patrol_heat,
                                          int autonomous_route,
                                          int memory_route,
                                          int patrol_deterrence)
{
    fprintf(fp, "{\"schema\":");
    snw_json_string(fp, SNW_PERSISTENT_SCHEMA);
    fprintf(fp, ",\"tick\":%" PRIu64, tick);
    fprintf(fp, ",\"npc_slot\":%d", npc_slot);
    fprintf(fp, ",\"home_station\":%d", home_station);
    fprintf(fp, ",\"faction\":%d", faction);
    fprintf(fp, ",\"selected_option\":");
    snw_json_string(fp, snw_hypothesis(selected)->name);
    fprintf(fp, ",\"oracle_option\":");
    snw_json_string(fp, snw_hypothesis(group->target)->name);
    fprintf(fp, ",\"utility\":%.9g", utility);
    fprintf(fp, ",\"social_utility\":%.9g", social_utility);
    fprintf(fp, ",\"pirate_damage\":%.9g", pirate_damage);
    fprintf(fp, ",\"npc_hull_after\":%.9g", hull_after);
    fprintf(fp, ",\"home_is_pirate_market\":%s", home_is_pirate ? "true" : "false");
    fprintf(fp, ",\"dest_is_pirate_market\":%s", dest_is_pirate ? "true" : "false");
    fprintf(fp, ",\"contraband_delta\":%.9g", contraband_delta);
    fprintf(fp, ",\"contraband_demand\":%.9g", contraband_demand);
    fprintf(fp, ",\"route_profit_memory\":%.9g", route_profit_memory);
    fprintf(fp, ",\"route_patrol_heat\":%.9g", route_patrol_heat);
    fprintf(fp, ",\"autonomous_route\":%s", autonomous_route ? "true" : "false");
    fprintf(fp, ",\"memory_route\":%s", memory_route ? "true" : "false");
    fprintf(fp, ",\"patrol_deterrence\":%s", patrol_deterrence ? "true" : "false");
    fprintf(fp, ",\"frontier_need\":%.9g", group->frontier_need);
    fprintf(fp, ",\"route_pirate_risk\":%.9g", group->route_pirate_risk);
    fprintf(fp, ",\"contract_at_own_station\":%s", group->contract_at_own_station ? "true" : "false");
    fprintf(fp, ",\"contract_at_faction_station\":%s", group->contract_at_faction_station ? "true" : "false");
    fputs("}\n", fp);
}

static int snw_persistent_run(const snw_config_t *config)
{
    if (config == NULL) return 0;
    size_t station_count = config->persistent_stations;
    if (station_count < SIGNAL_SEEDED_STATION_COUNT) station_count = SIGNAL_SEEDED_STATION_COUNT;
    if (station_count > SNW_PERSISTENT_MAX_STATIONS) station_count = SNW_PERSISTENT_MAX_STATIONS;
    if (station_count > MAX_STATIONS) station_count = MAX_STATIONS;
    size_t npc_count = config->persistent_npcs;
    if (npc_count > MAX_NPC_SHIPS) npc_count = MAX_NPC_SHIPS;
    if (npc_count == 0u) npc_count = 1u;

    world_t *w = calloc(1u, sizeof(*w));
    if (w == NULL) return 0;
    w->rng = config->seed;
    world_reset(w);
    snw_clear_contracts(w);
    snw_clear_npcs(w);
    for (size_t s = 0; s < station_count; ++s) snw_persistent_seed_station(w, s, config->seed);

    snw_persistent_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));

    int npc_slots[MAX_NPC_SHIPS];
    memset(npc_slots, -1, sizeof(npc_slots));
    size_t spawned = 0;
    for (size_t i = 0; i < npc_count; ++i) {
        int home = (int)(i % station_count);
        int role = (i & 1u) ? NPC_ROLE_HAULER : NPC_ROLE_MINER;
        int slot = spawn_npc(w, home, (npc_role_t)role);
        if (slot < 0) continue;
        npc_ship_t *npc = &w->npc_ships[slot];
        ship_t *ship = world_npc_ship_for(w, slot);
        if (ship == NULL) continue;
        npc->state = NPC_STATE_DOCKED;
        npc->home_station = home;
        npc->ship.mining_level = ship->mining_level = (int)(1u + (i % 3u));
        npc->ship.hold_level = ship->hold_level = (int)(1u + ((i / 3u) % 3u));
        npc->ship.tractor_level = ship->tractor_level = (int)(1u + ((i / 7u) % 3u));
        ledger_earn(&w->stations[home], npc->session_token, 250.0f + (float)(i % 9u) * 45.0f);
        if (snw_persistent_is_pirate_market((size_t)home, station_count)) metrics.pirate_npcs++;
        else metrics.lawful_npcs++;
        npc_slots[spawned++] = slot;
    }
    if (spawned == 0u) {
        free(w);
        return 0;
    }
    frontier_virtual_pilots_set(w, (int)snw_clip((double)spawned * 12.0, 0.0, 5000.0));

    for (size_t s = 0; s < station_count; ++s) {
        metrics.frontier_need[s] = 0.35 + (double)(s % 7u) * 0.11;
        metrics.pirate_risk[s] = 0.10 + (double)(s % 5u) * 0.13;
        if (snw_persistent_is_pirate_market(s, station_count)) {
            metrics.pirate_market_count++;
            metrics.contraband_stock[s] = 4.0 + (double)(s % 5u);
            metrics.contraband_demand[s] = 0.85 + (double)(s % 4u) * 0.18;
            metrics.pirate_risk[s] += 0.25;
        } else {
            metrics.contraband_stock[s] = (double)(s % 3u);
            metrics.contraband_demand[s] = 0.08 + (double)(s % 5u) * 0.03;
        }
        metrics.faction_control[s][s % SNW_PERSISTENT_FACTIONS] = 1.0;
    }

    FILE *trace = config->trace_path != NULL ? fopen(config->trace_path, "w") : NULL;
    if (config->trace_path != NULL && trace == NULL) {
        free(w);
        return 0;
    }

    uint32_t rng = config->seed ^ UINT32_C(0x70657273);
    for (size_t tick = 0; tick < config->persistent_ticks; ++tick) {
        size_t who = (size_t)(snw_rng_next(&rng) % (uint32_t)spawned);
        int slot = npc_slots[who];
        npc_ship_t *npc = slot >= 0 ? &w->npc_ships[slot] : NULL;
        ship_t *ship = slot >= 0 ? world_npc_ship_for(w, slot) : NULL;
        if (npc == NULL || ship == NULL || !npc->active) {
            world_sim_step(w, 1.0f / 30.0f);
            continue;
        }
        int active_before = npc->active ? 1 : 0;
        float hull_before = ship->hull;

        int home = npc->home_station >= 0 && npc->home_station < (int)station_count
            ? npc->home_station
            : (int)(who % station_count);
        int faction = home % (int)SNW_PERSISTENT_FACTIONS;
        int home_is_pirate = snw_persistent_is_pirate_market((size_t)home, station_count);
        int default_dest = (home + 1 + (int)(snw_rng_next(&rng) % (uint32_t)(station_count - 1u))) %
                           (int)station_count;
        int memory_route = 0;
        int patrol_deterrence = 0;
        int dest = snw_persistent_choose_destination(&metrics,
                                                     station_count,
                                                     home,
                                                     home_is_pirate,
                                                     default_dest,
                                                     &rng,
                                                     &memory_route,
                                                     &patrol_deterrence);
        int autonomous_route = dest != default_dest;
        int dest_is_pirate = snw_persistent_is_pirate_market((size_t)dest, station_count);
        metrics.route_profit_memory[home][dest] *= 0.995;
        metrics.route_patrol_heat[home][dest] *= 0.985;
        station_t *home_st = &w->stations[home];
        station_t *dest_st = &w->stations[dest];
        commodity_t contract_commodity = (tick + who) & 1u ? COMMODITY_CUPRITE_INGOT : COMMODITY_FERRITE_INGOT;
        int stock = station_finished_count(home_st, contract_commodity);
        double base_price = 35.0 + metrics.frontier_need[dest] * 60.0 + metrics.pirate_risk[dest] * 80.0;
        double contract_value = stock > 0 ? base_price * snw_clip((double)stock, 1.0, 8.0) : 0.0;
        double credit_weight = snw_station_credit_weight(home, dest, faction);
        double pirate = metrics.pirate_risk[dest];
        double frontier = metrics.frontier_need[dest];
        double expansion = metrics.faction_control[dest][faction] < 0.55 ? 1.0 : 0.35;
        double home_contraband_demand = metrics.contraband_demand[home];
        double dest_contraband_stock = metrics.contraband_stock[dest];
        double route_profit_memory = metrics.route_profit_memory[home][dest];
        double route_patrol_heat = metrics.route_patrol_heat[home][dest];
        double contraband_price = 70.0 + home_contraband_demand * 95.0 + pirate * 30.0;

        snw_group_t group;
        memset(&group, 0, sizeof(group));
        group.seed = (uint32_t)(config->seed + tick * 977u + who);
        group.group_id = ((uint64_t)tick << 32) | (uint64_t)who;
        group.scenario_kind = (int)(tick % 9u);
        group.home_station = home;
        group.role = npc->role;
        group.persona_risk = (double)((who * 37u) % 100u) / 100.0;
        group.persona_growth = (double)((who * 53u + tick) % 100u) / 100.0;
        group.persona_patience = (double)((who * 71u + tick * 3u) % 100u) / 100.0;
        group.faction_id = faction;
        group.faction_goal = pirate > 0.65 ? 1 : frontier > 0.85 ? 0 : 2;
        group.faction_reputation = metrics.faction_control[home][faction];
        group.frontier_need = frontier;
        group.expansion_pressure = expansion;
        group.route_pirate_risk = pirate;
        group.route_recent_attack = snw_clip((pirate > 0.7 ? 1.0 : pirate * 0.35) +
                                             route_patrol_heat * (home_is_pirate ? 0.40 : -0.15),
                                             0.0,
                                             1.5);
        group.escort_available = spawned > 1u ? 1.0 : 0.0;
        group.convoy_bonus = 0.20 + metrics.faction_control[home][faction] * 0.60;
        group.pirate_bounty = 0.20 + pirate;
        group.desired_upgrade = snw_preferred_upgrade(npc, ship);
        group.desired_commodity = snw_upgrade_commodity(group.desired_upgrade);
        group.desired_units = group.desired_upgrade < SHIP_UPGRADE_COUNT
            ? (int)ceilf(upgrade_product_cost(ship, group.desired_upgrade))
            : 0;
        group.home_balance = ledger_balance(home_st, npc->session_token);
        group.contract_at_own_station = dest == home;
        group.contract_at_faction_station = (dest % (int)SNW_PERSISTENT_FACTIONS) == faction;
        group.station_credit_weight = credit_weight;

        for (size_t oi = 0; oi < SNW_OPTION_COUNT; ++oi) {
            snw_outcome_t *out = &group.outcomes[oi];
            memset(out, 0, sizeof(*out));
            out->legal = 1;
            switch (snw_hypothesis(oi)->option) {
            case SNW_OPT_WAIT:
                out->utility = 0.02;
                break;
            case SNW_OPT_MINE_HOME:
                out->legal = npc->role == NPC_ROLE_MINER;
                out->credit_delta = 20.0;
                out->utility = out->legal ? 0.40 + frontier * 0.30 : -1.0;
                break;
            case SNW_OPT_HAUL_CONTRACT:
                out->legal = contract_value > 0.0;
                out->travel = out->legal;
                out->contract_value = contract_value;
                out->credit_delta = contract_value;
                out->cargo_moved = stock;
                out->faction_delta = group.contract_at_faction_station ? 0.15 : 0.0;
                out->utility = out->legal ? contract_value * credit_weight / 95.0 - pirate * 0.9 : -1.0;
                break;
            case SNW_OPT_GOSSIP_COURIER:
                out->travel = 1;
                out->utility = 0.18 + pirate * 0.25 + frontier * 0.15;
                break;
            case SNW_OPT_SELF_REFIT_HOME:
                out->legal = group.desired_commodity < COMMODITY_COUNT &&
                             station_finished_count(home_st, group.desired_commodity) >= group.desired_units &&
                             group.home_balance > 80.0;
                out->self_upgrade = out->legal;
                out->utility = out->legal ? 2.8 + group.persona_growth : -1.0;
                break;
            case SNW_OPT_IMPORT_FRAME:
            case SNW_OPT_IMPORT_LASER:
            case SNW_OPT_IMPORT_TRACTOR:
                out->legal = group.desired_commodity < COMMODITY_COUNT &&
                             station_finished_count(dest_st, group.desired_commodity) >= group.desired_units;
                out->travel = out->legal;
                out->import_module = out->legal;
                out->utility = out->legal ? 2.4 + group.persona_growth * 0.7 - pirate * 0.35 : -1.0;
                break;
            case SNW_OPT_SUPPLY_FRONTIER:
                out->legal = frontier > 0.20 && stock > 0;
                out->travel = out->legal;
                out->faction_work = out->legal;
                out->frontier_delta = out->legal ? frontier * 0.45 : 0.0;
                out->faction_delta = out->legal ? 0.20 + expansion * 0.30 : 0.0;
                out->credit_delta = out->legal ? 45.0 + frontier * 85.0 : 0.0;
                out->utility = out->legal ? frontier * 2.4 + expansion - pirate * 0.35 : -1.0;
                break;
            case SNW_OPT_ESCORT_CONVOY:
                out->legal = pirate > 0.30 && spawned > 1u;
                out->travel = out->legal;
                out->faction_work = out->legal;
                out->pirate_work = out->legal;
                out->pirate_delta = out->legal ? -pirate * 0.28 : 0.0;
                out->credit_delta = out->legal ? 40.0 + pirate * 90.0 : 0.0;
                out->utility = out->legal ? pirate * 2.3 + group.convoy_bonus : -1.0;
                break;
            case SNW_OPT_PATROL_ROUTE:
                out->legal = pirate > 0.45;
                out->travel = out->legal;
                out->faction_work = out->legal;
                out->pirate_work = out->legal;
                out->pirate_delta = out->legal ? -pirate * 0.45 : 0.0;
                out->credit_delta = out->legal ? 30.0 + pirate * 120.0 : 0.0;
                out->utility = out->legal ? pirate * 2.8 + group.pirate_bounty * 0.5 : -1.0;
                break;
            case SNW_OPT_RISKY_PROFIT:
                out->legal = home_is_pirate
                    ? dest_contraband_stock > 0.0 && home_contraband_demand > 0.20
                    : contract_value > 0.0 && pirate > 0.15;
                out->travel = out->legal;
                out->contract_value = home_is_pirate
                    ? contraband_price * snw_clip(dest_contraband_stock, 1.0, 5.0)
                    : contract_value;
                out->credit_delta = home_is_pirate ? out->contract_value : contract_value * 1.15;
                out->cargo_moved = out->legal
                    ? (home_is_pirate ? snw_clip(dest_contraband_stock, 1.0, 5.0) : 1.0)
                    : 0.0;
                out->pirate_delta = out->legal ? pirate * (home_is_pirate ? 0.16 : 0.10) : 0.0;
                out->faction_delta = home_is_pirate
                    ? 0.0
                    : (group.contract_at_faction_station ? 0.05 : -0.18);
                out->pirate_work = home_is_pirate && out->legal;
                out->utility = out->legal
                    ? (home_is_pirate
                       ? out->contract_value / 90.0 + home_contraband_demand * 1.4 -
                         pirate * (0.45 + group.persona_risk * 0.45) +
                         route_profit_memory * 0.55 - route_patrol_heat * 1.10
                       : contract_value / 85.0 - pirate * (0.8 + group.persona_risk))
                    : -1.0;
                break;
            default:
                out->legal = 0;
                out->utility = -1.0;
                break;
            }
            if (!out->legal) out->utility -= 1.0;
        }

        double best = -1e300;
        group.target = 0;
        for (size_t oi = 0; oi < SNW_OPTION_COUNT; ++oi) {
            if (group.outcomes[oi].utility > best) {
                best = group.outcomes[oi].utility;
                group.target = oi;
            }
        }
        size_t selected = group.target;
        const snw_outcome_t *out = &group.outcomes[selected];
        double social_utility = out->utility;
        if (pirate > 0.75 && group.outcomes[SNW_OPT_PATROL_ROUTE].legal) {
            selected = SNW_OPT_PATROL_ROUTE;
            out = &group.outcomes[selected];
            social_utility = out->utility + 0.20;
        }

        double contraband_delta = 0.0;
        station_t *credit_station = (selected == SNW_OPT_RISKY_PROFIT && home_is_pirate)
            ? home_st
            : (selected == SNW_OPT_HAUL_CONTRACT || selected == SNW_OPT_RISKY_PROFIT)
                ? dest_st
                : home_st;
        if (out->credit_delta > 0.0) {
            ledger_earn(credit_station, npc->session_token, (float)out->credit_delta);
            if (credit_station == home_st) metrics.own_station_credit += out->credit_delta;
            else if (((credit_station - w->stations) % (int)SNW_PERSISTENT_FACTIONS) == faction)
                metrics.faction_station_credit += out->credit_delta;
            else
                metrics.off_faction_credit += out->credit_delta;
            if (snw_persistent_is_pirate_market((size_t)(credit_station - w->stations), station_count))
                metrics.pirate_station_credit += out->credit_delta;
            else
                metrics.lawful_station_credit += out->credit_delta;
        }
        if (selected == SNW_OPT_MINE_HOME) {
            snw_seed_finished(home_st, COMMODITY_FERRITE_INGOT, 1);
        } else if ((selected == SNW_OPT_HAUL_CONTRACT ||
                    (selected == SNW_OPT_RISKY_PROFIT && !home_is_pirate) ||
                    selected == SNW_OPT_SUPPLY_FRONTIER) && stock > 0) {
            (void)station_finished_drain(home_st, contract_commodity, 1);
            snw_seed_finished(dest_st, contract_commodity, 1);
        }
        if (selected == SNW_OPT_RISKY_PROFIT && home_is_pirate && out->legal) {
            contraband_delta = snw_clip(out->cargo_moved, 0.0, metrics.contraband_stock[dest]);
            metrics.contraband_stock[dest] = snw_clip(metrics.contraband_stock[dest] - contraband_delta,
                                                       0.0,
                                                       50.0);
            metrics.contraband_stock[home] = snw_clip(metrics.contraband_stock[home] + contraband_delta,
                                                       0.0,
                                                       75.0);
            metrics.contraband_demand[home] = snw_clip(metrics.contraband_demand[home] - contraband_delta * 0.08,
                                                        0.0,
                                                        1.8);
            metrics.contraband_credit += out->credit_delta;
            metrics.contraband_volume += contraband_delta;
            metrics.lawful_supply_leakage += dest_is_pirate ? contraband_delta * 0.25 : contraband_delta;
            metrics.route_profit_memory[home][dest] =
                snw_clip(metrics.route_profit_memory[home][dest] +
                         out->credit_delta / 900.0 + contraband_delta * 0.12,
                         0.0,
                         6.0);
            metrics.pirate_risk[home] = snw_clip(metrics.pirate_risk[home] + contraband_delta * 0.012,
                                                  0.0,
                                                  1.7);
            metrics.smuggle_runs++;
        } else if (selected == SNW_OPT_PATROL_ROUTE && out->legal) {
            metrics.contraband_stock[dest] = snw_clip(metrics.contraband_stock[dest] - 0.4 - pirate * 0.2,
                                                       0.0,
                                                       75.0);
            metrics.route_patrol_heat[home][dest] =
                snw_clip(metrics.route_patrol_heat[home][dest] + 0.55 + pirate * 0.20,
                         0.0,
                         5.0);
            metrics.route_profit_memory[home][dest] =
                snw_clip(metrics.route_profit_memory[home][dest] - 0.35 - pirate * 0.10,
                         0.0,
                         6.0);
        } else if (selected == SNW_OPT_ESCORT_CONVOY && out->legal) {
            metrics.contraband_demand[dest] = snw_clip(metrics.contraband_demand[dest] + 0.03,
                                                        0.0,
                                                        1.8);
            metrics.route_patrol_heat[home][dest] =
                snw_clip(metrics.route_patrol_heat[home][dest] + 0.18,
                         0.0,
                         5.0);
        }
        if (selected == SNW_OPT_RISKY_PROFIT && !home_is_pirate && out->legal) {
            metrics.lawful_risky_profit_runs++;
        }
        if (!home_is_pirate && !dest_is_pirate && selected != SNW_OPT_PATROL_ROUTE) {
            metrics.contraband_stock[dest] = snw_clip(metrics.contraband_stock[dest] + pirate * 0.015,
                                                       0.0,
                                                       50.0);
        }
        metrics.frontier_need[dest] = snw_clip(metrics.frontier_need[dest] -
                                               (selected == SNW_OPT_SUPPLY_FRONTIER ? 0.18 : -0.015),
                                               0.0,
                                               1.5);
        metrics.pirate_risk[dest] = snw_clip(metrics.pirate_risk[dest] +
                                             (out->pirate_delta * 0.55) + 0.01,
                                             0.0,
                                             1.7);
        if (out->faction_delta > 0.0) {
            metrics.faction_control[dest][faction] =
                snw_clip(metrics.faction_control[dest][faction] + out->faction_delta * 0.08,
                         0.0,
                         1.0);
        }

        double pirate_damage = 0.0;
        if (out->travel &&
            selected != SNW_OPT_ESCORT_CONVOY &&
            selected != SNW_OPT_PATROL_ROUTE) {
            double hazard_prob = snw_clip((pirate - 0.35) * 0.85 +
                                          (selected == SNW_OPT_RISKY_PROFIT ? 0.22 : 0.0),
                                          0.0,
                                          0.82);
            double hazard_roll = (double)(snw_rng_next(&rng) % 1000u) / 1000.0;
            if (hazard_roll < hazard_prob) {
                float dmg = (float)(6.0 + pirate * 18.0 +
                                    (selected == SNW_OPT_RISKY_PROFIT ? 12.0 : 0.0));
                apply_npc_ship_damage(w, slot, dmg);
                ship = world_npc_ship_for(w, slot);
                float hull_now = ship != NULL ? ship->hull : npc->hull;
                pirate_damage = (double)fmaxf(0.0f, hull_before - hull_now);
                metrics.pirate_damage_events++;
                metrics.pirate_damage += pirate_damage;
            }
        }
        world_sim_step(w, 1.0f / 30.0f);
        ship = world_npc_ship_for(w, slot);
        double hull_after = ship != NULL ? (double)ship->hull : (double)npc->hull;
        if (active_before && (!npc->active || hull_after <= 0.0)) metrics.npc_losses++;

        metrics.ticks++;
        metrics.decisions++;
        metrics.legal_decisions += out->legal ? 1u : 0u;
        metrics.faction_aligned_decisions += group.contract_at_own_station || group.contract_at_faction_station ? 1u : 0u;
        metrics.pirate_events += out->pirate_work ? 1u : 0u;
        metrics.autonomous_route_choices += autonomous_route ? 1u : 0u;
        metrics.memory_route_reuses += memory_route ? 1u : 0u;
        metrics.patrol_deterrence_events += patrol_deterrence ? 1u : 0u;
        metrics.frontier_supply_events += selected == SNW_OPT_SUPPLY_FRONTIER ? 1u : 0u;
        metrics.utility_sum += out->utility;
        metrics.social_utility_sum += social_utility;
        metrics.action_count[selected]++;
        if (trace != NULL) {
            snw_persistent_write_decision(trace, (uint64_t)tick, slot, home, faction,
                                          &group, selected, out->utility, social_utility,
                                          pirate_damage, hull_after, home_is_pirate,
                                          dest_is_pirate, contraband_delta,
                                          metrics.contraband_demand[home],
                                          metrics.route_profit_memory[home][dest],
                                          metrics.route_patrol_heat[home][dest],
                                          autonomous_route,
                                          memory_route,
                                          patrol_deterrence);
        }
    }

    for (size_t s = 0; s < station_count; ++s) {
        metrics.pirate_pressure_sum += metrics.pirate_risk[s];
        metrics.frontier_need_sum += metrics.frontier_need[s];
        metrics.contraband_demand_sum += metrics.contraband_demand[s];
        for (size_t d = 0; d < station_count; ++d) {
            metrics.route_profit_memory_sum += metrics.route_profit_memory[s][d];
            metrics.route_patrol_heat_sum += metrics.route_patrol_heat[s][d];
        }
    }
    if (trace != NULL) fclose(trace);

    size_t active_scaffolds = 0;
    for (int i = 0; i < MAX_SCAFFOLDS; ++i) {
        if (w->scaffolds[i].active) active_scaffolds++;
    }

    FILE *summary = config->summary_path != NULL ? fopen(config->summary_path, "w") : NULL;
    if (config->summary_path != NULL && summary == NULL) {
        free(w);
        return 0;
    }
    FILE *outs[2] = {stdout, summary};
    for (size_t si = 0; si < 2; ++si) {
        FILE *fp = outs[si];
        if (fp == NULL) continue;
        fprintf(fp, "{\"schema\":");
        snw_json_string(fp, SNW_PERSISTENT_SUMMARY_SCHEMA);
        fprintf(fp, ",\"task\":");
        snw_json_string(fp, SNW_TASK_NAME);
        fprintf(fp, ",\"trace_schema\":");
        snw_json_string(fp, SNW_PERSISTENT_SCHEMA);
        fprintf(fp, ",\"ticks\":%zu", config->persistent_ticks);
        fprintf(fp, ",\"stations\":%zu", station_count);
        fprintf(fp, ",\"npcs\":%zu", spawned);
        fprintf(fp, ",\"decisions\":%zu", metrics.decisions);
        fprintf(fp, ",\"legal_rate\":%.9g", snw_rate(metrics.legal_decisions, metrics.decisions));
        fprintf(fp, ",\"faction_aligned_rate\":%.9g", snw_rate(metrics.faction_aligned_decisions, metrics.decisions));
        fprintf(fp, ",\"mean_utility\":%.9g", metrics.decisions ? metrics.utility_sum / (double)metrics.decisions : 0.0);
        fprintf(fp, ",\"mean_social_utility\":%.9g", metrics.decisions ? metrics.social_utility_sum / (double)metrics.decisions : 0.0);
        fprintf(fp, ",\"own_station_credit\":%.9g", metrics.own_station_credit);
        fprintf(fp, ",\"faction_station_credit\":%.9g", metrics.faction_station_credit);
        fprintf(fp, ",\"off_faction_credit\":%.9g", metrics.off_faction_credit);
        fprintf(fp, ",\"lawful_station_credit\":%.9g", metrics.lawful_station_credit);
        fprintf(fp, ",\"pirate_station_credit\":%.9g", metrics.pirate_station_credit);
        fprintf(fp, ",\"contraband_credit\":%.9g", metrics.contraband_credit);
        fprintf(fp, ",\"contraband_volume\":%.9g", metrics.contraband_volume);
        fprintf(fp, ",\"lawful_supply_leakage\":%.9g", metrics.lawful_supply_leakage);
        fprintf(fp, ",\"mean_pirate_risk\":%.9g", station_count ? metrics.pirate_pressure_sum / (double)station_count : 0.0);
        fprintf(fp, ",\"mean_frontier_need\":%.9g", station_count ? metrics.frontier_need_sum / (double)station_count : 0.0);
        fprintf(fp, ",\"mean_contraband_demand\":%.9g", station_count ? metrics.contraband_demand_sum / (double)station_count : 0.0);
        fprintf(fp, ",\"pirate_events\":%zu", metrics.pirate_events);
        fprintf(fp, ",\"pirate_damage_events\":%zu", metrics.pirate_damage_events);
        fprintf(fp, ",\"pirate_damage\":%.9g", metrics.pirate_damage);
        fprintf(fp, ",\"npc_losses\":%zu", metrics.npc_losses);
        fprintf(fp, ",\"pirate_markets\":%zu", metrics.pirate_market_count);
        fprintf(fp, ",\"pirate_npcs\":%zu", metrics.pirate_npcs);
        fprintf(fp, ",\"lawful_npcs\":%zu", metrics.lawful_npcs);
        fprintf(fp, ",\"smuggle_runs\":%zu", metrics.smuggle_runs);
        fprintf(fp, ",\"lawful_risky_profit_runs\":%zu", metrics.lawful_risky_profit_runs);
        fprintf(fp, ",\"autonomous_route_choices\":%zu", metrics.autonomous_route_choices);
        fprintf(fp, ",\"memory_route_reuses\":%zu", metrics.memory_route_reuses);
        fprintf(fp, ",\"patrol_deterrence_events\":%zu", metrics.patrol_deterrence_events);
        fprintf(fp, ",\"mean_route_profit_memory\":%.9g",
                station_count ? metrics.route_profit_memory_sum / (double)(station_count * station_count) : 0.0);
        fprintf(fp, ",\"mean_route_patrol_heat\":%.9g",
                station_count ? metrics.route_patrol_heat_sum / (double)(station_count * station_count) : 0.0);
        fprintf(fp, ",\"frontier_supply_events\":%zu", metrics.frontier_supply_events);
        fprintf(fp, ",\"frontier_virtual_pilots\":%d", w->frontier_virtual_pilots);
        fprintf(fp, ",\"frontier_plans_created\":%u", w->frontier_plans_created);
        fprintf(fp, ",\"frontier_scaffold_orders\":%u", w->frontier_scaffold_orders);
        fprintf(fp, ",\"frontier_module_plans_created\":%u", w->frontier_module_plans_created);
        fprintf(fp, ",\"frontier_module_scaffold_orders\":%u", w->frontier_module_scaffold_orders);
        fprintf(fp, ",\"frontier_virtual_scaffolds_manufactured\":%u", w->frontier_virtual_scaffolds_manufactured);
        fprintf(fp, ",\"frontier_virtual_scaffold_deliveries\":%u", w->frontier_virtual_scaffold_deliveries);
        fprintf(fp, ",\"frontier_virtual_supply_deliveries\":%u", w->frontier_virtual_supply_deliveries);
        fprintf(fp, ",\"active_scaffolds\":%zu", active_scaffolds);
        fprintf(fp, ",\"action_count\":");
        snw_write_option_counts(fp, metrics.action_count);
        fputs("}\n", fp);
    }
    if (summary != NULL) fclose(summary);
    free(w);
    return 1;
}

static void snw_usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--schema] [--train-groups n] [--test-groups n] "
            "[--epochs n] [--learning-rate r] [--seed n] "
            "[--trace path] [--summary path] [--save path] "
            "[--persistent-economy] [--ticks n] [--npcs n] [--stations n]\n",
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
    config->persistent_economy = 0;
    config->persistent_ticks = 240u;
    config->persistent_npcs = 48u;
    config->persistent_stations = 12u;
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
        } else if (strcmp(argv[i], "--persistent-economy") == 0) {
            config->persistent_economy = 1;
        } else if (strcmp(argv[i], "--ticks") == 0 && i + 1 < argc) {
            if (!snw_parse_size(argv[++i], &config->persistent_ticks)) return 0;
        } else if (strcmp(argv[i], "--npcs") == 0 && i + 1 < argc) {
            if (!snw_parse_size(argv[++i], &config->persistent_npcs)) return 0;
        } else if (strcmp(argv[i], "--stations") == 0 && i + 1 < argc) {
            if (!snw_parse_size(argv[++i], &config->persistent_stations)) return 0;
        } else {
            return 0;
        }
    }
    return *schema_only || (config->train_groups > 0u && config->test_groups > 0u && config->epochs > 0u);
}

static void snw_print_schema(void)
{
    printf("{\"task\":\"%s\",\"trace_schema\":\"%s\",\"summary_schema\":\"%s\","
           "\"social_slice_schema\":\"%s\","
           "\"candidate_count\":%u,\"feature_count\":%u,"
           "\"feature_set\":\"signal-npc-worker-v2\",\"options\":[",
           SNW_TASK_NAME, SNW_TRACE_SCHEMA, SNW_SUMMARY_SCHEMA, SNW_SOCIAL_SLICE_SCHEMA,
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
    if (config.persistent_economy) {
        if (!snw_persistent_run(&config)) {
            fprintf(stderr, "failed to run Signal NPC persistent economy\n");
            return 1;
        }
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
    if (!snw_append_social_slice_trace(config.trace_path, net, &test, &metrics)) {
        nn_free(net);
        snw_dataset_free(&train);
        snw_dataset_free(&test);
        return 1;
    }
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
