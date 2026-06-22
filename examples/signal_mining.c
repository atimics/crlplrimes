#include "nn.h"

#include "chain_log.h"
#include "game_sim.h"
#include "manifest.h"
#include "sim_autopilot.h"
#include "sim_flight.h"
#include "sim_nav.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SM_TRACE_SCHEMA "signal.mining_counterfactual.v1"
#define SM_SUMMARY_SCHEMA "crlp.signal_mining.summary.v1"
#define SM_TASK_NAME "signal_mining"
#define SM_BASE_OPTION_COUNT 8u
#define SM_OPTION_COUNT 14u
#define SM_FEATURE_COUNT 70u
#define SM_SIM_DT (1.0f / 120.0f)

typedef enum {
    SM_OPT_AUTOPILOT_LOOP = 0,
    SM_OPT_MINE_CLEAR = 1,
    SM_OPT_MINE_RICH = 2,
    SM_OPT_COLLECT_LOCAL = 3,
    SM_OPT_DELIVER_REFINERY = 4,
    SM_OPT_DOCK_REPAIR = 5,
    SM_OPT_SUPPLY_SHIPYARD = 6,
    SM_OPT_EXPORT_REPAIR_KITS = 7,
} sm_option_t;

typedef enum {
    SM_ROUTE_DIRECT = 0,
    SM_ROUTE_CLEAR_APPROACH = 1,
    SM_ROUTE_STATION_EGRESS = 2,
    SM_ROUTE_DOCK_LANE = 3,
    SM_ROUTE_COUNT = 4,
} sm_route_rule_t;

typedef enum {
    SM_SPEED_CRUISE = 0,
    SM_SPEED_SLOW_CONTACT = 1,
    SM_SPEED_FAST_TRANSIT = 2,
    SM_SPEED_COUNT = 3,
} sm_speed_rule_t;

typedef enum {
    SM_INTERACT_NONE = 0,
    SM_INTERACT_MINE = 1,
    SM_INTERACT_TRACTOR = 2,
    SM_INTERACT_DOCK = 3,
    SM_INTERACT_TRADE = 4,
    SM_INTERACT_COUNT = 5,
} sm_interaction_rule_t;

typedef struct {
    sm_option_t option;
    const char *name;
    const char *target_rule;
    sm_route_rule_t route_rule;
    sm_speed_rule_t speed_rule;
    sm_interaction_rule_t interaction_rule;
    bool opportunistic;
    bool station_legal;
} sm_policy_hypothesis_t;

typedef struct {
    uint32_t world_seed;
    uint32_t scenario_seed;
    int scenario_kind;
    int station_index;
    int prefix_ticks;
    vec2 spawn_pos;
    vec2 spawn_vel;
    float spawn_angle;
    float spawn_hull;
    bool enable_npcs;
    bool start_docked;
} sm_scenario_init_t;

typedef struct {
    int target_asteroid;
    int target_fragment;
    int target_station;
    vec2 target_pos;
    double target_dist;
    double target_alignment_cos;
    double target_alignment_sin;
    double target_ore;
    double target_hp;
    double target_radius;
    double target_grade;
    double target_clear;
    double raw_ore_need;
    double repair_kit_level;
    int mineable_count;
    int local_fragment_count;
    double nearest_mine_dist;
    double nearest_clear_mine_dist;
    double richest_mine_dist;
    double nearest_fragment_dist;
    double refinery_dist;
    double dock_dist;
} sm_state_view_t;

typedef struct {
    double utility;
    double hull_loss;
    double damage_amount;
    double credits_delta;
    double ore_delta;
    int damage_events;
    int death_events;
    int repair_events;
    int mining_ticks;
    int fractures;
    int pickups;
    int smelts;
    int peak_towed;
    int final_towed;
    int station_ingots_delta;
    int repair_kits_delta;
    double hull_gain;
    double dock_progress;
    int final_docked;
    int start_repair_kits;
    int final_repair_kits;
    int final_ship_repair_kits;
    int final_ship_shipyard_inputs;
    int shipyard_inputs_delta;
    int repair_dock_kits_delta;
} sm_outcome_t;

typedef struct {
    uint64_t group_id;
    sm_scenario_init_t init;
    nn_real features[SM_OPTION_COUNT][SM_FEATURE_COUNT];
    sm_outcome_t outcomes[SM_OPTION_COUNT];
    size_t target;
    size_t baseline;
    double second_best_utility;
} sm_group_t;

typedef struct {
    sm_group_t *groups;
    size_t count;
} sm_dataset_t;

typedef struct {
    const char *summary_path;
    const char *trace_path;
    const char *save_path;
    uint32_t seed;
    size_t train_groups;
    size_t test_groups;
    int horizon_ticks;
    size_t epochs;
    nn_real learning_rate;
} sm_config_t;

typedef struct {
    size_t groups;
    size_t top1;
    size_t top3;
    double selected_utility;
    double oracle_utility;
    double regret;
    double baseline_utility;
    double baseline_regret;
    size_t selected_damage;
    size_t selected_death;
    size_t selected_smelt;
    size_t selected_tow;
    size_t baseline_damage;
    size_t baseline_death;
    size_t baseline_smelt;
    size_t baseline_tow;
    size_t option_selected[SM_OPTION_COUNT];
    size_t option_target[SM_OPTION_COUNT];
} sm_metrics_t;

static const sm_policy_hypothesis_t SM_HYPOTHESES[SM_OPTION_COUNT] = {
    {SM_OPT_AUTOPILOT_LOOP, "autopilot_loop", "autopilot_target", SM_ROUTE_DIRECT, SM_SPEED_CRUISE, SM_INTERACT_NONE, false, false},
    {SM_OPT_MINE_CLEAR, "mine_clear_cruise", "nearest_clear_rock", SM_ROUTE_CLEAR_APPROACH, SM_SPEED_CRUISE, SM_INTERACT_MINE, false, false},
    {SM_OPT_MINE_CLEAR, "mine_clear_slow_contact", "nearest_clear_rock", SM_ROUTE_CLEAR_APPROACH, SM_SPEED_SLOW_CONTACT, SM_INTERACT_MINE, false, false},
    {SM_OPT_MINE_CLEAR, "mine_clear_station_egress", "nearest_clear_rock", SM_ROUTE_STATION_EGRESS, SM_SPEED_CRUISE, SM_INTERACT_MINE, false, false},
    {SM_OPT_MINE_RICH, "mine_rich_cruise", "richest_visible_rock", SM_ROUTE_CLEAR_APPROACH, SM_SPEED_CRUISE, SM_INTERACT_MINE, false, false},
    {SM_OPT_MINE_RICH, "mine_rich_slow_contact", "richest_visible_rock", SM_ROUTE_CLEAR_APPROACH, SM_SPEED_SLOW_CONTACT, SM_INTERACT_MINE, false, false},
    {SM_OPT_COLLECT_LOCAL, "collect_fragment_slow", "nearest_visible_fragment", SM_ROUTE_CLEAR_APPROACH, SM_SPEED_SLOW_CONTACT, SM_INTERACT_TRACTOR, true, false},
    {SM_OPT_COLLECT_LOCAL, "collect_fragment_fast", "nearest_visible_fragment", SM_ROUTE_CLEAR_APPROACH, SM_SPEED_FAST_TRANSIT, SM_INTERACT_TRACTOR, true, false},
    {SM_OPT_DELIVER_REFINERY, "deliver_refinery_lane", "nearest_matching_refinery", SM_ROUTE_DOCK_LANE, SM_SPEED_CRUISE, SM_INTERACT_DOCK, false, true},
    {SM_OPT_DELIVER_REFINERY, "deliver_refinery_fast", "nearest_matching_refinery", SM_ROUTE_DOCK_LANE, SM_SPEED_FAST_TRANSIT, SM_INTERACT_DOCK, false, true},
    {SM_OPT_DOCK_REPAIR, "dock_repair_slow", "repair_capable_dock", SM_ROUTE_DOCK_LANE, SM_SPEED_SLOW_CONTACT, SM_INTERACT_DOCK, false, true},
    {SM_OPT_DOCK_REPAIR, "dock_repair_fast", "repair_capable_dock", SM_ROUTE_DOCK_LANE, SM_SPEED_FAST_TRANSIT, SM_INTERACT_DOCK, false, true},
    {SM_OPT_SUPPLY_SHIPYARD, "supply_shipyard_inputs", "shipyard_input_trade", SM_ROUTE_DOCK_LANE, SM_SPEED_CRUISE, SM_INTERACT_TRADE, false, true},
    {SM_OPT_EXPORT_REPAIR_KITS, "export_repair_kits", "repair_kit_trade", SM_ROUTE_DOCK_LANE, SM_SPEED_CRUISE, SM_INTERACT_TRADE, false, true},
};

static const NNFeatureSchema SM_FEATURES[SM_FEATURE_COUNT] = {
    {"bias", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_hypothesis_index", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_autopilot_loop", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_mine_clear_cruise", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_mine_clear_slow_contact", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_mine_clear_station_egress", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_mine_rich_cruise", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_mine_rich_slow_contact", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_collect_fragment_slow", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_collect_fragment_fast", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_deliver_refinery_lane", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_deliver_refinery_fast", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_dock_repair_slow", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_dock_repair_fast", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_supply_shipyard_inputs", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hypothesis_export_repair_kits", NN_FEATURE_STATE_OBSERVATION, 1},
    {"route_direct", NN_FEATURE_STATE_OBSERVATION, 1},
    {"route_clear_approach", NN_FEATURE_STATE_OBSERVATION, 1},
    {"route_station_egress", NN_FEATURE_STATE_OBSERVATION, 1},
    {"route_dock_lane", NN_FEATURE_STATE_OBSERVATION, 1},
    {"speed_cruise", NN_FEATURE_STATE_OBSERVATION, 1},
    {"speed_slow_contact", NN_FEATURE_STATE_OBSERVATION, 1},
    {"speed_fast_transit", NN_FEATURE_STATE_OBSERVATION, 1},
    {"interaction_none", NN_FEATURE_STATE_OBSERVATION, 1},
    {"interaction_mine", NN_FEATURE_STATE_OBSERVATION, 1},
    {"interaction_tractor", NN_FEATURE_STATE_OBSERVATION, 1},
    {"interaction_dock", NN_FEATURE_STATE_OBSERVATION, 1},
    {"interaction_trade", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_opportunistic", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_station_legal", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_autopilot_loop", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_mine_nearest_clear", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_mine_richest", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_collect_local_fragments", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_deliver_refinery", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_dock_repair", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_supply_shipyard_inputs", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_base_export_repair_kits", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hull_ratio", NN_FEATURE_STATE_OBSERVATION, 1},
    {"speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"forward_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"signal_quality", NN_FEATURE_STATE_OBSERVATION, 1},
    {"docked", NN_FEATURE_STATE_OBSERVATION, 1},
    {"towed_ratio", NN_FEATURE_STATE_OBSERVATION, 1},
    {"towed_count", NN_FEATURE_STATE_OBSERVATION, 1},
    {"nearest_mineable_rock_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"nearest_clear_rock_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"richest_rock_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"nearest_fragment_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"nearest_refinery_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"nearest_dock_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_heading_cos", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_heading_sin", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_ore", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_hp", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_radius", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_grade", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_target_clear_approach", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_is_mining", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_is_delivery", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_is_service", NN_FEATURE_STATE_OBSERVATION, 1},
    {"refinery_raw_ore_need", NN_FEATURE_STATE_OBSERVATION, 1},
    {"station_repair_kit_level", NN_FEATURE_STATE_OBSERVATION, 1},
    {"local_fragment_count", NN_FEATURE_STATE_OBSERVATION, 1},
    {"mineable_rock_count", NN_FEATURE_STATE_OBSERVATION, 1},
    {"damaged", NN_FEATURE_STATE_OBSERVATION, 1},
    {"has_tow", NN_FEATURE_STATE_OBSERVATION, 1},
    {"prefix_phase", NN_FEATURE_STATE_OBSERVATION, 1},
    {"scenario_kind", NN_FEATURE_STATE_OBSERVATION, 1},
};

static const NNExperimentSpec SM_EXPERIMENT_SPEC = {
    .task_name = SM_TASK_NAME,
    .role = "Signal tactical mining and supply-chain hypothesis ranker over simulator-certified candidates",
    .state_space = "authoritative Signal world state after deterministic scenario prefix: ship hull/kinematics, towed fragments, nearby rocks, fragments, docks, refinery demand, shipyard inputs, and repair-kit stock",
    .state_distribution = "deterministic Signal world seeds with station starts, rock approaches, autopilot-generated mining prefixes, and live NPC economy prefixes",
    .action_space = "fourteen bounded policy hypotheses generated from a grammar over target rule, route constraint, speed/contact profile, and interaction primitive; each compiles to legal player controls",
    .candidate_enumerator = "enumerate the bounded policy grammar for every sampled state; each hypothesis owns a deterministic controller for the replay horizon",
    .observable_features = "candidate hypothesis id, grammar primitive one-hots, base option id, plus observable ship, rock, fragment, station, demand, and prefix-state features",
    .leakage_policy = NN_LEAKAGE_FORBIDDEN,
    .neural_scorer = "tiny MLP scores one tactical option row and softmax ranks options within a replay group",
    .grounding_authority = NN_GROUNDING_DETERMINISTIC_REPLAY,
    .grounding_operator = "authoritative Signal world replay over deterministic tactical option prefixes",
    .exact_verifier = "fresh deterministic Signal simulator replay from the same scenario seed and prefix for each candidate option",
    .verifier_feedback = "certified utility, smelts, pickups, fractures, mining ticks, towed fragments, shipyard-input deltas, repair-kit deltas, hull loss, damage events, and death events",
    .candidate_grammar = "target selector x route constraint x speed/contact profile x interaction primitive",
    .outcome_row_schema = SM_TRACE_SCHEMA " and " SM_SUMMARY_SCHEMA,
    .claim_scope = "offline tactical option ranking over the declared deterministic Signal mining scenarios",
    .known_non_claims =
        "does not claim live economy control or scenario families outside the replay-certified prefixes",
    .rule_grammar = "target selector x route constraint x speed/contact profile x interaction primitive; neural scorer ranks only replay-certified grammar hypotheses",
    .certificate_checker = "authoritative replay records candidate outcomes; verifier outputs are labels/diagnostics and are not scorer inputs",
    .gate_executor = "candidate policies drive normal Signal input_intent_t controls through world_sim_step; no sim state is shallow-copied",
    .proposer_inputs = "observable state/action features only; replay utility and event counters are targets",
    .proposer_targets = "argmax over replay-certified tactical utility",
    .proposer_mode = NN_PROPOSER_SELECTION,
    .baseline_orderings = "fixed grammar baseline: slow repair when damaged, refinery lane when towing, slow fragment collection when visible, otherwise nearest-clear-rock cruise",
    .generalization_axis = "train deterministic world seeds to held-out deterministic world seeds, economy prefixes, and supply-chain starts",
    .trace_schema = SM_TRACE_SCHEMA,
    .replay_protocol = "JSONL counterfactual groups with stable tactical option ids and identical deterministic scenario prefixes",
    .rl_protocol = "offline simulator replay ranking; later live shadow rows can append the same option ids",
    .features = SM_FEATURES,
    .feature_count = sizeof(SM_FEATURES) / sizeof(SM_FEATURES[0]),
};

static const sm_policy_hypothesis_t *sm_hypothesis(size_t candidate)
{
    if (candidate >= SM_OPTION_COUNT) candidate = 0u;
    return &SM_HYPOTHESES[candidate];
}

static float sm_hypothesis_speed(const sm_policy_hypothesis_t *hyp, float cruise_speed)
{
    if (hyp == NULL) return cruise_speed;
    switch (hyp->speed_rule) {
    case SM_SPEED_SLOW_CONTACT:
        return fminf(cruise_speed, 125.0f);
    case SM_SPEED_FAST_TRANSIT:
        return fmaxf(cruise_speed, 360.0f);
    case SM_SPEED_CRUISE:
    default:
        return cruise_speed;
    }
}

static double sm_clip(double x, double lo, double hi)
{
    if (!isfinite(x)) return 0.0;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static double sm_scale(double x, double denom)
{
    if (denom <= 0.0) return 0.0;
    return sm_clip(x / denom, 0.0, 1.0);
}

static double sm_signed_scale(double x, double denom)
{
    if (denom <= 0.0) return 0.0;
    return sm_clip(x / denom, -1.0, 1.0);
}

static uint32_t sm_rng_next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x ? x : 0x9e3779b9u;
    return *state;
}

static double sm_rng_unit(uint32_t *state)
{
    return (double)(sm_rng_next(state) >> 8) / 16777215.0;
}

static float sm_rng_range(uint32_t *state, float lo, float hi)
{
    return lo + (hi - lo) * (float)sm_rng_unit(state);
}

static int sm_parse_u32(const char *text, uint32_t *out_value)
{
    if (text == NULL || out_value == NULL || text[0] == '\0') return 0;
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) return 0;
    *out_value = (uint32_t)value;
    return 1;
}

static int sm_parse_size(const char *text, size_t *out_value)
{
    if (text == NULL || out_value == NULL || text[0] == '\0') return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out_value = (size_t)value;
    return 1;
}

static int sm_parse_int(const char *text, int *out_value)
{
    if (text == NULL || out_value == NULL || text[0] == '\0') return 0;
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out_value = (int)value;
    return 1;
}

static int sm_parse_real(const char *text, nn_real *out_value)
{
    if (text == NULL || out_value == NULL || text[0] == '\0') return 0;
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) return 0;
    *out_value = (nn_real)value;
    return 1;
}

static void sm_json_string(FILE *fp, const char *text)
{
    (void)nn_write_json_string(fp, text);
}

static const char *sm_scenario_name(int kind)
{
    switch (kind) {
    case 0: return "station_start";
    case 1: return "rock_approach";
    case 2: return "mid_mining_prefix";
    case 3: return "tow_or_return_prefix";
    case 4: return "damaged_station_recovery";
    case 5: return "damaged_field_recovery";
    case 6: return "shipyard_supply_chain";
    case 7: return "repair_kit_export";
    default: return "mixed";
    }
}

static int sm_tow_capacity(const ship_t *ship)
{
    int cap = 2 + ship->tractor_level * 2;
    if (cap < 1) cap = 1;
    if (cap > 10) cap = 10;
    return cap;
}

static commodity_t sm_towed_commodity(const world_t *w, const server_player_t *sp)
{
    for (int t = 0; t < sp->ship.towed_count; ++t) {
        int idx = sp->ship.towed_fragments[t];
        if (idx < 0 || idx >= MAX_ASTEROIDS) continue;
        const asteroid_t *a = &w->asteroids[idx];
        if (!a->active) continue;
        return a->commodity;
    }
    return COMMODITY_COUNT;
}

static const asteroid_t *sm_first_towed_fragment(const world_t *w, const server_player_t *sp)
{
    for (int t = 0; t < sp->ship.towed_count; ++t) {
        int idx = sp->ship.towed_fragments[t];
        if (idx < 0 || idx >= MAX_ASTEROIDS) continue;
        const asteroid_t *a = &w->asteroids[idx];
        if (!a->active) continue;
        return a;
    }
    return NULL;
}

static int sm_count_station_finished(const world_t *w, commodity_t c)
{
    int count = 0;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        if (!station_exists(&w->stations[s])) continue;
        count += station_finished_count(&w->stations[s], c);
    }
    return count;
}

static int sm_count_station_ingots(const world_t *w)
{
    return sm_count_station_finished(w, COMMODITY_FERRITE_INGOT) +
           sm_count_station_finished(w, COMMODITY_CUPRITE_INGOT) +
           sm_count_station_finished(w, COMMODITY_CRYSTAL_INGOT);
}

static int sm_count_shipyard_inputs(const world_t *w)
{
    int count = 0;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_SHIPYARD)) continue;
        if (!station_produces(st, COMMODITY_FRAME)) {
            count += station_finished_count(st, COMMODITY_FRAME);
        }
        if (!station_produces(st, COMMODITY_LASER_MODULE)) {
            count += station_finished_count(st, COMMODITY_LASER_MODULE);
        }
        if (!station_produces(st, COMMODITY_TRACTOR_MODULE)) {
            count += station_finished_count(st, COMMODITY_TRACTOR_MODULE);
        }
    }
    return count;
}

static int sm_count_repair_dock_kits(const world_t *w)
{
    int count = 0;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        if (station_has_module(st, MODULE_SHIPYARD)) continue;
        count += station_finished_count(st, COMMODITY_REPAIR_KIT);
    }
    return count;
}

static void sm_seed_supply_chain_fixture(world_t *w, const server_player_t *sp)
{
    static const uint8_t origin[8] = { 'B', 'R', 'N', 'T', 'R', 'A', 'D', 'E' };
    if (w == NULL || sp == NULL) return;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        ledger_earn(st, sp->session_token, 500.0f);
        if (station_produces(st, COMMODITY_FRAME)) {
            station_finished_mint(st, COMMODITY_FRAME, 8, origin);
        }
        if (station_produces(st, COMMODITY_LASER_MODULE)) {
            station_finished_mint(st, COMMODITY_LASER_MODULE, 8, origin);
        }
        if (station_produces(st, COMMODITY_TRACTOR_MODULE)) {
            station_finished_mint(st, COMMODITY_TRACTOR_MODULE, 8, origin);
        }
        if (station_has_module(st, MODULE_SHIPYARD)) {
            station_finished_mint(st, COMMODITY_REPAIR_KIT, 20, origin);
        } else if (station_has_module(st, MODULE_DOCK)) {
            station_finished_drain(st, COMMODITY_REPAIR_KIT, 1000);
            st->_inventory_cache[COMMODITY_REPAIR_KIT] = 0.0f;
        }
    }
}

static bool sm_station_can_smelt_ore(const station_t *st, commodity_t ore)
{
    if (ore == COMMODITY_COUNT) {
        return station_has_module(st, MODULE_HOPPER) &&
               station_has_module(st, MODULE_FURNACE);
    }
    return station_can_smelt(st, ore);
}

static bool sm_furnace_allowed_for_fragment(const asteroid_t *fragment,
                                            int station_idx,
                                            int module_idx)
{
    if (fragment == NULL) return true;
    if (fragment->commodity != COMMODITY_CRYSTAL_ORE ||
        fragment->crystal_stage != CRYSTAL_STAGE_INTERMEDIATE) {
        return true;
    }
    if (fragment->crystal_stage_station == 0xFFu ||
        fragment->crystal_stage_module == 0xFFu) {
        return true;
    }
    return !(fragment->crystal_stage_station == (uint8_t)station_idx &&
             fragment->crystal_stage_module == (uint8_t)module_idx);
}

static vec2 sm_station_smelt_drop_point(const station_t *st,
                                        int station_idx,
                                        commodity_t ore,
                                        const asteroid_t *fragment)
{
    vec2 best_mid = st->pos;
    float best_silo_d = 1e18f;
    bool found = false;
    for (int m = 0; m < st->module_count; ++m) {
        if (st->modules[m].scaffold) continue;
        if (st->modules[m].type != MODULE_FURNACE) continue;
        commodity_t furnace_ore = module_instance_input_ore(&st->modules[m]);
        if (ore != COMMODITY_COUNT && furnace_ore != ore) continue;
        if (!sm_furnace_allowed_for_fragment(fragment, station_idx, m)) continue;
        int ring = st->modules[m].ring;
        vec2 furnace_pos = module_world_pos_ring(st, ring, st->modules[m].slot);
        int adj_rings[2] = {ring + 1, ring - 1};
        for (int ri = 0; ri < 2; ++ri) {
            int adj = adj_rings[ri];
            if (adj < 1 || adj > STATION_NUM_RINGS) continue;
            for (int m2 = 0; m2 < st->module_count; ++m2) {
                if (st->modules[m2].ring != adj) continue;
                if (st->modules[m2].scaffold) continue;
                if (st->modules[m2].type != MODULE_HOPPER) continue;
                if ((commodity_t)st->modules[m2].commodity != furnace_ore) continue;
                vec2 hopper_pos = module_world_pos_ring(st, adj, st->modules[m2].slot);
                float d = v2_dist_sq(furnace_pos, hopper_pos);
                if (d < best_silo_d) {
                    best_silo_d = d;
                    best_mid = v2_scale(v2_add(furnace_pos, hopper_pos), 0.5f);
                    found = true;
                }
            }
        }
    }
    return found ? best_mid : st->pos;
}

static bool sm_can_mine_asteroid(const server_player_t *sp, const asteroid_t *a)
{
    if (a == NULL || !a->active || asteroid_is_collectible(a)) return false;
    return (int)a->tier >= (int)max_mineable_tier(sp->ship.mining_level);
}

static bool sm_clear_mining_approach(const world_t *w,
                                     const server_player_t *sp,
                                     const asteroid_t *a)
{
    if (a == NULL) return false;
    const hull_def_t *hull = ship_hull_def(&sp->ship);
    vec2 from_rock = v2_sub(sp->ship.pos, a->pos);
    float from_len = v2_len(from_rock);
    if (from_len < 1.0f) return true;
    vec2 outward = v2_scale(from_rock, 1.0f / from_len);
    vec2 approach = v2_add(a->pos, v2_scale(outward, a->radius + 120.0f));
    return nav_segment_clear(w, sp->ship.pos, approach, hull->ship_radius + 30.0f);
}

static int sm_find_nearest_mineable(const world_t *w,
                                    const server_player_t *sp,
                                    bool require_clear)
{
    int best = -1;
    float best_d = 1e18f;
    for (int i = 0; i < MAX_ASTEROIDS; ++i) {
        const asteroid_t *a = &w->asteroids[i];
        if (!sm_can_mine_asteroid(sp, a)) continue;
        if (signal_strength_at(w, a->pos) < 0.35f) continue;
        if (require_clear && !sm_clear_mining_approach(w, sp, a)) continue;
        float d = v2_dist_sq(sp->ship.pos, a->pos);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

static int sm_find_richest_mineable(const world_t *w, const server_player_t *sp)
{
    int best = -1;
    double best_score = -1e18;
    for (int i = 0; i < MAX_ASTEROIDS; ++i) {
        const asteroid_t *a = &w->asteroids[i];
        if (!sm_can_mine_asteroid(sp, a)) continue;
        if (signal_strength_at(w, a->pos) < 0.35f) continue;
        double dist = sqrt((double)v2_dist_sq(sp->ship.pos, a->pos));
        double score = (double)a->ore + (double)a->max_ore * 0.25 - dist * 0.015;
        if (sm_clear_mining_approach(w, sp, a)) score += 3.0;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

static int sm_find_nearest_fragment(const world_t *w, const server_player_t *sp)
{
    int best = -1;
    float best_d = 1e18f;
    for (int i = 0; i < MAX_ASTEROIDS; ++i) {
        const asteroid_t *a = &w->asteroids[i];
        if (!a->active || !asteroid_is_collectible(a)) continue;
        if (signal_strength_at(w, a->pos) < 0.25f) continue;
        float d = v2_dist_sq(sp->ship.pos, a->pos);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

static int sm_find_refinery(const world_t *w, const server_player_t *sp)
{
    commodity_t ore = sm_towed_commodity(w, sp);
    int best = -1;
    float best_d = 1e18f;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        if (!sm_station_can_smelt_ore(st, ore)) continue;
        float d = v2_dist_sq(sp->ship.pos, st->pos);
        if (d < best_d) {
            best_d = d;
            best = s;
        }
    }
    if (best >= 0) return best;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        float d = v2_dist_sq(sp->ship.pos, st->pos);
        if (d < best_d) {
            best_d = d;
            best = s;
        }
    }
    return best;
}

static int sm_find_nearest_dock(const world_t *w, const server_player_t *sp)
{
    int best = -1;
    float best_d = 1e18f;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        float d = v2_dist_sq(sp->ship.pos, st->pos);
        if (d < best_d) {
            best_d = d;
            best = s;
        }
    }
    return best;
}

static int sm_find_repair_dock(const world_t *w, const server_player_t *sp)
{
    int best = -1;
    float best_score = 1e18f;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        float dist = sqrtf(v2_dist_sq(sp->ship.pos, st->pos));
        float score = dist;
        if (station_finished_count(st, COMMODITY_REPAIR_KIT) > 0) score -= 3000.0f;
        if (station_has_module(st, MODULE_SHIPYARD)) score -= 1200.0f;
        if (score < best_score) {
            best_score = score;
            best = s;
        }
    }
    return best >= 0 ? best : sm_find_nearest_dock(w, sp);
}

static double sm_repair_dock_distance(const world_t *w, const server_player_t *sp)
{
    int s = sm_find_repair_dock(w, sp);
    if (s < 0) return 9000.0;
    return sqrt((double)v2_dist_sq(sp->ship.pos, w->stations[s].pos));
}

static bool sm_is_shipyard_input(commodity_t c)
{
    return c == COMMODITY_FRAME ||
           c == COMMODITY_LASER_MODULE ||
           c == COMMODITY_TRACTOR_MODULE;
}

static commodity_t sm_carried_shipyard_input(const server_player_t *sp)
{
    if (ship_finished_count(&sp->ship, COMMODITY_FRAME) > 0) return COMMODITY_FRAME;
    if (ship_finished_count(&sp->ship, COMMODITY_LASER_MODULE) > 0) return COMMODITY_LASER_MODULE;
    if (ship_finished_count(&sp->ship, COMMODITY_TRACTOR_MODULE) > 0) return COMMODITY_TRACTOR_MODULE;
    return COMMODITY_COUNT;
}

static int sm_count_ship_shipyard_inputs(const server_player_t *sp)
{
    return ship_finished_count(&sp->ship, COMMODITY_FRAME) +
           ship_finished_count(&sp->ship, COMMODITY_LASER_MODULE) +
           ship_finished_count(&sp->ship, COMMODITY_TRACTOR_MODULE);
}

static int sm_find_product_source(const world_t *w,
                                  const server_player_t *sp,
                                  commodity_t commodity)
{
    int best = -1;
    float best_score = 1e18f;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        if (!station_produces(st, commodity)) continue;
        int stock = station_finished_count(st, commodity);
        float dist = sqrtf(v2_dist_sq(sp->ship.pos, st->pos));
        float score = dist - (float)stock * 80.0f;
        if (stock > 0) score -= 2500.0f;
        if (score < best_score) {
            best_score = score;
            best = s;
        }
    }
    return best;
}

static int sm_find_shipyard_needing(const world_t *w,
                                    const server_player_t *sp,
                                    commodity_t commodity)
{
    int best = -1;
    float best_score = -1e18f;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        if (!station_has_module(st, MODULE_SHIPYARD)) continue;
        station_demand_t demand = station_demand_for(st, commodity);
        float dist = sqrtf(v2_dist_sq(sp->ship.pos, st->pos));
        float score = demand.severity * 6000.0f - dist * 0.15f;
        if (score > best_score) {
            best_score = score;
            best = s;
        }
    }
    return best;
}

static int sm_find_repair_kit_importer(const world_t *w, const server_player_t *sp)
{
    int best = -1;
    float best_score = -1e18f;
    for (int s = 0; s < MAX_STATIONS; ++s) {
        const station_t *st = &w->stations[s];
        if (!station_is_active(st)) continue;
        if (!station_has_module(st, MODULE_DOCK)) continue;
        if (station_has_module(st, MODULE_SHIPYARD)) continue;
        station_demand_t demand = station_demand_for(st, COMMODITY_REPAIR_KIT);
        float dist = sqrtf(v2_dist_sq(sp->ship.pos, st->pos));
        float score = demand.severity * 7000.0f - dist * 0.12f;
        if (score > best_score) {
            best_score = score;
            best = s;
        }
    }
    return best;
}

static commodity_t sm_best_shipyard_input_need(const world_t *w,
                                               const server_player_t *sp)
{
    commodity_t choices[3] = {
        COMMODITY_FRAME,
        COMMODITY_LASER_MODULE,
        COMMODITY_TRACTOR_MODULE,
    };
    commodity_t best = COMMODITY_COUNT;
    float best_score = -1e18f;
    for (size_t i = 0; i < 3u; ++i) {
        commodity_t c = choices[i];
        int dest = sm_find_shipyard_needing(w, sp, c);
        int src = sm_find_product_source(w, sp, c);
        if (dest < 0 || src < 0) continue;
        float severity = station_demand_for(&w->stations[dest], c).severity;
        float stock = (float)station_finished_count(&w->stations[src], c);
        float score = severity * 8.0f + sm_clip(stock, 0.0, 12.0) * 0.2f;
        if (score > best_score) {
            best_score = score;
            best = c;
        }
    }
    return best;
}

static void sm_reset_player(world_t *w, server_player_t *sp)
{
    ship_cleanup(&sp->ship);
    memset(sp, 0, sizeof(*sp));
    player_init_ship(sp, w);
    sp->id = 0;
    sp->connected = true;
    sp->session_ready = true;
    sp->session_token[0] = 1u;
    sp->docked = false;
    sp->current_station = -1;
    sp->nearby_station = -1;
    sp->autopilot_mode = 0;
    sp->autopilot_target = -1;
    sp->was_in_signal = true;
    sp->boost_hold_timer = 0.0f;
    memset(&sp->input, 0, sizeof(sp->input));
}

static void sm_setup_world(const sm_scenario_init_t *init,
                           world_t *w,
                           server_player_t **out_sp)
{
    memset(w, 0, sizeof(*w));
    w->rng = init->world_seed;
    world_reset(w);
    world_seed_station_manifests(w);
    if (!init->enable_npcs) {
        for (int n = 0; n < MAX_NPC_SHIPS; ++n) {
            w->npc_ships[n].active = false;
        }
    }
    server_player_t *sp = &w->players[0];
    sm_reset_player(w, sp);
    sp->ship_asset_id = SHIP_ASSET_ID_NONE;
    sp->ship.pos = init->spawn_pos;
    sp->ship.vel = init->spawn_vel;
    sp->ship.angle = init->spawn_angle;
    {
        float max_hull = ship_max_hull(&sp->ship);
        sp->ship.hull = init->spawn_hull > 0.0f
                            ? fminf(max_hull, fmaxf(1.0f, init->spawn_hull))
                            : max_hull;
    }
    sp->ship.towed_count = 0;
    memset(sp->ship.towed_fragments, -1, sizeof(sp->ship.towed_fragments));
    sp->ship.towed_scaffold = -1;
    sp->docked = false;
    sp->current_station = -1;
    memset(&sp->input, 0, sizeof(sp->input));
    if (init->start_docked &&
        init->station_index >= 0 &&
        init->station_index < MAX_STATIONS &&
        station_is_active(&w->stations[init->station_index])) {
        sp->docked = true;
        sp->current_station = init->station_index;
        sp->nearby_station = init->station_index;
        sp->in_dock_range = true;
        anchor_ship_in_station(sp, w);
    }
    *out_sp = sp;
}

static vec2 sm_random_station_nearby(uint32_t *rng,
                                     const station_t *station,
                                     float min_radius,
                                     float max_radius)
{
    float angle = sm_rng_range(rng, -PI_F, PI_F);
    float radius = sm_rng_range(rng, min_radius, max_radius);
    return v2_add(station->pos, v2(cosf(angle) * radius, sinf(angle) * radius));
}

static void sm_apply_cmd(server_player_t *sp, flight_cmd_t cmd)
{
    memset(&sp->input, 0, sizeof(sp->input));
    sp->input.turn = sm_clip(cmd.turn, -1.0, 1.0);
    sp->input.thrust = sm_clip(cmd.thrust, -1.0, 1.0);
    sp->input.reverse_thrust = sp->input.thrust < -0.1f;
}

static void sm_fly_to_point(const world_t *w,
                            server_player_t *sp,
                            nav_path_t *path,
                            vec2 target,
                            float standoff,
                            float max_speed)
{
    float dist = v2_len(v2_sub(target, sp->ship.pos));
    flight_cmd_t cmd = dist <= standoff + 55.0f
                     ? flight_hover_near(w, &sp->ship, target, standoff)
                     : flight_steer_to(w, &sp->ship, path, target, standoff, max_speed, SM_SIM_DT);
    sm_apply_cmd(sp, cmd);
}

static void sm_policy_mine_target(world_t *w,
                                  server_player_t *sp,
                                  nav_path_t *path,
                                  int target,
                                  const sm_policy_hypothesis_t *hyp)
{
    if (sp->ship.towed_count >= sm_tow_capacity(&sp->ship)) {
        target = -1;
    }
    if (target < 0 || target >= MAX_ASTEROIDS || !sm_can_mine_asteroid(sp, &w->asteroids[target])) {
        target = sm_find_nearest_mineable(w, sp, true);
        if (target < 0) target = sm_find_nearest_mineable(w, sp, false);
    }
    if (target < 0) {
        int fragment = sm_find_nearest_fragment(w, sp);
        if (fragment >= 0) {
            sm_fly_to_point(w, sp, path, w->asteroids[fragment].pos, 70.0f, 180.0f);
            sp->input.tractor_hold = true;
            return;
        }
        sm_apply_cmd(sp, flight_brake(&sp->ship));
        return;
    }

    asteroid_t *a = &w->asteroids[target];
    float standoff = a->radius + 118.0f;
    float dist = v2_len(v2_sub(a->pos, sp->ship.pos));
    if (dist > standoff + 45.0f) {
        sm_fly_to_point(w, sp, path, a->pos, standoff, sm_hypothesis_speed(hyp, 220.0f));
        sp->input.tractor_hold = true;
        return;
    }

    sm_fly_to_point(w, sp, path, a->pos, standoff, sm_hypothesis_speed(hyp, 150.0f));
    vec2 to_rock = v2_sub(a->pos, sp->ship.pos);
    float heading = atan2f(to_rock.y, to_rock.x);
    float diff = wrap_angle(heading - sp->ship.angle);
    sp->input.turn = flight_face_heading(&sp->ship, heading);
    sp->input.mine = fabsf(diff) < 0.42f && dist < a->radius + MINING_RANGE + 90.0f;
    sp->input.mining_target_hint = target;
    sp->input.tractor_hold = true;
}

static void sm_policy_collect(world_t *w,
                              server_player_t *sp,
                              nav_path_t *path,
                              const sm_policy_hypothesis_t *hyp)
{
    int cap = sm_tow_capacity(&sp->ship);
    if (sp->ship.towed_count >= cap) {
        int s = sm_find_refinery(w, sp);
        if (s >= 0) {
            commodity_t ore = sm_towed_commodity(w, sp);
            const asteroid_t *fragment = sm_first_towed_fragment(w, sp);
            vec2 pt = sm_station_smelt_drop_point(&w->stations[s], s, ore, fragment);
            sm_fly_to_point(w, sp, path, pt, 55.0f, sm_hypothesis_speed(hyp, 210.0f));
            sp->input.tractor_hold = true;
            return;
        }
    }

    int fragment = sm_find_nearest_fragment(w, sp);
    if (fragment >= 0) {
        sm_fly_to_point(w, sp, path, w->asteroids[fragment].pos, 55.0f, sm_hypothesis_speed(hyp, 190.0f));
        sp->input.tractor_hold = true;
        return;
    }
    sm_policy_mine_target(w, sp, path, sm_find_nearest_mineable(w, sp, true), hyp);
}

static void sm_policy_deliver(world_t *w,
                              server_player_t *sp,
                              nav_path_t *path,
                              const sm_policy_hypothesis_t *hyp)
{
    int s = sm_find_refinery(w, sp);
    if (s < 0) {
        sm_apply_cmd(sp, flight_brake(&sp->ship));
        return;
    }
    commodity_t ore = sm_towed_commodity(w, sp);
    const asteroid_t *fragment = sm_first_towed_fragment(w, sp);
    vec2 smelt_pt = sm_station_smelt_drop_point(&w->stations[s], s, ore, fragment);
    sm_fly_to_point(w, sp, path, smelt_pt, 50.0f, sm_hypothesis_speed(hyp, 220.0f));
    sp->input.tractor_hold = true;
    if (v2_dist_sq(sp->ship.pos, w->stations[s].pos) < w->stations[s].dock_radius * w->stations[s].dock_radius) {
        sp->input.dock = true;
        sp->input.interact = true;
    }
}

static void sm_policy_dock_repair(world_t *w,
                                  server_player_t *sp,
                                  nav_path_t *path,
                                  const sm_policy_hypothesis_t *hyp)
{
    int s = sm_find_repair_dock(w, sp);
    if (s < 0) {
        sm_apply_cmd(sp, flight_brake(&sp->ship));
        return;
    }
    if (sp->docked) {
        station_t *dock_st = &w->stations[sp->current_station];
        memset(&sp->input, 0, sizeof(sp->input));
        if (station_produces(dock_st, COMMODITY_REPAIR_KIT) &&
            ship_finished_count(&sp->ship, COMMODITY_REPAIR_KIT) <= 0) {
            sp->input.buy_product = true;
            sp->input.buy_commodity = COMMODITY_REPAIR_KIT;
            sp->input.buy_grade = MINING_GRADE_COUNT;
        }
        sp->input.service_repair = true;
        return;
    }
    if (sp->docking_approach) {
        memset(&sp->input, 0, sizeof(sp->input));
        sp->input.dock = true;
        sp->input.interact = true;
        return;
    }
    station_t *st = &w->stations[s];
    float center_dist = v2_len(v2_sub(sp->ship.pos, st->pos));
    vec2 approach = station_approach_target(st, sp->ship.pos);
    vec2 entry = station_entry_target(st, sp->ship.pos);
    vec2 target = center_dist > st->dock_radius + 260.0f ? entry : approach;
    sm_fly_to_point(w, sp, path, target, 25.0f, sm_hypothesis_speed(hyp, 150.0f));
    if (center_dist < DOCK_APPROACH_RANGE + 40.0f || sp->in_dock_range) {
        sp->input.dock = true;
        sp->input.interact = true;
        sp->input.service_repair = true;
    }
}

static void sm_route_to_dock(world_t *w,
                             server_player_t *sp,
                             nav_path_t *path,
                             int station_index,
                             float max_speed)
{
    if (station_index < 0 || station_index >= MAX_STATIONS ||
        !station_is_active(&w->stations[station_index])) {
        sm_apply_cmd(sp, flight_brake(&sp->ship));
        return;
    }
    if (sp->docked) {
        memset(&sp->input, 0, sizeof(sp->input));
        if (sp->current_station != station_index) {
            sp->input.launch = true;
            sp->input.interact = true;
        }
        return;
    }
    if (sp->docking_approach) {
        memset(&sp->input, 0, sizeof(sp->input));
        sp->input.dock = true;
        sp->input.interact = true;
        return;
    }

    station_t *st = &w->stations[station_index];
    float center_dist = v2_len(v2_sub(sp->ship.pos, st->pos));
    vec2 approach = station_approach_target(st, sp->ship.pos);
    vec2 entry = station_entry_target(st, sp->ship.pos);
    vec2 target = center_dist > st->dock_radius + 260.0f ? entry : approach;
    sm_fly_to_point(w, sp, path, target, 25.0f, max_speed);
    if (center_dist < DOCK_APPROACH_RANGE + 40.0f || sp->in_dock_range) {
        sp->input.dock = true;
        sp->input.interact = true;
    }
}

static void sm_policy_buy_deliver_product(world_t *w,
                                          server_player_t *sp,
                                          nav_path_t *path,
                                          commodity_t commodity,
                                          int destination,
                                          const sm_policy_hypothesis_t *hyp)
{
    if (commodity >= COMMODITY_COUNT) {
        sm_apply_cmd(sp, flight_brake(&sp->ship));
        return;
    }

    int carried = ship_finished_count(&sp->ship, commodity);
    if (sp->docked) {
        station_t *st = &w->stations[sp->current_station];
        memset(&sp->input, 0, sizeof(sp->input));
        if (carried > 0) {
            if (sp->current_station == destination ||
                station_demand_for(st, commodity).severity > 0.0f) {
                sp->input.service_sell = true;
                sp->input.service_sell_only = commodity;
                sp->input.service_sell_grade = MINING_GRADE_COUNT;
                return;
            }
            sp->input.launch = true;
            sp->input.interact = true;
            return;
        }
        if (station_produces(st, commodity)) {
            sp->input.buy_product = true;
            sp->input.buy_commodity = commodity;
            sp->input.buy_grade = MINING_GRADE_COUNT;
            return;
        }
        sp->input.launch = true;
        sp->input.interact = true;
        return;
    }

    int target = carried > 0 ? destination : sm_find_product_source(w, sp, commodity);
    if (target < 0 && carried > 0) target = destination;
    if (target < 0) {
        sm_apply_cmd(sp, flight_brake(&sp->ship));
        return;
    }
    sm_route_to_dock(w, sp, path, target, sm_hypothesis_speed(hyp, 550.0f));
}

static void sm_policy_supply_shipyard(world_t *w,
                                      server_player_t *sp,
                                      nav_path_t *path,
                                      const sm_policy_hypothesis_t *hyp)
{
    commodity_t c = sm_carried_shipyard_input(sp);
    if (c == COMMODITY_COUNT) c = sm_best_shipyard_input_need(w, sp);
    if (!sm_is_shipyard_input(c)) {
        sm_policy_collect(w, sp, path, hyp);
        return;
    }
    int dest = sm_find_shipyard_needing(w, sp, c);
    sm_policy_buy_deliver_product(w, sp, path, c, dest, hyp);
}

static void sm_policy_export_repair_kits(world_t *w,
                                         server_player_t *sp,
                                         nav_path_t *path,
                                         const sm_policy_hypothesis_t *hyp)
{
    int dest = sm_find_repair_kit_importer(w, sp);
    sm_policy_buy_deliver_product(w, sp, path, COMMODITY_REPAIR_KIT, dest, hyp);
}

static void sm_apply_option_policy(world_t *w,
                                   server_player_t *sp,
                                   size_t candidate,
                                   nav_path_t *path,
                                   int sticky_target)
{
    const sm_policy_hypothesis_t *hyp = sm_hypothesis(candidate);
    sm_option_t option = hyp->option;
    if (option == SM_OPT_AUTOPILOT_LOOP) {
        memset(&sp->input, 0, sizeof(sp->input));
        sp->autopilot_mode = 1;
        return;
    }

    sp->autopilot_mode = 0;
    sp->autopilot_target = -1;
    switch (option) {
    case SM_OPT_MINE_CLEAR:
        sm_policy_mine_target(w, sp, path,
                              sticky_target >= 0 ? sticky_target
                                                 : sm_find_nearest_mineable(w, sp, true),
                              hyp);
        break;
    case SM_OPT_MINE_RICH:
        sm_policy_mine_target(w, sp, path,
                              sticky_target >= 0 ? sticky_target
                                                 : sm_find_richest_mineable(w, sp),
                              hyp);
        break;
    case SM_OPT_COLLECT_LOCAL:
        sm_policy_collect(w, sp, path, hyp);
        break;
    case SM_OPT_DELIVER_REFINERY:
        sm_policy_deliver(w, sp, path, hyp);
        break;
    case SM_OPT_DOCK_REPAIR:
        sm_policy_dock_repair(w, sp, path, hyp);
        break;
    case SM_OPT_SUPPLY_SHIPYARD:
        sm_policy_supply_shipyard(w, sp, path, hyp);
        break;
    case SM_OPT_EXPORT_REPAIR_KITS:
        sm_policy_export_repair_kits(w, sp, path, hyp);
        break;
    case SM_OPT_AUTOPILOT_LOOP:
    default:
        break;
    }
}

static void sm_run_prefix(const sm_scenario_init_t *init, world_t *w, server_player_t *sp)
{
    if (init->prefix_ticks <= 0) return;
    sp->autopilot_mode = 1;
    sp->autopilot_state = AUTOPILOT_STEP_FIND_TARGET;
    sp->autopilot_target = -1;
    sp->autopilot_last_pos = sp->ship.pos;
    for (int t = 0; t < init->prefix_ticks; ++t) {
        memset(&sp->input, 0, sizeof(sp->input));
        world_sim_step(w, SM_SIM_DT);
    }
    sp->autopilot_mode = 0;
    sp->autopilot_target = -1;
    memset(&sp->input, 0, sizeof(sp->input));
}

static void sm_detach_world_heap_pointers(world_t *w)
{
    if (w == NULL) return;
    w->signal_cache.strength = NULL;
    w->asteroid_grid.entries = NULL;
    for (int i = 0; i < MAX_STATIONS; ++i) {
        w->stations[i].manifest = (manifest_t){0};
        w->stations[i].receipts_opaque = NULL;
    }
    for (int i = 0; i < MAX_PLAYERS; ++i) {
        w->players[i].ship.manifest = (manifest_t){0};
        w->players[i].ship.receipts_opaque = NULL;
    }
    for (int i = 0; i < MAX_SHIPS; ++i) {
        w->ships[i].manifest = (manifest_t){0};
        w->ships[i].receipts_opaque = NULL;
    }
    for (int i = 0; i < MAX_SHIP_ASSETS; ++i) {
        w->ship_assets[i].ship.manifest = (manifest_t){0};
        w->ship_assets[i].ship.receipts_opaque = NULL;
    }
}

static void sm_cleanup_world(world_t *w)
{
    if (w == NULL) return;
    for (int i = 0; i < MAX_STATIONS; ++i) {
        w->stations[i].manifest = (manifest_t){0};
        w->stations[i].receipts_opaque = NULL;
    }
    for (int i = 0; i < MAX_PLAYERS; ++i) {
        w->players[i].ship.manifest = (manifest_t){0};
        w->players[i].ship.receipts_opaque = NULL;
    }
    for (int i = 0; i < MAX_SHIP_ASSETS; ++i) {
        w->ship_assets[i].ship.manifest = (manifest_t){0};
        w->ship_assets[i].ship.receipts_opaque = NULL;
    }
    for (int i = 0; i < MAX_SHIPS; ++i) {
        w->ships[i].manifest = (manifest_t){0};
        w->ships[i].receipts_opaque = NULL;
    }
    world_cleanup(w);
}

static bool sm_clone_world_heap_buffers(world_t *dst, const world_t *src)
{
    if (src->signal_cache.strength != NULL) {
        size_t count = (size_t)SIGNAL_GRID_DIM * (size_t)SIGNAL_GRID_DIM;
        dst->signal_cache.strength = malloc(count * sizeof(*dst->signal_cache.strength));
        if (dst->signal_cache.strength == NULL) return false;
        memcpy(dst->signal_cache.strength,
               src->signal_cache.strength,
               count * sizeof(*dst->signal_cache.strength));
    }
    if (src->asteroid_grid.entries != NULL && src->asteroid_grid.capacity > 0u) {
        size_t count = (size_t)src->asteroid_grid.capacity;
        dst->asteroid_grid.entries = malloc(count * sizeof(*dst->asteroid_grid.entries));
        if (dst->asteroid_grid.entries == NULL) return false;
        memcpy(dst->asteroid_grid.entries,
               src->asteroid_grid.entries,
               count * sizeof(*dst->asteroid_grid.entries));
    }
    return true;
}

static bool sm_clone_world(world_t *dst, const world_t *src)
{
    if (dst == NULL || src == NULL) return false;
    memset(dst, 0, sizeof(*dst));
    *dst = *src;
    sm_detach_world_heap_pointers(dst);

    if (!sm_clone_world_heap_buffers(dst, src)) goto fail;
    for (int i = 0; i < MAX_STATIONS; ++i) {
        if (!station_copy(&dst->stations[i], &src->stations[i])) goto fail;
    }
    for (int i = 0; i < MAX_PLAYERS; ++i) {
        if (!ship_copy(&dst->players[i].ship, &src->players[i].ship)) goto fail;
    }
    for (int i = 0; i < MAX_SHIPS; ++i) {
        if (!ship_copy(&dst->ships[i], &src->ships[i])) goto fail;
    }
    return true;

fail:
    sm_cleanup_world(dst);
    memset(dst, 0, sizeof(*dst));
    return false;
}

static void sm_accumulate_events(const world_t *w, int player_id, sm_outcome_t *outcome)
{
    for (int i = 0; i < w->events.count; ++i) {
        const sim_event_t *ev = &w->events.events[i];
        if (ev->player_id != player_id &&
            ev->type != SIM_EVENT_CONTRACT_COMPLETE &&
            ev->type != SIM_EVENT_SCAFFOLD_READY) {
            continue;
        }
        switch (ev->type) {
        case SIM_EVENT_FRACTURE:
            outcome->fractures++;
            break;
        case SIM_EVENT_PICKUP:
            outcome->pickups += ev->pickup.fragments > 0 ? ev->pickup.fragments : 1;
            break;
        case SIM_EVENT_MINING_TICK:
            outcome->mining_ticks++;
            break;
        case SIM_EVENT_SELL:
            outcome->smelts++;
            break;
        case SIM_EVENT_DAMAGE:
            outcome->damage_events++;
            outcome->damage_amount += ev->damage.amount;
            break;
        case SIM_EVENT_DEATH:
            outcome->death_events++;
            break;
        case SIM_EVENT_REPAIR:
            outcome->repair_events++;
            break;
        default:
            break;
        }
    }
}

static bool sm_replay_option(const sm_config_t *config,
                             const sm_scenario_init_t *init,
                             const world_t *base_world,
                             size_t candidate,
                             int sticky_target,
                             sm_outcome_t *outcome)
{
    const sm_policy_hypothesis_t *hyp = sm_hypothesis(candidate);
    sm_option_t option = hyp->option;
    if (config == NULL || init == NULL || base_world == NULL || outcome == NULL) return false;
    memset(outcome, 0, sizeof(*outcome));
    nav_caches_reset();
    world_t *w = calloc(1u, sizeof(*w));
    if (w == NULL) return false;
    if (!sm_clone_world(w, base_world)) {
        free(w);
        return false;
    }
    server_player_t *sp = &w->players[0];
    double start_hull = sp->ship.hull;
    double start_dock_dist = sm_repair_dock_distance(w, sp);
    double start_credits = sp->ship.stat_credits_earned;
    double start_ore = sp->ship.stat_ore_mined;
    int start_ingots = sm_count_station_ingots(w);
    int start_repair_kits = sm_count_station_finished(w, COMMODITY_REPAIR_KIT);
    int start_shipyard_inputs = sm_count_shipyard_inputs(w);
    int start_repair_dock_kits = sm_count_repair_dock_kits(w);
    outcome->start_repair_kits = start_repair_kits;
    nav_path_t path;
    memset(&path, 0, sizeof(path));

    for (int tick = 0; tick < config->horizon_ticks; ++tick) {
        sm_apply_option_policy(w, sp, candidate, &path, sticky_target);
        world_sim_step(w, SM_SIM_DT);
        sm_accumulate_events(w, sp->id, outcome);
        if (sp->ship.towed_count > outcome->peak_towed) {
            outcome->peak_towed = sp->ship.towed_count;
        }
    }

    outcome->final_towed = sp->ship.towed_count;
    outcome->final_docked = sp->docked ? 1 : 0;
    outcome->hull_loss = fmax(0.0, start_hull - (double)sp->ship.hull);
    outcome->hull_gain = fmax(0.0, (double)sp->ship.hull - start_hull);
    {
        double final_dock_dist = sm_repair_dock_distance(w, sp);
        outcome->dock_progress =
            sm_clip((start_dock_dist - final_dock_dist) / 2200.0, 0.0, 1.0);
    }
    outcome->credits_delta = (double)sp->ship.stat_credits_earned - start_credits;
    outcome->ore_delta = (double)sp->ship.stat_ore_mined - start_ore;
    outcome->station_ingots_delta = sm_count_station_ingots(w) - start_ingots;
    outcome->final_repair_kits = sm_count_station_finished(w, COMMODITY_REPAIR_KIT);
    outcome->final_ship_repair_kits = ship_finished_count(&sp->ship, COMMODITY_REPAIR_KIT);
    outcome->final_ship_shipyard_inputs = sm_count_ship_shipyard_inputs(sp);
    outcome->repair_kits_delta = outcome->final_repair_kits - start_repair_kits;
    outcome->shipyard_inputs_delta = sm_count_shipyard_inputs(w) - start_shipyard_inputs;
    outcome->repair_dock_kits_delta = sm_count_repair_dock_kits(w) - start_repair_dock_kits;

    outcome->utility =
        (double)outcome->smelts * 18.0 +
        (double)outcome->pickups * 3.0 +
        (double)outcome->fractures * 2.0 +
        (double)outcome->peak_towed * 0.85 +
        outcome->credits_delta * 0.025 +
        outcome->ore_delta * 0.03 +
        (double)outcome->mining_ticks * 0.004 -
        outcome->hull_loss * 0.45 -
        outcome->damage_amount * 0.25 -
        (double)outcome->damage_events * 2.0 -
        (double)outcome->death_events * 80.0;
    if (option == SM_OPT_DOCK_REPAIR && outcome->repair_events > 0) {
        outcome->utility += 2.0 + outcome->hull_gain * 0.65;
    }
    if (start_hull < 55.0 && outcome->hull_gain > 0.0) {
        outcome->utility += outcome->hull_gain * 0.35;
    }
    if (option == SM_OPT_DOCK_REPAIR && start_hull < 55.0) {
        outcome->utility += outcome->dock_progress * 3.0;
        if (outcome->final_docked) outcome->utility += 2.0;
    }
    if (option == SM_OPT_DELIVER_REFINERY && outcome->final_towed < outcome->peak_towed) {
        outcome->utility += 4.0;
    }
    if (option == SM_OPT_SUPPLY_SHIPYARD) {
        if (outcome->shipyard_inputs_delta > 0) {
            outcome->utility += (double)outcome->shipyard_inputs_delta * 4.0;
        }
        if (outcome->repair_kits_delta > 0) {
            outcome->utility += (double)outcome->repair_kits_delta * 0.35;
        }
        if (outcome->final_ship_shipyard_inputs > 0) {
            outcome->utility += (double)outcome->final_ship_shipyard_inputs * 2.5;
            if (init->scenario_kind == 6) outcome->utility += 18.0;
        }
    }
    if (option == SM_OPT_EXPORT_REPAIR_KITS) {
        if (outcome->repair_dock_kits_delta > 0) {
            outcome->utility += (double)outcome->repair_dock_kits_delta * 0.9;
        }
        if (outcome->final_ship_repair_kits > 0) {
            outcome->utility += (double)outcome->final_ship_repair_kits * 2.5;
            if (init->scenario_kind == 7) outcome->utility += 18.0;
        }
    }

    sm_cleanup_world(w);
    free(w);
    return true;
}

static void sm_init_scenario(const sm_config_t *config,
                             uint32_t split_seed,
                             size_t index,
                             sm_scenario_init_t *init)
{
    uint32_t rng = split_seed ^ (uint32_t)(index * UINT32_C(0x9e3779b9)) ^ UINT32_C(0x51f15eed);
    uint32_t world_seed = sm_rng_next(&rng) ^ config->seed;
    memset(init, 0, sizeof(*init));
    init->world_seed = world_seed;
    init->scenario_seed = rng;

    world_t *template_world = calloc(1u, sizeof(*template_world));
    if (template_world == NULL) return;
    template_world->rng = world_seed;
    world_reset(template_world);
    for (int n = 0; n < MAX_NPC_SHIPS; ++n) {
        template_world->npc_ships[n].active = false;
    }
    server_player_t *sp = &template_world->players[0];
    sm_reset_player(template_world, sp);

    init->station_index = (int)(sm_rng_next(&rng) % 3u);
    if (init->station_index < 0 || init->station_index >= MAX_STATIONS) init->station_index = 0;
    station_t *station = &template_world->stations[init->station_index];
    init->scenario_kind = (int)(index % 8u);
    init->enable_npcs = init->scenario_kind >= 6;

    int target = sm_find_nearest_mineable(template_world, sp, true);
    if (target < 0) target = sm_find_nearest_mineable(template_world, sp, false);
    if (target >= 0 && (init->scenario_kind == 1 || init->scenario_kind == 5)) {
        asteroid_t *a = &template_world->asteroids[target];
        float angle = sm_rng_range(&rng, -PI_F, PI_F);
        float radius = a->radius + sm_rng_range(&rng,
                                                 init->scenario_kind == 5 ? 240.0f : 420.0f,
                                                 init->scenario_kind == 5 ? 720.0f : 950.0f);
        vec2 dir = v2(cosf(angle), sinf(angle));
        init->spawn_pos = v2_add(a->pos, v2_scale(dir, radius));
        vec2 to_rock = v2_sub(a->pos, init->spawn_pos);
        init->spawn_angle = atan2f(to_rock.y, to_rock.x);
        init->spawn_vel = v2_scale(ship_forward(init->spawn_angle),
                                   sm_rng_range(&rng,
                                                init->scenario_kind == 5 ? 40.0f : 0.0f,
                                                init->scenario_kind == 5 ? 180.0f : 90.0f));
        if (init->scenario_kind == 5) {
            init->spawn_hull = sm_rng_range(&rng, 8.0f, 34.0f);
        }
        init->prefix_ticks = 0;
    } else if (init->scenario_kind >= 6) {
        int station_slot = 1;
        if (station_slot >= template_world->station_count) station_slot = init->station_index;
        station = &template_world->stations[station_slot];
        init->station_index = station_slot;
        init->spawn_pos = station->pos;
        init->spawn_vel = v2(0.0f, 0.0f);
        init->spawn_angle = sm_rng_range(&rng, -PI_F, PI_F);
        init->start_docked = true;
        init->prefix_ticks = config->horizon_ticks;
    } else {
        init->spawn_pos = sm_random_station_nearby(&rng,
                                                   station,
                                                   init->scenario_kind == 4 ? 620.0f : 850.0f,
                                                   init->scenario_kind == 4 ? 1500.0f : 2600.0f);
        init->spawn_vel = v2(sm_rng_range(&rng, -45.0f, 45.0f),
                             sm_rng_range(&rng, -45.0f, 45.0f));
        vec2 to_station = v2_sub(station->pos, init->spawn_pos);
        init->spawn_angle = atan2f(to_station.y, to_station.x) +
                            sm_rng_range(&rng, -0.65f, 0.65f);
        if (init->scenario_kind == 4) {
            init->spawn_hull = sm_rng_range(&rng, 12.0f, 45.0f);
            init->spawn_vel = v2(sm_rng_range(&rng, -120.0f, 120.0f),
                                 sm_rng_range(&rng, -120.0f, 120.0f));
        }
        if (init->scenario_kind == 2) {
            init->prefix_ticks = config->horizon_ticks / 2;
        } else if (init->scenario_kind == 3) {
            init->prefix_ticks = config->horizon_ticks;
        } else {
            init->prefix_ticks = 0;
        }
    }

    sm_cleanup_world(template_world);
    free(template_world);
}

static void sm_build_state_view(const world_t *w,
                                const server_player_t *sp,
                                size_t candidate,
                                sm_state_view_t *view)
{
    sm_option_t option = sm_hypothesis(candidate)->option;
    memset(view, 0, sizeof(*view));
    view->target_asteroid = -1;
    view->target_fragment = -1;
    view->target_station = -1;
    view->target_pos = sp->ship.pos;
    view->target_dist = 0.0;
    view->nearest_mine_dist = 1e9;
    view->nearest_clear_mine_dist = 1e9;
    view->richest_mine_dist = 1e9;
    view->nearest_fragment_dist = 1e9;
    view->refinery_dist = 1e9;
    view->dock_dist = 1e9;

    int nearest = sm_find_nearest_mineable(w, sp, false);
    int clear = sm_find_nearest_mineable(w, sp, true);
    int rich = sm_find_richest_mineable(w, sp);
    int fragment = sm_find_nearest_fragment(w, sp);
    int refinery = sm_find_refinery(w, sp);
    int dock = sm_find_nearest_dock(w, sp);
    int repair_dock = sm_find_repair_dock(w, sp);
    if (nearest >= 0) view->nearest_mine_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->asteroids[nearest].pos));
    if (clear >= 0) view->nearest_clear_mine_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->asteroids[clear].pos));
    if (rich >= 0) view->richest_mine_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->asteroids[rich].pos));
    if (fragment >= 0) view->nearest_fragment_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->asteroids[fragment].pos));
    if (refinery >= 0) view->refinery_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->stations[refinery].pos));
    if (repair_dock >= 0) view->dock_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->stations[repair_dock].pos));
    else if (dock >= 0) view->dock_dist = sqrt((double)v2_dist_sq(sp->ship.pos, w->stations[dock].pos));

    for (int i = 0; i < MAX_ASTEROIDS; ++i) {
        const asteroid_t *a = &w->asteroids[i];
        if (sm_can_mine_asteroid(sp, a) && signal_strength_at(w, a->pos) >= 0.35f) {
            view->mineable_count++;
        }
        if (a->active && asteroid_is_collectible(a) &&
            v2_dist_sq(sp->ship.pos, a->pos) < 900.0f * 900.0f) {
            view->local_fragment_count++;
        }
    }

    if (option == SM_OPT_MINE_CLEAR) {
        view->target_asteroid = clear >= 0 ? clear : nearest;
    } else if (option == SM_OPT_MINE_RICH) {
        view->target_asteroid = rich >= 0 ? rich : nearest;
    } else if (option == SM_OPT_COLLECT_LOCAL) {
        view->target_fragment = fragment;
    } else if (option == SM_OPT_DELIVER_REFINERY || option == SM_OPT_AUTOPILOT_LOOP) {
        if (refinery >= 0) {
            commodity_t ore = sm_towed_commodity(w, sp);
            const asteroid_t *towed = sm_first_towed_fragment(w, sp);
            view->target_station = refinery;
            view->target_pos = sm_station_smelt_drop_point(&w->stations[refinery], refinery, ore, towed);
        }
    } else if (option == SM_OPT_DOCK_REPAIR) {
        if (repair_dock >= 0) {
            view->target_station = repair_dock;
            view->target_pos = station_approach_target(&w->stations[repair_dock], sp->ship.pos);
        }
    } else if (option == SM_OPT_SUPPLY_SHIPYARD) {
        commodity_t c = sm_carried_shipyard_input(sp);
        if (c == COMMODITY_COUNT) c = sm_best_shipyard_input_need(w, sp);
        int dest = sm_find_shipyard_needing(w, sp, c);
        int src = sm_find_product_source(w, sp, c);
        int station_target = ship_finished_count(&sp->ship, c) > 0 ? dest : src;
        if (station_target >= 0) {
            view->target_station = station_target;
            view->target_pos = station_approach_target(&w->stations[station_target], sp->ship.pos);
        }
    } else if (option == SM_OPT_EXPORT_REPAIR_KITS) {
        int dest = sm_find_repair_kit_importer(w, sp);
        int src = sm_find_product_source(w, sp, COMMODITY_REPAIR_KIT);
        int station_target = ship_finished_count(&sp->ship, COMMODITY_REPAIR_KIT) > 0 ? dest : src;
        if (station_target >= 0) {
            view->target_station = station_target;
            view->target_pos = station_approach_target(&w->stations[station_target], sp->ship.pos);
        }
    }

    if (view->target_asteroid >= 0) {
        const asteroid_t *a = &w->asteroids[view->target_asteroid];
        view->target_pos = a->pos;
        view->target_ore = a->ore;
        view->target_hp = a->hp;
        view->target_radius = a->radius;
        view->target_grade = (double)a->grade;
        view->target_clear = sm_clear_mining_approach(w, sp, a) ? 1.0 : 0.0;
    } else if (view->target_fragment >= 0) {
        const asteroid_t *a = &w->asteroids[view->target_fragment];
        view->target_pos = a->pos;
        view->target_ore = a->ore;
        view->target_hp = a->hp;
        view->target_radius = a->radius;
        view->target_grade = (double)a->grade;
        view->target_clear = 1.0;
    }

    if (view->target_station >= 0) {
        const station_t *st = &w->stations[view->target_station];
        commodity_t ore = sm_towed_commodity(w, sp);
        if (ore == COMMODITY_COUNT && nearest >= 0) ore = w->asteroids[nearest].commodity;
        if (ore < COMMODITY_RAW_ORE_COUNT) {
            view->raw_ore_need = station_raw_ore_need_score(st, ore);
        }
        view->repair_kit_level = (double)station_finished_count(st, COMMODITY_REPAIR_KIT);
    }

    vec2 delta = v2_sub(view->target_pos, sp->ship.pos);
    float dist = v2_len(delta);
    view->target_dist = dist;
    if (dist > 0.001f) {
        vec2 dir = v2_scale(delta, 1.0f / dist);
        vec2 fwd = ship_forward(sp->ship.angle);
        view->target_alignment_cos = v2_dot(fwd, dir);
        view->target_alignment_sin = v2_cross(fwd, dir);
    } else {
        view->target_alignment_cos = 1.0;
        view->target_alignment_sin = 0.0;
    }
}

static void sm_encode_features(const sm_scenario_init_t *init,
                               const world_t *w,
                               const server_player_t *sp,
                               size_t candidate,
                               nn_real out_features[SM_FEATURE_COUNT])
{
    const sm_policy_hypothesis_t *hyp = sm_hypothesis(candidate);
    sm_option_t option = hyp->option;
    sm_state_view_t view;
    sm_build_state_view(w, sp, candidate, &view);
    vec2 fwd = ship_forward(sp->ship.angle);
    double max_hull = fmax(1.0, (double)ship_max_hull(&sp->ship));
    double speed = sqrt((double)v2_len_sq(sp->ship.vel));
    double forward_speed = v2_dot(sp->ship.vel, fwd);
    double hull_ratio = sm_clip((double)sp->ship.hull / max_hull, 0.0, 1.0);
    double towed_ratio = sm_scale((double)sp->ship.towed_count, (double)sm_tow_capacity(&sp->ship));

    memset(out_features, 0, sizeof(nn_real) * SM_FEATURE_COUNT);
    out_features[0] = (nn_real)1.0;
    out_features[1] = (nn_real)sm_scale((double)candidate, (double)(SM_OPTION_COUNT - 1u));
    for (size_t i = 0; i < SM_OPTION_COUNT; ++i) {
        out_features[2u + i] = (nn_real)(i == candidate ? 1.0 : 0.0);
    }
    for (size_t i = 0; i < SM_ROUTE_COUNT; ++i) {
        out_features[16u + i] = (nn_real)(i == (size_t)hyp->route_rule ? 1.0 : 0.0);
    }
    for (size_t i = 0; i < SM_SPEED_COUNT; ++i) {
        out_features[20u + i] = (nn_real)(i == (size_t)hyp->speed_rule ? 1.0 : 0.0);
    }
    for (size_t i = 0; i < SM_INTERACT_COUNT; ++i) {
        out_features[23u + i] = (nn_real)(i == (size_t)hyp->interaction_rule ? 1.0 : 0.0);
    }
    out_features[28] = (nn_real)(hyp->opportunistic ? 1.0 : 0.0);
    out_features[29] = (nn_real)(hyp->station_legal ? 1.0 : 0.0);
    for (size_t i = 0; i < SM_BASE_OPTION_COUNT; ++i) {
        out_features[30u + i] = (nn_real)(i == (size_t)option ? 1.0 : 0.0);
    }
    out_features[38] = (nn_real)hull_ratio;
    out_features[39] = (nn_real)sm_scale(speed, 450.0);
    out_features[40] = (nn_real)sm_signed_scale(forward_speed, 300.0);
    out_features[41] = (nn_real)sm_clip(signal_strength_at(w, sp->ship.pos), 0.0, 1.0);
    out_features[42] = (nn_real)(sp->docked ? 1.0 : 0.0);
    out_features[43] = (nn_real)towed_ratio;
    out_features[44] = (nn_real)sm_scale((double)sp->ship.towed_count, 10.0);
    out_features[45] = (nn_real)sm_scale(view.nearest_mine_dist, 6000.0);
    out_features[46] = (nn_real)sm_scale(view.nearest_clear_mine_dist, 6000.0);
    out_features[47] = (nn_real)sm_scale(view.richest_mine_dist, 6000.0);
    out_features[48] = (nn_real)sm_scale(view.nearest_fragment_dist, 4000.0);
    out_features[49] = (nn_real)sm_scale(view.refinery_dist, 9000.0);
    out_features[50] = (nn_real)sm_scale(view.dock_dist, 9000.0);
    out_features[51] = (nn_real)sm_scale(view.target_dist, 9000.0);
    out_features[52] = (nn_real)sm_clip(view.target_alignment_cos, -1.0, 1.0);
    out_features[53] = (nn_real)sm_clip(view.target_alignment_sin, -1.0, 1.0);
    out_features[54] = (nn_real)sm_scale(view.target_ore, 80.0);
    out_features[55] = (nn_real)sm_scale(view.target_hp, 260.0);
    out_features[56] = (nn_real)sm_scale(view.target_radius, 220.0);
    out_features[57] = (nn_real)sm_scale(view.target_grade, 4.0);
    out_features[58] = (nn_real)view.target_clear;
    out_features[59] = (nn_real)(option == SM_OPT_MINE_CLEAR || option == SM_OPT_MINE_RICH);
    out_features[60] = (nn_real)(option == SM_OPT_DELIVER_REFINERY);
    out_features[61] = (nn_real)(option == SM_OPT_DOCK_REPAIR);
    out_features[62] = (nn_real)sm_clip(view.raw_ore_need, 0.0, 1.0);
    out_features[63] = (nn_real)sm_scale(view.repair_kit_level, 100.0);
    out_features[64] = (nn_real)sm_scale((double)view.local_fragment_count, 8.0);
    out_features[65] = (nn_real)sm_scale((double)view.mineable_count, 40.0);
    out_features[66] = (nn_real)(hull_ratio < 0.82 ? 1.0 : 0.0);
    out_features[67] = (nn_real)(sp->ship.towed_count > 0 ? 1.0 : 0.0);
    out_features[68] = (nn_real)sm_scale((double)init->prefix_ticks, 7200.0);
    out_features[69] = (nn_real)sm_scale((double)init->scenario_kind, 7.0);
}

static size_t sm_fixed_priority_baseline(const nn_real features[SM_OPTION_COUNT][SM_FEATURE_COUNT])
{
    double hull_ratio = (double)features[0][38];
    double has_tow = (double)features[0][67];
    double local_fragments = (double)features[0][64];
    if (hull_ratio < 0.55) return 10u;
    if (has_tow > 0.5) return 8u;
    if (local_fragments > 0.0) return 6u;
    return 1u;
}

static int sm_select_target_for_option(const world_t *w,
                                       const server_player_t *sp,
                                       size_t candidate)
{
    sm_option_t option = sm_hypothesis(candidate)->option;
    if (option == SM_OPT_MINE_CLEAR) return sm_find_nearest_mineable(w, sp, true);
    if (option == SM_OPT_MINE_RICH) return sm_find_richest_mineable(w, sp);
    return -1;
}

static void sm_trace_features(FILE *fp, const nn_real row[SM_FEATURE_COUNT])
{
    fputc('[', fp);
    for (size_t i = 0; i < SM_FEATURE_COUNT; ++i) {
        if (i > 0) fputc(',', fp);
        fprintf(fp, "%.6g", (double)row[i]);
    }
    fputc(']', fp);
}

static void sm_write_trace_row(FILE *fp,
                               const char *split,
                               const sm_group_t *group,
                               size_t option)
{
    if (fp == NULL || group == NULL || option >= SM_OPTION_COUNT) return;
    const sm_policy_hypothesis_t *hyp = sm_hypothesis(option);
    const sm_outcome_t *out = &group->outcomes[option];
    fprintf(fp, "{\"schema\":");
    sm_json_string(fp, SM_TRACE_SCHEMA);
    fprintf(fp, ",\"task\":");
    sm_json_string(fp, SM_TASK_NAME);
    fprintf(fp, ",\"split\":");
    sm_json_string(fp, split);
    fprintf(fp, ",\"group_id\":%" PRIu64, group->group_id);
    fprintf(fp, ",\"world_seed\":%u", group->init.world_seed);
    fprintf(fp, ",\"scenario\":");
    sm_json_string(fp, sm_scenario_name(group->init.scenario_kind));
    fprintf(fp, ",\"prefix_ticks\":%d", group->init.prefix_ticks);
    fprintf(fp, ",\"candidate_id\":%zu", option);
    fprintf(fp, ",\"candidate\":");
    sm_json_string(fp, hyp->name);
    fprintf(fp, ",\"base_option\":%d", (int)hyp->option);
    fprintf(fp, ",\"target_rule\":");
    sm_json_string(fp, hyp->target_rule);
    fprintf(fp, ",\"route_rule\":%d", (int)hyp->route_rule);
    fprintf(fp, ",\"speed_rule\":%d", (int)hyp->speed_rule);
    fprintf(fp, ",\"interaction_rule\":%d", (int)hyp->interaction_rule);
    fprintf(fp, ",\"opportunistic\":%s", hyp->opportunistic ? "true" : "false");
    fprintf(fp, ",\"station_legal\":%s", hyp->station_legal ? "true" : "false");
    fprintf(fp, ",\"certified\":true");
    fprintf(fp, ",\"is_oracle\":%s", option == group->target ? "true" : "false");
    fprintf(fp, ",\"is_baseline\":%s", option == group->baseline ? "true" : "false");
    fprintf(fp, ",\"utility\":%.9g", out->utility);
    fprintf(fp, ",\"smelts\":%d", out->smelts);
    fprintf(fp, ",\"pickups\":%d", out->pickups);
    fprintf(fp, ",\"fractures\":%d", out->fractures);
    fprintf(fp, ",\"mining_ticks\":%d", out->mining_ticks);
    fprintf(fp, ",\"peak_towed\":%d", out->peak_towed);
    fprintf(fp, ",\"final_towed\":%d", out->final_towed);
    fprintf(fp, ",\"final_docked\":%d", out->final_docked);
    fprintf(fp, ",\"station_ingots_delta\":%d", out->station_ingots_delta);
    fprintf(fp, ",\"repair_kits_delta\":%d", out->repair_kits_delta);
    fprintf(fp, ",\"start_repair_kits\":%d", out->start_repair_kits);
    fprintf(fp, ",\"final_repair_kits\":%d", out->final_repair_kits);
    fprintf(fp, ",\"final_ship_repair_kits\":%d", out->final_ship_repair_kits);
    fprintf(fp, ",\"final_ship_shipyard_inputs\":%d", out->final_ship_shipyard_inputs);
    fprintf(fp, ",\"shipyard_inputs_delta\":%d", out->shipyard_inputs_delta);
    fprintf(fp, ",\"repair_dock_kits_delta\":%d", out->repair_dock_kits_delta);
    fprintf(fp, ",\"hull_gain\":%.9g", out->hull_gain);
    fprintf(fp, ",\"dock_progress\":%.9g", out->dock_progress);
    fprintf(fp, ",\"credits_delta\":%.9g", out->credits_delta);
    fprintf(fp, ",\"hull_loss\":%.9g", out->hull_loss);
    fprintf(fp, ",\"damage_amount\":%.9g", out->damage_amount);
    fprintf(fp, ",\"damage_events\":%d", out->damage_events);
    fprintf(fp, ",\"death_events\":%d", out->death_events);
    fprintf(fp, ",\"features\":");
    sm_trace_features(fp, group->features[option]);
    fputs("}\n", fp);
}

static int sm_evaluate_group(const sm_config_t *config,
                             const char *split,
                             uint64_t group_id,
                             const sm_scenario_init_t *init,
                             FILE *trace_fp,
                             sm_group_t *group)
{
    if (config == NULL || init == NULL || group == NULL) return 0;
    memset(group, 0, sizeof(*group));
    group->group_id = group_id;
    group->init = *init;
    group->target = 0u;
    group->baseline = 0u;
    group->second_best_utility = -1e18;

    world_t *base_world = calloc(1u, sizeof(*base_world));
    if (base_world == NULL) return 0;
    server_player_t *base_sp = NULL;
    sm_setup_world(init, base_world, &base_sp);
    sm_run_prefix(init, base_world, base_sp);
    if (init->enable_npcs) {
        sm_seed_supply_chain_fixture(base_world, base_sp);
    }

    int sticky_targets[SM_OPTION_COUNT];
    for (size_t option = 0; option < SM_OPTION_COUNT; ++option) {
        sticky_targets[option] = sm_select_target_for_option(base_world, base_sp, option);
        sm_encode_features(init, base_world, base_sp, option, group->features[option]);
    }
    group->baseline = sm_fixed_priority_baseline(group->features);

    double best = -1e18;
    double second = -1e18;
    for (size_t option = 0; option < SM_OPTION_COUNT; ++option) {
        if (!sm_replay_option(config,
                              init,
                              base_world,
                              option,
                              sticky_targets[option],
                              &group->outcomes[option])) {
            sm_cleanup_world(base_world);
            free(base_world);
            return 0;
        }
        double utility = group->outcomes[option].utility;
        if (utility > best) {
            second = best;
            best = utility;
            group->target = option;
        } else if (utility > second) {
            second = utility;
        }
    }
    group->second_best_utility = isfinite(second) ? second : best;
    sm_cleanup_world(base_world);
    free(base_world);

    for (size_t option = 0; option < SM_OPTION_COUNT; ++option) {
        sm_write_trace_row(trace_fp, split, group, option);
    }
    return 1;
}

static int sm_generate_dataset(const sm_config_t *config,
                               const char *split,
                               uint32_t split_seed,
                               size_t count,
                               FILE *trace_fp,
                               sm_dataset_t *dataset)
{
    if (config == NULL || split == NULL || dataset == NULL) return 0;
    memset(dataset, 0, sizeof(*dataset));
    dataset->groups = calloc(count, sizeof(*dataset->groups));
    if (dataset->groups == NULL && count > 0u) return 0;
    dataset->count = count;

    uint64_t split_tag = strcmp(split, "train") == 0 ? UINT64_C(0x1000000000000000)
                                                     : UINT64_C(0x2000000000000000);
    for (size_t i = 0; i < count; ++i) {
        sm_scenario_init_t init;
        sm_init_scenario(config, split_seed, i, &init);
        uint64_t group_id = split_tag | (uint64_t)i;
        if (!sm_evaluate_group(config, split, group_id, &init, trace_fp, &dataset->groups[i])) {
            return 0;
        }
    }
    return 1;
}

static void sm_dataset_free(sm_dataset_t *dataset)
{
    if (dataset == NULL) return;
    free(dataset->groups);
    dataset->groups = NULL;
    dataset->count = 0u;
}

static NeuralNet *sm_create_model(uint32_t seed)
{
    static const size_t layers[] = {SM_FEATURE_COUNT, 32u, 16u, 1u};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int sm_train_model(NeuralNet *net, const sm_dataset_t *train, const sm_config_t *config)
{
    if (net == NULL || train == NULL || config == NULL || train->count == 0u) return 0;
    uint32_t rng = config->seed ^ UINT32_C(0x6d696e65);
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss_sum = 0.0;
        for (size_t step = 0; step < train->count; ++step) {
            size_t idx = (size_t)(sm_rng_next(&rng) % (uint32_t)train->count);
            const sm_group_t *group = &train->groups[idx];
            double gap = group->outcomes[group->target].utility - group->second_best_utility;
            nn_real weight = (nn_real)(1.0 + sm_clip(gap, 0.0, 8.0));
            nn_real loss = nn_train_softmax_scorer_weighted(net,
                                                            &group->features[0][0],
                                                            SM_OPTION_COUNT,
                                                            group->target,
                                                            weight,
                                                            config->learning_rate);
            if (!isfinite((double)loss)) return 0;
            loss_sum += (double)loss;
        }
        printf("signal_mining counterfactual epoch %zu loss=%.4f groups=%zu\n",
               epoch + 1u,
               loss_sum / (double)train->count,
               train->count);
    }
    return 1;
}

static sm_metrics_t sm_evaluate_model(NeuralNet *net, const sm_dataset_t *test)
{
    sm_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    if (net == NULL || test == NULL || test->count == 0u) return metrics;

    nn_real scores[SM_OPTION_COUNT];
    nn_real probabilities[SM_OPTION_COUNT];
    size_t ranked[SM_OPTION_COUNT];
    for (size_t i = 0; i < test->count; ++i) {
        const sm_group_t *group = &test->groups[i];
        if (!nn_score_action_rows(net,
                                  &group->features[0][0],
                                  SM_OPTION_COUNT,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, SM_OPTION_COUNT, ranked)) {
            continue;
        }
        size_t selected = ranked[0];
        size_t rank = SM_OPTION_COUNT;
        for (size_t j = 0; j < SM_OPTION_COUNT; ++j) {
            if (ranked[j] == group->target) {
                rank = j;
                break;
            }
        }
        if (rank >= SM_OPTION_COUNT) continue;

        const sm_outcome_t *sel = &group->outcomes[selected];
        const sm_outcome_t *oracle = &group->outcomes[group->target];
        const sm_outcome_t *base = &group->outcomes[group->baseline];
        ++metrics.groups;
        metrics.top1 += rank == 0 ? 1u : 0u;
        metrics.top3 += rank < 3u ? 1u : 0u;
        metrics.selected_utility += sel->utility;
        metrics.oracle_utility += oracle->utility;
        metrics.regret += oracle->utility - sel->utility;
        metrics.baseline_utility += base->utility;
        metrics.baseline_regret += oracle->utility - base->utility;
        metrics.selected_damage += (sel->damage_events > 0 || sel->hull_loss > 0.01) ? 1u : 0u;
        metrics.selected_death += sel->death_events > 0 ? 1u : 0u;
        metrics.selected_smelt += sel->smelts > 0 ? 1u : 0u;
        metrics.selected_tow += (sel->peak_towed > 0 || sel->pickups > 0) ? 1u : 0u;
        metrics.baseline_damage += (base->damage_events > 0 || base->hull_loss > 0.01) ? 1u : 0u;
        metrics.baseline_death += base->death_events > 0 ? 1u : 0u;
        metrics.baseline_smelt += base->smelts > 0 ? 1u : 0u;
        metrics.baseline_tow += (base->peak_towed > 0 || base->pickups > 0) ? 1u : 0u;
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

static int sm_save_checkpoint(NeuralNet *net,
                              const char *path,
                              const char *trace_path,
                              char out_hash[65])
{
    if (net == NULL || path == NULL) return 0;
    NNCheckpointMeta meta;
    nn_checkpoint_meta_init(&meta, 3u);
    if (!nn_checkpoint_meta_set_parent_model_hash(&meta, "none") ||
        !nn_checkpoint_meta_set_benchmark_report_hash(&meta, "pending") ||
        !nn_checkpoint_meta_set_training_trace_root(
            &meta,
            trace_path != NULL ? trace_path : "generated_in_memory") ||
        !nn_checkpoint_meta_set_notes(
            &meta,
            "Signal mining/tow/smelt/supply grammar scorer trained from replay-certified utility") ||
        !nn_checkpoint_meta_set_feature_set(&meta, "signal-mining-grammar-v1")) {
        return 0;
    }
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static double sm_rate(size_t numerator, size_t denominator)
{
    return denominator == 0u ? 0.0 : (double)numerator / (double)denominator;
}

static void sm_write_option_counts(FILE *fp, const size_t counts[SM_OPTION_COUNT])
{
    fputc('{', fp);
    for (size_t i = 0; i < SM_OPTION_COUNT; ++i) {
        if (i > 0) fputc(',', fp);
        sm_json_string(fp, sm_hypothesis(i)->name);
        fprintf(fp, ":%zu", counts[i]);
    }
    fputc('}', fp);
}

static void sm_write_summary(FILE *fp,
                             const sm_config_t *config,
                             const char *model_hash,
                             const NeuralNet *net,
                             const sm_dataset_t *train,
                             const sm_dataset_t *test,
                             const sm_metrics_t *metrics)
{
    fprintf(fp, "{\"schema\":");
    sm_json_string(fp, SM_SUMMARY_SCHEMA);
    fprintf(fp, ",\"task\":");
    sm_json_string(fp, SM_TASK_NAME);
    fprintf(fp, ",\"trace_schema\":");
    sm_json_string(fp, SM_TRACE_SCHEMA);
    fprintf(fp, ",\"training_mode\":\"counterfactual_grammar_utility\"");
    fprintf(fp, ",\"candidate_layer\":\"mine_tow_smelt_supply_grammar_v1\"");
    fprintf(fp, ",\"model_hash\":");
    sm_json_string(fp, model_hash != NULL ? model_hash : "unsaved");
    fprintf(fp, ",\"trace_path\":");
    sm_json_string(fp, config->trace_path);
    fprintf(fp, ",\"train_groups\":%zu", train != NULL ? train->count : 0u);
    fprintf(fp, ",\"test_groups\":%zu", test != NULL ? test->count : 0u);
    fprintf(fp, ",\"candidate_count\":%u", SM_OPTION_COUNT);
    fprintf(fp, ",\"horizon_ticks\":%d", config->horizon_ticks);
    fprintf(fp, ",\"epochs\":%zu", config->epochs);
    fprintf(fp, ",\"learning_rate\":%.9g", (double)config->learning_rate);
    fprintf(fp, ",\"model_parameters\":%zu", net != NULL ? nn_parameter_count(net) : 0u);
    fprintf(fp, ",\"model_forward_flops\":%zu", net != NULL ? nn_forward_flop_count(net) : 0u);
    fprintf(fp, ",\"top1\":%.9g", sm_rate(metrics->top1, metrics->groups));
    fprintf(fp, ",\"top3\":%.9g", sm_rate(metrics->top3, metrics->groups));
    fprintf(fp, ",\"utility_groups\":%zu", metrics->groups);
    fprintf(fp, ",\"selected_utility\":%.9g", metrics->selected_utility);
    fprintf(fp, ",\"oracle_utility\":%.9g", metrics->oracle_utility);
    fprintf(fp, ",\"regret\":%.9g", metrics->regret);
    fprintf(fp, ",\"baseline_utility\":%.9g", metrics->baseline_utility);
    fprintf(fp, ",\"baseline_regret\":%.9g", metrics->baseline_regret);
    fprintf(fp, ",\"selected_damage_rate\":%.9g", sm_rate(metrics->selected_damage, metrics->groups));
    fprintf(fp, ",\"selected_death_rate\":%.9g", sm_rate(metrics->selected_death, metrics->groups));
    fprintf(fp, ",\"selected_smelt_rate\":%.9g", sm_rate(metrics->selected_smelt, metrics->groups));
    fprintf(fp, ",\"selected_tow_rate\":%.9g", sm_rate(metrics->selected_tow, metrics->groups));
    fprintf(fp, ",\"baseline_damage_rate\":%.9g", sm_rate(metrics->baseline_damage, metrics->groups));
    fprintf(fp, ",\"baseline_death_rate\":%.9g", sm_rate(metrics->baseline_death, metrics->groups));
    fprintf(fp, ",\"baseline_smelt_rate\":%.9g", sm_rate(metrics->baseline_smelt, metrics->groups));
    fprintf(fp, ",\"baseline_tow_rate\":%.9g", sm_rate(metrics->baseline_tow, metrics->groups));
    fprintf(fp, ",\"target_options\":");
    sm_write_option_counts(fp, metrics->option_target);
    fprintf(fp, ",\"selected_options\":");
    sm_write_option_counts(fp, metrics->option_selected);
    fputs("}\n", fp);
}

static int sm_write_summary_file(const char *path,
                                 const sm_config_t *config,
                                 const char *model_hash,
                                 const NeuralNet *net,
                                 const sm_dataset_t *train,
                                 const sm_dataset_t *test,
                                 const sm_metrics_t *metrics)
{
    if (path == NULL) return 1;
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open Signal mining summary for write: %s\n", path);
        return 0;
    }
    sm_write_summary(fp, config, model_hash, net, train, test, metrics);
    if (fclose(fp) != 0) return 0;
    return 1;
}

static void sm_print_usage(FILE *stream)
{
    fputs("usage: signal_mining [--schema] [--train-groups N] [--test-groups N] "
          "[--horizon-ticks N] [--epochs N] [--learning-rate R] [--seed N] "
          "[--trace path] [--summary path] [--save path]\n",
          stream);
}

static int sm_parse_config(int argc, char **argv, sm_config_t *config, int *schema_only)
{
    if (config == NULL || schema_only == NULL) return 0;
    memset(config, 0, sizeof(*config));
    config->seed = 4242u;
    config->train_groups = 24u;
    config->test_groups = 8u;
    config->horizon_ticks = 3600;
    config->epochs = 4u;
    config->learning_rate = (nn_real)0.004;
    *schema_only = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schema") == 0) {
            *schema_only = 1;
        } else if (strcmp(argv[i], "--train-groups") == 0 && i + 1 < argc) {
            if (!sm_parse_size(argv[++i], &config->train_groups)) return 0;
        } else if (strcmp(argv[i], "--test-groups") == 0 && i + 1 < argc) {
            if (!sm_parse_size(argv[++i], &config->test_groups)) return 0;
        } else if (strcmp(argv[i], "--horizon-ticks") == 0 && i + 1 < argc) {
            if (!sm_parse_int(argv[++i], &config->horizon_ticks)) return 0;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!sm_parse_size(argv[++i], &config->epochs)) return 0;
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            if (!sm_parse_real(argv[++i], &config->learning_rate)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!sm_parse_u32(argv[++i], &config->seed)) return 0;
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--summary") == 0 && i + 1 < argc) {
            config->summary_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            sm_print_usage(stdout);
            exit(EXIT_SUCCESS);
        } else {
            return 0;
        }
    }

    if (*schema_only) return 1;
    return config->train_groups > 0u && config->test_groups > 0u &&
           config->horizon_ticks > 0 && config->epochs > 0u;
}


int main(int argc, char **argv)
{
    sm_config_t config;
    int schema_only = 0;
    if (!sm_parse_config(argc, argv, &config, &schema_only)) {
        sm_print_usage(stderr);
        return EXIT_FAILURE;
    }
    if (schema_only) {
        return nn_experiment_write_schema_json(stdout, &SM_EXPERIMENT_SPEC)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }

    chain_log_set_disk_enabled(false);
    signal_chain_set_disk_enabled(false);

    FILE *trace_fp = NULL;
    if (config.trace_path != NULL) {
        trace_fp = fopen(config.trace_path, "w");
        if (trace_fp == NULL) {
            fprintf(stderr, "failed to open Signal mining trace for write: %s\n", config.trace_path);
            return EXIT_FAILURE;
        }
    }

    sm_dataset_t train;
    sm_dataset_t test;
    memset(&train, 0, sizeof(train));
    memset(&test, 0, sizeof(test));
    if (!sm_generate_dataset(&config,
                             "train",
                             config.seed ^ UINT32_C(0x74726169),
                             config.train_groups,
                             trace_fp,
                             &train) ||
        !sm_generate_dataset(&config,
                             "test",
                             config.seed ^ UINT32_C(0x74657374),
                             config.test_groups,
                             trace_fp,
                             &test)) {
        if (trace_fp != NULL) fclose(trace_fp);
        sm_dataset_free(&train);
        sm_dataset_free(&test);
        return EXIT_FAILURE;
    }
    if (trace_fp != NULL && fclose(trace_fp) != 0) {
        sm_dataset_free(&train);
        sm_dataset_free(&test);
        return EXIT_FAILURE;
    }

    NeuralNet *net = sm_create_model(config.seed ^ UINT32_C(0x4d494e45));
    if (net == NULL) {
        sm_dataset_free(&train);
        sm_dataset_free(&test);
        return EXIT_FAILURE;
    }
    if (!sm_train_model(net, &train, &config)) {
        nn_free(net);
        sm_dataset_free(&train);
        sm_dataset_free(&test);
        return EXIT_FAILURE;
    }

    char model_hash[65] = "unsaved";
    if (config.save_path != NULL) {
        if (!sm_save_checkpoint(net, config.save_path, config.trace_path, model_hash)) {
            fprintf(stderr, "failed to save Signal mining checkpoint: %s\n", config.save_path);
            nn_free(net);
            sm_dataset_free(&train);
            sm_dataset_free(&test);
            return EXIT_FAILURE;
        }
        printf("saved signal_mining checkpoint: %s hash=%s\n", config.save_path, model_hash);
    }

    sm_metrics_t metrics = sm_evaluate_model(net, &test);
    sm_write_summary(stdout, &config, model_hash, net, &train, &test, &metrics);
    if (!sm_write_summary_file(config.summary_path,
                               &config,
                               model_hash,
                               net,
                               &train,
                               &test,
                               &metrics)) {
        nn_free(net);
        sm_dataset_free(&train);
        sm_dataset_free(&test);
        return EXIT_FAILURE;
    }

    nn_free(net);
    sm_dataset_free(&train);
    sm_dataset_free(&test);
    return metrics.groups == 0u ? EXIT_FAILURE : EXIT_SUCCESS;
}
