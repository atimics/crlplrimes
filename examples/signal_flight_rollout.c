#include "nn.h"

#include "game_sim.h"
#include "manifest.h"
#include "protocol.h"
#include "sim_flight.h"
#include "sim_nav.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SFR_FEATURE_COUNT 48u
#define SFR_ACTION_COUNT 9u
#define SFR_HYPOTHESIS_COUNT 5u
#define SFR_SIM_DT (1.0f / 120.0f)
#define SFR_SUMMARY_SCHEMA "crlp.signal_flight_rollout.summary.v1"

typedef enum {
    SFR_CONTROLLER_GOAL = 0,
    SFR_CONTROLLER_HEURISTIC = 1,
    SFR_CONTROLLER_NEURAL_RAW = 2,
    SFR_CONTROLLER_NEURAL_GUARDED = 3,
    SFR_CONTROLLER_COUNT = 4,
} sfr_controller_t;

typedef struct {
    int turn;
    int thrust;
    const char *name;
} sfr_action_t;

typedef enum {
    SFR_HYPOTHESIS_DIRECT = 0,
    SFR_HYPOTHESIS_NAV_PATH = 1,
    SFR_HYPOTHESIS_LEFT_BYPASS = 2,
    SFR_HYPOTHESIS_RIGHT_BYPASS = 3,
    SFR_HYPOTHESIS_BRAKE_ESCAPE = 4,
} sfr_hypothesis_kind_t;

typedef struct {
    sfr_hypothesis_kind_t kind;
    vec2 control_target;
    nav_path_t path;
    int certified;
    double certificate_score;
    double alignment;
    double clearance;
} sfr_hypothesis_t;

typedef struct {
    const char *checkpoint_path;
    const char *summary_path;
    int episodes;
    int ticks;
    uint32_t seed;
    int include_wall_probes;
    int controller_enabled[SFR_CONTROLLER_COUNT];
} sfr_config_t;

typedef struct {
    uint32_t world_seed;
    int station_index;
    vec2 spawn_pos;
    vec2 spawn_vel;
    float spawn_angle;
    vec2 goal;
    const char *scenario;
} sfr_episode_init_t;

typedef struct {
    double dist;
    double control_dist;
    double heading_cos;
    double heading_sin;
    double control_heading_cos;
    double control_heading_sin;
    double speed;
    double forward_speed;
    double lateral_speed;
    double brake_distance;
    double fwd_clear;
    double left_clear;
    double right_clear;
    double vel_clear;
    double signal;
    double hull_ratio;
    double path_count;
    double path_current;
    double progress;
    double next_dist;
    double next_speed;
} sfr_state_t;

typedef struct {
    int reached;
    int died;
    int stuck;
    int ticks;
    double start_dist;
    double final_dist;
    double min_dist;
    double progress;
    double damage_amount;
    double hull_loss;
    int damage_events;
    int death_events;
    double min_hull_ratio;
    double utility;
    int hypothesis_ticks;
    int hypothesis_candidates;
    int hypothesis_certified_candidates;
    int hypothesis_selected_certified;
    int hypothesis_selected[SFR_HYPOTHESIS_COUNT];
} sfr_episode_result_t;

typedef struct {
    int episodes;
    int reached;
    int died;
    int stuck;
    int damage_episodes;
    int total_ticks;
    int damage_events;
    int death_events;
    double start_dist;
    double final_dist;
    double min_dist;
    double progress;
    double damage_amount;
    double hull_loss;
    double min_hull_ratio_sum;
    double utility;
    int hypothesis_ticks;
    int hypothesis_candidates;
    int hypothesis_certified_candidates;
    int hypothesis_selected_certified;
    int hypothesis_selected[SFR_HYPOTHESIS_COUNT];
} sfr_summary_t;

static const sfr_action_t SFR_ACTIONS[SFR_ACTION_COUNT] = {
    {0, 0, "NONE"},
    {0, 1, "W"},
    {-1, 0, "A"},
    {1, 0, "D"},
    {0, -1, "S"},
    {-1, 1, "WA"},
    {1, 1, "WD"},
    {-1, -1, "SA"},
    {1, -1, "SD"},
};

static const char *sfr_controller_name(sfr_controller_t controller)
{
    switch (controller) {
    case SFR_CONTROLLER_GOAL: return "goal_heading_no_safety";
    case SFR_CONTROLLER_HEURISTIC: return "flight_steer_to";
    case SFR_CONTROLLER_NEURAL_RAW: return "neural_raw";
    case SFR_CONTROLLER_NEURAL_GUARDED: return "neural_guarded";
    default: return "unknown";
    }
}

static uint32_t sfr_rng_next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x ? x : 0x9e3779b9u;
    return *state;
}

static double sfr_rng_unit(uint32_t *state)
{
    return (double)(sfr_rng_next(state) >> 8) / 16777215.0;
}

static float sfr_rng_range(uint32_t *state, float lo, float hi)
{
    return lo + (hi - lo) * (float)sfr_rng_unit(state);
}

static double sfr_clip(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static double sfr_scale(double x, double denom)
{
    if (denom <= 0.0) return 0.0;
    return sfr_clip(x / denom, 0.0, 1.0);
}

static double sfr_signed_scale(double x, double denom)
{
    if (denom <= 0.0) return 0.0;
    return sfr_clip(x / denom, -1.0, 1.0);
}

static vec2 sfr_random_station_nearby(uint32_t *rng,
                                      const station_t *station,
                                      float min_radius,
                                      float max_radius)
{
    float angle = sfr_rng_range(rng, -PI_F, PI_F);
    float radius = sfr_rng_range(rng, min_radius, max_radius);
    return v2_add(station->pos, v2(cosf(angle) * radius, sinf(angle) * radius));
}

static float sfr_wall_clearance(const world_t *w,
                                vec2 pos,
                                vec2 vel,
                                float ship_radius,
                                float heading)
{
    vec2 fwd = v2(cosf(heading), sinf(heading));
    vec2 perp = v2(-fwd.y, fwd.x);
    float speed = sqrtf(v2_len_sq(vel));
    float lookahead = fmaxf(100.0f, fminf(speed * 1.5f, 500.0f));
    float worst = 1.0f;
    int station_count = w->station_count > 0 ? w->station_count : 3;

    for (int i = 0; i < station_count && i < MAX_STATIONS; i++) {
        const station_t *st = &w->stations[i];
        if (st->module_count <= 0 && st->radius <= 0.0f && st->dock_radius <= 0.0f) {
            continue;
        }

        float clearance = fmaxf(850.0f,
                                fmaxf(st->radius + 360.0f,
                                      st->dock_radius + 252.0f)) + ship_radius;
        vec2 to_station = v2_sub(st->pos, pos);
        float fd = v2_dot(to_station, fwd);
        if (fd < -clearance || fd > lookahead) continue;
        float lateral = fabsf(v2_dot(to_station, perp));
        if (lateral >= clearance) continue;

        float along = sqrtf(fmaxf(0.0f, clearance * clearance - lateral * lateral));
        float entry = fd - along;
        float exit_dist = fd + along;
        if (entry >= 0.0f) {
            worst = fminf(worst, entry / lookahead);
        } else if (exit_dist >= 0.0f) {
            worst = 0.0f;
        }
    }

    return fmaxf(0.0f, fminf(1.0f, worst));
}

static float sfr_forward_clearance(const world_t *w,
                                   vec2 pos,
                                   vec2 vel,
                                   float ship_radius,
                                   float heading)
{
    float asteroid_clearance = nav_forward_clearance(w, pos, vel, ship_radius, heading);
    float wall_clearance = sfr_wall_clearance(w, pos, vel, ship_radius, heading);
    return fminf(asteroid_clearance, wall_clearance);
}

static vec2 sfr_safe_unit(vec2 v, vec2 fallback)
{
    float len = sqrtf(v2_len_sq(v));
    if (len <= 0.001f) {
        return fallback;
    }
    return v2_scale(v, 1.0f / len);
}

static void sfr_set_single_waypoint(nav_path_t *path, vec2 target, vec2 final_goal)
{
    memset(path, 0, sizeof(*path));
    path->waypoints[0] = target;
    path->count = 1;
    path->current = 0;
    path->goal = final_goal;
}

static size_t sfr_action_from_controls(int turn, int thrust)
{
    for (size_t i = 0; i < SFR_ACTION_COUNT; i++) {
        if (SFR_ACTIONS[i].turn == turn && SFR_ACTIONS[i].thrust == thrust) {
            return i;
        }
    }
    return 0u;
}

static size_t sfr_quantize_flight_cmd(flight_cmd_t cmd)
{
    int turn = 0;
    int thrust = 0;
    if (cmd.turn < -0.25f) turn = -1;
    else if (cmd.turn > 0.25f) turn = 1;
    if (cmd.thrust < -0.25f) thrust = -1;
    else if (cmd.thrust > 0.25f) thrust = 1;
    return sfr_action_from_controls(turn, thrust);
}

static size_t sfr_goal_heading_action(const sfr_state_t *state)
{
    int turn = 0;
    int thrust = 0;
    if (state->heading_sin < -0.18) turn = -1;
    else if (state->heading_sin > 0.18) turn = 1;
    if (state->heading_cos > 0.73) thrust = 1;
    return sfr_action_from_controls(turn, thrust);
}

static void sfr_apply_action(server_player_t *sp, size_t action_id)
{
    const sfr_action_t *action = &SFR_ACTIONS[action_id < SFR_ACTION_COUNT ? action_id : 0u];
    memset(&sp->input, 0, sizeof(sp->input));
    sp->input.turn = (float)action->turn;
    sp->input.thrust = (float)action->thrust;
    sp->input.reverse_thrust = false;
}

static void sfr_reset_player(world_t *w, server_player_t *sp)
{
    ship_cleanup(&sp->ship);
    memset(sp, 0, sizeof(*sp));
    player_init_ship(sp, w);
    sp->id = 0;
    sp->connected = true;
    sp->docked = false;
    sp->autopilot_mode = 0;
    sp->was_in_signal = true;
    sp->boost_hold_timer = 0.0f;
    memset(&sp->input, 0, sizeof(sp->input));
}

static void sfr_setup_world(const sfr_episode_init_t *init,
                            world_t *w,
                            server_player_t **out_sp)
{
    memset(w, 0, sizeof(*w));
    w->rng = init->world_seed;
    world_reset(w);
    for (int n = 0; n < MAX_NPC_SHIPS; n++) {
        w->npc_ships[n].active = false;
    }
    server_player_t *sp = &w->players[0];
    sfr_reset_player(w, sp);
    sp->ship.pos = init->spawn_pos;
    sp->ship.vel = init->spawn_vel;
    sp->ship.angle = init->spawn_angle;
    sp->ship.hull = ship_max_hull(&sp->ship);
    sp->ship.towed_count = 0;
    sp->ship.towed_scaffold = -1;
    sp->docked = false;
    memset(&sp->input, 0, sizeof(sp->input));
    *out_sp = sp;
}

static void sfr_init_episode(const sfr_config_t *config,
                             int episode,
                             uint32_t *rng,
                             sfr_episode_init_t *init)
{
    world_t template_world = {0};
    uint32_t world_seed =
        config->seed ^ ((uint32_t)episode * 0x9e3779b9u) ^ 0x51f11a7eu;
    memset(init, 0, sizeof(*init));
    template_world.rng = world_seed;
    world_reset(&template_world);
    init->world_seed = world_seed;
    init->station_index = (int)(sfr_rng_next(rng) % 3u);
    station_t *station = &template_world.stations[init->station_index];

    if (config->include_wall_probes && (episode % 3) == 2) {
        float outer = STATION_RING_RADIUS[STATION_NUM_RINGS] + 100.0f;
        float wall_edge = STATION_RING_RADIUS[STATION_NUM_RINGS] + 58.0f;
        float angle = sfr_rng_range(rng, -PI_F, PI_F);
        float goal_angle = angle + PI_F + sfr_rng_range(rng, -0.55f, 0.55f);
        float spawn_radius = sfr_rng_range(rng, wall_edge - 10.0f, wall_edge + 70.0f);
        float goal_radius = sfr_rng_range(rng, outer + 700.0f, outer + 1800.0f);
        vec2 spawn_dir = v2(cosf(angle), sinf(angle));
        vec2 goal_dir = v2(cosf(goal_angle), sinf(goal_angle));
        init->spawn_pos = v2_add(station->pos, v2_scale(spawn_dir, spawn_radius));
        init->goal = v2_add(station->pos, v2_scale(goal_dir, goal_radius));
        vec2 to_goal = v2_sub(init->goal, init->spawn_pos);
        init->spawn_angle = atan2f(to_goal.y, to_goal.x);
        init->spawn_vel = v2_scale(ship_forward(init->spawn_angle),
                                   sfr_rng_range(rng, 140.0f, 240.0f));
        init->scenario = "station_wall";
    } else {
        init->spawn_pos = sfr_random_station_nearby(rng, station, 800.0f, 4200.0f);
        init->spawn_vel = v2(sfr_rng_range(rng, -70.0f, 70.0f),
                             sfr_rng_range(rng, -70.0f, 70.0f));
        init->spawn_angle = sfr_rng_range(rng, -PI_F, PI_F);
        init->goal = sfr_random_station_nearby(rng, station, 800.0f, 5200.0f);
        init->scenario = "transit";
    }

    world_cleanup(&template_world);
}

static double sfr_control_alignment(vec2 pos, vec2 goal, vec2 control_target)
{
    vec2 to_goal = v2_sub(goal, pos);
    vec2 to_control = v2_sub(control_target, pos);
    float goal_len = sqrtf(v2_len_sq(to_goal));
    float control_len = sqrtf(v2_len_sq(to_control));
    if (goal_len <= 0.001f || control_len <= 0.001f) {
        return 1.0;
    }
    return sfr_clip(v2_dot(to_goal, to_control) / (double)(goal_len * control_len), -1.0, 1.0);
}

static void sfr_certify_hypothesis(const world_t *w,
                                   const server_player_t *sp,
                                   vec2 goal,
                                   float clearance,
                                   sfr_hypothesis_t *hyp)
{
    vec2 pos = sp->ship.pos;
    vec2 to_control = v2_sub(hyp->control_target, pos);
    float control_dist = sqrtf(v2_len_sq(to_control));
    float control_heading = control_dist > 0.001f
                          ? atan2f(to_control.y, to_control.x)
                          : sp->ship.angle;
    float speed = sqrtf(v2_len_sq(sp->ship.vel));
    float vel_heading = speed > 0.5f
                      ? atan2f(sp->ship.vel.y, sp->ship.vel.x)
                      : sp->ship.angle;
    float fwd_clear = sfr_forward_clearance(w,
                                            pos,
                                            sp->ship.vel,
                                            clearance,
                                            control_heading);
    float vel_clear = sfr_forward_clearance(w,
                                            pos,
                                            sp->ship.vel,
                                            clearance,
                                            vel_heading);
    int segment_clear = nav_segment_clear(w, pos, hyp->control_target, clearance);
    double alignment = sfr_control_alignment(pos, goal, hyp->control_target);
    int danger = speed > 70.0f && (fwd_clear < 0.25f || vel_clear < 0.25f);

    hyp->alignment = alignment;
    hyp->clearance = fminf(fwd_clear, vel_clear);
    hyp->certified = 0;
    hyp->certificate_score = -1000.0;

    if (control_dist < 25.0f || !segment_clear) {
        return;
    }
    if (hyp->kind == SFR_HYPOTHESIS_BRAKE_ESCAPE) {
        if (!danger) {
            return;
        }
        hyp->certified = 1;
        hyp->certificate_score = 2.0 + (double)hyp->clearance - sfr_scale(speed, 300.0);
        return;
    }
    if (alignment < -0.35 || fwd_clear < 0.08f) {
        return;
    }

    hyp->certified = 1;
    hyp->certificate_score =
        alignment +
        0.55 * (double)hyp->clearance -
        0.10 * sfr_scale(control_dist, 1200.0);
    if (hyp->kind == SFR_HYPOTHESIS_DIRECT) {
        hyp->certificate_score += 0.20;
    } else if (hyp->kind == SFR_HYPOTHESIS_NAV_PATH) {
        hyp->certificate_score += 0.30;
    } else if (danger) {
        hyp->certificate_score += 0.25;
    }
}

static size_t sfr_enumerate_hypotheses(const world_t *w,
                                       const server_player_t *sp,
                                       vec2 goal,
                                       nav_path_t *nav_path,
                                       sfr_hypothesis_t *out,
                                       size_t capacity)
{
    const hull_def_t *hull = ship_hull_def(&sp->ship);
    float clearance = hull->ship_radius + 30.0f;
    vec2 pos = sp->ship.pos;
    vec2 to_goal = v2_sub(goal, pos);
    float goal_dist = sqrtf(v2_len_sq(to_goal));
    float speed = sqrtf(v2_len_sq(sp->ship.vel));
    vec2 goal_dir = sfr_safe_unit(to_goal, ship_forward(sp->ship.angle));
    vec2 side = v2(-goal_dir.y, goal_dir.x);
    float leg = fminf(fmaxf(goal_dist * 0.25f, 420.0f), 1100.0f);
    float offset = fminf(fmaxf(goal_dist * 0.18f, 360.0f), 850.0f);
    size_t count = 0;

    if (capacity < SFR_HYPOTHESIS_COUNT) {
        return 0;
    }

    out[count] = (sfr_hypothesis_t){
        .kind = SFR_HYPOTHESIS_DIRECT,
        .control_target = goal,
    };
    count++;

    nav_follow_path(w, nav_path, pos, goal, clearance, SFR_SIM_DT);
    vec2 nav_target = goal;
    if (nav_path->count > 0 && nav_path->current < nav_path->count) {
        nav_target = nav_path->waypoints[nav_path->current];
    }
    out[count] = (sfr_hypothesis_t){
        .kind = SFR_HYPOTHESIS_NAV_PATH,
        .control_target = nav_target,
        .path = *nav_path,
    };
    count++;

    out[count] = (sfr_hypothesis_t){
        .kind = SFR_HYPOTHESIS_LEFT_BYPASS,
        .control_target = v2_add(pos, v2_add(v2_scale(goal_dir, leg),
                                             v2_scale(side, offset))),
    };
    sfr_set_single_waypoint(&out[count].path, out[count].control_target, goal);
    count++;

    out[count] = (sfr_hypothesis_t){
        .kind = SFR_HYPOTHESIS_RIGHT_BYPASS,
        .control_target = v2_add(pos, v2_add(v2_scale(goal_dir, leg),
                                             v2_scale(side, -offset))),
    };
    sfr_set_single_waypoint(&out[count].path, out[count].control_target, goal);
    count++;

    vec2 brake_dir = speed > 0.5f
                   ? sfr_safe_unit(v2_scale(sp->ship.vel, -1.0f),
                                   v2_scale(ship_forward(sp->ship.angle), -1.0f))
                   : v2_scale(ship_forward(sp->ship.angle), -1.0f);
    float brake_leg = fminf(fmaxf(speed * 2.5f, 320.0f), 900.0f);
    out[count] = (sfr_hypothesis_t){
        .kind = SFR_HYPOTHESIS_BRAKE_ESCAPE,
        .control_target = v2_add(pos, v2_scale(brake_dir, brake_leg)),
    };
    sfr_set_single_waypoint(&out[count].path, out[count].control_target, goal);
    count++;

    for (size_t i = 0; i < count; i++) {
        if (out[i].kind == SFR_HYPOTHESIS_DIRECT) {
            memset(&out[i].path, 0, sizeof(out[i].path));
        } else if (out[i].kind == SFR_HYPOTHESIS_NAV_PATH && out[i].path.count == 0) {
            sfr_set_single_waypoint(&out[i].path, out[i].control_target, goal);
        }
        sfr_certify_hypothesis(w, sp, goal, clearance, &out[i]);
    }

    return count;
}

static void sfr_select_hypothesis(const world_t *w,
                                  const server_player_t *sp,
                                  vec2 goal,
                                  nav_path_t *nav_path,
                                  nav_path_t *out_path,
                                  sfr_episode_result_t *result)
{
    sfr_hypothesis_t hypotheses[SFR_HYPOTHESIS_COUNT];
    size_t count = sfr_enumerate_hypotheses(w,
                                            sp,
                                            goal,
                                            nav_path,
                                            hypotheses,
                                            SFR_HYPOTHESIS_COUNT);
    size_t selected = 0;
    int selected_set = 0;
    int certified_count = 0;

    for (size_t i = 0; i < count; i++) {
        if (hypotheses[i].certified) {
            certified_count++;
            if (!selected_set ||
                hypotheses[i].certificate_score > hypotheses[selected].certificate_score) {
                selected = i;
                selected_set = 1;
            }
        }
    }
    if (!selected_set && count > 0) {
        for (size_t i = 0; i < count; i++) {
            if (hypotheses[i].kind == SFR_HYPOTHESIS_NAV_PATH) {
                selected = i;
                break;
            }
        }
    }

    memset(out_path, 0, sizeof(*out_path));
    if (count > 0) {
        *out_path = hypotheses[selected].path;
        result->hypothesis_ticks++;
        result->hypothesis_candidates += (int)count;
        result->hypothesis_certified_candidates += certified_count;
        result->hypothesis_selected_certified += hypotheses[selected].certified ? 1 : 0;
        if ((unsigned)hypotheses[selected].kind < SFR_HYPOTHESIS_COUNT) {
            result->hypothesis_selected[hypotheses[selected].kind]++;
        }
    }
}

static void sfr_fill_state(const world_t *w,
                           const server_player_t *sp,
                           const nav_path_t *path,
                           vec2 goal,
                           sfr_state_t *state)
{
    const hull_def_t *hull = ship_hull_def(&sp->ship);
    vec2 to_goal = v2_sub(goal, sp->ship.pos);
    vec2 control_target = goal;
    if (path != NULL && path->count > 0 && path->current < path->count) {
        control_target = path->waypoints[path->current];
    }
    vec2 to_control = v2_sub(control_target, sp->ship.pos);
    vec2 forward = ship_forward(sp->ship.angle);
    vec2 right = v2(-forward.y, forward.x);
    float speed = sqrtf(v2_len_sq(sp->ship.vel));
    float max_hull = ship_max_hull(&sp->ship);
    float goal_angle = atan2f(to_goal.y, to_goal.x);
    float control_angle = atan2f(to_control.y, to_control.x);
    float heading_diff = wrap_angle(goal_angle - sp->ship.angle);
    float control_heading_diff = wrap_angle(control_angle - sp->ship.angle);

    memset(state, 0, sizeof(*state));
    state->dist = sqrtf(v2_len_sq(to_goal));
    state->control_dist = sqrtf(v2_len_sq(to_control));
    state->heading_cos = cosf(heading_diff);
    state->heading_sin = sinf(heading_diff);
    state->control_heading_cos = cosf(control_heading_diff);
    state->control_heading_sin = sinf(control_heading_diff);
    state->speed = speed;
    state->forward_speed = v2_dot(sp->ship.vel, forward);
    state->lateral_speed = v2_dot(sp->ship.vel, right);
    state->brake_distance = (speed * speed) / (2.0f * SHIP_BRAKE);
    state->fwd_clear = sfr_forward_clearance(w,
                                             sp->ship.pos,
                                             sp->ship.vel,
                                             hull->ship_radius,
                                             sp->ship.angle);
    state->left_clear = sfr_forward_clearance(w,
                                              sp->ship.pos,
                                              sp->ship.vel,
                                              hull->ship_radius,
                                              sp->ship.angle + 0.7f);
    state->right_clear = sfr_forward_clearance(w,
                                               sp->ship.pos,
                                               sp->ship.vel,
                                               hull->ship_radius,
                                               sp->ship.angle - 0.7f);
    state->vel_clear = speed > 0.5f
                     ? sfr_forward_clearance(w,
                                             sp->ship.pos,
                                             sp->ship.vel,
                                             hull->ship_radius,
                                             atan2f(sp->ship.vel.y, sp->ship.vel.x))
                     : state->fwd_clear;
    state->signal = signal_strength_at(w, sp->ship.pos);
    state->hull_ratio = max_hull > 0.0f ? sp->ship.hull / max_hull : 1.0f;
    state->path_count = path != NULL ? path->count : 0;
    state->path_current = path != NULL ? path->current : 0;
    state->progress = 0.0;
    state->next_dist = state->dist;
    state->next_speed = speed;
}

static void sfr_fill_action_row(const sfr_state_t *state, size_t action, nn_real *row)
{
    const sfr_action_t *a = &SFR_ACTIONS[action < SFR_ACTION_COUNT ? action : 0u];
    double turn = (double)a->turn;
    double thrust = (double)a->thrust;
    double dist = sfr_scale(state->dist, 6000.0);
    double control_dist = sfr_scale(state->control_dist, 6000.0);
    double speed = sfr_signed_scale(state->speed, 350.0);
    double forward_speed = sfr_signed_scale(state->forward_speed, 350.0);
    double lateral_speed = sfr_signed_scale(state->lateral_speed, 350.0);
    double brake_distance = sfr_scale(state->brake_distance, 700.0);
    double fwd_clear = sfr_clip(state->fwd_clear, 0.0, 1.0);
    double left_clear = sfr_clip(state->left_clear, 0.0, 1.0);
    double right_clear = sfr_clip(state->right_clear, 0.0, 1.0);
    double vel_clear = sfr_clip(state->vel_clear, 0.0, 1.0);
    double signal = sfr_clip(state->signal, 0.0, 1.0);
    double hull = sfr_clip(state->hull_ratio, 0.0, 1.0);
    double path_count = sfr_scale(state->path_count, 16.0);
    double path_current = sfr_scale(state->path_current, 16.0);
    double progress = 0.0;
    double next_dist = dist;
    double next_speed = speed;
    double close_goal = 1.0 - dist;
    double heading_cos = sfr_clip(state->heading_cos, -1.0, 1.0);
    double heading_sin = sfr_clip(state->heading_sin, -1.0, 1.0);
    double control_heading_cos = sfr_clip(state->control_heading_cos, -1.0, 1.0);
    double control_heading_sin = sfr_clip(state->control_heading_sin, -1.0, 1.0);

    memset(row, 0, SFR_FEATURE_COUNT * sizeof(*row));
    row[0] = (nn_real)1;
    row[1] = (nn_real)((double)action / (double)(SFR_ACTION_COUNT - 1u));
    row[2] = (nn_real)turn;
    row[3] = (nn_real)thrust;
    row[4] = (nn_real)(turn < 0.0 ? 1.0 : 0.0);
    row[5] = (nn_real)(turn > 0.0 ? 1.0 : 0.0);
    row[6] = (nn_real)(thrust > 0.0 ? 1.0 : 0.0);
    row[7] = (nn_real)(thrust < 0.0 ? 1.0 : 0.0);
    row[8] = (nn_real)(action == 0 ? 1.0 : 0.0);
    row[9] = (nn_real)dist;
    row[10] = (nn_real)control_dist;
    row[11] = (nn_real)heading_cos;
    row[12] = (nn_real)heading_sin;
    row[13] = (nn_real)control_heading_cos;
    row[14] = (nn_real)control_heading_sin;
    row[15] = (nn_real)speed;
    row[16] = (nn_real)forward_speed;
    row[17] = (nn_real)lateral_speed;
    row[18] = (nn_real)brake_distance;
    row[19] = (nn_real)fwd_clear;
    row[20] = (nn_real)left_clear;
    row[21] = (nn_real)right_clear;
    row[22] = (nn_real)vel_clear;
    row[23] = (nn_real)signal;
    row[24] = (nn_real)hull;
    row[25] = (nn_real)path_count;
    row[26] = (nn_real)path_current;
    row[27] = (nn_real)progress;
    row[28] = (nn_real)next_dist;
    row[29] = (nn_real)next_speed;
    row[30] = (nn_real)(turn * control_heading_sin);
    row[31] = (nn_real)(thrust * control_heading_cos);
    row[32] = (nn_real)(thrust * fwd_clear);
    row[33] = (nn_real)((thrust < 0.0 ? 1.0 : 0.0) * brake_distance);
    row[34] = (nn_real)((turn < 0.0 ? 1.0 : 0.0) * left_clear);
    row[35] = (nn_real)((turn > 0.0 ? 1.0 : 0.0) * right_clear);
    row[36] = (nn_real)((action == 0 ? 1.0 : 0.0) * fabs(speed));
    row[37] = (nn_real)(turn * lateral_speed);
    row[38] = (nn_real)(thrust * forward_speed);
    row[39] = (nn_real)(thrust * (1.0 - hull));
    row[40] = (nn_real)(thrust * next_speed);
    row[41] = (nn_real)(thrust * progress);
    row[42] = (nn_real)(thrust * close_goal);
    row[43] = (nn_real)(turn * heading_sin);
    row[44] = (nn_real)(thrust * heading_cos);
    row[45] = (nn_real)((thrust < 0.0 ? 1.0 : 0.0) * forward_speed);
    row[46] = (nn_real)fabs(turn);
    row[47] = (nn_real)fabs(thrust);
}

static int sfr_action_allowed(const sfr_state_t *state, size_t action)
{
    const sfr_action_t *a = &SFR_ACTIONS[action < SFR_ACTION_COUNT ? action : 0u];
    if (state->dist > 300.0 && state->speed < 12.0 && a->turn == 0 && a->thrust <= 0) {
        return 0;
    }
    if (a->thrust > 0 && state->control_heading_cos < 0.15) {
        return 0;
    }
    if (a->thrust > 0 && state->fwd_clear < 0.12) {
        return 0;
    }
    if (a->thrust > 0 && state->vel_clear < 0.10 && state->speed > 60.0) {
        return 0;
    }
    return 1;
}

static size_t sfr_neural_action(NeuralNet *net,
                                const sfr_state_t *state,
                                int guarded)
{
    nn_real inputs[SFR_ACTION_COUNT * SFR_FEATURE_COUNT];
    nn_real scores[SFR_ACTION_COUNT];
    nn_real probabilities[SFR_ACTION_COUNT];
    size_t ranked[SFR_ACTION_COUNT];

    for (size_t action = 0; action < SFR_ACTION_COUNT; ++action) {
        sfr_fill_action_row(state, action, &inputs[action * SFR_FEATURE_COUNT]);
    }
    if (!nn_score_action_rows(net, inputs, SFR_ACTION_COUNT, scores, probabilities) ||
        !nn_rank_action_scores(scores, SFR_ACTION_COUNT, ranked)) {
        return sfr_goal_heading_action(state);
    }
    for (size_t i = 0; i < SFR_ACTION_COUNT; ++i) {
        size_t action = ranked[i];
        if (!guarded || sfr_action_allowed(state, action)) {
            return action;
        }
    }
    return sfr_goal_heading_action(state);
}

static void sfr_scan_events(const world_t *w,
                            int *damage_events,
                            int *death_events,
                            double *damage_amount)
{
    for (int i = 0; i < w->events.count; i++) {
        const sim_event_t *ev = &w->events.events[i];
        if (ev->player_id != 0) continue;
        if (ev->type == SIM_EVENT_DAMAGE) {
            (*damage_events)++;
            *damage_amount += ev->damage.amount;
        } else if (ev->type == SIM_EVENT_DEATH) {
            (*death_events)++;
        }
    }
}

static sfr_episode_result_t sfr_run_episode(const sfr_episode_init_t *init,
                                            sfr_controller_t controller,
                                            NeuralNet *net,
                                            int max_ticks)
{
    world_t w = {0};
    server_player_t *sp = NULL;
    nav_path_t path = {0};
    sfr_episode_result_t result = {0};
    double start_hull;
    double max_hull;
    double last_progress_dist;
    int stuck_ticks = 0;

    sfr_setup_world(init, &w, &sp);
    start_hull = sp->ship.hull;
    max_hull = ship_max_hull(&sp->ship);
    result.start_dist = sqrtf(v2_dist_sq(sp->ship.pos, init->goal));
    result.min_dist = result.start_dist;
    result.min_hull_ratio = max_hull > 0.0 ? sp->ship.hull / max_hull : 1.0;
    last_progress_dist = result.start_dist;

    for (int tick = 0; tick < max_ticks; tick++) {
        sfr_state_t state;
        nav_path_t control_path = path;
        size_t action = 0;
        float dist = sqrtf(v2_dist_sq(sp->ship.pos, init->goal));
        if (dist < result.min_dist) result.min_dist = dist;
        if (dist < 120.0f) {
            result.reached = 1;
            result.ticks = tick;
            break;
        }
        if (sp->ship.hull <= 0.0f) {
            result.died = 1;
            result.ticks = tick;
            break;
        }

        if (controller == SFR_CONTROLLER_NEURAL_RAW ||
            controller == SFR_CONTROLLER_NEURAL_GUARDED) {
            sfr_select_hypothesis(&w, sp, init->goal, &path, &control_path, &result);
        }
        sfr_fill_state(&w, sp, &control_path, init->goal, &state);
        switch (controller) {
        case SFR_CONTROLLER_GOAL:
            action = sfr_goal_heading_action(&state);
            break;
        case SFR_CONTROLLER_HEURISTIC: {
            flight_cmd_t cmd = flight_steer_to(&w, &sp->ship, &path, init->goal, 0.0f, 160.0f, SFR_SIM_DT);
            action = sfr_quantize_flight_cmd(cmd);
            break;
        }
        case SFR_CONTROLLER_NEURAL_RAW:
            action = net != NULL ? sfr_neural_action(net, &state, 0) : sfr_goal_heading_action(&state);
            break;
        case SFR_CONTROLLER_NEURAL_GUARDED:
            action = net != NULL ? sfr_neural_action(net, &state, 1) : sfr_goal_heading_action(&state);
            break;
        default:
            action = sfr_goal_heading_action(&state);
            break;
        }

        sfr_apply_action(sp, action);
        world_sim_step(&w, SFR_SIM_DT);
        sfr_scan_events(&w, &result.damage_events, &result.death_events, &result.damage_amount);
        if (result.death_events > 0) {
            result.died = 1;
            result.ticks = tick + 1;
            break;
        }

        double hull_ratio = max_hull > 0.0 ? sp->ship.hull / max_hull : 1.0;
        if (hull_ratio < result.min_hull_ratio) result.min_hull_ratio = hull_ratio;

        if (tick > 0 && (tick % 120) == 0) {
            double current_dist = sqrtf(v2_dist_sq(sp->ship.pos, init->goal));
            if (last_progress_dist - current_dist < 20.0) {
                stuck_ticks += 120;
            } else {
                stuck_ticks = 0;
            }
            last_progress_dist = current_dist;
            if (stuck_ticks >= 8 * 120) {
                result.stuck = 1;
                result.ticks = tick;
                break;
            }
        }
        result.ticks = tick + 1;
    }

    result.final_dist = sqrtf(v2_dist_sq(sp->ship.pos, init->goal));
    if (result.final_dist < result.min_dist) result.min_dist = result.final_dist;
    result.progress = result.start_dist - result.final_dist;
    result.hull_loss = start_hull - sp->ship.hull;
    if (result.hull_loss < 0.0) result.hull_loss = 0.0;
    result.utility =
        (result.reached ? 2.0 : 0.0) +
        sfr_clip(result.progress / 3000.0, -1.0, 1.0) -
        (result.damage_amount * 0.03) -
        (result.died ? 10.0 : 0.0) -
        (result.stuck ? 1.0 : 0.0) -
        ((double)result.ticks / (double)max_ticks) * 0.05;

    world_cleanup(&w);
    return result;
}

static void sfr_summary_add(sfr_summary_t *summary, const sfr_episode_result_t *episode)
{
    summary->episodes++;
    summary->reached += episode->reached;
    summary->died += episode->died;
    summary->stuck += episode->stuck;
    summary->damage_episodes += episode->damage_events > 0 ? 1 : 0;
    summary->total_ticks += episode->ticks;
    summary->damage_events += episode->damage_events;
    summary->death_events += episode->death_events;
    summary->start_dist += episode->start_dist;
    summary->final_dist += episode->final_dist;
    summary->min_dist += episode->min_dist;
    summary->progress += episode->progress;
    summary->damage_amount += episode->damage_amount;
    summary->hull_loss += episode->hull_loss;
    summary->min_hull_ratio_sum += episode->min_hull_ratio;
    summary->utility += episode->utility;
    summary->hypothesis_ticks += episode->hypothesis_ticks;
    summary->hypothesis_candidates += episode->hypothesis_candidates;
    summary->hypothesis_certified_candidates += episode->hypothesis_certified_candidates;
    summary->hypothesis_selected_certified += episode->hypothesis_selected_certified;
    for (size_t i = 0; i < SFR_HYPOTHESIS_COUNT; i++) {
        summary->hypothesis_selected[i] += episode->hypothesis_selected[i];
    }
}

static double sfr_rate(int value, int denom)
{
    return denom > 0 ? (double)value / (double)denom : 0.0;
}

static double sfr_mean(double value, int denom)
{
    return denom > 0 ? value / (double)denom : 0.0;
}

static void sfr_write_summary_line(FILE *out,
                                   const sfr_config_t *config,
                                   const char *checkpoint_hash,
                                   sfr_controller_t controller,
                                   const sfr_summary_t *summary)
{
    int n = summary->episodes;
    fprintf(out,
            "{\"schema\":\"%s\","
            "\"controller\":\"%s\","
            "\"checkpoint_hash\":\"%s\","
            "\"episodes\":%d,"
            "\"ticks_per_episode\":%d,"
            "\"seed\":%u,"
            "\"wall_probes\":%s,"
            "\"reach_rate\":%.9f,"
            "\"death_rate\":%.9f,"
            "\"stuck_rate\":%.9f,"
            "\"damage_episode_rate\":%.9f,"
            "\"damage_events\":%d,"
            "\"death_events\":%d,"
            "\"avg_ticks\":%.9f,"
            "\"avg_start_dist\":%.9f,"
            "\"avg_final_dist\":%.9f,"
            "\"avg_min_dist\":%.9f,"
            "\"avg_progress\":%.9f,"
            "\"avg_damage_amount\":%.9f,"
            "\"avg_hull_loss\":%.9f,"
            "\"avg_min_hull_ratio\":%.9f,"
            "\"hypothesis_layer\":\"certified_control_target_v1\","
            "\"hypothesis_ticks\":%d,"
            "\"hypothesis_candidate_count\":%d,"
            "\"hypothesis_certified_candidate_rate\":%.9f,"
            "\"hypothesis_selected_certified_rate\":%.9f,"
            "\"hypothesis_direct_rate\":%.9f,"
            "\"hypothesis_nav_path_rate\":%.9f,"
            "\"hypothesis_left_bypass_rate\":%.9f,"
            "\"hypothesis_right_bypass_rate\":%.9f,"
            "\"hypothesis_brake_escape_rate\":%.9f,"
            "\"avg_utility\":%.9f}\n",
            SFR_SUMMARY_SCHEMA,
            sfr_controller_name(controller),
            checkpoint_hash != NULL ? checkpoint_hash : "",
            n,
            config->ticks,
            config->seed,
            config->include_wall_probes ? "true" : "false",
            sfr_rate(summary->reached, n),
            sfr_rate(summary->died, n),
            sfr_rate(summary->stuck, n),
            sfr_rate(summary->damage_episodes, n),
            summary->damage_events,
            summary->death_events,
            sfr_mean((double)summary->total_ticks, n),
            sfr_mean(summary->start_dist, n),
            sfr_mean(summary->final_dist, n),
            sfr_mean(summary->min_dist, n),
            sfr_mean(summary->progress, n),
            sfr_mean(summary->damage_amount, n),
            sfr_mean(summary->hull_loss, n),
            sfr_mean(summary->min_hull_ratio_sum, n),
            summary->hypothesis_ticks,
            summary->hypothesis_candidates,
            sfr_rate(summary->hypothesis_certified_candidates, summary->hypothesis_candidates),
            sfr_rate(summary->hypothesis_selected_certified, summary->hypothesis_ticks),
            sfr_rate(summary->hypothesis_selected[SFR_HYPOTHESIS_DIRECT], summary->hypothesis_ticks),
            sfr_rate(summary->hypothesis_selected[SFR_HYPOTHESIS_NAV_PATH], summary->hypothesis_ticks),
            sfr_rate(summary->hypothesis_selected[SFR_HYPOTHESIS_LEFT_BYPASS], summary->hypothesis_ticks),
            sfr_rate(summary->hypothesis_selected[SFR_HYPOTHESIS_RIGHT_BYPASS], summary->hypothesis_ticks),
            sfr_rate(summary->hypothesis_selected[SFR_HYPOTHESIS_BRAKE_ESCAPE], summary->hypothesis_ticks),
            sfr_mean(summary->utility, n));
}

static void sfr_print_usage(FILE *stream)
{
    fputs("usage: signal_flight_rollout --checkpoint path [--episodes N] [--ticks N] "
          "[--seed N] [--summary path] [--no-wall-probes]\n",
          stream);
}

static int sfr_parse_int(const char *text, int min_value, int *out)
{
    char *end = NULL;
    long value;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < min_value || value > INT32_MAX) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int sfr_parse_u32(const char *text, uint32_t *out)
{
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return 0;
    }
    *out = (uint32_t)value;
    return 1;
}

static int sfr_parse_config(int argc, char **argv, sfr_config_t *config)
{
    memset(config, 0, sizeof(*config));
    config->episodes = 24;
    config->ticks = 900;
    config->seed = 4242u;
    config->include_wall_probes = 1;
    for (int i = 0; i < SFR_CONTROLLER_COUNT; i++) {
        config->controller_enabled[i] = 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            config->checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            if (!sfr_parse_int(argv[++i], 1, &config->episodes)) return 0;
        } else if (strcmp(argv[i], "--ticks") == 0 && i + 1 < argc) {
            if (!sfr_parse_int(argv[++i], 1, &config->ticks)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!sfr_parse_u32(argv[++i], &config->seed)) return 0;
        } else if (strcmp(argv[i], "--summary") == 0 && i + 1 < argc) {
            config->summary_path = argv[++i];
        } else if (strcmp(argv[i], "--no-wall-probes") == 0) {
            config->include_wall_probes = 0;
        } else if (strcmp(argv[i], "--help") == 0) {
            sfr_print_usage(stdout);
            exit(EXIT_SUCCESS);
        } else {
            return 0;
        }
    }

    return config->checkpoint_path != NULL;
}

int main(int argc, char **argv)
{
    sfr_config_t config;
    if (!sfr_parse_config(argc, argv, &config)) {
        sfr_print_usage(stderr);
        return EXIT_FAILURE;
    }

    NNCheckpointMeta meta;
    NeuralNet *net = nn_load_checkpoint(config.checkpoint_path, &meta);
    if (net == NULL) {
        fprintf(stderr, "failed to load checkpoint: %s\n", config.checkpoint_path);
        return EXIT_FAILURE;
    }
    if (nn_input_size(net) != SFR_FEATURE_COUNT || nn_output_size(net) != 1u) {
        fprintf(stderr,
                "checkpoint shape mismatch: input=%zu output=%zu expected=%u/1\n",
                nn_input_size(net),
                nn_output_size(net),
                SFR_FEATURE_COUNT);
        nn_free(net);
        return EXIT_FAILURE;
    }

    char checkpoint_hash[65] = "";
    if (!nn_hash_checkpoint_hex(config.checkpoint_path, checkpoint_hash)) {
        strcpy(checkpoint_hash, "unknown");
    }

    sfr_summary_t summaries[SFR_CONTROLLER_COUNT];
    memset(summaries, 0, sizeof(summaries));
    uint32_t rng = config.seed ^ 0x7a11c0deu;

    for (int episode = 0; episode < config.episodes; episode++) {
        sfr_episode_init_t init;
        sfr_init_episode(&config, episode, &rng, &init);
        for (int c = 0; c < SFR_CONTROLLER_COUNT; c++) {
            if (!config.controller_enabled[c]) continue;
            sfr_episode_result_t result =
                sfr_run_episode(&init, (sfr_controller_t)c, net, config.ticks);
            sfr_summary_add(&summaries[c], &result);
        }
    }

    FILE *summary = NULL;
    if (config.summary_path != NULL) {
        summary = fopen(config.summary_path, "w");
        if (summary == NULL) {
            perror("signal_flight_rollout: open summary");
            nn_free(net);
            return EXIT_FAILURE;
        }
    }

    for (int c = 0; c < SFR_CONTROLLER_COUNT; c++) {
        if (!config.controller_enabled[c]) continue;
        sfr_write_summary_line(stdout, &config, checkpoint_hash, (sfr_controller_t)c, &summaries[c]);
        if (summary != NULL) {
            sfr_write_summary_line(summary, &config, checkpoint_hash, (sfr_controller_t)c, &summaries[c]);
        }
    }

    int ok = 1;
    if (summary != NULL && fclose(summary) != 0) {
        perror("signal_flight_rollout: close summary");
        ok = 0;
    }
    nn_free(net);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
