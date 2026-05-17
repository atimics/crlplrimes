#!/usr/bin/env node

import { createReadStream, existsSync } from "node:fs";
import { stat } from "node:fs/promises";

function usage() {
  console.log(`usage: node scripts/signal_live_status.mjs [options]

Options:
  --log PATH          live brain JSONL log (default build/float/signal_live_brain/live.jsonl)
  --interval-ms N     report interval in ms (default 1000)
  --window N          rolling action window in rows (default 250)
  --tail-lines N      seed from the last N rows on startup (default 5000)
  --stale-ms N        mark stale when no row arrives for N ms (default 3000)
  --from-start        seed from the whole file instead of the tail
  --once              print one summary and exit
  --json              print machine-readable status JSON
`);
}

function parseArgs(argv) {
  const config = {
    logPath: "build/float/signal_live_brain/live.jsonl",
    intervalMs: 1000,
    windowRows: 250,
    tailLines: 5000,
    staleMs: 3000,
    fromStart: false,
    once: false,
    json: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--log" && value) {
      config.logPath = value;
      i += 1;
    } else if (arg === "--interval-ms" && value) {
      config.intervalMs = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--window" && value) {
      config.windowRows = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--tail-lines" && value) {
      config.tailLines = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--stale-ms" && value) {
      config.staleMs = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--from-start") {
      config.fromStart = true;
    } else if (arg === "--once") {
      config.once = true;
    } else if (arg === "--json") {
      config.json = true;
    } else {
      throw new Error(`unknown or incomplete option: ${arg}`);
    }
  }

  return config;
}

function parsePositiveInt(text, name) {
  const value = Number(text);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function parseJsonLine(line) {
  if (!line || line.charCodeAt(0) !== 123) return null;
  try {
    const row = JSON.parse(line);
    return row && row.schema === "crlp.signal_live_brain.v1" ? row : null;
  } catch {
    return null;
  }
}

function readRange(path, start, endExclusive) {
  if (endExclusive <= start) return Promise.resolve("");
  return new Promise((resolve, reject) => {
    const chunks = [];
    const stream = createReadStream(path, {
      start,
      end: endExclusive - 1,
      encoding: "utf8",
    });
    stream.on("data", (chunk) => chunks.push(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(chunks.join("")));
  });
}

class LiveStatus {
  constructor(config) {
    this.config = config;
    this.offset = 0;
    this.partial = "";
    this.rowsSeen = 0;
    this.window = [];
    this.lastRow = null;
    this.prevRow = null;
    this.minHull = Infinity;
    this.maxHull = 0;
    this.damageEvents = 0;
    this.hullLoss = 0;
    this.deaths = 0;
    this.lastDamageAt = 0;
    this.startedAt = Date.now();
    this.lastRowWallMs = 0;
  }

  async seed() {
    if (!existsSync(this.config.logPath)) return;
    const info = await stat(this.config.logPath);
    if (this.config.fromStart || info.size === 0) {
      this.offset = 0;
      await this.readNew();
      return;
    }

    const maxSeedBytes = 4 * 1024 * 1024;
    const start = Math.max(0, info.size - maxSeedBytes);
    const text = await readRange(this.config.logPath, start, info.size);
    const lines = text.split(/\r?\n/);
    if (start > 0) lines.shift();
    const tail = lines.filter(Boolean).slice(-this.config.tailLines);
    for (const line of tail) this.ingest(parseJsonLine(line));
    this.offset = info.size;
  }

  async readNew() {
    if (!existsSync(this.config.logPath)) return;
    const info = await stat(this.config.logPath);
    if (info.size < this.offset) {
      this.offset = 0;
      this.partial = "";
      this.prevRow = null;
    }
    const text = await readRange(this.config.logPath, this.offset, info.size);
    this.offset = info.size;
    if (!text) return;
    const lines = (this.partial + text).split(/\r?\n/);
    this.partial = lines.pop() ?? "";
    for (const line of lines) this.ingest(parseJsonLine(line));
  }

  ingest(row) {
    if (!row) return;
    this.rowsSeen += 1;
    this.lastRow = row;
    this.lastRowWallMs = Date.now();
    this.window.push(row);
    if (this.window.length > this.config.windowRows) this.window.shift();

    if (Number.isFinite(row.hull)) {
      this.minHull = Math.min(this.minHull, row.hull);
      this.maxHull = Math.max(this.maxHull, row.hull);
      if (this.prevRow && Number.isFinite(this.prevRow.hull)) {
        const drop = this.prevRow.hull - row.hull;
        if (drop > 0.01) {
          this.damageEvents += 1;
          this.hullLoss += drop;
          this.lastDamageAt = row.t_ms || Date.now();
        }
        if (this.prevRow.hull > 0 && row.hull <= 0) {
          this.deaths += 1;
        }
      }
    }
    this.prevRow = row;
  }

  snapshot() {
    const row = this.lastRow;
    const now = Date.now();
    const ageMs = row ? now - this.lastRowWallMs : now - this.startedAt;
    const actions = new Map();
    let shadowCompared = 0;
    let shadowDisagreements = 0;
    const inference = [];
    for (const item of this.window) {
      const key = item.action || flagsName(item.flags);
      actions.set(key, (actions.get(key) || 0) + 1);
      if (item.neural_action !== null && item.neural_action !== undefined) {
        shadowCompared += 1;
        if (item.shadow_disagrees) shadowDisagreements += 1;
      }
      if (Number.isFinite(item.inference_ms)) inference.push(item.inference_ms);
    }
    const topActions = [...actions.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 4)
      .map(([action, count]) => ({
        action,
        count,
        pct: this.window.length ? count / this.window.length : 0,
      }));

    return {
      schema: "crlp.signal_live_status.v1",
      log: this.config.logPath,
      rows_seen: this.rowsSeen,
      window_rows: this.window.length,
      stale: ageMs > this.config.staleMs,
      age_ms: ageMs,
      player_id: row?.player_id ?? null,
      action: row?.action ?? null,
      brain_mode: row?.brain_mode ?? null,
      active_source: row?.active_source ?? null,
      neural_action: row?.neural_action ?? null,
      heuristic_action: row?.heuristic_action ?? null,
      shadow_disagreement_rate: shadowCompared ? shadowDisagreements / shadowCompared : null,
      inference_p50_ms: percentile(inference, 0.5),
      inference_p95_ms: percentile(inference, 0.95),
      inference_p99_ms: percentile(inference, 0.99),
      flags: row?.flags ?? null,
      hull: finiteOrNull(row?.hull),
      min_hull: Number.isFinite(this.minHull) ? round1(this.minHull) : null,
      max_hull: this.maxHull > 0 ? round1(this.maxHull) : null,
      hull_loss: round1(this.hullLoss),
      damage_events: this.damageEvents,
      deaths: this.deaths,
      docked: row?.docked ?? null,
      x: finiteOrNull(row?.x),
      y: finiteOrNull(row?.y),
      speed: finiteOrNull(row?.speed),
      dist: finiteOrNull(row?.dist),
      target_station: row?.target?.station ?? null,
      asteroids: row?.asteroids ?? null,
      top_actions: topActions,
      last_damage_t_ms: this.lastDamageAt || null,
    };
  }
}

function flagsName(flags) {
  if (!Number.isInteger(flags) || flags === 0) return "NONE";
  const parts = [];
  if (flags & 1) parts.push("W");
  if (flags & 16) parts.push("S");
  if (flags & 2) parts.push("A");
  if (flags & 4) parts.push("D");
  return parts.length ? parts.join("") : `flags:${flags}`;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? round1(value) : null;
}

function percentile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return Math.round(sorted[index] * 1000) / 1000;
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function formatStatus(s) {
  if (!s.rows_seen) {
    return `signal brain: waiting for rows log=${s.log}`;
  }
  const actionMix = s.top_actions
    .map((a) => `${a.action}:${Math.round(a.pct * 100)}%`)
    .join(" ");
  const stale = s.stale ? ` STALE age=${(s.age_ms / 1000).toFixed(1)}s` : "";
  const docked = s.docked ? " docked" : "";
  return [
    `signal brain${stale}`,
    `rows=${s.rows_seen}`,
    `p${s.player_id}`,
    `action=${s.action}`,
    s.brain_mode ? `mode=${s.brain_mode}/${s.active_source}` : "",
    s.neural_action ? `neural=${s.neural_action}` : "",
    s.shadow_disagreement_rate !== null ? `shadow_diff=${Math.round(s.shadow_disagreement_rate * 100)}%` : "",
    s.inference_p95_ms !== null ? `p95=${s.inference_p95_ms}ms` : "",
    `hull=${s.hull}`,
    `min=${s.min_hull}`,
    `loss=${s.hull_loss}`,
    `dmg=${s.damage_events}`,
    `death=${s.deaths}`,
    `speed=${s.speed}`,
    `dist=${s.dist}`,
    `pos=(${s.x},${s.y})`,
    `station=${s.target_station}`,
    `rocks=${s.asteroids}`,
    docked.trim(),
    `mix=${actionMix || "none"}`,
  ].filter(Boolean).join(" ");
}

async function main() {
  const config = parseArgs(process.argv);
  const live = new LiveStatus(config);
  await live.seed();

  async function report() {
    await live.readNew();
    const snapshot = live.snapshot();
    console.log(config.json ? JSON.stringify(snapshot) : formatStatus(snapshot));
  }

  await report();
  if (config.once) return;

  const timer = setInterval(() => {
    report().catch((err) => console.error(`[status] ${err.message}`));
  }, config.intervalMs);
  process.on("SIGINT", () => {
    clearInterval(timer);
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    clearInterval(timer);
    process.exit(0);
  });
}

main().catch((err) => {
  console.error(`[status] ${err.message}`);
  process.exit(1);
});
