#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import http from "node:http";
import https from "node:https";

const SELF_DIR = dirname(fileURLToPath(import.meta.url));
const BRAIN_SCRIPT = join(SELF_DIR, "signal_live_brain.mjs");

function usage() {
  console.log(`usage: node scripts/signal_live_swarm.mjs [options]

Options:
  --server URL       Signal websocket URL (default ws://127.0.0.1:9091/ws)
  --count N          number of pilots to launch (default 4)
  --min-active N     minimum pilots that must emit live rows (default count)
  --max-ms N         stop pilots after N ms (default 30000)
  --tick-ms N        pilot control period in ms (default 50)
  --stagger-ms N     launch delay between pilots (default 100)
  --out-dir PATH     swarm output directory (default build/float/signal_live_swarm)
  --callsign-prefix  prefix for 7-byte callsigns (default BRN)
  --brain-checkpoint PATH
  --brain-mode MODE  heuristic, shadow_neural, or neural_active (default shadow_neural)
  --option-checkpoint PATH
  --option-mode MODE off, fixed, shadow_neural, or neural_active (default off)
  --protocol-root PATH
  --skip-protocol-check
  --allow-protocol-drift
  --dry-run
`);
}

function parseArgs(argv) {
  const config = {
    server: "ws://127.0.0.1:9091/ws",
    count: 4,
    minActive: null,
    maxMs: 30000,
    tickMs: 50,
    staggerMs: 100,
    outDir: "build/float/signal_live_swarm",
    callsignPrefix: "BRN",
    checkpointPath: null,
    brainMode: "shadow_neural",
    optionCheckpointPath: null,
    optionMode: "off",
    protocolRoot: "../signal",
    skipProtocolCheck: false,
    allowProtocolDrift: false,
    dryRun: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--server" && value) {
      config.server = value;
      i += 1;
    } else if (arg === "--count" && value) {
      config.count = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--min-active" && value) {
      config.minActive = parseNonNegativeInt(value, arg);
      i += 1;
    } else if (arg === "--max-ms" && value) {
      config.maxMs = parseNonNegativeInt(value, arg);
      i += 1;
    } else if (arg === "--tick-ms" && value) {
      config.tickMs = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--stagger-ms" && value) {
      config.staggerMs = parseNonNegativeInt(value, arg);
      i += 1;
    } else if (arg === "--out-dir" && value) {
      config.outDir = value;
      i += 1;
    } else if (arg === "--callsign-prefix" && value) {
      config.callsignPrefix = value;
      i += 1;
    } else if (arg === "--brain-checkpoint" && value) {
      config.checkpointPath = value;
      i += 1;
    } else if (arg === "--brain-mode" && value) {
      if (!["heuristic", "shadow_neural", "neural_active"].includes(value)) {
        throw new Error("--brain-mode must be heuristic, shadow_neural, or neural_active");
      }
      config.brainMode = value;
      i += 1;
    } else if (arg === "--option-checkpoint" && value) {
      config.optionCheckpointPath = value;
      i += 1;
    } else if (arg === "--option-mode" && value) {
      if (!["off", "fixed", "shadow_neural", "neural_active"].includes(value)) {
        throw new Error("--option-mode must be off, fixed, shadow_neural, or neural_active");
      }
      config.optionMode = value;
      i += 1;
    } else if (arg === "--protocol-root" && value) {
      config.protocolRoot = value;
      i += 1;
    } else if (arg === "--skip-protocol-check") {
      config.skipProtocolCheck = true;
    } else if (arg === "--allow-protocol-drift") {
      config.allowProtocolDrift = true;
    } else if (arg === "--dry-run") {
      config.dryRun = true;
    } else {
      throw new Error(`unknown or incomplete option: ${arg}`);
    }
  }

  config.minActive = config.minActive ?? config.count;
  if (config.minActive > config.count) {
    throw new Error("--min-active must be <= --count");
  }
  if (config.callsignPrefix.length === 0) {
    throw new Error("--callsign-prefix must not be empty");
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

function parseNonNegativeInt(text, name) {
  const value = Number(text);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function callsign(prefix, index) {
  const suffix = String(index + 1).padStart(3, "0");
  return `${prefix}${suffix}`.slice(0, 7);
}

function healthUrl(serverUrl) {
  const url = new URL(serverUrl);
  url.protocol = url.protocol === "wss:" ? "https:" : "http:";
  url.pathname = "/health";
  url.search = "";
  return url.toString();
}

function fetchJson(urlText, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const url = new URL(urlText);
    const mod = url.protocol === "https:" ? https : http;
    const req = mod.get(url, { timeout: timeoutMs }, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
        } catch {
          resolve(null);
        }
      });
    });
    req.on("timeout", () => req.destroy());
    req.on("error", () => resolve(null));
  });
}

async function readPilotRows(logPath) {
  try {
    const text = await readFile(logPath, "utf8");
    const rows = [];
    for (const line of text.split(/\r?\n/)) {
      if (!line.startsWith("{")) continue;
      try {
        const row = JSON.parse(line);
        if (row.schema === "crlp.signal_live_brain.v1") rows.push(row);
      } catch {
        // Ignore partial rows from interrupted runs.
      }
    }
    return rows;
  } catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

function pilotArgs(config, index, pilotDir) {
  const args = [
    BRAIN_SCRIPT,
    "--server", config.server,
    "--session", join(pilotDir, "session.json"),
    "--log", join(pilotDir, "live.jsonl"),
    "--callsign", callsign(config.callsignPrefix, index),
    "--max-ms", String(config.maxMs),
    "--tick-ms", String(config.tickMs),
    "--brain-mode", config.brainMode,
    "--option-mode", config.optionMode,
  ];
  if (config.checkpointPath) args.push("--brain-checkpoint", config.checkpointPath);
  if (config.optionCheckpointPath) args.push("--option-checkpoint", config.optionCheckpointPath);
  if (config.skipProtocolCheck) {
    args.push("--skip-protocol-check");
  } else {
    args.push("--protocol-root", config.protocolRoot);
  }
  if (config.allowProtocolDrift) args.push("--allow-protocol-drift");
  if (config.dryRun) args.push("--dry-run");
  return args;
}

async function launchPilot(config, index, outDir) {
  const pilotDir = join(outDir, `pilot-${String(index + 1).padStart(2, "0")}`);
  await mkdir(pilotDir, { recursive: true });
  const stdoutPath = join(pilotDir, "stdout.log");
  const stderrPath = join(pilotDir, "stderr.log");
  const stdout = createWriteStream(stdoutPath, { flags: "w" });
  const stderr = createWriteStream(stderrPath, { flags: "w" });
  const child = spawn(process.execPath, pilotArgs(config, index, pilotDir), {
    cwd: resolve("."),
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.pipe(stdout);
  child.stderr.pipe(stderr);
  const pilot = {
    index,
    pilotDir,
    logPath: join(pilotDir, "live.jsonl"),
    stdoutPath,
    stderrPath,
    child,
    exitCode: null,
    signal: null,
  };
  pilot.done = new Promise((resolveDone) => {
    child.once("exit", (code, signal) => {
      pilot.exitCode = code;
      pilot.signal = signal;
      resolveDone();
    });
  });
  return pilot;
}

async function summarizePilot(config, pilot) {
  const rows = await readPilotRows(pilot.logPath);
  const last = rows.at(-1) ?? null;
  return {
    index: pilot.index,
    callsign: callsign(config.callsignPrefix, pilot.index),
    exit_code: pilot.exitCode,
    signal: pilot.signal,
    rows: rows.length,
    active: rows.length > 0,
    player_id: last?.player_id ?? null,
    last_action: last?.action ?? null,
    hull: last?.hull ?? null,
    x: last?.x ?? null,
    y: last?.y ?? null,
    log: pilot.logPath,
    stdout: pilot.stdoutPath,
    stderr: pilot.stderrPath,
  };
}

async function main() {
  const config = parseArgs(process.argv);
  const outDir = resolve(config.outDir);
  await mkdir(outDir, { recursive: true });
  const health = healthUrl(config.server);
  const startedAt = new Date().toISOString();
  console.log(`[swarm] launching ${config.count} pilot(s) against ${config.server}`);
  console.log(`[swarm] health ${health}`);
  console.log(`[swarm] out ${outDir}`);

  const pilots = [];
  let stopping = false;
  const stopAll = () => {
    if (stopping) return;
    stopping = true;
    for (const pilot of pilots) {
      if (pilot.child.exitCode === null && !pilot.child.killed) {
        pilot.child.kill("SIGTERM");
      }
    }
  };
  process.on("SIGINT", stopAll);
  process.on("SIGTERM", stopAll);

  for (let i = 0; i < config.count; i += 1) {
    const pilot = await launchPilot(config, i, outDir);
    pilots.push(pilot);
    if (i + 1 < config.count && config.staggerMs > 0) {
      await sleep(config.staggerMs);
    }
  }

  const statusTimer = setInterval(async () => {
    const doc = await fetchJson(health);
    const live = doc ? `players=${doc.players ?? "?"} live=${doc.live_connections ?? "?"}` : "health=unavailable";
    const running = pilots.filter((pilot) => pilot.child.exitCode === null).length;
    console.log(`[swarm] ${live} running=${running}/${config.count}`);
  }, 2000);

  await Promise.all(pilots.map((pilot) => pilot.done));
  clearInterval(statusTimer);
  const finalHealth = await fetchJson(health);
  const pilotSummaries = await Promise.all(pilots.map((pilot) => summarizePilot(config, pilot)));
  const activePilots = pilotSummaries.filter((pilot) => pilot.active).length;
  const summary = {
    schema: "crlp.signal_live_swarm.summary.v1",
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    server: config.server,
    health_url: health,
    requested_pilots: config.count,
    min_active: config.minActive,
    active_pilots: activePilots,
    success: activePilots >= config.minActive,
    final_health: finalHealth,
    pilots: pilotSummaries,
  };
  const summaryPath = join(outDir, "summary.json");
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`[swarm] active=${activePilots}/${config.count} min_active=${config.minActive}`);
  console.log(`[swarm] summary ${summaryPath}`);
  process.exit(summary.success ? 0 : 2);
}

main().catch((err) => {
  console.error(`[swarm] ${err.message}`);
  process.exit(1);
});
