#!/usr/bin/env node

import { mkdir, readFile, writeFile, appendFile } from "node:fs/promises";
import { dirname } from "node:path";
import { createHash, randomBytes } from "node:crypto";
import { createConnection } from "node:net";
import { connect as tlsConnect } from "node:tls";

const NET_MSG_JOIN = 0x01;
const NET_MSG_INPUT = 0x04;
const NET_MSG_WORLD_ASTEROIDS = 0x10;
const NET_MSG_PLAYER_SHIP = 0x15;
const NET_MSG_STATION_IDENTITY = 0x17;
const NET_MSG_WORLD_PLAYERS = 0x18;
const NET_MSG_SESSION = 0x20;

const NET_INPUT_THRUST = 1 << 0;
const NET_INPUT_LEFT = 1 << 1;
const NET_INPUT_RIGHT = 1 << 2;
const NET_INPUT_BRAKE = 1 << 4;

const NET_ACTION_NONE = 0;
const NET_ACTION_LAUNCH = 2;

const PLAYER_RECORD_SIZE = 77;
const ASTEROID_RECORD_SIZE = 34;
const STATION_IDENTITY_MIN_SIZE = 59;

const SHIP_BRAKE = 180.0;
const OBJECTIVE_WAYPOINTS_PER_STATION = 4;
const OBJECTIVE_COMPLETE_RADIUS = 450;
const STATION_ESCAPE_PADDING = 650;
const NAV_WAYPOINT_REACHED_RADIUS = 280;
const NAV_WAYPOINT_TTL_MS = 18000;
const NAV_ASTEROID_CLEARANCE = 300;
const NAV_ASTEROID_LOOKAHEAD = 1700;

function usage() {
  console.log(`usage: node scripts/signal_live_brain.mjs [options]

Options:
  --server URL       Signal websocket URL (default ws://127.0.0.1:9091/ws)
  --callsign NAME    7-byte callsign sent in SESSION (default BRN-001)
  --session PATH     persisted session token JSON
  --log PATH         append live observation/action JSONL
  --max-ms N         stop after N ms; 0 means run until interrupted
  --tick-ms N        control period in ms (default 50)
  --station N        station id to orbit; default nearest known station
  --radius N         orbit radius in world units (default 2200)
  --angular-rate N   orbit angular rate in rad/sec (default 0.045)
  --goal X,Y         fly toward a fixed point instead of orbiting
  --brain-checkpoint PATH
                    load a signal-flight-live-v2 checkpoint for WASD scoring
  --brain-mode MODE  heuristic, shadow_neural, or neural_active (default shadow_neural)
  --dry-run          connect and decode, but do not send movement inputs
`);
}

function parseArgs(argv) {
  const config = {
    server: "ws://127.0.0.1:9091/ws",
    callsign: "BRN-001",
    sessionPath: "build/float/signal_live_brain/session.json",
    logPath: "build/float/signal_live_brain/live.jsonl",
    maxMs: 0,
    tickMs: 50,
    station: null,
    radius: 2200,
    angularRate: 0.045,
    goal: null,
    checkpointPath: null,
    brainMode: "shadow_neural",
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
    } else if (arg === "--callsign" && value) {
      config.callsign = value.slice(0, 7);
      i += 1;
    } else if (arg === "--session" && value) {
      config.sessionPath = value;
      i += 1;
    } else if (arg === "--log" && value) {
      config.logPath = value;
      i += 1;
    } else if (arg === "--max-ms" && value) {
      config.maxMs = parseNonNegativeInt(value, arg);
      i += 1;
    } else if (arg === "--tick-ms" && value) {
      config.tickMs = parsePositiveInt(value, arg);
      i += 1;
    } else if (arg === "--station" && value) {
      config.station = parseNonNegativeInt(value, arg);
      i += 1;
    } else if (arg === "--radius" && value) {
      config.radius = parsePositiveNumber(value, arg);
      i += 1;
    } else if (arg === "--angular-rate" && value) {
      config.angularRate = parseNumber(value, arg);
      i += 1;
    } else if (arg === "--goal" && value) {
      const parts = value.split(",");
      if (parts.length !== 2) throw new Error("--goal must be X,Y");
      config.goal = {
        x: parseNumber(parts[0], "--goal x"),
        y: parseNumber(parts[1], "--goal y"),
      };
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
    } else if (arg === "--dry-run") {
      config.dryRun = true;
    } else {
      throw new Error(`unknown or incomplete option: ${arg}`);
    }
  }

  if (config.callsign.length === 0) {
    throw new Error("--callsign must not be empty");
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

function parsePositiveNumber(text, name) {
  const value = parseNumber(text, name);
  if (!(value > 0)) {
    throw new Error(`${name} must be positive`);
  }
  return value;
}

function parseNumber(text, name) {
  const value = Number(text);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be numeric`);
  }
  return value;
}

async function loadSessionToken(path) {
  await mkdir(dirname(path), { recursive: true });
  try {
    const json = JSON.parse(await readFile(path, "utf8"));
    if (typeof json.token === "string" && /^[0-9a-fA-F]{16}$/.test(json.token)) {
      return Buffer.from(json.token, "hex");
    }
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
  }

  const token = randomBytes(8);
  await writeFile(path, `${JSON.stringify({ token: token.toString("hex") }, null, 2)}\n`);
  return token;
}

function readF32(buffer, offset) {
  return buffer.readFloatLE(offset);
}

function readU16(buffer, offset) {
  return buffer.readUInt16LE(offset);
}

function readU32(buffer, offset) {
  return buffer.readUInt32LE(offset);
}

function writeU16(buffer, offset, value) {
  buffer.writeUInt16LE(value & 0xffff, offset);
}

function writeU32(buffer, offset, value) {
  buffer.writeUInt32LE(value >>> 0, offset);
}

class SignalWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(urlText) {
    this.url = new URL(urlText);
    this.readyState = SignalWebSocket.CONNECTING;
    this.socket = null;
    this.handlers = new Map();
    this.recv = Buffer.alloc(0);
    this.handshakeComplete = false;
    this.acceptKey = "";
    this.connect();
  }

  addEventListener(type, handler) {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set());
    this.handlers.get(type).add(handler);
  }

  dispatch(type, event = {}) {
    for (const handler of this.handlers.get(type) ?? []) {
      try {
        handler(event);
      } catch (err) {
        console.error(`[brain] websocket ${type} handler failed: ${err.message}`);
      }
    }
  }

  connect() {
    const secure = this.url.protocol === "wss:";
    if (!secure && this.url.protocol !== "ws:") {
      throw new Error(`unsupported websocket protocol: ${this.url.protocol}`);
    }
    const port = this.url.port ? Number(this.url.port) : secure ? 443 : 80;
    const host = this.url.hostname;
    const path = `${this.url.pathname || "/"}${this.url.search || ""}`;
    const key = randomBytes(16).toString("base64");
    this.acceptKey = createHash("sha1")
      .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest("base64");
    const headers = [
      `GET ${path} HTTP/1.1`,
      `Host: ${host}${this.url.port ? `:${port}` : ""}`,
      "Upgrade: websocket",
      "Connection: Upgrade",
      `Sec-WebSocket-Key: ${key}`,
      "Sec-WebSocket-Version: 13",
      "\r\n",
    ].join("\r\n");

    const onConnect = () => this.socket.write(headers);
    this.socket = secure
      ? tlsConnect({ host, port, servername: host }, onConnect)
      : createConnection({ host, port }, onConnect);
    this.socket.on("data", (chunk) => this.onData(chunk));
    this.socket.on("error", (error) => {
      this.readyState = SignalWebSocket.CLOSED;
      this.dispatch("error", { error });
    });
    this.socket.on("close", () => {
      this.readyState = SignalWebSocket.CLOSED;
      this.dispatch("close", {});
    });
  }

  onData(chunk) {
    this.recv = Buffer.concat([this.recv, chunk]);
    if (!this.handshakeComplete) {
      const end = this.recv.indexOf("\r\n\r\n");
      if (end < 0) return;
      const head = this.recv.subarray(0, end).toString("latin1");
      this.recv = this.recv.subarray(end + 4);
      this.finishHandshake(head);
    }
    this.readFrames();
  }

  finishHandshake(head) {
    const lines = head.split("\r\n");
    const status = lines[0] || "";
    const headers = new Map();
    for (const line of lines.slice(1)) {
      const idx = line.indexOf(":");
      if (idx > 0) headers.set(line.slice(0, idx).toLowerCase(), line.slice(idx + 1).trim());
    }
    if (!/^HTTP\/1\.[01] 101\b/.test(status)) {
      throw new Error(`websocket upgrade failed: ${status}`);
    }
    if (headers.get("sec-websocket-accept") !== this.acceptKey) {
      throw new Error("websocket upgrade failed: bad Sec-WebSocket-Accept");
    }
    this.handshakeComplete = true;
    this.readyState = SignalWebSocket.OPEN;
    this.dispatch("open", {});
  }

  readFrames() {
    for (;;) {
      if (this.recv.length < 2) return;
      const b0 = this.recv[0];
      const b1 = this.recv[1];
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f;
      let offset = 2;
      if (len === 126) {
        if (this.recv.length < offset + 2) return;
        len = this.recv.readUInt16BE(offset);
        offset += 2;
      } else if (len === 127) {
        if (this.recv.length < offset + 8) return;
        const high = this.recv.readUInt32BE(offset);
        const low = this.recv.readUInt32BE(offset + 4);
        if (high !== 0) throw new Error("websocket frame too large");
        len = low;
        offset += 8;
      }
      const maskOffset = offset;
      if (masked) offset += 4;
      if (this.recv.length < offset + len) return;
      let payload = this.recv.subarray(offset, offset + len);
      if (masked) {
        payload = Buffer.from(payload);
        const mask = this.recv.subarray(maskOffset, maskOffset + 4);
        for (let i = 0; i < payload.length; i += 1) payload[i] ^= mask[i & 3];
      }
      this.recv = this.recv.subarray(offset + len);

      if (opcode === 0x8) {
        this.close();
        return;
      }
      if (opcode === 0x9) {
        this.writeFrame(payload, 0x0a);
        continue;
      }
      if (opcode === 0x1 || opcode === 0x2) {
        this.dispatch("message", { data: payload });
      }
    }
  }

  send(data) {
    this.writeFrame(Buffer.from(data), 0x2);
  }

  close() {
    if (this.readyState === SignalWebSocket.CLOSED || this.readyState === SignalWebSocket.CLOSING) return;
    this.readyState = SignalWebSocket.CLOSING;
    if (this.socket && !this.socket.destroyed) {
      this.writeFrame(Buffer.alloc(0), 0x8);
      this.socket.end();
    }
  }

  writeFrame(payload, opcode) {
    if (!this.socket || this.socket.destroyed) return;
    const mask = randomBytes(4);
    let header;
    if (payload.length < 126) {
      header = Buffer.alloc(2);
      header[0] = 0x80 | opcode;
      header[1] = 0x80 | payload.length;
    } else if (payload.length <= 0xffff) {
      header = Buffer.alloc(4);
      header[0] = 0x80 | opcode;
      header[1] = 0x80 | 126;
      header.writeUInt16BE(payload.length, 2);
    } else {
      header = Buffer.alloc(10);
      header[0] = 0x80 | opcode;
      header[1] = 0x80 | 127;
      header.writeUInt32BE(0, 2);
      header.writeUInt32BE(payload.length, 6);
    }
    const masked = Buffer.from(payload);
    for (let i = 0; i < masked.length; i += 1) masked[i] ^= mask[i & 3];
    this.socket.write(Buffer.concat([header, mask, masked]));
  }
}

function wrapAngle(angle) {
  let x = angle;
  while (x > Math.PI) x -= Math.PI * 2;
  while (x < -Math.PI) x += Math.PI * 2;
  return x;
}

function turnFlagForHeading(headingError, deadband = 0.12) {
  if (headingError > deadband) return NET_INPUT_LEFT;
  if (headingError < -deadband) return NET_INPUT_RIGHT;
  return 0;
}

function len(x, y) {
  return Math.hypot(x, y);
}

function segmentProjection(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const denom = abx * abx + aby * aby;
  const rawT = denom > 0 ? ((px - ax) * abx + (py - ay) * aby) / denom : 0;
  const t = clip(rawT, 0, 1);
  const x = ax + abx * t;
  const y = ay + aby * t;
  return {
    t,
    x,
    y,
    dist: len(px - x, py - y),
  };
}

function clip(value, minValue, maxValue) {
  if (!Number.isFinite(value)) return 0;
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

function scale(value, denominator) {
  return denominator > 0 ? clip(value / denominator, 0, 1) : 0;
}

function signedScale(value, denominator) {
  return denominator > 0 ? clip(value / denominator, -1, 1) : 0;
}

function stationName(buffer, offset) {
  const end = Math.min(buffer.length, offset + 32);
  let zero = offset;
  while (zero < end && buffer[zero] !== 0) zero += 1;
  return buffer.subarray(offset, zero).toString("utf8");
}

async function frameDataToBuffer(data) {
  if (Buffer.isBuffer(data)) return data;
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  if (ArrayBuffer.isView(data)) {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data && typeof data.arrayBuffer === "function") {
    return Buffer.from(await data.arrayBuffer());
  }
  throw new Error("unsupported websocket frame data");
}

const SIGNAL_FLIGHT_ACTIONS = [
  { id: 0, name: "NONE", flags: 0, turn: 0, thrust: 0 },
  { id: 1, name: "W", flags: NET_INPUT_THRUST, turn: 0, thrust: 1 },
  { id: 2, name: "A", flags: NET_INPUT_LEFT, turn: -1, thrust: 0 },
  { id: 3, name: "D", flags: NET_INPUT_RIGHT, turn: 1, thrust: 0 },
  { id: 4, name: "S", flags: NET_INPUT_BRAKE, turn: 0, thrust: -1 },
  { id: 5, name: "WA", flags: NET_INPUT_THRUST | NET_INPUT_LEFT, turn: -1, thrust: 1 },
  { id: 6, name: "WD", flags: NET_INPUT_THRUST | NET_INPUT_RIGHT, turn: 1, thrust: 1 },
  { id: 7, name: "SA", flags: NET_INPUT_BRAKE | NET_INPUT_LEFT, turn: -1, thrust: -1 },
  { id: 8, name: "SD", flags: NET_INPUT_BRAKE | NET_INPUT_RIGHT, turn: 1, thrust: -1 },
];

const NN_ACTIVATION_SIGMOID = 0;
const NN_ACTIVATION_RELU = 1;
const NN_ACTIVATION_TANH = 2;
const NN_ACTIVATION_LINEAR = 3;

async function loadSignalFlightModel(path) {
  if (!path) return null;
  const buffer = await readFile(path);
  const hash = createHash("sha256").update(buffer).digest("hex");
  let offset = 0;

  const requireBytes = (count) => {
    if (offset + count > buffer.length) {
      throw new Error(`truncated checkpoint: ${path}`);
    }
  };
  const readU32 = () => {
    requireBytes(4);
    const value = buffer.readUInt32LE(offset);
    offset += 4;
    return value;
  };
  const readU64 = () => {
    requireBytes(8);
    const value = Number(buffer.readBigUInt64LE(offset));
    offset += 8;
    if (!Number.isSafeInteger(value)) {
      throw new Error(`checkpoint integer is too large: ${path}`);
    }
    return value;
  };
  const readScalar = (scalarType) => {
    if (scalarType === 1) {
      requireBytes(4);
      const value = buffer.readFloatLE(offset);
      offset += 4;
      return value;
    }
    if (scalarType === 2) {
      requireBytes(8);
      const value = buffer.readDoubleLE(offset);
      offset += 8;
      return value;
    }
    throw new Error(`unsupported checkpoint scalar type: ${scalarType}`);
  };
  const readString = (length) => {
    requireBytes(length);
    const bytes = buffer.subarray(offset, offset + length);
    offset += length;
    const zero = bytes.indexOf(0);
    return bytes.subarray(0, zero >= 0 ? zero : bytes.length).toString("utf8");
  };

  requireBytes(8);
  const magic = buffer.subarray(offset, offset + 8).toString("ascii");
  offset += 8;
  if (magic !== "NNCKPT01") throw new Error(`bad checkpoint magic: ${path}`);

  const checkpointVersion = readU32();
  const scalarType = readU32();
  const layerCount = readU64();
  const sizes = [];
  for (let i = 0; i < layerCount; i += 1) sizes.push(readU64());
  const hiddenActivation = readU32();
  const outputActivation = readU32();
  const loss = readU32();

  const weights = [null];
  const biases = [null];
  for (let layer = 1; layer < layerCount; layer += 1) {
    const previous = sizes[layer - 1];
    const current = sizes[layer];
    const layerWeights = new Float64Array(previous * current);
    const layerBiases = new Float64Array(current);
    for (let i = 0; i < layerWeights.length; i += 1) layerWeights[i] = readScalar(scalarType);
    for (let i = 0; i < layerBiases.length; i += 1) layerBiases[i] = readScalar(scalarType);
    weights.push(layerWeights);
    biases.push(layerBiases);
  }

  const metadataSize = readU64();
  if (metadataSize !== 487 && metadataSize !== 455) {
    throw new Error(`unsupported checkpoint metadata size ${metadataSize}: ${path}`);
  }
  const metadata = {
    checkpointVersion,
    featureEncoderVersion: readU32(),
    parentModelHash: readString(65),
    trainingTraceRoot: readString(65),
    benchmarkReportHash: readString(65),
    notes: readString(256),
    featureSet: metadataSize === 487 ? readString(32) : "",
  };

  if (sizes.length !== 4 || sizes[0] !== 48 || sizes[1] !== 32 || sizes[2] !== 16 || sizes[3] !== 1) {
    throw new Error(`unsupported signal flight checkpoint shape: ${sizes.join("x")}`);
  }
  if (metadata.featureEncoderVersion !== 2 || metadata.featureSet !== "signal-flight-live-v2") {
    throw new Error(
      `refusing non-live checkpoint ${path}: feature_encoder_version=${metadata.featureEncoderVersion} feature_set=${metadata.featureSet}`,
    );
  }

  return {
    path,
    hash,
    sizes,
    hiddenActivation,
    outputActivation,
    loss,
    weights,
    biases,
    metadata,
  };
}

function activate(value, activation) {
  switch (activation) {
    case NN_ACTIVATION_RELU:
      return value > 0 ? value : 0;
    case NN_ACTIVATION_TANH:
      return Math.tanh(value);
    case NN_ACTIVATION_LINEAR:
      return value;
    case NN_ACTIVATION_SIGMOID:
    default:
      return value >= 0
        ? 1 / (1 + Math.exp(-value))
        : Math.exp(value) / (1 + Math.exp(value));
  }
}

function forwardSignalFlightModel(model, input) {
  let activations = input;
  for (let layer = 1; layer < model.sizes.length; layer += 1) {
    const previous = model.sizes[layer - 1];
    const current = model.sizes[layer];
    const weights = model.weights[layer];
    const biases = model.biases[layer];
    const output = new Float64Array(current);
    const activation = layer + 1 === model.sizes.length
      ? model.outputActivation
      : model.hiddenActivation;
    for (let neuron = 0; neuron < current; neuron += 1) {
      let sum = biases[neuron];
      const base = neuron * previous;
      for (let i = 0; i < previous; i += 1) {
        sum += weights[base + i] * activations[i];
      }
      output[neuron] = activate(sum, activation);
    }
    activations = output;
  }
  return activations[0];
}

function scoreProbabilities(scores) {
  if (!scores.length) return [];
  const maxScore = Math.max(...scores.map((item) => item.score));
  const weights = scores.map((item) => Math.exp(item.score - maxScore));
  const denom = weights.reduce((sum, value) => sum + value, 0) || 1;
  return scores.map((item, index) => ({
    action: item.action.name,
    probability: round6(weights[index] / denom),
  }));
}

class SignalLiveBrain {
  constructor(config, sessionToken, model) {
    this.config = config;
    this.sessionToken = sessionToken;
    this.model = model;
    this.ws = null;
    this.opened = false;
    this.localId = null;
    this.inputSeq = 0;
    this.actionId = 1;
    this.lastFlags = 0;
    this.lastActionAt = 0;
    this.startMs = Date.now();
    this.lastStatusAt = 0;
    this.lastLogAt = 0;
    this.cruiseTurnSign = 1;
    this.cruiseTurnLocked = false;
    this.burnHoldUntilMs = 0;
    this.burnHoldTurnFlag = 0;
    this.stationEscapeTurnSign = this.sessionToken[1] >= 128 ? 1 : -1;
    this.objective = null;
    this.objectiveSerial = 1;
    this.navWaypoint = null;
    this.navSerial = 1;
    this.players = new Map();
    this.stations = new Map();
    this.asteroids = new Map();
    this.ship = {
      hull: 0,
      docked: false,
      currentStation: -1,
    };
  }

  async run() {
    await mkdir(dirname(this.config.logPath), { recursive: true });
    this.ws = new SignalWebSocket(this.config.server);
    this.ws.addEventListener("open", () => this.onOpen());
    this.ws.addEventListener("message", (event) => {
      frameDataToBuffer(event.data)
        .then((buffer) => this.onMessage(buffer))
        .catch((err) => console.error(`[brain] message decode failed: ${err.message}`));
    });
    this.ws.addEventListener("error", (event) => {
      console.error("[brain] websocket error", event.error ?? "");
    });
    this.ws.addEventListener("close", () => {
      this.opened = false;
      console.log("[brain] disconnected");
    });

    const tick = setInterval(() => {
      this.controlTick().catch((err) => console.error(`[brain] tick failed: ${err.message}`));
    }, this.config.tickMs);

    const timeout = this.config.maxMs > 0
      ? setTimeout(() => {
          clearInterval(tick);
          this.ws?.close();
        }, this.config.maxMs)
      : null;

    await new Promise((resolve) => {
      this.ws.addEventListener("close", resolve, { once: true });
    });
    clearInterval(tick);
    if (timeout) clearTimeout(timeout);
  }

  onOpen() {
    this.opened = true;
    console.log(`[brain] connected ${this.config.server}`);
    if (this.model) {
      console.log(`[brain] model ${this.model.hash} feature_set=${this.model.metadata.featureSet}`);
    }
    this.sendSession();
  }

  sendSession() {
    const buffer = Buffer.alloc(16);
    buffer[0] = NET_MSG_SESSION;
    this.sessionToken.copy(buffer, 1);
    Buffer.from(this.config.callsign.padEnd(7, "\0").slice(0, 7), "ascii").copy(buffer, 9);
    this.send(buffer);
    console.log(`[brain] session ${this.sessionToken.toString("hex")} callsign=${this.config.callsign}`);
  }

  onMessage(buffer) {
    if (buffer.length < 1) return;
    switch (buffer[0]) {
      case NET_MSG_JOIN:
        this.handleJoin(buffer);
        break;
      case NET_MSG_WORLD_PLAYERS:
        this.handleWorldPlayers(buffer);
        break;
      case NET_MSG_PLAYER_SHIP:
        this.handlePlayerShip(buffer);
        break;
      case NET_MSG_WORLD_ASTEROIDS:
        this.handleWorldAsteroids(buffer);
        break;
      case NET_MSG_STATION_IDENTITY:
        this.handleStationIdentity(buffer);
        break;
      default:
        break;
    }
  }

  handleJoin(buffer) {
    if (buffer.length < 2) return;
    const id = buffer[1];
    if (this.localId === null) {
      this.localId = id;
      console.log(`[brain] joined as player ${id}`);
    }
  }

  handleWorldPlayers(buffer) {
    if (buffer.length < 2) return;
    const count = buffer[1];
    const recordSize = buffer.length >= 2 + count * PLAYER_RECORD_SIZE ? PLAYER_RECORD_SIZE : 67;
    if (buffer.length < 2 + count * recordSize) return;
    for (let i = 0; i < count; i += 1) {
      const offset = 2 + i * recordSize;
      const id = buffer[offset];
      const state = {
        id,
        x: readF32(buffer, offset + 1),
        y: readF32(buffer, offset + 5),
        vx: readF32(buffer, offset + 9),
        vy: readF32(buffer, offset + 13),
        angle: readF32(buffer, offset + 17),
        flags: buffer[offset + 21],
        serverTick: recordSize >= 73 ? readU32(buffer, offset + 69) : 0,
        inputTickAck: recordSize >= 77 ? readU32(buffer, offset + 73) : 0,
        seenAt: Date.now(),
      };
      this.players.set(id, state);
    }
  }

  handlePlayerShip(buffer) {
    if (buffer.length < 16) return;
    const id = buffer[1];
    if (this.localId !== null && id !== this.localId) return;
    this.ship.hull = readF32(buffer, 2);
    this.ship.docked = buffer[10] !== 0;
    this.ship.currentStation = buffer[11] === 0xff ? -1 : buffer[11];
  }

  handleWorldAsteroids(buffer) {
    if (buffer.length < 3) return;
    const count = readU16(buffer, 1);
    if (buffer.length < 3 + count * ASTEROID_RECORD_SIZE) return;
    for (let i = 0; i < count; i += 1) {
      const offset = 3 + i * ASTEROID_RECORD_SIZE;
      const index = readU16(buffer, offset);
      const flags = buffer[offset + 2];
      const active = (flags & 1) !== 0;
      if (!active) {
        this.asteroids.delete(index);
        continue;
      }
      this.asteroids.set(index, {
        index,
        x: readF32(buffer, offset + 3),
        y: readF32(buffer, offset + 7),
        vx: readF32(buffer, offset + 11),
        vy: readF32(buffer, offset + 15),
        hp: readF32(buffer, offset + 19),
        ore: readF32(buffer, offset + 23),
        radius: readF32(buffer, offset + 27),
        seenAt: Date.now(),
      });
    }
  }

  handleStationIdentity(buffer) {
    if (buffer.length < STATION_IDENTITY_MIN_SIZE) return;
    const id = buffer[1];
    this.stations.set(id, {
      id,
      flags: buffer[2],
      x: readF32(buffer, 7),
      y: readF32(buffer, 11),
      radius: readF32(buffer, 15),
      dockRadius: readF32(buffer, 19),
      signalRange: readF32(buffer, 23),
      name: stationName(buffer, 27),
    });
  }

  async controlTick() {
    if (!this.opened || !this.ws || this.ws.readyState !== SignalWebSocket.OPEN) return;
    const player = this.localPlayer();
    const now = Date.now();
    if (!player) {
      this.maybeStatus(now, "waiting for player state");
      return;
    }

    let action = NET_ACTION_NONE;
    if (this.ship.docked && now - this.lastActionAt > 1000) {
      action = NET_ACTION_LAUNCH;
      this.lastActionAt = now;
    }

    const decision = this.decide(player);
    if (!this.config.dryRun) {
      this.sendInput(decision.flags, action, player);
    }
    await this.logDecision(now, player, decision, action);
    this.maybeStatus(now, this.statusText(player, decision, action));
  }

  localPlayer() {
    if (this.localId === null) return null;
    return this.players.get(this.localId) ?? null;
  }

  decide(player) {
    const heuristic = this.heuristicDecision(player);
    if (!this.model || !heuristic.target || !heuristic.state || this.config.brainMode === "heuristic") {
      return this.attachDecisionMode(heuristic, heuristic, null, 0);
    }

    const started = process.hrtime.bigint();
    const neural = this.neuralDecision(player, heuristic.target, heuristic.state);
    const inferenceMs = Number(process.hrtime.bigint() - started) / 1e6;
    const active = this.config.brainMode === "neural_active" ? neural : heuristic;
    return this.attachDecisionMode(active, heuristic, neural, inferenceMs);
  }

  heuristicDecision(player) {
    if (this.ship.docked) {
      return {
        flags: 0,
        actionName: "DOCKED",
        target: null,
        obstacle: null,
        dist: 0,
        headingError: 0,
        policy: "heuristic",
      };
    }

    const target = this.targetFor(player);
    if (!target) {
      return {
        flags: 0,
        actionName: "WAIT",
        target: null,
        obstacle: null,
        dist: 0,
        headingError: 0,
        policy: "heuristic",
      };
    }

    const state = this.flightState(player, target);

    const targetTurn = turnFlagForHeading(state.headingError);

    let flags = targetTurn;

    if (state.dist > 180 && Math.abs(state.headingError) < 1.25) {
      flags |= NET_INPUT_THRUST;
    }

    return {
      flags,
      actionName: flagsToActionName(flags),
      target,
      obstacle: state.obstacle,
      dist: state.dist,
      headingError: state.headingError,
      velocityHeadingError: state.velocityHeadingError,
      speed: state.speed,
      forwardSpeed: state.forwardSpeed,
      lateralSpeed: state.lateralSpeed,
      closingSpeed: state.closingSpeed,
      velocityGoalCos: state.velocityGoalCos,
      brakeDistance: state.brakeDistance,
      policy: "heuristic",
      state,
    };
  }

  neuralDecision(player, target, state) {
    const recovery = this.recoveryDecision(player, target, state);
    if (recovery) return recovery;
    return this.modelDecision(player, target, state);
  }

  attachDecisionMode(active, heuristic, neural, inferenceMs) {
    const activeSource = neural && this.config.brainMode === "neural_active" ? "neural" : "heuristic";
    return {
      ...active,
      state: undefined,
      brainMode: this.model ? this.config.brainMode : "heuristic",
      activeSource,
      activeAction: active.actionName,
      heuristicAction: heuristic.actionName,
      heuristicFlags: heuristic.flags,
      neuralAction: neural?.actionName ?? null,
      neuralFlags: neural?.flags ?? null,
      shadowDisagrees: neural ? neural.flags !== heuristic.flags : false,
      inferenceMs,
      checkpointPath: this.model?.path ?? null,
      checkpointHash: this.model?.hash ?? null,
      featureSet: this.model?.metadata?.featureSet ?? null,
      candidateScores: neural?.candidateScores ?? null,
      candidateProbabilities: neural?.candidateProbabilities ?? null,
      safetyGate: neural?.safetyGate ?? null,
      safetyRejectReasons: neural?.safetyRejectReasons ?? [],
    };
  }

  modelDecision(player, target, state) {
    const scores = this.scoreCandidates(state);
    scores.sort((a, b) => b.score - a.score);
    const bestRaw = scores[0];
    const velocityAlign = this.velocityAlignDecision(player, target, state, scores);
    if (velocityAlign) return velocityAlign;
    const cruise = this.goalCruiseDecision(player, target, state, scores);
    if (cruise) return cruise;
    const best = scores.find((item) => this.goalActionAllowed(state, item.action)) ?? bestRaw;
    const bestAction = best.action;
    return {
      flags: bestAction.flags,
      actionName: bestAction.name,
      target,
      obstacle: state.obstacle,
      dist: state.dist,
      headingError: state.headingError,
      velocityHeadingError: state.velocityHeadingError,
      speed: state.speed,
      forwardSpeed: state.forwardSpeed,
      lateralSpeed: state.lateralSpeed,
      closingSpeed: state.closingSpeed,
      velocityGoalCos: state.velocityGoalCos,
      brakeDistance: state.brakeDistance,
      policy: "model",
      modelHash: this.model.hash,
      modelScore: best.score,
      maskedAction: bestAction !== bestRaw.action ? bestRaw.action.name : null,
      safetyGate: bestAction !== bestRaw.action ? "goal_action_mask" : "pass",
      safetyRejectReasons: bestAction !== bestRaw.action ? ["goal_action_mask"] : [],
      scoreTop: scores.slice(0, 3).map((item) => ({
        action: item.action.name,
        score: round6(item.score),
      })),
      candidateScores: scores.map((item) => ({
        action: item.action.name,
        score: round6(item.score),
      })),
      candidateProbabilities: scoreProbabilities(scores),
    };
  }

  scoreCandidates(state) {
    return SIGNAL_FLIGHT_ACTIONS.map((action) => ({
      action,
      score: forwardSignalFlightModel(this.model, this.modelFeatures(state, action)),
    }));
  }

  recoveryDecision(player, target, state) {
    if (!this.isObstacleContact(state) || !state.obstacle) return null;
    const desired = state.obstacle.escapeAngle ?? wrapAngle(player.angle + Math.PI);
    const headingError = wrapAngle(desired - player.angle);
    const turnFlag = turnFlagForHeading(headingError);
    const thrustFlag = Math.abs(headingError) < 2.15 || state.speed < 20 ? NET_INPUT_THRUST : 0;
    const flags = turnFlag | thrustFlag || turnFlag || NET_INPUT_THRUST;
    return {
      flags,
      actionName: flagsToActionName(flags),
      target,
      obstacle: state.obstacle,
      dist: state.dist,
      headingError,
      velocityHeadingError: state.velocityHeadingError,
      speed: state.speed,
      forwardSpeed: state.forwardSpeed,
      lateralSpeed: state.lateralSpeed,
      closingSpeed: state.closingSpeed,
      velocityGoalCos: state.velocityGoalCos,
      brakeDistance: state.brakeDistance,
      policy: "recovery",
      modelHash: this.model.hash,
      modelScore: undefined,
      maskedAction: "CONTACT",
      safetyGate: "obstacle_contact_recovery",
      safetyRejectReasons: ["obstacle_contact_recovery"],
      scoreTop: null,
    };
  }

  velocityAlignDecision(player, target, state, scores) {
    if (this.isObstacleContact(state) || state.speed < 25 || state.forwardSpeed > -6) {
      return null;
    }
    const turnFlag = turnFlagForHeading(state.velocityHeadingError, 0.12);
    if (!turnFlag) return null;
    const actionName = flagsToActionName(turnFlag);
    const bestRaw = scores[0];
    const chosenScore = scores.find((item) => item.action.name === actionName)?.score;
    return {
      flags: turnFlag,
      actionName,
      target,
      obstacle: state.obstacle,
      dist: state.dist,
      headingError: state.headingError,
      velocityHeadingError: state.velocityHeadingError,
      speed: state.speed,
      forwardSpeed: state.forwardSpeed,
      lateralSpeed: state.lateralSpeed,
      closingSpeed: state.closingSpeed,
      velocityGoalCos: state.velocityGoalCos,
      brakeDistance: state.brakeDistance,
      policy: "velocity_align",
      modelHash: this.model.hash,
      modelScore: chosenScore,
      maskedAction: bestRaw.action.name,
      safetyGate: "velocity_align_guard",
      safetyRejectReasons: ["velocity_align_guard"],
      scoreTop: scores.slice(0, 3).map((item) => ({
        action: item.action.name,
        score: round6(item.score),
      })),
      candidateScores: scores.map((item) => ({
        action: item.action.name,
        score: round6(item.score),
      })),
      candidateProbabilities: scoreProbabilities(scores),
    };
  }

  goalCruiseDecision(player, target, state, scores) {
    const forcedWaypoint = target?.navKind === "station_escape" || target?.navKind === "station_bypass";
    if (this.isObstacleContact(state) || (!forcedWaypoint && state.dist < 700)) return null;

    const bestRaw = scores[0];
    const shouldCruise =
      bestRaw.action.thrust <= 0 ||
      forcedWaypoint ||
      state.speed < 65 ||
      state.closingSpeed < 20 ||
      (state.dist > 1600 && state.speed < 110);
    if (!shouldCruise) return null;

    const absHeadingError = Math.abs(state.headingError);
    if (!this.cruiseTurnLocked && absHeadingError > 0.1 && absHeadingError < 2.75) {
      this.cruiseTurnSign = state.headingError > 0 ? 1 : -1;
    }
    if (absHeadingError > 2.75) {
      this.cruiseTurnLocked = true;
    } else if (absHeadingError < 1.15) {
      this.cruiseTurnLocked = false;
    }

    let turnFlag = turnFlagForHeading(state.headingError, 0.1);
    if (this.cruiseTurnLocked && absHeadingError > 1.15) {
      turnFlag = turnFlagForHeading(this.cruiseTurnSign, 0);
    }
    const now = Date.now();
    const thrustWindow = state.speed < 20 ? 1.35 : 0.95;
    let thrustFlag = absHeadingError < thrustWindow ? NET_INPUT_THRUST : 0;
    if (thrustFlag) {
      if (now >= this.burnHoldUntilMs) {
        this.burnHoldUntilMs = now + 450;
        this.burnHoldTurnFlag = turnFlag;
      } else {
        turnFlag = this.burnHoldTurnFlag;
      }
    } else if (now >= this.burnHoldUntilMs) {
      this.burnHoldTurnFlag = 0;
    }
    if (state.forwardSpeed < -18 && state.speed > 18 && state.closingSpeed < 12) {
      thrustFlag = 0;
    }
    const flags = turnFlag | thrustFlag || turnFlag || NET_INPUT_THRUST;
    const actionName = flagsToActionName(flags);
    const chosenScore = scores.find((item) => item.action.name === actionName)?.score;

    return {
      flags,
      actionName,
      target,
      obstacle: state.obstacle,
      dist: state.dist,
      headingError: state.headingError,
      velocityHeadingError: state.velocityHeadingError,
      speed: state.speed,
      forwardSpeed: state.forwardSpeed,
      lateralSpeed: state.lateralSpeed,
      closingSpeed: state.closingSpeed,
      velocityGoalCos: state.velocityGoalCos,
      brakeDistance: state.brakeDistance,
      policy: "goal_cruise",
      modelHash: this.model.hash,
      modelScore: chosenScore,
      maskedAction: bestRaw.action.name,
      safetyGate: "goal_cruise_guard",
      safetyRejectReasons: ["goal_cruise_guard"],
      scoreTop: scores.slice(0, 3).map((item) => ({
        action: item.action.name,
        score: round6(item.score),
      })),
      candidateScores: scores.map((item) => ({
        action: item.action.name,
        score: round6(item.score),
      })),
      candidateProbabilities: scoreProbabilities(scores),
    };
  }

  goalActionAllowed(state, action) {
    if (this.isObstacleContact(state)) {
      if (action.id === 0) return false;
      return action.thrust < 0 || action.turn !== 0;
    }
    if (state.dist <= 450 || state.speed >= 20) return true;
    if (action.id === 0 || action.thrust < 0) return false;
    if (Math.abs(state.headingError) > 0.35 && action.thrust > 0 && action.turn === 0) {
      return false;
    }
    return true;
  }

  isObstacleContact(state) {
    if (!state.obstacle) return false;
    return state.obstacle.forward < 120 && state.obstacle.clearance < -4;
  }

  flightState(player, target) {
    const dx = target.x - player.x;
    const dy = target.y - player.y;
    const dist = len(dx, dy);
    const desired = Math.atan2(dy, dx);
    const headingError = wrapAngle(desired - player.angle);
    const forwardX = Math.cos(player.angle);
    const forwardY = Math.sin(player.angle);
    const rightX = -forwardY;
    const rightY = forwardX;
    const speed = len(player.vx, player.vy);
    const forwardSpeed = player.vx * forwardX + player.vy * forwardY;
    const lateralSpeed = player.vx * rightX + player.vy * rightY;
    const closingSpeed = dist > 0 ? (player.vx * dx + player.vy * dy) / dist : 0;
    const velocityHeadingError = speed > 0.5 ? wrapAngle(Math.atan2(player.vy, player.vx) - player.angle) : 0;
    const velocityGoalCos = speed > 0.5 && dist > 0 ? (player.vx * dx + player.vy * dy) / (speed * dist) : 0;
    const brakeDistance = (speed * speed) / (2 * SHIP_BRAKE);
    const obstacle = this.findForwardObstacle(player, forwardX, forwardY, rightX, rightY, brakeDistance);
    const signal = this.signalStrength(player);
    const fwdClear = this.forwardClearance(player, player.angle);
    const leftClear = this.forwardClearance(player, player.angle + 0.7);
    const rightClear = this.forwardClearance(player, player.angle - 0.7);
    const velClear = speed > 0.5
      ? this.forwardClearance(player, Math.atan2(player.vy, player.vx))
      : fwdClear;

    return {
      dist,
      controlDist: dist,
      headingError,
      headingCos: Math.cos(headingError),
      headingSin: Math.sin(headingError),
      controlHeadingCos: Math.cos(headingError),
      controlHeadingSin: Math.sin(headingError),
      speed,
      forwardSpeed,
      lateralSpeed,
      closingSpeed,
      velocityHeadingError,
      velocityGoalCos,
      brakeDistance,
      fwdClear,
      leftClear,
      rightClear,
      velClear,
      signal,
      hullRatio: this.ship.hull > 0 ? clip(this.ship.hull / 100, 0, 1) : 1,
      pathCount: 0,
      pathCurrent: 0,
      obstacle,
    };
  }

  modelFeatures(state, action) {
    const dist = scale(state.dist, 6000);
    const controlDist = scale(state.controlDist, 6000);
    const speed = signedScale(state.speed, 350);
    const forwardSpeed = signedScale(state.forwardSpeed, 350);
    const lateralSpeed = signedScale(state.lateralSpeed, 350);
    const brakeDistance = scale(state.brakeDistance, 700);
    const fwdClear = clip(state.fwdClear, 0, 1);
    const leftClear = clip(state.leftClear, 0, 1);
    const rightClear = clip(state.rightClear, 0, 1);
    const velClear = clip(state.velClear, 0, 1);
    const signal = clip(state.signal, 0, 1);
    const hull = clip(state.hullRatio, 0, 1);
    const pathCount = scale(state.pathCount, 16);
    const pathCurrent = scale(state.pathCurrent, 16);
    const progress = 0;
    const nextDist = dist;
    const nextSpeed = speed;
    const closeGoal = 1 - dist;
    const turn = action.turn;
    const thrust = action.thrust;
    const headingCos = clip(state.headingCos, -1, 1);
    const headingSin = clip(state.headingSin, -1, 1);
    const controlHeadingCos = clip(state.controlHeadingCos, -1, 1);
    const controlHeadingSin = clip(state.controlHeadingSin, -1, 1);

    const row = new Float64Array(48);
    row[0] = 1;
    row[1] = action.id / (SIGNAL_FLIGHT_ACTIONS.length - 1);
    row[2] = turn;
    row[3] = thrust;
    row[4] = turn < 0 ? 1 : 0;
    row[5] = turn > 0 ? 1 : 0;
    row[6] = thrust > 0 ? 1 : 0;
    row[7] = thrust < 0 ? 1 : 0;
    row[8] = action.id === 0 ? 1 : 0;
    row[9] = dist;
    row[10] = controlDist;
    row[11] = headingCos;
    row[12] = headingSin;
    row[13] = controlHeadingCos;
    row[14] = controlHeadingSin;
    row[15] = speed;
    row[16] = forwardSpeed;
    row[17] = lateralSpeed;
    row[18] = brakeDistance;
    row[19] = fwdClear;
    row[20] = leftClear;
    row[21] = rightClear;
    row[22] = velClear;
    row[23] = signal;
    row[24] = hull;
    row[25] = pathCount;
    row[26] = pathCurrent;
    row[27] = progress;
    row[28] = nextDist;
    row[29] = nextSpeed;
    row[30] = turn * controlHeadingSin;
    row[31] = thrust * controlHeadingCos;
    row[32] = thrust * fwdClear;
    row[33] = (thrust < 0 ? 1 : 0) * brakeDistance;
    row[34] = (turn < 0 ? 1 : 0) * leftClear;
    row[35] = (turn > 0 ? 1 : 0) * rightClear;
    row[36] = (action.id === 0 ? 1 : 0) * Math.abs(speed);
    row[37] = turn * lateralSpeed;
    row[38] = thrust * forwardSpeed;
    row[39] = thrust * (1 - hull);
    row[40] = thrust * nextSpeed;
    row[41] = thrust * progress;
    row[42] = thrust * closeGoal;
    row[43] = turn * headingSin;
    row[44] = thrust * headingCos;
    row[45] = (thrust < 0 ? 1 : 0) * forwardSpeed;
    row[46] = Math.abs(turn);
    row[47] = Math.abs(thrust);
    return row;
  }

  targetFor(player) {
    if (this.config.goal) {
      const target = {
        ...this.config.goal,
        objectiveKind: "fixed_goal",
        objectiveId: 0,
      };
      return this.plotWaypoint(player, target);
    }

    const objective = this.currentObjective(player);
    if (!objective) return null;
    const target = {
      x: objective.x,
      y: objective.y,
      station: objective.stationId,
      objectiveKind: objective.kind,
      objectiveId: objective.id,
      waypoint: objective.waypoint,
      completeRadius: objective.completeRadius,
    };
    return this.plotWaypoint(player, target);
  }

  plotWaypoint(player, target) {
    const key = this.targetKey(target);
    const now = Date.now();
    if (this.navWaypoint && this.navWaypoint.targetKey !== key) {
      this.navWaypoint = null;
    }
    if (this.navWaypoint) {
      const d = len(this.navWaypoint.x - player.x, this.navWaypoint.y - player.y);
      if (d > this.navWaypoint.completeRadius && now < this.navWaypoint.expiresAt) {
        return this.decorateWaypointTarget(target, this.navWaypoint);
      }
      this.navWaypoint = null;
    }

    const waypoint = this.computeWaypoint(player, target, key, now);
    if (!waypoint) return target;
    this.navWaypoint = waypoint;
    return this.decorateWaypointTarget(target, waypoint);
  }

  targetKey(target) {
    return [
      target.objectiveKind ?? "target",
      target.objectiveId ?? 0,
      target.waypoint ?? 0,
      Math.round(target.x),
      Math.round(target.y),
    ].join(":");
  }

  decorateWaypointTarget(target, waypoint) {
    return {
      ...target,
      x: waypoint.x,
      y: waypoint.y,
      objectiveKind: waypoint.navKind,
      parentObjectiveKind: target.objectiveKind,
      navKind: waypoint.navKind,
      navId: waypoint.id,
      finalX: target.x,
      finalY: target.y,
      completeRadius: waypoint.completeRadius,
      escapeRadius: waypoint.escapeRadius ?? null,
      stationDistance: waypoint.stationDistance ?? null,
      blocker: waypoint.blocker ?? null,
    };
  }

  computeWaypoint(player, target, key, now) {
    const stationWaypoint = this.stationWaypoint(player, target, key, now);
    if (stationWaypoint) return stationWaypoint;
    const asteroidWaypoint = this.asteroidWaypoint(player, target, key, now);
    if (asteroidWaypoint) return asteroidWaypoint;
    return null;
  }

  stationWaypoint(player, target, key, now) {
    const containingStation = this.containingStation(player);
    if (containingStation) {
      return this.makeStationBypassWaypoint(player, target, containingStation, key, now, "station_escape");
    }

    let best = null;
    for (const station of this.stations.values()) {
      const clearance = this.stationClearanceRadius(station);
      const hit = segmentProjection(station.x, station.y, player.x, player.y, target.x, target.y);
      if (hit.t <= 0.03 || hit.t >= 0.97 || hit.dist >= clearance) continue;
      if (!best || hit.t < best.hit.t) best = { station, hit };
    }
    if (!best) return null;
    return this.makeStationBypassWaypoint(player, target, best.station, key, now, "station_bypass");
  }

  containingStation(player) {
    for (const station of this.stations.values()) {
      const stationDist = len(player.x - station.x, player.y - station.y);
      if (stationDist < this.stationClearanceRadius(station)) {
        return { station, stationDist };
      }
    }
    return null;
  }

  makeStationBypassWaypoint(player, target, stationRef, key, now, navKind) {
    const station = stationRef.station ?? stationRef;
    const stationDist = stationRef.stationDist ?? len(player.x - station.x, player.y - station.y);
    const dx = player.x - station.x;
    const dy = player.y - station.y;
    const tx = target.x - station.x;
    const ty = target.y - station.y;
    const basis = stationDist > 1 ? { x: dx, y: dy } : { x: tx, y: ty };
    const escapeRadius = this.stationClearanceRadius(station);
    const offset = navKind === "station_escape" ? 0.95 : 0.72;
    const escapeAngle = Math.atan2(basis.y, basis.x) + this.stationEscapeTurnSign * offset;
    return {
      id: this.navSerial++,
      targetKey: key,
      navKind,
      x: station.x + Math.cos(escapeAngle) * escapeRadius,
      y: station.y + Math.sin(escapeAngle) * escapeRadius,
      completeRadius: NAV_WAYPOINT_REACHED_RADIUS,
      expiresAt: now + NAV_WAYPOINT_TTL_MS,
      escapeRadius,
      stationDistance: stationDist,
      blocker: {
        kind: "station",
        station: station.id,
        clearance: round3(escapeRadius),
      },
    };
  }

  stationClearanceRadius(station) {
    return Math.max(
      850,
      (station.radius || 0) + STATION_ESCAPE_PADDING,
      (station.dockRadius || 0) + STATION_ESCAPE_PADDING * 0.7,
    );
  }

  asteroidWaypoint(player, target, key, now) {
    let best = null;
    for (const asteroid of this.asteroids.values()) {
      if (now - asteroid.seenAt > 3000) continue;
      const hit = segmentProjection(asteroid.x, asteroid.y, player.x, player.y, target.x, target.y);
      if (hit.t <= 0.04 || hit.t >= 0.92) continue;
      const pathDist = len(hit.x - player.x, hit.y - player.y);
      if (pathDist > NAV_ASTEROID_LOOKAHEAD) continue;
      const clearance = Math.max(NAV_ASTEROID_CLEARANCE, (asteroid.radius || 0) + 220);
      if (hit.dist >= clearance) continue;
      if (!best || pathDist < best.pathDist) best = { asteroid, hit, pathDist, clearance };
    }
    if (!best) return null;

    const dx = target.x - player.x;
    const dy = target.y - player.y;
    const pathLen = Math.max(1, len(dx, dy));
    const side = this.stationEscapeTurnSign;
    const nx = (-dy / pathLen) * side;
    const ny = (dx / pathLen) * side;
    return {
      id: this.navSerial++,
      targetKey: key,
      navKind: "asteroid_bypass",
      x: best.asteroid.x + nx * best.clearance,
      y: best.asteroid.y + ny * best.clearance,
      completeRadius: NAV_WAYPOINT_REACHED_RADIUS,
      expiresAt: now + Math.min(NAV_WAYPOINT_TTL_MS, 9000),
      blocker: {
        kind: "asteroid",
        index: best.asteroid.index,
        clearance: round3(best.clearance),
      },
    };
  }

  currentObjective(player) {
    if (this.objective && this.objectiveComplete(player, this.objective)) {
      this.advanceObjective(player);
    }
    if (!this.objective) {
      this.startObjective(player);
    }
    return this.objective;
  }

  objectiveComplete(player, objective) {
    return len(objective.x - player.x, objective.y - player.y) <= objective.completeRadius;
  }

  startObjective(player) {
    const station = this.selectedStation(player);
    if (!station) return null;
    this.objective = this.stationWaypointObjective(station, 0);
    this.logObjective("started", this.objective);
    return this.objective;
  }

  advanceObjective(player) {
    const previous = this.objective;
    const station = this.stations.get(previous.stationId) ?? this.selectedStation(player);
    if (!station) {
      this.objective = null;
      return null;
    }
    const waypoint = (previous.waypoint + 1) % OBJECTIVE_WAYPOINTS_PER_STATION;
    this.objective = this.stationWaypointObjective(station, waypoint);
    this.logObjective("advanced", this.objective);
    return this.objective;
  }

  stationWaypointObjective(station, waypoint) {
    const phase = this.sessionToken[0] / 255 * Math.PI * 2;
    const angle = phase + waypoint * (Math.PI * 2 / OBJECTIVE_WAYPOINTS_PER_STATION);
    return {
      kind: "station_patrol_waypoint",
      id: this.objectiveSerial++,
      stationId: station.id,
      stationName: station.name,
      waypoint,
      x: station.x + Math.cos(angle) * this.config.radius,
      y: station.y + Math.sin(angle) * this.config.radius,
      completeRadius: Math.max(OBJECTIVE_COMPLETE_RADIUS, Math.min(700, this.config.radius * 0.25)),
    };
  }

  logObjective(event, objective) {
    console.log(
      `[brain] objective ${event} #${objective.id} ${objective.kind} station=${objective.stationId} waypoint=${objective.waypoint} target=(${objective.x.toFixed(0)},${objective.y.toFixed(0)})`,
    );
  }

  selectedStation(player) {
    if (this.config.station !== null) {
      return this.stations.get(this.config.station) ?? null;
    }
    let best = null;
    let bestDist = Infinity;
    for (const station of this.stations.values()) {
      const d = len(station.x - player.x, station.y - player.y);
      if (d < bestDist) {
        best = station;
        bestDist = d;
      }
    }
    return best;
  }

  findForwardObstacle(player, forwardX, forwardY, rightX, rightY, brakeDistance) {
    let best = null;
    const horizon = Math.max(420, Math.min(1000, brakeDistance + 260));
    const now = Date.now();
    for (const asteroid of this.asteroids.values()) {
      if (now - asteroid.seenAt > 3000) continue;
      const rx = asteroid.x - player.x;
      const ry = asteroid.y - player.y;
      const fwd = rx * forwardX + ry * forwardY;
      if (fwd < 0 || fwd > horizon) continue;
      const lateral = rx * rightX + ry * rightY;
      const clearance = Math.abs(lateral) - (asteroid.radius + 85);
      if (clearance > 0) continue;
      if (!best || fwd < best.forward) {
        best = {
          index: asteroid.index,
          forward: fwd,
          lateral,
          clearance,
          side: lateral >= 0 ? -1 : 1,
          escapeAngle: Math.atan2(player.y - asteroid.y, player.x - asteroid.x),
        };
      }
    }
    return best;
  }

  forwardClearance(player, heading) {
    const forwardX = Math.cos(heading);
    const forwardY = Math.sin(heading);
    const perpX = -forwardY;
    const perpY = forwardX;
    const speed = len(player.vx, player.vy);
    const lookahead = Math.max(100, Math.min(speed * 1.5, 500));
    const shipRadius = 24;
    const now = Date.now();
    let worst = 1;

    for (const asteroid of this.asteroids.values()) {
      if (now - asteroid.seenAt > 3000) continue;
      const rx = asteroid.x - player.x;
      const ry = asteroid.y - player.y;
      const fd = rx * forwardX + ry * forwardY;
      if (fd < -asteroid.radius || fd > lookahead) continue;
      const lateral = Math.abs(rx * perpX + ry * perpY);
      const clearance = asteroid.radius + shipRadius + 20;
      if (lateral < clearance) {
        worst = Math.min(worst, fd / lookahead);
      }
    }

    for (const station of this.stations.values()) {
      const rx = station.x - player.x;
      const ry = station.y - player.y;
      const fd = rx * forwardX + ry * forwardY;
      const clearance = this.stationClearanceRadius(station) + shipRadius;
      if (fd < -clearance || fd > lookahead) continue;
      const lateral = Math.abs(rx * perpX + ry * perpY);
      if (lateral >= clearance) continue;
      const along = Math.sqrt(Math.max(0, clearance * clearance - lateral * lateral));
      const entry = fd - along;
      const exit = fd + along;
      if (entry >= 0) {
        worst = Math.min(worst, entry / lookahead);
      } else if (exit >= 0) {
        worst = 0;
      }
    }

    return clip(worst, 0, 1);
  }

  signalStrength(player) {
    let best = 0;
    let overlaps = 0;
    for (const station of this.stations.values()) {
      if (!(station.signalRange > 0)) continue;
      const dist = len(station.x - player.x, station.y - player.y);
      const strength = Math.max(0, 1 - dist / station.signalRange);
      if (strength > 0) overlaps += 1;
      if (strength > best) best = strength;
    }
    if (overlaps <= 1) return clip(best, 0, 1);
    return clip(best * Math.min(overlaps, 3), 0, 1);
  }

  sendInput(flags, action, player) {
    this.inputSeq = (this.inputSeq + 1) & 0xffff;
    if (this.inputSeq === 0) this.inputSeq = 1;
    const actionId = action === NET_ACTION_NONE ? 0 : this.nextActionId();
    const inputTick = player.serverTick ? player.serverTick + 1 : 0;
    const buffer = Buffer.alloc(18);
    buffer[0] = NET_MSG_INPUT;
    buffer[1] = flags;
    buffer[2] = action;
    buffer[3] = 0xff;
    buffer[4] = 0xff;
    buffer[5] = 0xff;
    buffer[6] = 0xff;
    buffer[7] = 0xff;
    writeU16(buffer, 8, this.inputSeq);
    writeU16(buffer, 10, 0xffff);
    writeU16(buffer, 12, actionId);
    writeU32(buffer, 14, inputTick);
    this.send(buffer);
    this.lastFlags = flags;
  }

  nextActionId() {
    const id = this.actionId;
    this.actionId = (this.actionId + 1) & 0xffff;
    if (this.actionId === 0) this.actionId = 1;
    return id;
  }

  send(buffer) {
    if (!this.ws || this.ws.readyState !== SignalWebSocket.OPEN) return;
    this.ws.send(buffer);
  }

  async logDecision(now, player, decision, action) {
    if (!this.config.logPath || now - this.lastLogAt < 200) return;
    this.lastLogAt = now;
    const row = {
      schema: "crlp.signal_live_brain.v1",
      t_ms: now,
      player_id: this.localId,
      dry_run: this.config.dryRun,
      policy: decision.policy ?? "unknown",
      brain_mode: decision.brainMode ?? "heuristic",
      active_source: decision.activeSource ?? "heuristic",
      model_hash: decision.modelHash ?? null,
      checkpoint_path: decision.checkpointPath ?? null,
      checkpoint_hash: decision.checkpointHash ?? null,
      feature_set: decision.featureSet ?? null,
      action: decision.actionName,
      active_action: decision.activeAction ?? decision.actionName,
      heuristic_action: decision.heuristicAction ?? null,
      heuristic_flags: decision.heuristicFlags ?? null,
      neural_action: decision.neuralAction ?? null,
      neural_flags: decision.neuralFlags ?? null,
      shadow_disagrees: decision.shadowDisagrees ?? false,
      flags: decision.flags,
      one_shot_action: action,
      x: round3(player.x),
      y: round3(player.y),
      vx: round3(player.vx),
      vy: round3(player.vy),
      angle: round6(player.angle),
      player_flags: player.flags,
      server_tick: player.serverTick ?? 0,
      input_tick_ack: player.inputTickAck ?? 0,
      hull: round3(this.ship.hull),
      docked: this.ship.docked,
      target: decision.target
        ? {
            x: round3(decision.target.x),
            y: round3(decision.target.y),
            station: decision.target.station ?? null,
            objective_kind: decision.target.objectiveKind ?? null,
            parent_objective_kind: decision.target.parentObjectiveKind ?? null,
            nav_kind: decision.target.navKind ?? null,
            nav_id: decision.target.navId ?? null,
            objective_id: decision.target.objectiveId ?? null,
            waypoint: decision.target.waypoint ?? null,
            final_x: decision.target.finalX !== undefined ? round3(decision.target.finalX) : null,
            final_y: decision.target.finalY !== undefined ? round3(decision.target.finalY) : null,
            escape_radius: decision.target.escapeRadius ? round3(decision.target.escapeRadius) : null,
            station_distance: decision.target.stationDistance ? round3(decision.target.stationDistance) : null,
            blocker: decision.target.blocker ?? null,
          }
        : null,
      dist: round3(decision.dist),
      heading_error: round6(decision.headingError),
      velocity_heading_error: round6(decision.velocityHeadingError ?? 0),
      speed: round3(decision.speed ?? 0),
      forward_speed: round3(decision.forwardSpeed ?? 0),
      lateral_speed: round3(decision.lateralSpeed ?? 0),
      closing_speed: round3(decision.closingSpeed ?? 0),
      velocity_goal_cos: round6(decision.velocityGoalCos ?? 0),
      obstacle: decision.obstacle,
      model_score: decision.modelScore !== undefined ? round6(decision.modelScore) : null,
      masked_action: decision.maskedAction ?? null,
      score_top: decision.scoreTop ?? null,
      candidate_scores: decision.candidateScores ?? null,
      candidate_probabilities: decision.candidateProbabilities ?? null,
      safety_gate: decision.safetyGate ?? null,
      safety_reject_reasons: decision.safetyRejectReasons ?? [],
      inference_ms: round6(decision.inferenceMs ?? 0),
      stations: this.stations.size,
      asteroids: this.asteroids.size,
    };
    await appendFile(this.config.logPath, `${JSON.stringify(row)}\n`);
  }

  maybeStatus(now, text) {
    if (now - this.lastStatusAt < 1000) return;
    this.lastStatusAt = now;
    console.log(`[brain] ${text}`);
  }

  statusText(player, decision, action) {
    const stationText = decision.target?.station !== undefined ? ` station=${decision.target.station}` : "";
    const objectiveText = decision.target?.objectiveId ? ` obj=${decision.target.objectiveId}` : "";
    const actionText = action === NET_ACTION_LAUNCH ? " +LAUNCH" : "";
    const policyText =
      decision.policy === "model"
        ? " model"
        : decision.policy === "recovery"
          ? " recovery"
          : decision.policy === "velocity_align"
            ? " align"
          : decision.policy === "goal_cruise"
            ? " cruise"
            : "";
    const scoreText = decision.scoreTop ? ` top=${decision.scoreTop.map((item) => item.action).join("/")}` : "";
    const maskText = decision.maskedAction ? ` masked=${decision.maskedAction}` : "";
    const modeText = decision.brainMode ? ` mode=${decision.brainMode}/${decision.activeSource}` : "";
    const shadowText = decision.shadowDisagrees ? ` shadow=${decision.neuralAction}` : "";
    return `${decision.actionName}${actionText}${policyText}${modeText}${shadowText}${maskText}${scoreText} pos=(${player.x.toFixed(0)},${player.y.toFixed(0)}) dist=${decision.dist.toFixed(0)} heading=${decision.headingError.toFixed(2)}${stationText}${objectiveText}`;
  }
}

function flagsToActionName(flags) {
  const parts = [];
  if (flags & NET_INPUT_THRUST) parts.push("W");
  if (flags & NET_INPUT_BRAKE) parts.push("S");
  if (flags & NET_INPUT_LEFT) parts.push("A");
  if (flags & NET_INPUT_RIGHT) parts.push("D");
  return parts.length === 0 ? "NONE" : parts.join("");
}

function round3(value) {
  return Math.round(value * 1000) / 1000;
}

function round6(value) {
  return Math.round(value * 1000000) / 1000000;
}

async function main() {
  const config = parseArgs(process.argv);
  const sessionToken = await loadSessionToken(config.sessionPath);
  const model = await loadSignalFlightModel(config.checkpointPath);
  const brain = new SignalLiveBrain(config, sessionToken, model);
  await brain.run();
}

main().catch((err) => {
  console.error(`[brain] ${err.message}`);
  process.exit(1);
});
