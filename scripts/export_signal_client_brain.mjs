#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { createHash } from "node:crypto";

const NN_ACTIVATION_SIGMOID = 0;
const NN_ACTIVATION_RELU = 1;
const NN_ACTIVATION_TANH = 2;
const NN_ACTIVATION_LINEAR = 3;
const NN_ACTIVATION_EML = 4;
const NN_ACTIVATION_EML_PAIRWISE = 5;

function usage() {
  console.log(`usage: node scripts/export_signal_client_brain.mjs --checkpoint PATH --out DIR [options]

Options:
  --name NAME                 artifact name, e.g. flight (default from checkpoint)
  --symbol SYMBOL             C symbol prefix, e.g. signal_client_flight
  --require-feature-set NAME  fail unless checkpoint metadata matches
  --require-feature-version N fail unless checkpoint metadata matches
  --emit-readme               write README.md in output dir
`);
}

function parseArgs(argv) {
  const config = {
    checkpoint: null,
    outDir: null,
    name: null,
    symbol: null,
    requireFeatureSet: null,
    requireFeatureVersion: null,
    emitReadme: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const value = argv[i + 1];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--checkpoint" && value) {
      config.checkpoint = value;
      i += 1;
    } else if (arg === "--out" && value) {
      config.outDir = value;
      i += 1;
    } else if (arg === "--name" && value) {
      config.name = value;
      i += 1;
    } else if (arg === "--symbol" && value) {
      config.symbol = value;
      i += 1;
    } else if (arg === "--require-feature-set" && value) {
      config.requireFeatureSet = value;
      i += 1;
    } else if (arg === "--require-feature-version" && value) {
      const parsed = Number(value);
      if (!Number.isInteger(parsed) || parsed < 0) {
        throw new Error("--require-feature-version must be a non-negative integer");
      }
      config.requireFeatureVersion = parsed;
      i += 1;
    } else if (arg === "--emit-readme") {
      config.emitReadme = true;
    } else {
      throw new Error(`unknown or incomplete option: ${arg}`);
    }
  }
  if (!config.checkpoint) throw new Error("--checkpoint is required");
  if (!config.outDir) throw new Error("--out is required");
  config.name = sanitizeIdent(config.name ?? basename(config.checkpoint, ".nnckpt"));
  config.symbol = sanitizeIdent(config.symbol ?? `signal_client_${config.name}`);
  return config;
}

function sanitizeIdent(text) {
  const ident = String(text ?? "")
    .replace(/[^A-Za-z0-9_]/g, "_")
    .replace(/^[^A-Za-z_]+/, "");
  if (!ident) throw new Error(`invalid identifier: ${text}`);
  return ident;
}

function readCString(buffer, offset, len) {
  const end = offset + len;
  let n = offset;
  while (n < end && buffer[n] !== 0) n += 1;
  return buffer.toString("utf8", offset, n);
}

function loadCheckpoint(path, checkpointBytes) {
  const bytes = Buffer.from(checkpointBytes);
  let offset = 0;
  const need = (n) => {
    if (offset + n > bytes.length) throw new Error(`truncated checkpoint: ${path}`);
  };
  const read = (n) => {
    need(n);
    const start = offset;
    offset += n;
    return bytes.subarray(start, start + n);
  };
  const readU32 = () => {
    const b = read(4);
    return b.readUInt32LE(0);
  };
  const readU64 = () => {
    const b = read(8);
    const v = b.readBigUInt64LE(0);
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`checkpoint integer is too large: ${path}`);
    }
    return Number(v);
  };
  const readScalar = (scalarType) => {
    if (scalarType === 1) return read(4).readFloatLE(0);
    if (scalarType === 2) return read(8).readDoubleLE(0);
    throw new Error(`unsupported checkpoint scalar type: ${scalarType}`);
  };
  const readFixedString = (len) => {
    const start = offset;
    read(len);
    return readCString(bytes, start, len);
  };

  const magic = read(8).toString("ascii");
  if (magic !== "NNCKPT01") throw new Error(`bad checkpoint magic: ${path}`);
  const checkpointVersion = readU32();
  const scalarType = readU32();
  const layerCount = readU64();
  const sizes = [];
  for (let i = 0; i < layerCount; i += 1) sizes.push(readU64());
  const hiddenActivation = readU32();
  const outputActivation = readU32();
  const loss = readU32();
  const weights = [];
  const biases = [];
  for (let layer = 1; layer < layerCount; layer += 1) {
    const previous = sizes[layer - 1];
    const current = sizes[layer];
    const layerWeights = new Array(previous * current);
    const layerBiases = new Array(current);
    for (let i = 0; i < layerWeights.length; i += 1) {
      layerWeights[i] = readScalar(scalarType);
    }
    for (let i = 0; i < layerBiases.length; i += 1) {
      layerBiases[i] = readScalar(scalarType);
    }
    weights.push(layerWeights);
    biases.push(layerBiases);
  }
  const metadataSize = readU64();
  if (metadataSize !== 455 && metadataSize !== 487) {
    throw new Error(`unsupported checkpoint metadata size ${metadataSize}: ${path}`);
  }
  const metadata = {
    featureEncoderVersion: readU32(),
    parentModelHash: "",
    trainingTraceRoot: "",
    benchmarkReportHash: "",
    notes: "",
    featureSet: "",
  };
  metadata.parentModelHash = readFixedString(65);
  metadata.trainingTraceRoot = readFixedString(65);
  metadata.benchmarkReportHash = readFixedString(65);
  metadata.notes = readFixedString(256);
  if (metadataSize === 487) metadata.featureSet = readFixedString(32);

  return {
    path,
    hash: sha256Hex(bytes),
    scalarType,
    sizes,
    hiddenActivation,
    outputActivation,
    loss,
    weights,
    biases,
    metadata,
  };
}

function sha256Hex(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

function activationName(kind) {
  switch (kind) {
    case NN_ACTIVATION_SIGMOID: return "sigmoid";
    case NN_ACTIVATION_RELU: return "relu";
    case NN_ACTIVATION_TANH: return "tanh";
    case NN_ACTIVATION_LINEAR: return "linear";
    case NN_ACTIVATION_EML: return "eml";
    case NN_ACTIVATION_EML_PAIRWISE: return "eml_pairwise";
    default: return `unknown_${kind}`;
  }
}

function cFloat(value) {
  if (!Number.isFinite(value)) throw new Error(`non-finite weight: ${value}`);
  if (Object.is(value, -0)) return "-0.0f";
  if (value === 0) return "0.0f";
  const text = Number(value).toPrecision(9);
  return `${text.includes(".") || text.includes("e") ? text : `${text}.0`}f`;
}

function cString(text) {
  return JSON.stringify(String(text));
}

function formatArray(name, values) {
  const lines = [];
  lines.push(`static const float ${name}[${values.length}] SIGNAL_CLIENT_ALIGN64 = {`);
  for (let i = 0; i < values.length; i += 8) {
    const chunk = values.slice(i, i + 8).map(cFloat).join(", ");
    lines.push(`    ${chunk}${i + 8 < values.length ? "," : ""}`);
  }
  lines.push("};");
  return lines.join("\n");
}

function headerGuard(symbol) {
  return `${symbol.toUpperCase()}_H`;
}

function emitHeader(model, config) {
  const guard = headerGuard(config.symbol);
  const sizes = model.sizes;
  return `/* @generated by scripts/export_signal_client_brain.mjs. Do not edit by hand. */
#ifndef ${guard}
#define ${guard}

#include <stddef.h>

#define ${config.symbol.toUpperCase()}_LAYER_COUNT ${sizes.length}u
#define ${config.symbol.toUpperCase()}_INPUT_COUNT ${sizes[0]}u
#define ${config.symbol.toUpperCase()}_OUTPUT_COUNT ${sizes[sizes.length - 1]}u
#define ${config.symbol.toUpperCase()}_HIDDEN0 ${sizes[1] ?? 0}u
#define ${config.symbol.toUpperCase()}_HIDDEN1 ${sizes[2] ?? 0}u

extern const char *const ${config.symbol}_checkpoint_hash;
extern const char *const ${config.symbol}_feature_set;
extern const unsigned ${config.symbol}_feature_encoder_version;

float ${config.symbol}_score(const float features[${sizes[0]}]);
int ${config.symbol}_select_best(const float *features,
                                 size_t candidate_count,
                                 float *scores_out);

#endif /* ${guard} */
`;
}

function activationExpr(symbol, value, activation) {
  switch (activation) {
    case NN_ACTIVATION_SIGMOID:
      return `${symbol}_sigmoid(${value})`;
    case NN_ACTIVATION_RELU:
      return `${symbol}_relu(${value})`;
    case NN_ACTIVATION_TANH:
      return `tanhf(${value})`;
    case NN_ACTIVATION_LINEAR:
      return value;
    case NN_ACTIVATION_EML:
      return `${symbol}_eml(${value})`;
    case NN_ACTIVATION_EML_PAIRWISE:
      throw new Error("eml_pairwise activation is not supported by the Signal client exporter");
    default:
      throw new Error(`unsupported activation kind: ${activation}`);
  }
}

function emitC(model, config) {
  const sizes = model.sizes;
  if (sizes.length < 2) throw new Error("checkpoint must have at least two layers");
  const prefix = config.symbol;
  const arrays = [];
  for (let layer = 0; layer < model.weights.length; layer += 1) {
    arrays.push(formatArray(`${prefix}_w${layer}`, model.weights[layer]));
    arrays.push(formatArray(`${prefix}_b${layer}`, model.biases[layer]));
  }

  const usedActivations = new Set([
    model.hiddenActivation,
    model.outputActivation,
  ]);
  const activationHelpers = [];
  if (usedActivations.has(NN_ACTIVATION_RELU)) {
    activationHelpers.push(`SIGNAL_CLIENT_ALWAYS_INLINE float ${prefix}_relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}`);
  }
  if (usedActivations.has(NN_ACTIVATION_SIGMOID)) {
    activationHelpers.push(`SIGNAL_CLIENT_ALWAYS_INLINE float ${prefix}_sigmoid(float x)
{
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}`);
  }
  if (usedActivations.has(NN_ACTIVATION_EML)) {
    activationHelpers.push(`SIGNAL_CLIENT_ALWAYS_INLINE float ${prefix}_eml(float x)
{
    const float ex = expf(x);
    return ex - logf(1.0f + ex);
}`);
  }

  const layerFns = [];
  for (let layer = 0; layer < model.weights.length; layer += 1) {
    const previous = sizes[layer];
    const current = sizes[layer + 1];
    const act = layer + 1 === sizes.length - 1 ? model.outputActivation : model.hiddenActivation;
    layerFns.push(`SIGNAL_CLIENT_ALWAYS_INLINE void ${prefix}_layer${layer}(const float *SIGNAL_CLIENT_RESTRICT src,
                                                    float *SIGNAL_CLIENT_RESTRICT dst)
{
    for (size_t neuron = 0; neuron < ${current}u; ++neuron) {
        float sum = ${prefix}_b${layer}[neuron];
        const float *SIGNAL_CLIENT_RESTRICT w = &${prefix}_w${layer}[neuron * ${previous}u];
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
        for (size_t i = 0; i < ${previous}u; ++i) {
            sum += w[i] * src[i];
        }
        dst[neuron] = ${activationExpr(prefix, "sum", act)};
    }
}`);
  }

  const scratchDecls = sizes.slice(1).map((size, i) => `    float h${i}[${size}] = {0};`).join("\n");
  const layerCalls = model.weights.map((_, layer) => {
    const src = layer === 0 ? "features" : `h${layer - 1}`;
    const dst = `h${layer}`;
    return `    ${prefix}_layer${layer}(${src}, ${dst});`;
  }).join("\n");
  const outputName = `h${model.weights.length - 1}`;

  return `/* @generated by scripts/export_signal_client_brain.mjs. Do not edit by hand. */
#include "${prefix}.h"

#include <math.h>
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define SIGNAL_CLIENT_ALWAYS_INLINE static inline __attribute__((always_inline))
#define SIGNAL_CLIENT_ALIGN64 __attribute__((aligned(64)))
#define SIGNAL_CLIENT_RESTRICT __restrict__
#else
#define SIGNAL_CLIENT_ALWAYS_INLINE static inline
#define SIGNAL_CLIENT_ALIGN64
#define SIGNAL_CLIENT_RESTRICT
#endif

const char *const ${prefix}_checkpoint_hash = ${cString(model.hash)};
const char *const ${prefix}_feature_set = ${cString(model.metadata.featureSet)};
const unsigned ${prefix}_feature_encoder_version = ${model.metadata.featureEncoderVersion}u;

${activationHelpers.join("\n\n")}

${arrays.join("\n\n")}

${layerFns.join("\n\n")}

float ${prefix}_score(const float features[${sizes[0]}])
{
${scratchDecls}
${layerCalls}
    return ${outputName}[0];
}

int ${prefix}_select_best(const float *features, size_t candidate_count, float *scores_out)
{
    if (features == 0 || candidate_count == 0u) return -1;
    int best = -1;
    float best_score = -INFINITY;
    for (size_t i = 0; i < candidate_count; ++i) {
        const float score = ${prefix}_score(&features[i * ${sizes[0]}u]);
        if (scores_out != 0) scores_out[i] = score;
        if (best < 0 || score > best_score) {
            best = (int)i;
            best_score = score;
        }
    }
    return best;
}
`;
}

function emitManifest(model, config, files) {
  return `${JSON.stringify({
    schema: "crlp.signal_client_brain_bundle.v1",
    name: config.name,
    symbol: config.symbol,
    checkpoint_path: resolve(config.checkpoint),
    checkpoint_hash: model.hash,
    scalar_type: model.scalarType === 1 ? "float32" : "float64",
    sizes: model.sizes,
    hidden_activation: activationName(model.hiddenActivation),
    output_activation: activationName(model.outputActivation),
    feature_encoder_version: model.metadata.featureEncoderVersion,
    feature_set: model.metadata.featureSet,
    notes: model.metadata.notes,
    files,
    generated_at: new Date().toISOString(),
    authority: "checkpoint_to_static_c_export",
    hot_path_contract: [
      "no checkpoint parser",
      "no heap allocation",
      "stack scratch buffers",
      "static const neuron arrays",
    ],
  }, null, 2)}\n`;
}

function emitReadme(model, config, files) {
  return `# Signal Client Brain Bundle

Generated C inference bundle for Signal.

- Checkpoint: \`${resolve(config.checkpoint)}\`
- Checkpoint hash: \`${model.hash}\`
- Feature set: \`${model.metadata.featureSet || "unknown"}\`
- Feature encoder version: \`${model.metadata.featureEncoderVersion}\`
- Shape: \`${model.sizes.join(" -> ")}\`
- Files: ${files.map((file) => `\`${file}\``).join(", ")}

The hot path uses static neuron arrays and stack scratch buffers. It does not
parse \`.nnckpt\`, allocate heap memory, or write temporary checkpoint files.
Regenerate this directory from CRLPLRIMES after retraining the checkpoint.
`;
}

async function main() {
  const config = parseArgs(process.argv);
  const checkpointBytes = await readFile(config.checkpoint);
  const model = loadCheckpoint(config.checkpoint, checkpointBytes);
  if (config.requireFeatureSet !== null && model.metadata.featureSet !== config.requireFeatureSet) {
    throw new Error(`feature_set mismatch: got ${model.metadata.featureSet}, want ${config.requireFeatureSet}`);
  }
  if (config.requireFeatureVersion !== null &&
      model.metadata.featureEncoderVersion !== config.requireFeatureVersion) {
    throw new Error(
      `feature_encoder_version mismatch: got ${model.metadata.featureEncoderVersion}, want ${config.requireFeatureVersion}`,
    );
  }

  await mkdir(config.outDir, { recursive: true });
  const headerFile = `${config.symbol}.h`;
  const sourceFile = `${config.symbol}.c`;
  const manifestFile = `${config.symbol}.manifest.json`;
  await writeFile(join(config.outDir, headerFile), emitHeader(model, config));
  await writeFile(join(config.outDir, sourceFile), emitC(model, config));
  await writeFile(join(config.outDir, manifestFile), emitManifest(model, config, [headerFile, sourceFile, manifestFile]));
  if (config.emitReadme) {
    await writeFile(join(config.outDir, "README.md"), emitReadme(model, config, [headerFile, sourceFile, manifestFile]));
  }
  console.log(JSON.stringify({
    out_dir: resolve(config.outDir),
    header: join(resolve(config.outDir), headerFile),
    source: join(resolve(config.outDir), sourceFile),
    manifest: join(resolve(config.outDir), manifestFile),
    checkpoint_hash: model.hash,
    feature_set: model.metadata.featureSet,
    feature_encoder_version: model.metadata.featureEncoderVersion,
    sizes: model.sizes,
  }, null, 2));
}

main().catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
