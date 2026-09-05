#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, statSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SELF = fileURLToPath(import.meta.url);
const SHA = /^[a-f0-9]{64}$/;
const COMMIT = /^[a-f0-9]{40}$/;
const ID = /^[A-Za-z0-9._:/-]+$/;
const fail = (message) => { throw new Error(message); };
export const sha256File = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const digest = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));
const writeJson = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });

function absoluteFile(path, label) {
  if (typeof path !== 'string' || !isAbsolute(path) || !statSync(path).isFile()) fail(`${label} requires an absolute file`);
  return path;
}

function command(program, args, options = {}) {
  const result = spawnSync(program, args, { encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, ...options });
  if (result.error || result.status !== 0) fail(`${program} failed: ${result.error?.message || result.stderr || `exit ${result.status}`}`);
  return result.stdout;
}

function checkout(binding, label) {
  if (!binding || !isAbsolute(binding.root || '') || !COMMIT.test(binding.commit || '')) fail(`${label} requires an absolute root and full commit`);
  const actual = command('git', ['-C', binding.root, 'rev-parse', 'HEAD']).trim();
  if (actual !== binding.commit) fail(`${label} commit drift: ${actual}`);
}

export function validateConfig(config) {
  if (!config || !ID.test(config.id || '') || typeof config.title !== 'string' || !config.title.trim()) fail('id and title are required');
  if (!Number.isSafeInteger(config.seed) || config.seed < 0) fail('seed must be a nonnegative integer');
  checkout(config.source, 'source');
  if (typeof config.source.repository !== 'string' || !config.source.repository.trim()) fail('source.repository is required');
  checkout(config.ilxyr, 'ilxyr');
  absoluteFile(config.ilxyr.cli, 'ilxyr.cli');
  absoluteFile(config.runner?.program, 'runner.program');
  if (!Array.isArray(config.runner.args) || config.runner.args.some((arg) => typeof arg !== 'string')) fail('runner.args must be strings');
  if (!isAbsolute(config.runner.cwd || '') || !statSync(config.runner.cwd).isDirectory()) fail('runner.cwd requires an absolute directory');
  if (!Number.isSafeInteger(config.runner.timeout_seconds) || config.runner.timeout_seconds < 1 || config.runner.timeout_seconds > 300) fail('runner timeout must be between 1 and 300 seconds');
  if (!Array.isArray(config.inputs) || !config.inputs.length) fail('inputs must bind the prepared data and runner');
  if (!Array.isArray(config.outputs) || !config.outputs.length) fail('outputs must declare result files');
  for (const [kind, entries] of [['inputs', config.inputs], ['outputs', config.outputs]]) {
    const names = new Set();
    const paths = new Set();
    for (const item of entries) {
      if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(item.name || '') || names.has(item.name)) fail(`${kind} names must be unique simple names`);
      if (!isAbsolute(item.path || '') || paths.has(resolve(item.path))) fail(`${kind} paths must be unique absolute paths`);
      names.add(item.name);
      paths.add(resolve(item.path));
      if (kind === 'inputs') {
        absoluteFile(item.path, item.name);
        if (!SHA.test(item.sha256 || '') || sha256File(item.path) !== item.sha256) fail(`input hash mismatch: ${item.name}`);
      }
    }
  }
  const inputPaths = new Set(config.inputs.map((input) => realpathSync(input.path)));
  absoluteFile(config.runner.args[0], 'runner script');
  if (!inputPaths.has(realpathSync(config.runner.args[0]))) fail('inputs must hash-bind the runner script');
  for (const output of config.outputs) {
    if (inputPaths.has(resolve(output.path))) fail(`output overlaps an input: ${output.name}`);
  }
  if (!Array.isArray(config.metrics) || !config.metrics.length) fail('metrics are required');
  const metricNames = new Set();
  for (const metric of config.metrics) {
    if (!ID.test(metric.name || '') || metricNames.has(metric.name) || !metric.description || typeof metric.unit !== 'string') fail('metrics require unique names, units, and descriptions');
    metricNames.add(metric.name);
  }
  if (!metricNames.has(config.outcome?.primary_metric) || !Number.isFinite(config.outcome.threshold)) fail('outcome requires a declared metric and finite threshold');
  return config;
}

export function validateMetrics(metrics, config) {
  if (!metrics || Array.isArray(metrics) || typeof metrics !== 'object') fail('runner metrics must be an object');
  const actual = Object.keys(metrics).sort();
  const expected = config.metrics.map((metric) => metric.name).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected) || Object.values(metrics).some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
    fail('runner metrics must exactly match the frozen finite numeric metric set');
  }
  return metrics;
}

function verifyBindings(bindings) {
  for (const binding of bindings) {
    if (sha256File(binding.path) !== binding.sha256) fail(`bound file changed: ${binding.name}`);
  }
}

function executeFrozen(path, expectedHash) {
  if (!SHA.test(expectedHash || '') || sha256File(path) !== expectedHash) fail('frozen configuration hash mismatch');
  const frozen = readJson(path);
  const config = validateConfig(frozen.config);
  verifyBindings(frozen.bindings);
  if (sha256File(SELF) !== frozen.adapter_sha256) fail('frozen adapter changed');
  for (const output of config.outputs) if (existsSync(output.path)) fail(`output already exists: ${output.name}`);
  const result = spawnSync(config.runner.program, config.runner.args, {
    cwd: config.runner.cwd,
    env: {},
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024,
    timeout: config.runner.timeout_seconds * 1000,
  });
  writeFileSync(join(dirname(path), 'runner.stdout.log'), result.stdout || '', { flag: 'wx' });
  writeFileSync(join(dirname(path), 'runner.stderr.log'), result.stderr || '', { flag: 'wx' });
  if (result.error || result.status !== 0) fail(`demo runner failed: ${result.error?.message || `exit ${result.status}`}; see runner.stderr.log`);
  const payload = JSON.parse(result.stdout);
  if (JSON.stringify(Object.keys(payload)) !== JSON.stringify(['metrics'])) fail('runner stdout must contain only the metrics envelope');
  const metrics = validateMetrics(payload.metrics, config);
  verifyBindings(frozen.bindings);
  const outputs = config.outputs.map((output) => {
    absoluteFile(output.path, output.name);
    return { name: `outputs/${output.name}`, path: output.path, sha256: sha256File(output.path), bytes: statSync(output.path).size };
  });
  const artifacts = [...frozen.bindings, ...outputs, { name: 'contract/frozen.json', path, sha256: expectedHash }];
  writeJson(join(dirname(path), 'artifact-bindings.json'), artifacts);
  const source = {
    repository: config.source.repository,
    commit: config.source.commit,
    artifacts: artifacts.map(({ name, sha256 }) => ({ path: name, sha256 })),
  };
  process.stdout.write(JSON.stringify({ metrics, source }) + '\n');
}

function nativePackage(config, frozenPath, frozenHash, adapterPath) {
  const actor = (name) => ({ id: `service://signal-demo/${name}`, kind: 'service' });
  const stages = ['hypothesis', 'mathematical_foundation', 'engineering_review', 'experiment_design'];
  const descriptions = [
    'Measure whether the neural controller has higher matched rollout utility than flight_steer_to.',
    'The primary quantity is neural mean utility minus the matched heuristic mean utility. A strict positive difference resolves improved; zero or a negative difference resolves not_improved.',
    'Automated input review checks exact source commits, file hashes, explicit executable paths, finite metric names, fresh output files, and the local time limit. This is a local protocol fixture with one adapter, not an independent scientific review.',
    `Frozen local simulation design and complete file bindings: ${JSON.stringify(config)}. Training starts only through ilXyr after compilation and admission. Fixture credits and forecasts exercise the local accounting path; they represent no payment or independent forecast evidence.`,
  ];
  const contributions = stages.map((stage, index) => ({
    schema: 'ilxyr.contribution.v1', id: `${config.id}.${stage}`, stage,
    actor: actor(stage === 'engineering_review' ? 'input-checker' : 'contract-renderer'),
    title: `Signal local demo ${stage}`, body: descriptions[index],
    input_refs: stages.slice(0, index).map((prior) => `${config.id}.${prior}`),
    claims: ['This contribution records the declared local demo procedure.'], confidence: 1,
  }));
  const metricPredicate = (operator) => ({ kind: 'metric', metric: config.outcome.primary_metric, operator, threshold: config.outcome.threshold });
  const experiment = {
    schema: 'ilxyr.experiment.v1', id: config.id, title: config.title,
    hypothesis: descriptions[0], rationale: 'A bounded local simulation study with an immutable input and output contract. Fixture accounting exercises ilXyr admission.',
    proposer: actor('contract-renderer'),
    lineage: Object.fromEntries(stages.map((stage) => [stage, `${config.id}.${stage}`])),
    baseline: 'baseline://signal/flight_steer_to',
    datasets: [`dataset://signal-demo/${digest(config.inputs)}`], models: [],
    metrics: config.metrics, seeds: [config.seed],
    outcome_contract: {
      primary_metric: config.outcome.primary_metric, success_outcome: 'improved',
      outcomes: [
        { id: 'improved', description: 'The measured utility difference is strictly above the frozen threshold.', predicate: metricPredicate('gt') },
        { id: 'not_improved', description: 'The measured utility difference is at or below the frozen threshold.', predicate: metricPredicate('lte') },
        { id: 'execution_failure', description: 'The runner fails, times out, or violates its metric contract.', predicate: { kind: 'execution_failure' } },
      ],
    },
    execution: { executor: 'local-command', program: process.execPath, args: [adapterPath, '--internal-execute', frozenPath, '--expected-sha256', frozenHash], timeout_seconds: config.runner.timeout_seconds + 15, max_cost_credits: 1, network: 'open' },
    funding: { required_compute_credits: 1, minimum_forecasters: 1, minimum_total_stake: 1 },
    security: { weight_class: 'public', code_policy: 'arbitrary', export_policy: 'artifacts' },
    evidence_authority: { level: 'deterministic_replay', scope: { seeds: [config.seed], coverage: 1 }, provenance: { artifact_hashes: [], checker: 'checker://signal-demo/frozen-local-run/v1' } },
    expected_outputs: [...config.metrics.map(({ name }) => `metrics.${name}`), 'resolved_outcome', 'forecast_settlements', 'source_attestation.artifacts'],
  };
  const forecast = { schema: 'ilxyr.forecast.v1', id: `${config.id}.fixture-forecast`, experiment_id: config.id, forecaster: actor('fixture-forecast'), probabilities: { improved: 0.45, not_improved: 0.45, execution_failure: 0.1 }, stake: 1, rationale: 'Fixed illustrative local protocol fixture. These probabilities are not an independent forecast or a research conclusion.' };
  const funding = { schema: 'ilxyr.funding.v1', id: `${config.id}.fixture-credit`, experiment_id: config.id, funder: actor('fixture-credit'), compute_credits: 1, rationale: 'One local fixture accounting credit. Monetary value and external payment are zero.' };
  return { contributions, experiment, forecast, funding };
}

export function runDemo(configPath, output, execute = false) {
  const config = validateConfig(readJson(configPath));
  const outputRoot = resolve(output);
  if (existsSync(outputRoot) && readdirSync(outputRoot).length) fail('choose a fresh ilXyr output directory');
  for (const artifact of config.outputs) if (existsSync(artifact.path)) fail(`output already exists: ${artifact.name}`);
  mkdirSync(outputRoot, { recursive: true });
  const adapterPath = join(outputRoot, 'signal_demo_ilxyr.mjs');
  copyFileSync(SELF, adapterPath);
  const bindings = [
    ...config.inputs.map((input) => ({ ...input, name: `inputs/${input.name}` })),
    ...[['runtime/node', process.execPath], ['runtime/runner', config.runner.program], ['runtime/ilxyr', config.ilxyr.cli], ['contract/adapter', adapterPath]].map(([name, path]) => ({ name, path, sha256: sha256File(path) })),
  ];
  const frozenPath = join(outputRoot, 'frozen-config.json');
  writeJson(frozenPath, { schema: 'crlp.signal_demo_ilxyr_frozen.v1', config, bindings, adapter_sha256: sha256File(adapterPath), node_version: process.version, accounting: 'one local fixture credit; zero monetary value' });
  const frozenHash = sha256File(frozenPath);
  const packageFiles = nativePackage(config, frozenPath, frozenHash, adapterPath);
  const ledger = join(outputRoot, 'ledger');
  const steps = [];
  function ilxyr(name, args) {
    const result = JSON.parse(command(config.ilxyr.cli, args));
    steps.push(name);
    writeJson(join(outputRoot, `${name}.json`), result);
    return result;
  }
  ilxyr('init', ['init', ledger]);
  for (const contribution of packageFiles.contributions) {
    const path = join(outputRoot, `${contribution.stage}.json`);
    writeJson(path, contribution);
    ilxyr(`contribute-${contribution.stage}`, ['contribute', ledger, path]);
  }
  for (const name of ['experiment', 'forecast', 'funding']) writeJson(join(outputRoot, `${name}.json`), packageFiles[name]);
  ilxyr('compile', ['compile', ledger, join(outputRoot, 'experiment.json')]);
  ilxyr('forecast-record', ['forecast', ledger, join(outputRoot, 'forecast.json')]);
  ilxyr('fund', ['fund', ledger, join(outputRoot, 'funding.json')]);
  const admission = ilxyr('admit', ['admit', ledger, config.id]);
  if (!admission.accepted) fail(`ilXyr admission rejected: ${JSON.stringify(admission.checks.filter((check) => !check.passed))}`);
  verifyBindings(bindings);
  if (execute) ilxyr('run', ['run', ledger, config.id, '--execute']);
  const status = ilxyr('status', ['status', ledger, config.id]);
  const verification = ilxyr('verify', ['verify', ledger]);
  if (!verification.valid) fail('ilXyr ledger verification failed');
  let evidenceRef = null;
  if (execute) {
    const events = readFileSync(join(ledger, '.ilxyr/events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    evidenceRef = events.findLast((event) => event.event_type === 'EvidenceRecorded' && event.aggregate_id === config.id)?.artifact_ref;
    if (!evidenceRef) fail('ilXyr evidence reference is missing');
    ilxyr('evidence', ['export-evidence', ledger, evidenceRef, 'native']);
  }
  const receipt = { schema: 'crlp.signal_demo_ilxyr_receipt.v1', experiment_id: config.id, executed: execute, outcome: status.latest_evidence?.resolved_outcome ?? null, metrics: status.latest_run?.metrics ?? {}, frozen_config_sha256: frozenHash, ledger, evidence_ref: evidenceRef, verification, steps, accounting: 'one local fixture credit; zero monetary value' };
  writeJson(join(outputRoot, 'receipt.json'), receipt);
  if (execute && receipt.outcome === 'execution_failure') fail(`experiment recorded execution_failure: ${status.latest_run?.stderr || status.latest_run?.output_error || 'runner failed'}`);
  return receipt;
}

export function main(argv = process.argv.slice(2)) {
  if (argv[0] === '--internal-execute' && argv[2] === '--expected-sha256') return executeFrozen(argv[1], argv[3]);
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--execute') options.execute = true;
    else if (['--config', '--output'].includes(arg) && argv[index + 1]) options[arg.slice(2)] = argv[++index];
    else fail(`unexpected argument: ${arg}`);
  }
  if (!options.config || !options.output) fail('usage: signal_demo_ilxyr.mjs --config CONFIG --output FRESH_DIRECTORY [--execute]');
  process.stdout.write(JSON.stringify(runDemo(options.config, options.output, options.execute), null, 2) + '\n');
}

if (process.argv[1] && realpathSync(process.argv[1]) === realpathSync(SELF)) {
  try { main(); } catch (error) { console.error(`signal-demo-ilxyr: ${error.message}`); process.exitCode = 1; }
}
