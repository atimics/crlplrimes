import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256File, validateMetrics } from './signal_demo_ilxyr.mjs';

const adapter = fileURLToPath(new URL('./signal_demo_ilxyr.mjs', import.meta.url));
const ilxyr = process.env.ILXYR_TEST_CLI;
const ilxyrRoot = process.env.ILXYR_TEST_REPO;
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));

test('metric contract rejects missing, extra, and nonfinite values', () => {
  const config = { metrics: [{ name: 'utility_delta' }] };
  assert.deepEqual(validateMetrics({ utility_delta: -1 }, config), { utility_delta: -1 });
  for (const metrics of [{}, { utility_delta: 1, extra: 0 }, { utility_delta: Infinity }, { utility_delta: '1' }]) {
    assert.throws(() => validateMetrics(metrics, config), /metric/);
  }
});

function fixture(delta, tamper = false) {
  const root = mkdtempSync(join(tmpdir(), 'signal-ilxyr-test-'));
  const input = join(root, 'train.csv');
  const runner = join(root, 'runner.mjs');
  const outputFile = join(root, 'result.json');
  const output = join(root, 'ilxyr-run');
  writeFileSync(input, 'x,y\n1,2\n');
  writeFileSync(runner, `import {readFileSync,writeFileSync} from 'node:fs';
const events=readFileSync(process.argv[3]+'/.ilxyr/events.jsonl','utf8').trim().split('\\n').map(JSON.parse);
for(const type of ['ExperimentCompiled','AdmissionDecided','ExecutionStarted']) if(!events.some(e=>e.event_type===type)) throw Error(type+' must precede the runner');
${tamper ? `writeFileSync(${JSON.stringify(input)},'changed');` : ''}
writeFileSync(process.argv[2],JSON.stringify({delta:${delta}})+'\\n');
process.stdout.write(JSON.stringify({metrics:{utility_delta:${delta},train_rows:1}})+'\\n');
`);
  execFileSync('git', ['init', '-q', root]);
  execFileSync('git', ['-C', root, 'add', 'train.csv', 'runner.mjs']);
  execFileSync('git', ['-C', root, '-c', 'user.name=Demo Test', '-c', 'user.email=demo@example.invalid', 'commit', '-qm', 'Fixture']);
  const commit = execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const config = {
    id: `signal-demo.test.${delta > 0 ? 'positive' : 'negative'}.${tamper ? 'tamper' : 'valid'}`,
    title: 'Native ilXyr local fixture', seed: 4242,
    source: { root, repository: 'fixture://signal-demo', commit },
    ilxyr: { root: resolve(ilxyrRoot), cli: resolve(ilxyr), commit: execFileSync('git', ['-C', ilxyrRoot, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim() },
    runner: { program: process.execPath, args: [runner, outputFile, join(output, 'ledger')], cwd: root, timeout_seconds: 5 },
    inputs: [{ name: 'train.csv', path: input, sha256: sha256File(input) }, { name: 'runner.mjs', path: runner, sha256: sha256File(runner) }],
    outputs: [{ name: 'result.json', path: outputFile }],
    metrics: [{ name: 'utility_delta', unit: 'utility', description: 'Neural minus heuristic utility.' }, { name: 'train_rows', unit: 'rows', description: 'Training rows.' }],
    outcome: { primary_metric: 'utility_delta', threshold: 0 },
  };
  const configPath = join(root, 'config.json');
  writeFileSync(configPath, JSON.stringify(config));
  return { config, configPath, output, input, outputFile };
}

for (const delta of [0.25, 0, -0.25]) {
  test(`native ilXyr freezes before execution and settles utility delta ${delta}`, { skip: !ilxyr || !ilxyrRoot }, () => {
    const f = fixture(delta);
    const result = spawnSync(process.execPath, [adapter, '--config', f.configPath, '--output', f.output, '--execute'], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const receipt = JSON.parse(result.stdout);
    assert.equal(receipt.outcome, delta > 0 ? 'improved' : 'not_improved');
    assert.equal(receipt.verification.valid, true);
    const evidence = readJson(join(f.output, 'evidence.json'));
    assert.equal(evidence.run.metrics.utility_delta, delta);
    assert.ok(evidence.settlements.length);
    const artifact = evidence.run.source_attestation.artifacts.find((item) => item.path === 'outputs/result.json');
    assert.equal(artifact.sha256, sha256File(f.outputFile));
    const repeated = spawnSync(process.execPath, [adapter, '--config', f.configPath, '--output', f.output, '--execute'], { encoding: 'utf8' });
    assert.equal(repeated.status, 1);
    assert.match(repeated.stderr, /fresh ilXyr output/);
  });
}

test('changed input blocks preparation before ledger creation', { skip: !ilxyr || !ilxyrRoot }, () => {
  const f = fixture(1);
  writeFileSync(f.input, 'changed');
  const result = spawnSync(process.execPath, [adapter, '--config', f.configPath, '--output', f.output, '--execute'], { encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /input hash mismatch/);
});

test('input mutation by runner records a verified execution failure', { skip: !ilxyr || !ilxyrRoot }, () => {
  const f = fixture(1, true);
  const result = spawnSync(process.execPath, [adapter, '--config', f.configPath, '--output', f.output, '--execute'], { encoding: 'utf8' });
  assert.equal(result.status, 1);
  const receipt = readJson(join(f.output, 'receipt.json'));
  assert.equal(receipt.outcome, 'execution_failure');
  assert.equal(receipt.verification.valid, true);
  assert.match(result.stderr, /bound file changed/);
});
