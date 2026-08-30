#!/usr/bin/env node

import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { sha256File, validateManifest } from "./ilxyr_cloud_bridge.mjs";

const digest = "a".repeat(64);
const commit = "b".repeat(40);
const scriptsDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptsDirectory, "..");

function manifest() {
  return {
    schema: "crlp.ilxyr_cloud_job.v1",
    id: "bridge-smoke-v1",
    purpose: "infrastructure_smoke",
    claim_ids: [],
    grounding_authority: "exact_check",
    source: {
      repository: "https://github.com/atimics/crlplrimes.git",
      commit: "CURRENT",
      artifacts: [{ path: "scripts/ilxyr_cloud_bridge.mjs", sha256: digest }],
    },
    control_plane: {
      repository: "https://github.com/cenetex/ilxyr.git",
      commit,
      family_manifest: "integrations/ilxyr/families/bridge-smoke/family.json",
    },
    corpus: { kind: "none" },
    outputs: { directory: "build/ilxyr/bridge-smoke-v1", retention_days: 30 },
  };
}

assert.equal(validateManifest(manifest()).id, "bridge-smoke-v1");

{
  const smokeJobPath = join(repositoryRoot, "integrations", "ilxyr", "jobs", "bridge-smoke.json");
  const smokeJob = validateManifest(JSON.parse(readFileSync(smokeJobPath, "utf8")));
  assert.equal(smokeJob.purpose, "infrastructure_smoke");
  assert.deepEqual(smokeJob.claim_ids, []);
  const artifact = smokeJob.source.artifacts[0];
  assert.equal(sha256File(join(repositoryRoot, artifact.path)), artifact.sha256);
}

{
  const value = manifest();
  value.purpose = "experiment";
  value.source.commit = "short";
  assert.throws(() => validateManifest(value), /full lowercase Git commit/u);
}

{
  const value = manifest();
  value.grounding_authority = "corpus_proxy";
  assert.throws(() => validateManifest(value), /verified Braid release/u);
}

{
  const value = manifest();
  value.source.artifacts[0].path = "../outside";
  assert.throws(() => validateManifest(value), /safe relative path/u);
}

{
  const value = manifest();
  value.extra = true;
  assert.throws(() => validateManifest(value), /job.extra is not allowed/u);
}

{
  const value = manifest();
  value.grounding_authority = "corpus_proxy";
  value.corpus = {
    kind: "braid_release",
    repository: "https://github.com/cenetex/braid.git",
    commit: "c".repeat(40),
    release_id: "braid-example-v1-0123456789abcdef",
    release_manifest_sha256: "d".repeat(64),
    release_directory: "inputs/braid/example-v1",
    materialization: { kind: "existing" },
  };
  assert.equal(validateManifest(value).corpus.kind, "braid_release");
}

function run(program, args, cwd) {
  const result = spawnSync(program, args, { cwd, encoding: "utf8" });
  assert.equal(result.status, 0, `${program} ${args.join(" ")}\n${result.stderr || result.stdout}`);
  return result.stdout.trim();
}

function initRepo(path) {
  mkdirSync(path, { recursive: true });
  run("git", ["init", "-q"], path);
  run("git", ["config", "user.email", "bridge-test@example.invalid"], path);
  run("git", ["config", "user.name", "Bridge Test"], path);
}

function commitAll(path) {
  run("git", ["add", "."], path);
  run("git", ["commit", "-q", "-m", "fixture"], path);
  return run("git", ["rev-parse", "HEAD"], path);
}

const temporaryRoot = mkdtempSync(join(tmpdir(), "crlp-ilxyr-bridge-test-"));
try {
  const sourceRepo = join(temporaryRoot, "crlplrimes");
  const ilxyrRepo = join(temporaryRoot, "ilxyr");
  const braidRepo = join(temporaryRoot, "braid");
  initRepo(sourceRepo);
  initRepo(ilxyrRepo);
  initRepo(braidRepo);

  const fakeIlxyr = join(ilxyrRepo, "ilxyr.mjs");
  writeFileSync(fakeIlxyr, `#!/usr/bin/env node
import { mkdirSync } from "node:fs";
if (process.argv[2] === "init") mkdirSync(process.argv[3], { recursive: true });
if (process.argv[2] === "family" && process.argv[3] === "check") {
  const ready = process.env.ILXYR_REJECT !== "1";
  process.stdout.write(JSON.stringify({
    ready,
    members: ready ? [] : [{ experiment_id: "fixture.rejected.v1", ready: false, failed_gates: ["fixture_gate"] }],
  }));
}
`);
  chmodSync(fakeIlxyr, 0o755);
  const ilxyrCommit = commitAll(ilxyrRepo);

  const fakeBraid = join(braidRepo, "braid.mjs");
  writeFileSync(fakeBraid, "process.exitCode = 0;\n");
  const braidCommit = commitAll(braidRepo);

  const familyDirectory = join(sourceRepo, "integrations", "ilxyr", "families", "bridge-smoke");
  mkdirSync(familyDirectory, { recursive: true });
  writeFileSync(join(familyDirectory, "family.json"), `${JSON.stringify({
    commit: "__CRLP_COMMIT__",
    root: "__CRLP_ROOT__",
    release: "__BRAID_RELEASE_ID__",
    artifact: "__SOURCE_ARTIFACT_0_SHA256__",
  }, null, 2)}\n`);
  const sourceArtifact = join(sourceRepo, "scripts", "runner.txt");
  mkdirSync(dirname(sourceArtifact), { recursive: true });
  writeFileSync(sourceArtifact, "frozen runner bytes\n");

  const releaseDirectory = join(sourceRepo, "inputs", "braid", "bridge-release");
  mkdirSync(releaseDirectory, { recursive: true });
  const braidSpec = join(sourceRepo, "inputs", "braid", "bridge.braid.yaml");
  writeFileSync(braidSpec, "apiVersion: braid/v1\nkind: DatasetBuild\n");
  const releaseId = "braid-bridge-v1-0123456789abcdef";
  const releaseManifest = join(releaseDirectory, "release.json");
  writeFileSync(releaseManifest, `${JSON.stringify({ releaseId })}\n`);

  const jobPath = join(sourceRepo, "integrations", "ilxyr", "bridge-job.json");
  const job = manifest();
  job.source.artifacts[0].path = "scripts/runner.txt";
  job.source.artifacts[0].sha256 = sha256File(sourceArtifact);
  job.control_plane.commit = ilxyrCommit;
  job.corpus = {
    kind: "braid_release",
    repository: "https://github.com/cenetex/braid.git",
    commit: braidCommit,
    release_id: releaseId,
    release_manifest_sha256: sha256File(releaseManifest),
    release_directory: "inputs/braid/bridge-release",
    materialization: {
      kind: "build",
      spec_path: "inputs/braid/bridge.braid.yaml",
      output_root: "inputs/braid",
    },
  };
  writeFileSync(jobPath, `${JSON.stringify(job, null, 2)}\n`);
  const rejectedJobPath = join(sourceRepo, "integrations", "ilxyr", "bridge-rejected-job.json");
  const rejectedJob = structuredClone(job);
  rejectedJob.id = "bridge-rejected-v1";
  rejectedJob.outputs.directory = "build/ilxyr/bridge-rejected-v1";
  writeFileSync(rejectedJobPath, `${JSON.stringify(rejectedJob, null, 2)}\n`);
  const sourceCommit = commitAll(sourceRepo);

  const bridgePath = resolve(scriptsDirectory, "ilxyr_cloud_bridge.mjs");
  const result = spawnSync(process.execPath, [
    bridgePath,
    "--manifest", jobPath,
    "--repo", sourceRepo,
    "--ilxyr-repo", ilxyrRepo,
    "--ilxyr-cli", fakeIlxyr,
    "--braid-repo", braidRepo,
    "--braid-cli", fakeBraid,
  ], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const receipt = JSON.parse(readFileSync(join(sourceRepo, "build", "ilxyr", "bridge-smoke-v1", "bridge-receipt.json"), "utf8"));
  assert.equal(receipt.source.commit, sourceCommit);
  assert.equal(receipt.corpus.release_id, releaseId);
  assert.deepEqual(receipt.lifecycle, ["freeze", "check"]);
  const rendered = JSON.parse(readFileSync(receipt.control_plane.rendered_family_manifest, "utf8"));
  assert.equal(rendered.commit, sourceCommit);
  assert.equal(rendered.release, releaseId);
  assert.equal(rendered.artifact, job.source.artifacts[0].sha256);

  const rejected = spawnSync(process.execPath, [
    bridgePath,
    "--manifest", rejectedJobPath,
    "--repo", sourceRepo,
    "--ilxyr-repo", ilxyrRepo,
    "--ilxyr-cli", fakeIlxyr,
    "--braid-repo", braidRepo,
    "--braid-cli", fakeBraid,
  ], { encoding: "utf8", env: { ...process.env, ILXYR_REJECT: "1" } });
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stderr, /ilXyr rejected family bridge-rejected-v1: fixture\.rejected\.v1: fixture_gate/u);
} finally {
  rmSync(temporaryRoot, { recursive: true, force: true });
}

process.stdout.write("ilXyr/Braid cloud bridge contract tests passed\n");
