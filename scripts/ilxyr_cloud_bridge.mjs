#!/usr/bin/env node

import { createHash } from "node:crypto";
import {
  existsSync, lstatSync, readFileSync, readdirSync, mkdirSync, writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const COMMIT = /^[0-9a-f]{40}$/u;
const DIGEST = /^[0-9a-f]{64}$/u;
const IDENTIFIER = /^[a-z0-9][a-z0-9._-]*$/u;
const RELEASE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/u;
const SAFE_PATH = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/u;
const GITHUB_REPOSITORY = /^https:\/\/github\.com\/([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+?)(?:\.git)?$/u;
const AUTHORITIES = new Set(["exact_check", "deterministic_replay", "corpus_proxy", "human_review"]);
const PURPOSES = new Set(["infrastructure_smoke", "experiment"]);

export function sha256File(path) {
  const hash = createHash("sha256");
  hash.update(readFileSync(path));
  return hash.digest("hex");
}

function fail(message) {
  throw new Error(message);
}

function object(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
  return value;
}

function exactKeys(value, allowed, label) {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) fail(`${label}.${key} is not allowed`);
  }
}

function string(value, label) {
  if (typeof value !== "string" || value.length === 0) fail(`${label} must be a non-empty string`);
  return value;
}

function repository(value, label) {
  string(value, label);
  if (!GITHUB_REPOSITORY.test(value)) fail(`${label} must be an https GitHub repository URL`);
  return value;
}

function githubRepository(value, label) {
  repository(value, label);
  const match = GITHUB_REPOSITORY.exec(value);
  return `${match[1]}/${match[2]}`;
}

function commit(value, label, allowCurrent = false) {
  if (allowCurrent && value === "CURRENT") return value;
  if (typeof value !== "string" || !COMMIT.test(value)) fail(`${label} must be a full lowercase Git commit`);
  return value;
}

function digest(value, label) {
  if (typeof value !== "string" || !DIGEST.test(value)) fail(`${label} must be a lowercase SHA-256 digest`);
  return value;
}

function relativePath(value, label) {
  string(value, label);
  if (isAbsolute(value) || !SAFE_PATH.test(value)) fail(`${label} must be a safe relative path`);
  const normalized = value.split(/[\\/]+/u);
  if (normalized.includes("..") || normalized.includes(".") || normalized.includes("")) {
    fail(`${label} must be a normalized repository-relative path`);
  }
  return value;
}

function uniqueStrings(value, label) {
  if (!Array.isArray(value)) fail(`${label} must be an array`);
  for (const [index, item] of value.entries()) string(item, `${label}[${index}]`);
  if (new Set(value).size !== value.length) fail(`${label} must not contain duplicates`);
  return value;
}

function validateArtifacts(value) {
  if (!Array.isArray(value) || value.length === 0) fail("source.artifacts must contain at least one artifact");
  const seen = new Set();
  for (const [index, entryValue] of value.entries()) {
    const entry = object(entryValue, `source.artifacts[${index}]`);
    exactKeys(entry, ["path", "sha256"], `source.artifacts[${index}]`);
    relativePath(entry.path, `source.artifacts[${index}].path`);
    digest(entry.sha256, `source.artifacts[${index}].sha256`);
    if (seen.has(entry.path)) fail(`source.artifacts repeats ${entry.path}`);
    seen.add(entry.path);
  }
}

export function validateManifest(raw) {
  const manifest = object(raw, "job");
  exactKeys(manifest, [
    "schema", "id", "purpose", "claim_ids", "grounding_authority",
    "source", "control_plane", "corpus", "outputs",
  ], "job");
  if (manifest.schema !== "crlp.ilxyr_cloud_job.v1") fail("job.schema must be crlp.ilxyr_cloud_job.v1");
  if (typeof manifest.id !== "string" || !IDENTIFIER.test(manifest.id)) fail("job.id is not a simple identifier");
  if (!PURPOSES.has(manifest.purpose)) fail("job.purpose is invalid");
  uniqueStrings(manifest.claim_ids, "job.claim_ids");
  if (!AUTHORITIES.has(manifest.grounding_authority)) fail("job.grounding_authority is invalid");

  const source = object(manifest.source, "source");
  exactKeys(source, ["repository", "commit", "artifacts"], "source");
  repository(source.repository, "source.repository");
  commit(source.commit, "source.commit", true);
  validateArtifacts(source.artifacts);

  const control = object(manifest.control_plane, "control_plane");
  exactKeys(control, ["repository", "commit", "family_manifest"], "control_plane");
  repository(control.repository, "control_plane.repository");
  commit(control.commit, "control_plane.commit");
  relativePath(control.family_manifest, "control_plane.family_manifest");

  const corpus = object(manifest.corpus, "corpus");
  if (corpus.kind === "none") {
    exactKeys(corpus, ["kind"], "corpus");
    if (manifest.grounding_authority === "corpus_proxy") {
      fail("corpus_proxy cloud jobs must bind a verified Braid release");
    }
  } else if (corpus.kind === "braid_release") {
    exactKeys(corpus, [
      "kind", "repository", "commit", "release_id", "release_manifest_sha256", "release_directory",
      "materialization",
    ], "corpus");
    repository(corpus.repository, "corpus.repository");
    commit(corpus.commit, "corpus.commit");
    if (typeof corpus.release_id !== "string" || !RELEASE_ID.test(corpus.release_id)) {
      fail("corpus.release_id is invalid");
    }
    digest(corpus.release_manifest_sha256, "corpus.release_manifest_sha256");
    relativePath(corpus.release_directory, "corpus.release_directory");
    const materialization = object(corpus.materialization, "corpus.materialization");
    if (materialization.kind === "existing") {
      exactKeys(materialization, ["kind"], "corpus.materialization");
    } else if (materialization.kind === "build") {
      exactKeys(materialization, ["kind", "spec_path", "output_root"], "corpus.materialization");
      relativePath(materialization.spec_path, "corpus.materialization.spec_path");
      relativePath(materialization.output_root, "corpus.materialization.output_root");
    } else {
      fail("corpus.materialization.kind must be existing or build");
    }
  } else {
    fail("corpus.kind must be none or braid_release");
  }

  const outputs = object(manifest.outputs, "outputs");
  exactKeys(outputs, ["directory", "retention_days"], "outputs");
  relativePath(outputs.directory, "outputs.directory");
  if (!Number.isInteger(outputs.retention_days) || outputs.retention_days < 1 || outputs.retention_days > 90) {
    fail("outputs.retention_days must be an integer from 1 to 90");
  }
  return manifest;
}

function contained(root, path, label) {
  const rootPath = resolve(root);
  const candidate = resolve(path);
  const within = relative(rootPath, candidate);
  if (within === "" || (!within.startsWith(`..${sep}`) && within !== ".." && !isAbsolute(within))) return candidate;
  fail(`${label} leaves ${rootPath}`);
}

function run(program, args, options = {}) {
  const rendered = [program, ...args].join(" ");
  process.stderr.write(`+ ${rendered}\n`);
  const result = spawnSync(program, args, {
    cwd: options.cwd,
    env: options.env ?? process.env,
    encoding: "utf8",
    stdio: options.capture ? ["ignore", "pipe", "pipe"] : "inherit",
  });
  if (result.error) fail(`${rendered}: ${result.error.message}`);
  if (result.status !== 0) {
    const detail = options.capture ? `\n${result.stderr || result.stdout || ""}` : "";
    fail(`${rendered} exited ${result.status}${detail}`);
  }
  return result.stdout ?? "";
}

function gitHead(repoPath) {
  return run("git", ["-C", repoPath, "rev-parse", "HEAD"], { capture: true }).trim();
}

function verifyCheckout(repoPath, expectedCommit, label, allowCurrent = false) {
  if (!existsSync(repoPath)) fail(`${label} checkout is missing: ${repoPath}`);
  const head = gitHead(repoPath);
  if (!(allowCurrent && expectedCommit === "CURRENT") && head !== expectedCommit) {
    fail(`${label} checkout is ${head}, expected ${expectedCommit}`);
  }
  return head;
}

function verifyTrackedFile(repoPath, path, label) {
  const absolute = contained(repoPath, path, label);
  const within = relative(resolve(repoPath), absolute).split(sep).join("/");
  run("git", ["-C", repoPath, "ls-files", "--error-unmatch", "--", within], { capture: true });
  run("git", ["-C", repoPath, "diff", "--quiet", "HEAD", "--", within], { capture: true });
  return absolute;
}

function verifySourceArtifacts(repoPath, artifacts) {
  return artifacts.map((artifact) => {
    const path = verifyTrackedFile(repoPath, resolve(repoPath, artifact.path), `source artifact ${artifact.path}`);
    if (!existsSync(path)) fail(`source artifact is missing: ${artifact.path}`);
    const actual = sha256File(path);
    if (actual !== artifact.sha256) fail(`source artifact ${artifact.path} has SHA-256 ${actual}, expected ${artifact.sha256}`);
    return { ...artifact };
  });
}

function walkFiles(root) {
  const files = [];
  for (const name of readdirSync(root).sort()) {
    const path = join(root, name);
    const entry = lstatSync(path);
    if (entry.isSymbolicLink()) fail(`family template must not contain symbolic links: ${path}`);
    if (entry.isDirectory()) files.push(...walkFiles(path));
    else if (entry.isFile()) files.push(path);
    else fail(`family template contains an unsupported entry: ${path}`);
  }
  return files;
}

function replaceAll(value, replacements) {
  let rendered = value;
  for (const [token, replacement] of Object.entries(replacements)) {
    rendered = rendered.replaceAll(token, replacement);
  }
  return rendered;
}

function renderFamily(manifest, repoPath, sourceCommit, outputDirectory) {
  const templateManifest = verifyTrackedFile(
    repoPath,
    resolve(repoPath, manifest.control_plane.family_manifest),
    "ilXyr family template",
  );
  const templateRoot = dirname(templateManifest);
  const renderedRoot = resolve(outputDirectory, "rendered-family");
  if (existsSync(renderedRoot)) fail(`rendered family directory already exists: ${renderedRoot}`);
  const replacements = {
    __JOB_ID__: manifest.id,
    __CRLP_REPOSITORY__: manifest.source.repository,
    __CRLP_COMMIT__: sourceCommit,
    __CRLP_ROOT__: "/workspace/crlplrimes",
    __ILXYR_ROOT__: "/workspace/ilxyr",
    __BRAID_ROOT__: "/workspace/braid",
    __NODE_EXECUTABLE__: process.execPath,
    __OUTPUT_ROOT__: `/workspace/crlplrimes/${manifest.outputs.directory}`,
    __BRAID_RELEASE_ID__: manifest.corpus.kind === "braid_release" ? manifest.corpus.release_id : "",
    __BRAID_RELEASE_MANIFEST_SHA256__: manifest.corpus.kind === "braid_release"
      ? manifest.corpus.release_manifest_sha256
      : "",
    __BRAID_RELEASE_DIRECTORY__: manifest.corpus.kind === "braid_release"
      ? `/workspace/crlplrimes/${manifest.corpus.release_directory}`
      : "",
  };
  for (const [index, artifact] of manifest.source.artifacts.entries()) {
    replacements[`__SOURCE_ARTIFACT_${index}_PATH__`] = artifact.path;
    replacements[`__SOURCE_ARTIFACT_${index}_SHA256__`] = artifact.sha256;
  }
  const artifacts = [];
  for (const sourcePath of walkFiles(templateRoot)) {
    verifyTrackedFile(repoPath, sourcePath, `family template file ${sourcePath}`);
    const within = relative(templateRoot, sourcePath);
    const destination = resolve(renderedRoot, within);
    const text = readFileSync(sourcePath, "utf8");
    const rendered = replaceAll(text, replacements);
    const unresolved = rendered.match(/__[A-Z0-9_]+__/gu);
    if (unresolved) fail(`${sourcePath} has unresolved template tokens: ${[...new Set(unresolved)].join(", ")}`);
    if (sourcePath.endsWith(".json")) {
      try {
        JSON.parse(rendered);
      } catch (error) {
        fail(`rendered family JSON is invalid at ${within}: ${error.message}`);
      }
    }
    mkdirSync(dirname(destination), { recursive: true });
    writeFileSync(destination, rendered, { flag: "wx" });
    artifacts.push({ path: relative(outputDirectory, destination).split(sep).join("/"), sha256: sha256File(destination) });
  }
  const renderedManifest = resolve(renderedRoot, relative(templateRoot, templateManifest));
  return { manifest: renderedManifest, artifacts };
}

function verifyBraid(manifest, options) {
  if (manifest.corpus.kind === "none") return null;
  if (!options.braidRepo || !options.braidCli) fail("Braid jobs require --braid-repo and --braid-cli");
  const braidCommit = verifyCheckout(options.braidRepo, manifest.corpus.commit, "Braid");
  const releaseDirectory = options.braidRelease
    ? resolve(options.braidRelease)
    : contained(options.repo, resolve(options.repo, manifest.corpus.release_directory), "Braid release directory");
  if (!options.braidRelease && manifest.corpus.materialization.kind === "build") {
    const spec = verifyTrackedFile(
      options.repo,
      resolve(options.repo, manifest.corpus.materialization.spec_path),
      "Braid build specification",
    );
    const outputRoot = contained(
      options.repo,
      resolve(options.repo, manifest.corpus.materialization.output_root),
      "Braid build output root",
    );
    run(process.execPath, [options.braidCli, "build", spec, "--output", outputRoot]);
  }
  const releaseManifest = resolve(releaseDirectory, "release.json");
  if (!existsSync(releaseManifest)) fail(`Braid release manifest is missing: ${releaseManifest}`);
  const actualManifestHash = sha256File(releaseManifest);
  if (actualManifestHash !== manifest.corpus.release_manifest_sha256) {
    fail(`Braid release.json has SHA-256 ${actualManifestHash}, expected ${manifest.corpus.release_manifest_sha256}`);
  }
  const release = JSON.parse(readFileSync(releaseManifest, "utf8"));
  if (release.releaseId !== manifest.corpus.release_id) {
    fail(`Braid release ID is ${release.releaseId}, expected ${manifest.corpus.release_id}`);
  }
  run(process.execPath, [options.braidCli, "verify", releaseDirectory, "--expected", manifest.corpus.release_id]);
  return {
    repository: manifest.corpus.repository,
    commit: braidCommit,
    release_id: manifest.corpus.release_id,
    release_manifest_sha256: actualManifestHash,
    materialization: manifest.corpus.materialization.kind,
  };
}

function readManifest(path) {
  return validateManifest(JSON.parse(readFileSync(path, "utf8")));
}

function parseArguments(argv) {
  const options = { execute: false, validateOnly: false };
  const valueOptions = {
    "--manifest": "manifest",
    "--repo": "repo",
    "--ilxyr-repo": "ilxyrRepo",
    "--ilxyr-cli": "ilxyrCli",
    "--braid-repo": "braidRepo",
    "--braid-cli": "braidCli",
    "--braid-release": "braidRelease",
    "--ledger": "ledger",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--execute") options.execute = true;
    else if (arg === "--validate-only") options.validateOnly = true;
    else if (arg === "--github-outputs") options.githubOutputs = true;
    else if (valueOptions[arg]) {
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) fail(`${arg} requires a value`);
      options[valueOptions[arg]] = value;
      index += 1;
    } else fail(`unknown argument ${arg}`);
  }
  if (!options.manifest) fail("--manifest is required");
  return options;
}

function githubOutputs(manifest) {
  const values = {
    job_id: manifest.id,
    source_commit: manifest.source.commit,
    ilxyr_repository: githubRepository(manifest.control_plane.repository, "control_plane.repository"),
    ilxyr_commit: manifest.control_plane.commit,
    braid_required: String(manifest.corpus.kind === "braid_release"),
    braid_repository: manifest.corpus.kind === "braid_release"
      ? githubRepository(manifest.corpus.repository, "corpus.repository")
      : "",
    braid_commit: manifest.corpus.kind === "braid_release" ? manifest.corpus.commit : "",
    output_directory: manifest.outputs.directory,
    retention_days: String(manifest.outputs.retention_days),
  };
  return Object.entries(values).map(([key, value]) => `${key}=${value}`).join("\n");
}

function executeBridge(manifest, options) {
  if (!options.repo || !options.ilxyrRepo || !options.ilxyrCli) {
    fail("bridge preparation requires --repo, --ilxyr-repo, and --ilxyr-cli");
  }
  const sourceCommit = verifyCheckout(options.repo, manifest.source.commit, "CRLPLRIMES", true);
  const ilxyrCommit = verifyCheckout(options.ilxyrRepo, manifest.control_plane.commit, "ilXyr");
  verifyTrackedFile(options.repo, resolve(options.manifestPath), "cloud job manifest");
  const artifacts = verifySourceArtifacts(options.repo, manifest.source.artifacts);
  const braid = verifyBraid(manifest, options);
  const outputDirectory = contained(options.repo, resolve(options.repo, manifest.outputs.directory), "output directory");
  const ledger = options.ledger ? resolve(options.ledger) : resolve(outputDirectory, "ledger");
  mkdirSync(outputDirectory, { recursive: true });
  const family = renderFamily(manifest, options.repo, sourceCommit, outputDirectory);
  if (!existsSync(ledger)) run(options.ilxyrCli, ["init", ledger]);
  run(options.ilxyrCli, ["family", "freeze", ledger, family.manifest]);
  run(options.ilxyrCli, ["family", "check", ledger, family.manifest]);
  const lifecycle = ["freeze", "check"];
  if (options.execute) {
    run(options.ilxyrCli, ["family", "run", ledger, family.manifest, "--execute"]);
    run(options.ilxyrCli, ["family", "settle", ledger, family.manifest]);
    run(options.ilxyrCli, ["verify", ledger]);
    lifecycle.push("run", "settle", "verify");
  }
  const receipt = {
    schema: "crlp.ilxyr_cloud_bridge_receipt.v1",
    job_id: manifest.id,
    purpose: manifest.purpose,
    claim_ids: manifest.claim_ids,
    grounding_authority: manifest.grounding_authority,
    executed: options.execute,
    lifecycle,
    source: { repository: manifest.source.repository, commit: sourceCommit, artifacts },
    control_plane: {
      repository: manifest.control_plane.repository,
      commit: ilxyrCommit,
      family_template: manifest.control_plane.family_manifest,
      rendered_family_manifest: family.manifest,
      rendered_artifacts: family.artifacts,
      node_executable: process.execPath,
    },
    corpus: braid ?? { kind: "none" },
    ledger,
  };
  const receiptPath = resolve(outputDirectory, "bridge-receipt.json");
  writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "w" });
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
}

export function main(argv = process.argv.slice(2)) {
  const options = parseArguments(argv);
  const manifestPath = resolve(options.manifest);
  options.manifestPath = manifestPath;
  const manifest = readManifest(manifestPath);
  if (options.githubOutputs) {
    process.stdout.write(`${githubOutputs(manifest)}\n`);
    return;
  }
  if (options.validateOnly) {
    process.stdout.write(`${JSON.stringify({ valid: true, schema: manifest.schema, id: manifest.id })}\n`);
    return;
  }
  executeBridge(manifest, options);
}

const invoked = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (invoked) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
