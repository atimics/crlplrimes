#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const [siteDirArg, realArg, runIdArg, shaArg, generatedUtcArg] = process.argv.slice(2);

if (!siteDirArg) {
  console.error("usage: node scripts/render_pages_report.mjs <site-dir> [real] [run-id] [sha] [generated-utc]");
  process.exit(2);
}

const siteDir = path.resolve(siteDirArg);
const resultsDir = path.join(siteDir, "results");
const real = realArg || "unknown";
const runId = runIdArg || "local";
const commit = shaArg || "unknown";
const generatedUtc = generatedUtcArg || new Date().toISOString().replace(/\.\d{3}Z$/, "Z");

function readText(file, fallback = "") {
  return existsSync(file) ? readFileSync(file, "utf8") : fallback;
}

function parseJsonl(file) {
  return readText(file)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line, index) => {
      try {
        return [JSON.parse(line)];
      } catch (error) {
        console.warn(`Skipping invalid JSONL row in ${file}:${index + 1}: ${error.message}`);
        return [];
      }
    });
}

function parseJson(file, fallback = null) {
  const text = readText(file, "");
  if (!text.trim()) return fallback;
  try {
    return JSON.parse(text);
  } catch (error) {
    console.warn(`Skipping invalid JSON file ${file}: ${error.message}`);
    return fallback;
  }
}

function parseManifest(file) {
  const lines = readText(file)
    .split(/\r?\n/)
    .filter(Boolean);
  if (!lines.length) return [];
  const headers = lines[0].split("\t");
  return lines.slice(1).map((line) => {
    const cols = line.split("\t");
    return Object.fromEntries(headers.map((header, index) => [header, cols[index] || ""]));
  });
}

function sha256File(file) {
  if (!existsSync(file)) return "";
  return createHash("sha256").update(readFileSync(file)).digest("hex");
}

function walkFiles(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) return walkFiles(file);
    return entry.isFile() ? [file] : [];
  });
}

function buildManifestRows() {
  const rows = [];
  for (const category of ["results", "models"]) {
    const categoryDir = path.join(siteDir, category);
    for (const file of walkFiles(categoryDir).sort()) {
      const relativePath = path.relative(siteDir, file).split(path.sep).join("/");
      rows.push({
        category,
        path: relativePath,
        bytes: String(statSync(file).size),
        sha256: sha256File(file),
        source_commit: commit,
        github_run: runId,
        generated_utc: generatedUtc,
      });
    }
  }
  return rows;
}

function writeManifest(rows) {
  const headers = ["category", "path", "bytes", "sha256", "source_commit", "github_run", "generated_utc"];
  const lines = [
    headers.join("\t"),
    ...rows.map((row) => headers.map((header) => row[header] || "").join("\t")),
  ];
  writeFileSync(manifestPath, `${lines.join("\n")}\n`);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function attr(value) {
  return escapeHtml(value);
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value, digits = 2) {
  const number = finite(value);
  if (number === null) return "n/a";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}

function formatPct(value, digits = 1) {
  const number = finite(value);
  if (number === null) return "n/a";
  return `${(number * 100).toFixed(digits)}%`;
}

function formatPctPoints(value, digits = 1) {
  const number = finite(value);
  if (number === null) return "n/a";
  return `${number.toFixed(digits)}%`;
}

function shortHash(value, length = 12) {
  if (!value || value === "none" || value === "unknown") return value || "n/a";
  return String(value).slice(0, length);
}

function modeLabel(row) {
  return String(row.evaluation_mode || row.selection_strategy || "unspecified")
    .replace(/^certified_threshold_margin_gate_/, "certified ")
    .replace(/_/g, " ");
}

function weightedAccuracy(row) {
  const top1 = finite(row.top1);
  if (top1 !== null) return top1;
  const bins = Array.isArray(row.reliability_bins) ? row.reliability_bins : [];
  let count = 0;
  let correct = 0;
  for (const bin of bins) {
    const binCount = finite(bin.count);
    const accuracy = finite(bin.accuracy);
    if (binCount !== null && accuracy !== null) {
      count += binCount;
      correct += binCount * accuracy;
    }
  }
  return count > 0 ? correct / count : null;
}

function workSavedPct(row) {
  const explicit = finite(row.expanded_saved_pct);
  if (explicit !== null) return explicit;
  const direct = finite(row.direct_exact);
  const proposal = finite(row.proposal_steps) ?? finite(row.expanded_exact);
  if (direct === null || direct <= 0 || proposal === null) return null;
  return Math.max(0, Math.min(100, (1 - proposal / direct) * 100));
}

function rowScore(row) {
  return (weightedAccuracy(row) ?? -Infinity)
    + ((workSavedPct(row) ?? -Infinity) / 1000)
    - ((finite(row.ece) ?? 0) / 1000);
}

function isRawRow(row) {
  return String(row.evaluation_mode || "") === "raw";
}

function isCertifiedOrGatedRow(row) {
  const mode = String(row.evaluation_mode || "");
  return mode.includes("certified")
    || mode.includes("expanded")
    || (finite(row.accepted_rules) ?? 0) > 0
    || (row.rule_set_hash && row.rule_set_hash !== "unknown" && row.rule_set_hash !== "hand-wheel");
}

function hrefFor(filePath) {
  return String(filePath || "").split("/").map(encodeURIComponent).join("/");
}

function renderTable(headers, rows, className = "") {
  if (!rows.length) return '<div class="empty">No rows available.</div>';
  const cls = className ? ` class="${attr(className)}"` : "";
  return `<table${cls}><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function renderBarList(rows, max, options = {}) {
  if (!rows.length) return '<div class="empty">No rows available.</div>';
  return rows.map((item) => {
    const width = max > 0 ? Math.max(1, Math.min(100, (item.value / max) * 100)) : 0;
    const fill = item.kind === "orange" ? "fill orange" : item.kind === "warn" ? "fill warn" : "fill";
    const dataAttrs = options.dataValue ? ` data-value="${attr(item.value)}"` : "";
    return `<div class="bar-row"${dataAttrs}>
      <div class="bar-label"><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.sub || "")}</small></div>
      <div class="track"><div class="${fill}" style="width:${width.toFixed(2)}%"></div></div>
      <div class="bar-value">${escapeHtml(item.display)}</div>
    </div>`;
  }).join("");
}

function bestRowsByTask(rows, valueFn) {
  const byTask = new Map();
  for (const row of rows) {
    if (!row.task) continue;
    const value = valueFn(row);
    if (value === null || value === undefined) continue;
    const current = byTask.get(row.task);
    if (!current || value > current.value || (value === current.value && rowScore(row) > rowScore(current.row))) {
      byTask.set(row.task, { row, value });
    }
  }
  return [...byTask.values()].sort((a, b) => b.value - a.value || String(a.row.task).localeCompare(String(b.row.task)));
}

const comparisonPath = path.join(resultsDir, "comparison.jsonl");
const canonicalPath = path.join(resultsDir, "comparison.canonical.jsonl");
const manifestPath = path.join(siteDir, "manifest.tsv");
const rows = parseJsonl(comparisonPath);
const compareRows = rows.filter((row) => row.schema === "crlp.compare.v1");
const ruleRows = rows.filter((row) => row.schema === "crlp.rule_search.summary.v1");
const schemaRows = rows.filter((row) => row.schema === "crlp.experiment_schema.v1");
const env = parseJson(path.join(resultsDir, "environment.json"), {});
const paperClaims = parseJson(path.join(resultsDir, "paper_claims.json"), { claims: [] })?.claims || [];

const comparisonSha = sha256File(comparisonPath);
const canonicalSha = sha256File(canonicalPath);
const canonicalIdentical = comparisonSha && canonicalSha && comparisonSha === canonicalSha;
let manifestRows = [];
let modelFiles = [];
let resultFiles = [];
const tasks = new Set([...schemaRows.map((row) => row.task), ...compareRows.map((row) => row.task)].filter(Boolean));
const modelHashes = new Set(compareRows.map((row) => row.model_hash).filter((hash) => hash && hash !== "none" && hash !== "unknown"));
const reliabilityRows = compareRows.filter((row) => Array.isArray(row.reliability_bins) && row.reliability_bins.length);
const zeroFalseRejectRows = compareRows.filter((row) => finite(row.false_rejections) === 0);
const nonZeroFalseRejectRows = compareRows.filter((row) => (finite(row.false_rejections) ?? 0) > 0);
const c1Claim = paperClaims.find((claim) => claim.id === "C1.prime_gate_delta");
const c2Claim = paperClaims.find((claim) => claim.id === "C2.prime_proposer_fixed_budget");
const staleClaims = paperClaims.filter((claim) => claim.status === "stale");
const underpoweredClaims = paperClaims.filter((claim) => claim.status === "underpowered");
const supportedClaims = paperClaims.filter((claim) => claim.status === "supported");
const bestSavedRow = [...ruleRows, ...compareRows]
  .filter((row) => workSavedPct(row) !== null)
  .sort((a, b) => (workSavedPct(b) ?? -1) - (workSavedPct(a) ?? -1))[0];
const bestAccuracyRow = compareRows
  .filter((row) => weightedAccuracy(row) !== null)
  .sort((a, b) => (weightedAccuracy(b) ?? -1) - (weightedAccuracy(a) ?? -1))[0];
const bestCalibrationRow = compareRows
  .filter((row) => finite(row.ece) !== null && Array.isArray(row.reliability_bins) && row.reliability_bins.length)
  .sort((a, b) => (finite(a.ece) ?? Infinity) - (finite(b.ece) ?? Infinity))[0];

function primeGateSentence() {
  const metrics = c1Claim?.metrics;
  if (metrics) {
    return `Prime-gap learned filtering raises top-1 from ${formatPct(metrics.raw_top1)} raw and ${formatPct(metrics.hand_top1)} hand-wheel to ${formatPct(metrics.learned_top1)}, a +${formatNumber(metrics.learned_vs_hand_delta_pp, 2)} percentage-point gain over hand-wheel, with ${formatNumber(metrics.false_rejections, 0)} verifier-counted true-answer deletions.`;
  }
  return bestAccuracyRow
    ? `Best observed accuracy is ${formatPct(weightedAccuracy(bestAccuracyRow))} on ${bestAccuracyRow.task} / ${modeLabel(bestAccuracyRow)}.`
    : "No comparison rows with accuracy were found.";
}

function c2Sentence() {
  if (c2Claim?.summary) return c2Claim.summary;
  return "The fixed-budget proposer comparison is not available in the current claim compiler output.";
}

function canonicalSentence() {
  if (!comparisonSha || !canonicalSha) return "Canonical comparison output was not available for hashing.";
  if (canonicalIdentical) {
    return "comparison.jsonl and comparison.canonical.jsonl are byte-identical in this run, so canonicalization is an idempotency check here rather than a second distinct dataset.";
  }
  return "comparison.jsonl and comparison.canonical.jsonl differ, so the canonical file is the timing-stripped comparison surface.";
}

const abstractText = `CRLPLRIMES studies verifier-grounded action ranking: enumerate candidate actions, certify rules that reject impossible candidates, and rank the surviving set with residual neural scorers. This live paper is pre-rendered from the evidence bundle, not from browser-only JavaScript. The current bundle covers ${tasks.size} declared task families, ${compareRows.length} comparison rows, ${ruleRows.length} rule-search summaries, and ${paperClaims.length} claim-compiler rows. ${primeGateSentence()} ${c2Claim ? `The C2 row is explicitly negative: ${c2Sentence()}` : ""}`;

function renderMetrics() {
  const bestSaved = bestSavedRow ? formatPctPoints(workSavedPct(bestSavedRow)) : "n/a";
  return [
    { value: tasks.size, label: "declared task families", sub: `${schemaRows.length} experiment-schema rows` },
    { value: compareRows.length, label: "comparison rows", sub: "model, gate, and calibration outcomes" },
    { value: bestSaved, label: "best verifier work saved", sub: bestSavedRow ? `${bestSavedRow.task || "task"} / ${modeLabel(bestSavedRow)}` : "no row" },
    { value: supportedClaims.length, label: "supported claims", sub: `${underpoweredClaims.length} underpowered, ${staleClaims.length} stale` },
    { value: modelHashes.size, label: "referenced model hashes", sub: `${modelFiles.length} model/rule files published` },
    { value: zeroFalseRejectRows.length, label: "zero false-reject rows", sub: `${nonZeroFalseRejectRows.length} rows report nonzero false rejections` },
  ].map((metric) => `<div class="metric"><strong>${escapeHtml(metric.value)}</strong><span>${escapeHtml(metric.label)}</span><small>${escapeHtml(metric.sub)}</small></div>`).join("");
}

function pairedBar(label, value, klass) {
  const width = Math.max(1, Math.min(100, value * 100));
  return `<div class="paired-bar">
    <span>${escapeHtml(label)}</span>
    <div class="track"><div class="${klass}" style="width:${width.toFixed(2)}%"></div></div>
    <div class="bar-value">${escapeHtml(formatPct(value))}</div>
  </div>`;
}

function renderGateFigure() {
  const rawByTask = new Map();
  const gatedByTask = new Map();
  for (const row of compareRows) {
    if (!row.task) continue;
    const acc = weightedAccuracy(row);
    if (acc === null) continue;
    if (isRawRow(row)) {
      const current = rawByTask.get(row.task);
      if (!current || acc > current.acc) rawByTask.set(row.task, { row, acc });
    } else if (isCertifiedOrGatedRow(row)) {
      const current = gatedByTask.get(row.task);
      if (!current || acc > current.acc || (acc === current.acc && rowScore(row) > rowScore(current.row))) {
        gatedByTask.set(row.task, { row, acc });
      }
    }
  }
  const pairs = [...rawByTask.keys()]
    .filter((task) => gatedByTask.has(task))
    .map((task) => ({ task, raw: rawByTask.get(task), gated: gatedByTask.get(task) }))
    .sort((a, b) => (b.gated.acc - b.raw.acc) - (a.gated.acc - a.raw.acc))
    .slice(0, 7);
  if (!pairs.length) {
    return {
      html: '<div class="empty">No raw/gated task pairs were found.</div>',
      caption: "Figure 1 has no paired evidence in the current comparison file.",
    };
  }
  const gains = pairs.map((pair) => pair.gated.acc - pair.raw.acc);
  const html = pairs.map((pair) => `<div class="comparison-row">
    <div class="bar-label"><strong>${escapeHtml(pair.task)}</strong><small>${escapeHtml(modeLabel(pair.gated.row))}</small></div>
    <div class="paired-bars">${pairedBar("Raw", pair.raw.acc, "fill")}${pairedBar("Gated", pair.gated.acc, "fill orange")}</div>
  </div>`).join("");
  return {
    html,
    caption: `Figure 1. Raw accuracy compared with the strongest certified or rule-gated row. Largest displayed gain: ${(Math.max(...gains) * 100).toFixed(1)} percentage points.`,
  };
}

function renderWorkFigure() {
  const leaders = ruleRows
    .filter((row) => workSavedPct(row) !== null)
    .sort((a, b) => (workSavedPct(b) ?? -1) - (workSavedPct(a) ?? -1))
    .slice(0, 7)
    .map((row) => ({
      label: row.task || "task",
      sub: `${String(row.selection_strategy || "strategy").replace(/_/g, " ")} / ${row.accepted_rules ?? "n/a"} rules`,
      value: workSavedPct(row),
      display: formatPctPoints(workSavedPct(row)),
      kind: "orange",
    }));
  return {
    html: renderBarList(leaders, Math.max(100, ...leaders.map((row) => row.value || 0))),
    caption: "Figure 2. Rule-search leaders ranked by exact verifier work saved, with accepted-rule counts shown in each row label.",
  };
}

function renderCalibrationFigure() {
  const leaders = compareRows
    .filter((row) => finite(row.ece) !== null && Array.isArray(row.reliability_bins) && row.reliability_bins.length)
    .sort((a, b) => (finite(a.ece) ?? Infinity) - (finite(b.ece) ?? Infinity))
    .slice(0, 7)
    .map((row) => ({
      label: row.task || "task",
      sub: modeLabel(row),
      value: finite(row.ece),
      display: formatPct(row.ece),
      kind: "warn",
    }));
  return {
    html: renderBarList(leaders, Math.max(0.12, ...leaders.map((row) => row.value || 0))),
    caption: "Figure 3. Lowest expected calibration error rows with reliability bins available. Smaller bars are better; the interactive panel expands bins.",
  };
}

function renderClaimLedger() {
  const bestSaved = bestSavedRow ? formatPctPoints(workSavedPct(bestSavedRow)) : "n/a";
  return renderTable(
    ["Claim", "Live evidence", "Primary artifact", "Status"],
    [
      [
        escapeHtml("Task scope and leakage policy are explicit."),
        escapeHtml(`${schemaRows.length} crlp.experiment_schema.v1 rows declare task roles, claim scope, features, and leakage policy.`),
        '<a href="results/comparison.jsonl.txt"><code>comparison.jsonl.txt</code></a>',
        '<span class="ledger-status live">pre-rendered</span>',
      ],
      [
        escapeHtml(c1Claim?.title || "Audited next-prime filter improves top-1."),
        escapeHtml(c1Claim?.summary || primeGateSentence()),
        '<a href="#model-results"><code>C1 / crlp.compare.v1</code></a>',
        `<span class="ledger-status check">${escapeHtml(c1Claim?.status || "computed")}</span>`,
      ],
      [
        escapeHtml(c2Claim?.title || "Fixed-budget proposer is a negative baseline."),
        escapeHtml(c2Sentence()),
        '<a href="#rule-search"><code>C2 / rule_search.summary</code></a>',
        `<span class="ledger-status warn">${escapeHtml(c2Claim?.status || "negative")}</span>`,
      ],
      [
        escapeHtml("Certified gates can shrink exact verification."),
        escapeHtml(`Best observed saved work is ${bestSaved} across rule-search and comparison rows.`),
        '<a href="#rule-search"><code>rule_search.summary</code></a>',
        '<span class="ledger-status check">quantified</span>',
      ],
      [
        escapeHtml("False-rejection behavior is audited, not hidden."),
        escapeHtml(`${zeroFalseRejectRows.length} comparison rows explicitly report false_rejections = 0; ${nonZeroFalseRejectRows.length} rows report nonzero false rejections and remain visible.`),
        '<a href="#model-results"><code>crlp.compare.v1</code></a>',
        '<span class="ledger-status live">inspectable</span>',
      ],
      [
        escapeHtml("Canonicalization is explicitly reported."),
        escapeHtml(canonicalSentence()),
        '<a href="results/comparison.canonical.jsonl.txt"><code>comparison.canonical.jsonl.txt</code></a>',
        `<span class="ledger-status ${canonicalIdentical ? "check" : "live"}">${canonicalIdentical ? "byte-identical" : "canonicalized"}</span>`,
      ],
      [
        escapeHtml("Models and certified rules are published artifacts."),
        escapeHtml(`${modelFiles.length} checkpoint or rule files are published with byte counts, SHA-256 hashes, commit, run, and generation timestamp.`),
        '<a href="manifest.tsv"><code>manifest.tsv</code></a>',
        '<span class="ledger-status check">reproducible</span>',
      ],
    ],
  );
}

function renderClaimCompilerTable() {
  if (!paperClaims.length) return '<div class="empty">No paper claim compiler output was bundled.</div>';
  const rows = paperClaims.slice(0, 40).map((claim) => [
    `<strong>${escapeHtml(claim.id || "")}</strong><br>${escapeHtml(claim.title || "")}`,
    escapeHtml(claim.summary || claim.claim_text || ""),
    `<span class="ledger-status ${claim.status === "supported" ? "check" : claim.status === "underpowered" || claim.status === "stale" ? "warn" : "live"}">${escapeHtml(claim.status || "unknown")}</span>`,
    `<code>${escapeHtml(claim.source || "")}</code>`,
  ]);
  return renderTable(["Claim", "Compiler summary", "Status", "Source"], rows, "claim-table");
}

function renderTaskCards() {
  if (!schemaRows.length) return '<div class="empty">No experiment-schema rows were found.</div>';
  return schemaRows.map((row) => {
    const audit = row.feature_audit || {};
    const warning = audit.has_warning ? '<span class="tag warn">feature warning</span>' : '<span class="tag good">policy compliant</span>';
    return `<article class="task-card">
      <h3>${escapeHtml(row.task || "unknown task")}</h3>
      <p>${escapeHtml(row.role || row.claim_scope || "No role declared.")}</p>
      <p>${escapeHtml(row.claim_scope || "")}</p>
      <div class="task-meta">
        <span class="tag">${escapeHtml(row.leakage_policy || "leakage policy n/a")}</span>
        <span class="tag">${escapeHtml(row.proposer_mode || "mode n/a")}</span>
        <span class="tag">${escapeHtml(`${audit.enabled_features ?? "n/a"}/${audit.total_features ?? "n/a"} features`)}</span>
        ${warning}
      </div>
    </article>`;
  }).join("");
}

function renderModelHashTable() {
  const groups = new Map();
  for (const row of compareRows) {
    const hash = row.model_hash;
    if (!hash || hash === "none" || hash === "unknown") continue;
    if (!groups.has(hash)) {
      groups.set(hash, { hash, tasks: new Set(), modes: new Set(), bestAcc: null, bestEce: null, bestWork: null });
    }
    const group = groups.get(hash);
    if (row.task) group.tasks.add(row.task);
    if (row.evaluation_mode) group.modes.add(row.evaluation_mode);
    const acc = weightedAccuracy(row);
    const ece = finite(row.ece);
    const work = workSavedPct(row);
    if (acc !== null) group.bestAcc = group.bestAcc === null ? acc : Math.max(group.bestAcc, acc);
    if (ece !== null) group.bestEce = group.bestEce === null ? ece : Math.min(group.bestEce, ece);
    if (work !== null) group.bestWork = group.bestWork === null ? work : Math.max(group.bestWork, work);
  }
  const rows = [...groups.values()]
    .sort((a, b) => (b.bestAcc ?? -1) - (a.bestAcc ?? -1) || a.hash.localeCompare(b.hash))
    .slice(0, 24)
    .map((group) => [
      `<span class="mono hash" title="${attr(group.hash)}">${escapeHtml(shortHash(group.hash))}</span>`,
      escapeHtml([...group.tasks].sort().join(", ")),
      escapeHtml([...group.modes].slice(0, 4).map((mode) => mode.replace(/_/g, " ")).join(", ") + (group.modes.size > 4 ? " ..." : "")),
      escapeHtml(formatPct(group.bestAcc)),
      escapeHtml(formatPct(group.bestEce)),
      escapeHtml(formatPctPoints(group.bestWork)),
    ]);
  return renderTable(["Model hash", "Tasks", "Modes", "Best accuracy", "Best ECE", "Best work saved"], rows);
}

function renderResultsRows() {
  return compareRows
    .slice()
    .sort((a, b) => String(a.task).localeCompare(String(b.task)) || rowScore(b) - rowScore(a) || String(a.evaluation_mode).localeCompare(String(b.evaluation_mode)))
    .slice(0, 180)
    .map((row, index) => `<tr class="selectable result-data-row" data-row-index="${index}">
      <td>${escapeHtml(row.task || "")}</td>
      <td>${escapeHtml(modeLabel(row))}</td>
      <td>${escapeHtml(formatPct(weightedAccuracy(row)))}</td>
      <td>${escapeHtml(formatPct(row.top5))}</td>
      <td>${escapeHtml(formatPct(row.ece))}</td>
      <td>${escapeHtml(formatPctPoints(workSavedPct(row)))}</td>
      <td>${escapeHtml(row.accepted_rules ?? "n/a")}</td>
      <td>${escapeHtml(row.false_rejections ?? "n/a")}</td>
      <td><span class="mono hash" title="${attr(row.model_hash || "")}">${escapeHtml(shortHash(row.model_hash))}</span></td>
    </tr>`)
    .join("");
}

function renderRuleTable() {
  const rows = ruleRows
    .filter((row) => workSavedPct(row) !== null)
    .sort((a, b) => (workSavedPct(b) ?? -1) - (workSavedPct(a) ?? -1))
    .slice(0, 18)
    .map((row) => [
      escapeHtml(row.task || ""),
      escapeHtml(String(row.selection_strategy || "").replace(/_/g, " ")),
      escapeHtml(formatPctPoints(workSavedPct(row))),
      escapeHtml(row.accepted_rules ?? "n/a"),
      escapeHtml(`${formatNumber(row.expanded_exact, 3)} / ${formatNumber(row.direct_exact, 3)}`),
      `<span class="mono hash" title="${attr(row.rule_set_hash || "")}">${escapeHtml(shortHash(row.rule_set_hash))}</span>`,
    ]);
  return renderTable(["Task", "Strategy", "Saved", "Accepted", "Exact work", "Rule set"], rows);
}

function renderReliability() {
  const row = reliabilityRows.slice().sort((a, b) => rowScore(b) - rowScore(a))[0];
  if (!row) return { title: "Reliability bins", html: '<div class="empty">No reliability bins were found.</div>' };
  const html = row.reliability_bins.map((bin) => {
    const confidence = Math.max(0, Math.min(1, finite(bin.confidence) ?? 0));
    const accuracy = Math.max(0, Math.min(1, finite(bin.accuracy) ?? 0));
    const label = `${formatPct(bin.low, 0)}-${formatPct(bin.high, 0)}`;
    return `<div class="bin">
      <div class="bin-bars">
        <div class="bin-bar conf" title="confidence ${attr(formatPct(confidence))}" style="height:${(confidence * 100).toFixed(2)}%"></div>
        <div class="bin-bar acc" title="accuracy ${attr(formatPct(accuracy))}" style="height:${(accuracy * 100).toFixed(2)}%"></div>
      </div>
      <span title="${attr(`${formatNumber(bin.count, 0)} samples`)}">${escapeHtml(label)}</span>
    </div>`;
  }).join("");
  return { title: `${row.task || "task"} - ${modeLabel(row)}`, html };
}

function renderArtifactTable(files) {
  return renderTable(
    ["Path", "Bytes", "SHA-256", "Commit"],
    files.map((row) => [
      `<a href="${attr(hrefFor(row.path))}"><code>${escapeHtml(row.path)}</code></a>`,
      escapeHtml(formatNumber(row.bytes, 0)),
      `<span class="mono hash" title="${attr(row.sha256 || "")}">${escapeHtml(shortHash(row.sha256))}</span>`,
      `<span class="mono hash">${escapeHtml(shortHash(row.source_commit || commit))}</span>`,
    ]),
  );
}

function markdownSummary() {
  const statusCounts = paperClaims.reduce((counts, claim) => {
    counts[claim.status || "unknown"] = (counts[claim.status || "unknown"] || 0) + 1;
    return counts;
  }, {});
  const statusText = Object.entries(statusCounts).map(([status, count]) => `${count} ${status}`).join(", ") || "no bundled claim compiler rows";
  return `# CRLPLRIMES Live Evidence Summary

Generated: \`${generatedUtc}\`

Commit: \`${commit}\`

GitHub run: \`${runId}\`

Scalar type: \`${real}\`

## Headline

- ${primeGateSentence()}
- C2 is a negative baseline, not a success claim: ${c2Sentence()}
- Best exact verifier work saved in the published bundle: ${bestSavedRow ? `${formatPctPoints(workSavedPct(bestSavedRow))} on \`${bestSavedRow.task || "task"}\` / \`${modeLabel(bestSavedRow)}\`` : "n/a"}.
- False-rejection audit surface: ${zeroFalseRejectRows.length} comparison rows report \`false_rejections = 0\`; ${nonZeroFalseRejectRows.length} rows report nonzero false rejections and remain visible in the report table.
- Claim compiler statuses: ${statusText}.
- ${canonicalSentence()}

## Reader Contract

The HTML page is pre-rendered at build time. Browser JavaScript only verifies and hydrates the static claims; it is not required to read the abstract, figures, claim ledger, task declarations, result tables, calibration bins, or artifact manifest.

## Inspectable Artifacts

- Human-readable JSONL mirror: \`results/comparison.jsonl.txt\`
- Canonical JSONL mirror: \`results/comparison.canonical.jsonl.txt\`
- Manifest with provenance columns: \`manifest.tsv\`
- Environment manifest: \`results/environment.json\`
- Paper claim compiler bundle: \`results/paper_claims.json\`

## Environment

- Runner OS: \`${env.runner_os || "unknown"}\`
- Runner arch: \`${env.runner_arch || "unknown"}\`
- Compiler: \`${env.compiler_version || env.cc || "unknown"}\`
- libc: \`${env.libc || "unknown"}\`
- CPU: \`${env.cpu_model || "unknown"}\`
`;
}

writeFileSync(path.join(resultsDir, "summary.md"), markdownSummary());

manifestRows = buildManifestRows();
writeManifest(manifestRows);
manifestRows = parseManifest(manifestPath);
modelFiles = manifestRows.filter((row) => row.category === "models");
resultFiles = manifestRows.filter((row) => row.category === "results");

const gateFigure = renderGateFigure();
const workFigure = renderWorkFigure();
const calibrationFigure = renderCalibrationFigure();
const bestAccuracyBars = bestRowsByTask(compareRows, weightedAccuracy).map((item) => ({
  label: item.row.task,
  sub: modeLabel(item.row),
  value: item.value,
  display: formatPct(item.value),
}));
const bestWorkBars = bestRowsByTask(compareRows, workSavedPct).map((item) => ({
  label: item.row.task,
  sub: modeLabel(item.row),
  value: item.value,
  display: formatPctPoints(item.value),
  kind: "orange",
}));
const ruleSavedBars = bestRowsByTask(ruleRows, workSavedPct).map((item) => ({
  label: item.row.task,
  sub: String(item.row.selection_strategy || "strategy n/a").replace(/_/g, " "),
  value: item.value,
  display: formatPctPoints(item.value),
  kind: "orange",
}));
const reliability = renderReliability();

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRLPLRIMES Live Paper</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f4ee;
      --band: #ebe7dc;
      --panel: #fffefa;
      --panel-soft: #f8f5ed;
      --text: #202223;
      --muted: #636a70;
      --line: #d7d0c1;
      --accent: #146c78;
      --accent-2: #b75f34;
      --accent-3: #637b3d;
      --warn: #9a4d5d;
      --shadow: 0 10px 30px rgba(42, 35, 22, 0.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #151514;
        --band: #22211d;
        --panel: #1b1c1a;
        --panel-soft: #22231f;
        --text: #eceae2;
        --muted: #b6b0a6;
        --line: #39382f;
        --accent: #6ec7cf;
        --accent-2: #e29a72;
        --accent-3: #b6ca77;
        --warn: #e192a5;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.24);
      }
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: var(--accent); }
    code, .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }
    main {
      width: min(1240px, calc(100% - 32px));
      margin: 0 auto 52px;
    }
    header {
      background: linear-gradient(90deg, rgba(20, 108, 120, 0.10), transparent 38%), linear-gradient(180deg, var(--band), var(--bg));
      border-bottom: 1px solid var(--line);
      margin-bottom: 28px;
    }
    .header-inner {
      width: min(1240px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 24px;
    }
    .paper-title {
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
      align-items: end;
    }
    .eyebrow {
      color: var(--accent-2);
      font-size: 0.82rem;
      font-weight: 750;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    h1, h2, h3 { line-height: 1.15; margin: 0; letter-spacing: 0; }
    h1 {
      font-size: clamp(2.1rem, 4vw, 4.2rem);
      max-width: 980px;
      margin-top: 8px;
    }
    h2 { font-size: 1.35rem; }
    h3 { font-size: 1rem; }
    p { color: var(--muted); max-width: 850px; }
    .lede {
      font-size: 1.08rem;
      margin: 16px 0 0;
    }
    .paper-kicker {
      color: var(--muted);
      font-size: 0.95rem;
      margin: 8px 0 0;
      max-width: 820px;
    }
    .paper-note, .metric, .panel, .task-card, .argument-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .paper-note { padding: 14px; }
    .paper-note strong { display: block; font-size: 0.92rem; margin-bottom: 6px; }
    .paper-note p { margin: 0; }
    .run-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 20px;
    }
    .pill {
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      display: inline-flex;
      gap: 6px;
      max-width: 100%;
      padding: 6px 10px;
      white-space: nowrap;
    }
    .pill strong { color: var(--text); font-weight: 700; }
    nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 22px;
    }
    nav a {
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      padding: 7px 10px;
      text-decoration: none;
    }
    section {
      border-top: 1px solid var(--line);
      padding: 28px 0;
    }
    section:first-child { border-top: 0; }
    .section-head {
      align-items: end;
      display: grid;
      gap: 10px 24px;
      grid-template-columns: minmax(0, 1fr) auto;
      margin-bottom: 16px;
    }
    .section-head p { margin: 8px 0 0; }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .grid.wide { grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); }
    .argument-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .argument-card, .task-card, .panel { padding: 16px; }
    .argument-card strong {
      color: var(--accent);
      display: block;
      font-size: 0.82rem;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .argument-card p, .task-card p { margin: 8px 0 0; }
    .metric {
      padding: 14px 15px;
      min-height: 108px;
    }
    .metric strong {
      display: block;
      font-size: clamp(1.45rem, 4vw, 2rem);
      letter-spacing: 0;
      line-height: 1;
    }
    .metric span, .metric small { color: var(--muted); display: block; }
    .metric small { margin-top: 8px; }
    .panel.figure {
      display: grid;
      gap: 12px;
      min-height: 330px;
    }
    .figure-label {
      color: var(--accent-2);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .caption {
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.9rem;
      margin: 0;
      padding-top: 10px;
    }
    .comparison-bars, .bar-list {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }
    .comparison-row {
      display: grid;
      gap: 8px;
      grid-template-columns: minmax(120px, 0.7fr) minmax(180px, 1.5fr);
    }
    .paired-bars { display: grid; gap: 5px; }
    .paired-bar, .bar-row {
      align-items: center;
      display: grid;
      gap: 8px;
    }
    .paired-bar { grid-template-columns: 84px minmax(120px, 1fr) 64px; }
    .bar-row { grid-template-columns: minmax(130px, 0.9fr) minmax(140px, 1.8fr) minmax(70px, auto); }
    .paired-bar span, .bar-label small { color: var(--muted); font-size: 0.82rem; }
    .bar-label { min-width: 0; overflow-wrap: anywhere; }
    .bar-label small { display: block; }
    .track {
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 999px;
      height: 14px;
      overflow: hidden;
      position: relative;
    }
    .fill {
      background: linear-gradient(90deg, var(--accent), var(--accent-3));
      border-radius: inherit;
      height: 100%;
      min-width: 2px;
    }
    .fill.orange { background: linear-gradient(90deg, var(--accent-2), #d2a53b); }
    .fill.warn { background: linear-gradient(90deg, var(--warn), var(--accent-2)); }
    .bar-value {
      color: var(--text);
      font-variant-numeric: tabular-nums;
      text-align: right;
      white-space: nowrap;
    }
    .task-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
    }
    .tag {
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      display: inline-flex;
      font-size: 0.78rem;
      padding: 3px 8px;
    }
    .tag.good { color: var(--accent-3); }
    .tag.warn { color: var(--warn); }
    .two-col {
      display: grid;
      gap: 14px;
      grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr);
    }
    .reliability {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(72px, 1fr));
      min-height: 200px;
      align-items: end;
      margin-top: 14px;
    }
    .bin {
      display: grid;
      gap: 5px;
      grid-template-rows: minmax(126px, 1fr) auto;
      min-width: 0;
    }
    .bin-bars {
      align-items: end;
      display: grid;
      gap: 4px;
      grid-template-columns: 1fr 1fr;
      height: 126px;
    }
    .bin-bar {
      border-radius: 5px 5px 0 0;
      min-height: 2px;
    }
    .bin-bar.conf { background: var(--accent); }
    .bin-bar.acc { background: var(--accent-2); }
    .bin span {
      color: var(--muted);
      font-size: 0.76rem;
      overflow-wrap: anywhere;
      text-align: center;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 0.86rem;
      margin-top: 10px;
    }
    .legend i {
      border-radius: 999px;
      display: inline-block;
      height: 10px;
      margin-right: 5px;
      width: 10px;
    }
    .tools {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    input[type="search"], select {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      min-height: 36px;
      padding: 7px 9px;
    }
    table {
      background: var(--panel);
      border: 1px solid var(--line);
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    td {
      max-width: 420px;
      overflow-wrap: anywhere;
    }
    tr:last-child td { border-bottom: 0; }
    tbody tr.selectable { cursor: pointer; }
    tbody tr.selectable:hover { background: var(--panel-soft); }
    .table-wrap {
      border-radius: 8px;
      overflow-x: auto;
      width: 100%;
    }
    .hash { color: var(--muted); }
    .ledger-status {
      border-radius: 999px;
      display: inline-flex;
      font-size: 0.78rem;
      font-weight: 750;
      padding: 3px 8px;
      white-space: nowrap;
    }
    .ledger-status.live {
      background: rgba(20, 108, 120, 0.12);
      color: var(--accent);
    }
    .ledger-status.check {
      background: rgba(99, 123, 61, 0.14);
      color: var(--accent-3);
    }
    .ledger-status.warn {
      background: rgba(154, 77, 93, 0.13);
      color: var(--warn);
    }
    details {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 12px;
      padding: 12px 14px;
    }
    summary { cursor: pointer; font-weight: 700; }
    .empty, .error {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 14px;
    }
    .verification {
      margin-top: 14px;
      padding: 12px 14px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
    }
    .verification strong { color: var(--text); }
    @media (max-width: 780px) {
      .section-head, .two-col, .paper-title, .argument-grid, .comparison-row { grid-template-columns: 1fr; }
      .tools { justify-content: flex-start; }
      .bar-row { grid-template-columns: 1fr; }
      .paired-bar { grid-template-columns: 72px minmax(120px, 1fr) 58px; }
      .bar-value { text-align: left; }
      header { margin-bottom: 16px; }
    }
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="paper-title">
      <div>
        <div class="eyebrow">Pre-rendered live paper generated from the evidence workflow</div>
        <h1>CRLPLRIMES: certified rule learning for verifier-grounded action ranking</h1>
        <p class="lede">The paper argument, figures, model results, and reproducibility ledger are all rendered from the current JSONL evidence and checkpoint manifest at build time.</p>
        <p class="paper-kicker">Browser JavaScript is only a verification and hydration layer. The claims below are readable in crawlers, archives, text browsers, and no-JS environments.</p>
      </div>
      <aside class="paper-note">
        <strong>Reader contract</strong>
        <p>Prose cannot outrun evidence here: the public story is generated from artifacts, and the artifacts carry hashes, provenance, and text-readable mirrors.</p>
      </aside>
    </div>
    <div class="run-meta">
      <span class="pill">Generated: <strong>${escapeHtml(generatedUtc)}</strong></span>
      <span class="pill">Run: <strong>${escapeHtml(runId)}</strong></span>
      <span class="pill">Commit: <strong>${escapeHtml(shortHash(commit))}</strong></span>
      <span class="pill">Scalar: <strong>${escapeHtml(real)}</strong></span>
      <span class="pill">Models: <strong>${modelFiles.length}</strong></span>
      <span class="pill">Results: <strong>${resultFiles.length}</strong></span>
    </div>
    <nav aria-label="Report sections">
      <a href="#abstract">Abstract</a>
      <a href="#argument">Argument</a>
      <a href="#figures">Figures</a>
      <a href="#ledger">Claim ledger</a>
      <a href="#paper-claims">Claim compiler</a>
      <a href="#claims">Task declarations</a>
      <a href="#model-results">Model results</a>
      <a href="#rule-search">Rule search</a>
      <a href="#calibration">Calibration</a>
      <a href="#artifacts">Artifacts</a>
    </nav>
    <div class="verification" id="verification-status"><strong>Static render:</strong> readable without JavaScript. Browser verification has not run.</div>
    <noscript><div class="verification"><strong>No JavaScript:</strong> this page remains readable; only hash re-verification and table filtering are disabled.</div></noscript>
  </div>
</header>
<main>
  <section id="abstract">
    <div class="section-head">
      <div>
        <h2>Abstract</h2>
        <p>${escapeHtml(abstractText)}</p>
      </div>
    </div>
    <div class="grid">${renderMetrics()}</div>
  </section>

  <section id="argument">
    <div class="section-head">
      <div>
        <h2>Argument</h2>
        <p>The live paper follows the same spine as the experiment: declare scope, certify gates against a verifier, measure model ranking and saved verifier work, then publish failures and calibration beside successes.</p>
      </div>
    </div>
    <div class="argument-grid">
      <article class="argument-card"><strong>Question</strong><p>Can a learned rule proposer discover certified gates that shrink verifier work while preserving the correct action?</p></article>
      <article class="argument-card"><strong>Positive row</strong><p>${escapeHtml(primeGateSentence())}</p></article>
      <article class="argument-card"><strong>Negative row</strong><p>${escapeHtml(c2Sentence())}</p></article>
    </div>
  </section>

  <section id="figures">
    <div class="section-head">
      <div>
        <h2>Main Figures</h2>
        <p>These figure panels are pre-rendered from the published evidence bundle. They remain readable without client-side parsing.</p>
      </div>
    </div>
    <div class="grid wide">
      <figure class="panel figure">
        <div><div class="figure-label">Figure 1</div><h3>Certified gates change the ranking problem</h3></div>
        <div class="comparison-bars">${gateFigure.html}</div>
        <figcaption class="caption">${escapeHtml(gateFigure.caption)}</figcaption>
      </figure>
      <figure class="panel figure">
        <div><div class="figure-label">Figure 2</div><h3>Rule search reduces exact verifier work</h3></div>
        <div class="bar-list">${workFigure.html}</div>
        <figcaption class="caption">${escapeHtml(workFigure.caption)}</figcaption>
      </figure>
      <figure class="panel figure">
        <div><div class="figure-label">Figure 3</div><h3>Calibration remains inspectable</h3></div>
        <div class="bar-list">${calibrationFigure.html}</div>
        <figcaption class="caption">${escapeHtml(calibrationFigure.caption)}</figcaption>
      </figure>
    </div>
  </section>

  <section id="ledger">
    <div class="section-head">
      <div>
        <h2>Claim Ledger</h2>
        <p>Each row links a public claim to the live evidence class that supports it. The C2 negative baseline is intentionally visible here.</p>
      </div>
    </div>
    <div class="table-wrap">${renderClaimLedger()}</div>
  </section>

  <section id="paper-claims">
    <div class="section-head">
      <div>
        <h2>Paper Claim Compiler</h2>
        <p>Machine-checked claim statuses from <code>paper/generated/claims.json</code>, copied into the published evidence bundle.</p>
      </div>
    </div>
    <div class="table-wrap">${renderClaimCompilerTable()}</div>
  </section>

  <section id="claims">
    <div class="section-head">
      <div>
        <h2>Task Declarations</h2>
        <p>Experiment declarations are rendered from <code>crlp.experiment_schema.v1</code> rows so displayed scope follows captured evidence.</p>
      </div>
    </div>
    <div class="grid wide">${renderTaskCards()}</div>
  </section>

  <section id="model-results">
    <div class="section-head">
      <div>
        <h2>Model Results</h2>
        <p>Best observed model or gate result per task, plus the pre-rendered comparison table. JavaScript only filters; it does not create the table.</p>
      </div>
      <div class="tools"><input id="result-filter" type="search" placeholder="Filter task, mode, hash"></div>
    </div>
    <div class="two-col">
      <div class="panel"><h3>Best accuracy by task</h3><div class="bar-list">${renderBarList(bestAccuracyBars, 1)}</div></div>
      <div class="panel"><h3>Best verifier work saved</h3><div class="bar-list">${renderBarList(bestWorkBars, Math.max(100, ...bestWorkBars.map((row) => row.value || 0)))}</div></div>
    </div>
    <div class="panel" style="margin-top:14px"><h3>Referenced model hashes</h3><div class="table-wrap">${renderModelHashTable()}</div></div>
    <div class="table-wrap" style="margin-top:14px">
      <table id="results-table">
        <thead><tr><th>Task</th><th>Mode</th><th>Accuracy</th><th>Top-5</th><th>ECE</th><th>Work</th><th>Rules</th><th>False rejects</th><th>Model hash</th></tr></thead>
        <tbody>${renderResultsRows()}</tbody>
      </table>
    </div>
  </section>

  <section id="rule-search">
    <div class="section-head">
      <div>
        <h2>Certified Rule Search</h2>
        <p>Rule-search summaries compare proposer strategies, accepted certificates, and exact verifier work saved.</p>
      </div>
    </div>
    <div class="two-col">
      <div class="panel"><h3>Best saved work by strategy</h3><div class="bar-list">${renderBarList(ruleSavedBars, Math.max(100, ...ruleSavedBars.map((row) => row.value || 0)))}</div></div>
      <div class="panel"><h3>Rule search leaders</h3><div class="table-wrap">${renderRuleTable()}</div></div>
    </div>
  </section>

  <section id="calibration">
    <div class="section-head">
      <div>
        <h2>Calibration</h2>
        <p>Reliability bins compare model confidence against empirical correctness. The first row is pre-rendered; JavaScript can update the selection.</p>
      </div>
    </div>
    <div class="panel">
      <h3 id="reliability-title">${escapeHtml(reliability.title)}</h3>
      <div class="legend"><span><i style="background:var(--accent)"></i>confidence</span><span><i style="background:var(--accent-2)"></i>accuracy</span></div>
      <div id="reliability-chart" class="reliability">${reliability.html}</div>
    </div>
  </section>

  <section id="artifacts">
    <div class="section-head">
      <div>
        <h2>Models And Raw Artifacts</h2>
        <p>Model checkpoints and certified rule checkpoints are first-class outputs. JSONL files also have <code>.txt</code> mirrors for browser and tool readability.</p>
      </div>
    </div>
    <div class="panel"><h3>Published model files</h3><div class="table-wrap">${renderArtifactTable(modelFiles)}</div></div>
    <details>
      <summary>Raw result files and manifest</summary>
      <div class="table-wrap" style="margin-top:12px">${renderArtifactTable(resultFiles)}</div>
    </details>
    <div class="run-meta" style="margin-top:14px">
      <a class="pill" href="results/summary.md"><strong>summary.md</strong></a>
      <a class="pill" href="results/comparison.jsonl.txt"><strong>comparison.jsonl.txt</strong></a>
      <a class="pill" href="results/comparison.canonical.jsonl.txt"><strong>comparison.canonical.jsonl.txt</strong></a>
      <a class="pill" href="results/commands.txt"><strong>commands.txt</strong></a>
      <a class="pill" href="results/environment.json"><strong>environment.json</strong></a>
      <a class="pill" href="results/paper_claims.json"><strong>paper_claims.json</strong></a>
      <a class="pill" href="manifest.tsv"><strong>manifest.tsv</strong></a>
    </div>
  </section>
</main>
<script>
(function () {
  const expectedComparisonSha = ${JSON.stringify(comparisonSha)};
  const expectedCanonicalSha = ${JSON.stringify(canonicalSha)};

  async function sha256Hex(text) {
    if (!window.crypto || !window.crypto.subtle) return "";
    const bytes = new TextEncoder().encode(text);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest)).map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  async function verify() {
    const status = document.getElementById("verification-status");
    if (!status) return;
    try {
      const [comparison, canonical] = await Promise.all([
        fetch("results/comparison.jsonl", { cache: "no-store" }).then((response) => response.text()),
        fetch("results/comparison.canonical.jsonl", { cache: "no-store" }).then((response) => response.text()),
      ]);
      const comparisonSha = await sha256Hex(comparison);
      const canonicalSha = await sha256Hex(canonical);
      const rows = comparison.split(/\\r?\\n/).filter(Boolean).length;
      const ok = comparisonSha === expectedComparisonSha && canonicalSha === expectedCanonicalSha;
      status.innerHTML = ok
        ? "<strong>Browser verification passed:</strong> fetched " + rows + " comparison rows and matched published SHA-256 hashes."
        : "<strong>Browser verification mismatch:</strong> fetched artifacts did not match the build-time hashes.";
    } catch (error) {
      status.innerHTML = "<strong>Browser verification unavailable:</strong> " + String(error && error.message ? error.message : error);
    }
  }

  function bindFilter() {
    const input = document.getElementById("result-filter");
    const table = document.getElementById("results-table");
    if (!input || !table) return;
    input.addEventListener("input", function () {
      const query = input.value.trim().toLowerCase();
      for (const row of table.querySelectorAll("tbody tr")) {
        row.hidden = query && !row.textContent.toLowerCase().includes(query);
      }
    });
  }

  verify();
  bindFilter();
}());
</script>
</body>
</html>
`;

writeFileSync(path.join(siteDir, "index.html"), html);
