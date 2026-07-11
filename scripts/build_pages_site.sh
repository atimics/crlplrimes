#!/bin/sh
set -eu

real="${REAL:-float}"
results_dir="${PAGES_RESULTS_DIR:-results/latest}"
site_dir="${PAGES_SITE_DIR:-build/pages-site}"
model_dirs="${PAGES_MODEL_DIRS:-build/$real}"

if [ -e "$site_dir" ]; then
    echo "refusing to overwrite existing site dir: $site_dir" >&2
    exit 1
fi

if [ ! -d "$results_dir" ]; then
    echo "missing results dir: $results_dir" >&2
    exit 1
fi

mkdir -p "$site_dir/results" "$site_dir/models"
cp -R "$results_dir"/. "$site_dir/results/"
: > "$site_dir/.nojekyll"

for model_dir in $model_dirs; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    find "$model_dir" -type f \( -name '*.nnckpt' -o -name '*.crlprule' \) | sort | while IFS= read -r src; do
        rel="${src#"$model_dir"/}"
        dest="$site_dir/models/$rel"
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
    done
done

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

file_size() {
    if stat -c %s "$1" >/dev/null 2>&1; then
        stat -c %s "$1"
    else
        stat -f %z "$1"
    fi
}

html_escape() {
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g'
}

js_escape() {
    sed 's/\\/\\\\/g; s/"/\\"/g'
}

write_manifest() {
    manifest="$site_dir/manifest.tsv"
    printf 'category\tpath\tbytes\tsha256\n' > "$manifest"
    for category in results models; do
        if [ ! -d "$site_dir/$category" ]; then
            continue
        fi
        find "$site_dir/$category" -type f | sort | while IFS= read -r file; do
            rel="${file#"$site_dir"/}"
            bytes="$(file_size "$file")"
            sha="$(hash_file "$file")"
            printf '%s\t%s\t%s\t%s\n' "$category" "$rel" "$bytes" "$sha"
        done
    done >> "$manifest"
}

write_index() {
    generated_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    run_id="${GITHUB_RUN_ID:-local}"
    sha="${GITHUB_SHA:-unknown}"
    model_count="$(find "$site_dir/models" -type f | wc -l | tr -d ' ')"
    result_count="$(find "$site_dir/results" -type f | wc -l | tr -d ' ')"

    generated_utc_js="$(printf '%s' "$generated_utc" | js_escape)"
    run_id_js="$(printf '%s' "$run_id" | js_escape)"
    sha_js="$(printf '%s' "$sha" | js_escape)"
    real_js="$(printf '%s' "$real" | js_escape)"

    cat > "$site_dir/index.html" <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRLPLRIMES Live Evidence Report</title>
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
      background:
        linear-gradient(90deg, rgba(20, 108, 120, 0.10), transparent 38%),
        linear-gradient(180deg, var(--band), var(--bg));
      border-bottom: 1px solid var(--line);
      margin-bottom: 28px;
    }
    .header-inner {
      width: min(1240px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 24px;
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
    .metric, .panel, .task-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
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
    .panel { padding: 16px; }
    .task-card { padding: 14px; }
    .task-card p { margin: 8px 0 0; }
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
    .bar-list {
      display: grid;
      gap: 11px;
      margin-top: 14px;
    }
    .bar-row {
      display: grid;
      gap: 7px;
      grid-template-columns: minmax(130px, 0.9fr) minmax(140px, 1.8fr) minmax(70px, auto);
      align-items: center;
    }
    .bar-label {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .bar-label small { color: var(--muted); display: block; }
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
      max-width: 360px;
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
    .error { border-color: var(--warn); color: var(--warn); }
    .loading {
      animation: pulse 1.4s ease-in-out infinite;
      color: var(--muted);
    }
    @keyframes pulse { 50% { opacity: 0.45; } }
    @media (max-width: 780px) {
      .section-head, .two-col { grid-template-columns: 1fr; }
      .tools { justify-content: flex-start; }
      .bar-row { grid-template-columns: 1fr; }
      .bar-value { text-align: left; }
      header { margin-bottom: 16px; }
    }
  </style>
  <script>
    window.CRLPLRIMES_PAGE_META = {
      generatedUtc: "${generated_utc_js}",
      runId: "${run_id_js}",
      commit: "${sha_js}",
      real: "${real_js}",
      modelFileCount: ${model_count},
      resultFileCount: ${result_count}
    };
  </script>
</head>
<body>
HTML

    cat >> "$site_dir/index.html" <<'HTML'
<header>
  <div class="header-inner">
    <div class="eyebrow">Published evidence workflow</div>
    <h1>CRLPLRIMES live evidence report</h1>
    <p class="lede">A static, reproducible report generated from the same JSONL artifacts and model checkpoints that ship with the repository. The page parses the latest comparison rows, certified rule search summaries, calibration bins, and checkpoint manifest directly in the browser.</p>
    <div class="run-meta" id="run-meta"></div>
    <nav aria-label="Report sections">
      <a href="#abstract">Abstract</a>
      <a href="#claims">Claims</a>
      <a href="#model-results">Model results</a>
      <a href="#rule-search">Rule search</a>
      <a href="#calibration">Calibration</a>
      <a href="#artifacts">Artifacts</a>
    </nav>
  </div>
</header>
<main>
  <section id="abstract">
    <div class="section-head">
      <div>
        <h2>Abstract</h2>
        <p id="summary-text">Loading report data...</p>
      </div>
    </div>
    <div class="grid" id="headline-metrics">
      <div class="metric loading"><strong>...</strong><span>loading comparison rows</span></div>
    </div>
  </section>

  <section id="claims">
    <div class="section-head">
      <div>
        <h2>Task Claims</h2>
        <p>Experiment declarations are rendered from <code>crlp.experiment_schema.v1</code> rows so the displayed claim scope follows the captured evidence.</p>
      </div>
    </div>
    <div class="grid wide" id="claim-cards"></div>
  </section>

  <section id="model-results">
    <div class="section-head">
      <div>
        <h2>Model Results</h2>
        <p>Best observed model or gate result per task, plus a filterable comparison table. Click a comparison row to update the reliability panel.</p>
      </div>
      <div class="tools">
        <input id="result-filter" type="search" placeholder="Filter task, mode, hash">
      </div>
    </div>
    <div class="two-col">
      <div class="panel">
        <h3>Best accuracy by task</h3>
        <div id="accuracy-chart" class="bar-list"></div>
      </div>
      <div class="panel">
        <h3>Best verifier work saved</h3>
        <div id="work-chart" class="bar-list"></div>
      </div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>Referenced model hashes</h3>
      <div class="table-wrap" id="model-hash-table"></div>
    </div>
    <div class="table-wrap" style="margin-top:14px">
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Mode</th>
            <th>Accuracy</th>
            <th>Top-5</th>
            <th>ECE</th>
            <th>Work</th>
            <th>Rules</th>
            <th>False rejects</th>
            <th>Model hash</th>
          </tr>
        </thead>
        <tbody id="result-rows"></tbody>
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
      <div class="panel">
        <h3>Best saved work by strategy</h3>
        <div id="rule-saved-chart" class="bar-list"></div>
      </div>
      <div class="panel">
        <h3>Rule search leaders</h3>
        <div class="table-wrap" id="rule-table"></div>
      </div>
    </div>
  </section>

  <section id="calibration">
    <div class="section-head">
      <div>
        <h2>Calibration</h2>
        <p>Reliability bins compare model confidence against empirical correctness. Use the result table above to inspect another row.</p>
      </div>
      <div class="tools">
        <select id="reliability-select" aria-label="Reliability row"></select>
      </div>
    </div>
    <div class="panel">
      <h3 id="reliability-title">Reliability bins</h3>
      <div class="legend">
        <span><i style="background:var(--accent)"></i>confidence</span>
        <span><i style="background:var(--accent-2)"></i>accuracy</span>
      </div>
      <div id="reliability-chart" class="reliability"></div>
    </div>
  </section>

  <section id="artifacts">
    <div class="section-head">
      <div>
        <h2>Models And Raw Artifacts</h2>
        <p>Model checkpoints and certified rule checkpoints are first-class outputs. Raw JSONL files remain available for reproducibility and downstream analysis.</p>
      </div>
    </div>
    <div class="panel">
      <h3>Published model files</h3>
      <div class="table-wrap" id="model-artifacts"></div>
    </div>
    <details>
      <summary>Raw result files and manifest</summary>
      <div class="table-wrap" id="result-artifacts" style="margin-top:12px"></div>
    </details>
    <div class="run-meta" style="margin-top:14px">
      <a class="pill" href="results/summary.md"><strong>summary.md</strong></a>
      <a class="pill" href="results/comparison.jsonl"><strong>comparison.jsonl</strong></a>
      <a class="pill" href="results/comparison.canonical.jsonl"><strong>comparison.canonical.jsonl</strong></a>
      <a class="pill" href="results/commands.txt"><strong>commands.txt</strong></a>
      <a class="pill" href="manifest.tsv"><strong>manifest.tsv</strong></a>
    </div>
  </section>
</main>
<script>
(function () {
  "use strict";

  const meta = window.CRLPLRIMES_PAGE_META || {};
  const state = {
    rows: [],
    compareRows: [],
    ruleRows: [],
    schemaRows: [],
    manifestRows: [],
    reliabilityRows: []
  };

  const els = {
    runMeta: document.getElementById("run-meta"),
    summary: document.getElementById("summary-text"),
    metrics: document.getElementById("headline-metrics"),
    claims: document.getElementById("claim-cards"),
    accuracyChart: document.getElementById("accuracy-chart"),
    workChart: document.getElementById("work-chart"),
    resultFilter: document.getElementById("result-filter"),
    resultRows: document.getElementById("result-rows"),
    modelHashTable: document.getElementById("model-hash-table"),
    ruleSavedChart: document.getElementById("rule-saved-chart"),
    ruleTable: document.getElementById("rule-table"),
    reliabilitySelect: document.getElementById("reliability-select"),
    reliabilityTitle: document.getElementById("reliability-title"),
    reliabilityChart: document.getElementById("reliability-chart"),
    modelArtifacts: document.getElementById("model-artifacts"),
    resultArtifacts: document.getElementById("result-artifacts")
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function hrefFor(path) {
    return String(path || "").split("/").map(encodeURIComponent).join("/");
  }

  function finite(value) {
    if (value == null || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatNumber(value, digits) {
    const number = finite(value);
    if (number == null) return "n/a";
    return number.toLocaleString(undefined, {
      maximumFractionDigits: digits == null ? 2 : digits,
      minimumFractionDigits: digits == null ? 0 : 0
    });
  }

  function formatPct(value, digits) {
    const number = finite(value);
    if (number == null) return "n/a";
    return (number * 100).toFixed(digits == null ? 1 : digits) + "%";
  }

  function formatPctPoints(value, digits) {
    const number = finite(value);
    if (number == null) return "n/a";
    return number.toFixed(digits == null ? 1 : digits) + "%";
  }

  function shortHash(hash) {
    if (!hash || hash === "none" || hash === "unknown") return hash || "n/a";
    return String(hash).slice(0, 12);
  }

  function modeLabel(row) {
    return String(row.evaluation_mode || row.selection_strategy || "unspecified")
      .replace(/^certified_threshold_margin_gate_/, "certified ")
      .replace(/_/g, " ");
  }

  function weightedAccuracy(row) {
    if (finite(row.top1) != null) return finite(row.top1);
    const bins = Array.isArray(row.reliability_bins) ? row.reliability_bins : [];
    let count = 0;
    let correct = 0;
    bins.forEach(function (bin) {
      const binCount = finite(bin.count);
      const acc = finite(bin.accuracy);
      if (binCount != null && acc != null) {
        count += binCount;
        correct += binCount * acc;
      }
    });
    return count > 0 ? correct / count : null;
  }

  function workSavedPct(row) {
    const explicit = finite(row.expanded_saved_pct);
    if (explicit != null) return explicit;
    const direct = finite(row.direct_exact);
    const proposal = finite(row.proposal_steps) != null ? finite(row.proposal_steps) : finite(row.expanded_exact);
    if (direct == null || direct <= 0 || proposal == null) return null;
    return Math.max(0, Math.min(100, (1 - proposal / direct) * 100));
  }

  function rowScore(row) {
    const acc = weightedAccuracy(row);
    const work = workSavedPct(row);
    const ece = finite(row.ece);
    return [
      acc == null ? -Infinity : acc,
      work == null ? -Infinity : work / 1000,
      ece == null ? 0 : -ece / 1000
    ].reduce(function (sum, value) { return sum + value; }, 0);
  }

  function parseJsonl(text, source) {
    return String(text || "").split(/\r?\n/).reduce(function (rows, line, index) {
      const trimmed = line.trim();
      if (!trimmed) return rows;
      try {
        rows.push(JSON.parse(trimmed));
      } catch (error) {
        console.warn("Skipping invalid JSONL row in " + source + ":" + (index + 1), error);
      }
      return rows;
    }, []);
  }

  function parseManifest(text) {
    return String(text || "").split(/\r?\n/).slice(1).reduce(function (rows, line) {
      if (!line.trim()) return rows;
      const cols = line.split("\t");
      rows.push({
        category: cols[0],
        path: cols[1],
        bytes: Number(cols[2]),
        sha256: cols[3]
      });
      return rows;
    }, []);
  }

  async function readText(path, optional) {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      if (optional) return "";
      throw new Error(path + " returned HTTP " + response.status);
    }
    return response.text();
  }

  function renderRunMeta() {
    els.runMeta.innerHTML = [
      ["Generated", meta.generatedUtc || "unknown"],
      ["Run", meta.runId || "local"],
      ["Commit", shortHash(meta.commit || "unknown")],
      ["Scalar", meta.real || "unknown"],
      ["Models", meta.modelFileCount],
      ["Results", meta.resultFileCount]
    ].map(function (item) {
      return '<span class="pill">' + escapeHtml(item[0]) + ': <strong>' + escapeHtml(item[1]) + "</strong></span>";
    }).join("");
  }

  function renderMetrics() {
    const tasks = new Set();
    state.schemaRows.forEach(function (row) { if (row.task) tasks.add(row.task); });
    state.compareRows.forEach(function (row) { if (row.task) tasks.add(row.task); });
    const bestAccuracy = state.compareRows.reduce(function (best, row) {
      const acc = weightedAccuracy(row);
      return acc == null ? best : Math.max(best, acc);
    }, -Infinity);
    const bestSaved = state.ruleRows.concat(state.compareRows).reduce(function (best, row) {
      const saved = workSavedPct(row);
      return saved == null ? best : Math.max(best, saved);
    }, -Infinity);
    const zeroFalseRejectRows = state.compareRows.filter(function (row) { return finite(row.false_rejections) === 0; }).length;
    const modelHashes = new Set(state.compareRows.map(function (row) { return row.model_hash; }).filter(function (hash) {
      return hash && hash !== "none" && hash !== "unknown";
    }));

    els.summary.textContent =
      "CRLPLRIMES evaluates verifier-grounded action ranking, certified candidate pruning, and rule-proposer behavior across declared task families. This page is regenerated from the promoted evidence workflow and keeps the paper claims, charts, comparison rows, calibration bins, and model artifacts in one inspectable report.";

    els.metrics.innerHTML = [
      { value: tasks.size, label: "declared task families", sub: state.schemaRows.length + " experiment-schema rows" },
      { value: state.compareRows.length, label: "comparison rows", sub: "model, gate, and calibration outcomes" },
      { value: bestAccuracy === -Infinity ? "n/a" : formatPct(bestAccuracy), label: "best observed accuracy", sub: "from top-1 or reliability bins" },
      { value: bestSaved === -Infinity ? "n/a" : formatPctPoints(bestSaved), label: "best verifier work saved", sub: "from certified search summaries" },
      { value: modelHashes.size, label: "referenced model hashes", sub: meta.modelFileCount + " checkpoint/rule files published" },
      { value: zeroFalseRejectRows, label: "zero false-reject rows", sub: "rows with explicit false_rejections = 0" }
    ].map(function (metric) {
      return '<div class="metric"><strong>' + escapeHtml(metric.value) + '</strong><span>' + escapeHtml(metric.label) + '</span><small>' + escapeHtml(metric.sub) + "</small></div>";
    }).join("");
  }

  function renderClaims() {
    if (!state.schemaRows.length) {
      els.claims.innerHTML = '<div class="empty">No experiment-schema rows were found.</div>';
      return;
    }
    els.claims.innerHTML = state.schemaRows.map(function (row) {
      const audit = row.feature_audit || {};
      const warning = audit.has_warning ? '<span class="tag warn">feature warning</span>' : '<span class="tag good">policy compliant</span>';
      return '<article class="task-card">'
        + '<h3>' + escapeHtml(row.task || "unknown task") + '</h3>'
        + '<p>' + escapeHtml(row.role || row.claim_scope || "No role declared.") + '</p>'
        + '<p>' + escapeHtml(row.claim_scope || "") + '</p>'
        + '<div class="task-meta">'
        + '<span class="tag">' + escapeHtml(row.leakage_policy || "leakage policy n/a") + '</span>'
        + '<span class="tag">' + escapeHtml(row.proposer_mode || "mode n/a") + '</span>'
        + '<span class="tag">' + escapeHtml((audit.enabled_features == null ? "n/a" : audit.enabled_features) + "/" + (audit.total_features == null ? "n/a" : audit.total_features) + " features") + '</span>'
        + warning
        + '</div>'
        + '</article>';
    }).join("");
  }

  function bestRowsByTask(rows, valueFn) {
    const byTask = new Map();
    rows.forEach(function (row) {
      if (!row.task) return;
      const value = valueFn(row);
      if (value == null) return;
      const current = byTask.get(row.task);
      if (!current || value > current.value || (value === current.value && rowScore(row) > rowScore(current.row))) {
        byTask.set(row.task, { row: row, value: value });
      }
    });
    return Array.from(byTask.values()).sort(function (a, b) {
      return b.value - a.value || String(a.row.task).localeCompare(String(b.row.task));
    });
  }

  function renderBarList(el, rows, options) {
    if (!rows.length) {
      el.innerHTML = '<div class="empty">No rows available.</div>';
      return;
    }
    const max = options.max == null ? Math.max.apply(null, rows.map(function (row) { return row.value; })) : options.max;
    el.innerHTML = rows.map(function (item) {
      const width = max > 0 ? Math.max(1, Math.min(100, item.value / max * 100)) : 0;
      const klass = item.kind === "orange" ? "fill orange" : item.kind === "warn" ? "fill warn" : "fill";
      return '<div class="bar-row">'
        + '<div class="bar-label"><strong>' + escapeHtml(item.label) + '</strong><small>' + escapeHtml(item.sub || "") + '</small></div>'
        + '<div class="track"><div class="' + klass + '" style="width:' + width.toFixed(2) + '%"></div></div>'
        + '<div class="bar-value">' + escapeHtml(item.display) + '</div>'
        + '</div>';
    }).join("");
  }

  function renderModelCharts() {
    const bestAccuracy = bestRowsByTask(state.compareRows, weightedAccuracy).map(function (item) {
      return {
        label: item.row.task,
        sub: modeLabel(item.row),
        value: item.value,
        display: formatPct(item.value)
      };
    });
    renderBarList(els.accuracyChart, bestAccuracy, { max: 1 });

    const bestWork = bestRowsByTask(state.compareRows, workSavedPct).map(function (item) {
      return {
        label: item.row.task,
        sub: modeLabel(item.row),
        value: item.value,
        display: formatPctPoints(item.value),
        kind: "orange"
      };
    });
    renderBarList(els.workChart, bestWork, { max: Math.max(100, bestWork.reduce(function (max, row) { return Math.max(max, row.value); }, 0)) });
  }

  function renderModelHashTable() {
    const groups = new Map();
    state.compareRows.forEach(function (row) {
      const hash = row.model_hash;
      if (!hash || hash === "none" || hash === "unknown") return;
      if (!groups.has(hash)) {
        groups.set(hash, { hash: hash, rows: [], tasks: new Set(), modes: new Set(), bestAcc: null, bestEce: null, bestWork: null });
      }
      const group = groups.get(hash);
      group.rows.push(row);
      if (row.task) group.tasks.add(row.task);
      if (row.evaluation_mode) group.modes.add(row.evaluation_mode);
      const acc = weightedAccuracy(row);
      const ece = finite(row.ece);
      const work = workSavedPct(row);
      if (acc != null) group.bestAcc = group.bestAcc == null ? acc : Math.max(group.bestAcc, acc);
      if (ece != null) group.bestEce = group.bestEce == null ? ece : Math.min(group.bestEce, ece);
      if (work != null) group.bestWork = group.bestWork == null ? work : Math.max(group.bestWork, work);
    });
    const rows = Array.from(groups.values()).sort(function (a, b) {
      return (b.bestAcc || -1) - (a.bestAcc || -1) || a.hash.localeCompare(b.hash);
    }).slice(0, 24);
    els.modelHashTable.innerHTML = renderTable(
      ["Model hash", "Tasks", "Modes", "Best accuracy", "Best ECE", "Best work saved"],
      rows.map(function (group) {
        return [
          '<span class="mono hash" title="' + escapeHtml(group.hash) + '">' + escapeHtml(shortHash(group.hash)) + '</span>',
          escapeHtml(Array.from(group.tasks).sort().join(", ")),
          escapeHtml(Array.from(group.modes).slice(0, 4).map(function (mode) { return mode.replace(/_/g, " "); }).join(", ") + (group.modes.size > 4 ? " ..." : "")),
          escapeHtml(formatPct(group.bestAcc)),
          escapeHtml(formatPct(group.bestEce)),
          escapeHtml(formatPctPoints(group.bestWork))
        ];
      })
    );
  }

  function renderResultsTable() {
    const query = String(els.resultFilter.value || "").trim().toLowerCase();
    const rows = state.compareRows.filter(function (row) {
      if (!query) return true;
      return [row.task, row.evaluation_mode, row.model_hash, row.rule_set_hash, row.feature_set].join(" ").toLowerCase().indexOf(query) >= 0;
    }).sort(function (a, b) {
      return String(a.task).localeCompare(String(b.task))
        || rowScore(b) - rowScore(a)
        || String(a.evaluation_mode).localeCompare(String(b.evaluation_mode));
    }).slice(0, 180);

    els.resultRows.innerHTML = rows.map(function (row, index) {
      const originalIndex = state.compareRows.indexOf(row);
      return '<tr class="selectable" data-row-index="' + originalIndex + '">'
        + '<td>' + escapeHtml(row.task || "") + '</td>'
        + '<td>' + escapeHtml(modeLabel(row)) + '</td>'
        + '<td>' + escapeHtml(formatPct(weightedAccuracy(row))) + '</td>'
        + '<td>' + escapeHtml(formatPct(row.top5)) + '</td>'
        + '<td>' + escapeHtml(formatPct(row.ece)) + '</td>'
        + '<td>' + escapeHtml(formatPctPoints(workSavedPct(row))) + '</td>'
        + '<td>' + escapeHtml(row.accepted_rules == null ? "n/a" : row.accepted_rules) + '</td>'
        + '<td>' + escapeHtml(row.false_rejections == null ? "n/a" : row.false_rejections) + '</td>'
        + '<td><span class="mono hash" title="' + escapeHtml(row.model_hash || "") + '">' + escapeHtml(shortHash(row.model_hash)) + '</span></td>'
        + '</tr>';
    }).join("") || '<tr><td colspan="9">No matching rows.</td></tr>';
  }

  function renderRuleSearch() {
    const leaders = bestRowsByTask(state.ruleRows, workSavedPct).map(function (item) {
      return {
        label: item.row.task,
        sub: String(item.row.selection_strategy || "strategy n/a").replace(/_/g, " "),
        value: item.value,
        display: formatPctPoints(item.value),
        kind: "orange"
      };
    });
    renderBarList(els.ruleSavedChart, leaders, { max: Math.max(100, leaders.reduce(function (max, row) { return Math.max(max, row.value); }, 0)) });

    const rows = state.ruleRows.slice().filter(function (row) {
      return workSavedPct(row) != null;
    }).sort(function (a, b) {
      return (workSavedPct(b) || -1) - (workSavedPct(a) || -1);
    }).slice(0, 18);
    els.ruleTable.innerHTML = renderTable(
      ["Task", "Strategy", "Saved", "Accepted", "Exact work", "Rule set"],
      rows.map(function (row) {
        return [
          escapeHtml(row.task || ""),
          escapeHtml(String(row.selection_strategy || "").replace(/_/g, " ")),
          escapeHtml(formatPctPoints(workSavedPct(row))),
          escapeHtml(row.accepted_rules == null ? "n/a" : row.accepted_rules),
          escapeHtml(formatNumber(row.expanded_exact, 3) + " / " + formatNumber(row.direct_exact, 3)),
          '<span class="mono hash" title="' + escapeHtml(row.rule_set_hash || "") + '">' + escapeHtml(shortHash(row.rule_set_hash)) + '</span>'
        ];
      })
    );
  }

  function renderReliabilityControls() {
    state.reliabilityRows = state.compareRows.filter(function (row) {
      return Array.isArray(row.reliability_bins) && row.reliability_bins.length;
    }).sort(function (a, b) {
      return rowScore(b) - rowScore(a);
    });
    els.reliabilitySelect.innerHTML = state.reliabilityRows.map(function (row, index) {
      return '<option value="' + index + '">' + escapeHtml((row.task || "task") + " - " + modeLabel(row) + " - " + formatPct(weightedAccuracy(row))) + '</option>';
    }).join("");
    if (state.reliabilityRows.length) renderReliabilityRow(0);
  }

  function renderReliabilityRow(index) {
    const row = state.reliabilityRows[index];
    if (!row) {
      els.reliabilityTitle.textContent = "Reliability bins";
      els.reliabilityChart.innerHTML = '<div class="empty">No reliability bins were found.</div>';
      return;
    }
    els.reliabilityTitle.textContent = (row.task || "task") + " - " + modeLabel(row);
    const bins = row.reliability_bins || [];
    els.reliabilityChart.innerHTML = bins.map(function (bin) {
      const confidence = Math.max(0, Math.min(1, finite(bin.confidence) || 0));
      const accuracy = Math.max(0, Math.min(1, finite(bin.accuracy) || 0));
      const label = formatPct(bin.low, 0) + "-" + formatPct(bin.high, 0);
      return '<div class="bin">'
        + '<div class="bin-bars">'
        + '<div class="bin-bar conf" title="confidence ' + escapeHtml(formatPct(confidence)) + '" style="height:' + (confidence * 100).toFixed(2) + '%"></div>'
        + '<div class="bin-bar acc" title="accuracy ' + escapeHtml(formatPct(accuracy)) + '" style="height:' + (accuracy * 100).toFixed(2) + '%"></div>'
        + '</div>'
        + '<span title="' + escapeHtml(formatNumber(bin.count, 0) + " samples") + '">' + escapeHtml(label) + '</span>'
        + '</div>';
    }).join("");
  }

  function renderArtifacts() {
    const modelFiles = state.manifestRows.filter(function (row) { return row.category === "models"; });
    const resultFiles = state.manifestRows.filter(function (row) { return row.category === "results"; });
    els.modelArtifacts.innerHTML = renderArtifactTable(modelFiles);
    els.resultArtifacts.innerHTML = renderArtifactTable(resultFiles);
  }

  function renderArtifactTable(rows) {
    return renderTable(
      ["Path", "Bytes", "SHA-256"],
      rows.map(function (row) {
        return [
          '<a href="' + escapeHtml(hrefFor(row.path)) + '"><code>' + escapeHtml(row.path) + '</code></a>',
          escapeHtml(formatNumber(row.bytes, 0)),
          '<span class="mono hash" title="' + escapeHtml(row.sha256 || "") + '">' + escapeHtml(shortHash(row.sha256)) + '</span>'
        ];
      })
    );
  }

  function renderTable(headers, rows) {
    if (!rows.length) return '<div class="empty">No rows available.</div>';
    return '<table><thead><tr>' + headers.map(function (header) {
      return '<th>' + escapeHtml(header) + '</th>';
    }).join("") + '</tr></thead><tbody>' + rows.map(function (row) {
      return '<tr>' + row.map(function (cell) {
        return '<td>' + cell + '</td>';
      }).join("") + '</tr>';
    }).join("") + '</tbody></table>';
  }

  function bindEvents() {
    els.resultFilter.addEventListener("input", renderResultsTable);
    els.resultRows.addEventListener("click", function (event) {
      const tr = event.target.closest("tr[data-row-index]");
      if (!tr) return;
      const row = state.compareRows[Number(tr.getAttribute("data-row-index"))];
      const index = state.reliabilityRows.indexOf(row);
      if (index >= 0) {
        els.reliabilitySelect.value = String(index);
        renderReliabilityRow(index);
        document.getElementById("calibration").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
    els.reliabilitySelect.addEventListener("change", function () {
      renderReliabilityRow(Number(els.reliabilitySelect.value));
    });
  }

  function renderAll(summaryText) {
    state.compareRows = state.rows.filter(function (row) { return row.schema === "crlp.compare.v1"; });
    state.ruleRows = state.rows.filter(function (row) { return row.schema === "crlp.rule_search.summary.v1"; });
    state.schemaRows = state.rows.filter(function (row) { return row.schema === "crlp.experiment_schema.v1"; });

    if (summaryText && summaryText.trim()) {
      els.summary.textContent = summaryText.trim().replace(/^#.*$/gm, "").replace(/\s+/g, " ").trim();
    }
    renderMetrics();
    renderClaims();
    renderModelCharts();
    renderModelHashTable();
    renderResultsTable();
    renderRuleSearch();
    renderReliabilityControls();
    renderArtifacts();
  }

  function showError(error) {
    console.error(error);
    els.summary.innerHTML = '<span class="error">Could not load the published result artifacts: ' + escapeHtml(error.message || error) + '</span>';
    els.metrics.innerHTML = '<div class="error">Open this report through GitHub Pages or a local web server so the browser can fetch JSONL artifacts.</div>';
  }

  async function main() {
    renderRunMeta();
    bindEvents();
    const results = await Promise.all([
      readText("results/comparison.jsonl"),
      readText("manifest.tsv"),
      readText("results/summary.md", true)
    ]);
    state.rows = parseJsonl(results[0], "results/comparison.jsonl");
    state.manifestRows = parseManifest(results[1]);
    renderAll(results[2] || "");
  }

  main().catch(showError);
}());
</script>
</body>
</html>
HTML
}

write_manifest
write_index

echo "pages site staged at $site_dir"
