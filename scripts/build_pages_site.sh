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
    .paper-title {
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
      align-items: end;
    }
    .paper-kicker {
      color: var(--muted);
      font-size: 0.95rem;
      margin: 8px 0 0;
      max-width: 820px;
    }
    .paper-note {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
    }
    .paper-note strong {
      display: block;
      font-size: 0.92rem;
      margin-bottom: 6px;
    }
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
    .argument-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .argument-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .argument-card strong {
      color: var(--accent);
      display: block;
      font-size: 0.82rem;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .argument-card p { margin: 8px 0 0; }
    .comparison-bars {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }
    .comparison-row {
      display: grid;
      gap: 8px;
      grid-template-columns: minmax(120px, 0.7fr) minmax(180px, 1.5fr);
    }
    .paired-bars {
      display: grid;
      gap: 5px;
    }
    .paired-bar {
      align-items: center;
      display: grid;
      gap: 8px;
      grid-template-columns: 84px minmax(120px, 1fr) 64px;
    }
    .paired-bar span {
      color: var(--muted);
      font-size: 0.82rem;
    }
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
      .section-head, .two-col, .paper-title, .argument-grid, .comparison-row { grid-template-columns: 1fr; }
      .tools { justify-content: flex-start; }
      .bar-row { grid-template-columns: 1fr; }
      .paired-bar { grid-template-columns: 72px minmax(120px, 1fr) 58px; }
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
    <div class="paper-title">
      <div>
        <div class="eyebrow">Live paper generated from the evidence workflow</div>
        <h1>CRLPLRIMES: certified rule learning for verifier-grounded action ranking</h1>
        <p class="lede">The paper argument, figures, model results, and reproducibility ledger are all rendered from the current JSONL evidence and checkpoint manifest.</p>
        <p class="paper-kicker">This is not a static screenshot of results. It is the published paper surface reading <code>comparison.jsonl</code>, <code>manifest.tsv</code>, calibration bins, rule-search summaries, and model hashes directly from GitHub Pages.</p>
      </div>
      <aside class="paper-note">
        <strong>How to read this page</strong>
        <p>Start with the abstract and main figures. Use the claim ledger to jump from each public claim to the exact rows and artifacts supporting it.</p>
      </aside>
    </div>
    <div class="run-meta" id="run-meta"></div>
    <nav aria-label="Report sections">
      <a href="#abstract">Abstract</a>
      <a href="#argument">Argument</a>
      <a href="#figures">Figures</a>
      <a href="#ledger">Claim ledger</a>
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

  <section id="argument">
    <div class="section-head">
      <div>
        <h2>Argument</h2>
        <p>The live paper follows the same spine as the experiment: declare the task and leakage policy, certify rules against a verifier, then measure model ranking, saved verifier work, false rejections, and calibration.</p>
      </div>
    </div>
    <div class="argument-grid">
      <article class="argument-card">
        <strong>Question</strong>
        <p>Can a learned rule proposer discover certified gates that shrink verifier work while preserving the correct action?</p>
      </article>
      <article class="argument-card">
        <strong>Mechanism</strong>
        <p>Every task exposes candidates, verifier feedback, and rule grammars. Accepted rules become certificates; the residual neural scorer ranks what survives.</p>
      </article>
      <article class="argument-card">
        <strong>Evidence</strong>
        <p>The figures below are built from comparison rows, rule-search summaries, reliability bins, and published checkpoint hashes.</p>
      </article>
    </div>
  </section>

  <section id="figures">
    <div class="section-head">
      <div>
        <h2>Main Figures</h2>
        <p>Paper-style figures emphasize the central deltas before exposing the full tables. Values are recomputed in the browser from the published evidence files.</p>
      </div>
    </div>
    <div class="grid wide">
      <figure class="panel figure">
        <div>
          <div class="figure-label">Figure 1</div>
          <h3>Certified gates change the ranking problem</h3>
        </div>
        <div id="figure-gate" class="comparison-bars"></div>
        <figcaption class="caption" id="figure-gate-caption">Loading gate comparison...</figcaption>
      </figure>
      <figure class="panel figure">
        <div>
          <div class="figure-label">Figure 2</div>
          <h3>Rule search reduces exact verifier work</h3>
        </div>
        <div id="figure-work" class="bar-list"></div>
        <figcaption class="caption" id="figure-work-caption">Loading rule-search leaders...</figcaption>
      </figure>
      <figure class="panel figure">
        <div>
          <div class="figure-label">Figure 3</div>
          <h3>Calibration remains inspectable</h3>
        </div>
        <div id="figure-calibration" class="bar-list"></div>
        <figcaption class="caption" id="figure-calibration-caption">Loading calibration rows...</figcaption>
      </figure>
    </div>
  </section>

  <section id="ledger">
    <div class="section-head">
      <div>
        <h2>Claim Ledger</h2>
        <p>Each row links a paper claim to the live evidence class that supports it. The raw files remain available, but the ledger is the reader-facing map.</p>
      </div>
    </div>
    <div class="table-wrap" id="claim-ledger"></div>
  </section>

  <section id="claims">
    <div class="section-head">
      <div>
        <h2>Task Declarations</h2>
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
    figureGate: document.getElementById("figure-gate"),
    figureGateCaption: document.getElementById("figure-gate-caption"),
    figureWork: document.getElementById("figure-work"),
    figureWorkCaption: document.getElementById("figure-work-caption"),
    figureCalibration: document.getElementById("figure-calibration"),
    figureCalibrationCaption: document.getElementById("figure-calibration-caption"),
    claimLedger: document.getElementById("claim-ledger"),
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

  function isRawRow(row) {
    return String(row.evaluation_mode || "") === "raw";
  }

  function isCertifiedOrGatedRow(row) {
    const mode = String(row.evaluation_mode || "");
    return mode.indexOf("certified") >= 0
      || mode.indexOf("expanded") >= 0
      || finite(row.accepted_rules) > 0
      || (row.rule_set_hash && row.rule_set_hash !== "unknown" && row.rule_set_hash !== "hand-wheel");
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
      "CRLPLRIMES studies verifier-grounded action ranking: enumerate candidate actions, certify rules that can reject impossible candidates, and train residual scorers on the surviving set. The current evidence bundle covers "
      + tasks.size + " declared task families, " + state.compareRows.length + " comparison rows, and " + state.ruleRows.length
      + " rule-search summaries. The strongest published rule-search row saves " + (bestSaved === -Infinity ? "n/a" : formatPctPoints(bestSaved))
      + " of exact verifier work; the tables below keep every row, checkpoint hash, calibration bin, and raw artifact inspectable.";

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

  function renderMainFigures() {
    renderGateFigure();
    renderWorkFigure();
    renderCalibrationFigure();
  }

  function renderGateFigure() {
    const rawByTask = new Map();
    const gatedByTask = new Map();
    state.compareRows.forEach(function (row) {
      if (!row.task) return;
      const acc = weightedAccuracy(row);
      if (acc == null) return;
      if (isRawRow(row)) {
        const current = rawByTask.get(row.task);
        if (!current || acc > current.acc) rawByTask.set(row.task, { row: row, acc: acc });
      } else if (isCertifiedOrGatedRow(row)) {
        const current = gatedByTask.get(row.task);
        if (!current || acc > current.acc || (acc === current.acc && rowScore(row) > rowScore(current.row))) {
          gatedByTask.set(row.task, { row: row, acc: acc });
        }
      }
    });
    const rows = Array.from(rawByTask.keys()).filter(function (task) {
      return gatedByTask.has(task);
    }).map(function (task) {
      return {
        task: task,
        raw: rawByTask.get(task),
        gated: gatedByTask.get(task)
      };
    }).sort(function (a, b) {
      return (b.gated.acc - b.raw.acc) - (a.gated.acc - a.raw.acc);
    }).slice(0, 7);

    if (!rows.length) {
      els.figureGate.innerHTML = '<div class="empty">No raw/gated task pairs were found.</div>';
      els.figureGateCaption.textContent = "Figure 1 has no paired evidence in the current comparison file.";
      return;
    }

    els.figureGate.innerHTML = rows.map(function (item) {
      return '<div class="comparison-row">'
        + '<div class="bar-label"><strong>' + escapeHtml(item.task) + '</strong><small>' + escapeHtml(modeLabel(item.gated.row)) + '</small></div>'
        + '<div class="paired-bars">'
        + pairedBar("Raw", item.raw.acc, "fill")
        + pairedBar("Gated", item.gated.acc, "fill orange")
        + '</div>'
        + '</div>';
    }).join("");

    const gains = rows.map(function (item) { return item.gated.acc - item.raw.acc; });
    const bestGain = Math.max.apply(null, gains);
    els.figureGateCaption.textContent =
      "Figure 1. For each task with both raw and gated rows, the chart compares raw accuracy with the strongest certified or rule-gated row. Largest displayed gain: "
      + (bestGain * 100).toFixed(1) + " percentage points.";
  }

  function pairedBar(label, value, klass) {
    const width = Math.max(1, Math.min(100, value * 100));
    return '<div class="paired-bar">'
      + '<span>' + escapeHtml(label) + '</span>'
      + '<div class="track"><div class="' + klass + '" style="width:' + width.toFixed(2) + '%"></div></div>'
      + '<div class="bar-value">' + escapeHtml(formatPct(value)) + '</div>'
      + '</div>';
  }

  function renderWorkFigure() {
    const leaders = state.ruleRows.slice().filter(function (row) {
      return workSavedPct(row) != null;
    }).sort(function (a, b) {
      return (workSavedPct(b) || -1) - (workSavedPct(a) || -1);
    }).slice(0, 7).map(function (row) {
      return {
        label: row.task || "task",
        sub: String(row.selection_strategy || "strategy").replace(/_/g, " ") + " / " + (row.accepted_rules == null ? "n/a" : row.accepted_rules) + " rules",
        value: workSavedPct(row),
        display: formatPctPoints(workSavedPct(row)),
        kind: "orange"
      };
    });
    renderBarList(els.figureWork, leaders, { max: Math.max(100, leaders.reduce(function (max, row) { return Math.max(max, row.value); }, 0)) });
    els.figureWorkCaption.textContent =
      "Figure 2. Rule-search leaders ranked by exact verifier work saved, with accepted-rule counts shown in each row label.";
  }

  function renderCalibrationFigure() {
    const rows = state.compareRows.filter(function (row) {
      return finite(row.ece) != null && Array.isArray(row.reliability_bins) && row.reliability_bins.length;
    }).sort(function (a, b) {
      return (finite(a.ece) || Infinity) - (finite(b.ece) || Infinity);
    }).slice(0, 7).map(function (row) {
      return {
        label: row.task || "task",
        sub: modeLabel(row),
        value: finite(row.ece),
        display: formatPct(row.ece),
        kind: "warn"
      };
    });
    renderBarList(els.figureCalibration, rows, { max: Math.max(0.12, rows.reduce(function (max, row) { return Math.max(max, row.value || 0); }, 0)) });
    els.figureCalibrationCaption.textContent =
      "Figure 3. Lowest expected calibration error rows with reliability bins available. Smaller bars are better; the interactive calibration panel expands the bins.";
  }

  function renderClaimLedger() {
    const bestSaved = state.ruleRows.concat(state.compareRows).reduce(function (best, row) {
      const saved = workSavedPct(row);
      return saved == null ? best : Math.max(best, saved);
    }, -Infinity);
    const zeroFalseRejectRows = state.compareRows.filter(function (row) {
      return finite(row.false_rejections) === 0;
    }).length;
    const reliabilityRows = state.compareRows.filter(function (row) {
      return Array.isArray(row.reliability_bins) && row.reliability_bins.length;
    }).length;
    const modelFiles = state.manifestRows.filter(function (row) { return row.category === "models"; }).length;
    const resultFiles = state.manifestRows.filter(function (row) { return row.category === "results"; }).length;

    els.claimLedger.innerHTML = renderTable(
      ["Claim", "Live evidence", "Primary artifact", "Status"],
      [
        [
          escapeHtml("Task scope and leakage policy are explicit."),
          escapeHtml(state.schemaRows.length + " crlp.experiment_schema.v1 rows declare task roles, claim scope, features, and leakage policy."),
          '<a href="results/comparison.jsonl"><code>comparison.jsonl</code></a>',
          '<span class="ledger-status live">live row-backed</span>'
        ],
        [
          escapeHtml("Certified gates can shrink exact verification."),
          escapeHtml("Best observed saved work is " + (bestSaved === -Infinity ? "n/a" : formatPctPoints(bestSaved)) + " across rule-search and comparison rows."),
          '<a href="#rule-search"><code>rule_search.summary</code></a>',
          '<span class="ledger-status check">quantified</span>'
        ],
        [
          escapeHtml("False-rejection behavior is audited, not hidden."),
          escapeHtml(zeroFalseRejectRows + " comparison rows explicitly report false_rejections = 0; nonzero rows remain visible in the full table."),
          '<a href="#model-results"><code>crlp.compare.v1</code></a>',
          '<span class="ledger-status live">inspectable</span>'
        ],
        [
          escapeHtml("Calibration is part of the evidence surface."),
          escapeHtml(reliabilityRows + " comparison rows include reliability bins for confidence versus empirical correctness."),
          '<a href="#calibration"><code>reliability_bins</code></a>',
          '<span class="ledger-status live">interactive</span>'
        ],
        [
          escapeHtml("Models and certified rules are published artifacts."),
          escapeHtml(modelFiles + " checkpoint or rule files are published with byte counts and SHA-256 hashes."),
          '<a href="manifest.tsv"><code>manifest.tsv</code></a>',
          '<span class="ledger-status check">reproducible</span>'
        ],
        [
          escapeHtml("Raw evidence stays available for independent checks."),
          escapeHtml(resultFiles + " result files are mirrored under results/ with commands and canonical comparison output."),
          '<a href="#artifacts"><code>results/</code></a>',
          '<span class="ledger-status live">downloadable</span>'
        ]
      ]
    );
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
    renderMainFigures();
    renderClaimLedger();
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
