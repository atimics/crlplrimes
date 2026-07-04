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

    {
        cat <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRLPLRIMES Results and Models</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f7f4;
      --text: #1c1f24;
      --muted: #5f6977;
      --line: #cdd3dc;
      --panel: #ffffff;
      --accent: #276a82;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111317;
        --text: #eef1f4;
        --muted: #a7b0bb;
        --line: #2f3742;
        --panel: #171b21;
        --accent: #73bfd2;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 40px auto;
    }
    h1, h2 { line-height: 1.15; margin: 0; }
    h1 { font-size: clamp(2rem, 4vw, 3.5rem); max-width: 900px; }
    h2 { font-size: 1.2rem; margin-top: 32px; }
    p { color: var(--muted); max-width: 760px; }
    a { color: var(--accent); }
    .meta {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      margin: 28px 0;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px 16px;
    }
    .metric strong { display: block; font-size: 1.45rem; }
    .metric span { color: var(--muted); }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 22px 0 8px;
    }
    .links a {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      text-decoration: none;
      background: var(--panel);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th { font-size: 0.82rem; color: var(--muted); text-transform: uppercase; }
    td code { word-break: break-all; }
    tr:last-child td { border-bottom: 0; }
    .hash { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
<main>
  <h1>CRLPLRIMES Results and Models</h1>
  <p>Published output from the promoted evidence workflow. The result files mirror <code>results/latest</code>; model files are generated checkpoints and certified rule checkpoints from <code>build/${real}</code>.</p>
  <div class="meta">
    <div class="metric"><strong>${result_count}</strong><span>result files</span></div>
    <div class="metric"><strong>${model_count}</strong><span>model files</span></div>
    <div class="metric"><strong>${real}</strong><span>scalar type</span></div>
    <div class="metric"><strong>${run_id}</strong><span>GitHub run</span></div>
  </div>
  <p>Generated at <code>${generated_utc}</code> from commit <code>${sha}</code>.</p>
  <div class="links">
    <a href="results/summary.md">summary.md</a>
    <a href="results/comparison.jsonl">comparison.jsonl</a>
    <a href="results/comparison.canonical.jsonl">comparison.canonical.jsonl</a>
    <a href="results/commands.txt">commands.txt</a>
    <a href="manifest.tsv">manifest.tsv</a>
  </div>
HTML

        for category in models results; do
            printf '  <h2>%s</h2>\n' "$(printf '%s' "$category" | tr '[:lower:]' '[:upper:]')"
            printf '  <table>\n'
            printf '    <thead><tr><th>Path</th><th>Bytes</th><th>SHA-256</th></tr></thead>\n'
            printf '    <tbody>\n'
            awk -F '\t' -v category="$category" 'NR > 1 && $1 == category { print $2 "\t" $3 "\t" $4 }' "$site_dir/manifest.tsv" | while IFS='	' read -r path bytes sha; do
                escaped_path="$(printf '%s' "$path" | html_escape)"
                short_sha="$(printf '%s' "$sha" | cut -c 1-16)"
                printf '      <tr><td><a href="%s"><code>%s</code></a></td><td>%s</td><td class="hash" title="%s">%s</td></tr>\n' \
                    "$escaped_path" "$escaped_path" "$bytes" "$sha" "$short_sha"
            done
            printf '    </tbody>\n'
            printf '  </table>\n'
        done

        cat <<HTML
</main>
</body>
</html>
HTML
    } > "$site_dir/index.html"
}

write_manifest
write_index

echo "pages site staged at $site_dir"
