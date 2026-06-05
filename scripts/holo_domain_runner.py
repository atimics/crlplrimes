#!/usr/bin/env python3
"""Universal holonet experiment harness for crlp.trace.v1 domains."""
import json, sys, os, subprocess, hashlib

DOMAIN = sys.argv[1]
TRACE = sys.argv[2]
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)

BINARY = os.path.join(os.path.dirname(OUT), "holo_experiments")
if not os.path.exists(BINARY):
    BINARY = "build/float/holo_experiments"

rows = []
with open(TRACE) as f:
    for line in f:
        d = json.loads(line)
        if d.get("schema") != "crlp.trace.v1":
            continue
        sh = d.get("state_hash", "")
        target = d.get("target_action_id", 0)
        scores = d.get("scores", [])
        verifier = d.get("verifier_result", "")
        verified = 1 if verifier in ("correct", "accepted", "corrected", "dataset_target") else 0
        if not scores:
            continue

        # Use raw features for VSA embedding if available
        # For ARC, strip position-dependent features (output_x, output_y, cell_index)
        # to prevent the holo store from becoming a cell-position lookup table
        features_raw = d.get("features", None)
        if features_raw and len(features_raw) > 0:
            features = list(features_raw)
            # ARC features: 0=output_x, 1=output_y, 6=output_cell_index
            # Zero out position-dependent features
            if DOMAIN == "arc" and len(features) >= 7:
                features[0] = 0.0  # output_x
                features[1] = 0.0  # output_y
                features[6] = 0.0  # output_cell_index
        else:
            features = None
        if features and len(features) > 0:
            state_key = "features:" + ",".join(f"{f:.6f}" for f in features)
        else:
            state_key = "hash:" + str(int(hashlib.md5(sh.encode()).hexdigest()[:8], 16))

        parts = [sh, state_key, str(target), str(verified)]
        parts.extend(f"{s:.6f}" for s in scores)
        rows.append("\t".join(parts))

if len(rows) < 20:
    print(f"SKIP {DOMAIN}: only {len(rows)} rows", file=sys.stderr)
    sys.exit(0)

input_path = os.path.join(OUT, f"{DOMAIN}_holo_input.txt")
with open(input_path, "w") as f:
    for r in rows:
        f.write(r + "\n")

n_fields = len(rows[0].split("\t"))
print(f"DOMAIN {DOMAIN}: {len(rows)} rows, {n_fields-4} candidates")

cmd = [BINARY, "--mode", "domain-integration",
       "--domain", DOMAIN,
       "--input", input_path,
       "--output-dir", OUT]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
