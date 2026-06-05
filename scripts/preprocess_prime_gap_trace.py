#!/usr/bin/env python3
"""Preprocess prime gap trace for holo integration."""
import json, sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <trace.jsonl> <output.txt>", file=sys.stderr)
    sys.exit(1)

rows = []
with open(sys.argv[1]) as f:
    for line in f:
        rows.append(json.loads(line))

with open(sys.argv[2], 'w') as f:
    for r in rows:
        p_val = r.get('p', 0)
        sh = r['state_hash']
        target = r['target_action_id']
        scores = r['scores']
        verifier = 1 if r.get('verifier_result','') in ('accepted','corrected') else 0
        state_id = p_val  # use prime number as state id
        f.write(f"{sh}\t{state_id}\t{target}\t{verifier}")
        for s in scores:
            f.write(f"\t{s:.6f}")
        f.write("\n")

print(f"Wrote {len(rows)} rows")
