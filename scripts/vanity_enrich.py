#!/usr/bin/env python3
"""
vanity_enrich.py -- Post-process trebuchet grind output with holographic proof.

Usage:
  python3 scripts/vanity_enrich.py \
    --grind-output grind.json \
    --hologram-bin ./build/float/vanity_hologram \
    [--dim 2048] [--sample-rate 4] \
    --out enriched.json
"""

import argparse, json, subprocess, sys

def enrich(grind_json, holo_bin, dim, sample_rate):
    secret_key = grind_json.get("secretKey")
    attempts = grind_json.get("attempts", 0)
    if not secret_key or len(secret_key) != 64:
        raise ValueError("grind output missing valid secretKey (64 bytes)")
    if attempts == 0:
        raise ValueError("grind output missing attempts")
    seed_hex = "".join(f"{b:02x}" for b in secret_key[:32])
    cmd = [holo_bin, "--seed-hex", seed_hex, "--attempts", str(attempts),
           "--dim", str(dim), "--sample-rate", str(sample_rate)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"vanity_hologram failed: {result.stderr}")
    holo_json = json.loads(result.stdout)
    return {"hologram": holo_json.get("hologram", {}),
            "selfCheck": holo_json.get("selfCheck", {})}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grind-output", required=True)
    p.add_argument("--hologram-bin", default="./build/float/vanity_hologram")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--sample-rate", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.grind_output) as f:
        grind = json.load(f)
    meta = enrich(grind, args.hologram_bin, args.dim, args.sample_rate)
    out = dict(grind)
    out["hologram"] = meta["hologram"]
    out["selfCheck"] = meta["selfCheck"]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Enriched: {args.out}  dim={meta['hologram']['dim']} "
          f"positions={meta['hologram']['positions']} "
          f"fidelity={meta['hologram']['fidelity']:.6f}")

if __name__ == "__main__":
    main()
