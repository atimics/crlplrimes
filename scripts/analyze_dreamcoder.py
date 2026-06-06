# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anonymous Authors

#!/usr/bin/env python3
"""Parse DREAMCODER stderr output and emit composition frequency analysis."""
import sys, re, collections

def analyze(stderr_path):
    pairs = collections.Counter()
    singletons = collections.Counter()
    total_tasks = 0
    solved_tasks = 0
    
    with open(stderr_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ' task=' in line:
                total_tasks += 1
                if 'exact=1' in line:
                    solved_tasks += 1
                continue
            if 'accepted_rules' not in line:
                continue
            parts = line.split('=', 1)
            if len(parts) < 2:
                continue
            rules = [x.strip() for x in parts[1].split(',')]
            for r in rules:
                if not r:
                    continue
                if r.startswith('compose_'):
                    comp_parts = r.replace('compose_', '').split('_then_')
                    if len(comp_parts) == 2:
                        a, b = sorted(comp_parts)
                        pairs[f"{a} + {b}"] += 1
                    singletons[r] += 1
                elif r.startswith('hypothesis_'):
                    m = re.match(r'hypothesis_(\w+)_', r)
                    if m:
                        singletons[f"hypothesis:{m.group(1)}"] += 1
                else:
                    singletons[r] += 1
    
    print(f"Tasks: {solved_tasks}/{total_tasks} solved ({100*solved_tasks/total_tasks:.1f}%)\n")
    
    print("=== Pair compositions (promotable refactorings) ===")
    for pair, count in pairs.most_common(20):
        pct = 100 * count / solved_tasks if solved_tasks else 0
        print(f"  {count:3d} ({pct:5.1f}%)  {pair}")
    
    print(f"\n=== Singleton transforms ===")
    for name, count in singletons.most_common(30):
        pct = 100 * count / solved_tasks if solved_tasks else 0
        print(f"  {count:3d} ({pct:5.1f}%)  {name}")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dreamcoder_ind_400.txt'
    analyze(path)
