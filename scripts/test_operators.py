# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anonymous Authors

#!/usr/bin/env python3
"""Test ARC operators against hypothesis fixtures."""
import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_operators import gen_cands, certify, apply_to_test, load_task

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures', 'arc_hypothesis')

def test_fixtures():
    fixture_files = sorted(glob.glob(os.path.join(FIXTURE_DIR, '*.json')))
    if not fixture_files:
        print("No fixture files found in", FIXTURE_DIR)
        return
    solved, unsolved = [], []
    for fpath in fixture_files:
        name = os.path.basename(fpath).replace('.json', '')
        task = load_task(fpath)
        cands = gen_cands(task)
        best = None
        for cand_name, fn in cands:
            passed, _, _, _ = certify(task, fn)
            if passed:
                test_preds = apply_to_test(task, fn)
                all_ok = True
                for pi, pred in enumerate(test_preds):
                    if pred is None:
                        all_ok = False; break
                    if pi < len(task.get('test', [])):
                        expected = task['test'][pi].get('output')
                        if expected is not None and pred != expected:
                            all_ok = False; break
                if all_ok:
                    best = cand_name; break
        if best:
            print(f"  SOLVED {name:30s} by {best}")
            solved.append((name, best))
        else:
            print(f"  MISSED {name:30s}")
            unsolved.append(name)
    print(f"\nSummary: {len(solved)}/{len(fixture_files)} solved")
    if unsolved:
        print(f"Unsolved: {', '.join(unsolved)}")

if __name__ == '__main__':
    test_fixtures()
