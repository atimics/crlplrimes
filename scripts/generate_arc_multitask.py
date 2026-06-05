#!/usr/bin/env python3
"""Generate holo-compatible traces from multiple ARC task JSONs.
Uses the same feature encoding as arc_agi's fill_arc_sample."""
import json, os, glob, hashlib, random

import sys
FIXTURE_DIR = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/arc_hypothesis"
OUTPUT = "build/float/holo_results/arc_multitask_input.txt"
MAX_INPUT = 30   # max grid dimension for padding
MAX_OUTPUT = 30

COMMON_FEATURES = 9
INPUT_CELLS = MAX_INPUT * MAX_INPUT
OUTPUT_CELLS = MAX_OUTPUT * MAX_OUTPUT
ACTION_COUNT = 10
FEATURE_COUNT = COMMON_FEATURES + INPUT_CELLS + OUTPUT_CELLS + ACTION_COUNT

def encode_sample(input_grid, output_grid, input_w, input_h, output_w, output_h, 
                  cell, target_color, candidate_color):
    """Encode one candidate action for one output cell. Same as arc_agi fill_arc_sample."""
    output_x = cell % output_w
    output_y = cell // output_w
    row = [0.0] * FEATURE_COUNT
    
    row[0] = output_x / (output_w - 1) if output_w > 1 else 0
    row[1] = output_y / (output_h - 1) if output_h > 1 else 0
    row[2] = input_w / MAX_INPUT
    row[3] = input_h / MAX_INPUT
    row[4] = output_w / MAX_OUTPUT
    row[5] = output_h / MAX_OUTPUT
    row[6] = cell / (output_w * output_h - 1) if output_w * output_h > 1 else 0
    row[7] = 1.0 if candidate_color == 0 else 0.0
    row[8] = candidate_color / 9.0
    
    # Input grid (zero-padded to MAX_INPUT x MAX_INPUT)
    for y in range(input_h):
        for x in range(input_w):
            idx = y * input_w + x
            row[COMMON_FEATURES + y * MAX_INPUT + x] = input_grid[idx] / 9.0
    
    # Output cell one-hot (zero-padded to MAX_OUTPUT x MAX_OUTPUT)
    row[COMMON_FEATURES + INPUT_CELLS + cell] = 1.0
    
    # Action one-hot
    row[COMMON_FEATURES + INPUT_CELLS + OUTPUT_CELLS + candidate_color] = 1.0
    
    return row

def process_task(task_path):
    """Process one ARC task, return list of (state_hash, target, verified, scores, features)."""
    with open(task_path) as f:
        task = json.load(f)
    
    results = []
    task_id = os.path.basename(task_path).replace('.json', '')
    
    for pair_idx, pair in enumerate(task['train']):
        inp = pair['input']
        out = pair['output']
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        
        # Flatten grids
        inp_flat = [inp[y][x] for y in range(ih) for x in range(iw)]
        out_flat = [out[y][x] for y in range(oh) for x in range(ow)]
        
        for cell in range(oh * ow):
            target_color = out_flat[cell]
            
            # Generate features and scores for all 10 colors
            features = []
            scores = []
            for color in range(ACTION_COUNT):
                feat = encode_sample(inp_flat, out_flat, iw, ih, ow, oh, cell, target_color, color)
                features.append(feat)
                # Simulate neural score: correct answer gets boost, others get noise
                score = 0.8 if color == target_color else 0.0
                score += random.uniform(-0.3, 0.3)
                scores.append(score)
            
            state_hash = hashlib.md5(f"{task_id}:{pair_idx}:{cell}".encode()).hexdigest()[:16]
            
            # Format: state_key\tfeatures_as_csv\ttarget\tverified\tscore0\t...
            features_str = ",".join(f"{v:.6f}" for v in features[target_color])
            parts = [state_hash, f"features:{features_str}", str(target_color), "1"]
            parts.extend(f"{s:.6f}" for s in scores)
            results.append("\t".join(parts))
    
    # Also process test pair
    if 'test' in task:
        for pair in task['test']:
            inp = pair['input']
            out = pair.get('output', None)
            ih, iw = len(inp), len(inp[0]) if inp else 0
            oh, ow = len(out), len(out[0]) if out and out else (ih, iw)
            
            inp_flat = [inp[y][x] for y in range(ih) for x in range(iw)]
            out_flat = [out[y][x] for y in range(oh) for x in range(ow)] if out else [0] * (oh * ow)
            
            for cell in range(oh * ow):
                target_color = out_flat[cell] if out else 0
                features = []
                scores = []
                for color in range(ACTION_COUNT):
                    feat = encode_sample(inp_flat, out_flat, iw, ih, ow, oh, cell, target_color, color)
                    features.append(feat)
                    score = 0.8 if color == target_color else 0.0
                    score += random.uniform(-0.3, 0.3)
                    scores.append(score)
                
                state_hash = hashlib.md5(f"{task_id}:test:{cell}".encode()).hexdigest()[:16]
                features_str = ",".join(f"{v:.6f}" for v in features[target_color])
                parts = [state_hash, f"features:{features_str}", str(target_color), str(1 if out else 0)]
                parts.extend(f"{s:.6f}" for s in scores)
                results.append("\t".join(parts))
    
    return results

# Main
all_rows = []
for task_file in sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json"))):
    rows = process_task(task_file)
    all_rows.extend(rows)
    print(f"{os.path.basename(task_file)}: {len(rows)} rows")

with open(OUTPUT, 'w') as f:
    for r in all_rows:
        f.write(r + '\n')
print(f"\nTotal: {len(all_rows)} rows -> {OUTPUT}")
