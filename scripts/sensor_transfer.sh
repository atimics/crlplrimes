#!/bin/sh
set -eu

BIN="${BIN:-./build/float/dataset_action_score}"
OUT_DIR="${OUT_DIR:-build/float/sensor_transfer}"
BUDGET="${BUDGET:-200}"
HIDDEN="${HIDDEN:-16}"
LR="${LR:-0.04}"
SEED="${SEED:-20260531}"
MAX_GROUPS="${MAX_GROUPS:-128}"

mkdir -p "$OUT_DIR"
RESULT="$OUT_DIR/transfer_comparison.jsonl"
: > "$RESULT"

run_experiment() {
    local label="$1" experiment="$2" task="$3" savepath="$4"
    echo "=== $label ==="
    "$BIN" --experiment "$experiment" --task "$task" \
        --epochs "$BUDGET" --hidden "$HIDDEN" --lr "$LR" --seed "$SEED" \
        --max-groups "$MAX_GROUPS" ${5:-} \
        --save "$savepath" 2>/dev/null \
      | sed 's/$/,"baseline":"'"$label"'","transfer_source":"'"${6:-none}"'"}/' \
      >> "$RESULT"
}

# A: music-native
run_experiment "music_native" music_sequence music_seq_A "$OUT_DIR/mA.nnckpt" "" "none"

# Haiku source
run_experiment "haiku_source" haiku_adjacency haiku_src "$OUT_DIR/haiku.nnckpt" "--max-candidates 64" "haiku"

# B: music+random (matched capacity)
run_experiment "music_random_matched" music_sequence music_seq_B "$OUT_DIR/mB.nnckpt" "" "random"

# C: music with haiku structural sensors
run_experiment "music_haiku_structural" music_sequence music_seq_C "$OUT_DIR/mC.nnckpt" "" "haiku"

# D: music-native anchor
run_experiment "music_native_anchor" music_sequence music_seq_D "$OUT_DIR/mD.nnckpt" "" "none"

# E: haiku-derived + native
run_experiment "music_haiku_plus_native" music_sequence music_seq_E "$OUT_DIR/mE.nnckpt" "" "haiku"

# Summary
printf '{"schema":"crlp.sensor_transfer.summary.v1","experiment":"haiku_to_music","source_domain":"haiku","target_domain":"music","source_authority":"corpus_proxy","target_authority":"corpus_proxy","sensor_families":["adjacency","shape","position","repetition","contrast"],"transfer_mode":"feature_mapping","baselines":["music_native","haiku_source","music_random_matched","music_haiku_structural","music_native_anchor","music_haiku_plus_native"],"metrics":["top1_delta","nll_delta"],"authority_note":"Source authority does not transfer. Only measured target-domain performance counts.","status":"experiment_complete"}\n' \
  >> "$RESULT"

echo "Sensor transfer results: $RESULT"
echo "---"
cat "$RESULT"
