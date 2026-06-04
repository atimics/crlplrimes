#!/bin/bash
# chunk_merge_train.sh — parallel holographic chunk training with merge
set -e
CHUNKS=${1:-4}
EPOCHS=${2:-4}
CORPUS="data/haiku_open/clean/hf_haiku_clean.txt"
BIN="./build/float/haiku_word_lm"
ARTIFACTS="build/float/haiku_word_lm_artifacts"
HOLO_DIM=1024
echo "=== Chunk-merge: $CHUNKS chunks, $EPOCHS epochs each ==="
TOTAL=$(grep -c "^$" "$CORPUS")
TOTAL=$((TOTAL + 1))
CHUNK_SIZE=$((TOTAL / CHUNKS))
echo "Total poems: $TOTAL, chunk size: $CHUNK_SIZE"
for ((c=0; c<CHUNKS; c++)); do
    START=$((c * CHUNK_SIZE))
    END=$((c == CHUNKS - 1 ? TOTAL : (c + 1) * CHUNK_SIZE))
    echo "  Chunk $c: $START-$END ($((END - START)) poems)"
done
echo "Run chunks manually or use the commands below:"
for ((c=0; c<CHUNKS; c++)); do
    START=$((c * CHUNK_SIZE))
    END=$((c == CHUNKS - 1 ? TOTAL : (c + 1) * CHUNK_SIZE))
    HOLO="${ARTIFACTS}/holo_chunk_${c}.bin"
    CKPT="${ARTIFACTS}/haiku_chunk_${c}.nnckpt"
    echo "$BIN --corpus $CORPUS --online 1 --epochs $EPOCHS --lr 0.004 --max-candidates 128 --holographic 1 --holo-dim $HOLO_DIM --anki 1 --chunk-start $START --chunk-end $END --samples 0 --save-holo $HOLO --save $CKPT"
done
