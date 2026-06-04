#!/bin/bash
# chunk_train.sh — train haiku model in chunks with holographic memory
set -e
CORPUS="data/haiku_open/clean/hf_haiku_clean.txt"
BIN="./build/float/haiku_word_lm"
ARTIFACTS="build/float/haiku_word_lm_artifacts"
CHUNKS=${1:-10}
EPOCHS=${2:-3}

echo "=== Chunk training: $CHUNKS chunks, $EPOCHS epochs each ==="
# Count poems
TOTAL=$(grep -c "^$" "$CORPUS")
TOTAL=$((TOTAL + 1))
CHUNK_SIZE=$((TOTAL / CHUNKS))
echo "Total poems: $TOTAL, chunk size: $CHUNK_SIZE"

for ((c=0; c<CHUNKS; c++)); do
    START=$((c * CHUNK_SIZE))
    END=$((c == CHUNKS - 1 ? TOTAL : (c + 1) * CHUNK_SIZE))
    HOLO="${ARTIFACTS}/chunk_${c}.bin"
    CKPT="${ARTIFACTS}/chunk_${c}.nnckpt"
    echo "=== Chunk $c/$CHUNKS: poems $START-$END ($((END - START)) poems) ==="
    $BIN --corpus "$CORPUS" \
         --online 1 --epochs "$EPOCHS" --lr 0.004 --max-candidates 128 \
         --holo-gate 1 --anki 0 \
         --chunk-start "$START" --chunk-end "$END" \
         --samples 2 --save-holo "$HOLO" --save "$CKPT"
    echo ""
done

echo "=== All chunks complete ==="
echo "Traces: ${ARTIFACTS}/chunk_*.bin"
