#!/bin/bash
set -euo pipefail
# Auto-rebuild + restart tournament on source change
# Usage: make watch
PORT=8087
SRC=examples/go_tournament.c
BIN=build/float/go_tournament
OBJS="build/float/nn.o build/float/nn_experiment.o build/float/nn_symbolic.o build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o"

mkdir -p build/float/tournament_logs

LAST_HASH=""
while true; do
    HASH=$(cat "$SRC" include/nn.h include/nn_core.h include/nn_layer.h include/nn_dataset.h include/nn_experiment.h include/go_rules.h 2>/dev/null | md5 2>/dev/null || md5 -q "$SRC" 2>/dev/null)
    if [ "$HASH" != "$LAST_HASH" ]; then
        echo "[watch] $(date +%H:%M:%S) source changed, rebuilding..."
        pkill -f "go_tournament --agents" 2>/dev/null || true
        sleep 3
        # Ensure old processes are fully gone
        pkill -9 -f "go_tournament --agents" 2>/dev/null || true
        sleep 1
        
        # Rebuild object files (headers may have changed)
        make -j4 $OBJS 2>&1 | tail -5
        # Compile tournament
        cc -std=c11 -O3 -Iinclude -DNN_REAL=float -DNN_REAL_IS_FLOAT=1 \
            "$SRC" $OBJS -lm -o "${BIN}.new" 2>&1
        BUILD_OK=$?
        
        if [ "$BUILD_OK" -eq 0 ] && [ -x "${BIN}.new" ]; then
            # Wipe old checkpoints before launching (new architecture may differ)
            rm -f build/float/tournament_Alpha*.nnckpt build/float/tournament_elo.txt build/float/tournament_Alpha*.nnckpt.tmp
            mv "${BIN}.new" "$BIN"
            sync
            sleep 0.5
            echo "[watch] build OK, launching..."
            nohup "$BIN" --agents 16 --games 1 --threads 8 --display 4 \
                > "build/float/tournament_logs/run_$(date +%H%M%S).log" 2>&1 &
            echo "[watch] http://127.0.0.1:$PORT"
        else
            echo "[watch] BUILD FAILED (exit=$BUILD_OK)"
            rm -f "${BIN}.new"
        fi
        LAST_HASH="$HASH"
    fi
    sleep 2
done
