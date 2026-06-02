#!/bin/bash
# Auto-rebuild + restart tournament on source change
# Usage: ./scripts/watch_tournament.sh
PORT=8087
SRC=examples/go_tournament.c
BIN=build/float/go_tournament

mkdir -p build/float/tournament_logs

LAST_HASH=""
while true; do
    HASH=$(cat "$SRC" include/nn.h include/nn_core.h include/nn_layer.h include/nn_dataset.h include/nn_experiment.h include/go_rules.h 2>/dev/null | md5 2>/dev/null || md5 -q "$SRC" 2>/dev/null)
    if [ "$HASH" != "$LAST_HASH" ]; then
        echo "[watch] $(date +%H:%M:%S) source changed, rebuilding..."
        # Kill all tournament processes
        pkill -f "go_tournament --agents" 2>/dev/null || true
        sleep 2
        
        # Rebuild all object files (headers may have changed too)
        make -j4 build/float/nn.o build/float/nn_experiment.o build/float/nn_symbolic.o \
            build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o \
            build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o \
            build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o 2>&1 | tail -3
        cc -std=c11 -O3 -Iinclude -DNN_REAL=float -DNN_REAL_IS_FLOAT=1 \
            "$SRC" \
            build/float/nn.o build/float/nn_experiment.o build/float/nn_symbolic.o \
            build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o \
            build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o \
            build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o \
            -lm -o "${BIN}.new" 2>&1 | tail -3
        BUILD_OK=${PIPESTATUS[0]}
        
        if [ "$BUILD_OK" -eq 0 ] && [ -x "${BIN}.new" ]; then
            mv "${BIN}.new" "$BIN"
            # Wipe old checkpoints (new architecture may be incompatible)
            rm -f build/float/tournament_Alpha*.nnckpt build/float/tournament_elo.txt
            echo "[watch] build OK, launching..."
            nohup "$BIN" --agents 16 --games 1 --threads 8 --display 4 \
                > build/float/tournament_logs/run_$(date +%H%M%S).log 2>&1 &
            echo "[watch] http://127.0.0.1:$PORT"
        else
            echo "[watch] BUILD FAILED (exit=$BUILD_OK)"
            rm -f "${BIN}.new"
        fi
        LAST_HASH="$HASH"
    fi
    sleep 2
done
