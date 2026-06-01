#!/bin/bash
# Auto-rebuild + restart tournament on source change
# Usage: make watch
PORT=8087
SRC=examples/go_tournament.c
BIN=build/float/go_tournament

mkdir -p build/float/tournament_logs

LAST_HASH=""
while true; do
    HASH=$(md5 -q "$SRC" 2>/dev/null || md5sum "$SRC" 2>/dev/null | cut -d' ' -f1)
    if [ "$HASH" != "$LAST_HASH" ]; then
        echo "[watch] $(date +%H:%M:%S) source changed, rebuilding..."
        # Kill old
        PID=$(lsof -ti :$PORT 2>/dev/null)
        [ -n "$PID" ] && kill $PID 2>/dev/null && sleep 0.5
        
        cc -std=c11 -O3 -Iinclude -DNN_REAL=float -DNN_REAL_IS_FLOAT=1 \
            "$SRC" \
            build/float/nn.o build/float/nn_experiment.o build/float/nn_symbolic.o \
            build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o \
            build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o \
            build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o \
            -lm -o "$BIN" 2>&1 | tail -3
        
        if [ -x "$BIN" ]; then
            echo "[watch] build OK, launching..."
            nohup "$BIN" --agents 16 --games 1 --threads 8 --display 4 \
                > build/float/tournament_logs/run_$(date +%H%M%S).log 2>&1 &
            echo "[watch] http://127.0.0.1:$PORT"
        else
            echo "[watch] BUILD FAILED"
        fi
        LAST_HASH="$HASH"
    fi
    sleep 2
done
