#!/bin/bash
# Launch the Go tournament. Kills any existing instance first.
# Usage: ./scripts/tournament.sh [--agents N] [--games K] [--threads T] [--display D]

PORT=8087
BIN=./build/float/go_tournament

# Kill existing
PID=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Killing existing tournament (PID $PID)..."
    kill $PID 2>/dev/null
    sleep 1
    kill -9 $PID 2>/dev/null
fi

# Build if needed
if [ ! -x "$BIN" ] || [ "$BIN" -ot examples/go_tournament.c ]; then
    echo "Building..."
    make -j4 go_tournament 2>/dev/null || {
        cc -std=c11 -O3 -Iinclude -DNN_REAL=float -DNN_REAL_IS_FLOAT=1 \
            examples/go_tournament.c \
            build/float/nn.o build/float/nn_experiment.o build/float/nn_symbolic.o \
            build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o \
            build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o \
            build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o \
            -lm -o "$BIN" || { echo "Build failed"; exit 1; }
    }
fi

echo "Starting tournament..."
exec "$BIN" --agents "${AGENTS:-16}" --games "${GAMES:-1}" --threads "${THREADS:-4}" --display "${DISPLAY:-4}" "$@"
