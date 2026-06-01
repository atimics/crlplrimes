#!/bin/bash
# Persistent tournament daemon — auto-restarts on crash, saves ELO
PORT=8087
BIN=./build/float/go_tournament
LOG_DIR=build/float/tournament_logs
ELO_FILE="$LOG_DIR/elo_state.txt"
mkdir -p "$LOG_DIR"

# Load ELO if exists
if [ -f "$ELO_FILE" ]; then
    echo "Loading ELO state from $ELO_FILE"
fi

RESTART_COUNT=0
while true; do
    # Kill any stale instance
    PID=$(lsof -ti :$PORT 2>/dev/null)
    [ -n "$PID" ] && kill -9 $PID 2>/dev/null && sleep 1
    
    LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
    echo "[daemon] Starting run #$((RESTART_COUNT+1)) at $(date)" | tee -a "$LOG_DIR/daemon.log"
    
    # Build if needed
    if [ ! -x "$BIN" ] || [ "$BIN" -ot examples/go_tournament.c ]; then
        echo "[daemon] Rebuilding..." | tee -a "$LOG_DIR/daemon.log"
        cc -std=c11 -O3 -Iinclude -DNN_REAL=float -DNN_REAL_IS_FLOAT=1 \
            examples/go_tournament.c \
            build/float/nn.o build/float/nn_experiment.o build/float/nn_symbolic.o \
            build/float/nn_bitgrid.o build/float/nn_tilebot.o build/float/nn_layer.o \
            build/float/nn_program.o build/float/nn_hypothesis.o build/float/nn_protocol.o \
            build/float/nn_json.o build/float/nn_text.o build/float/nn_dataset.o \
            -lm -o "$BIN" || { echo "Build failed"; exit 1; }
    fi
    
    # Run tournament
    "$BIN" --agents "${AGENTS:-16}" --games "${GAMES:-1}" \
          --threads "${THREADS:-8}" --display "${DISPLAY:-4}" \
          2>&1 | tee "$LOG"
    
    EXIT_CODE=$?
    RESTART_COUNT=$((RESTART_COUNT+1))
    
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        echo "[daemon] Clean exit (code $EXIT_CODE)" | tee -a "$LOG_DIR/daemon.log"
        break
    fi
    
    echo "[daemon] Crash (code $EXIT_CODE), restarting in 2s..." | tee -a "$LOG_DIR/daemon.log"
    sleep 2
done
