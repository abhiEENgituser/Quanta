#!/usr/bin/env bash
# calibrate.sh — fit the cost model against the real engine at the project's
# standard config: -t 3, engine pinned to cores 0-2, calibrator on core 3.
#
# Conditions are enforced, not remembered: wrong governor, battery, or swap
# refuse to run. The shim is started fresh (stale socket removed first — a
# killed shim never runs its unlink), and the calibrator talks to it directly
# over the Backend client: no quantad, no HTTP, no queue in the measurement.
set -eu
cd "$(dirname "$0")/../.."

MODEL=models/qwen2.5-0.5b-q4km.gguf
SOCK=/tmp/quanta-calibrate.sock

gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
[ "$gov" = "performance" ] || {
    echo "REFUSING: governor is '$gov' — sudo cpupower frequency-set -g performance" >&2
    exit 1
}
on_ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1 || echo "?")
[ "$on_ac" = "1" ] || { echo "REFUSING: not on AC power" >&2; exit 1; }
swap=$(vmstat 1 2 | tail -1 | awk '{print $7+$8}')
[ "$swap" = "0" ] || { echo "REFUSING: swap active" >&2; exit 1; }

mkdir -p bench/bin
go build -o bench/bin/quanta-calibrate ./cmd/quanta-calibrate
cmake --build shim/build -j3 >/dev/null

cleanup() { [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

start_shim() {  # $1 = -q value, $2 = log suffix
    [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null && wait 2>/dev/null || true
    rm -f "$SOCK"
    taskset -c 0-2 ./shim/build/quanta_shim -m "$MODEL" -s "$SOCK" -t 3 -q "$1" \
        2>"bench/results/calibrate_shim_$2.log" &
    SPID=$!
    for _ in $(seq 1 150); do [ -S "$SOCK" ] && break; sleep 0.1; done
    [ -S "$SOCK" ] || { echo "FATAL: shim never bound (calibrate_shim_$2.log)" >&2; exit 1; }
}

# Phase A — the per-length lines. Needs big per-sequence windows (prefill up
# to 512 tokens), so a single sequence slot: -q 1 keeps the full 2048 window.
start_shim 1 lines
taskset -c 3 bench/bin/quanta-calibrate -mode lines \
    -socket "$SOCK" \
    -engine-args "-t 3, engine cores 0-2, calibrator core 3, governor performance"

# Phase B — the batch curve. Needs 6 sequence slots; n_ctx divides, giving
# each sequence a 341-token window — plenty for a 128-token prompt + steps.
start_shim 6 batch
taskset -c 3 bench/bin/quanta-calibrate -mode batch \
    -socket "$SOCK" -batch-max 6 -batch-prompt 128 -thermal-warmup 45s