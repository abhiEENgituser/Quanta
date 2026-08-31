#!/usr/bin/env bash
# validate.sh — referee run for the cost model: real engine measured at
# held-out prompt lengths, compared against pure model prediction.
# Same enforced conditions and pinning as calibration — both sides of a
# comparison must be taken under the same conditions.
set -eu
cd "$(dirname "$0")/../.."

MODEL=models/qwen2.5-0.5b-q4km.gguf
SOCK=/tmp/quanta-validate.sock

gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
[ "$gov" = "performance" ] || {
    echo "REFUSING: governor is '$gov' — sudo cpupower frequency-set -g performance" >&2
    exit 1
}
on_ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1 || echo "?")
[ "$on_ac" = "1" ] || { echo "REFUSING: not on AC power" >&2; exit 1; }
swap=$(vmstat 1 2 | tail -1 | awk '{print $7+$8}')
[ "$swap" = "0" ] || { echo "REFUSING: swap active" >&2; exit 1; }

mkdir -p bench/bin bench/results
go build -o bench/bin/quanta-validate ./cmd/quanta-validate
cmake --build shim/build -j3 >/dev/null

cleanup() { [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

rm -f "$SOCK"
taskset -c 0-2 ./shim/build/quanta_shim -m "$MODEL" -s "$SOCK" -t 3 2>bench/results/validate_shim.log &
SPID=$!
for _ in $(seq 1 150); do [ -S "$SOCK" ] && break; sleep 0.1; done
[ -S "$SOCK" ] || { echo "FATAL: shim never bound (bench/results/validate_shim.log)" >&2; exit 1; }

taskset -c 3 bench/bin/quanta-validate -socket "$SOCK"