#!/usr/bin/env bash
# bench_single.sh — Phase 1 exit runs: single-stream baselines.
#
# Sweeps prompt length (~32/128/512 tokens) x engine threads (4 = library
# default, 3 = one core reserved for the control plane), 5 recorded repeats
# each. Every (config, repeat) gets a FRESH shim and quantad, because server
# histograms accumulate since process start and runs must not blend.
#
# Standing rules enforced, not remembered:
#   - refuses to run unless the governor is `performance`
#   - engine pinned to cores 0-2 for -t 3 (cores 0-3 for -t 4);
#     quantad + bench pinned to core 3
#   - MHz sampled DURING runs (quanta-bench does this itself)
#   - drift gate printed per run (in the .log)
#
# Usage:  ./bench/configs/bench_single.sh
# Output: bench/results/single/  (per-run CSVs + logs; plot them with
#         bench/plot/plot_single.py)
set -eu

cd "$(dirname "$0")/../.."          # repo root, wherever invoked from

MODEL=models/qwen2.5-0.5b-q4km.gguf
SOCK=/tmp/quanta-bench.sock         # private socket — never fights a dev shim
ADDR=127.0.0.1:8087
OUT=bench/results/single
REPEATS=5
WARM=5s
MAXTOK=32
# prompt-repeat values -> approximate prompt tokens (base prompt ~5-6 tokens/repeat);
# actual token counts land in each CSV row from the server's done event.
REPEAT_VALUES="5 21 85"
THREADS="4 3"

# Rate must scale DOWN as prompts grow, or utilization varies wildly per cell:
# service time at ~425 prompt tokens is ~3.3s, and a fixed 0.5 req/s there is
# rho ~ 1.6 — an unstable, ever-growing queue. That mistake was made once: the
# long-prompt "baselines" measured queue growth, not the engine. Targets below
# hold rho ~ 0.3 everywhere; durations stretch so each run still records >= ~12.
rate_for()  { case "$1" in 5) echo 0.5 ;; 21) echo 0.25 ;; 85) echo 0.08 ;; esac; }
dur_for()   { case "$1" in 5) echo 40s ;; 21) echo 80s  ;; 85) echo 180s ;; esac; }

# ---- pre-flight: conditions are requirements, not suggestions ---------------
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
if [ "$gov" != "performance" ]; then
    echo "REFUSING: governor is '$gov', need performance:" >&2
    echo "    sudo cpupower frequency-set -g performance" >&2
    exit 1
fi
on_ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1 || echo "?")
[ "$on_ac" = "1" ] || { echo "REFUSING: not on AC power" >&2; exit 1; }
swap=$(vmstat 1 2 | tail -1 | awk '{print $7+$8}')
[ "$swap" = "0" ] || { echo "REFUSING: swap active (si+so=$swap)" >&2; exit 1; }
load=$(cut -d' ' -f1 /proc/loadavg)
awk "BEGIN{exit !($load < 0.6)}" || echo "WARNING: load=$load — close the browser" >&2

mkdir -p "$OUT" bench/bin
go build -o bench/bin/quantad ./cmd/quantad
go build -o bench/bin/quanta-bench ./cmd/quanta-bench
cmake --build shim/build -j3 >/dev/null

cleanup() {
    [ -n "${QPID:-}" ] && kill "$QPID" 2>/dev/null || true
    [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

for t in $THREADS; do
    # -t 3 tests the reservation policy: engine on 0-2, control plane on 3.
    # -t 4 is the default-config comparison: engine allowed all cores, control
    # plane still on 3 — the contention that implies is exactly what the
    # comparison exists to price.
    if [ "$t" = "3" ]; then ENGINE_CORES=0-2; else ENGINE_CORES=0-3; fi

    for rep in $REPEAT_VALUES; do
        for run in $(seq 1 $REPEATS); do
            tag="t${t}_rep${rep}_run${run}"

            # A killed shim never runs its unlink() cleanup, so the socket
            # FILE outlives the process. Without this rm, the next readiness
            # check "passes" against the corpse while the new shim is still
            # loading the model — and everything downstream dies quietly.
            rm -f "$SOCK"

            taskset -c "$ENGINE_CORES" ./shim/build/quanta_shim \
                -m "$MODEL" -s "$SOCK" -t "$t" 2>"$OUT/shim_$tag.log" &
            SPID=$!
            for _ in $(seq 1 150); do [ -S "$SOCK" ] && break; sleep 0.1; done
            [ -S "$SOCK" ] || { echo "FATAL $tag: shim never bound (see $OUT/shim_$tag.log)"; exit 1; }

            taskset -c 3 bench/bin/quantad -socket "$SOCK" -listen "$ADDR" \
                2>"$OUT/quantad_$tag.log" &
            QPID=$!
            ready=0
            for _ in $(seq 1 50); do
                curl -sf "http://$ADDR/v1/metrics" >/dev/null 2>&1 && { ready=1; break; }
                kill -0 "$QPID" 2>/dev/null || break   # quantad already died
                sleep 0.1
            done
            [ "$ready" = "1" ] || { echo "FATAL $tag: quantad not ready (see $OUT/quantad_$tag.log)"; exit 1; }

            taskset -c 3 bench/bin/quanta-bench \
                -target "http://$ADDR" -rate "$(rate_for "$rep")" \
                -duration "$(dur_for "$rep")" \
                -warmup "$WARM" -max-tokens "$MAXTOK" -prompt-repeat "$rep" \
                -out "$OUT/$tag.csv" 2>"$OUT/$tag.log" \
                || { echo "FATAL $tag: bench reported no successful requests"; exit 1; }

            kill "$QPID" "$SPID" 2>/dev/null || true
            wait 2>/dev/null || true
            QPID=""; SPID=""

            # recorded AND error-free — counting ',true,' alone once reported
            # "32 recorded" for a run in which every request failed.
            good=$(awk -F, '$7=="true" && $8=="" {n++} END{print n+0}' "$OUT/$tag.csv")
            [ "$good" -ge 10 ] || { echo "FATAL $tag: only $good good rows"; exit 1; }
            echo "done: $tag  ($good good)"
            sleep 3   # cooldown
        done
    done
done

echo
echo "all runs complete -> $OUT"
echo "plot: python3 bench/plot/plot_single.py $OUT"