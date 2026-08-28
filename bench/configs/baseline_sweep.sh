#!/usr/bin/env bash
# baseline_sweep.sh — decode@100 with repeats, and prefill vs prompt length.
#
# Standing rules applied here: quiet machine, repeats with cooldown, run 1 of
# each series discarded at analysis time, MHz sampled DURING runs (not after),
# swap checked. Results go to stdout as CSV; redirect into bench/results/.
#
# Usage:
#   ./bench/configs/baseline_sweep.sh > bench/results/baseline_sweep_$(date +%Y%m%d).csv
set -u

PROBE=shim/build/probe
REPEATS=6          # first run of each series is warmup — discard in analysis
COOLDOWN=2

if [ ! -x "$PROBE" ]; then
    echo "error: $PROBE not built" >&2
    exit 1
fi

# --- pre-flight: record the conditions the numbers were taken under ---------
echo "# governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" >&2
echo "# on_ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" >&2
echo "# loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >&2
swap=$(vmstat 1 2 | tail -1 | awk '{print $7+$8}')
echo "# swap_activity=$swap (must be 0)" >&2

echo "kind,prompt_tokens,decode_steps,run,prefill_ms,decode_ms,mhz_mean,mhz_min"

run_probe() {  # $1=kind $2=prompt_len(-l, 0=natural) $3=steps $4=run_idx
    local mhzfile
    mhzfile=$(mktemp)
    ( while :; do
        awk '/cpu MHz/{s+=$4;n++} END{printf "%.0f\n",s/n}' /proc/cpuinfo
        sleep 0.05
      done ) > "$mhzfile" 2>/dev/null &
    local sampler=$!

    local args=(-n "$3")
    [ "$2" -gt 0 ] && args+=(-l "$2")

    local out
    out=$("$PROBE" "${args[@]}" 2>&1 >/dev/null)

    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null

    local prefill decode
    prefill=$(echo "$out" | awk '/^prefill:/{print $2}')
    decode=$(echo "$out" | awk '/^decode:/{print $2}')
    local mhz_mean mhz_min
    mhz_mean=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}' "$mhzfile")
    mhz_min=$(awk 'NR==1{m=$1} $1<m{m=$1} END{print m}' "$mhzfile")
    rm -f "$mhzfile"

    echo "$1,$2,$3,$4,${prefill:-NA},${decode:-NA},${mhz_mean:-NA},${mhz_min:-NA}"
    sleep "$COOLDOWN"
}

# --- series 1: decode at 100 steps, natural (5-token) prompt ----------------
for r in $(seq 1 $REPEATS); do
    run_probe decode100 0 100 "$r"
done

# --- series 2: prefill vs prompt length, minimal decode ---------------------
for len in 16 64 128 256 512; do
    for r in $(seq 1 $REPEATS); do
        run_probe prefill "$len" 1 "$r"
    done
done
