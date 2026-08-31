.PHONY: build test shim clean lint-clock bench-single calibrate validate

# Referee for the cost model: real engine at HELD-OUT prompt lengths vs pure
# prediction. Exits nonzero if worst end-to-end error exceeds 10%. ~4 minutes.
validate:
	systemd-inhibit --what=sleep:idle --why="quanta validation" \
	  ./bench/configs/validate.sh

# Fit the cost model against the real engine at the standard config (-t 3,
# pinned). Writes internal/engine/costmodel/params.json — an inspectable
# artifact, committed, with fit diagnostics embedded. ~4 minutes.
calibrate:
	systemd-inhibit --what=sleep:idle --why="quanta calibration" \
	  ./bench/configs/calibrate.sh

# Phase 1 exit runs: single-stream baselines across prompt length x threads.
# Refuses to run under bad conditions (governor, AC, swap). ~30 minutes.
# systemd-inhibit blocks idle-suspend for the session's duration: a locked
# screen once suspended the machine mid-run, freezing a run invisibly — the
# monotonic clock stops during suspend, so the data looks plausible instead
# of broken, which is the worst kind of wrong.
bench-single:
	systemd-inhibit --what=sleep:idle --why="quanta bench run" \
	  ./bench/configs/bench_single.sh
	python3 bench/plot/plot_single.py bench/results/single

build:
	go build ./...

test: lint-clock
	go test -race ./...

# No direct time use below cmd/. One stray time.Now() in the request path
# silently breaks the Phase 2 virtual-clock simulator — code must receive
# timestamps (or, later, a Clock) from its caller instead.
lint-clock:
	@! grep -rn "time\.Now()\|time\.Sleep(\|time\.After(" internal/ \
	  --include="*.go" | grep -v "internal/clock/" | grep -v "_test.go" \
	  || (echo "direct time use outside internal/clock:"; \
	      grep -rn "time\.Now()\|time\.Sleep(\|time\.After(" internal/ \
	        --include="*.go" | grep -v "internal/clock/" | grep -v "_test.go"; exit 1)

shim:
	cmake -S shim -B shim/build && cmake --build shim/build

clean:
	rm -rf shim/build
