.PHONY: build test shim clean lint-clock

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
