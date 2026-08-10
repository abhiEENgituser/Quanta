.PHONY: build test shim clean

build:
	go build ./...

test:
	go test ./...

shim:
	cmake -S shim -B shim/build && cmake --build shim/build

clean:
	rm -rf shim/build
