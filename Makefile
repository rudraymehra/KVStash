# kvblockd — Week-1 Makefile (fuzz/bench become real in Week 2+)

GO      ?= go
PKGS    := ./...
BIN_DIR := bin

.PHONY: all build test race lint fuzz bench clean

all: build

build:
	$(GO) build -o $(BIN_DIR)/ $(PKGS)

test:
	$(GO) test -short -count=1 $(PKGS)

race:
	$(GO) test -race -short -count=1 $(PKGS)

lint:
	$(GO) vet $(PKGS)
	@command -v golangci-lint >/dev/null 2>&1 && golangci-lint run || echo "golangci-lint not installed (CI runs it); skipping"

fuzz:
	@echo "fuzz: no fuzz targets yet (arrives Week 2 with internal/protocol)"

bench:
	@echo "bench: no benchmarks yet (arrives Week 2)"

clean:
	rm -rf $(BIN_DIR) dist
