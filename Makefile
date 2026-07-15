# kvblockd — Week-1 Makefile (fuzz becomes real in Week 2+)

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
	@if command -v golangci-lint >/dev/null 2>&1; then golangci-lint run; else echo "golangci-lint not installed (CI runs it); skipping"; fi

fuzz:
	@echo "fuzz: no fuzz targets yet (arrives Week 2 with internal/protocol)"

bench:
	$(GO) test -run='^$$' -bench=. -benchmem $(PKGS)

clean:
	rm -rf $(BIN_DIR) dist
