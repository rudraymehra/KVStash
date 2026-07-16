# kvblockd

GO      ?= go
PKGS    := ./...
BIN_DIR := bin

# Mutation gate: 0.90 floor on the pure codec package (header + body codecs).
# Latest ~0.93; surviving mutants are provably equivalent (copy() capped by dst
# length, panic sentinels that still panic later, 1>>0 == 1<<0).
MUTATE_PKG := ./internal/protocol/
MUTATE_MIN := 0.9

.PHONY: all build test race lint fuzz mutate bench clean

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
	$(GO) test -fuzz=FuzzParseHeader -fuzztime=90s ./internal/protocol/

mutate:
	@command -v go-mutesting >/dev/null 2>&1 || { echo "installing go-mutesting..."; $(GO) install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest; }
	@go-mutesting --exec-timeout 30 $(MUTATE_PKG) | tail -1 | tee /tmp/kvb-msi.txt
	@awk -v min=$(MUTATE_MIN) '{ for (i=1;i<=NF;i++) if ($$i ~ /^[0-9]+\.[0-9]+$$/) { score=$$i; break } } END { if (score+0 < min+0) { printf "FAIL: mutation score %s < %s floor\n", score, min; exit 1 } else printf "OK: mutation score %s >= %s floor\n", score, min }' /tmp/kvb-msi.txt

bench:
	$(GO) test -run='^$$' -bench=. -benchmem $(PKGS)

clean:
	rm -rf $(BIN_DIR) dist
