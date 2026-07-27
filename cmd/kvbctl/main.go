// Command kvbctl is the kvblockd CLI. Data plane: exists/get/put/delete/stats
// against a live daemon over pkg/client. Ops plane (cmd_ops.go): namespace
// add/list and quota set over the daemon's loopback admin socket
// (admin_addr). Stdlib flag only — no cobra (dependency count is a feature).
//
// Key convention: block keys are [32]byte. With -hex a key argument must be
// exactly 64 hex chars; otherwise the argument is hashed with SHA-256 — a
// smoke-tool convenience so `kvbctl put k1 blob.bin` works out of the box.
// (Adapters derive real wire keys via the BLAKE3 prefix chain —
// client.WireKey; pass those with -hex.)
//
// Exit codes: 0 success · 1 not-found or per-key/status failure · 2 usage,
// connect, or I/O error.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/kvstash/kvblockd/pkg/client"
)

// version is stamped by the release build (-ldflags "-X main.version=…");
// "dev" means a non-release build.
var version = "dev"

// common holds the flags every subcommand shares.
type common struct {
	addr      string
	token     string
	tokenFile string
	ns        string
	timeout   time.Duration
	hexKeys   bool
}

// register wires the shared flags into a subcommand's FlagSet.
func (c *common) register(fs *flag.FlagSet) {
	fs.StringVar(&c.addr, "addr", "127.0.0.1:9440", "daemon address")
	fs.StringVar(&c.token, "token", "", "bearer token (VISIBLE in ps/shell history — smoke use only; prefer -token-file or KVBCTL_TOKEN)")
	fs.StringVar(&c.tokenFile, "token-file", "", "file holding the bearer token (keeps the credential out of argv; matches the admin plane)")
	fs.StringVar(&c.ns, "ns", "", "namespace")
	fs.DurationVar(&c.timeout, "timeout", 10*time.Second, "dial/request timeout")
	fs.BoolVar(&c.hexKeys, "hex", false, "key args are 64-char hex [32]byte literals (default: sha256 of the arg)")
}

// resolveToken picks the credential by precedence flag > file > KVBCTL_TOKEN.
// The token grants full namespace data access, so it must never be forced
// through argv (ps output, shell history, process accounting all see argv) —
// -token stays only as the documented smoke-use escape hatch.
func (c *common) resolveToken() (string, error) {
	if c.token != "" {
		return c.token, nil
	}
	if c.tokenFile != "" {
		b, err := os.ReadFile(c.tokenFile) //nolint:gosec // G304: operator-supplied path (same posture as the admin plane's -token-file)
		if err != nil {
			return "", fmt.Errorf("-token-file: %w", err)
		}
		return strings.TrimSpace(string(b)), nil
	}
	return os.Getenv("KVBCTL_TOKEN"), nil
}

// key derives the [32]byte block key from a CLI argument.
func (c *common) key(arg string) ([32]byte, error) {
	var k [32]byte
	if c.hexKeys {
		b, err := hex.DecodeString(arg)
		if err != nil || len(b) != 32 {
			return k, fmt.Errorf("-hex key %q: want exactly 64 hex chars", arg)
		}
		copy(k[:], b)
		return k, nil
	}
	return sha256.Sum256([]byte(arg)), nil
}

// dial opens a single-stream client with the shared options.
func (c *common) dial(ctx context.Context) (*client.Client, error) {
	tok, err := c.resolveToken()
	if err != nil {
		return nil, err
	}
	return client.Dial(ctx, c.addr, client.Options{
		Streams:     1,
		Namespace:   c.ns,
		Token:       tok,
		DialTimeout: c.timeout,
	})
}

func usage() {
	fmt.Fprintf(os.Stderr, `kvbctl — kvblockd data-plane CLI

Usage:
  kvbctl exists [flags] KEY...          probe the prefix chain
  kvbctl get    [flags] [-o FILE] KEY   fetch one block (stdout by default)
  kvbctl put    [flags] KEY <FILE|->    store one block (stdin with -)
  kvbctl delete [flags] [-force] KEY... remove blocks
  kvbctl stats  [flags]                 server stats JSON
  kvbctl namespace add|list [flags]     tenant admin (loopback admin socket)
  kvbctl quota set [flags]              per-tier tenant quota (admin socket)
  kvbctl version                        print version

Shared flags: -addr -token-file -ns -timeout -hex   (see 'kvbctl CMD -h')
Token sources, by precedence: -token (smoke use ONLY — argv is visible in ps
and shell history) > -token-file > KVBCTL_TOKEN.
`)
	os.Exit(2)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var code int
	switch os.Args[1] {
	case "exists":
		code = cmdExists(os.Args[2:])
	case "get":
		code = cmdGet(os.Args[2:])
	case "put":
		code = cmdPut(os.Args[2:])
	case "delete":
		code = cmdDelete(os.Args[2:])
	case "stats":
		code = cmdStats(os.Args[2:])
	case "namespace":
		code = cmdNamespace(os.Args[2:])
	case "quota":
		code = cmdQuota(os.Args[2:])
	case "-version", "--version", "version":
		fmt.Printf("kvbctl %s (%s/%s)\n", version, runtime.GOOS, runtime.GOARCH)
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "kvbctl: unknown command %q\n", os.Args[1])
		usage()
	}
	os.Exit(code)
}

// fail prints an error and returns the exit code for it.
func fail(err error) int {
	fmt.Fprintln(os.Stderr, "kvbctl:", err)
	return 2
}
