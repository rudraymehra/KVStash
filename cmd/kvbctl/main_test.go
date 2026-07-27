package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestResolveTokenPrecedence: flag > file > env, file contents trimmed, a
// named-but-unreadable file is an error (never a silent empty credential).
func TestResolveTokenPrecedence(t *testing.T) {
	dir := t.TempDir()
	tf := filepath.Join(dir, "token")
	if err := os.WriteFile(tf, []byte("file-token\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("KVBCTL_TOKEN", "env-token")

	c := common{token: "flag-token", tokenFile: tf}
	if tok, err := c.resolveToken(); err != nil || tok != "flag-token" {
		t.Fatalf("flag precedence: %q, %v", tok, err)
	}

	c = common{tokenFile: tf}
	if tok, err := c.resolveToken(); err != nil || tok != "file-token" {
		t.Fatalf("file token (trimmed): %q, %v", tok, err)
	}

	c = common{}
	if tok, err := c.resolveToken(); err != nil || tok != "env-token" {
		t.Fatalf("env fallback: %q, %v", tok, err)
	}

	c = common{tokenFile: filepath.Join(dir, "no-such-file")}
	if _, err := c.resolveToken(); err == nil {
		t.Fatal("missing -token-file resolved silently instead of erroring")
	}
}
