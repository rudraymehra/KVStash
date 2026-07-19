// Admin-plane subcommands: namespace add / quota set / namespace list over
// the daemon's loopback admin socket (admin_addr, default 127.0.0.1:9441).
// These mutate the RUNNING process only — persist changes in the namespaces
// file too; the daemon says so in every response.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

func adminPost(addr, path string, payload any) int {
	body, err := json.Marshal(payload)
	if err != nil {
		return fail(err)
	}
	c := &http.Client{Timeout: 10 * time.Second}
	resp, err := c.Post("http://"+addr+path, "application/json", bytes.NewReader(body)) //nolint:noctx // CLI-local
	if err != nil {
		return fail(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	fmt.Print(string(out))
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}

func cmdNamespace(args []string) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "kvbctl namespace <add|list> …")
		return 2
	}
	switch args[0] {
	case "add":
		fs := flag.NewFlagSet("namespace add", flag.ExitOnError)
		admin := fs.String("admin", "127.0.0.1:9441", "daemon admin address (loopback)")
		name := fs.String("name", "", "namespace name")
		id := fs.Uint("id", 0, "stable nonzero namespace id")
		tokenFile := fs.String("token-file", "", "file holding the bearer token (hashed client-side; never sent)")
		qd := fs.Int64("quota-dram", 0, "DRAM byte quota (0 = unlimited)")
		qn := fs.Int64("quota-nvme", 0, "NVMe byte quota (0 = unlimited)")
		qs := fs.Int64("quota-s3", 0, "S3 byte quota (0 = unlimited)")
		pq := fs.Int64("pin-quota", 0, "pinned-bytes quota (0 = unlimited)")
		_ = fs.Parse(args[1:])
		if *name == "" || *id == 0 || *tokenFile == "" {
			fmt.Fprintln(os.Stderr, "namespace add needs -name, -id, -token-file")
			return 2
		}
		tok, err := os.ReadFile(*tokenFile) //nolint:gosec // G304: operator-supplied path
		if err != nil {
			return fail(err)
		}
		h := sha256.Sum256(bytes.TrimRight(tok, "\r\n"))
		return adminPost(*admin, "/v1/namespace", map[string]any{
			"name": *name, "id": *id, "token_sha256": hex.EncodeToString(h[:]),
			"quota_dram": *qd, "quota_nvme": *qn, "quota_s3": *qs, "pin_quota": *pq,
		})
	case "list":
		fs := flag.NewFlagSet("namespace list", flag.ExitOnError)
		admin := fs.String("admin", "127.0.0.1:9441", "daemon admin address (loopback)")
		_ = fs.Parse(args[1:])
		c := &http.Client{Timeout: 10 * time.Second}
		resp, err := c.Get("http://" + *admin + "/v1/namespaces") //nolint:noctx // CLI-local
		if err != nil {
			return fail(err)
		}
		defer resp.Body.Close()
		out, _ := io.ReadAll(resp.Body)
		fmt.Print(string(out))
		if resp.StatusCode != http.StatusOK {
			return 1
		}
		return 0
	default:
		fmt.Fprintf(os.Stderr, "kvbctl namespace: unknown subcommand %q\n", args[0])
		return 2
	}
}

func cmdQuota(args []string) int {
	if len(args) < 1 || args[0] != "set" {
		fmt.Fprintln(os.Stderr, "kvbctl quota set -name X -tier dram|nvme|s3 -bytes N")
		return 2
	}
	fs := flag.NewFlagSet("quota set", flag.ExitOnError)
	admin := fs.String("admin", "127.0.0.1:9441", "daemon admin address (loopback)")
	name := fs.String("name", "", "namespace name")
	tier := fs.String("tier", "", "dram|nvme|s3")
	bytesN := fs.Int64("bytes", 0, "quota bytes (0 = unlimited)")
	_ = fs.Parse(args[1:])
	switch *tier {
	case "dram", "nvme", "s3":
	default:
		fmt.Fprintln(os.Stderr, "quota set needs -tier dram|nvme|s3")
		return 2
	}
	if *name == "" {
		fmt.Fprintln(os.Stderr, "quota set needs -name")
		return 2
	}
	return adminPost(*admin, "/v1/quota", map[string]any{
		"name": *name, "tier": *tier, "bytes": *bytesN,
	})
}
