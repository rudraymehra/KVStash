// Package config loads and validates the kvblockd daemon configuration.
//
// Precedence, lowest to highest: built-in defaults → YAML file → environment
// (the fixed KVBLOCKD_* table) → command-line overrides. Validation never
// silently clamps: a config that violates a PROTOCOL.md §4 floor is an error
// the operator sees, not a surprise the peer negotiates around.
package config

import (
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// Config is the daemon configuration. Byte sizes are plain integers (no size
// suffix parsing — example.yaml carries the arithmetic in comments).
type Config struct {
	// ListenAddr is the data-plane TCP listener ("host:port").
	ListenAddr string `yaml:"listen_addr"`
	// AdminAddr and MetricsAddr are accepted for schema stability but not yet
	// served by this build (the admin/metrics surfaces land with the metrics
	// package); the daemon logs them as configured-but-inactive.
	AdminAddr   string `yaml:"admin_addr"`
	MetricsAddr string `yaml:"metrics_addr"`

	// MaxConns is a global accept cap — a cheap DoS floor. Per-tenant
	// connection accounting is the tenant package's job, not this field's.
	MaxConns int `yaml:"max_conns"`

	// PROTOCOL.md §4 negotiated-limit ceilings this server offers. Floors are
	// enforced by Validate against the protocol package's constants.
	MaxBatchKeys  uint32 `yaml:"max_batch_keys"`
	MaxFrameLen   uint32 `yaml:"max_frame_len"`
	MaxBlobLen    uint32 `yaml:"max_blob_len"`
	InitialCredit uint32 `yaml:"initial_credit"`

	// StreamTimeoutMS is the PUT_STREAM inactivity reaper (§5; floor 5000).
	StreamTimeoutMS uint32 `yaml:"stream_timeout_ms"`
	// LeaseDefaultMS / LeaseMaxMS are the §3.1 lease parameters.
	LeaseDefaultMS uint32 `yaml:"lease_default_ms"`
	LeaseMaxMS     uint32 `yaml:"lease_max_ms"`

	// NamespacesPath points at the namespaces/tokens YAML (server-side auth).
	NamespacesPath string `yaml:"namespaces_path"`

	// SockSndBuf / SockRcvBuf are per-connection socket buffer requests in
	// bytes. 16 MiB is the value the A1 rig saturated 50 GbE with; the kernel
	// silently clamps on untuned hosts (the transport logs the effective size).
	SockSndBuf int `yaml:"sock_sndbuf"`
	SockRcvBuf int `yaml:"sock_rcvbuf"`
}

// Overrides are the command-line flags an operator actually needs at launch;
// nil pointer = not set. Deliberately short.
type Overrides struct {
	ListenAddr     *string
	MetricsAddr    *string
	NamespacesPath *string
	MaxConns       *int
}

// Default returns the built-in defaults (PROTOCOL.md §4 defaults for the
// wire-visible limits).
func Default() Config {
	return Config{
		ListenAddr:      ":9440",
		AdminAddr:       "127.0.0.1:9441",
		MetricsAddr:     "127.0.0.1:9442",
		MaxConns:        1024,
		MaxBatchKeys:    protocol.DefaultMaxBatchKeys,
		MaxFrameLen:     protocol.DefaultMaxFrameLen,
		MaxBlobLen:      protocol.DefaultMaxBlobLen,
		InitialCredit:   protocol.DefaultInitialCredit,
		StreamTimeoutMS: protocol.DefaultStreamTimeoutMS,
		LeaseDefaultMS:  protocol.DefaultLeaseMS,
		LeaseMaxMS:      protocol.MaxLeaseMS,
		SockSndBuf:      16 << 20,
		SockRcvBuf:      16 << 20,
	}
}

// envTable is the fixed environment layer: variable → setter. One loop, no
// reflection, no surprises.
var envTable = []struct {
	name string
	set  func(*Config, string) error
}{
	{"KVBLOCKD_LISTEN_ADDR", func(c *Config, v string) error { c.ListenAddr = v; return nil }},
	{"KVBLOCKD_METRICS_ADDR", func(c *Config, v string) error { c.MetricsAddr = v; return nil }},
	{"KVBLOCKD_NAMESPACES", func(c *Config, v string) error { c.NamespacesPath = v; return nil }},
	{"KVBLOCKD_MAX_CONNS", func(c *Config, v string) error {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("KVBLOCKD_MAX_CONNS: %w", err)
		}
		c.MaxConns = n
		return nil
	}},
}

// Load builds the effective configuration. path == "" means "no file" (the
// daemon runs on defaults); a non-empty path that cannot be read or parsed is
// an error — an explicitly configured file must exist. The returned Config is
// already validated.
func Load(path string, ov Overrides) (Config, error) {
	c := Default()

	if path != "" {
		f, err := os.Open(path) //nolint:gosec // G304: path is the operator's own --config flag
		if err != nil {
			return Config{}, fmt.Errorf("config: %w", err)
		}
		defer f.Close()
		dec := yaml.NewDecoder(f)
		dec.KnownFields(true) // a typo'd key fails loudly, never silently defaults
		if err := dec.Decode(&c); err != nil {
			return Config{}, fmt.Errorf("config %s: %w", path, err)
		}
	}

	for _, e := range envTable {
		if v, ok := os.LookupEnv(e.name); ok {
			if err := e.set(&c, v); err != nil {
				return Config{}, err
			}
		}
	}

	if ov.ListenAddr != nil {
		c.ListenAddr = *ov.ListenAddr
	}
	if ov.MetricsAddr != nil {
		c.MetricsAddr = *ov.MetricsAddr
	}
	if ov.NamespacesPath != nil {
		c.NamespacesPath = *ov.NamespacesPath
	}
	if ov.MaxConns != nil {
		c.MaxConns = *ov.MaxConns
	}

	if err := c.Validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

// Validate checks the configuration against the PROTOCOL.md floors and basic
// sanity. It mutates nothing: a wrong config is an error, not a clamp.
func (c *Config) Validate() error {
	var errs []error
	check := func(cond bool, format string, args ...any) {
		if !cond {
			errs = append(errs, fmt.Errorf(format, args...))
		}
	}

	for _, a := range []struct{ name, addr string }{
		{"listen_addr", c.ListenAddr},
		{"admin_addr", c.AdminAddr},
		{"metrics_addr", c.MetricsAddr},
	} {
		if a.addr == "" {
			continue
		}
		if _, err := net.ResolveTCPAddr("tcp", a.addr); err != nil {
			errs = append(errs, fmt.Errorf("%s %q: %w", a.name, a.addr, err))
		}
	}

	check(c.MaxConns >= 1, "max_conns %d: must be >= 1", c.MaxConns)
	check(c.MaxBatchKeys >= protocol.FloorMaxBatchKeys,
		"max_batch_keys %d: below the §4 floor %d", c.MaxBatchKeys, protocol.FloorMaxBatchKeys)
	check(c.MaxFrameLen >= protocol.FloorMaxFrameLen,
		"max_frame_len %d: below the §4 floor %d", c.MaxFrameLen, protocol.FloorMaxFrameLen)
	check(c.MaxBlobLen >= protocol.FloorMaxBlobLen,
		"max_blob_len %d: below the §4 floor %d", c.MaxBlobLen, protocol.FloorMaxBlobLen)
	check(c.InitialCredit >= protocol.FloorInitialCredit,
		"initial_credit %d: below the §4 floor %d", c.InitialCredit, protocol.FloorInitialCredit)
	check(c.MaxBlobLen <= c.MaxFrameLen,
		"max_blob_len %d exceeds max_frame_len %d", c.MaxBlobLen, c.MaxFrameLen)
	check(c.StreamTimeoutMS >= 5000,
		"stream_timeout_ms %d: §5 says not negotiable below 5s", c.StreamTimeoutMS)
	check(c.LeaseDefaultMS >= 1 && c.LeaseDefaultMS <= c.LeaseMaxMS,
		"lease_default_ms %d: must be in [1, lease_max_ms %d]", c.LeaseDefaultMS, c.LeaseMaxMS)
	check(c.LeaseMaxMS <= protocol.MaxLeaseMS,
		"lease_max_ms %d: exceeds the protocol clamp %d", c.LeaseMaxMS, protocol.MaxLeaseMS)
	check(c.SockSndBuf >= 0 && c.SockRcvBuf >= 0,
		"sock_sndbuf/sock_rcvbuf must be >= 0 (0 = OS default)")

	return errors.Join(errs...)
}

// WireLimits is the config→protocol bridge: the §4 ceilings this server
// offers at HELLO negotiation.
func (c Config) WireLimits() protocol.Limits {
	return protocol.Limits{
		MaxBatchKeys:  c.MaxBatchKeys,
		MaxFrameLen:   c.MaxFrameLen,
		MaxBlobLen:    c.MaxBlobLen,
		InitialCredit: c.InitialCredit,
	}
}
