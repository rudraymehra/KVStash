package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/kvstash/kvblockd/internal/protocol"
)

func TestDefaultsValidate(t *testing.T) {
	c := Default()
	if err := c.Validate(); err != nil {
		t.Fatalf("built-in defaults do not validate: %v", err)
	}
	if c.WireLimits() != protocol.DefaultLimits() {
		t.Fatalf("default wire limits diverge from the protocol defaults: %+v", c.WireLimits())
	}
}

func TestLoadNoFileIsDefaults(t *testing.T) {
	c, err := Load("", Overrides{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c, Default()) { // Config holds a slice (nvme_paths) — not ==-comparable
		t.Fatalf("configless load is not the defaults:\n got %+v\nwant %+v", c, Default())
	}
}

func TestLoadMissingExplicitFileFails(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.yaml"), Overrides{}); err == nil {
		t.Fatal("an explicitly configured missing file must be an error")
	}
}

// TestExampleYAMLIsDefaultsAndValid pins example.yaml against drift: it must
// load, validate, and equal the built-in defaults it documents.
func TestExampleYAMLIsDefaultsAndValid(t *testing.T) {
	c, err := Load("example.yaml", Overrides{})
	if err != nil {
		t.Fatalf("example.yaml: %v", err)
	}
	if !reflect.DeepEqual(c, Default()) {
		t.Fatalf("example.yaml diverged from the built-in defaults:\n got %+v\nwant %+v", c, Default())
	}
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestUnknownKeyRejected(t *testing.T) {
	p := writeTemp(t, "listen_adr: \":1\"\n") // typo'd key
	if _, err := Load(p, Overrides{}); err == nil || !strings.Contains(err.Error(), "listen_adr") {
		t.Fatalf("typo'd key silently accepted: %v", err)
	}
}

func TestPrecedence(t *testing.T) {
	p := writeTemp(t, "listen_addr: \":7001\"\nmax_conns: 7\n")

	t.Setenv("KVBLOCKD_LISTEN_ADDR", ":7002")

	c, err := Load(p, Overrides{})
	if err != nil {
		t.Fatal(err)
	}
	if c.ListenAddr != ":7002" || c.MaxConns != 7 {
		t.Fatalf("env must beat file, file must beat defaults: %+v", c)
	}

	flag := ":7003"
	conns := 3
	c, err = Load(p, Overrides{ListenAddr: &flag, MaxConns: &conns})
	if err != nil {
		t.Fatal(err)
	}
	if c.ListenAddr != ":7003" || c.MaxConns != 3 {
		t.Fatalf("flags must beat env and file: %+v", c)
	}
}

func TestEnvParseErrorSurfaces(t *testing.T) {
	t.Setenv("KVBLOCKD_MAX_CONNS", "not-a-number")
	if _, err := Load("", Overrides{}); err == nil {
		t.Fatal("garbage env int silently accepted")
	}
}

func TestValidateRejections(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
		expect string
	}{
		{"batch keys below floor", func(c *Config) { c.MaxBatchKeys = protocol.FloorMaxBatchKeys - 1 }, "max_batch_keys"},
		{"frame below floor", func(c *Config) { c.MaxFrameLen = protocol.FloorMaxFrameLen - 1 }, "max_frame_len"},
		{"blob below floor", func(c *Config) { c.MaxBlobLen = protocol.FloorMaxBlobLen - 1 }, "max_blob_len"},
		{"credit below floor", func(c *Config) { c.InitialCredit = protocol.FloorInitialCredit - 1 }, "initial_credit"},
		{"blob over frame", func(c *Config) { c.MaxBlobLen = c.MaxFrameLen + 1; c.MaxFrameLen = protocol.FloorMaxFrameLen }, "max_blob_len"},
		{"stream timeout below 5s", func(c *Config) { c.StreamTimeoutMS = 4999 }, "stream_timeout_ms"},
		{"lease default over max", func(c *Config) { c.LeaseDefaultMS = c.LeaseMaxMS + 1 }, "lease_default_ms"},
		{"lease max over clamp", func(c *Config) { c.LeaseMaxMS = protocol.MaxLeaseMS + 1 }, "lease_max_ms"},
		{"zero conns", func(c *Config) { c.MaxConns = 0 }, "max_conns"},
		{"bad addr", func(c *Config) { c.ListenAddr = "not-an-addr:port:extra" }, "listen_addr"},
		{"negative sndbuf", func(c *Config) { c.SockSndBuf = -1 }, "sock_sndbuf"},
		{
			"nvme paths without budget",
			func(c *Config) { c.NvmePaths = []string{"/mnt/nvme0"} },
			"nvme_max_bytes",
		},
		{
			"nvme segment not 4KiB-aligned",
			func(c *Config) {
				c.NvmePaths = []string{"/a"}
				c.NvmeMaxBytes = 8 << 30
				c.NvmeSegmentBytes = 1<<20 + 1
			},
			"nvme_segment_bytes",
		},
		{
			"nvme segment too small for one max blob",
			func(c *Config) { c.NvmePaths = []string{"/a"}; c.NvmeMaxBytes = 8 << 30; c.NvmeSegmentBytes = 4 << 20 },
			"nvme_segment_bytes",
		},
		{
			"nvme demote watermark not below eviction's",
			func(c *Config) { c.NvmePaths = []string{"/a"}; c.NvmeMaxBytes = 8 << 30; c.NvmeDemoteWatermarkPct = 95 },
			"nvme_demote_watermark_pct",
		},
		{
			"nvme duplicate path",
			func(c *Config) { c.NvmePaths = []string{"/a", "/a"}; c.NvmeMaxBytes = 8 << 30 },
			"duplicate path",
		},
		{
			"nvme with eviction none",
			func(c *Config) { c.NvmePaths = []string{"/a"}; c.NvmeMaxBytes = 8 << 30; c.EvictionPolicy = "none" },
			"eviction_policy",
		},
	}
	for _, tc := range cases {
		c := Default()
		tc.mutate(&c)
		err := c.Validate()
		if err == nil {
			t.Errorf("%s: accepted", tc.name)
			continue
		}
		if !strings.Contains(err.Error(), tc.expect) {
			t.Errorf("%s: error %q does not name %q", tc.name, err, tc.expect)
		}
	}

	// max_blob_len == max_frame_len is legal (blob over frame is not).
	c := Default()
	c.MaxBlobLen = c.MaxFrameLen
	if err := c.Validate(); err != nil {
		t.Errorf("blob == frame must be legal: %v", err)
	}

	// A well-formed two-volume NVMe config validates.
	c = Default()
	c.NvmePaths = []string{"/mnt/nvme0/kvb", "/mnt/nvme1/kvb"}
	c.NvmeMaxBytes = 64 << 30
	if err := c.Validate(); err != nil {
		t.Errorf("well-formed nvme config rejected: %v", err)
	}
}

func TestNvmePathsEnv(t *testing.T) {
	t.Setenv("KVBLOCKD_NVME_PATHS", " /mnt/nvme0 ,/mnt/nvme1, ")
	t.Setenv("KVBLOCKD_NVME_MAX_BYTES", "68719476736")
	c, err := Load("", Overrides{})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.NvmePaths) != 2 || c.NvmePaths[0] != "/mnt/nvme0" || c.NvmePaths[1] != "/mnt/nvme1" {
		t.Fatalf("env paths parsed wrong: %v", c.NvmePaths)
	}
	if c.NvmeMaxBytes != 64<<30 {
		t.Fatalf("env max bytes: %d", c.NvmeMaxBytes)
	}
}
