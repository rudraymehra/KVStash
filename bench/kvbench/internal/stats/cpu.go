package stats

import (
	"bufio"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

// CPUSample is the cores-per-GB/s input — a first-class metric (the
// vs-128-vCPU story needs it honest on both sides of the wire).
type CPUSample struct {
	ClientCores    float64 `json:"client_cores"`
	DaemonCores    float64 `json:"daemon_cores,omitempty"`
	DaemonRSSBytes int64   `json:"daemon_rss_bytes,omitempty"`
	DaemonSource   string  `json:"daemon_source,omitempty"` // "metrics" | "proc" | ""
}

// SelfCPU returns this process's cumulative user+system CPU seconds —
// xferspike's getrusage pattern; delta/wall = cores.
func SelfCPU() float64 {
	var ru unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	sec := func(tv unix.Timeval) float64 { return float64(tv.Sec) + float64(tv.Usec)/1e6 }
	return sec(ru.Utime) + sec(ru.Stime)
}

// DaemonProbe reads the daemon's cumulative CPU seconds + RSS from one of
// two sources:
//
//   - metricsURL (e.g. http://host:9442/metrics): scrapes
//     process_cpu_seconds_total / process_resident_memory_bytes — the ONLY
//     source that works across hosts (Rig T's client cannot read node B's
//     /proc). Requires the daemon's ProcessCollector (registered in
//     internal/metrics as part of this week).
//   - pid (loopback only): /proc/<pid>/stat utime+stime — Linux-only
//     fallback needing no daemon change.
type DaemonProbe struct {
	MetricsURL string
	PID        int
}

// Sample reads the current cumulative values (0s when unconfigured).
func (p DaemonProbe) Sample() (cpuSeconds float64, rssBytes int64, source string, err error) {
	switch {
	case p.MetricsURL != "":
		cpu, rss, err := scrapeProcess(p.MetricsURL)
		return cpu, rss, "metrics", err
	case p.PID > 0:
		cpu, rss, err := procPIDStat(p.PID)
		return cpu, rss, "proc", err
	default:
		return 0, 0, "", nil
	}
}

// scrapeProcess pulls the two process_* series off a Prometheus endpoint
// with a plain line-prefix parse (the verify.py pattern — no client lib).
func scrapeProcess(url string) (cpuSeconds float64, rssBytes int64, err error) {
	cl := http.Client{Timeout: 5 * time.Second}
	resp, err := cl.Get(url) //nolint:noctx,gosec // G107: operator-provided metrics URL, bounded timeout
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue // a metric line with no value (truncated scrape) — skip, don't panic
		}
		switch fields[0] {
		case "process_cpu_seconds_total":
			cpuSeconds, _ = strconv.ParseFloat(fields[1], 64)
		case "process_resident_memory_bytes":
			f, _ := strconv.ParseFloat(fields[1], 64)
			rssBytes = int64(f)
		}
	}
	if cpuSeconds == 0 {
		return 0, 0, fmt.Errorf("stats: %s exposes no process_cpu_seconds_total (daemon built without the process collector?)", url)
	}
	return cpuSeconds, rssBytes, sc.Err()
}
