//go:build linux

package stats

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// procPIDStat parses /proc/<pid>/stat: fields 14+15 (utime+stime, USER_HZ)
// → CPU seconds, field 24 (rss pages) → bytes. USER_HZ=100 is the
// documented assumption (universally true on the rigs' kernels).
func procPIDStat(pid int) (cpuSeconds float64, rssBytes int64, err error) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, 0, err
	}
	return parseProcStat(string(raw))
}

func parseProcStat(s string) (cpuSeconds float64, rssBytes int64, err error) {
	// comm (field 2) may contain spaces/parens — split after the LAST ')'.
	close := strings.LastIndexByte(s, ')')
	if close < 0 {
		return 0, 0, fmt.Errorf("stats: malformed /proc stat line")
	}
	fields := strings.Fields(s[close+1:]) // fields[0] = field 3 (state)
	if len(fields) < 22 {
		return 0, 0, fmt.Errorf("stats: short /proc stat line (%d fields)", len(fields))
	}
	utime, err1 := strconv.ParseFloat(fields[11], 64) // field 14
	stime, err2 := strconv.ParseFloat(fields[12], 64) // field 15
	rssPages, err3 := strconv.ParseInt(fields[21], 10, 64)
	if err := errors.Join(err1, err2, err3); err != nil {
		return 0, 0, fmt.Errorf("stats: /proc stat parse: %w", err)
	}
	const userHZ = 100
	return (utime + stime) / userHZ, rssPages * int64(os.Getpagesize()), nil
}
