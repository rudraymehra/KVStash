//go:build !linux

package stats

import "errors"

// procPIDStat is Linux-only; on the dev box use --daemon-metrics instead.
func procPIDStat(int) (float64, int64, error) {
	return 0, 0, errors.New("stats: --daemon-pid needs /proc (linux); use --daemon-metrics")
}
