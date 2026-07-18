package loadgen

import "runtime"

func yield() { runtime.Gosched() }
