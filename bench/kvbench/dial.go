package main

import (
	"context"
	"fmt"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/target"
)

// dialTarget builds the store under test from the shared flags. blobBytes
// feeds the nvmefs driver's fixed-size invariant.
func dialTarget(ctx context.Context, tf *targetFlags, blobBytes int) (target.Target, string, error) {
	switch tf.kind {
	case "kvblockd":
		t, err := target.DialKVBlockd(ctx, target.KVBOptions{
			Addr: tf.addr, Namespace: tf.ns, Token: tf.token,
			Streams: tf.streams, SockBuf: tf.sockbuf, SkipVerify: !tf.verify,
		})
		return t, "kvblockd", err
	case "redis", "valkey":
		t, err := target.DialRedis(ctx, target.RedisOptions{Addr: tf.addr, Streams: tf.streams})
		return t, tf.kind, err
	case "nvmefs":
		if tf.dir == "" {
			return nil, "", fmt.Errorf("nvmefs needs --dir")
		}
		t, err := target.OpenFS(target.FSOptions{
			Dir: tf.dir, BlobBytes: blobBytes, Workers: tf.streams, Fdatasync: tf.fsync,
		})
		return t, "nvmefs", err
	case "mem":
		return target.NewMem(0, 0), "mem", nil
	default:
		return nil, "", fmt.Errorf("unknown target %q", tf.kind)
	}
}
