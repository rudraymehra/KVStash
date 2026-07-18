package target

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"
)

// Redis drives Redis 7 / Valkey 8 (RESP-compatible) through go-redis
// v9.21.0 — deliberately the STRONGEST available client configuration (the
// two-client rule: this bar shows what Redis can do with a zero-copy
// client; the redis-py bar shows the path LMCache actually ships).
//
// Fairness notes, disclosed in the README:
//   - GETs use the zero-copy GetToBuffer into per-slot reusable buffers,
//     one pipeline per batch — the per-key-buffer equivalent of MGET
//     (MGET proper cannot deliver into caller buffers).
//   - PUTs use SetFromBuffer, pipelined.
//   - Keys are the raw 32 bytes (RESP is binary-safe).
type Redis struct {
	c *redis.Client
}

// RedisOptions configures the driver. Streams maps to the connection pool
// size, mirroring the kvblockd driver's stream count.
type RedisOptions struct {
	Addr    string
	Streams int
}

// DialRedis connects and pings.
func DialRedis(ctx context.Context, o RedisOptions) (*Redis, error) {
	c := redis.NewClient(&redis.Options{
		Addr:     o.Addr,
		PoolSize: max(o.Streams, 1),
	})
	if err := c.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("target: dial redis %s: %w", o.Addr, err)
	}
	return &Redis{c: c}, nil
}

func rkey(k [32]byte) string { return string(k[:]) }

// BatchPut pipelines SetFromBuffer for every key.
func (t *Redis) BatchPut(ctx context.Context, keys [][32]byte, blobs [][]byte) ([]Status, error) {
	pipe := t.c.Pipeline()
	cmds := make([]*redis.StatusCmd, len(keys))
	for i, k := range keys {
		cmds[i] = pipe.SetFromBuffer(ctx, rkey(k), blobs[i])
	}
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		return nil, err
	}
	out := make([]Status, len(keys))
	for i, c := range cmds {
		if c.Err() != nil {
			out[i] = Errored
			continue
		}
		out[i] = OK
	}
	return out, nil
}

// BatchGet pipelines zero-copy GetToBuffer into the caller's dst slots.
// The reply is read DIRECTLY into dst[i] (no intermediate string) — the
// caller pre-sizes slots to the cell's blob size, exactly like the
// kvblockd driver's reuse contract.
func (t *Redis) BatchGet(ctx context.Context, keys [][32]byte, dst [][]byte) ([]Status, error) {
	pipe := t.c.Pipeline()
	cmds := make([]*redis.ZeroCopyStringCmd, len(keys))
	for i, k := range keys {
		if cap(dst[i]) == 0 {
			dst[i] = make([]byte, 4<<20) // first-use allocation; reused afterwards
		}
		cmds[i] = pipe.GetToBuffer(ctx, rkey(k), dst[i][:cap(dst[i])])
	}
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		return nil, err
	}
	out := make([]Status, len(keys))
	for i, c := range cmds {
		_, err := c.Result()
		switch {
		case errors.Is(err, redis.Nil):
			out[i] = Miss
		case err != nil:
			out[i] = Errored
		default:
			dst[i] = c.Bytes() // buf[:n] of the buffer we passed in
			out[i] = OK
		}
	}
	return out, nil
}

// BatchExists pipelines EXISTS and counts the consecutive-from-0 prefix.
func (t *Redis) BatchExists(ctx context.Context, chain [][32]byte) (int, error) {
	pipe := t.c.Pipeline()
	cmds := make([]*redis.IntCmd, len(chain))
	for i, k := range chain {
		cmds[i] = pipe.Exists(ctx, rkey(k))
	}
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		return 0, err
	}
	n := 0
	for _, c := range cmds {
		v, err := c.Result()
		if err != nil || v == 0 {
			break
		}
		n++
	}
	return n, nil
}

// Close releases the pool.
func (t *Redis) Close() error { return t.c.Close() }
