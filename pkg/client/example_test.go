package client_test

import (
	"context"
	"fmt"

	"github.com/kvstash/kvblockd/pkg/client"
)

// ExampleClient_BatchGet shows the read path: dial a namespace, fetch a batch
// of block keys into caller-owned buffers, and read the per-key statuses. A
// NOT_FOUND leaves its buffer untouched; a hit streams straight into into[i].
func ExampleClient_BatchGet() {
	ctx := context.Background()
	c, err := client.Dial(ctx, "127.0.0.1:9440", client.Options{
		Streams:   4,
		Namespace: "tenant-a",
		Token:     "s3cr3t",
	})
	if err != nil {
		return // no daemon in the example environment; this is illustrative
	}
	defer c.Close()

	keys := [][32]byte{
		client.WireKey("vllm", "facebook/opt-125m", "1", "0", "12345"),
		client.WireKey("vllm", "facebook/opt-125m", "1", "0", "67890"),
	}
	into := make([][]byte, len(keys))
	for i := range into {
		into[i] = make([]byte, 2<<20) // pre-sized to the block
	}
	statuses, err := c.BatchGet(ctx, keys, into)
	if err != nil {
		return
	}
	for i, st := range statuses {
		fmt.Printf("key %d: %s\n", i, st)
	}
}
