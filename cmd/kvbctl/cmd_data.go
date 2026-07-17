package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/pkg/client"
)

// cmdExists probes the prefix chain: prints n_consecutive and, when the
// server negotiated the bitmap, one status line per key.
func cmdExists(args []string) int {
	var c common
	fs := flag.NewFlagSet("exists", flag.ExitOnError)
	c.register(fs)
	_ = fs.Parse(args)
	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "kvbctl exists: need at least one KEY")
		return 2
	}
	keys := make([][32]byte, fs.NArg())
	for i, arg := range fs.Args() {
		k, err := c.key(arg)
		if err != nil {
			return fail(err)
		}
		keys[i] = k
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	cl, err := c.dial(ctx)
	if err != nil {
		return fail(err)
	}
	defer cl.Close()

	n, perKey, err := cl.BatchExists(ctx, keys)
	if err != nil {
		return fail(err)
	}
	fmt.Printf("consecutive=%d\n", n)
	for i, st := range perKey {
		fmt.Printf("%s\t%s\n", fs.Arg(i), st)
	}
	if n < len(keys) {
		return 1
	}
	return 0
}

// cmdGet fetches one block to stdout or -o FILE. The client verifies the
// xxh3_64 descriptor checksum before any byte is emitted.
func cmdGet(args []string) int {
	var c common
	var out string
	fs := flag.NewFlagSet("get", flag.ExitOnError)
	c.register(fs)
	fs.StringVar(&out, "o", "", "write the block to FILE instead of stdout")
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "kvbctl get: need exactly one KEY")
		return 2
	}
	k, err := c.key(fs.Arg(0))
	if err != nil {
		return fail(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	cl, err := c.dial(ctx)
	if err != nil {
		return fail(err)
	}
	defer cl.Close()

	into := make([][]byte, 1)
	statuses, err := cl.BatchGet(ctx, [][32]byte{k}, into)
	if err != nil {
		return fail(err)
	}
	if statuses[0] != protocol.StatusOK && statuses[0] != protocol.StatusOKExists {
		fmt.Fprintf(os.Stderr, "kvbctl get: %s\n", statuses[0])
		return 1
	}

	if out != "" {
		f, err := os.Create(out) //nolint:gosec // G304: operator-chosen output path
		if err != nil {
			return fail(err)
		}
		if _, err := f.Write(into[0]); err != nil {
			_ = f.Close()
			return fail(err)
		}
		// A close-time write-back failure (full disk, NFS flush) must not
		// exit 0 pretending the block was saved.
		if err := f.Close(); err != nil {
			return fail(err)
		}
		return 0
	}
	if _, err := os.Stdout.Write(into[0]); err != nil {
		return fail(err)
	}
	return 0
}

// cmdPut stores one block from FILE (or stdin with "-"). The client streams
// BEGIN→CHUNK→COMMIT and treats a write-once OK_EXISTS hit as success.
func cmdPut(args []string) int {
	var c common
	fs := flag.NewFlagSet("put", flag.ExitOnError)
	c.register(fs)
	_ = fs.Parse(args)
	if fs.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "kvbctl put: need KEY and FILE (or - for stdin)")
		return 2
	}
	k, err := c.key(fs.Arg(0))
	if err != nil {
		return fail(err)
	}
	var data []byte
	if fs.Arg(1) == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(fs.Arg(1)) //nolint:gosec // G304: operator-chosen input path
	}
	if err != nil {
		return fail(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	cl, err := c.dial(ctx)
	if err != nil {
		return fail(err)
	}
	defer cl.Close()

	if err := cl.Put(ctx, k, data); err != nil {
		// A protocol-level status (quota, too-large, busy) is a per-key
		// failure (exit 1); only transport/dial/I-O errors are exit 2.
		var se *client.StatusError
		if errors.As(err, &se) {
			fmt.Fprintf(os.Stderr, "kvbctl put: %s\n", se.Status)
			return 1
		}
		return fail(err)
	}
	fmt.Printf("stored %d bytes\n", len(data))
	return 0
}

// cmdDelete removes blocks; -force sets F_FORCE (evict even leased/pinned
// once the tiers enforce lifecycle). Prints one status line per key.
func cmdDelete(args []string) int {
	var c common
	var force bool
	fs := flag.NewFlagSet("delete", flag.ExitOnError)
	c.register(fs)
	fs.BoolVar(&force, "force", false, "force-evict even leased/pinned blocks")
	_ = fs.Parse(args)
	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "kvbctl delete: need at least one KEY")
		return 2
	}
	keys := make([][32]byte, fs.NArg())
	for i, arg := range fs.Args() {
		k, err := c.key(arg)
		if err != nil {
			return fail(err)
		}
		keys[i] = k
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	cl, err := c.dial(ctx)
	if err != nil {
		return fail(err)
	}
	defer cl.Close()

	perKey, err := cl.Delete(ctx, keys, force)
	if err != nil {
		return fail(err)
	}
	code := 0
	for i, st := range perKey {
		fmt.Printf("%s\t%s\n", fs.Arg(i), st)
		if st != protocol.StatusOK {
			code = 1
		}
	}
	return code
}

// cmdStats prints the server's stats JSON document.
func cmdStats(args []string) int {
	var c common
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	c.register(fs)
	_ = fs.Parse(args)

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	cl, err := c.dial(ctx)
	if err != nil {
		return fail(err)
	}
	defer cl.Close()

	doc, err := cl.Stats(ctx)
	if err != nil {
		return fail(err)
	}
	fmt.Println(string(doc))
	return 0
}
