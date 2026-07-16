# kvblockd Wire Protocol v1 ("KVB1")

Status: **FROZEN v1.** Any wire change after this freeze requires updating this
document, the golden vectors (`internal/protocol/testdata/frames/`), and the
fuzz corpus in the same PR.

Locked constraints honored: TCP-only, batch-only verbs, 64-byte fixed header,
8 opcode families, two-phase PUT, CRC32C header protection, credit
backpressure, 16–64 parallel connections, zero-copy send/recv paths.

Byte order (locked, matching the transport rig's convention): the 4-byte
`magic` is carried so the wire bytes read literally as ASCII `"KVB1"`
(`4B 56 42 31`) in a hexdump — the universal convention for protocol magics.
Every other integer is **little-endian** (native on x86-64 and ARM64; the hot
path never byte-swaps). All strings are UTF-8, length-prefixed, no NUL
terminators; **length prefixes count bytes, never characters**. "Frame" = one
64-byte header + `payload_len` bytes of payload. One TCP connection carries an
ordered stream of frames; a client opens 16–64 connections and stripes batches
across them.

**Padding rule (applies everywhere a layout says "pad"):** pad bytes MUST be
`0x00`, ARE included in `payload_len`, and receivers MUST ignore their
contents. "Pad to an N-byte multiple" always means zero-padding the region so
the **total payload length** becomes the next multiple of N (no pad if it
already is one; a zero-length region gets zero pad bytes).

---

## 1. The 64-byte frame header

Every frame, both directions, starts with exactly 64 bytes:

| Offset | Size | Field | Type | Justification |
|---|---|---|---|---|
| 0 | 4 | `magic` | `"KVB1"` = `4B 56 42 31` | Instant rejection of port scans, TLS handshakes, HTTP probes hitting the port; also the version-generation anchor (a wire-incompatible v2 becomes `"KVB2"`). |
| 4 | 1 | `version` | u8, `0x01` | Header-layout generation. Bumped only if this 64B layout ever changes (intended: never). Capability evolution happens in feature bits, not here. |
| 5 | 1 | `opcode` | u8 | 8 verb families (§3). Response = same opcode with `F_RESP` flag. u8 leaves 247 unassigned values for future ops; unknown opcodes are skippable (length-prefixed) → forward compatible. |
| 6 | 2 | `flags` | u16 bitfield | Frame role + sub-op selector (§2). 16 bits keeps the header at 64B while giving 8 reserved bits for negotiated extensions. |
| 8 | 4 | `namespace_id` | u32 | Tenant routing on EVERY frame, resolved once at HELLO from name→u32. Integer compare instead of string lookup per op; makes cross-tenant confusion structurally impossible and lets a future L4 proxy shard by tenant without parsing bodies. |
| 12 | 4 | `credit` | u32 | Incremental flow-control grant in bytes (§8). Server→client frames: bytes added to the client's send window. Client→server frames: MUST be 0 in v1 (reserved for symmetric credit later). Doubles as the header's alignment pad so `request_id` lands on an 8-byte boundary. |
| 16 | 8 | `request_id` | u64 | Client-chosen, unique among outstanding requests on that connection; correlates pipelined/out-of-order responses (§7) and binds PUT_STREAM chunks to their stream (§5). `0` is reserved for control frames (NOP/CREDIT). |
| 24 | 32 | `key` | [32]u8 | BLAKE3 prefix-chain hash, **computed by the adapter, opaque to the server** (the server never inspects or recomputes it). In-header (not body) for the one verb that is per-key on the wire — PUT_STREAM — so chunk frames need zero body parsing to find their staging buffer. Batch verbs carry keys in the body and MUST zero this field. |
| 56 | 4 | `payload_len` | u32 | Length-prefixed framing: receiver always knows exactly how many bytes to read or skip — errors and unknown opcodes can never desynchronize the stream. Caps a payload at 4 GiB−1 (a frame is that plus the 64B header); the negotiated `max_frame_len` (§4) caps it far lower. |
| 60 | 4 | `header_crc32c` | u32 | CRC32C (Castagnoli, hardware-accelerated: SSE4.2 / ARMv8-CRC; Go stdlib `hash/crc32` with `crc32.Castagnoli`) over header bytes 0–59. Protects the length field — a corrupted `payload_len` is the one error that *would* desync framing. Payload integrity is per-block `xxh3_64` in descriptors (§3), not a frame CRC: end-to-end (stored at PUT, verified at final consumer) and not paid twice on multi-hundred-MB frames. |

Header CRC or magic mismatch, or unsupported `version` → receiver sends a FATAL
error frame if it can (§9) and closes the connection. There is no
resynchronization attempt, ever.

## 2. Flags (u16)

| Bit | Name | Meaning |
|---|---|---|
| 0 (0x0001) | `F_RESP` | Frame is a response. |
| 1 (0x0002) | `F_MORE` | More response frames follow for this `request_id` (BATCH_GET continuation, §3.3). |
| 2 (0x0004) | `F_FATAL` | Sender closes the connection after this frame (protocol-fatal error report). |
| 3 (0x0008) | `F_FORCE` | DELETE only: override lease/soft-pin protection (never hard pins). |
| 4–7 | `sub-op` | 4-bit sub-operation selector within an opcode family (see each opcode). |
| 8–15 | reserved | MUST be zero on send; receiver MUST ignore (new meanings only ever assigned behind negotiated feature bits). |

Defined-but-inapplicable flags (e.g. `F_FORCE` on BATCH_GET) and a nonzero
`key` field on non-PUT_STREAM verbs are ignored by the receiver, same as
reserved bits. A server receiving a frame with `F_RESP` set skips it via
`payload_len` and answers `ERR_MALFORMED`; a client receiving a response whose
`request_id` it never sent (or an unsolicited non-NOP frame) discards it and
SHOULD log — neither case is connection-fatal.

## 3. Opcodes and body layouts

Eight families. Response frames reuse the request opcode with `F_RESP` set.
Plus one control opcode outside the families:

`0x00 NOP/CREDIT` — no payload, `request_id`=0. Keepalive (recommended when
idle >10 s) and unsolicited credit grants via the header `credit` field. Never
responded to. Receivers MUST tolerate a nonconforming NOP — skip any payload
via `payload_len`, ignore a nonzero `request_id` — and still never answer it.

**Uniform response preamble** — every response payload begins with 8 bytes:
```
status u8 | reserved u8[3] | count u32     // count = op-specific item count in THIS frame
```
`status` is the batch-level code (§9); per-key outcomes ride in descriptors or
status-byte arrays. **On any non-OK/OK_EXISTS batch-level status, the response
payload is exactly the 8-byte preamble with `count`=0** — the op-specific
fields and per-key regions shown in §3.1–3.8 appear only on success.

**Descriptor** — the load-bearing 16-byte pattern for batch data responses
(padded from 13B to 16B for natural alignment and future width; alignment beats
3 bytes/block at 0.4–2.5 MB payloads):
```
status u8 | reserved u8[3] | len u32 | xxh3_64 u64
```
`xxh3_64` (`github.com/zeebo/xxh3` in Go) is the payload checksum computed by
the writer at PUT COMMIT, stored with the block, echoed on every GET — clients
verify after `recv_into`, giving end-to-end integrity across DRAM→NVMe→S3
round-trips without the server re-hashing on read.

### 3.1 `0x01 HELLO` (auth + negotiation)
MUST be the first frame on every connection; anything else first is FATAL.
Header `namespace_id`=0, `key`=zeros, `request_id` = a client-chosen **nonzero**
value, echoed in the response like any other request (0 is reserved for
NOP/CREDIT). A second HELLO on an established connection → `ERR_MALFORMED`;
negotiated limits are fixed for the connection's lifetime. If the server
supports no protocol version in the client's `[proto_min, proto_max]` range,
it answers `ERR_UNSUPPORTED` with `F_FATAL` and closes.

Request payload:
```
proto_min u8 | proto_max u8 | reserved u16 |
feature_bits u64 |                                  // client's supported set (§10)
max_batch_keys u32 | max_frame_len u32 |            // client's proposed caps
reserved u64 |
token_len u16 | ns_len u16 | client_name_len u16 | reserved u16 |
token bytes | ns bytes | client_name bytes | pad to 8B multiple
```
Response payload:
```
preamble (status: OK | ERR_AUTH_FAILED | ERR_NAMESPACE_UNKNOWN | ERR_FORBIDDEN; count=0)
proto u8 | reserved u8[3] |
feature_bits u64 |                                  // intersection — the contract for this connection
max_batch_keys u32 | max_frame_len u32 |            // negotiated: min(client, server); defaults 512 keys, 256 MiB
max_blob_len u32 |                                  // default 32 MiB (blocks are 0.4–2.5 MB)
namespace_id u32 |                                  // binding of ns name for this connection's lifetime
initial_credit u32 |                                // opening send window, bytes (default 128 MiB)
lease_default_ms u32 | lease_max_ms u32 |           // 5000 / 60000
stream_timeout_ms u32 |                             // 30000 — the PUT zombie reaper (§5)
server_name_len u16 | reserved u16 | server_name bytes | pad to 8
```
One bearer token, one namespace, per connection — auth is connection-scoped,
never per-request (token bytes cross the wire once). Multi-namespace clients
open separate connection pools; this is also what keeps per-tenant accounting
and QoS trivially attributable. Non-OK HELLO → server sets `F_FATAL` and
closes. TLS is out of scope for v1 (same-AZ trusted network; terminate
externally if needed); a feature bit is reserved for STARTTLS-style upgrade
later.

### 3.2 `0x02 BATCH_EXISTS` — the scheduler-blocking probe, p99 < 1 ms
Request payload: `n_keys u32 | reserved u32 | key[32] × n` — keys MUST be in
prefix-chain order (position 0 = root block). `n_keys`=0 is legal on every
batch verb: the response is well-formed with `count`=0 and empty per-key
regions (zero pad bytes, per the §0 padding rule).
Response payload:
```
preamble (count = n_keys)
n_consecutive u32 | reserved u32          // hits from position 0 until first miss — the number every
                                          // framework actually wants
[ per-key status bytes × n, zero-padded   // only if FEAT_EXISTS_BITMAP negotiated; OK / NOT_FOUND / EVICTED
  to the next 8-byte multiple ]
```
(The same "per-key status bytes × n, zero-padded to the next 8-byte multiple"
region shape is used by TOUCH_LEASE, PIN, and DELETE responses.)
Served entirely from the sharded DRAM index (negative lookups included); MUST
never touch NVMe or S3. This is the op whose latency budget justifies
dedicating connections to it (lane pattern, §7).

### 3.3 `0x03 BATCH_GET`
Request payload: `n_keys u32 | reserved u32 | key[32] × n`.
Every block returned with `status=OK` is automatically granted a **read lease
of `lease_default_ms` (5 s)** starting when the server queues the response — it
cannot be evicted or reclaimed mid-transfer or before the client finishes
copying to GPU. (The lease clock starts at a moment the client cannot observe;
clients MUST treat it as starting at their own receive time and re-LEASE or
pin for anything slower than a prompt copy.)

Response = one or more frames (split only when descriptors+payloads would
exceed `max_frame_len`; all but the last carry `F_MORE`). Frames of one
response MUST be sent in ascending `first_index` order with contiguous
coverage of the request keys. Each frame's payload:
```
preamble (count = n_desc in THIS frame)
first_index u32 | total_keys u32          // this frame covers request keys [first_index, first_index+count)
descriptor × count                        // 16B each, in request-key order
payloads, concatenated                    // bytes of every OK descriptor in this frame, same order
```
Server-side this is **one `writev`** per frame: iovec[0] = preamble+descriptors
from a pooled scratch buffer, iovec[1..] = block bytes straight out of
arena/page-cache memory (refcount held until the write completes; `net.Buffers`
in Go). Client side: read 64B header, read `16+16×count` descriptor region into
a scratch buffer, then `recv_into` pre-registered destination buffers sized
from the descriptors. No copies, no delimiter scanning.

### 3.4 `0x04 PUT_STREAM` (sub-ops: 0=BEGIN, 1=CHUNK, 2=COMMIT, 3=ABORT) — see §5
The only per-key verb on the wire (a "batch PUT" is N pipelined streams with
distinct `request_id`s striped across connections — blocks are 0.4–2.5 MB, so
per-block framing costs nothing and keeps commit atomicity per block). Header
`key` = block key on ALL four sub-ops.

- **BEGIN** payload: `total_len u32 | ttl_ms u32 | xxh3_64_hint u64 (0 if unknown) | flags u32 | reserved u32`. No BEGIN flags are defined in v1: the field MUST be 0 on send and is ignored on receive. `total_len`=0 is legal (an empty block; its GET descriptor is `status=OK, len=0`). Response always sent: `OK` (staging reserved) | `OK_EXISTS` (write-once idempotent hit — client stops sending) | `ERR_QUOTA_BYTES` | `ERR_TOO_LARGE` | `ERR_BUSY`.
- **CHUNK** payload: raw bytes (recommended 512 KiB–4 MiB per chunk; each ≤ `max_frame_len`). No response per chunk. Chunks for one stream MUST stay on the BEGIN connection, in order; the header `key`+`request_id` route bytes into the staged arena extent with zero body parsing. Zero-length CHUNKs are permitted but do NOT reset the inactivity timer (§5) — an idle client cannot pin staging with free empty frames.
- **COMMIT** payload: `xxh3_64 u64` (authoritative — client computes while streaming from GPU; overrides hint). Response always sent: `OK` | `ERR_SHORT_STREAM` (bytes ≠ total_len) | `ERR_CHECKSUM` | `ERR_STALE_STREAM` | `OK_EXISTS` (lost a same-key race; server verifies xxh3_64 equality, mismatch → `ERR_IMMUTABLE_CONFLICT`, which with content-derived keys means corruption — alert, never overwrite).
- **ABORT** payload: empty. Response always sent, exactly one: `OK` on a live stream (staging freed immediately) | `ERR_STALE_STREAM` on a tombstoned, timed-out, or unknown stream.

### 3.5 `0x05 TOUCH_LEASE` (sub-ops: 0=TOUCH, 1=LEASE, 2=RELEASE)
Request payload: `n_keys u32 | ttl_ms u32 | key[32] × n`. Response: preamble +
per-key status bytes (zero-padded to the next 8-byte multiple).
TOUCH bumps S3-FIFO recency and extends TTL (`ttl_ms`=0 → leave TTL, recency
only). **TOUCH is metadata-only**: touching an S3-resident block does NOT
trigger a lazy restore to a hotter tier — GET restores, TOUCH never does. LEASE
grants/extends an eviction-protection lease: `ttl_ms` is the requested lease
duration (0 → `lease_default_ms`), clamped to `lease_max_ms`; leases are
extensions of the same mechanism GET auto-grants. RELEASE drops it early
(`ttl_ms` ignored; polite clients shrink eviction pressure). Leases protect
against **eviction and non-forced DELETE** (§3.7), never against explicit
`F_FORCE` DELETE by the namespace owner.

### 3.6 `0x06 PIN` (sub-ops: 0=PIN_SOFT, 1=PIN_HARD, 2=UNPIN)
Request payload: `n_keys u32 | reserved u32 | key[32] × n`. Response: preamble
+ per-key status bytes (zero-padded to the next 8-byte multiple).
Soft pin = survives normal S3-FIFO pressure, may be demoted a tier and, in
quota emergency, dropped (observable in STATS); hard pin = never evicted, never
demoted below NVMe, debits the per-tenant pinned-bytes quota → `ERR_PIN_QUOTA`
per key when exhausted. UNPIN clears both levels.

### 3.7 `0x07 DELETE`
Request payload: `n_keys u32 | reserved u32 | key[32] × n`. Response: preamble
+ per-key status bytes (zero-padded to the next 8-byte multiple).
Logical delete (index removal; space reclaimed lazily by FIFO segment reclaim —
immutability means no compaction). Leased → `ERR_LEASED`, soft-pinned →
`ERR_PINNED`, both overridable with `F_FORCE`; hard-pinned → `ERR_PINNED`
always (unpin first). Exists so connector-driven eviction-from-above works.

### 3.8 `0x08 STATS`
Request payload: `sections u32` bitmask (0 = all) `| reserved u32`. No section
bits are assigned in v1: servers MAY ignore the mask and return all sections;
bit assignments arrive behind a feature bit. Response: preamble + UTF-8 JSON
document.
Deliberately JSON: cold path, human-debuggable, maps onto storage-metrics and
Prometheus label sets without a wire-schema migration per release. The JSON
carries a `schema` field inside the document (schema evolution never burns a
feature bit). Includes per-namespace bytes/blocks per tier, quota headroom,
pinned/leased bytes, hit/miss counters, credit-stall time, segment counts.

## 4. Negotiated limits (defaults)

| Limit | Default | Floor a server MUST accept |
|---|---|---|
| `max_batch_keys` | 512 | 128 |
| `max_frame_len` | 256 MiB | 16 MiB (the measured coalescing knee) |
| `max_blob_len` | 32 MiB | 4 MiB |
| `initial_credit` | 128 MiB | 16 MiB |
| connections/client | 16 (tune to 64; ESnet multi-stream guidance) | — |

Violations (`n_keys` > negotiated, frame > negotiated, blob > negotiated) →
`ERR_BATCH_TOO_LARGE` / `ERR_TOO_LARGE`, non-fatal (the frame is
length-prefixed, so the server skips it cleanly).

## 5. PUT_STREAM two-phase state machine

```
                    BEGIN
                      │  quota check, reserve staging extent (total_len, arena)
        ┌─────────────┼──────────────────────────────┐
   resp OK       resp OK_EXISTS                 resp ERR_*
        │             │                              │
     STREAMING     CLOSED (tombstone)             CLOSED (tombstone)
        │  CHUNK×n (in-order, same conn; each resets the 30s timer)
        │──────────────► bytes land in staged extent (invisible to all reads)
        │
   ┌────┴─────┬───────────────┬──────────────────┐
 COMMIT     ABORT        30s inactivity      connection close
   │           │               │                  │
 verify     free staging   free staging,      free ALL streams
 len+xxh3   resp OK        tombstone id       on this conn
   │
 atomic publish: index insert — block becomes visible to BATCH_EXISTS/GET
 on ALL connections of the namespace
   │
 resp OK → CLOSED (request_id reusable)
```

Rules that make zombies impossible:
- **Visibility only at successful COMMIT.** Staged bytes never appear in the
  index, never satisfy EXISTS/GET, and on crash are garbage by construction
  (NVMe log records without a commit footer are truncated at recovery — wire
  contract and storage contract agree).
- **Optimistic streaming:** the client MAY pipeline CHUNKs (and even COMMIT)
  immediately after BEGIN without waiting for the BEGIN response — zero added
  RTT. If BEGIN was rejected (or answered `OK_EXISTS`), the stream is
  tombstoned: the server silently consumes and discards subsequent CHUNKs for
  that `request_id` (skip via `payload_len`) and answers any COMMIT with
  `ERR_STALE_STREAM`. Deterministic invariant: **BEGIN, COMMIT, and ABORT each
  get exactly one response; CHUNK frames and tombstone-discarded CHUNKs get
  none.**
- **Unknown and duplicate streams:** a CHUNK/COMMIT/ABORT whose `request_id`
  never had a BEGIN on this connection is treated exactly as tombstoned
  (CHUNKs discarded; COMMIT/ABORT → `ERR_STALE_STREAM`). A BEGIN on a
  `request_id` with a live stream → `ERR_MALFORMED`; the original stream is
  unaffected.
- **30 s inactivity timeout** (`stream_timeout_ms`, reset on every CHUNK):
  reaper frees the staging extent and tombstones the id until connection close.
  Late frames → discarded / `ERR_STALE_STREAM`. Sized as ~12× the worst
  legitimate inter-chunk gap for a 2.5 MB block on a congested 10 GbE link; not
  negotiable below 5 s.
- Tombstones are bounded: cleared when the connection closes or the id gets a
  terminal response and the client reuses it.

## 6. Lease & pin semantics on the wire (summary of §3.3/3.5/3.6)

- Read lease (5 s default): auto-granted per block by BATCH_GET; extendable via
  LEASE; blocks eviction, segment reclaim, and non-forced DELETE — never
  F_FORCE deletes. Expiry is silent (no wire event) — clients needing longer
  protection re-LEASE or pin.
- Soft pin: eviction-resistant, quota-emergency droppable, demotable across
  tiers.
- Hard pin: guaranteed resident (≥NVMe), counts against pinned-bytes quota,
  blocks DELETE.
- TTL: set at PUT BEGIN (`ttl_ms`, 0 = namespace default), extended by TOUCH.
  Expiry makes the block eviction-eligible, not instantly deleted.
- Ladder ordering under pressure: unpinned+expired → unpinned → soft-pinned
  (emergency only) → leased (never while lease valid) → hard-pinned (never).

## 7. Pipelining and out-of-order responses

Any number of outstanding requests per connection (bounded by credit, §8).
`request_id` is the correlator; uniqueness among that connection's outstanding
requests is the client's job; reuse allowed after the terminal response.

- With `FEAT_OOO` negotiated (both reference implementations support it; it
  exists so trivial clients can opt out): the server responds in completion
  order — a 1 ms BATCH_EXISTS overtakes a 320 MB BATCH_GET on the same
  connection. Multi-frame responses (`F_MORE`) are never interleaved with other
  responses *between their frames*; a response sequence is contiguous.
- Without `FEAT_OOO`: strict FIFO responses. Adapters SHOULD then use dedicated
  per-op-class connection lanes (EXISTS lane / GET lanes / PUT lanes) to avoid
  head-of-line blocking. This is the recommended deployment shape even with
  OOO.

## 8. Credit-based backpressure

Byte-granular window on **client→server payload bytes** (PUT chunks dominate;
this is the slow-client / overload protection):

1. HELLO response sets `initial_credit` (window W, bytes).
2. Every request frame the client sends debits W by its `payload_len` (headers
   are free; EXISTS/GET requests debit their small bodies too — one rule, no
   exemptions).
3. W ≤ 0 → client MUST NOT send further non-control frames on that connection
   (NOP keepalive is allowed). A client at small-positive W may still send one
   frame up to `max_frame_len` — overshoot by at most one frame is by design
   and consistent with rule 5's enforcement threshold.
4. Server replenishes via the header `credit` field on ANY server→client frame
   and via unsolicited `NOP/CREDIT` frames when it has nothing else to say.
   **Every byte a client debits MUST eventually be re-granted** once the
   server has consumed or discarded the frame — including rejected, skipped,
   and tombstone-discarded frames; the "grant ≈ bytes durably drained from
   staging" pacing heuristic applies only to PUT CHUNK bytes. (Without this
   rule, a read-heavy client's window would leak ~100 bytes per probe until it
   stalled permanently.)
5. Server enforcement: a client exceeding its window by more than
   `max_frame_len` gets `ERR_BUSY` then, on repeat, `F_FATAL` close. The server
   never buffers unbounded bytes for a misbehaving client; a hard
   per-connection deadline (2× `stream_timeout_ms` with zero drain) closes
   stuck connections.

Server→client direction is governed by TCP flow control plus the client's own
batch sizing in v1; the request-direction `credit` field is reserved (always 0)
so symmetric credit can be added behind a feature bit without a header change.

## 9. Status codes (u8)

| Code | Name | Notes |
|---|---|---|
| 0x00 | `OK` | |
| 0x01 | `OK_EXISTS` | Write-once idempotent hit; success, block already sealed |
| 0x10 | `NOT_FOUND` | Never stored, or expired+reclaimed |
| 0x11 | `EVICTED` | Was here, evicted (observability nicety; treat as NOT_FOUND) |
| 0x20 | `ERR_AUTH_REQUIRED` | Non-HELLO before HELLO (sent with F_FATAL) |
| 0x21 | `ERR_AUTH_FAILED` | Bad token (F_FATAL) |
| 0x22 | `ERR_NAMESPACE_UNKNOWN` | (F_FATAL at HELLO) |
| 0x23 | `ERR_FORBIDDEN` | Token lacks op permission (e.g., read-only token PUTs) |
| 0x30 | `ERR_QUOTA_BYTES` | Per-tenant tier quota exhausted |
| 0x31 | `ERR_PIN_QUOTA` | Pinned-bytes quota exhausted |
| 0x32 | `ERR_TOO_LARGE` | Blob/frame over negotiated cap |
| 0x33 | `ERR_BATCH_TOO_LARGE` | n_keys over negotiated cap |
| 0x34 | `ERR_BUSY` | Backpressure / credit violation — retry with backoff |
| 0x40 | `ERR_CHECKSUM` | COMMIT xxh3_64 mismatch |
| 0x41 | `ERR_SHORT_STREAM` | COMMIT with bytes ≠ total_len |
| 0x42 | `ERR_STALE_STREAM` | COMMIT/ABORT on timed-out/tombstoned stream |
| 0x43 | `ERR_IMMUTABLE_CONFLICT` | Same key, different xxh3_64 — corruption alarm |
| 0x44 | `ERR_LEASED` | DELETE blocked by live lease (override: F_FORCE) |
| 0x45 | `ERR_PINNED` | DELETE blocked by pin (F_FORCE overrides soft only) |
| 0x50 | `ERR_UNSUPPORTED` | Unknown opcode/sub-op — frame skipped, connection healthy |
| 0x51 | `ERR_MALFORMED` | Body unparseable — frame skipped, connection healthy |
| 0x60 | `ERR_INTERNAL` | Server fault; request MAY be retried |
| 0xF0 | `FATAL_PROTOCOL` | Header CRC/magic/version violation; always with F_FATAL |

Design rule: **only three things are connection-fatal** (magic/CRC/version,
pre-auth traffic, sustained credit abuse). Everything else is a per-frame or
per-key status — an error can never desynchronize framing because
`payload_len` always tells the receiver how much to skip.

**The protocol-fatal report frame** (the "FATAL error frame" of §1) has a
fixed shape, because the offending frame's own opcode/request_id are
untrustworthy garbage: opcode `0x00`, flags `F_RESP|F_FATAL`, `request_id`=0,
`namespace_id`=0, payload = the 8-byte preamble carrying the fatal status
(`FATAL_PROTOCOL`, `ERR_AUTH_REQUIRED`, …), `count`=0. It is sent best-effort;
receivers MUST NOT rely on it arriving before the close.

## 10. Feature bits (u64, negotiated as intersection at HELLO)

| Bit | Name | v1 reference impl |
|---|---|---|
| 0 | `FEAT_OOO` — out-of-order responses (§7) | yes |
| 1 | `FEAT_EXISTS_BITMAP` — per-key bytes in BATCH_EXISTS (§3.2) | yes |
| 2 | `FEAT_PAYLOAD_CRC32C` — additional whole-payload CRC trailer per frame | no (xxh3_64 descriptors suffice) |
| 3 | `FEAT_CREDIT_SYMMETRIC` — client grants server response credit | no |
| 4 | `FEAT_TLS_UPGRADE` — reserved | no |
| 5–63 | reserved (0) | |

**Versioning strategy:** the 64B header layout is frozen for the life of
`version=1` / magic `KVB1`. Evolution = (a) new opcodes — old servers answer
`ERR_UNSUPPORTED` harmlessly; (b) new sub-ops/flags/fields **gated behind
feature bits** so nothing new is ever sent unnegotiated; (c) reserved fields
are zero-on-send / ignore-on-receive. A client supporting proto [min,max] sends
both in HELLO; the server picks. Magic bump = flag day, avoided at all costs.

## 11. Worked example A — BATCH_EXISTS, 3 keys

Client probes chain K0→K1→K2 in namespace 7, request_id 0x1001. Keys
(illustrative): K0=`aa`×32, K1=`bb`×32, K2=`cc`×32. Server has K0,K1; K2
missing. `FEAT_EXISTS_BITMAP` negotiated.

**Request frame — 64B header + 104B payload = 168 bytes:**
```
off 0   4B 56 42 31                                        magic "KVB1"
off 4   01                                                 version 1
off 5   02                                                 opcode BATCH_EXISTS
off 6   00 00                                              flags (request, no sub-op)
off 8   07 00 00 00                                        namespace_id 7
off 12  00 00 00 00                                        credit (client→server: 0)
off 16  01 10 00 00 00 00 00 00                            request_id 0x1001
off 24  00 ×32                                             key zeroed (batch verb)
off 56  68 00 00 00                                        payload_len 104
off 60  ff a7 2e 5f                                        crc32c(bytes 0..59) = 0x5F2EA7FF [GOLDEN: testdata/frames/example-a-request.hex]
--- payload ---
off 64  03 00 00 00  00 00 00 00                           n_keys=3, reserved
off 72  aa ×32                                             K0
off 104 bb ×32                                             K1
off 136 cc ×32                                             K2
```

**Response frame — 64B header + 24B payload = 88 bytes** (sub-millisecond,
DRAM index only):
```
off 0   4B 56 42 31 | 01 | 02 | 01 00                      magic, ver, BATCH_EXISTS, flags=F_RESP
off 8   07 00 00 00                                        namespace_id 7
off 12  00 00 10 00                                        credit grant +1 MiB (0x00100000) piggybacked
off 16  01 10 00 00 00 00 00 00                            request_id 0x1001 (echoed)
off 24  00 ×32 | 18 00 00 00 | 23 2e 93 d6                 key=0, payload_len=24, crc32c = 0xD6932E23 [GOLDEN: testdata/frames/example-a-response.hex]
--- payload ---
off 64  00 00 00 00  03 00 00 00                           preamble: status=OK, count=3
off 72  02 00 00 00  00 00 00 00                           n_consecutive=2, reserved
off 80  00 00 10  00 00 00 00 00                           bitmap: K0=OK, K1=OK, K2=NOT_FOUND(0x10); pad×5
```
The scheduler needs exactly one u32 — `n_consecutive=2` — to decide how many
layers to load vs recompute.

## 12. Worked example B — BATCH_GET, 3 blocks (2 hits: 1 MiB + 2.5 MiB, 1 miss)

**Request:** identical shape to example A but opcode `03`, request_id 0x1002;
168 bytes total.

**Response — one frame (fits under 256 MiB): 64B header + 3,670,080B payload.**
Golden-vector payload convention: K0 = 1,048,576 bytes of `0xAA`, K1 =
2,621,440 bytes of `0xBB` (matching their key fill bytes); credit grant
0x00100000 piggybacked. Vector: `testdata/frames/example-b-response-header.hex`
(frame header + response header region; payloads follow the convention).
```
header: opcode 03, flags 0x0001 (F_RESP, no F_MORE → final), request_id 0x1002,
        payload_len = 0x00380040 (3,670,080), credit 00 00 10 00,
        crc32c = b9 9d 4f c5 (0xC54F9DB9) [GOLDEN]
--- payload ---
+0      00 00 00 00  03 00 00 00                           preamble: status=OK, count=3 descriptors
+8      00 00 00 00  03 00 00 00                           first_index=0, total_keys=3
+16     00 00 00 00 | 00 00 10 00 | 4f fc 81 89 60 eb 7d c4   desc0: OK, len=1,048,576,  xxh3_64=0xC47DEB608981FC4F [GOLDEN]
+32     00 00 00 00 | 00 00 28 00 | 5d c4 66 ba 22 16 9e d6   desc1: OK, len=2,621,440,  xxh3_64=0xD69E1622BA66C45D [GOLDEN]
+48     10 00 00 00 | 00 00 00 00 | 00 ×8                  desc2: NOT_FOUND, len=0, xxh3_64=0
+64     <1,048,576 bytes of K0>                            payloads: OK blocks only, key order
+1048640 <2,621,440 bytes of K1>
```
Server emits this as **one `writev`**: `iov[0]`=64B header + 64B
preamble/descriptors (one pooled scratch buffer), `iov[1]`=K0 arena bytes,
`iov[2]`=K1 arena bytes (refcounts held until the kernel accepts; both blocks
now carry a 5 s read lease). Client: 64B header read → 64B descriptor-region
read → two `recv_into` calls straight into pre-registered, GPU-bound
destination buffers → xxh3_64 verify. Every payload byte crosses userspace once
on each side.

## 13. Deliberate divergences from prior art (and why)

1. **Fixed binary frames, not RESP text framing.** A single `-ERR` or nil
   `$-1\r\n` in a RESP stream desynchronizes the TCP stream permanently unless
   every reply is parsed defensively. Our `payload_len`-always +
   per-descriptor `status` means every error is skippable in O(1); desync is
   impossible by construction. We keep what actually makes RESP-based caches
   fast — batch-only verbs, tiling across N connections, one-writev responses,
   single wakeups client-side — and drop the text protocol.
2. **Per-descriptor explicit lengths instead of negotiated fixed-chunk sizes.**
   Fixed-chunk tricks halve metadata round-trips by forcing every chunk to the
   same size. Our descriptor table gets the same zero-round-trip property while
   supporting variable 0.4–2.5 MB blocks and partial batch failure natively.
3. **In-band negotiation, no external metadata service.** kvblockd is a single
   static binary on TCP: HELLO carries everything (auth, limits, features), and
   there are no master/replication verbs in v1 — a deployment-weight decision,
   not an oversight.
4. **No RDMA descriptor exchange; NIXL-shaped batching instead.** We copy the
   *shape* (batched descriptor lists, non-blocking post, coalesced doorbells →
   one syscall per batch) but bind it to writev/TCP, because the entire wedge
   is "works on the NICs you already have."
5. **Lease/pin/TTL/quota as first-class wire ops.** Prior art proved the
   semantics at scale but kept them internal and single-tenant. Elevating that
   ladder to authenticated, per-namespace wire verbs is the product.
6. **`namespace_id` in every header.** No shared-cache ambiguity, no
   isolation-header bolt-on, no cross-tenant timing surface: identity is
   (namespace, key) on every frame, and dedup never crosses it.
7. **Write-once on the wire.** No SET-overwrite, no update verbs: re-PUT =
   `OK_EXISTS` no-op ack. This one contract deletes overwrite/consistency
   semantics, makes retries free, and is what allows FIFO segment reclaim with
   zero compaction downstream.

## 14. Implementation notes

- Go packages: `hash/crc32` (`crc32.MakeTable(crc32.Castagnoli)`,
  HW-accelerated), `github.com/zeebo/xxh3` (payload checksums — the descriptor
  field is named `xxh3_64`; any earlier draft references to xxh64/cespare are
  superseded), `golang.org/x/sys/unix` (`Readv`, socket options via
  `ListenConfig.Control`, `TCP_NODELAY`), `net.Buffers` for writev. BLAKE3
  lives only in adapters/clients — the server never computes it.
- Repo layout: `internal/protocol` (header codec, descriptor codec, status
  tables — pure, fuzzable), `internal/transport` (conn loop, credit ledger,
  lanes), `internal/server` (dispatch, PUT state machine + 30 s reaper).
- Conformance artifacts: golden byte vectors (real CRCs and xxh3_64 values)
  live at `internal/protocol/testdata/frames/` — example A request/response
  and the example B response header region — pinned by `TestGoldenVectors` +
  `TestGoldenCRCsMatchSpec` and regenerable with `-update`; `FuzzParseHeader`
  and `FuzzParseBatch` carry the seed corpora. Still pending: a model-based
  test asserting the §5 invariant "BEGIN, COMMIT, and ABORT each get exactly
  one response" (lands with the server's stream table).
