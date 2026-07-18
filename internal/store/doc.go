// Package store orchestrates the storage tiers: the Tiered store routes the
// server's Store surface through DRAM first and the NVMe index second, and
// owns tier movement — watermark demotion (90%, below the evictor's 95%),
// 2nd-hit promotion, and FIFO whole-segment reclaim. DRAM-only deployments
// never construct a Tiered store; the dram tier serves the server directly,
// byte-for-byte the pre-tiering behavior.
package store
