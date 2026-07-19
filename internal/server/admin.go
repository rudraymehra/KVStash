package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/kvstash/kvblockd/internal/tenant"
)

// The admin surface: a LOCALHOST-ONLY HTTP listener on admin_addr serving
// namespace management. Deliberately not the data plane (no KVB1 frames, no
// bearer tokens) — reaching it requires a shell on the box, the same trust
// boundary as editing namespaces.yaml. Mutations affect the RUNNING process
// only; persist them in the namespaces file too (the response says so).
//
//	POST /v1/namespace  {"name":"a","id":2,"token_sha256":"64hex",
//	                     "quota_dram":N,"quota_nvme":N,"quota_s3":N,"pin_quota":N}
//	POST /v1/quota      {"name":"a","tier":"dram|nvme|s3","bytes":N}
//	GET  /v1/namespaces → [{name,id,quotas,usage...}] (tokens never listed)

// AdminServer wires the registry + accountant behind the two handlers.
type AdminServer struct {
	reg *tenant.Registry
	q   *tenant.Quotas
}

// NewAdminServer builds the admin surface (nil quotas = usage reads 0).
func NewAdminServer(reg *tenant.Registry, q *tenant.Quotas) *AdminServer {
	return &AdminServer{reg: reg, q: q}
}

// Serve binds addr (loopback enforced) and serves until ctx cancels.
// Returns the bound address and a wait func, mirroring metrics.Serve.
func (a *AdminServer) Serve(ctx context.Context, addr string) (string, func(), error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return "", nil, fmt.Errorf("admin_addr %q: %w", addr, err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
		return "", nil, fmt.Errorf("admin_addr %q must be loopback — the admin surface is shell-trust only", addr)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/namespace", a.handleAddNamespace)
	mux.HandleFunc("POST /v1/quota", a.handleSetQuota)
	mux.HandleFunc("GET /v1/namespaces", a.handleList)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return "", nil, err
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = srv.Serve(ln)
	}()
	go func() { //nolint:gosec // G118: shutdown must outlive the cancelled ctx; the fresh timeout context is the point
		<-ctx.Done()
		shCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(shCtx)
	}()
	return ln.Addr().String(), func() { <-done }, nil
}

type nsAddReq struct {
	Name        string `json:"name"`
	ID          uint32 `json:"id"`
	TokenSHA256 string `json:"token_sha256"`
	QuotaDRAM   int64  `json:"quota_dram"`
	QuotaNVMe   int64  `json:"quota_nvme"`
	QuotaS3     int64  `json:"quota_s3"`
	PinQuota    int64  `json:"pin_quota"`
}

func (a *AdminServer) handleAddNamespace(w http.ResponseWriter, r *http.Request) {
	var req nsAddReq
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	ns := &tenant.Namespace{
		ID: req.ID, Name: req.Name,
		Quota:    [3]int64{req.QuotaDRAM, req.QuotaNVMe, req.QuotaS3},
		PinQuota: req.PinQuota,
	}
	if err := decodeTokenHash(req.TokenSHA256, &ns.TokenHash); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := a.reg.Add(ns); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	writeJSON(w, map[string]any{
		"ok":   true,
		"note": "runtime only — add the entry to the namespaces file to survive a restart",
	})
}

type quotaReq struct {
	Name  string `json:"name"`
	Tier  string `json:"tier"`
	Bytes int64  `json:"bytes"`
}

func (a *AdminServer) handleSetQuota(w http.ResponseWriter, r *http.Request) {
	var req quotaReq
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<12)).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	tier, ok := tierByName(req.Tier)
	if !ok {
		http.Error(w, "tier must be dram|nvme|s3", http.StatusBadRequest)
		return
	}
	if !a.reg.SetQuota(req.Name, tier, req.Bytes) {
		http.Error(w, "unknown namespace", http.StatusNotFound)
		return
	}
	if a.q != nil {
		a.q.Reload()
	}
	writeJSON(w, map[string]any{
		"ok":   true,
		"note": "runtime only — update the namespaces file to survive a restart",
	})
}

func (a *AdminServer) handleList(w http.ResponseWriter, _ *http.Request) {
	type row struct {
		Name   string           `json:"name"`
		ID     uint32           `json:"id"`
		Quota  map[string]int64 `json:"quota_bytes"`
		Usage  map[string]int64 `json:"usage_bytes"`
		PinCap int64            `json:"pin_quota"`
	}
	var out []row
	tiers := []tenant.Tier{tenant.TierDRAM, tenant.TierNVMe, tenant.TierS3}
	a.reg.Each(func(ns *tenant.Namespace) {
		r := row{
			Name: ns.Name, ID: ns.ID, PinCap: ns.PinQuota,
			Quota: make(map[string]int64, 3), Usage: make(map[string]int64, 3),
		}
		for _, t := range tiers {
			r.Quota[t.String()] = ns.Quota[t]
			if a.q != nil {
				r.Usage[t.String()] = a.q.Usage(ns.ID, t)
			}
		}
		out = append(out, r)
	})
	writeJSON(w, out)
}

func tierByName(s string) (tenant.Tier, bool) {
	switch s {
	case "dram":
		return tenant.TierDRAM, true
	case "nvme":
		return tenant.TierNVMe, true
	case "s3":
		return tenant.TierS3, true
	}
	return 0, false
}

func decodeTokenHash(hexStr string, dst *[32]byte) error {
	if len(hexStr) != 64 {
		return fmt.Errorf("token_sha256 must be 64 hex chars (echo -n SECRET | shasum -a 256)")
	}
	for i := 0; i < 32; i++ {
		var b byte
		if _, err := fmt.Sscanf(hexStr[i*2:i*2+2], "%02x", &b); err != nil {
			return fmt.Errorf("token_sha256: %w", err)
		}
		dst[i] = b
	}
	return nil
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
