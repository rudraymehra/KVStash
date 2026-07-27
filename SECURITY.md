# Security Policy

This file is the single authority on reporting; anything else in the repo
that mentions security reporting defers to it.

## Supported versions

Pre-1.0: only the **latest published release** (see the GitHub releases
page) receives security fixes. The transport is plaintext TCP by design in
v1 — deploy on a trusted network segment (the daemon warns on non-loopback
binds); TLS termination guidance lives in
[docs/deployment-guide.md](docs/deployment-guide.md).

## Reporting a vulnerability

1. **Preferred:** [GitHub private vulnerability reporting](https://github.com/rudraymehra/KVStash/security/advisories/new)
   — private to the maintainer, and the advisory machinery handles
   disclosure and credit.
2. **Fallback:** email rudraymehra@gmail.com with subject
   `[kvblockd security]`.

Please do not open public issues for security reports. We aim to
acknowledge within **72 hours** and to ship or publicly document a fix
within **90 days** of triage.

## Scope

The daemon (`kvblockd`), the CLI (`kvbctl`), the Go SDK (`pkg/client`), the
Python packages (`python/`), and the release pipeline. Design notes
relevant to security (multi-tenant isolation, hash-flood resistance,
cross-tenant side channels) live in [docs/DESIGN.md](docs/DESIGN.md).
