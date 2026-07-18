# The release image: FROM scratch — the binary is the image.
# Built by goreleaser (the static linux binary is the build context's
# `kvblockd`); ca-certificates ride along for the S3 tier's TLS.
#
#   docker run --rm -p 9440:9440 -p 9442:9442 \
#     -v $PWD/config:/config \
#     -e KVBLOCKD_NAMESPACES=/config/namespaces.yaml \
#     -e KVBLOCKD_METRICS_ADDR=0.0.0.0:9442 \
#     ghcr.io/kvstash/kvblockd:VERSION
#
# The env vars matter: with no namespaces file the daemon accepts no clients
# (by design), and the default metrics bind is 127.0.0.1 — loopback INSIDE
# the container, unreachable through -p without the 0.0.0.0 override.
# The container runs non-root (65532, the distroless "nonroot" convention).
# Hugepages/MEMLOCK need host-side setup — see deploy/kvblockd.service and
# docs/deployment-guide.md; the scratch image itself has no shell to debug
# with, by design.

FROM alpine:3.22 AS certs
RUN apk --no-cache add ca-certificates

FROM scratch
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY kvblockd /kvblockd
USER 65532:65532
EXPOSE 9440 9442
ENTRYPOINT ["/kvblockd"]
