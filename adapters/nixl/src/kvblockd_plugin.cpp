// kvblockd_plugin.cpp — the C export surface for libplugin_KVBLOCKD.so.
//
// Discovered by the NIXL plugin manager via NIXL_PLUGIN_DIR (the MinIO AIStor
// external-plugin precedent). The creator template supplies the required
// extern "C" nixl_plugin_init/nixl_plugin_fini pair plus get_plugin_name,
// get_plugin_version, create_engine, destroy_engine, get_backend_mems and
// get_backend_options.

#include "kvblockd_backend.h"

#include "backend/backend_plugin.h"

using kvblockd_plugin_t = nixlBackendPluginCreator<nixlKvblockdEngine>;

static const nixl_mem_list_t supported_segments = {DRAM_SEG, OBJ_SEG};

// Tuned defaults surfaced through get_backend_options — the anti-#1021 move:
// the out-of-the-box connection count and behaviors are the ones that actually
// perform against a kvblockd daemon (16 striped connections per PROTOCOL.md §4,
// verification on because corrupt KV blocks poison inference silently).
static const nixl_b_params_t default_options = {
    {"endpoint", "127.0.0.1:9440"}, // kvblockd data listener (host:port)
    {"namespace", ""},              // tenant namespace name (required)
    {"token", ""},                  // bearer token for the namespace (required)
    {"num_connections", "16"},      // wire pool size, 1..64
    {"verify_reads", "true"},       // xxh3-verify every GET payload
    {"put_ttl_ms", "0"},            // 0 = namespace default TTL
    {"op_timeout_ms", "30000"},     // per-syscall socket timeout
};

#ifdef STATIC_PLUGIN_KVBLOCKD

nixlBackendPlugin *
createStaticKVBLOCKDPlugin() {
    return kvblockd_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "KVBLOCKD", "0.1.0", default_options, supported_segments);
}

#else

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return kvblockd_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "KVBLOCKD", "0.1.0", default_options, supported_segments);
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}

#endif
