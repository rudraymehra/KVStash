"""vllm-kvblockd — vLLM native adapter for the kvblockd KV-cache block store.

Two integration altitudes, one package (see UPSTREAM.lock for the pinned
vLLM contract):

  CONNECTOR (CPU-capable, the e2e-gated one):
    --kv-transfer-config '{"kv_connector": "KvblockdConnector",
                           "kv_role": "kv_both",
                           "kv_connector_module_path": "vllm_kvblockd.connector",
                           "kv_connector_extra_config": {
                             "kvblockd_endpoint": "kvblockd://127.0.0.1:9440",
                             "kvblockd_namespace": "vllm",
                             "kvblockd_token": "..."}}'
    (v0.25 factory note: serve with --disable-hybrid-kv-cache-manager.)

  SECONDARY TIER (GPU serving via OffloadingConnector; e2e deferred, DEFER.md):
    --kv-transfer-config '{"kv_connector": "OffloadingConnector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {
                             "cpu_bytes_to_use": 64000000000,
                             "spec_name": "KvblockdTieringSpec",
                             "spec_module_path": "vllm_kvblockd.tier_manager",
                             "secondary_tiers": [{
                               "type": "kvblockd",
                               "endpoint": "kvblockd://127.0.0.1:9440",
                               "namespace": "vllm",
                               "token": "..."}]}}'
    Importing vllm_kvblockd.tier_manager (which spec_module_path does) is
    what registers the "kvblockd" tier type with SecondaryTierFactory.

Both classes import and instantiate WITHOUT vllm installed (the A6
interface-tripwire relies on this); vllm is an optional extra, never a hard
dependency — the plugin is loaded BY vllm.
"""

from vllm_kvblockd.config import AdapterConfig, block_chain_keys, chain_seed, fingerprint
from vllm_kvblockd.connector import KvblockdConnector, KvblockdConnectorMetadata
from vllm_kvblockd.tier_manager import KvblockdTierManager

__all__ = [
    "AdapterConfig",
    "KvblockdConnector",
    "KvblockdConnectorMetadata",
    "KvblockdTierManager",
    "block_chain_keys",
    "chain_seed",
    "fingerprint",
]
__version__ = "0.1.0"
