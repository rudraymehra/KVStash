# e2e — end-to-end integration rigs + economics

- `cpu/` — vLLM (LMCache and native adapter) CPU-backend e2e: byte-identity,
  hit counters, restart survival. Runs locally and in CI.
- `vllm-native-cpu.sh` — the native-adapter recipe (see header for phases).
- `sglang/` — SGLang HiCache L3 rig (scripts written and code-read; not yet
  run — needs a GPU box).
- `economics.py` — fetch-vs-recompute cost model, pure stdlib.
- `isolation_demo.sh`, `kill9_demo.sh` — recorded demos.
