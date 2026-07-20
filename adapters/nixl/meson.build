# kvblockd NIXL adapter (beta) — see README.md.
#
# Two build shapes:
#   * default: the shared C++ wire-client core + golden-vector test +
#     wire-level transfer gate. No NIXL dependency; builds on macOS/clang and
#     Linux/gcc alike.
#   * -Dnixl_path=<checkout> [-Dnixl_build_path=<its build dir>]: additionally
#     builds libplugin_KVBLOCKD.so (matching the in-tree plugin conventions:
#     name_prefix 'libplugin_', discovered via NIXL_PLUGIN_DIR) and the
#     NIXL-engine flavor of test_transfer. Needs abseil headers (nixl_log.h)
#     — pkg-config 'absl_log' (apt: libabsl-dev, brew: abseil).

project('kvblockd-nixl', 'cpp',
  version: '0.1.0',
  license: 'Apache-2.0',
  default_options: ['cpp_std=c++20', 'warning_level=2', 'buildtype=release'],
  meson_version: '>=1.1')

cxx = meson.get_compiler('cpp')
threads_dep = dependency('threads')

src_inc = include_directories('src')
xxhash_inc = include_directories('third_party/xxhash') # vendored v0.8.3, BSD-2-Clause

repo_root = meson.project_source_root() / '..' / '..'

# --- the shared native client core (SPEC 5: SGLang reuses this in Week 12) ---
kvb_core = static_library('kvb_core',
  ['src/kvblockd_client.cpp', 'src/executor.cpp'],
  include_directories: [src_inc, xxhash_inc],
  dependencies: [threads_dep],
  pic: true)

# --- NIXL wiring (optional) ---------------------------------------------------
nixl_path = get_option('nixl_path')
nixl_build_path = get_option('nixl_build_path')
have_nixl = nixl_path != ''

if have_nixl
  nixl_incs = include_directories(
    nixl_path / 'src/api/cpp',
    nixl_path / 'src/api/cpp/backend',
    nixl_path / 'src/core',
    nixl_path / 'src/core/telemetry',
    nixl_path / 'src/utils',
    nixl_path / 'src/infra',
  )
  # backend_aux.h -> common/nixl_log.h -> absl headers. Our code never calls
  # absl (compile-time include only), but link the dep where found — cheap and
  # keeps -Wl,--no-undefined viable for the test executable.
  absl_log_dep = dependency('absl_log', required: false)
  if not absl_log_dep.found()
    error('nixl_path given but abseil not found (pkg-config absl_log). apt: libabsl-dev, brew: abseil.')
  endif

  # libnixl_build (nixl src/infra): nixlDescList/nixlBlobDesc symbols. The
  # plugin and test link it exactly like in-tree plugins link nixl_infra.
  nixl_build_lib = []
  if nixl_build_path != ''
    nixl_build_lib = cxx.find_library('nixl_build',
      dirs: [nixl_build_path / 'src/infra'],
      required: true)
  endif

  # The plugin: sources compiled together (no dependence on this repo's static
  # lib layout), b_lundef=false so any symbol not satisfied by libnixl_build
  # resolves from the host process (which loaded libnixl) at dlopen — the
  # MinIO AIStor external-plugin pattern.
  plugin = shared_library('KVBLOCKD',
    ['src/kvblockd_client.cpp', 'src/executor.cpp',
     'src/kvblockd_backend.cpp', 'src/kvblockd_plugin.cpp'],
    name_prefix: 'libplugin_',
    include_directories: [src_inc, xxhash_inc, nixl_incs],
    dependencies: [threads_dep, absl_log_dep, nixl_build_lib],
    override_options: ['b_lundef=false'],
    install: false)
endif

# --- tests ---------------------------------------------------------------------

test_env = environment()
test_env.set('KVB_GOLDEN_DIR', repo_root / 'internal/protocol/testdata/frames')
test_env.set('KVB_ADAPTER_GOLDEN_DIR', meson.project_source_root() / 'test/golden')
if get_option('kvblockd_bin') != ''
  test_env.set('KVB_DAEMON', get_option('kvblockd_bin'))
else
  test_env.set('KVB_REPO_ROOT', repo_root)
endif

test_protocol = executable('test_protocol',
  'test/test_protocol.cpp',
  include_directories: [src_inc, xxhash_inc],
  link_with: kvb_core,
  dependencies: [threads_dep])

test('protocol-goldens', test_protocol,
  env: test_env,
  suite: 'nixl',
  timeout: 60)

transfer_srcs = ['test/test_transfer.cpp']
transfer_incs = [src_inc, xxhash_inc]
transfer_deps = [threads_dep]
transfer_args = []
if have_nixl
  transfer_srcs += ['src/kvblockd_backend.cpp']
  transfer_incs += [nixl_incs]
  transfer_deps += [absl_log_dep, nixl_build_lib]
  transfer_args += ['-DKVB_HAVE_NIXL']
endif

test_transfer = executable('test_transfer',
  transfer_srcs,
  include_directories: transfer_incs,
  link_with: kvb_core,
  dependencies: transfer_deps,
  cpp_args: transfer_args)

# The gate: 32×2.5MB write -> read -> verify -> abort against a real daemon
# the test spawns itself. 10-minute ceiling covers a cold `go build`.
test('transfer-gate', test_transfer,
  env: test_env,
  suite: 'nixl',
  timeout: 600,
  is_parallel: false)
