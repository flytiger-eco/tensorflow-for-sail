#!/bin/bash
# Usage: SAIL_ENV_FILE=ci/sail/ppu_env.sh bash ci/sail/run_tests.sh --profile cuda --run
# Proxy settings and other sensitive configuration come from the container login shell; no credentials are stored here.
ppu_config="${SAIL_PPU_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/ppu_config.sh}"
if [[ ! -f "$ppu_config" ]]; then
  printf 'Missing PPU build configuration: %s\n' "$ppu_config" >&2
  exit 2
fi
source "$ppu_config"
# Do not load other environment scripts; retain SDK paths and build proxies from the container login environment.
unset BAZELISK_BASE_URL PYTHON_PKG_URL
export TFCI=py312,linux_x86,no_docker
export TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX="linux_${profile}"
export TFCI_BAZEL_COMMON_ARGS="$(build_flags | tr '\n' ' ') $(test_flags | tr '\n' ' ') --repo_env=HERMETIC_PYTHON_VERSION=3.12 --jobs=${SAIL_BUILD_JOBS:-16} --local_test_jobs=1 --test_timeout=900 --test_env=all_proxy= --test_env=ALL_PROXY="
# Forward SDK location variables explicitly; otherwise the convolution offline cache root incorrectly falls back to /rtccache.
# Match the login shell without changing precision, algorithms, or test assertions.
export TFCI_BAZEL_COMMON_ARGS="$TFCI_BAZEL_COMMON_ARGS --test_env=PATH --test_env=HOME --test_env=CUDA_HOME --test_env=CUDA_PATH --test_env=CUDA_SDK --test_env=CUDA_TOOLKIT_ROOT --test_env=PPU_HOME --test_env=PPU_PATH --test_env=PPU_SDK --test_env=HGGC_DRIVER_CANDIDATE"
if [[ "$profile" == cpu ]]; then
  export TFCI_BAZEL_COMMON_ARGS="$TFCI_BAZEL_COMMON_ARGS --test_env=CUDA_VISIBLE_DEVICES="
else
  export TFCI_BAZEL_COMMON_ARGS="$TFCI_BAZEL_COMMON_ARGS --test_env=CUDA_VISIBLE_DEVICES=${SAIL_CUDA_VISIBLE_DEVICES:-0}"
fi
# Official NVIDIA driver packaging options do not apply to PPU; reuse official CI for selection and Bazel execution.
export SAIL_BAZEL_PLATFORM_ARGS=""
export TFCI_DOCKER_ENABLE=0
export TFCI_INDEX_HTML_ENABLE=0
export TFCI_OUTPUT_DIR="${SAIL_OUTPUT_ROOT:-/root/sail_final_work/ci_output}"
