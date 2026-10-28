#!/bin/bash
# Runtime options from the validated PPU build; retain only build_flags/test_flags used by the launcher.
# The environment must already provide the SDK, toolchain, runtime libraries, and build proxies; no external tools are installed or loaded here.
SDK_ROOT="${SDK_ROOT:-/usr/local/PPU_SDK/CUDA_SDK}"
CUDA_VER="${CUDA_VER:-12.8.0}"
CUDNN_VER="${CUDNN_VER:-8.9.5}"
NCCL_VER="${NCCL_VER:-2.27.3}"
LOCAL_TEST_JOBS="${LOCAL_TEST_JOBS:-8}"

# Preserve build options to avoid changing Bazel action keys when packaging the scripts.
build_flags() {
  cat <<EOF
--verbose_failures
--config=opt
--config=cuda_nvcc
--config=cuda_wheel
--repo_env=CUDA_NVCC=1
--repo_env=HERMETIC_CUDA_VERSION=${CUDA_VER}
--repo_env=HERMETIC_CUDNN_VERSION=${CUDNN_VER}
--repo_env=HERMETIC_NCCL_VERSION=${NCCL_VER}
--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.0
--repo_env=LOCAL_CUDA_PATH=${SDK_ROOT}
--repo_env=LOCAL_CUDNN_PATH=${SDK_ROOT}
--repo_env=LOCAL_NCCL_PATH=${SDK_ROOT}
--//tensorflow/core/kernels/mlir_generated:enable_gpu=False
--copt=-Wno-error=unused-command-line-argument
--copt=-Wno-gnu-offsetof-extensions
--copt=-fclang-abi-compat=17
EOF
}

# Clear HTTP proxies in test processes to prevent interception of local communication.
# ppu_env.sh overrides test concurrency and device visibility.
test_flags() {
  cat <<EOF
--test_output=summary
--test_summary=short
--test_env=TF_FORCE_GPU_ALLOW_GROWTH=true
--test_env=SAIL_DISABLE_MULTI_PROCESS_TESTING=1
--test_env=http_proxy=
--test_env=https_proxy=
--test_env=HTTP_PROXY=
--test_env=HTTPS_PROXY=
--test_env=no_proxy=localhost,127.0.0.1
--local_test_jobs=${LOCAL_TEST_JOBS}
--keep_going
EOF
}
