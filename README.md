# TensorFlow-for-SAIL

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## Table of Contents

- [Introduction](#introduction)
- [Supported Hardware](#supported-hardware)
- [User Guide](#user-guide)
- [Build from Source](#build-from-source)
- [Verify the Installation](#verify-the-installation)
- [Known Limitations](#known-limitations)
- [Resources](#resources)
- [Security](#security)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

TensorFlow-for-SAIL is developed based on the community open-source TensorFlow
project. It provides system-level adaptation and performance optimization for
Zhenwu PPU hardware, aiming to offer developers an out-of-the-box experience
for deep learning training and inference.

The current release is based on TensorFlow v2.21.0. While maintaining
compatibility with upstream TensorFlow features such as Eager Execution, graph
execution, Keras, and XLA, it adds backend adaptations and optimizations for
Zhenwu PPU devices.

## Supported Hardware

- Zhenwu 810
- Zhenwu 810E




## Build from Source

Build TensorFlow-for-SAIL inside the SAIL Docker image. The commands below use
`pkg.flytiger-eco.com/docker_release/ppu:v2.1.1-cuda12.8-ubuntu24-py312`.

```bash
# 1. Start the build container on a host with a Zhenwu PPU.
docker run -dit --name tensorflow-for-sail \
  --privileged \
  --ipc=host \
  --pid=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/alixpu_ctl \
  --network=host \
  pkg.flytiger-eco.com/docker_release/ppu:v2.1.1-cuda12.8-ubuntu24-py312

docker exec -it tensorflow-for-sail /bin/bash
```

Run the remaining commands inside the container:

```bash
# 2. Install build dependencies.
apt-get update
apt-get install -y clang-18 lld-18 patchelf wget xxd

wget -O /usr/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x /usr/bin/bazel

# 3. Clone the source code and initialize submodules.
git clone --recursive \
  https://github.com/flytiger-eco/tensorflow-for-sail.git \
  -b v2.21.0
cd tensorflow-for-sail

# If --recursive was not used or submodules are incomplete, run:
git submodule sync
git submodule update --init --recursive

# 4. Configure the PPU SDK and TensorFlow build.
source /usr/local/PPU_SDK/envsetup.sh
source /usr/local/pccl/envsetup.sh

SDK_ROOT=/usr/local/PPU_SDK/CUDA_SDK
CUDA_VER=12.8.0
CUDNN_VER=8.9.5
NCCL_VER=2.27.3

if [ ! -e "${SDK_ROOT}/lib" ]; then
  ln -s targets/x86_64-linux/lib "${SDK_ROOT}/lib"
fi
cp -f "${SDK_ROOT}"/extras/CUPTI/include/*.h "${SDK_ROOT}/include/"

export TF_NEED_CUDA=1
export TF_CUDA_CLANG=0
export TF_NVCC_CLANG=1
export CUDA_NVCC=1
export TF_ENABLE_XLA=1

export HERMETIC_CUDA_VERSION="${CUDA_VER}"
export HERMETIC_CUDNN_VERSION="${CUDNN_VER}"
export HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.0
export LOCAL_CUDA_PATH="${SDK_ROOT}"
export LOCAL_CUDNN_PATH="${SDK_ROOT}"
export LOCAL_NCCL_PATH="${NCCL_HOME}"

yes "" | ./configure

# 5. Build the wheel package.
bazel build -s --verbose_failures \
  --config=opt \
  --config=cuda_nvcc \
  --config=cuda_wheel \
  --repo_env=CUDA_NVCC=1 \
  --repo_env=HERMETIC_CUDA_VERSION="${CUDA_VER}" \
  --repo_env=HERMETIC_CUDNN_VERSION="${CUDNN_VER}" \
  --repo_env=HERMETIC_NCCL_VERSION="${NCCL_VER}" \
  --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=8.0 \
  --repo_env=LOCAL_CUDA_PATH="${SDK_ROOT}" \
  --repo_env=LOCAL_CUDNN_PATH="${SDK_ROOT}" \
  --repo_env=LOCAL_NCCL_PATH="${SDK_ROOT}" \
  --//tensorflow/core/kernels/mlir_generated:enable_gpu=False \
  --copt=-Wno-error=unused-command-line-argument \
  --copt=-Wno-gnu-offsetof-extensions \
  --copt=-fclang-abi-compat=17 \
  //tensorflow/python/kernel_tests/custom_ops:ackermann_op.so \
  //tensorflow/python/kernel_tests/custom_ops:duplicate_op.so \
  //tensorflow/tools/pip_package:wheel

# 6. Install the built wheel package.
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.21.0-cp312-cp312-linux_x86_64.whl
```



## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/)
- [TensorFlow Guide](https://www.tensorflow.org/guide/)
- [TensorFlow Models](https://github.com/tensorflow/models/tree/master/official)

## Security

For security information, please see [SECURITY.md](SECURITY.md).

## Disclaimer

- This software is provided for development and debugging purposes. Users
  assume all risks associated with its use.
- Users are responsible for managing data generated during use and complying
  with applicable security and compliance requirements.

## License

TensorFlow-for-SAIL is licensed under the
[Apache License 2.0](LICENSE).

## Acknowledgments

TensorFlow-for-SAIL is developed based on the community open-source TensorFlow
project. We thank the TensorFlow team and the open-source community for their
contributions, and welcome developers to contribute code, documentation, and
tests to TensorFlow-for-SAIL.
