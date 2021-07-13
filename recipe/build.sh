#!/usr/bin/env bash

set -ex

mkdir build
cd build

# enable components explicitly so we get build error when unsatisfied
cmake_config_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PROGRAM_PATH=$BUILD_PREFIX/bin
    -DCMAKE_INSTALL_PREFIX=$PREFIX
    -DLIB_SUFFIX=""
    -DENABLE_DOXYGEN=OFF
)

# Copy TensorRT headers to a local folder to separate them from system headers,
# only needed on aarch64. Also explicitly searches for them.
if [ ${ARCH} == "aarch64" ]; then
    mkdir -p ./tensorrt-headers/include
    cp -v /usr/include/aarch64-linux-gnu/Nv*.h ./tensorrt-headers/include
    cmake -GNinja ${CMAKE_ARGS} .. "${cmake_config_args[@]}" -DTensorRT_ROOT=./tensorrt-headers
else
    cmake -GNinja ${CMAKE_ARGS} .. "${cmake_config_args[@]}"
fi

cmake --build . --config Release -- -j${CPU_COUNT}
cmake --build . --config Release --target install
