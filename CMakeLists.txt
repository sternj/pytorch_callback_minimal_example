cmake_minimum_required(VERSION 3.17)
project(cb_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
option(USE_CLANG "build application with clang" ON)
# Set the C++ standard


# Add subdirectory for C++ extension
add_subdirectory(src/cpp)

