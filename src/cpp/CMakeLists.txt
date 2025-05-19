
cmake_minimum_required(VERSION 3.17)
project(cb_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(USE_CLANG "build application with clang" ON)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

project(cb_test LANGUAGES CXX)
if (NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(FATAL_ERROR "Please use Ninja: cmake -G Ninja")
endif()
include(CMakePrintHelpers)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sys; print(sys.executable, sys.path)"
    RESULT_VARIABLE out
    OUTPUT_VARIABLE out_txt
)

cmake_print_variables(out_txt)
cmake_print_variables(Python_EXECUTABLE)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    RESULT_VARIABLE pybind11_result
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT pybind11_result EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11 installation directory")
endif()
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE torch_result
    OUTPUT_VARIABLE torch_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT torch_result EQUAL 0)
    message(FATAL_ERROR "Failed to find torch installation directory")
endif()

set(CMAKE_PREFIX_PATH ${pybind11_cmake_dir} ${torch_cmake_dir} ${CMAKE_PREFIX_PATH})
include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

# Set CMAKE_PREFIX_PATH to help CMake find pybind11
cmake_print_variables(CMAKE_PREFIX_PATH)
# cmake_print_variables(pybind11_cmake_dir)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED COMPONENTS Development)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
cmake_print_variables(TORCH_CXX_FLAGS)
cmake_print_variables(pybind11_INCLUDE_DIRS)
set(SOURCES cb_test/libcbtest.cpp)
pybind11_add_module(_C ${SOURCES})
target_include_directories(_C PRIVATE 
    ${TORCH_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Include the header files
)
cmake_print_variables(TORCH_LIBRARIES)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
target_link_libraries(_C PRIVATE  "${TORCH_LIBRARIES}"  fmt::fmt-header-only)
