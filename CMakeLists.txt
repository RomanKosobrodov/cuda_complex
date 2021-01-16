cmake_minimum_required( VERSION 3.17 )
cmake_policy(SET CMP0074 NEW)

project( fft_arm64 VERSION 0.1 LANGUAGES C CXX )

set(CMAKE_CXX_STANDARD           11)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)

# Add cmake code for finding FFTW3 library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/support)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Tests 
enable_testing()
add_subdirectory(tests)
