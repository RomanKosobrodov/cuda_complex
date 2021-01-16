cmake_minimum_required( VERSION 3.17 )
cmake_policy(SET CMP0074 NEW)

project( cuda_complex VERSION 0.1 LANGUAGES C CXX )

set(CMAKE_CXX_STANDARD           11)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Tests 
enable_testing()
add_subdirectory(tests)
