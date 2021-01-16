project(cuda_complex_ut CXX)

set(CUDA_COMPLEX_UT_HEADERS
   catch.hpp
)

set(CUDA_COMPLEX_UT_SOURCES
    buffer_ut.cpp
    main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}
)

add_executable(${PROJECT_NAME}
    ${CUDA_COMPLEX_UT_HEADERS}
    ${CUDA_COMPLEX_UT_SOURCES}
)

target_include_directories(${PROJECT_NAME}
   PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}    
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC CUDA::cudart
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 53)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror)

add_test(NAME "CUDA_Complex_Unit_Tests" COMMAND ${PROJECT_NAME})