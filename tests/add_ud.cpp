#include "catch.hpp"
#include <cuda_complex.hpp>


__global__ add_float(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* sum) {
    using namespace complex;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sum[index] = a[index] + b[index];
}


TEST_CASE("Single precision complex numbers added correctly") {
    
}