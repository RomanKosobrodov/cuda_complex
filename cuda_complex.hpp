#include <cuda_runtime.h>

namespace complex {
__host__ __device__ static __inline__ cuFloatComplex operator+(cuFloatComplex x, cuFloatComplex y) { 
    return cuCaddf(x, y); 
}
}