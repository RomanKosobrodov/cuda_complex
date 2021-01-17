#include <cuda_runtime.h>
#include <initializer_list>
#include <cuComplex.h>

namespace complex {
struct complex_float: public cuFloatComplex {
  complex_float() = default;  
  complex_float(const std::initializer_list<float>& z) {
    const float* p = z.begin();
    x = *p;
    y = *(++p);
  }
  complex_float(const cuFloatComplex& other) {
    x = other.x;
    y = other.y; 
  }
  complex_float(float real, float imag) {
    x = real;
    y = imag;  
  }
  complex_float(float real) {
    x = real;
    y = 0.0f;  
  }
  operator cuFloatComplex() const {return make_cuFloatComplex(x, y);}
  operator cuFloatComplex&() {return *this;}

};


__host__ __device__ static __inline__ complex_float operator+(const complex_float& x, const complex_float& y) { 
    return cuCaddf(x, y); 
}

__host__ __device__ static __inline__ complex_float operator-(const complex_float& x, const complex_float& y) { 
    return cuCsubf(x, y); 
}

__host__ __device__ static __inline__ complex_float operator*(const complex_float& x, const complex_float& y) { 
    return cuCmulf(x, y); 
}

__host__ __device__ static __inline__ complex_float operator/(const complex_float& x, const complex_float& y) { 
    return cuCdivf(x, y); 
}

}